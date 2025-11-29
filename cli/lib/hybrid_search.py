#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

# Ensure the project root (one level above the `cli` package) is on sys.path
# so `from cli.helpers import ...` works when running the script directly.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from cli.helpers import InvertedIndex, Movie
from cli.lib.semantic_search import ChunkedSemanticSearch

NumberId = Union[int, str]


class HybridSearch:
    def __init__(self, documents: List[Movie]):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        # Ensure chunk embeddings / metadata are available for semantic search.
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        # InvertedIndex expects (index, docmap) dicts on construction.
        self.idx = InvertedIndex({}, {})

        # Try to load an existing persisted index; if it's not available,
        # build from the supplied documents and persist.
        try:
            self.idx.load()
        except FileNotFoundError:
            # Build expects a list of movie dicts
            self.idx.build(documents)
            self.idx.save()

    def _bm25_search(self, query: str, limit: int):
        # Ensure index is loaded before searching (load is idempotent).
        try:
            self.idx.load()
        except FileNotFoundError:
            # If loading fails unexpectedly, ensure we have an index in memory.
            self.idx.build(self.documents)
            self.idx.save()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5):
        """Perform a hybrid weighted search combining BM25 and chunked semantic scores.

        - Normalize both score sets via min-max normalization.
        - Combine scores per-document using the provided alpha weight for BM25.
        - Return the top `limit` results sorted by the hybrid score.
        """
        # Get BM25 results (movies list and a dict of id->score)
        bm25_docs, bm25_scores = self._bm25_search(query, limit)

        # Use chunked semantic search to get per-document semantic scores.
        # `search_chunks` returns formatted dicts with 'id' (persistent movie id) and 'score'.
        semantic_results = self.semantic_search.search_chunks(query, limit)

        # Normalize representations to list[(doc_id, score)]
        bm25_pairs: List[Tuple[NumberId, float]] = [
            (doc["id"], bm25_scores.get(doc["id"], 0.0)) for doc in bm25_docs
        ]
        bm25_scores_list = [score for _, score in bm25_pairs]

        # Convert semantic formatted results into (actual_doc_id, score) pairs.
        # `search_chunks` returns persistent movie ids in the `id` field; if an int
        # is provided that looks like an index, we try to resolve index->id as fallback.
        semantic_pairs: List[Tuple[NumberId, float]] = []

        # Build a lookup of persistent ids -> movie dict for robust mapping.
        id_to_movie_map: Dict[NumberId, Movie] = {
            cast(NumberId, doc["id"]): doc for doc in self.documents
        }

        for item in semantic_results:
            # item is expected to be a dict like {"id": movie_id, "score": score, ...}
            movie_identifier = item.get("id")
            raw_score = item.get("score", 0.0)

            resolved_id: Optional[NumberId] = None
            if isinstance(movie_identifier, int):
                # Prefer interpreting the integer as a persistent movie id if present.
                if movie_identifier in id_to_movie_map:
                    resolved_id = movie_identifier
                # Otherwise, if it appears to be an index into self.documents, map index -> id.
                elif 0 <= movie_identifier < len(self.documents):
                    doc = self.documents[movie_identifier]
                    resolved_id = cast(NumberId, doc.get("id"))
                else:
                    # Unknown integer id (stale index), skip this semantic hit.
                    continue
            else:
                # Non-int ids are treated as persistent ids (e.g. string ids)
                resolved_id = movie_identifier

            if resolved_id is None:
                continue
            semantic_pairs.append((resolved_id, float(raw_score)))

        semantic_scores_list = [score for _, score in semantic_pairs]

        # Normalize both score lists into [0.0, 1.0]
        normalized_bm25 = normalize_scores(bm25_scores_list)
        normalized_semantic = normalize_scores(semantic_scores_list)

        # Build a map of all documents by persistent id for easy lookup
        id_to_movie_map = {cast(NumberId, doc["id"]): doc for doc in self.documents}

        # Create maps from doc_id -> normalized score for bm25 and semantic results.
        # Use zip to align the normalized scores with the original ordered pairs.
        bm25_norm_map: Dict[NumberId, float] = {
            doc_id: score for (doc_id, _), score in zip(bm25_pairs, normalized_bm25)
        }
        semantic_norm_map: Dict[NumberId, float] = {
            doc_id: score
            for (doc_id, _), score in zip(semantic_pairs, normalized_semantic)
        }

        # Collect the union of candidate document ids
        candidate_ids = set(bm25_norm_map.keys()) | set(semantic_norm_map.keys())

        # Build list of combined result dicts
        combined_results: List[Dict[str, Any]] = []
        for doc_id in candidate_ids:
            kw_score = bm25_norm_map.get(doc_id, 0.0)
            sem_score = semantic_norm_map.get(doc_id, 0.0)
            hyb = hybrid_score(kw_score, sem_score, alpha)
            combined_results.append(
                {
                    "doc": id_to_movie_map.get(doc_id),
                    "keyword_score": kw_score,
                    "semantic_score": sem_score,
                    "hybrid_score": hyb,
                }
            )

        # Return results sorted by hybrid_score descending, limited to `limit` entries
        combined_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return combined_results[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10):
        """Perform Reciprocal Rank Fusion (RRF) search combining BM25 and chunked semantic scores."""
        # Increase the candidate pool so RRF has more documents to fuse.
        # This helps prevent missing ranks when one method retrieves documents
        # that the other method did not include in the small top-k window.
        search_limit = limit * 25

        # Get BM25 results (movies list and a dict of id->score)
        bm25_docs, _ = self._bm25_search(query, search_limit)

        # Use chunked semantic search to get per-document semantic scores.
        semantic_results = self.semantic_search.search_chunks(query, search_limit)

        # Build rank maps: doc_id -> rank (1-based)
        bm25_rank_map: Dict[NumberId, int] = {
            cast(NumberId, doc["id"]): rank + 1 for rank, doc in enumerate(bm25_docs)
        }
        semantic_rank_map: Dict[NumberId, int] = {}

        # Build a persistent-id map to prefer persistent ids in semantic results
        id_to_movie_map: Dict[NumberId, Movie] = {
            cast(NumberId, doc["id"]): doc for doc in self.documents
        }

        for rank, item in enumerate(semantic_results):
            movie_identifier = item.get("id")
            resolved_id: Optional[NumberId] = None

            if isinstance(movie_identifier, int):
                # If the integer matches a persistent id, use it.
                if movie_identifier in id_to_movie_map:
                    resolved_id = movie_identifier
                # Otherwise, if it's an index into self.documents, map index -> id.
                elif 0 <= movie_identifier < len(self.documents):
                    doc = self.documents[movie_identifier]
                    resolved_id = cast(NumberId, doc.get("id"))
                else:
                    # stale/out-of-range index; skip this semantic hit
                    continue
            else:
                # assume it's already a persistent doc ID (could be a string)
                resolved_id = movie_identifier

            if resolved_id is None:
                continue
            semantic_rank_map[resolved_id] = rank + 1

        # Collect the union of candidate document ids
        candidate_ids = set(bm25_rank_map.keys()) | set(semantic_rank_map.keys())

        # Build list of combined result dicts
        combined_results: List[Dict[str, Any]] = []
        for doc_id in candidate_ids:
            bm25_rank = bm25_rank_map.get(doc_id)
            semantic_rank = semantic_rank_map.get(doc_id)

            rrf_score = 0.0
            if bm25_rank is not None:
                rrf_score += 1.0 / (k + bm25_rank)
            else:
                # If no BM25 rank, treat as worst possible rank (search_limit + 1)
                rrf_score += 1.0 / (k + search_limit + 1)
            if semantic_rank is not None:
                rrf_score += 1.0 / (k + semantic_rank)
            else:
                # If no semantic rank, treat as worst possible rank (search_limit + 1)
                rrf_score += 1.0 / (k + search_limit + 1)

            # Resolve movie dict for display; linear scan used as a fallback for mixed key types.
            movie_doc = None
            # Prefer direct lookup by persistent id
            movie_doc = id_to_movie_map.get(doc_id)
            if movie_doc is None:
                # fallback: try to find by integer equality with movie['id'] when doc_id is str or unexpected
                movie_doc = next(
                    (
                        d
                        for d in self.documents
                        if cast(NumberId, d.get("id")) == doc_id
                    ),
                    None,
                )

            combined_results.append(
                {
                    "doc": movie_doc,
                    "rrf_score": rrf_score,
                    "bm25_rank": bm25_rank if bm25_rank is not None else "N/A",
                    "semantic_rank": semantic_rank
                    if semantic_rank is not None
                    else "N/A",
                }
            )

        # Return results sorted by rrf_score descending, limited to `limit` entries
        combined_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return combined_results[:limit]


def normalize_scores(scores: List[float]) -> List[float]:
    """Min-max normalize a list of numeric scores into [0.0, 1.0].

    Returns an empty list if `scores` is empty. If all scores are equal,
    returns a list of 1.0s to avoid division by zero (preserves ranking).
    """
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0 for _ in scores]  # Avoid division by zero; all scores are equal.
    return [(score - min_score) / (max_score - min_score) for score in scores]


def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    """Combine BM25 and semantic scores using weight alpha for BM25."""
    return (alpha * bm25_score) + ((1.0 - alpha) * semantic_score)
