#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure the project root (one level above the `cli` package) is on sys.path
# so `from cli.helpers import ...` works when running the script directly.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from cli.helpers import InvertedIndex
from cli.lib.semantic_search import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
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

        - Request an oversampled set of candidates from both BM25 and chunked semantic search
          (limit * 500) to ensure sufficient overlap.
        - Normalize both score sets via min-max normalization.
        - Combine scores per-document using the provided alpha weight for BM25.
        - Return the top `limit` results sorted by the hybrid score.
        """
        # Get BM25 results (movies list and a dict of id->score)
        bm25_docs, bm25_scores = self._bm25_search(query, limit * 500)

        # Use chunked semantic search to get per-document semantic scores.
        # `search_chunks` returns formatted dicts with 'id' (movie index) and 'score'.
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        # Normalize representations to list[(doc_id, score)]
        bm25_pairs = [(doc["id"], bm25_scores.get(doc["id"], 0.0)) for doc in bm25_docs]
        bm25_scores_list = [score for _, score in bm25_pairs]

        # Convert semantic formatted results into (actual_doc_id, score) pairs.
        # Chunked results return 'id' as the movie index (position in self.documents),
        # so map that to the real movie id when possible.
        semantic_pairs: list[tuple[int, float]] = []
        for item in semantic_results:
            # item is expected to be a dict like {"id": movie_idx, "score": score, ...}
            movie_idx = item.get("id")
            score = item.get("score", 0.0)
            doc_id = None
            if isinstance(movie_idx, int) and 0 <= movie_idx < len(self.documents):
                doc = self.documents[movie_idx]
                doc_id = doc.get("id")
            else:
                # Fallback: if `id` already reflects the real doc id or is unexpected,
                # use it directly.
                doc_id = movie_idx
            semantic_pairs.append((doc_id, float(score)))

        semantic_scores_list = [score for _, score in semantic_pairs]

        # Normalize both score lists into [0.0, 1.0]
        normalized_bm25 = normalize_scores(bm25_scores_list)
        normalized_semantic = normalize_scores(semantic_scores_list)

        # Build a map of all documents by id for easy lookup
        doc_map = {doc["id"]: doc for doc in self.documents}

        # Create maps from doc_id -> normalized score for bm25 and semantic results.
        # Use zip to align the normalized scores with the original ordered pairs.
        bm25_norm_map = {
            doc_id: score for (doc_id, _), score in zip(bm25_pairs, normalized_bm25)
        }
        semantic_norm_map = {
            doc_id: score
            for (doc_id, _), score in zip(semantic_pairs, normalized_semantic)
        }

        # Collect the union of candidate document ids
        candidate_ids = set(bm25_norm_map.keys()) | set(semantic_norm_map.keys())

        # Build list of combined result dicts
        combined_results = []
        for doc_id in candidate_ids:
            kw_score = bm25_norm_map.get(doc_id, 0.0)
            sem_score = semantic_norm_map.get(doc_id, 0.0)
            hyb = hybrid_score(kw_score, sem_score, alpha)
            combined_results.append(
                {
                    "doc": doc_map.get(doc_id),
                    "keyword_score": kw_score,
                    "semantic_score": sem_score,
                    "hybrid_score": hyb,
                }
            )

        # Return results sorted by hybrid_score descending, limited to `limit` entries
        combined_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return combined_results[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10):
        """Perform Reciprocal Rank Fusion (RRF) search combining BM25 and chunked semantic scores.

        - Request an oversampled set of candidates from both BM25 and chunked semantic search
          (limit * 500) to ensure sufficient overlap.
        """
        # Get BM25 results (movies list and a dict of id->score)
        bm25_docs, _ = self._bm25_search(query, limit * 500)

        # Use chunked semantic search to get per-document semantic scores.
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)

        # Build rank maps: doc_id -> rank (1-based)
        bm25_rank_map = {doc["id"]: rank + 1 for rank, doc in enumerate(bm25_docs)}
        semantic_rank_map = {}
        for rank, item in enumerate(semantic_results):
            movie_idx = item.get("id")
            doc_id = None
            if isinstance(movie_idx, int) and 0 <= movie_idx < len(self.documents):
                doc = self.documents[movie_idx]
                doc_id = doc.get("id")
            else:
                doc_id = movie_idx
            semantic_rank_map[doc_id] = rank + 1

        # Collect the union of candidate document ids
        candidate_ids = set(bm25_rank_map.keys()) | set(semantic_rank_map.keys())

        # Build list of combined result dicts
        combined_results = []
        for doc_id in candidate_ids:
            bm25_rank = bm25_rank_map.get(doc_id)
            semantic_rank = semantic_rank_map.get(doc_id)

            rrf_score = 0.0
            if bm25_rank is not None:
                rrf_score += 1.0 / (k + bm25_rank)
            if semantic_rank is not None:
                rrf_score += 1.0 / (k + semantic_rank)

            combined_results.append(
                {
                    "doc": next((d for d in self.documents if d["id"] == doc_id), None),
                    "rrf_score": rrf_score,
                    "bm25_rank": bm25_rank,
                    "semantic_rank": semantic_rank,
                }
            )

        # Return results sorted by rrf_score descending, limited to `limit` entries
        combined_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return combined_results[:limit]


def normalize_scores(scores: list[float]) -> list[float]:
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
