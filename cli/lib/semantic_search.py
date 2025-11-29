#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure the project root (one level above the `cli` package) is on sys.path
# so `from cli.helpers import ...` works when running the script directly.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import re

import numpy as np
from sentence_transformers import SentenceTransformer

from cli.helpers import Movie, load_movies


class SemanticSearch:
    """
    Basic semantic search that creates and queries embedding vectors for whole documents.
    """

    model: SentenceTransformer
    embeddings: np.ndarray | None
    documents: list[Movie] | None
    document_map: dict[int, Movie]

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str) -> np.ndarray:
        if text is None or text.strip() == "":
            raise ValueError("Input text cannot be empty or whitespace.")
        # SentenceTransformer.encode accepts lists; return the first vector.
        vecs = self.model.encode([text])
        return vecs[0]

    def build_embeddings(self, documents: list[Movie]) -> np.ndarray:
        if not documents:
            raise ValueError("Cannot build embeddings: documents list is empty.")

        # Build text inputs from the document fields to embed.
        texts = [
            f"{doc.get('title', '')}: {doc.get('description', '') or ''}"
            for doc in documents
        ]

        try:
            self.embeddings = self.model.encode(texts, show_progress_bar=True)
        except Exception as e:
            raise RuntimeError(f"Failed to build embeddings: {e}") from e

        self.documents = documents
        # document_map maps numeric index -> movie dict (used by index-based methods)
        self.document_map = {i: doc for i, doc in enumerate(documents)}

        # Try to persist embeddings to cache; failure should not stop execution.
        cache_path = Path("cache/movie_embeddings.npy")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            np.save(cache_path, self.embeddings)
        except Exception as e:
            print(f"Warning: failed to write embeddings cache: {e}")

        assert isinstance(self.embeddings, np.ndarray)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[Movie]) -> np.ndarray:
        if not documents:
            raise ValueError(
                "Cannot load or create embeddings: documents list is empty."
            )

        cache_path = Path("cache/movie_embeddings.npy")
        if cache_path.exists():
            try:
                loaded = np.load(cache_path)
            except Exception as e:
                print(f"Cached embeddings could not be read ({e}); rebuilding.")
                return self.build_embeddings(documents)

            # Basic validation: must be 2D and rows should match number of documents
            if (
                not isinstance(loaded, np.ndarray)
                or loaded.ndim != 2
                or loaded.shape[0] != len(documents)
            ):
                print("Cached embeddings do not match current documents; rebuilding.")
                return self.build_embeddings(documents)

            self.embeddings = loaded
            self.documents = documents
            self.document_map = {i: doc for i, doc in enumerate(documents)}
        else:
            return self.build_embeddings(documents)

        if self.embeddings is None:
            raise RuntimeError("Embeddings could not be loaded or built.")
        assert isinstance(self.embeddings, np.ndarray)
        return self.embeddings

    def search(self, query: str, limit: int = 10) -> list[tuple[Movie, float]]:
        if self.embeddings is None or self.documents is None:
            raise RuntimeError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)
        similarities = [
            (idx, cosine_similarity(query_embedding, emb))
            for idx, emb in enumerate(self.embeddings)
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        top = similarities[:limit]
        return [(self.document_map[idx], score) for idx, score in top]


class ChunkedSemanticSearch(SemanticSearch):
    """
    Semantic search which chunks document descriptions into smaller pieces,
    builds embeddings for chunks, and aggregates chunk-level scores into
    document-level scores.

    Key change: chunk metadata stores the persistent `movie_id` and
    `search_chunks` returns `id` as the persistent `movie_id`. Aggregation
    is done with `movie_id` as the key so results are compatible with BM25's
    persistent document ids.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[Movie]) -> np.ndarray:
        if not documents:
            raise ValueError("Cannot build chunk embeddings: documents list is empty.")

        # Keep both index-based map and persistent-id aware document map.
        self.documents = documents
        self.document_map = {i: doc for i, doc in enumerate(documents)}
        # doc_map_by_id for fast lookup by persistent id
        # doc_map_by_id = {doc.get("id"): doc for doc in documents}

        all_chunks: list[str] = []
        chunk_metadata: list[dict] = []

        for doc_idx, doc in enumerate(documents):
            description = doc.get("description", "") or ""
            chunks = list(semantic_chunks(description, max_chunk_size=4, overlap=1))
            if not chunks:
                continue
            total_chunks = len(chunks)
            # append chunk text and metadata for each chunk
            for chunk_idx, chunk_text in enumerate(chunks):
                all_chunks.append(chunk_text)
                chunk_metadata.append(
                    {
                        "movie_idx": doc_idx,  # position in documents list
                        "movie_id": doc.get("id"),  # persistent id
                        "chunk_idx": chunk_idx,
                        "total_chunks": total_chunks,
                    }
                )

        if not all_chunks:
            raise ValueError(
                "No chunks produced from documents; cannot build chunk embeddings."
            )

        try:
            self.chunk_embeddings = self.model.encode(
                all_chunks, show_progress_bar=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to build chunk embeddings: {e}") from e

        self.chunk_metadata = chunk_metadata

        # persist chunk embeddings and metadata when possible
        cache_dir = Path("cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            np.save(cache_dir / "chunk_embeddings.npy", self.chunk_embeddings)
        except Exception as e:
            print(f"Warning: failed to write chunk embeddings cache: {e}")

        try:
            with open(cache_dir / "chunk_metadata.json", "w", encoding="utf-8") as f:
                json.dump(
                    {"chunks": chunk_metadata, "total_chunks": len(all_chunks)},
                    f,
                    indent=2,
                )
        except Exception as e:
            print(f"Warning: failed to write chunk metadata cache: {e}")

        assert isinstance(self.chunk_embeddings, np.ndarray)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[Movie]) -> np.ndarray:
        if not documents:
            raise ValueError(
                "Cannot load or create chunk embeddings: documents list is empty."
            )

        self.documents = documents
        self.document_map = {i: doc for i, doc in enumerate(documents)}

        cache_dir = Path("cache")
        embeddings_path = cache_dir / "chunk_embeddings.npy"
        metadata_path = cache_dir / "chunk_metadata.json"

        if embeddings_path.exists() and metadata_path.exists():
            try:
                loaded_embeddings = np.load(embeddings_path)
            except Exception as e:
                print(f"Cached chunk embeddings could not be read ({e}); rebuilding.")
                return self.build_chunk_embeddings(documents)

            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"Cached chunk metadata could not be read ({e}); rebuilding.")
                return self.build_chunk_embeddings(documents)

            # Basic validations
            if (
                not isinstance(loaded_embeddings, np.ndarray)
                or loaded_embeddings.ndim != 2
            ):
                print(
                    "Cached chunk embeddings do not match expected format; rebuilding."
                )
                return self.build_chunk_embeddings(documents)

            total_chunks_meta = metadata.get("total_chunks")
            if (
                total_chunks_meta is not None
                and int(total_chunks_meta) != loaded_embeddings.shape[0]
            ):
                print(
                    "Cached chunk embeddings length does not match metadata; rebuilding."
                )
                return self.build_chunk_embeddings(documents)

            self.chunk_embeddings = loaded_embeddings
            # Ensure chunk list structure is present
            self.chunk_metadata = metadata.get("chunks", [])
        else:
            return self.build_chunk_embeddings(documents)

        if self.chunk_embeddings is None:
            raise RuntimeError("Chunk embeddings could not be loaded or built.")
        assert isinstance(self.chunk_embeddings, np.ndarray)
        return self.chunk_embeddings

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        """
        Search chunk embeddings with the given query and return a list of formatted
        results keyed by persistent `movie_id`. Each returned item is a dict:
          {
            "id": <persistent movie id>,
            "title": <title>,
            "document": <truncated description>,
            "score": <aggregated score>,
            "metadata": { ... }
          }
        Aggregation uses the best chunk score per persistent movie_id.
        """
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise RuntimeError(
                "No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first."
            )

        # Use this instance to generate the query embedding (avoids reloading model)
        query_embedding = self.generate_embedding(query)

        # Build chunk_scores including persistent movie_id
        chunk_scores: list[dict] = []
        for idx, chunk_emb in enumerate(self.chunk_embeddings):
            # defensive: metadata list might be shorter if cache mismatch; guard with try/except
            try:
                chunk_info = self.chunk_metadata[idx]
            except Exception:
                # skip entries without metadata
                continue
            score = float(cosine_similarity(query_embedding, chunk_emb))
            chunk_scores.append(
                {
                    "chunk_idx": chunk_info.get("chunk_idx"),
                    "movie_idx": chunk_info.get("movie_idx"),
                    "movie_id": chunk_info.get("movie_id"),
                    "score": score,
                }
            )

        # Aggregate best score per persistent movie_id (preferred) or by movie_idx fallback
        movie_scores: dict[int, float] = {}
        # Also keep lists of matching chunks per movie_id for metadata
        chunks_by_movie: dict[int, list[dict]] = {}

        for c in chunk_scores:
            movie_id = c.get("movie_id")
            if movie_id is None:
                # Fall back to movie_idx if metadata lacks movie_id
                movie_id = c.get("movie_idx")

            if movie_id is None:
                # Skip chunk if no identifier is available
                continue

            score = c["score"]
            # track matching chunks
            chunks_by_movie.setdefault(movie_id, []).append(c)

            # keep the best score per movie_id
            if (movie_id not in movie_scores) or (score > movie_scores[movie_id]):
                movie_scores[movie_id] = score

        # Sort results by aggregated score descending
        sorted_movie_scores = sorted(
            movie_scores.items(), key=lambda x: x[1], reverse=True
        )
        top_results = sorted_movie_scores[:limit]

        SCORE_PRECISION = 4
        # Build a fast lookup by persistent id for document info
        doc_map_by_id = {doc.get("id"): doc for doc in (self.documents or [])}

        formatted_results: list[dict] = []
        for movie_id, score in top_results:
            doc = doc_map_by_id.get(movie_id, {}) or {}
            title = doc.get("title", "")
            description = doc.get("description", "") or ""
            truncated = description[:100]

            matching_chunks = chunks_by_movie.get(movie_id, [])
            metadata = {}
            if matching_chunks:
                best_chunk = max(matching_chunks, key=lambda x: x.get("score", 0.0))
                metadata = {
                    "best_chunk_idx": best_chunk.get("chunk_idx"),
                    "best_chunk_score": round(
                        best_chunk.get("score", 0.0), SCORE_PRECISION
                    ),
                    "matched_chunks": len(matching_chunks),
                }

            formatted_results.append(
                {
                    "id": movie_id,  # persistent movie id (compatible with BM25 outputs)
                    "title": title,
                    "document": truncated,
                    "score": round(float(score), SCORE_PRECISION),
                    "metadata": metadata or {},
                }
            )

        return formatted_results


# Utility functions ------------------------------------------------------------


def verify_model():
    try:
        semantic_search = SemanticSearch()
        print(f"Model loaded: {semantic_search.model}")
        print(f"Max sequence length: {semantic_search.model.max_seq_length}")
    except Exception as e:
        print(f"Error loading model: {e}")


def embed_text(text: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    # Create an instance of SemanticSearch.
    semantic_search = SemanticSearch()

    # Load the documents from movies.json into a list.
    path_movies = Path("data/movies.json")
    documents = load_movies(path_movies)

    # Build or load embeddings for the documents.
    embeddings = semantic_search.load_or_create_embeddings(documents)

    if embeddings is None:
        print("No embeddings available.")
        return

    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return float(dot_product / (norm_vec1 * norm_vec2))


def fixed_size_chunks(text: str, size: int, overlap: int = 0):
    """Yield successive fixed-size chunks from text by grouping 'size' words."""
    words = text.split()
    start = 0
    while start < len(words):
        end = start + size
        yield " ".join(words[start:end])
        if start + size >= len(words):
            break
        start += size - overlap


def semantic_chunks(text: str, max_chunk_size: int = 4, overlap: int = 0):
    """
    Yield successive semantic chunks from text by grouping sentences.

    Each chunk will contain up to `max_chunk_size` sentences. Adjacent chunks
    will overlap by `overlap` sentences (the number of sentences, not words).
    """
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be a positive integer.")
    if overlap < 0 or overlap >= max_chunk_size:
        raise ValueError("overlap must be between 0 and max_chunk_size-1.")

    # Split the input into individual sentences using punctuation boundaries.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    if not sentences:
        return

    step = max_chunk_size - overlap
    for start in range(0, len(sentences), step):
        end = start + max_chunk_size
        chunk_sentences = sentences[start:end]
        if chunk_sentences:
            yield " ".join(chunk_sentences)
        if end >= len(sentences):
            break
