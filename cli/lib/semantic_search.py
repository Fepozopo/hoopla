#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure the project root (one level above the `cli` package) is on sys.path
# so `from cli.helpers import ...` works when running the script directly.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from cli.helpers import Movie, load_movies


# Define a class called SemanticSearch
class SemanticSearch:
    model: SentenceTransformer
    embeddings: np.ndarray | None
    documents: list[Movie] | None
    document_map: dict[int, Movie] | dict[int, Movie]

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        if text.strip() == "":
            raise ValueError("Input text cannot be empty or whitespace.")

        text_list = [text]  # The encode method expects a list of inputs
        embedding = self.model.encode(text_list)
        return embedding[0]

    def build_embeddings(self, documents: list[Movie]) -> np.ndarray:
        # Defensive guard: do not attempt to build embeddings for an empty corpus.
        if not documents:
            raise ValueError("Cannot build embeddings: documents list is empty.")

        # Create strings of "Title: Description" for each movie.
        movies = [f"{doc['title']}: {doc['description']}" for doc in documents]

        # Build embeddings and provide a clearer error if the encoder fails.
        try:
            self.embeddings = self.model.encode(movies, show_progress_bar=True)
        except Exception as e:
            raise RuntimeError(f"Failed to build embeddings: {e}") from e

        self.documents = documents
        self.document_map = {i: doc for i, doc in enumerate(documents)}

        # Ensure cache directory exists before saving; writing the cache is best-effort.
        cache_path = Path("cache/movie_embeddings.npy")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            np.save(cache_path, self.embeddings)
        except Exception as e:
            # Do not fail the entire operation if the cache cannot be written;
            # keep embeddings available in memory but notify the user.
            print(f"Warning: failed to write embeddings cache: {e}")

        # Ensure we always return an ndarray at runtime for callers / static analysis.
        assert isinstance(self.embeddings, np.ndarray)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[Movie]) -> np.ndarray:
        # Ensure callers supply a non-empty documents list. This prevents silent
        # mismatches where embeddings exist but the document map is empty.
        if not documents:
            raise ValueError(
                "Cannot load or create embeddings: documents list is empty."
            )

        cache_path = Path("cache/movie_embeddings.npy")

        if cache_path.exists():
            try:
                loaded = np.load(cache_path)
            except Exception as e:
                # If the cache cannot be read (corrupt file, permission issues, etc.)
                # rebuild from the provided documents and communicate the reason.
                print(f"Cached embeddings could not be read ({e}); rebuilding.")
                return self.build_embeddings(documents)

            # If the cached embeddings don't match the current documents, rebuild them.
            if (
                not isinstance(loaded, np.ndarray)
                or loaded.ndim != 2
                or loaded.shape[0] != len(documents)
            ):
                print(
                    "Cached embeddings do not match the current documents; rebuilding."
                )
                return self.build_embeddings(documents)

            # Cache looks valid: use it and wire up the in-memory document map.
            self.embeddings = loaded
            self.documents = documents
            self.document_map = {i: doc for i, doc in enumerate(documents)}
        else:
            # If cached embeddings are not found, build them.
            return self.build_embeddings(documents)

        # Runtime guard to ensure we return a concrete ndarray (not None).
        if self.embeddings is None:
            raise RuntimeError("Embeddings could not be loaded or built.")
        assert isinstance(self.embeddings, np.ndarray)
        return self.embeddings

    def search(self, query: str, limit: int) -> list[tuple[Movie, float]]:
        if self.embeddings is None or self.documents is None:
            raise RuntimeError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)
        similarities = [
            (idx, cosine_similarity(query_embedding, doc_embedding))
            for idx, doc_embedding in enumerate(self.embeddings)
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:limit]

        # Return the top results (up to limit) as a list of dictionaries
        return [(self.document_map[idx], score) for idx, score in top_results]


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[Movie]):
        all_chunks = []
        chunk_metadata = []

        for doc_id, doc in enumerate(documents):
            description = doc["description"]
            if not description:
                continue
            chunks = list(semantic_chunks(description, max_chunk_size=4, overlap=1))
            all_chunks.extend(chunks)
            total_chunks = len(chunks)
            for chunk_idx, _ in enumerate(chunks):
                chunk_metadata.append(
                    {
                        "movie_idx": doc_id,
                        "chunk_idx": chunk_idx,
                        "total_chunks": total_chunks,
                    }
                )

        try:
            self.chunk_embeddings = self.model.encode(
                all_chunks, show_progress_bar=True
            )
        except Exception as e:
            raise RuntimeError(f"Failed to build chunk embeddings: {e}") from e

        self.documents = documents
        self.chunk_metadata = chunk_metadata

        # Ensure cache directory exists before saving; writing the cache is best-effort.
        cache_dir = Path("cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            np.save(cache_dir / "chunk_embeddings.npy", self.chunk_embeddings)
        except Exception as e:
            # Do not fail the entire operation if the cache cannot be written;
            # keep embeddings available in memory but notify the user.
            print(f"Warning: failed to write chunk embeddings cache: {e}")

        try:
            import json

            with open(cache_dir / "chunk_metadata.json", "w") as f:
                json.dump(
                    {"chunks": chunk_metadata, "total_chunks": len(all_chunks)},
                    f,
                    indent=2,
                )
        except Exception as e:
            print(f"Warning: failed to write chunk metadata cache: {e}")

        # Ensure we always return an ndarray at runtime for callers / static analysis.
        assert isinstance(self.chunk_embeddings, np.ndarray)
        return self.chunk_embeddings


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

    # Extra runtime check for safety / static-analysis friendliness.
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


def cosine_similarity(vec1: float, vec2: float):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)


def fixed_size_chunks(text: str, size: int, overlap: int = 0):
    """Yield successive fixed-size chunks from text by grouping 'size'
    words together into a single string.
    """
    words = text.split()
    start = 0
    while start < len(words):
        end = start + size
        chunk = " ".join(words[start:end])
        yield chunk
        if start + size >= len(words):
            break
        start += size - overlap


def semantic_chunks(text: str, max_chunk_size: int = 4, overlap: int = 0):
    """Yield successive semantic chunks from text by grouping sentences.

    Each chunk will contain up to `max_chunk_size` sentences. Adjacent chunks
    will overlap by `overlap` sentences (the number of sentences, not words).

    Args:
        text: The input text to chunk.
        max_chunk_size: Maximum number of sentences per chunk (must be >= 1).
        overlap: Number of sentences to overlap between consecutive chunks
                 (must be >= 0 and < max_chunk_size).
    """
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be a positive integer.")
    if overlap < 0:
        raise ValueError("overlap must be a non-negative integer.")
    if overlap >= max_chunk_size:
        raise ValueError("overlap must be smaller than max_chunk_size.")

    # Split the input into individual sentences. Use \s+ after lookbehind to
    # handle newlines and multiple spaces. Strip out any empty entries.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    if not sentences:
        return

    # Step is how many sentences we advance for the next chunk.
    step = max_chunk_size - overlap

    # Slide a window of up to `max_chunk_size` sentences, advancing by `step`.
    for start in range(0, len(sentences), step):
        end = start + max_chunk_size
        chunk_sentences = sentences[start:end]
        if chunk_sentences:
            yield " ".join(chunk_sentences)
        if end >= len(sentences):
            break
