#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure the project root (one level above the `cli` package) is on sys.path
# so `from cli.helpers import ...` works when running the script directly.
sys.path.append(str(Path(__file__).resolve().parent.parent))

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
        # Create strings of "Title: Description" for each movie.
        movies = [f"{doc['title']}: {doc['description']}" for doc in documents]
        self.embeddings = self.model.encode(movies, show_progress_bar=True)
        self.documents = documents
        self.document_map = {i: doc for i, doc in enumerate(documents)}

        np.save("cache/movie_embeddings.npy", self.embeddings)
        # Ensure we always return an ndarray at runtime for callers / static analysis.
        assert isinstance(self.embeddings, np.ndarray)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[Movie]) -> np.ndarray:
        try:
            self.embeddings = np.load("cache/movie_embeddings.npy")
            self.documents = documents
            self.document_map = {i: doc for i, doc in enumerate(documents)}
        except FileNotFoundError:
            # If cached embeddings are not found, build them.
            self.build_embeddings(documents)

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

        return [(self.document_map[idx], score) for idx, score in top_results]


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
