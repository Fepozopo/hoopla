#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure the project root (one level above the `cli` package) is on sys.path
# so `from cli.helpers import ...` works when running the script directly.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Any, cast

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from cli.helpers import Movie, load_movies


class MultimodalSearch:
    def __init__(
        self, model_name: str = "clip-ViT-B-32", documents: list[Movie] | None = None
    ):
        """
        Initialize the multimodal search helper with a SentenceTransformer model.
        The model is stored on the instance as `self.model`.

        Also accepts an optional list of `documents` (movie dicts). When
        provided, builds `self.texts` and `self.text_embeddings` for image-to-text
        similarity searches.
        """
        self.model = SentenceTransformer(model_name)

        # Documents provided by the caller (list of Movie TypedDicts)
        self.documents: list[Movie] = documents or []

        # Build textual representations for each document in the form:
        # "{title}: {description}"
        self.texts: list[str] = [
            f"{doc.get('title', '')}: {doc.get('description', '') or ''}"
            for doc in self.documents
        ]

        # Pre-compute text embeddings for faster image->text similarity searches.
        # If there are no texts, keep an empty numpy array.
        if self.texts:
            try:
                self.text_embeddings = self.model.encode(
                    self.texts, show_progress_bar=True
                )
            except Exception as e:
                # If encoding fails, fall back to an empty embeddings array but keep going.
                print(f"Warning: failed to build text embeddings: {e}")
                self.text_embeddings = np.array([])
        else:
            self.text_embeddings = np.array([])

    def embed_image(self, image_path: Path):
        """
        Generate an embedding for the image at `image_path`.

        Loads the image using PIL.Image.open, converts to RGB, then passes a
        one-element list containing the image to the model's `encode` method.
        Returns the first (and only) embedding from the resulting list.
        """
        image = Image.open(image_path).convert("RGB")
        # Pass the PIL Image directly so the SentenceTransformer CLIP model treats
        # it as an image input instead of attempting to tokenize it as text.
        embeddings = self.model.encode([cast(Any, image)])
        return embeddings[0]

    def search_with_image(self, image_path: Path, limit: int = 5) -> list[dict]:
        """
        Search the loaded text documents for those most similar to the provided image.

        Returns a list of dicts:
          {
            "id": <document id>,
            "title": <title>,
            "description": <description or None>,
            "similarity": <float similarity score>
          }

        Results are sorted by similarity descending and limited to `limit`.
        """
        # If there are no documents or no text embeddings available, return empty list.
        if (
            not self.documents
            or self.text_embeddings is None
            or len(self.text_embeddings) == 0
        ):
            return []

        # Generate embedding for the query image
        image_embedding = self.embed_image(image_path)

        # Compute cosine similarity between image embedding and each text embedding
        def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
            dot_product = np.dot(vec1, vec2)
            norm_vec1 = np.linalg.norm(vec1)
            norm_vec2 = np.linalg.norm(vec2)
            if norm_vec1 == 0 or norm_vec2 == 0:
                return 0.0
            return float(dot_product / (norm_vec1 * norm_vec2))

        scores: list[tuple[int, float]] = []
        for idx, text_emb in enumerate(self.text_embeddings):
            try:
                score = float(cosine_similarity(image_embedding, text_emb))
            except Exception:
                # Defensive: if any embedding is malformed, skip it.
                continue
            scores.append((idx, score))

        # Sort by score descending and take the top `limit`
        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:limit]

        # Format results mapping back to original document fields
        results: list[dict] = []
        for idx, score in top:
            try:
                doc = self.documents[idx]
            except Exception:
                continue
            results.append(
                {
                    "id": doc.get("id"),
                    "title": doc.get("title"),
                    "description": doc.get("description"),
                    "similarity": float(score),
                }
            )

        return results


def verify_image_embedding(image_path: Path):
    """
    Top-level helper that creates a `MultimodalSearch` instance, generates an
    embedding for the provided image, and prints its dimensionality in the
    required format.
    """
    ms = MultimodalSearch()
    embedding = ms.embed_image(image_path)
    try:
        dims = embedding.shape[0]
    except Exception:
        # Fallback for list-like embeddings
        dims = len(embedding)
    print(f"Embedding shape: {dims} dimensions")


def image_search_command(image_path: Path) -> list[dict]:
    """
    Top-level helper to perform an image->text search against the movie dataset.

    Loads the movie dataset, creates a `MultimodalSearch` instance initialized
    with the documents, runs an image search, and returns the top results.
    """
    # Load movies from the repository data file
    path_movies = Path("data/movies.json")
    documents = load_movies(path_movies)
    if not documents:
        return []

    ms = MultimodalSearch(documents=documents)
    results = ms.search_with_image(image_path, limit=5)
    return results
