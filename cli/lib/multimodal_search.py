#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure the project root (one level above the `cli` package) is on sys.path
# so `from cli.helpers import ...` works when running the script directly.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from typing import Any, cast

from PIL import Image
from sentence_transformers import SentenceTransformer


class MultimodalSearch:
    def __init__(self, model_name: str = "clip-ViT-B-32"):
        """
        Initialize the multimodal search helper with a SentenceTransformer model.
        The model is stored on the instance as `self.model`.
        """
        self.model = SentenceTransformer(model_name)

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
