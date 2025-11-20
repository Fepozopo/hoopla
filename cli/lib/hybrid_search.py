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
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query: str, k: int, limit: int = 10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize_scores(scores: list[int]):
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0 for _ in scores]  # Avoid division by zero; all scores are equal.
    return [(score - min_score) / (max_score - min_score) for score in scores]
