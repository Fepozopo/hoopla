#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure the project root (one level above the `cli` package) is on sys.path
# so `from cli.helpers import ...` works when running the script directly.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import json

from cli.helpers import load_movies
from cli.lib.hybrid_search import HybridSearch


def _normalize_title(t: str) -> str:
    return t.strip().lower()


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # Load the golden_dataset.json file using the json.load function
    golden_dataset_path = (
        Path(__file__).resolve().parent.parent / "data" / "golden_dataset.json"
    )
    try:
        with open(golden_dataset_path, "r", encoding="utf-8") as f:
            golden_data = json.load(f)
    except FileNotFoundError:
        print(f"Golden dataset not found: {golden_dataset_path}")
        return
    except json.JSONDecodeError as e:
        print(f"Failed to parse golden dataset: {e}")
        return

    test_cases = golden_data.get("test_cases", [])
    if not isinstance(test_cases, list) or not test_cases:
        print("No test cases found in golden dataset.")
        return

    # Load movies and initialize HybridSearch
    path_movies = Path(__file__).resolve().parent.parent / "data" / "movies.json"
    movies = load_movies(path_movies)
    if not movies:
        print("No movies loaded; ensure data/movies.json exists and is valid.")
        return

    hs = HybridSearch(movies)

    # Print header showing the evaluation k
    print(f"k={limit}\n")

    RRF_K = 60  # Fixed RRF k parameter

    for case in test_cases:
        query = case.get("query", "")
        relevant_list = case.get("relevant_docs", []) or []

        # Run RRF search: k parameter fixed, top-k equals the --limit flag
        rrf_results = hs.rrf_search(query, k=RRF_K, limit=limit)

        # Extract retrieved titles in order
        retrieved_titles: list[str] = []
        for item in rrf_results:
            doc = item.get("doc") if isinstance(item, dict) else None
            title = None
            if isinstance(doc, dict):
                title = doc.get("title")
            if not title:
                title = "(no title)"
            retrieved_titles.append(title)

        # Compute precision: fraction of retrieved titles (out of `limit`) that are in the golden relevant set.
        relevant_set = {
            _normalize_title(t) for t in relevant_list if isinstance(t, str)
        }
        match_count = sum(
            1 for t in retrieved_titles if _normalize_title(t) in relevant_set
        )

        precision = (match_count / limit) if limit > 0 else 0.0

        # Format lists for printing
        retrieved_str = ", ".join(retrieved_titles) if retrieved_titles else "(none)"
        relevant_str = ", ".join(relevant_list) if relevant_list else "(none)"

        # Print result block
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Retrieved: {retrieved_str}")
        print(f"  - Relevant: {relevant_str}\n")


if __name__ == "__main__":
    main()
