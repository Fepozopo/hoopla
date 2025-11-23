#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure the project root (one level above the `cli` package) is on sys.path
# so `from cli.helpers import ...` works when running the script directly.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse

from cli.helpers import load_movies
from cli.keyword_search_cli import CLIArgs
from cli.lib.hybrid_search import HybridSearch, normalize_scores
from cli.prompts.enhance import get_response


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        "normalize", help="Normalize a list of scores"
    )
    _ = normalize_parser.add_argument(
        "scores",
        type=float,
        nargs="+",
        help="List of scores to normalize",
    )
    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="Perform a weighted hybrid search"
    )
    _ = weighted_search_parser.add_argument("query", type=str, help="Search query")
    _ = weighted_search_parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weighting factor between keyword and semantic scores (default: 0.5)",
    )
    _ = weighted_search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of top results to return (default: 5)",
    )
    rrf_search_parser = subparsers.add_parser(
        "rrf-search", help="Perform a reciprocal rank fusion hybrid search"
    )
    _ = rrf_search_parser.add_argument("query", type=str, help="Search query")
    _ = rrf_search_parser.add_argument(
        "--k",
        type=int,
        default=60,
        help="Reciprocal rank fusion constant k (default: 60)",
    )
    _ = rrf_search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of top results to return (default: 5)",
    )
    _ = rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )

    # Use a typed namespace so static checkers know the types of attributes
    namespace = CLIArgs()
    _ = parser.parse_args(namespace=namespace)
    args: CLIArgs = namespace

    match args.command:
        case "normalize":
            scores = getattr(args, "scores")
            if scores is None:
                print("No scores provided for normalization.")
                return
            normalized = normalize_scores(scores)
            for score in normalized:
                print(f"* {score:.4f}")
        case "weighted-search":
            query = getattr(args, "query")
            alpha = getattr(args, "alpha")
            limit = getattr(args, "limit")
            if query is None:
                print("No query provided for weighted search.")
                return

            # Load movies from the repository data file and perform a hybrid weighted search.
            path_movies = Path(__file__).parent.parent / "data" / "movies.json"
            movies = load_movies(path_movies)
            if not movies:
                print(
                    f"No movies loaded from {path_movies}. Ensure the data file exists."
                )
                return

            hs = HybridSearch(movies)
            results = hs.weighted_search(query, alpha, limit)

            # Print formatted, truncated results
            for idx, item in enumerate(results, start=1):
                doc = item.get("doc") or {}
                title = doc.get("title", "(no title)")
                hybrid = item.get("hybrid_score", 0.0)
                bm25 = item.get("keyword_score", 0.0)
                semantic = item.get("semantic_score", 0.0)
                description = doc.get("description", "")
                # Truncate description to a reasonable length for display
                excerpt = description
                max_len = 100
                if len(description) > max_len:
                    excerpt = description[:max_len].rstrip() + "..."
                print(f"{idx}. {title}")
                print(f"   Hybrid Score: {hybrid:.3f}")
                print(f"   BM25: {bm25:.3f}, Semantic: {semantic:.3f}")
                if excerpt:
                    print(f"   {excerpt}")
        case "rrf-search":
            query = getattr(args, "query")
            k = getattr(args, "k")
            limit = getattr(args, "limit")
            method = getattr(args, "enhance")
            if query is None:
                print("No query provided for RRF search.")
                return
            if method is not None:
                enhanced_query = get_response(method, query)
                print(f"Enhanced query ({method}): '{query}' -> '{enhanced_query}'\n")
                if enhanced_query is not None:
                    query = enhanced_query

            # Load movies from the repository data file and perform a hybrid RRF search.
            path_movies = Path(__file__).parent.parent / "data" / "movies.json"
            movies = load_movies(path_movies)
            if not movies:
                print(
                    f"No movies loaded from {path_movies}. Ensure the data file exists."
                )
                return

            hs = HybridSearch(movies)
            results = hs.rrf_search(query, k, limit)

            # Print formatted, truncated results
            for idx, item in enumerate(results, start=1):
                doc = item.get("doc") or {}
                title = doc.get("title", "(no title)")
                rrf_score = item.get("rrf_score", 0.0)
                bm25_rank = item.get("bm25_rank", 0)
                semantic_rank = item.get("semantic_rank", 0)
                description = doc.get("description", "")
                # Truncate description to a reasonable length for display
                excerpt = description
                max_len = 100
                if len(description) > max_len:
                    excerpt = description[:max_len].rstrip() + "..."

                print(f"{idx}. {title}")
                print(f"   RRF Score: {rrf_score:.3f}")
                print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
                if excerpt:
                    print(f"   {excerpt}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
