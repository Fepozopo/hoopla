#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure the project root (one level above the `cli` package) is on sys.path
# so `from cli.helpers import ...` works when running the script directly.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import os

from dotenv import load_dotenv

from cli.helpers import load_movies
from cli.keyword_search_cli import CLIArgs
from cli.lib.hybrid_search import HybridSearch, normalize_scores
from cli.prompts.enhance import ai_enhance
from cli.prompts.rerank_method import ai_rerank_method


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
    _ = rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Reranking method to apply after RRF",
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
            if limit <= 0:
                print("Limit must be a positive integer.")
                return
            else:
                limit = limit * 500
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
            load_dotenv()
            debug = os.environ.get("DEBUG", "0") == "1"
            query = getattr(args, "query")
            k = getattr(args, "k")
            limit = getattr(args, "limit")
            method = getattr(args, "enhance")
            rerank_method = getattr(args, "rerank_method")

            if query is None:
                print("No query provided for RRF search.")
                return

            if limit <= 0:
                print("Limit must be a positive integer.")
                return
            # Determine RRF limit based on whether reranking will be applied
            rrf_limit = 0
            if rerank_method is not None:
                rrf_limit = limit * 5
            else:
                rrf_limit = limit * 500

            if debug:
                print(
                    f"RRF Search Parameters:\nquery='{query}'\nk={k}\nlimit={limit}\nrrf_limit={rrf_limit}\nenhance={method}\nrerank_method={rerank_method}\n\n"
                )

            if method is not None:
                enhanced_query = ai_enhance(method, query)
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
            results = hs.rrf_search(query, k, rrf_limit)

            if rerank_method is not None:
                if debug:
                    print(f"Raw RRF results for query '{query}':")
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
                        print(
                            f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}"
                        )
                        if excerpt:
                            print(f"   {excerpt}")
                    print("\n\n")
                reranked_results = ai_rerank_method(rerank_method, query, results)
                if reranked_results is not None:
                    results = reranked_results
                    # Print formatted, truncated results
                    print(
                        f"Reranking top {limit} results using {rerank_method} method..."
                    )
                    print(f"Reciprocal Rank Fusion Results for '{query}' (k={k})")
                    for idx, item in enumerate(results[:limit], start=1):
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

                        # Get rerank info
                        rerank_title = ""
                        rerank_result = ""
                        rerank_score = item.get("rerank_score", 0.0)
                        rerank_rank = item.get("rerank_rank", 0)
                        rerank_cs_score = item.get("rerank_cs_score", 0.0)
                        if rerank_score is not None:
                            rerank_title = "Rerank Score"
                            rerank_result = f"{rerank_score:.3f}/10"
                        if rerank_rank is not None:
                            rerank_title = "Rerank Rank"
                            rerank_result = f"{rerank_rank}"
                        if rerank_cs_score is not None:
                            rerank_title = "Cross Encoder Score"
                            rerank_result = f"{rerank_cs_score:.3f}"

                        print(f"{idx}. {title}")
                        print(f"   {rerank_title}: {rerank_result}")
                        print(f"   RRF Score: {rrf_score:.3f}")
                        print(
                            f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}"
                        )
                        if excerpt:
                            print(f"   {excerpt}")
            else:
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
