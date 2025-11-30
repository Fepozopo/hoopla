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
from cli.lib.hybrid_search import HybridSearch
from cli.prompts.augment import (
    ai_augment_citations,
    ai_augment_question,
    ai_augment_rag,
    ai_augment_summarize,
)


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    _ = rag_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarize documents using AI"
    )
    _ = summarize_parser.add_argument(
        "query", type=str, help="Search query for RRF search"
    )
    _ = summarize_parser.add_argument(
        "--limit", type=int, default=5, help="Number of top documents to summarize"
    )
    citations_parser = subparsers.add_parser(
        "citations", help="Generate citations for a given text"
    )
    _ = citations_parser.add_argument(
        "query", type=str, help="Search query for RRF search"
    )
    _ = citations_parser.add_argument(
        "--limit", type=int, default=5, help="Number of top documents to cite"
    )
    question_parser = subparsers.add_parser(
        "question", help="Answer questions using AI"
    )
    _ = question_parser.add_argument("question", type=str, help="Question to answer")
    _ = question_parser.add_argument(
        "--limit", type=int, default=5, help="Number of top documents to use"
    )

    # Use a typed namespace so static checkers know the types of attributes
    namespace = CLIArgs()
    _ = parser.parse_args(namespace=namespace)
    args: CLIArgs = namespace

    match args.command:
        case "rag":
            load_dotenv()
            debug = os.environ.get("DEBUG", "0") == "1"
            query = getattr(args, "query")
            k = 10
            limit = 5
            if query is None:
                print("No query provided for RRF search.")
                return

            # Load movies from the repository data file and perform a hybrid RRF search.
            path_movies = Path(__file__).parent.parent / "data" / "movies.json"
            movies = load_movies(path_movies)
            if not movies:
                print(
                    f"No movies loaded from {path_movies}. Ensure the data file exists."
                )
                return

            hs = HybridSearch(movies)
            docs = hs.rrf_search(query, k, limit)

            response = ai_augment_rag(query, docs)

            print("Search Results:")
            for _, item in enumerate(docs[:limit], start=1):
                doc = item.get("doc") or {}
                title = doc.get("title", "(no title)")
                rrf_score = item.get("rrf_score", 0.0)
                bm25_rank = item.get("bm25_rank", 0)
                semantic_rank = item.get("semantic_rank", 0)

                print(f"   - {title}")
                if debug:
                    print(f"        RRF Score: {rrf_score:.4f}")
                    print(
                        f"        BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}"
                    )

            print("RAG Response:")
            print(f"{response}")
        case "summarize":
            load_dotenv()
            debug = os.environ.get("DEBUG", "0") == "1"
            query = getattr(args, "query")
            k = 10
            limit = getattr(args, "limit")
            if limit <= 0:
                print("Limit must be a positive integer.")
                return
            if not limit:
                limit = 5
            if query is None:
                print("No query provided for RRF search.")
                return

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

            response = ai_augment_summarize(query, results)

            print("Search Results:")
            for _, item in enumerate(results[:limit], start=1):
                doc = item.get("doc") or {}
                title = doc.get("title", "(no title)")
                rrf_score = item.get("rrf_score", 0.0)
                bm25_rank = item.get("bm25_rank", 0)
                semantic_rank = item.get("semantic_rank", 0)

                print(f"   - {title}")
                if debug:
                    print(f"        RRF Score: {rrf_score:.4f}")
                    print(
                        f"        BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}"
                    )

            print("LLM Summary:")
            print(f"{response}")
        case "citations":
            load_dotenv()
            debug = os.environ.get("DEBUG", "0") == "1"
            query = getattr(args, "query")
            k = 60
            limit = getattr(args, "limit")
            if limit <= 0:
                print("Limit must be a positive integer.")
                return
            if not limit:
                limit = 5
            if query is None:
                print("No query provided for RRF search.")
                return

            # Load movies from the repository data file and perform a hybrid RRF search.
            path_movies = Path(__file__).parent.parent / "data" / "movies.json"
            movies = load_movies(path_movies)
            if not movies:
                print(
                    f"No movies loaded from {path_movies}. Ensure the data file exists."
                )
                return

            hs = HybridSearch(movies)
            documents = hs.rrf_search(query, k, limit)

            response = ai_augment_citations(query, documents)

            print("Search Results:")
            for _, item in enumerate(documents[:limit], start=1):
                doc = item.get("doc") or {}
                title = doc.get("title", "(no title)")
                rrf_score = item.get("rrf_score", 0.0)
                bm25_rank = item.get("bm25_rank", 0)
                semantic_rank = item.get("semantic_rank", 0)

                print(f"   - {title}")
                if debug:
                    print(f"        RRF Score: {rrf_score:.4f}")
                    print(
                        f"        BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}"
                    )

            print("LLM Answer:")
            print(f"{response}")
        case "question":
            load_dotenv()
            debug = os.environ.get("DEBUG", "0") == "1"
            question = getattr(args, "question")
            k = 60
            limit = getattr(args, "limit")
            if limit <= 0:
                print("Limit must be a positive integer.")
                return
            if not limit:
                limit = 5
            if question is None:
                print("No query provided for RRF search.")
                return

            # Load movies from the repository data file and perform a hybrid RRF search.
            path_movies = Path(__file__).parent.parent / "data" / "movies.json"
            movies = load_movies(path_movies)
            if not movies:
                print(
                    f"No movies loaded from {path_movies}. Ensure the data file exists."
                )
                return

            hs = HybridSearch(movies)
            documents = hs.rrf_search(question, k, limit)

            response = ai_augment_question(question, documents)

            print("Search Results:")
            for _, item in enumerate(documents[:limit], start=1):
                doc = item.get("doc") or {}
                title = doc.get("title", "(no title)")
                rrf_score = item.get("rrf_score", 0.0)
                bm25_rank = item.get("bm25_rank", 0)
                semantic_rank = item.get("semantic_rank", 0)

                print(f"   - {title}")
                if debug:
                    print(f"        RRF Score: {rrf_score:.4f}")
                    print(
                        f"        BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}"
                    )

            print("Answer:")
            print(f"{response}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
