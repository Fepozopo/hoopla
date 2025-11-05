#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure the project root (one level above the `cli` package) is on sys.path
# so `from cli.helpers import ...` works when running the script directly.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse

from cli.keyword_search_cli import CLIArgs
from cli.lib.semantic_search import (
    SemanticSearch,
    embed_text,
    verify_embeddings,
    verify_model,
)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    _ = subparsers.add_parser("verify", help="Verify the semantic search model")
    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Generate embedding for a given text"
    )
    _ = embed_text_parser.add_argument("text", type=str, help="Text to embed")
    _ = subparsers.add_parser(
        "verify_embeddings", help="Verify the embeddings against cached values"
    )
    embedquery_parser = subparsers.add_parser(
        "embedquery", help="Generate embedding for a search query"
    )
    _ = embedquery_parser.add_argument("query", type=str, help="Search query to embed")
    search_parser = subparsers.add_parser(
        "search", help="Search movies using semantic search"
    )
    _ = search_parser.add_argument("query", type=str, help="Search query")
    _ = search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of top results to return (default: 5)",
    )

    # Use a typed namespace so static checkers know the types of attributes
    namespace = CLIArgs()
    _ = parser.parse_args(namespace=namespace)
    args: CLIArgs = namespace

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            text = getattr(args, "text")
            embed_text(text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            query = getattr(args, "query")
            embed_text(query)
        case "search":
            query = getattr(args, "query")
            limit = getattr(args, "limit")
            if limit <= 0:
                print("Limit must be a positive integer.")
                return
            if not limit:
                limit = 5
            semantic_search = SemanticSearch()
            semantic_search.load_or_create_embeddings(semantic_search.documents or [])
            results = semantic_search.search(query, limit=limit)
            for idx, item in enumerate(results, start=1):
                movie, score = item
                print(f"{movie.get('title')} ({score:.4f})")
                print(f"{movie.get('description')}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
