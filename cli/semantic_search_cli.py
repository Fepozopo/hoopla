#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure the project root (one level above the `cli` package) is on sys.path
# so `from cli.helpers import ...` works when running the script directly.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse

from cli.keyword_search_cli import CLIArgs
from cli.lib.semantic_search import embed_text, verify_embeddings, verify_model


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
        "embed_query", help="Generate embedding for a search query"
    )
    _ = embedquery_parser.add_argument("query", type=str, help="Search query to embed")

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
        case "embed_query":
            query = getattr(args, "query")
            embed_text(query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
