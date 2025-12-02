#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure the project root (one level above the `cli` package) is on sys.path
# so `from cli.helpers import ...` works when running the script directly.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse

from cli.keyword_search_cli import CLIArgs
from cli.lib.multimodal_search import verify_image_embedding


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify image embeddings against cached values"
    )
    _ = verify_image_embedding_parser.add_argument(
        "image_path", type=str, help="Path to the image file to verify"
    )

    # Use a typed namespace so static checkers know the types of attributes
    namespace = CLIArgs()
    _ = parser.parse_args(namespace=namespace)
    args: CLIArgs = namespace

    match args.command:
        case "verify_image_embedding":
            image_path = getattr(args, "image_path")
            verify_image_embedding(Path(image_path))
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
