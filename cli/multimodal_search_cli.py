#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure the project root (one level above the `cli` package) is on sys.path
# so `from cli.helpers import ...` works when running the script directly.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse

from cli.keyword_search_cli import CLIArgs
from cli.lib.multimodal_search import image_search_command, verify_image_embedding


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify image embeddings against cached values"
    )
    _ = verify_image_embedding_parser.add_argument(
        "image_path", type=str, help="Path to the image file to verify"
    )

    image_search_parser = subparsers.add_parser(
        "image_search", help="Search the movie dataset using an image"
    )
    _ = image_search_parser.add_argument(
        "image_path", type=str, help="Path to the image file to search"
    )

    # Use a typed namespace so static checkers know the types of attributes
    namespace = CLIArgs()
    _ = parser.parse_args(namespace=namespace)
    args: CLIArgs = namespace

    match args.command:
        case "verify_image_embedding":
            image_path = getattr(args, "image_path")
            verify_image_embedding(Path(image_path))
        case "image_search":
            image_path = getattr(args, "image_path")
            results = image_search_command(Path(image_path))
            for idx, r in enumerate(results, start=1):
                title = r.get("title") or ""
                similarity = r.get("similarity") or 0.0
                description = r.get("description") or ""
                print(f"{idx}. {title} (similarity: {similarity:.3f})")
                print(f"   {description[:100]}...")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
