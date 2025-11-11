#!/usr/bin/env python3
import sys
from pathlib import Path

# Ensure the project root (one level above the `cli` package) is on sys.path
# so `from cli.helpers import ...` works when running the script directly.
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse

from cli.helpers import load_movies
from cli.keyword_search_cli import CLIArgs
from cli.lib.semantic_search import (
    ChunkedSemanticSearch,
    SemanticSearch,
    embed_text,
    fixed_size_chunks,
    semantic_chunks,
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
    chunk_parser = subparsers.add_parser(
        "chunk", help="Chunk text into smaller pieces for embedding"
    )
    _ = chunk_parser.add_argument("text", type=str, help="Text to chunk")
    _ = chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, help="Chunk size"
    )
    _ = chunk_parser.add_argument(
        "--overlap", type=int, default=0, help="Chunk overlap size"
    )
    semantic_chunck_parser = subparsers.add_parser(
        "semantic_chunk", help="Chunk text semantically"
    )
    _ = semantic_chunck_parser.add_argument("text", type=str, help="Text to chunk")
    _ = semantic_chunck_parser.add_argument(
        "--max-chunk-size", type=int, default=4, help="Maximum chunk size"
    )
    _ = semantic_chunck_parser.add_argument(
        "--overlap", type=int, default=0, help="Chunk overlap size"
    )
    embed_chunks_parser = subparsers.add_parser(
        "embed_chunks", help="Embed text chunks"
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
            # Load movies from the canonical data file so document indices align
            # with any cached embeddings.
            path_movies = Path("data/movies.json")
            documents = load_movies(path_movies)
            if not documents:
                print(
                    f"No movies found at {path_movies}. Please add movies to the dataset."
                )
                return
            # Guard against the SemanticSearch requiring a non-empty document list
            # and bubble up any validation errors as user-friendly messages.
            try:
                semantic_search.load_or_create_embeddings(documents)
            except ValueError as e:
                print(f"Error preparing embeddings: {e}")
                return
            results = semantic_search.search(query, limit=limit)
            for idx, item in enumerate(results, start=1):
                movie, score = item
                print(
                    "============================================================================"
                )
                print(f"{idx}. {movie.get('title')} ({score:.4f})")
                print(f"{movie.get('description')}")
                print(
                    "============================================================================"
                )
                print()
        case "chunk":
            text = getattr(args, "text")
            chunk_size = getattr(args, "chunk_size")
            overlap = getattr(args, "overlap")
            if overlap >= chunk_size:
                parser.error("overlap must be smaller than chunk_size")
            print(f"Chunking {len(text)} characters")
            for idx, chunk in enumerate(
                fixed_size_chunks(text, chunk_size, overlap), start=1
            ):
                print(f"{idx}. {chunk}")
        case "semantic_chunk":
            text = getattr(args, "text")
            max_chunk_size = getattr(args, "max_chunk_size")
            overlap = getattr(args, "overlap")
            if overlap >= max_chunk_size:
                parser.error("overlap must be smaller than max_chunk_size")
            print(f"Semantically chunking {len(text)} characters")
            results = semantic_chunks(text, max_chunk_size, overlap)
            for idx, chunk in enumerate(results, start=1):
                print(f"{idx}. {chunk}")
        case "embed_chunks":
            # Load movies from the canonical data file so document indices align
            # with any cached chunk embeddings.
            path_movies = Path("data/movies.json")
            documents = load_movies(path_movies)
            if not documents:
                print(
                    f"No movies found at {path_movies}. Please add movies to the dataset."
                )
                return

            chunked_search = ChunkedSemanticSearch()

            # Use the chunked search's method to load or create embeddings
            load_fn = chunked_search.load_or_create_chunk_embeddings  # <-- fixed typo

            try:
                embeddings = load_fn(documents)
            except ValueError as e:
                print(f"Error preparing chunk embeddings: {e}")
                return

            print(f"Generated {len(embeddings)} chunked embeddings")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
