#!/usr/bin/env python3
from pathlib import Path
import sys

# Ensure the project root (one level above the `cli` package) is on sys.path
# so `from cli.helpers import ...` works when running the script directly.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from cli.helpers import (
    build_inverted_index,
    search_inverted_index,
    get_term_frequency,
    get_inverse_document_frequency,
)
from dataclasses import dataclass
import argparse


@dataclass
class CLIArgs:
    """A tiny typed container used as argparse's namespace so attributes
    have precise types for static type checkers (avoids Attributes being
    inferred as Any).

    parse_args will populate these attributes so it's safe to pass an
    instance as the "namespace" argument.
    """

    command: str | None = None
    query: str | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    _ = search_parser.add_argument("query", type=str, help="Search query")
    _ = subparsers.add_parser("build", help="Build the inverted index")
    tf_parser = subparsers.add_parser(
        "tf", help="Display term frequencies for a given term"
    )
    _ = tf_parser.add_argument(
        "id", type=int, help="Movie ID to get term frequency for"
    )
    _ = tf_parser.add_argument("term", type=str, help="Term to get frequencies for")
    idf_parser = subparsers.add_parser(
        "idf", help="Display inverse document frequencies for a given term"
    )
    _ = idf_parser.add_argument("term", type=str, help="Term to get IDF for")

    # Use a typed namespace so static checkers know the types of attributes
    namespace = CLIArgs()
    _ = parser.parse_args(namespace=namespace)
    args: CLIArgs = namespace

    path_movies = Path(__file__).parent.parent / "data" / "movies.json"

    match args.command:
        case "search":
            # argparse will populate `query` when the `search` subparser is
            # used, but the type checker doesn't know that. Validate at
            # runtime so static analysis can reason about the type safely.
            query = args.query
            if not isinstance(query, str):
                parser.error("the 'search' command requires a query argument")

            results = search_inverted_index(query)
            print(f"Searching for: {query}")
            for idx, movie in enumerate(results, start=1):
                # `parse_movie` guarantees `title` is present and is a str.
                print(f"{idx}. {movie['title']}")
        case "build":
            build_inverted_index(path_movies)
        case "tf":
            movie_id = getattr(args, "id", None)
            term = getattr(args, "term", None)
            if not isinstance(movie_id, int) or not isinstance(term, str):
                parser.error("the 'tf' command requires both id and term arguments")

            tf = get_term_frequency(movie_id, term)
            print(f"Term Frequency of '{term}' in movie ID {movie_id}: {tf}")
        case "idf":
            term = getattr(args, "term", None)
            if not isinstance(term, str):
                parser.error("the 'idf' command requires a term argument")

            idf = get_inverse_document_frequency(term)
            print(f"Inverse Document Frequency of '{term}': {idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
