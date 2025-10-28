#!/usr/bin/env python3
from pathlib import Path
import sys

# Ensure the project root (one level above the `cli` package) is on sys.path
# so `from cli.helpers import ...` works when running the script directly.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from cli.helpers import (
    BM25_B,
    BM25_K1,
    bm25_idf_command,
    bm25_search_command,
    bm25_tf_command,
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
    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Display TF-IDF scores for a given term"
    )
    _ = tfidf_parser.add_argument("id", type=int, help="Movie ID to get TF-IDF for")
    _ = tfidf_parser.add_argument("term", type=str, help="Term to get TF-IDF for")
    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    _ = bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )
    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    _ = bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    _ = bm25_tf_parser.add_argument(
        "term", type=str, help="Term to get BM25 TF score for"
    )
    _ = bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    _ = bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )
    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    _ = bm25search_parser.add_argument("query", type=str, help="Search query")
    _ = bm25search_parser.add_argument(
        "limit", type=int, nargs="?", default=5, help="Number of results to return"
    )

    # Use a typed namespace so static checkers know the types of attributes
    namespace = CLIArgs()
    _ = parser.parse_args(namespace=namespace)
    args: CLIArgs = namespace

    path_movies = Path(__file__).parent.parent / "data" / "movies.json"

    match args.command:
        case "search":
            query = getattr(args, "query", None)
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
        case "tfidf":
            movie_id = getattr(args, "id", None)
            term = getattr(args, "term", None)
            if not isinstance(movie_id, int) or not isinstance(term, str):
                parser.error("the 'tfidf' command requires both id and term arguments")

            tf = get_term_frequency(movie_id, term)
            idf = get_inverse_document_frequency(term)
            tfidf = tf * idf

            print(f"TF-IDF of '{term}' in movie ID {movie_id}: {tfidf:.2f}")
        case "bm25idf":
            term = getattr(args, "term", None)
            if not isinstance(term, str):
                parser.error("the 'bm25idf' command requires a term argument")

            idf = bm25_idf_command(term)
            print(f"BM25 IDF of '{term}': {idf:.2f}")
        case "bm25tf":
            doc_id = getattr(args, "doc_id", None)
            term = getattr(args, "term", None)
            k1 = getattr(args, "k1", BM25_K1)
            b = getattr(args, "b", BM25_B)
            if not isinstance(doc_id, int) or not isinstance(term, str):
                parser.error("the 'bm25tf' command requires doc_id and term arguments")

            tf = bm25_tf_command(doc_id, term, k1, b)
            print(f"BM25 TF score of '{term}' in document '{doc_id}': {tf:.2f}")
        case "bm25search":
            query = getattr(args, "query", None)
            limit = getattr(args, "limit", 5)
            if not isinstance(query, str):
                parser.error("the 'bm25search' command requires a query argument")

            results, scores = bm25_search_command(query, limit)
            print(f"BM25 Searching for: {query}")
            for idx, movie in enumerate(results, start=1):
                score = scores.get(movie["id"], 0.0)
                print(f"{idx}. ({movie['id']}) {movie['title']} - Score: {score:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
