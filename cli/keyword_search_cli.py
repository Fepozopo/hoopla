#!/usr/bin/env python3

from pathlib import Path
import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    path_movies = Path(__file__).parent.parent / "data" / "movies.json"

    match args.command:
        case "search":
            results = []
            with path_movies.open("r", encoding="utf-8") as file:
                data = json.load(file)

            query = args.query.lower()
            for movie in data.get("movies", []):
                title = movie.get("title", "")
                if title and query in title.lower():
                    results.append(movie)

            def id_key(movie):
                try:
                    return int(movie.get("id", 0))
                except (TypeError, ValueError):
                    return 0

            results.sort(key=id_key)
            results = results[:5]

            print(f"Searching for: {args.query}")
            for idx, movie in enumerate(results, start=1):
                print(f"{idx}. {movie.get('title', '<no title>')}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
