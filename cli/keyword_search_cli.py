#!/usr/bin/env python3

from pathlib import Path
from typing import Any
import argparse
import json
import unicodedata


def normalize_text(
    text: str | None,
    *,
    strip_diacritics: bool = True,
    remove_punctuation: bool = True,
) -> str:
    """
    Normalize a text string in a Unicode-aware way.

    Parameters
    - text: The input string. If falsy (None or empty), returns an empty string.
    - strip_diacritics: If True, decompose Unicode characters (NFKD) and remove
        combining marks (Unicode categories starting with 'M') to strip diacritics.
    - remove_punctuation: If True, remove characters whose Unicode category starts
        with 'P' (all punctuation characters).

    Returns
    - A normalized string: lowercased, with diacritics/punctuation removed according
      to the flags, and with runs of whitespace collapsed to single spaces.

    Examples
    - normalize_text("Café — The Movie!") -> "cafe the movie"
    - normalize_text("El Niño", strip_diacritics=False) -> "el niño"
    """
    if not text:
        return ""

    # Lowercase first to make comparisons case-insensitive
    s = text.lower()

    # Decompose characters so diacritics become separate combining characters
    if strip_diacritics:
        s = unicodedata.normalize("NFKD", s)
    else:
        # still normalize in a stable way but don't decompose for diacritic stripping
        s = unicodedata.normalize("NFC", s)

    # Build filtered result, skipping combining marks (M*) when stripping diacritics
    # and skipping punctuation (P*) when removing punctuation.
    filtered_chars = []
    for ch in s:
        cat = unicodedata.category(ch)  # e.g. 'Ll', 'Mn', 'Pd', 'Po', etc.
        if strip_diacritics and cat.startswith("M"):
            # Skip combining marks (diacritics)
            continue
        if remove_punctuation and cat.startswith("P"):
            # Skip punctuation
            continue
        filtered_chars.append(ch)

    filtered = "".join(filtered_chars)

    # Collapse any whitespace runs and trim
    normalized = " ".join(filtered.split())

    return normalized


def id_key(movie: dict[str, Any]) -> int:
    try:
        return int(movie.get("id", 0))
    except (TypeError, ValueError):
        return 0


def search_movies(
    query: str, path_movies: Path, limit: int = 5
) -> list[dict[str, Any]]:
    """
    Search movies by normalized substring match (simple fallback for BM25).
    Returns up to `limit` matching movie dicts sorted by id.
    """
    results: list[dict[str, Any]] = []
    with path_movies.open("r", encoding="utf-8") as file:
        data: dict[str, Any] = json.load(file)

    normalized_query = normalize_text(query)
    for movie in data.get("movies", []):
        # movie is expected to be a mapping from strings to arbitrary JSON values
        title = movie.get("title", "")
        normalized_title = normalize_text(title)
        if title and normalized_query and normalized_query in normalized_title:
            results.append(movie)

    results.sort(key=id_key)
    return results[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    path_movies = Path(__file__).parent.parent / "data" / "movies.json"

    match args.command:
        case "search":
            results = search_movies(args.query, path_movies)
            print(f"Searching for: {args.query}")
            for idx, movie in enumerate(results, start=1):
                print(f"{idx}. {movie.get('title', '<no title>')}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
