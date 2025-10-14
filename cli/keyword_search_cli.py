#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias, cast
from dataclasses import dataclass
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
    filtered_chars: list[str] = []
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


Json: TypeAlias = str | int | float | bool | None | list["Json"] | dict[str, "Json"]


def id_key(movie: dict[str, Json]) -> int:
    """Return an integer id for sorting; be defensive about the stored type.

    The JSON schema may store the id as a number or string. Only convert when
    it's a sensible type and fall back to 0 otherwise.
    """
    val = movie.get("id", 0)
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        try:
            return int(val)
        except ValueError:
            return 0
    if isinstance(val, float):
        # Convert floats by truncation; non-numeric objects are rejected above.
        return int(val)
    return 0


def search_movies(
    query: str, path_movies: Path, limit: int = 5
) -> list[dict[str, Json]]:
    """
    Search movies by normalized substring match (simple fallback for BM25).
    Returns up to `limit` matching movie dicts sorted by id.
    """
    results: list[dict[str, Json]] = []
    with path_movies.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    # Be defensive about the structure returned from json.load; if it's not a
    # mapping we can't search and just return an empty list.
    if not isinstance(raw, dict):
        return results

    data = cast(dict[str, Json], raw)

    normalized_query = normalize_text(query)
    movies = data.get("movies")
    if not isinstance(movies, list):
        return results

    for item in movies:
        # movie is expected to be a mapping from strings to JSON values; skip
        # entries that are not mappings to keep runtime behaviour predictable
        # and the type checker happy.
        if not isinstance(item, dict):
            continue

        movie = cast(dict[str, Json], item)

        raw_title = movie.get("title")
        title = raw_title if isinstance(raw_title, str) else ""
        normalized_title = normalize_text(title)
        if title and normalized_query and normalized_query in normalized_title:
            results.append(movie)

    results.sort(key=id_key)
    return results[:limit]


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

            results = search_movies(query, path_movies)
            print(f"Searching for: {query}")
            for idx, movie in enumerate(results, start=1):
                title = movie.get("title")
                if isinstance(title, str):
                    print(f"{idx}. {title}")
                else:
                    print(f"{idx}. <no title>")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
