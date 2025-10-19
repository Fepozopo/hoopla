from nltk.stem import PorterStemmer
from pathlib import Path
from typing import TypeAlias, cast, TypedDict
import json
import pickle
import unicodedata


Json: TypeAlias = str | int | float | bool | None | list["Json"] | dict[str, "Json"]


class Movie(TypedDict):
    """A validated subset of the movie JSON structure used by this CLI.

    We keep only the fields we need for searching and display. The
    parsing function below coerces/normalizes values (for example
    converting an id represented as a string into an int) and returns
    None for entries that are missing required information (title).
    """

    id: int
    title: str
    description: str | None


class InvertedIndex:
    """A inverted index mapping tokens to lists of document ids."""

    index: dict[str, list[int]]
    docmap: dict[int, Movie]

    def __init__(self, index: dict[str, list[int]], docmap: dict[int, Movie]) -> None:
        self.index = index
        self.docmap = docmap

    def __add_document(self, doc_id: int, text: str | None) -> None:
        normalized = normalize_text(text)
        tokens = tokenize_text(normalized)
        for token in tokens:
            if token not in self.index:
                self.index[token] = []
            if doc_id not in self.index[token]:
                self.index[token].append(doc_id)

    def get_documents(self, term: str) -> list[Movie]:
        normalized_term = normalize_text(term)
        token = tokenize_text(normalized_term)
        if not token:
            return []
        token = token[0]
        doc_ids = self.index.get(token, [])
        return [self.docmap[doc_id] for doc_id in doc_ids if doc_id in self.docmap]

    def build(self, movies: list[Movie]) -> None:
        for movie in movies:
            self.docmap[movie["id"]] = movie
            title = movie.get("title")
            description = movie.get("description")
            input = f"{title} {description}"
            self.__add_document(movie["id"], input)

    def save(self) -> None:
        """Persist the in-memory index and docmap to disk using pickle.

        Files:
        - cache/index.pkl   -> self.index
        - cache/docmap.pkl  -> self.docmap

        Creates the cache directory if it does not already exist.
        """
        cache_dir = Path("cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        index_path = cache_dir / "index.pkl"
        docmap_path = cache_dir / "docmap.pkl"

        with index_path.open("wb") as f:
            pickle.dump(self.index, f, protocol=pickle.HIGHEST_PROTOCOL)

        with docmap_path.open("wb") as f:
            pickle.dump(self.docmap, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self) -> None:
        """Load the inverted index and docmap from disk using pickle.

        Files:
        - cache/index.pkl   -> self.index
        - cache/docmap.pkl  -> self.docmap
        """
        cache_dir = Path("cache")
        index_path = cache_dir / "index.pkl"
        docmap_path = cache_dir / "docmap.pkl"

        # Raise FileNotFoundError if files do not exist
        if not index_path.exists() or not docmap_path.exists():
            raise FileNotFoundError(
                "Index or docmap file not found in cache directory."
            )

        with index_path.open("rb") as f:
            self.index = pickle.load(f)

        with docmap_path.open("rb") as f:
            self.docmap = pickle.load(f)


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


def tokenize_text(
    text: str | None,
) -> list[str]:
    """Split a text string into tokens based on whitespace.

    This is a simple fallback tokenizer that doesn't use any NLP libraries.
    It lowercases the input and splits on whitespace, returning a list of
    non-empty tokens.

    Parameters
    - text: The input string to tokenize. If falsy (None or empty), returns
        an empty list.

    Returns
    - A list of tokens (substrings) extracted from the input string.

    Examples
    - tokenize_text("The quick brown fox") -> ["the", "quick", "brown", "fox"]
    - tokenize_text("  Hello,   world!  ") -> ["hello,", "world!"]
    """
    if not text:
        return []

    # Lowercase and split on whitespace, filtering out empty tokens
    tokens = [token for token in text.lower().split() if token]

    # List of stop words to exclude from tokens
    stop_words_path = Path(__file__).parent.parent / "data" / "stopwords.txt"
    stop_words: set[str]
    try:
        with stop_words_path.open("r", encoding="utf-8") as f:
            stop_words = {line.strip().lower() for line in f if line.strip()}
    except FileNotFoundError:
        stop_words = set()
    tokens = [token for token in tokens if token not in stop_words]

    # Reduce tokens to their stems using Porter Stemmer
    stemmer = PorterStemmer()
    tokens: list[str]
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens


def parse_movie(raw: Json) -> Movie | None:
    """Validate and coerce a single raw JSON mapping into a `Movie`.

    Returns
    - A `Movie` dict with normalized types when the entry is valid.
    - `None` when the entry doesn't contain the minimal required data
        (currently a string `title`).
    """
    if not isinstance(raw, dict):
        return None

    raw_title = raw.get("title")
    if not isinstance(raw_title, str) or not raw_title:
        # Title is required for our search/display pipeline.
        return None

    # Normalise the id to an int with a sensible fallback.
    raw_id = raw.get("id", 0)
    if isinstance(raw_id, int):
        movie_id = raw_id
    elif isinstance(raw_id, float):
        movie_id = int(raw_id)
    elif isinstance(raw_id, str):
        try:
            movie_id = int(raw_id)
        except ValueError:
            try:
                movie_id = int(float(raw_id))
            except ValueError:
                movie_id = 0
    else:
        movie_id = 0

    raw_description = raw.get("description")
    description = raw_description if isinstance(raw_description, str) else None

    # Construct a value that already matches the `Movie` TypedDict so no
    # cast from a generic mapping is necessary.
    movie: Movie = {
        "id": movie_id,
        "title": raw_title,
        "description": description,
    }
    return movie


def id_key(movie: Movie) -> int:
    """Return the stored integer id for sorting.

    `parse_movie` guarantees the `id` field is an int (falling back to 0)
    so we can rely on that invariant here and keep the function trivial.
    """
    return movie["id"]


def search_movies(query: str, path_movies: Path, limit: int = 5) -> list[Movie]:
    """
    Search movies by normalized substring match (simple fallback for BM25).
    Returns up to `limit` matching movie dicts sorted by id.
    """
    results: list[Movie] = []
    with path_movies.open("r", encoding="utf-8") as file:
        # json.load returns Any; cast it to the Json alias so static
        # type checkers don't treat `raw` as Any. We still validate the
        # structure at runtime below.
        raw = cast(Json, json.load(file))

    # Be defensive about the structure returned from json.load; if it's not a
    # mapping we can't search and just return an empty list.
    if not isinstance(raw, dict):
        return results

    data = cast(dict[str, Json], raw)

    normalized_query = normalize_text(query)
    tokenized_query = tokenize_text(normalized_query)
    movies_raw = data.get("movies")
    if not isinstance(movies_raw, list):
        return results

    for item in movies_raw:
        # Each list item should be a mapping representing a movie; parse and
        # validate it into our `Movie` TypedDict. Skip entries that aren't
        # mappings or fail validation.
        if not isinstance(item, dict):
            continue

        parsed = parse_movie(item)
        if parsed is None:
            continue

        normalized_title = normalize_text(parsed["title"])
        if any(token in normalized_title for token in tokenized_query):
            results.append(parsed)

    results.sort(key=id_key)
    return results[:limit]


def build_inverted_index(path_movies: Path) -> None:
    """Build an inverted index from a movies JSON file and persist it.

    The function reads the JSON at `path_movies`, parses and validates the
    contained movie entries with `parse_movie`, builds an InvertedIndex,
    saves it to disk using InvertedIndex.save(), and then prints the first
    document id for the token 'merida' if any documents contain that token.
    """
    try:
        with path_movies.open("r", encoding="utf-8") as f:
            raw = cast(Json, json.load(f))
    except FileNotFoundError:
        print(f"Movies file not found: {path_movies}")
        return
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON from {path_movies}: {e}")
        return

    if not isinstance(raw, dict):
        print(f"Unexpected JSON structure in {path_movies}: expected a mapping")
        return

    data = cast(dict[str, Json], raw)
    movies_raw = data.get("movies")
    if not isinstance(movies_raw, list):
        print(f"No 'movies' list found in {path_movies}")
        return

    movies: list[Movie] = []
    for item in movies_raw:
        if not isinstance(item, dict):
            continue
        parsed = parse_movie(item)
        if parsed is None:
            continue
        movies.append(parsed)

    # Build the inverted index and persist it
    index = InvertedIndex({}, {})
    index.build(movies)
    index.save()


def search_inverted_index(query: str, limit: int = 5) -> list[Movie]:
    """Search the persisted inverted index for movies matching the query.

    Loads the inverted index and docmap from disk, tokenizes the query,
    retrieves matching documents, and returns up to `limit` results
    sorted by id.
    """
    index = InvertedIndex({}, {})
    try:
        index.load()
    except FileNotFoundError:
        print("Inverted index not found in cache. Please build it first.")
        return []

    normalized_query = normalize_text(query)
    tokenized_query = tokenize_text(normalized_query)

    results: list[Movie] = []
    for token in tokenized_query:
        docs = index.get_documents(token)
        results.extend(docs)

    # Remove duplicates while preserving order
    seen_ids: set[int]
    seen_ids = set()
    unique_results: list[Movie] = []
    for movie in results:
        if movie["id"] not in seen_ids:
            seen_ids.add(movie["id"])
            unique_results.append(movie)

    unique_results.sort(key=id_key)
    return unique_results[:limit]
