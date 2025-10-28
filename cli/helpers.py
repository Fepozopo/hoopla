from collections import Counter
from nltk.stem import PorterStemmer
from pathlib import Path
from typing import TypeAlias, cast, TypedDict
import json
import pickle
import unicodedata


BM25_K1 = 1.5
BM25_B = 0.75

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
    term_frequencies: dict[int, Counter[str]]
    doc_lengths: dict[int, int]

    def __init__(self, index: dict[str, list[int]], docmap: dict[int, Movie]) -> None:
        self.index = index
        self.docmap = docmap
        self.term_frequencies = {}
        self.doc_lengths = {}

    def __add_document(self, doc_id: int, text: str | None) -> None:
        normalized = normalize_text(text)
        tokens = tokenize_text(normalized)

        self.doc_lengths[doc_id] = len(tokens)

        # Ensure a Counter exists for this document's term frequencies
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        tf = self.term_frequencies[doc_id]

        for token in tokens:
            if token not in self.index:
                self.index[token] = []
            if doc_id not in self.index[token]:
                self.index[token].append(doc_id)
            # Increment the term frequency for this token in this document
            tf[token] += 1

    def get_documents(self, term: str) -> list[Movie]:
        normalized_term = normalize_text(term)
        token = tokenize_text(normalized_term)
        if not token:
            return []
        token = token[0]
        doc_ids = self.index.get(token, [])
        return [self.docmap[doc_id] for doc_id in doc_ids if doc_id in self.docmap]

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        total_length = sum(self.doc_lengths.values())
        return total_length / len(self.doc_lengths)

    def build(self, movies: list[Movie]) -> None:
        for movie in movies:
            self.docmap[movie["id"]] = movie
            title = movie.get("title")
            description = movie.get("description")
            input = f"{title} {description}"
            self.__add_document(movie["id"], input)

    def save(self) -> None:
        """Persist the in-memory index, docmap, term_frequencies, and doc_lengths to disk using pickle.

        Files:
        - cache/index.pkl   -> self.index
        - cache/docmap.pkl  -> self.docmap
        - cache/term_frequencies.pkl -> self.term_frequencies
        - cache/doc_lengths.pkl -> self.doc_lengths

        Creates the cache directory if it does not already exist.
        """
        cache_dir = Path("cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        index_path = cache_dir / "index.pkl"
        docmap_path = cache_dir / "docmap.pkl"
        term_frequencies_path = cache_dir / "term_frequencies.pkl"
        doc_lengths_path = cache_dir / "doc_lengths.pkl"

        with index_path.open("wb") as f:
            pickle.dump(self.index, f, protocol=pickle.HIGHEST_PROTOCOL)

        with docmap_path.open("wb") as f:
            pickle.dump(self.docmap, f, protocol=pickle.HIGHEST_PROTOCOL)

        with term_frequencies_path.open("wb") as f:
            pickle.dump(self.term_frequencies, f, protocol=pickle.HIGHEST_PROTOCOL)

        with doc_lengths_path.open("wb") as f:
            pickle.dump(self.doc_lengths, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self) -> None:
        """Load the inverted index, docmap, term_frequencies, and doc_lengths from disk using pickle.

        Files:
        - cache/index.pkl   -> self.index
        - cache/docmap.pkl  -> self.docmap
        - cache/term_frequencies.pkl -> self.term_frequencies
        - cache/doc_lengths.pkl -> self.doc_lengths
        """
        cache_dir = Path("cache")
        index_path = cache_dir / "index.pkl"
        docmap_path = cache_dir / "docmap.pkl"
        term_frequencies_path = cache_dir / "term_frequencies.pkl"
        doc_lengths_path = cache_dir / "doc_lengths.pkl"

        # Raise FileNotFoundError if files do not exist
        if (
            not index_path.exists()
            or not docmap_path.exists()
            or not term_frequencies_path.exists()
            or not doc_lengths_path.exists()
        ):
            raise FileNotFoundError(
                "Index, docmap, term_frequencies, or doc_lengths file not found in cache directory."
            )

        with index_path.open("rb") as f:
            self.index = pickle.load(f)

        with docmap_path.open("rb") as f:
            self.docmap = pickle.load(f)

        with term_frequencies_path.open("rb") as f:
            self.term_frequencies = pickle.load(f)

        with doc_lengths_path.open("rb") as f:
            self.doc_lengths = pickle.load(f)

    def get_tf(self, doc_id: int, term: str) -> int:
        """Return the number of times `term` appears in the document with id `doc_id`.

        The `term` is normalized and tokenized. We expect the tokenization to
        produce exactly one token; if it produces more than one token an
        exception is raised. If the document or term is not present, returns 0.
        """
        # Normalize and tokenize the provided term; expect exactly one token
        normalized = normalize_text(term)
        tokens = tokenize_text(normalized)

        if not tokens:
            return 0
        if len(tokens) > 1:
            raise ValueError("term must tokenize to a single token")

        token = tokens[0]

        # Look up term frequencies for the document
        tf_counter = self.term_frequencies.get(doc_id)
        if not tf_counter:
            return 0

        return tf_counter.get(token, 0)

    def get_bm25_idf(self, term: str) -> float:
        """Calculate the BM25 inverse document frequency (IDF) for `term`.

        Uses the formula:
        IDF(term) = log((N - df + 0.5) / (df + 0.5) + 1)
        where N is the total number of documents and n is the number of documents
        containing the term.

        Returns 0.0 if the term is not found in any document.
        """
        import math

        normalized_term = normalize_text(term)
        tokens = tokenize_text(normalized_term)

        if not tokens:
            return 0.0
        if len(tokens) > 1:
            raise ValueError("term must tokenize to a single token")

        token = tokens[0]
        N = len(self.docmap)
        df = len(self.index.get(token, []))

        if df == 0:
            return 0.0

        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return idf

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        """Calculate the BM25 term frequency (TF) for `term` in document `doc_id`.

        Uses the formula:
        TF(term, doc) = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avgdl)))
        where f is the raw term frequency in the document.

        Returns 0.0 if the term is not found in the document.
        """
        tf = self.get_tf(doc_id, term)
        if tf == 0:
            return 0.0

        dl = self.doc_lengths.get(doc_id, 0)
        avgdl = self.__get_avg_doc_length()
        bmtf = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avgdl)))
        return bmtf

    def bm25(self, doc_id: int, term: str) -> float:
        """Calculate the BM25 score for `term` in document `doc_id`.

        Uses the formula:
        BM25(term, doc) = IDF(term) * TF(term, doc)

        Returns 0.0 if the term is not found in the document.
        """
        idf = self.get_bm25_idf(term)
        tf = self.get_bm25_tf(doc_id, term)
        return idf * tf

    def bm25_search(
        self, query: str, limit: int
    ) -> tuple[list[Movie], dict[int, float]]:
        """Search the inverted index using BM25 scoring for the given query.

        Returns:
          - A list of up to `limit` matching movie dicts sorted by BM25 score.
          - A dict mapping document ids to their BM25 scores (only top results).
        """
        normalized_query = normalize_text(query)
        tokenized_query = tokenize_text(normalized_query)

        if not tokenized_query:
            return ([], {})

        scores: dict[int, float] = {}

        for token in tokenized_query:
            doc_ids = self.index.get(token, [])
            for doc_id in doc_ids:
                score = self.bm25(doc_id, token)
                if score == 0.0:
                    continue
                scores[doc_id] = scores.get(doc_id, 0.0) + score

        # Sort documents by score in descending order and get top `limit` as (doc_id, score) pairs
        top_pairs = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]

        # Map Movies with their scores (preserving ordering)
        results: list[Movie] = []
        top_scores: dict[int, float] = {}
        for doc_id, score in top_pairs:
            if doc_id in self.docmap:
                results.append(self.docmap[doc_id])
                top_scores[doc_id] = score

        return (results, top_scores)


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


def get_term_frequency(doc_id: int, term: str) -> int:
    """Retrieve the term frequency of `term` in the document with id `doc_id`.

    Loads the inverted index from disk and uses InvertedIndex.get_tf()
    to retrieve the term frequency.

    Returns
    - The term frequency as an integer. If the document or term is not
      found, returns 0.
    """
    index = InvertedIndex({}, {})
    try:
        index.load()
    except FileNotFoundError:
        print("Inverted index not found in cache. Please build it first.")
        return 0

    try:
        tf = index.get_tf(doc_id, term)
    except ValueError as e:
        print(f"Error retrieving term frequency: {e}")
        return 0

    return tf


def get_inverse_document_frequency(term: str) -> float:
    """Retrieve the inverse document frequency (IDF) of `term` in the corpus.

    Loads the inverted index from disk and calculates the IDF using the formula:
    IDF(term) = log(Total number of documents / Number of documents containing term)

    Returns
    - The IDF as a float. If the term is not found, returns 0.0.
    """
    import math

    index = InvertedIndex({}, {})
    try:
        index.load()
    except FileNotFoundError:
        print("Inverted index not found in cache. Please build it first.")
        return 0.0

    normalized_term = normalize_text(term)
    tokens = tokenize_text(normalized_term)

    if not tokens:
        return 0.0
    if len(tokens) > 1:
        print("Error: term must tokenize to a single token")
        return 0.0

    token = tokens[0]
    total_docs = len(index.docmap) + 1
    docs_with_term = len(index.index.get(token, [])) + 1

    if docs_with_term == 0:
        return 0.0

    idf = math.log(total_docs / docs_with_term)
    return idf


def bm25_idf_command(term: str) -> float:
    """Retrieve the BM25 inverse document frequency (IDF) of `term` in the corpus.

    Loads the inverted index from disk and uses InvertedIndex.get_bm25_idf()
    to calculate the BM25 IDF.

    Returns
    - The BM25 IDF as a float. If the term is not found, returns 0.0.
    """
    index = InvertedIndex({}, {})
    try:
        index.load()
    except FileNotFoundError:
        print("Inverted index not found in cache. Please build it first.")
        return 0.0

    try:
        idf = index.get_bm25_idf(term)
    except ValueError as e:
        print(f"Error retrieving BM25 IDF: {e}")
        return 0.0

    return idf


def bm25_tf_command(
    doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
) -> float:
    """Retrieve the BM25 term frequency (TF) of `term` in the document with id `doc_id`.

    Loads the inverted index from disk and uses InvertedIndex.get_bm25_tf()
    to calculate the BM25 TF.

    Returns
    - The BM25 TF as a float. If the document or term is not found, returns 0.0.
    """
    index = InvertedIndex({}, {})
    try:
        index.load()
    except FileNotFoundError:
        print("Inverted index not found in cache. Please build it first.")
        return 0.0

    try:
        tf = index.get_bm25_tf(doc_id, term, k1, b)
    except ValueError as e:
        print(f"Error retrieving BM25 TF: {e}")
        return 0.0

    return tf


def bm25_search_command(
    query: str, limit: int = 5
) -> tuple[list[Movie], dict[int, float]]:
    """Search the inverted index using BM25 scoring for the given query.

    Loads the inverted index from disk and uses InvertedIndex.bm25_search()
    to perform the search.

    Returns
    - A tuple containing:
      - A list of up to `limit` matching movie dicts sorted by BM25 score.
      - A dict mapping document ids to their BM25 scores (only top results).
    """
    index = InvertedIndex({}, {})
    try:
        index.load()
    except FileNotFoundError:
        print("Inverted index not found in cache. Please build it first.")
        return ([], {})

    results, scores = index.bm25_search(query, limit)
    return (results, scores)
