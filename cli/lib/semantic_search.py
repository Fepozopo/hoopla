from sentence_transformers import SentenceTransformer


# Define a class called SemanticSearch
class SemanticSearch:
    model: SentenceTransformer

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)


def verify_model():
    try:
        semantic_search = SemanticSearch()
        print(f"Model loaded: {semantic_search.model}")
        print(f"Max sequence length: {semantic_search.model.max_seq_length}")
    except Exception as e:
        print(f"Error loading model: {e}")
