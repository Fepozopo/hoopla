from sentence_transformers import SentenceTransformer


# Define a class called SemanticSearch
class SemanticSearch:
    model: SentenceTransformer

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text: str):
        if text.strip() == "":
            raise ValueError("Input text cannot be empty or whitespace.")

        text_list = [text]  # The encode method expects a list of inputs
        embedding = self.model.encode(text_list)
        return embedding[0]


def verify_model():
    try:
        semantic_search = SemanticSearch()
        print(f"Model loaded: {semantic_search.model}")
        print(f"Max sequence length: {semantic_search.model.max_seq_length}")
    except Exception as e:
        print(f"Error loading model: {e}")


def embed_text(text: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
