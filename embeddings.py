from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)


def create_embeddings(texts):
    """Convert document chunks into numerical vectors."""
    return model.encode(texts, normalize_embeddings=True).tolist()


def create_query_embedding(query):
    """Convert a user question into a vector."""
    return model.encode(query, normalize_embeddings=True).tolist()