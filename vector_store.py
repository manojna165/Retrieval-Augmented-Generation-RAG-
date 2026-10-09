from pathlib import Path
import chromadb

from embeddings import create_embeddings, create_query_embedding
from document_processor import load_and_split_documents

DB_PATH = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "task5_rag_chunks"

client = chromadb.PersistentClient(path=DB_PATH)


def initialize_database():
    """Create embeddings for chunks and store them in ChromaDB."""
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}}
    )

    chunks = load_and_split_documents()
    texts = [chunk.page_content for chunk in chunks]
    embeddings = create_embeddings(texts)

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": chunk.metadata["source"],
            "page": chunk.metadata["page"]
        }
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"Stored {collection.count()} chunks in ChromaDB.")
    return collection


def search_chunks(collection, query, top_k=4):
    """Retrieve the most relevant chunks for a question."""
    query_embedding = create_query_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    matches = []

    for i, text in enumerate(results["documents"][0]):
        distance = results["distances"][0][i]
        metadata = results["metadatas"][0][i]

        matches.append({
            "chunk_id": results["ids"][0][i],
            "text": text,
            "source": metadata["source"],
            "page": metadata["page"],
            "distance": round(distance, 4)
        })

    return matches