import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google import genai

from vector_store import initialize_database, search_chunks

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL")

app = FastAPI(
    title="Task 5 - RAG Document Q&A",
    description="Ask questions about documents using retrieval-augmented generation."
)

# Load and index the PDF chunks when the app starts
collection = initialize_database()


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=8)


def generate_answer(question, chunks):
    """Use retrieved chunks as context for Gemini."""
    if not API_KEY or not MODEL:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY or GEMINI_MODEL is not configured."
        )

    context = "\n\n".join(
        f"[Source: {chunk['source']}, page: {chunk['page']}]\n"
        f"{chunk['text']}"
        for chunk in chunks
    )

    prompt = f"""
You are a document question-answering assistant.

Answer the question using only the context provided below.
Do not invent facts or use outside knowledge.
If the context does not contain the answer, say:
"The provided documents do not contain enough information to answer this question."

Context:
{context}

Question:
{question}

Provide a clear, concise answer. Mention relevant source pages when useful.
"""

    try:
        client = genai.Client(api_key=API_KEY)
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )

        if not response.text:
            raise HTTPException(
                status_code=502,
                detail="Gemini returned an empty response."
            )

        return response.text

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Could not generate an answer using Gemini. Check the model, API key, quota, and network."
        )


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "RAG document Q&A API is running"
    }


@app.get("/documents/count")
def document_count():
    return {
        "indexed_chunks": collection.count()
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        # Step 1: Retrieve relevant chunks from ChromaDB
        chunks = search_chunks(
            collection,
            request.question,
            request.top_k
        )

        if not chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant document chunks were found."
            )

        # Step 2: Send the retrieved context to Gemini
        answer = generate_answer(request.question, chunks)

        # Step 3: Return the answer and its retrieved sources
        return {
            "question": request.question,
            "answer": answer,
            "sources": [
                {
                    "document": chunk["source"],
                    "page": chunk["page"],
                    "chunk_id": chunk["chunk_id"],
                    "distance": chunk["distance"],
                    "text": chunk["text"]
                }
                for chunk in chunks
            ]
        }

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the question."
        )