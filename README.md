# Task 5 - Retrieval-Augmented Generation (RAG)

## Objective
Build a document question-answering system using embeddings, a vector database, and an LLM.

## Technologies
- Python
- Sentence Transformers: all-MiniLM-L6-v2
- ChromaDB
- LangChain RecursiveCharacterTextSplitter
- PyPDF
- Google Gemini API
- FastAPI
- Pydantic

## Pipeline
1. Extract text from PDF documents.
2. Split text into chunks using chunk_size=800 and chunk_overlap=120.
3. Generate embeddings for each chunk.
4. Store the chunks, embeddings, and metadata in ChromaDB.
5. Convert the user's question into an embedding.
6. Retrieve the most relevant chunks using cosine distance.
7. Pass the retrieved chunks and question to Gemini.
8. Return the generated answer and source information.

## Setup
Create and activate a Python environment, then install:

pip install -r requirements.txt

Create a .env file containing:

GEMINI_API_KEY=your_api_key
GEMINI_MODEL=your_working_gemini_model

Never commit the .env file.

Place PDF files in the documents directory.

## Run
uvicorn main:app --reload

Open http://127.0.0.1:8000/docs

## Endpoints
- GET /: API health check
- GET /documents/count: Number of indexed chunks
- POST /ask: Ask a question about the indexed documents

## Example Request
{
  "question": "What is the Transformer architecture?",
  "top_k": 4
}

## Expected Output
The API returns a generated answer and the retrieved document chunks, including source filenames and page numbers.

## Limitations
Answers depend on the content retrieved from the supplied documents. Retrieval may miss relevant information, and the LLM can still make mistakes. The system instructs Gemini to acknowledge when the context is insufficient.