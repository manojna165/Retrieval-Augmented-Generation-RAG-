from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import re
DOCUMENTS_DIR = Path(__file__).parent / "documents"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    separators=["\n\n", "\n", ". ", " ", ""]
)


def load_and_split_documents():
    """Extract PDF text and split it into smaller chunks."""
    pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {DOCUMENTS_DIR}"
        )

    pages = []

    for pdf_path in pdf_files:
        reader = PdfReader(str(pdf_path))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            # After extracting text from each PDF page
            text = re.sub(r"\s+", " ", text).strip()
            if text.strip():
                pages.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path.name,
                            "page": page_number
                        }
                    )
                )

    if not pages:
        raise ValueError(
            "No extractable text found in the PDF files."
        )

    chunks = splitter.split_documents(pages)

    if not chunks:
        raise ValueError("No document chunks were created.")

    return chunks