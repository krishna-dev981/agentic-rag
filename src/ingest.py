"""
ingest.py — PDFs ni load chesi, chunks ga split chesi,
FAISS vector store lo save chestundi.
"""

import os
import pickle
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "pdfs"
VECTOR_DIR = BASE_DIR / "vectorstore" / "faiss_index"


# ---------- Config ----------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_pdfs():
    """PDF folder lo unna anni PDFs load cheyyi."""
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {PDF_DIR}")

    print(f"📄 Found {len(pdf_files)} PDFs:")
    for p in pdf_files:
        print(f"   - {p.name}")

    documents = []
    for pdf in pdf_files:
        print(f"\n📖 Loading: {pdf.name}")
        loader = PyPDFLoader(str(pdf))
        docs = loader.load()
        # Metadata lo source add cheyyi
        for d in docs:
            d.metadata["source"] = pdf.name
        documents.extend(docs)
        print(f"   → {len(docs)} pages loaded")

    return documents


def split_documents(documents):
    """Documents ni chinnadi chunks ga split cheyyi."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"\n✂️  Split into {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks):
    """Embeddings create chesi FAISS lo save cheyyi."""
    print(f"\n🧠 Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("🔨 Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save cheyyi
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTOR_DIR))
    print(f"✅ Vector store saved to: {VECTOR_DIR}")

    return vectorstore


def main():
    print("=" * 60)
    print("🚀 RAG Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Load PDFs
    documents = load_pdfs()

    # Step 2: Split into chunks
    chunks = split_documents(documents)

    # Step 3: Build vector store
    build_vectorstore(chunks)

    print("\n" + "=" * 60)
    print("🎉 Ingestion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()