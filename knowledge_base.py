"""
knowledge_base.py
Loads agricultural knowledge documents for KrushiBandhu RAG system.
"""

import os

KNOWLEDGE_DIR = "knowledge"


def load_documents():
    """Load all .txt knowledge files."""
    documents = []

    for file in os.listdir(KNOWLEDGE_DIR):

        if file.endswith(".txt"):

            path = os.path.join(KNOWLEDGE_DIR, file)

            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            documents.append({
                "source": file,
                "text": text
            })

    return documents


def split_text(text, chunk_size=400, overlap=50):
    """Split long text into chunks."""

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size - overlap):

        chunk = " ".join(words[i:i + chunk_size])

        chunks.append(chunk)

    return chunks


def get_all_documents():
    """Return chunked documents."""

    raw_docs = load_documents()

    all_chunks = []

    for doc in raw_docs:

        chunks = split_text(doc["text"])

        for chunk in chunks:

            all_chunks.append({
                "source": doc["source"],
                "text": chunk
            })

    return all_chunks