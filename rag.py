"""
rag.py — Knowledge retrieval system
"""

import os
import pickle
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from knowledge_base import get_all_documents
from symptom_map import SYMPTOM_MAP

EMBED_MODEL = "all-MiniLM-L6-v2"

INDEX_PATH = "rag_index/faiss.index"
DOCS_PATH = "rag_index/documents.pkl"

TOP_K = 4


def load_embedding_model():
    return SentenceTransformer(EMBED_MODEL)


def build_vector_store():

    os.makedirs("rag_index", exist_ok=True)

    embed_model = load_embedding_model()

    documents = get_all_documents()

    texts = [doc["text"] for doc in documents]

    embeddings = embed_model.encode(texts, convert_to_numpy=True)

    embeddings = embeddings.astype(np.float32)

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    with open(DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)

    print("RAG index built successfully.")


def load_vector_store():

    if not os.path.exists(INDEX_PATH):
        build_vector_store()

    index = faiss.read_index(INDEX_PATH)

    with open(DOCS_PATH, "rb") as f:
        documents = pickle.load(f)

    embed_model = load_embedding_model()

    return index, documents, embed_model


def retrieve_for_disease(disease_name):

    index, documents, embed_model = load_vector_store()

    symptoms = SYMPTOM_MAP.get(disease_name, [])

    symptom_text = " ".join(symptoms)

    query = f"""
    cotton crop disease {disease_name}
    symptoms {symptom_text}
    treatment prevention organic remedy
    """

    query_vec = embed_model.encode([query], convert_to_numpy=True)

    query_vec = query_vec.astype(np.float32)

    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, TOP_K)

    results = []

    for idx in indices[0]:

        if idx == -1:
            continue

        results.append(documents[idx])

    return results