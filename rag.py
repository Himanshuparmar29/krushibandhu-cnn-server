"""
rag.py — Lightweight TF-IDF knowledge retrieval (no PyTorch/FAISS)
Replaces the original faiss + sentence-transformers implementation
which is too heavy for Render free tier (512 MB RAM).
"""

import os
import re
import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from symptom_map import SYMPTOM_MAP

KNOWLEDGE_DIR = "knowledge"   # folder with .txt disease files
CHUNK_SIZE    = 600
CHUNK_OVERLAP = 60
TOP_K         = 4

# ── Build index once at import time ───────────────────────────────────────────
_chunks: List[Dict] = []
_vectorizer: TfidfVectorizer = None
_matrix = None


def _chunk_text(text: str, source: str) -> List[Dict]:
    chunks, start = [], 0
    text = re.sub(r'\r\n', '\n', text)
    while start < len(text):
        end   = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 60:
            chunks.append({"text": chunk, "source": source})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _build_index():
    global _chunks, _vectorizer, _matrix
    path = Path(KNOWLEDGE_DIR)
    if not path.exists():
        print(f"[RAG] Knowledge dir '{KNOWLEDGE_DIR}' not found — RAG disabled.")
        return

    all_chunks = []
    for txt in sorted(path.glob("*.txt")):
        raw    = txt.read_text(encoding="utf-8", errors="ignore")
        chunks = _chunk_text(raw, txt.stem)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("[RAG] No .txt files found.")
        return

    _chunks = all_chunks
    texts   = [c["text"] for c in all_chunks]
    _vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)
    _matrix = _vectorizer.fit_transform(texts)
    print(f"[RAG] TF-IDF index: {len(_chunks)} chunks, {_matrix.shape[1]} features")


# Build at import time (module-level, works with Gunicorn)
_build_index()


def retrieve_for_disease(disease_name: str) -> List[Dict]:
    """Return top-K relevant chunks for the given disease name."""
    if not _chunks or _vectorizer is None:
        return []

    symptoms     = SYMPTOM_MAP.get(disease_name, [])
    symptom_text = " ".join(symptoms)
    query        = f"cotton crop disease {disease_name} symptoms {symptom_text} treatment prevention organic remedy"

    q_vec  = _vectorizer.transform([query])
    scores = cosine_similarity(q_vec, _matrix)[0]
    top_k  = scores.argsort()[::-1][:TOP_K]

    return [
        {**_chunks[i], "score": float(scores[i])}
        for i in top_k if scores[i] > 0.01
    ]