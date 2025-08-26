"""
Vector store helpers for Chroma (persistent local index).
"""
from __future__ import annotations

from pathlib import Path
import chromadb


def get_collection(persist_dir: str = "data/chroma", collection_name: str = "books_v1"):
    """
    Return a Chroma collection, creating it if needed.
    """
    p = Path(persist_dir)
    p.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(p))

    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # cosine distance for OpenAI embeddings
        )
    except TypeError:
        # Fallback for versions not supporting metadata kw
        collection = client.get_or_create_collection(name=collection_name)
    return collection
