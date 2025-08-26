"""
Retriever: embed the user query and fetch top-k documents from Chroma.
"""
from __future__ import annotations

from typing import Any, Dict, List
from app.llm.openai_client import OpenAIClient
from .vectorstore import get_collection


def retrieve(
    query: str,
    top_k: int = 5,
    persist_dir: str = "data/chroma",
    collection_name: str = "books_v1",
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k documents for a given natural-language query.

    Returns a list of dicts:
        {
            "id": str,
            "document": str,
            "title": str,
            "themes": str | None,   # comma-separated
            "source": str,
            "distance": float,      # lower is better (cosine distance)
        }
    """
    client = OpenAIClient()
    [q_emb] = client.embed_texts([query])

    col = get_collection(persist_dir=persist_dir, collection_name=collection_name)
    res = col.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    out: List[Dict[str, Any]] = []
    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    for i in range(len(ids)):
        md = metas[i] or {}
        out.append({
            "id": ids[i],
            "document": docs[i],
            "title": md.get("title"),
            "themes": md.get("themes"),
            "source": md.get("source"),
            "distance": dists[i],
        })
    return out


if __name__ == "__main__":
    # Quick manual test (requires ingested index and .env)
    results = retrieve("friendship and magic at a wizard school", top_k=3)
    for r in results:
        print(r["title"], "| distance:", round(r["distance"], 4))
