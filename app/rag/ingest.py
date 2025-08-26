# app/rag/ingest.py
"""
Ingest short book summaries into Chroma (persistent, local).

- Parses data/book_summaries.md (Title, short summary, Themes)
- Creates embeddings with OpenAI (text-embedding-3-small by default)
- Upserts into a Chroma persistent collection
- Stores useful metadata (title, themes, source)

Run:
    python -m app.rag.ingest \
        --data-file data/book_summaries.md \
        --persist-dir data/chroma \
        --collection books_v1
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import chromadb

from app.llm.openai_client import OpenAIClient


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "untitled"


def parse_book_summaries_md(path: Path) -> List[Dict]:
    """
    Parse the markdown file with blocks like:

    ## Title: 1984
    <3-5 lines summary>
    Themes: surveillance, totalitarianism, freedom

    Returns:
        List[dict] with keys: title, summary, themes (list[str])
    """
    text = path.read_text(encoding="utf-8")
    # Split by "## Title:" headers
    parts = re.split(r"(?m)^##\s*Title:\s*", text)
    entries: List[Dict] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # First line is the title up to newline; rest is content
        if "\n" in part:
            title_line, rest = part.split("\n", 1)
        else:
            title_line, rest = part, ""
        title = title_line.strip()

        # Find Themes line
        m = re.search(r"(?mi)^Themes:\s*(.+)$", rest)
        themes: List[str] = []
        if m:
            themes_line = m.group(1)
            themes = [t.strip() for t in re.split(r"[,|]", themes_line) if t.strip()]
            summary_text = rest[: m.start()].strip()
        else:
            summary_text = rest.strip()

        # Normalize summary (condense blank lines)
        summary_lines = [ln.strip() for ln in summary_text.splitlines() if ln.strip()]
        summary = "\n".join(summary_lines)
        if not title or not summary:
            # Skip malformed blocks
            continue

        entries.append({"title": title, "summary": summary, "themes": themes})
    return entries


def build_documents(entries: List[Dict]) -> List[str]:
    """
    Build the text to be embedded/searched in Chroma for each book.
    Include title, summary, and themes in a compact form.
    """
    docs: List[str] = []
    for e in entries:
        themes_str = ", ".join(e.get("themes", [])) if e.get("themes") else ""
        doc = f"Title: {e['title']}\nSummary: {e['summary']}"
        if themes_str:
            doc += f"\nThemes: {themes_str}"
        docs.append(doc)
    return docs


def upsert_into_chroma(
    docs: List[str],
    entries: List[Dict],
    embeddings: List[List[float]],
    persist_dir: Path,
    collection_name: str,
) -> None:
    """
    Upsert documents with precomputed embeddings into a persistent Chroma collection.
    """
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    # Create or get collection (cosine distance is typical for OpenAI embeddings)
    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    except TypeError:
        # Fallback for versions not supporting metadata kw
        collection = client.get_or_create_collection(name=collection_name)

    ids = [slugify(e["title"]) for e in entries]
    metadatas = [
    {
        "title": e["title"],
        "themes": ", ".join(e.get("themes", [])) if e.get("themes") else None,
        "source": "book_summaries.md",
    }
    for e in entries
]


    # Upsert if available, else add
    if hasattr(collection, "upsert"):
        collection.upsert(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)
    else:
        # Legacy: delete then add to avoid duplicates
        try:
            collection.delete(ids=ids)
        except Exception:
            pass
        collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)

    print(f"Upserted {len(ids)} documents into collection '{collection_name}' at {persist_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest short book summaries into Chroma.")
    parser.add_argument("--data-file", type=str, default="data/book_summaries.md")
    parser.add_argument("--persist-dir", type=str, default="data/chroma")
    parser.add_argument("--collection", type=str, default="books_v1")
    args = parser.parse_args()

    data_path = Path(args.data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    entries = parse_book_summaries_md(data_path)
    if len(entries) < 10:
        raise RuntimeError(f"Expected at least 10 book entries, found {len(entries)}")

    docs = build_documents(entries)

    # Create embeddings with OpenAI
    client = OpenAIClient()  # loads settings from .env
    embeddings = client.embed_texts(docs)

    persist_dir = Path(args.persist_dir)
    upsert_into_chroma(
        docs=docs,
        entries=entries,
        embeddings=embeddings,
        persist_dir=persist_dir,
        collection_name=args.collection,
    )

    # Optional: write a manifest for traceability
    manifest = {
        "collection": args.collection,
        "count": len(entries),
        "data_file": str(data_path),
        "persist_dir": str(persist_dir),
    }
    manifest_path = persist_dir / f"{args.collection}_manifest.json"
    try:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote manifest: {manifest_path}")
    except Exception as e:
        print(f"Manifest write skipped: {e}")


if __name__ == "__main__":
    main()
