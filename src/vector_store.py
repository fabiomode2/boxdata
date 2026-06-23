from __future__ import annotations

import chromadb
from pathlib import Path

DB_DIR = Path(__file__).resolve().parent.parent / "data"

_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None


def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        DB_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(str(DB_DIR / "chroma_db"))
    return _client


def get_collection(name: str = "movies") -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def count_movies() -> int:
    col = get_collection()
    return col.count()


def delete_collection(name: str = "movies") -> None:
    """Drop the entire collection from ChromaDB. All embeddings gone."""
    global _collection
    _collection = None
    client = get_client()
    try:
        client.delete_collection(name)
    except ValueError:
        pass  # already doesn't exist
