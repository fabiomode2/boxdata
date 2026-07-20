from __future__ import annotations

import chromadb
from pathlib import Path

DB_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CHROMA_DIR = DB_DIR / "chroma_db"

_clients: dict[str, chromadb.PersistentClient] = {}
_collections: dict[str, chromadb.Collection] = {}


def _chroma_path(db_path: str | None = None) -> str:
    return str(Path(db_path).resolve()) if db_path else str(DEFAULT_CHROMA_DIR)


def _collection_key(name: str = "movies", db_path: str | None = None) -> str:
    return f"{_chroma_path(db_path)}:{name}"


def get_client(db_path: str | None = None) -> chromadb.PersistentClient:
    path = _chroma_path(db_path)
    if path not in _clients:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        _clients[path] = chromadb.PersistentClient(path)
    return _clients[path]


def get_collection(name: str = "movies", db_path: str | None = None) -> chromadb.Collection:
    key = _collection_key(name, db_path)
    if key not in _collections:
        client = get_client(db_path=db_path)
        _collections[key] = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
    return _collections[key]


def count_movies(db_path: str | None = None) -> int:
    col = get_collection(db_path=db_path)
    return col.count()


def delete_collection(name: str = "movies", db_path: str | None = None) -> None:
    """Drop the entire collection from ChromaDB. All embeddings gone."""
    key = _collection_key(name, db_path)
    _collections.pop(key, None)
    client = get_client(db_path=db_path)
    try:
        client.delete_collection(name)
    except ValueError:
        pass  # already doesn't exist
