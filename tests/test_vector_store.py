from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from src.vector_store import get_collection, count_movies, delete_collection


def test_vector_store_custom_path() -> None:
    tmp = tempfile.mkdtemp()
    try:
        custom_path = str(Path(tmp) / "custom_chroma")

        col = get_collection(db_path=custom_path)
        assert col.count() == 0

        col.add(ids=["1"], embeddings=[[0.1, 0.2, 0.3]], metadatas=[{"title": "Test"}])

        assert count_movies(db_path=custom_path) == 1

        col2 = get_collection(db_path=custom_path)
        assert col2.count() == 1
        assert col2.get(ids=["1"])["metadatas"][0]["title"] == "Test"

        delete_collection(db_path=custom_path)
        assert count_movies(db_path=custom_path) == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
