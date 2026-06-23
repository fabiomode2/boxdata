#!/usr/bin/env python3
"""Load collected movies from JSONL, generate embeddings, store in ChromaDB."""
import json
from pathlib import Path

from src.indexer import index_movies
from src.models import Movie
from src.vector_store import count_movies

DATA_FILE = Path(__file__).parent / "data" / "movies.jsonl"


def run_embed() -> None:
    if not DATA_FILE.exists():
        print(f"ERROR: {DATA_FILE} not found. Run collect.py first.")
        return

    with open(DATA_FILE) as f:
        movies = [Movie.from_dict(json.loads(line)) for line in f]

    print(f"Loaded {len(movies)} movies from {DATA_FILE}", flush=True)
    indexed = index_movies(movies)
    print(f"Done. {indexed} new movies indexed. Total in ChromaDB: {count_movies()}", flush=True)


if __name__ == "__main__":
    run_embed()
