#!/usr/bin/env python3
"""Load collected movies from JSONL, generate embeddings, store in ChromaDB."""
import json
import logging
from pathlib import Path

from src.indexer import index_movies
from src.models import Movie
from src.vector_store import count_movies

logger = logging.getLogger(__name__)

DATA_FILE = Path(__file__).parent / "data" / "movies.jsonl"


def run_embed() -> None:
    if not DATA_FILE.exists():
        logger.error("%s not found. Run collect.py first.", DATA_FILE)
        return

    with open(DATA_FILE) as f:
        movies = [Movie.from_dict(json.loads(line)) for line in f]

    logger.info("Loaded %d movies from %s", len(movies), DATA_FILE)
    indexed = index_movies(movies)
    logger.info("Done. %d new movies indexed. Total in ChromaDB: %d", indexed, count_movies())


if __name__ == "__main__":
    run_embed()
