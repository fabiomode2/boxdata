#!/usr/bin/env python3
"""Load collected movies from JSONL, generate embeddings, store in ChromaDB."""
import json
import logging
from pathlib import Path

from src.indexer import index_movies
from src.models import Movie
from src.vector_store import count_movies

logger = logging.getLogger(__name__)

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "movies.jsonl"


def run_embed(input_file: str | None = None) -> None:
    data_file = Path(input_file) if input_file else DATA_FILE
    if not data_file.exists():
        logger.error("%s not found. Run collect first.", data_file)
        return

    with open(data_file) as f:
        movies = [Movie.from_dict(json.loads(line)) for line in f]

    logger.info("Loaded %d movies from %s", len(movies), data_file)
    indexed = index_movies(movies)
    logger.info("Done. %d new movies indexed. Total in ChromaDB: %d", indexed, count_movies())
