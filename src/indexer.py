from __future__ import annotations

import logging

from src.models import Movie
from src.vector_store import get_collection

logger = logging.getLogger(__name__)

_BATCH_SIZE = 50


def index_movies(
    movies: list[Movie],
) -> int:
    """Generate embeddings in batches and store movies in ChromaDB."""
    from src.embedders import OllamaEmbedder

    embedder = OllamaEmbedder()

    col = get_collection()

    seen_ids: set[int] = set()
    uniq_movies: list[Movie] = []
    for m in movies:
        if m.tmdb_id not in seen_ids:
            seen_ids.add(m.tmdb_id)
            uniq_movies.append(m)

    all_id_strs = [str(m.tmdb_id) for m in uniq_movies]
    existing = col.get(ids=all_id_strs)["ids"]
    in_db: set[int] = {int(tid) for tid in existing}
    to_index = [m for m in uniq_movies if m.tmdb_id not in in_db]

    if not to_index:
        return 0

    indexed = 0
    total = len(to_index)

    for i in range(0, total, _BATCH_SIZE):
        batch = to_index[i : i + _BATCH_SIZE]
        texts = [m.enriched_text() for m in batch]
        vectors = embedder.embed_batch(texts)

        col.add(
            ids=[str(m.tmdb_id) for m in batch],
            embeddings=vectors,
            documents=texts,
            metadatas=[m.metadata() for m in batch],
        )
        indexed += len(batch)

        logger.info("  → embedded %d/%d", indexed, total)

    return indexed
