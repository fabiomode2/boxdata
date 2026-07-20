from __future__ import annotations

import logging

from tqdm import tqdm

from src.models import Movie
from src.vector_store import get_collection

logger = logging.getLogger(__name__)

_BATCH_SIZE = 50


def index_movies(
    movies: list[Movie],
    overwrite: bool = False,
    db_path: str | None = None,
) -> int:
    """Generate embeddings in batches and store movies in ChromaDB.

    overwrite=True uses upsert() to re-embed existing movies.
    db_path sets a custom ChromaDB directory (e.g. for dev DBs).
    """
    logging.getLogger("httpx").setLevel(logging.WARNING)

    from src.embedders import OllamaEmbedder

    embedder = OllamaEmbedder()

    col = get_collection(db_path=db_path)

    seen_ids: set[int] = set()
    uniq_movies: list[Movie] = []
    for m in movies:
        if m.tmdb_id not in seen_ids:
            seen_ids.add(m.tmdb_id)
            uniq_movies.append(m)

    if overwrite:
        to_index = uniq_movies
    else:
        all_id_strs = [str(m.tmdb_id) for m in uniq_movies]
        in_db: set[int] = set()
        for i in range(0, len(all_id_strs), 999):
            batch_ids = all_id_strs[i : i + 999]
            existing = col.get(ids=batch_ids)["ids"]
            in_db.update(int(tid) for tid in existing)
        to_index = [m for m in uniq_movies if m.tmdb_id not in in_db]

    if not to_index:
        return 0

    total = len(to_index)
    indexed = 0

    add_or_upsert = col.upsert if overwrite else col.add

    with tqdm(total=total, desc="Embedding", unit="movie") as pbar:
        for i in range(0, total, _BATCH_SIZE):
            batch = to_index[i : i + _BATCH_SIZE]
            texts = [m.enriched_text() for m in batch]
            vectors = embedder.embed_batch(texts)

            add_or_upsert(
                ids=[str(m.tmdb_id) for m in batch],
                embeddings=vectors,
                documents=texts,
                metadatas=[m.metadata() for m in batch],
            )
            indexed += len(batch)
            pbar.update(len(batch))

    return indexed
