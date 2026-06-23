from __future__ import annotations

import time
from google import genai
from src.config import GEMINI_API_KEY
from src.models import Movie
from src.vector_store import get_collection

_client: genai.Client | None = None
_BATCH_SIZE = 50


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set in .env")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


_EMBED_CONFIG = {
    "taskType": "RETRIEVAL_DOCUMENT",
    "outputDimensionality": 768,
}


def _embed_batch(texts: list[str]) -> list[list[float]]:
    client = _get_client()
    while True:
        try:
            res = client.models.embed_content(
                model="gemini-embedding-001",
                contents=texts,
                config=_EMBED_CONFIG,
            )
            return [e.values for e in res.embeddings]
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                import re
                match = re.search(r"retry in (\d+)s", err)
                wait = int(match.group(1)) + 1 if match else 60
                print(f"  Gemini 429: retrying in {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise


def index_movies(movies: list[Movie]) -> int:
    """Generate embeddings in batches and store movies in ChromaDB."""
    col = get_collection()
    to_index = [m for m in movies if not col.get(str(m.tmdb_id))["ids"]]

    if not to_index:
        return 0

    indexed = 0
    total = len(to_index)
    for i in range(0, total, _BATCH_SIZE):
        batch = to_index[i : i + _BATCH_SIZE]
        texts = [m.enriched_text() for m in batch]
        vectors = _embed_batch(texts)

        col.add(
            ids=[str(m.tmdb_id) for m in batch],
            embeddings=vectors,
            documents=texts,
            metadatas=[m.metadata() for m in batch],
        )
        indexed += len(batch)

        print(f"  → embedded {indexed}/{total}", flush=True)

        if indexed < total:
            time.sleep(30)

    return indexed
