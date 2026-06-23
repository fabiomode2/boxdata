# BoxData — Movie Recommendation System

## Overview
Recommend movies that multiple people will enjoy, using TMDB data + Gemini 2 embeddings + ChromaDB vector similarity.

## Stack
- Python 3.11+ (API data collection, embedding generation)
- TMDB API (movie data)
- Gemini API (embeddings — `gemini-embedding-001`, 768-dim, taskType=`RETRIEVAL_DOCUMENT`)
- ChromaDB (persistent vector DB, stores both embeddings + metadata)

## Key Links
- Letterboxd TMDB page: `https://letterboxd.com/tmdb/{TMDB_ID}`
- TMDB search: `/search/movie`

## Credentials
- TMDB API key/token + Gemini API key in `.env`

## Running
- Activate venv first: `source .venv/bin/activate`

## Data Pipeline
1. `python main.py collect --start 1900 --end 2026 --max-per-year 100` fetches from TMDB
2. `python main.py collect --test` runs only 2 years for quick testing
3. `python main.py count` shows total movies in ChromaDB
4. Fetches `/discover/movie?sort_by=popularity.desc&primary_release_year=YEAR` + `/movie/{id}/credits` + `/movie/{id}/reviews`
5. Text profile for embedding = natural language narrative: `"Title is a YEAR film. Genres: ... Directed by ... Starring ... Plot: ... It has been [quality_text] by [audience_text]."` (qualitative rating/audience embedded for semantic signal; exact numbers stay in metadata)
6. Generates 768-dim embedding with Gemini (`taskType=RETRIEVAL_DOCUMENT`) → stores in ChromaDB: `id=tmdb_id, embedding=vector, document=text, metadata={title, genres, year, vote_average, vote_count}`
7. Ratings and vote counts stay in ChromaDB metadata for post-query filtering/reranking

## Recommendation Logic (pending)
- Given 2+ Letterboxd profile URLs, fetch watched/watchlist movies
- Query ChromaDB with embedding + metadata filters (genre, year, etc.)
- Re-rank with `score * (vote_average * log(vote_count))` per Estrategia B
- Filter out already-watched movies in memory

## Structure
- `main.py` — CLI entry point
- `src/collector.py` — TMDB API fetcher
- `src/models.py` — `Movie` dataclass with `enriched_text()` and `metadata()`
- `src/vector_store.py` — ChromaDB client wrapper
- `src/indexer.py` — embeds with Gemini and indexes into ChromaDB
- `data/chroma_db/` — persistent vector store (gitignored)

## Rate Limits
- TMDB: keep 0.25s delay between calls (already in collector)
- Gemini: batch embedding (50/request), automatic 429 retry with backoff (already in indexer)
