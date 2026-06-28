# BoxData — Movie Recommendation System

Recommend movies that multiple people will enjoy, using TMDB data + Ollama embeddings + ChromaDB vector similarity.

## Stack
- Python 3.11+, TMDB API (movie data), Ollama + nomic-embed-text (local, 768-dim, ~274MB), ChromaDB (persistent vector DB)
- API key/token in `.env`

## Quickstart
```bash
source .venv/bin/activate
python main.py help
python main.py collect --test       # fetch 2 years for testing
python main.py embed                # generate embeddings → ChromaDB
python main.py explore              # UMAP + HDBSCAN visualization
python main.py recommend --user alice bob   # group recs
cd frontend && npm run dev           # React dev server (Vite)
docker compose up -d                 # API + Ollama on Oracle VPS
```

## Commands

| Command | Description |
|---------|-------------|
| `collect --start Y --end Y --max-per-year N` | Fetch movies from TMDB → JSONL |
| `collect --test` | 2 years quick test |
| `collect --skip-imdb` | Disable IMDb enrichment |
| `embed` | Generate embeddings from JSONL → ChromaDB |
| `count` | Show movie count in ChromaDB |
| `clear-db` | Delete all embeddings from ChromaDB |
| `analyze --export DIR` | Analyze Letterboxd taste profile |
| `explore` | UMAP + HDBSCAN + Plotly visualization |
| `recommend --user alice bob` | Group recs (scrapes `/films/page/N/`) |
| `recommend --export DIR` | Recs from Letterboxd CSV export |
| `recommend --ids 550 680` | Recs from TMDB IDs |
| `recommend --query "cyberpunk"` | Pure text semantic search |
| `recommend --export DIR --query "noir"` | Hybrid (profile + text) |

Common options: `--n`, `--min-votes`, `--min-rating`, `--diversity` (0=relevance, 1=variety), `--sigma`, `--max-distance`, `--auto-sigma`, `--genre`, `--min-year`/`--max-year`, `--randomize`, `--query-weight`.

## How It Works

### Data Pipeline
1. `collect` fetches `/discover/movie` per year (sorted by popularity, oversampled 3x, re-ranked by `vote_count * max(0.1, vote_avg - 2.5)`)
2. Fetches credits (director, top-5 cast, `imdb_id`) via `/movie/{id}?append_to_response=credits`
3. Optionally enriches via `imdbapi.dev`: IMDb rating/votes, Metacritic, interests/subgenres, origin countries, spoken languages
4. Text profile for embedding = natural language narrative: `"Title (year). Genres: ... Directed by ... Starring ... Plot: ... It has been [quality] by [audience]."`
5. 768-dim embedding via Ollama (nomic-embed-text) → ChromaDB with `hnsw:space: cosine`

### Recommendation Engine

**Group consensus** (used by `--user`, `--export`):
- Each user's taste = full watch history as embedded vectors + ratings
- For each candidate, predicts each user's rating via **Nadaraya-Watson kernel regression**:
  - `weight_i = exp(-dist² / sigma²)` where `dist = 1 - cos_sim(emb_candidate, emb_history_i)`
  - `f(user, candidate) = Σ(weight_i * rating_i) / Σ(weight_i)`
- **Consensus** = geometric mean across users (square reinforces agreement)
- `--auto-sigma` calibrates bandwidth via k-fold CV over [0.05-1.0]
- Re-ranked via **Reciprocal Rank Fusion** balancing similarity + popularity

**Single user**: taste clusters (weighted k-means, K=2-4, with fallbacks for <15 movies)

**Text query**: embed query + ChromaDB cosine search + RRF

**Hybrid**: blends kernel regression consensus with query similarity via `--query-weight`

**Diversity**: optional Maximum Marginal Relevance (`--diversity`)

### Taste Similarity
Bidirectional top-3 mean nearest-neighbour between two users' movie embeddings.

## Project Structure
```
main.py               CLI entry point (argparse dispatcher)
pyproject.toml        Project config + [project.scripts] entry
src/
├── collect_runner.py TMDB collection runner
├── embed_runner.py   Embedding runner
├── collector.py      TMDB fetcher (rate-limited, parallel credits)
├── imdb_enricher.py  IMDb API enrichment (batch + individual)
├── models.py         Movie dataclass (enriched_text, metadata)
├── vector_store.py   ChromaDB wrapper (cosine space)
├── indexer.py        Embed + index pipeline
├── embedders.py      OllamaEmbedder (nomic-embed-text)
├── profile.py        Letterboxd CSV parser + TMDB resolver
├── recommender.py    Clustering, kernel regression, MMR, scraper
└── explorer.py       UMAP + HDBSCAN + Plotly
api/server.py         FastAPI JSON API (port 8000)
Dockerfile.api        Docker image for the API
docker-compose.yml    Ollama + API stack for Oracle VPS
frontend/             React + Vite frontend (deploy to Vercel)
tests/                Pytest suite (31 tests)
data/chroma_db/       Vector store (gitignored)
data/cache/           Scraped profiles (gitignored)
docs/                 Technical notes and changelog
misc/                 Experimental files (viewer.html)
```

## Rate Limits & Scraping
- TMDB: 0.25s between calls (lock-protected)
- Letterboxd scraper: 1s between `/films/` page fetches, Semaphore(5) on slug→TMDB resolution
- Profiles cached to `data/cache/{username}.json`; use `--no-cache` to bypass

## Key Technical Decisions
- **RRF over z-score normalization**: distribution-agnostic, rank-based, avoids noise from small pools
- **Multi-seed discovery > single centroid**: queries from each user centroid + random seeds → covers all taste regions
- **Numpy batch scoring**: `_predict_rating_batch()` vectorizes kernel regression as matrix multiply (~60x faster)
- **CSS selectors over regex** for Letterboxd scraping (BeautifulSoup, `li.griditem`, `[data-tmdb-id]`)
- **Geometric mean over arithmetic mean** for group consensus: softer than min, harder than avg, rewards genuine agreement

## Lessons Learned
1. ChromaDB cosine distance → `similarity = 1.0 - dist` (not `1/(1+dist)` which is for L2)
2. Quality factor in primary score penalises niche movies twice — quality belongs in reranking only
3. Taste weights should be rating-only; popularity handled in reranking
4. Taste similarity must be bidirectional top-3 (single-direction max inflates with large collections)
5. Recency bonus distorts ratings — use raw rating (0.1-1.0) as weight
6. Letterboxd URI → `data-tmdb-id` is faster and more reliable than TMDB search
7. Single source scraper (`/films/page/N/`) shows ALL watched films with current ratings
8. Profile resolver needs rate limiting to avoid Letterboxd blocking
9. `print()` for CLI output, `logging` for internal events (timestamps break table formatting)
10. Using `curl-cffi` consistently (not mixing with `requests`) avoids inconsistent blocking
11. Caching `_build_chromadb_index()` avoids reloading all metadata on every TMDB resolution

## Changelog

### Jun 2026 — Web App Redesign (React + Vite + Docker)
- Replaced `web/app.py` (FastAPI + server-side templates) with `frontend/` (React + Vite, deploy to Vercel)
- Created `api/server.py` — new FastAPI JSON API for the frontend
- Created `Dockerfile.api` + `docker-compose.yml` for Oracle VPS deployment
- Removed `web/` directory

### Jun 2026 — Project Structure Cleanup
- Moved `collect.py` → `src/collect_runner.py` and `embed.py` → `src/embed_runner.py`
- Fixed `embed.py` bug (missing `count_movies` import when run standalone)
- Removed stale `src/boxdata.egg-info/`
- Added `[project.scripts]` entry point to `pyproject.toml`

### Jun 2026 — Refactor & Bugfixes
- Unified `recommend_for_group`/`recommend_hybrid` into `_recommend_pipeline` (-160 LOC)
- Extracted `_fetch_user_embeddings` helper (DRY get_taste_vector/clusters)
- Removed quality factor from primary score (was double-penalising niche movies)
- Fixed `_matches_year_filter` falsy-0 bug
- Added Semaphore(5) rate limiting to profile URI resolver
- Switched `_resolve_slug_parallel` to curl_cffi for consistency
- Cached `_build_chromadb_index()` to avoid reloading ChromaDB metadata
- Replaced dead-code tests with real geometric mean + MMR + batch predict tests
- Added missing deps (numpy, umap, hdbscan, plotly, fastapi, uvicorn)
- Cleaned up dead code and imports; removed orphan Gemini API key
- Consolidated documentation into single AGENTS.md
- Organized `idea.txt`, PDF, and `viewer.html` into `docs/` and `misc/`

### May 2026 — Initial Implementation
- TMDB collection pipeline with IMDb enrichment
- Ollama nomic-embed-text embeddings → ChromaDB
- Letterboxd scraper (single-source `/films/page/N/`)
- Kernel regression (Nadaraya-Watson) + geometric mean consensus
- Numpy batch scoring (~60x speedup)
- Auto-sigma calibration via k-fold CV
- UMAP + HDBSCAN embedding space explorer
- FastAPI web demo
- 24 unit tests
