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
| `filter` | Filter movies by rating/vote thresholds |
| `llm-text` | Generate rich text descriptions with LLM |
| `count` | Show movie count in ChromaDB |
| `clear-db` | Delete all embeddings from ChromaDB |
| `analyze --export DIR` | Analyze Letterboxd taste profile |
| `explore` | UMAP + HDBSCAN + Plotly visualization |
| `recommend --user alice bob` | Group recs (scrapes `/films/page/N/`) |
| `recommend --export DIR` | Recs from Letterboxd CSV export |
| `recommend --ids 550 680` | Recs from TMDB IDs |
| `recommend --query "cyberpunk"` | Pure text semantic search |
| `recommend --export DIR --query "noir"` | Hybrid (profile + text) |

Common options: `--n`, `--min-votes`, `--min-rating`, `--diversity` (0=relevance, 1=variety), `--sigma`, `--max-distance`, `--auto-sigma`, `--genre`, `--min-year`/`--max-year`, `--randomize`, `--query-weight`, `--no-feature-rerank`.

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
├── collector.py      TMDB fetcher (rate-limited, parallel credits, curl-cffi)
├── imdbapi_enricher.py  IMDb API enrichment via imdbapi.dev (batch + individual, curl-cffi)
├── models.py         Movie dataclass (enriched_text, metadata)
├── vector_store.py   ChromaDB wrapper (cosine space)
├── indexer.py        Embed + index pipeline
├── embedders.py      OllamaEmbedder (nomic-embed-text)
├── profile.py        Letterboxd CSV parser + TMDB resolver (curl-cffi)
├── recommender.py    Clustering, kernel regression, MMR, scraper
├── review_fetcher.py TMDB review fetch for LLM-enhanced embeddings
└── explorer.py       UMAP + HDBSCAN + Plotly
api/server.py         FastAPI JSON API (port 8000)
Dockerfile.api        Docker image for the API
docker-compose.yml    Ollama + API stack for Oracle VPS
frontend/             React + Vite frontend (deploy to Vercel)
tests/                Pytest suite (31 tests, all passing)
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

## Agent Skills (OpenCode)

Skills loaded for BoxData development:

| Skill | Source | Area |
|-------|--------|------|
| data-pipeline | local | ETL pipeline patterns |
| web-scraping | local | Letterboxd scraping, curl-cffi, anti-bot |
| python-design-patterns | local | Architecture, SRP, composition |
| python-testing-patterns | local | pytest, fixtures, mocking |
| python-performance-optimization | local | Profiling, batch scoring |
| react-dev | local | React 18 type-safe components |
| vite | local | Vite 6 build tool |
| google-gemini-embeddings | local | Alternative embedding model |
| cavecrew | local | Compressed subagents for long sessions |
| fastapi-python | skills.sh (11.4K) | FastAPI API server patterns |
| machine-learning | skills.sh (1.5K) | ML patterns for kernel regression |
| frontend-react-best-practices | skills.sh (983) | React best practices |
| local-llm-ops | skills.sh (289) | Ollama/local LLM operations |
| umap-learn | skills.sh (225) | UMAP dimensionality reduction |

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
10. `curl-cffi` everywhere (not mixing with `requests`) avoids inconsistent blocking
11. Caching `_build_chromadb_index()` avoids reloading all metadata on every TMDB resolution
12. Feature reranking bonus debe tener cap (máx 10%) o distorsiona el ranking
13. Tests que cambian de semántica (IMDb→TMDB) deben renombrarse, no solo cambiarse las aserciones
14. Cuando migras HTTP lib, cazar la excepción específica (`CurlError`) en vez de `RequestException`

## Changelog

### Jul 2026 — Evaluation Framework (Fase 1)
- Created `src/evaluator.py` — `evaluate_user()` con k-fold CV + Nadaraya-Watson
- Added `evaluate` CLI command (`--users`, `--export`, `--k`, `--sigma`, `--output`)
- Metrics: MAE, RMSE, Precision@5, nDCG@5, MRR (mean + std across folds)
- LOO fallback for <10 movies, random seed fijo (42) para reproducibilidad
- `_print_evaluation_results()` tabla formateada + macro-average entre usuarios
- 15 tests in `tests/test_evaluator.py` — todos pasando (56/56 total)
- Baseline ejecutado: 4 usuarios reales (fabiomode2, narrowd, efrx, georgemcnuggets)
- MAE promedio: 0.1250 (±0.0186) con nomic-embed-text

### Jul 2026 — Code Quality + Feature Reranking + Pipeline Prep
- Reordered roadmap: F1 (evaluation) now precedes F2 (feature reranking) for measurable progress
- Feature reranking implemented: `_feature_rerank()` boosts candidates sharing director (+10%), decade (+5%), or actors (Jaccard, +5% max), capped at 10% total
- Added `--feature-rerank`/`--no-feature-rerank` flag to CLI (on by default)
- Added `cast` field to ChromaDB metadata for actor overlap computation
- Eliminated redundant N individual ChromaDB calls in `taste_similarity()` via batch `_fetch_embeddings()`
- Removed dead code: `_get_embedding()` (no callers after taste_similarity fix)
- Fixed all 9 bare `except Exception: pass/blind return` with logger.debug/warning + context
- Added `CurlError` import to recommender.py for specific exception handling in slug resolver
- Integrated `review_fetcher.py` into collector: `--enrich-reviews` flag fetches top reviews during collection (default on)
- Created `src/filter_runner.py` + `filter` CLI command (rating/vote thresholds → JSONL)
- Created `src/llm_text_generator.py` + `llm-text` CLI command (Ollama + OpenAI-compatible, resumable)
- Added `--skip-reviews` flag to collect command

### Jul 2026 — Bugfixes & Refinements
- Fixed ChromaDB batch size limit: `_fetch_user_embeddings` now chunks at 999 IDs (was unlimited)
- DRY refactor: `_load_user_history` now delegates to `_fetch_user_embeddings` (-20 LOC)
- `embed` command now accepts `--input` for alternative JSONL files (needed for LLM text pipeline)
- MMR diversity now uses numpy-vectorized cosine (`_cosine_similarity_np`)
- Renamed `imdb_enricher.py` → `imdbapi_enricher.py` (clarifies it uses imdbapi.dev, not official IMDb)
- Added OpenCode Go / DeepSeek V4 Flash strategy to embedding upgrade plan

### Jul 2026 — Codebase Cleanup & Review Fetcher
- Fixed broken test `test_best_vote_data_prefers_imdb` → `test_tmdb_vote_data_uses_tmdb_only`
- Renamed `_best_vote_data` → `_tmdb_vote_data` for consistency
- Fixed resume bug (off-by-one) in `collect_runner.py`
- Unified HTTP libs: collector.py, imdbapi_enricher.py, profile.py → `curl-cffi` (was `requests`)
- Letterboxd scraper in profile.py now uses `Session(impersonate="safari17_0")`
- Created `src/review_fetcher.py` — `fetch_top_reviews()` + `enrich_movie_with_reviews()`
- Updated roadmap-v2.md with bonus cap (10%), review fetcher, HTTP cleanup
- Added lessons 12-14 to AGENTS.md

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

---

## Temp Plan File Lifecycle

`docs/plan_actual_temp.md` holds current work-in-progress plan for active phase.
- Created at phase start with full implementation details
- Deleted when phase completes
- Results documented concisely in `docs/roadmap-v2.md` (Changelog section)
- One temp file at a time; outdated plans are cleaned
