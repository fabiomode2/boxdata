# BoxData

Movie recommendation system using TMDB + Ollama embeddings + ChromaDB.

Group movie recommender. Reads Letterboxd histories, embeds movie metadata with
`nomic-embed-text` through Ollama, stores vectors in ChromaDB, predicts each
user's rating with Nadaraya-Watson kernel regression, and combines predictions
into one ranked list.

Demo: https://boxdata.vercel.app

## Stack

- Python 3.11, FastAPI, ChromaDB, Ollama
- TMDB metadata, Letterboxd histories
- React + Vite frontend
- Docker Compose for API + Ollama

## Pipeline

1. `collect` fetches movies from TMDB into `data/movies.jsonl`.
2. `embed` builds a text profile per movie and indexes its 768-d vector.
3. `recommend` loads each user's Letterboxd ratings.
4. Kernel regression estimates each user's rating for every candidate.
5. Predictions are combined with a geometric mean and reranked by popularity
   plus optional director/decade/actor overlap.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env

ollama pull nomic-embed-text

python main.py collect --test
python main.py embed
python main.py recommend --user alice bob
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Docker:

```bash
docker compose up -d --build
```

## CLI

| Command | Description |
|---|---|
| `collect` | TMDB movies → JSONL |
| `embed` | Text profiles → ChromaDB vectors |
| `recommend` | Group, single-user, or text-query recommendations |
| `evaluate` | k-fold CV: MAE, RMSE, Precision@5, nDCG@5, MRR |
| `explore` | UMAP + HDBSCAN visualization |
| `filter` | Rating/vote filters |
| `llm-text` | Optional LLM descriptions |
| `analyze` | Letterboxd taste profile |

Useful flags: `--n`, `--diversity`, `--auto-sigma`, `--genre`, `--min-year`,
`--min-rating`, `--query`, `--query-weight`.

## API

| Endpoint | Auth | Description |
|---|---|---|
| `POST /api/recommend` | `X-API-Key` | Start recommendation job |
| `GET /api/job/{id}` | none | Poll job status |
| `GET /api/avatar/{username}` | `X-API-Key` | Cached Letterboxd avatar |
| `GET /health` | none | Health + movie count |

## Configuration

| Variable | Description |
|---|---|
| `API_TOKEN` | TMDB bearer token |
| `API_KEY` | Shared secret for FastAPI |
| `CORS_ORIGINS` | Allowed frontend origins |
| `OLLAMA_HOST` | Ollama URL |
| `BOXSCRAPE_CACHE_TTL` | Letterboxd cache TTL |
| `BOXSCRAPE_FAILURE_TTL` | Failed-scrape cooldown |
| `MAX_CONCURRENT_JOBS` | Recommendation job limit |
| `BOXDATA_UID` / `BOXDATA_GID` | Owner of `./data` |

`.env` is gitignored. Use `.env.example` as the template.

## Tests

```bash
pytest -q
```

67 tests.

## Data

Movie metadata comes from TMDB; watch histories come from Letterboxd. No movie
database or user data is included in the repository. Respect both services'
terms and rate limits.

## Docs

- [`docs/SECURITY.md`](docs/SECURITY.md)
- [`docs/technical-analysis.md`](docs/technical-analysis.md)
- [`docs/performance-tuning.md`](docs/performance-tuning.md)
- [`docs/roadmap-v2.md`](docs/roadmap-v2.md)
- [`docs/embedding-upgrade-plan.md`](docs/embedding-upgrade-plan.md)
