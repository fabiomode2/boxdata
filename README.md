# 🎬 BoxData

**Movie night for a group, not for an algorithm.**

BoxData reads everyone's Letterboxd history, turns it into taste vectors, and ranks films the whole group is likely to rate highly. No cloud LLM, no "because you watched" — just local embeddings and rating-aware kernel regression.

```
Letterboxd ──► scraper ──► taste vectors ─┐
                                           ├──► consensus ranking ──► React UI
TMDB ──► collector ──► 768-d embeddings ──┘         ▲
              (Ollama + ChromaDB)                   │
                                          FastAPI (jobs + progress)
```

## Why it is different

- **Group consensus, not one profile.** Each user gets a predicted rating per candidate via Nadaraya-Watson kernel regression; ratings are combined with a geometric mean so agreement wins.
- **Local and private.** `nomic-embed-text` runs through Ollama. Your watch history never goes to a third-party LLM.
- **Explainable rankings.** Similarity to your rated films, popularity, and optional director/decade/actor overlap — no black box.
- **Built for real Letterboxd data.** Cached scraping, stale-cache fallback, negative caching, and a persistent slug→TMDB map survive Cloudflare challenges.

## Quickstart

```bash
git clone https://github.com/fabiomode2/boxdata.git
cd boxdata

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env          # add your TMDB token and a strong API_KEY
python main.py collect --test # 2 years of movies
python main.py embed          # Ollama embeddings → ChromaDB
python main.py recommend --user alice bob
```

Start Ollama first (`ollama pull nomic-embed-text`). For the web app:

```bash
cd frontend && npm install && npm run dev
```

Or run the API + Ollama stack:

```bash
docker compose up -d --build
```

## CLI

| Command | What it does |
|---|---|
| `collect` | TMDB movies → `data/movies.jsonl` |
| `embed` | Movie text → ChromaDB vectors |
| `recommend` | Group, single-user, or text-query recommendations |
| `evaluate` | k-fold CV: MAE, RMSE, Precision@5, nDCG@5, MRR |
| `explore` | UMAP + HDBSCAN map of the embedding space |
| `filter` | Quality/rating filters over the dataset |
| `llm-text` | Optional richer descriptions via LLM |
| `analyze` | Letterboxd taste profile |

Useful flags: `--n`, `--diversity`, `--auto-sigma`, `--genre`, `--min-year`, `--min-rating`, `--query`, `--query-weight`.

## API

| Endpoint | Auth | Purpose |
|---|---|---|
| `POST /api/recommend` | `X-API-Key` | Start a background recommendation job |
| `GET /api/job/{id}` | none | Poll progress/results (unpredictable job ID) |
| `GET /api/avatar/{username}` | `X-API-Key` | Cached Letterboxd avatar |
| `GET /health` | none | Liveness + indexed movie count |

Rate limits, request-size caps, concurrent-job caps, and CORS allow-listing are enabled by default. API docs are disabled in production.

## Configuration

| Variable | Purpose |
|---|---|
| `API_TOKEN` | TMDB bearer token |
| `API_KEY` | Shared secret between the Vercel proxy and FastAPI |
| `CORS_ORIGINS` | Comma-separated allowed frontend origins |
| `OLLAMA_HOST` | Ollama URL (default `http://localhost:11434`) |
| `BOXSCRAPE_CACHE_TTL` | Letterboxd cache freshness (default 7 days) |
| `BOXSCRAPE_FAILURE_TTL` | Cooldown after a failed scrape (default 15 min) |
| `MAX_CONCURRENT_JOBS` | Max simultaneous recommendation jobs (default 4) |
| `BOXDATA_UID` / `BOXDATA_GID` | Host UID/GID that owns `./data` (default 1000) |

`.env` is gitignored. Only `.env.example` belongs in the repo.

## Tests

```bash
pytest -q
```

## Docs

- [`docs/SECURITY.md`](docs/SECURITY.md) — deployment trust model
- [`docs/technical-analysis.md`](docs/technical-analysis.md) — scoring details
- [`docs/roadmap-v2.md`](docs/roadmap-v2.md) — history and next steps
