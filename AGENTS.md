# BoxData — Movie Recommendation System

## Overview
Recommend movies that multiple people will enjoy, using TMDB data + Ollama embeddings + ChromaDB vector similarity.

## Stack
- Python 3.11+ (API data collection, embedding generation)
- TMDB API (movie data)
- Ollama + nomic-embed-text (local embeddings, 768-dim, no rate limits, ~274MB)
- ChromaDB (persistent vector DB, stores both embeddings + metadata)

## Key Links
- Letterboxd TMDB page: `https://letterboxd.com/tmdb/{TMDB_ID}`
- TMDB search: `/search/movie`

## Credentials
- TMDB API key/token in `.env`

## Running
- Activate venv first: `source .venv/bin/activate`
- Help overview: `python main.py help`
- Per-command help: `python main.py <comando> -h`

## Commands
| Command | Description |
|---------|-------------|
| `python main.py collect --start 1900 --end 2026 --max-per-year 100` | Fetch movies from TMDB → JSONL (with IMDb enrichment) |
| `python main.py collect --test` | 2 years for quick testing |
| `python main.py collect --skip-imdb` | Disable IMDb API enrichment |
| `python main.py embed` | Generate embeddings from JSONL → ChromaDB |
| `python main.py count` | Show movie count in ChromaDB |
| `python main.py clear-db` | Delete ALL embeddings from ChromaDB |
| `python main.py analyze --export DIR` | Analyze Letterboxd taste profile |
| `python main.py explore` | UMAP + HDBSCAN + Plotly visualization of embedding space |
| `python main.py recommend --user alice bob` | Group recommendations from Letterboxd usernames (scrapes `/films/page/N/`) |
| `python main.py recommend --export DIR` | Recommendations from Letterboxd CSV export |
| `python main.py recommend --ids 550 680` | Recommendations from TMDB IDs one user likes |
| `python main.py recommend --query "cyberpunk"` | Pure text-based semantic search |
| `python main.py recommend --export DIR --query "noir"` | Hybrid: profile + text, blended by `--query-weight` (default 0.3) |
| `python web/app.py` | Start web demo (FastAPI, port 8080) |

All accept `--n` (results count), `--min-votes` (minimum vote_count filter), and `--diversity` (0=relevance, 1=variety, default 0), `--sigma`, `--max-distance`, `--auto-sigma`, `--genre` (comma-separated genre filter).

## Data Pipeline
1. `python main.py collect` fetches `/discover/movie?sort_by=popularity.desc&primary_release_year=YEAR` + `/movie/{id}?append_to_response=credits` (gets director, cast, AND `imdb_id`)
2. By default, also enriches via `imdbapi.dev`: `/titles/{imdb_id}` or `:batchGet` — adds IMDb rating/votes (larger sample), Metacritic score, interests/subgenres, origin countries, spoken languages
3. Text profile for embedding = natural language narrative: `"Title is a YEAR film. Genres: ... Subgenres: ... Directed by ... Starring ... Plot: ... It has been [quality_text] by [audience_text]."` (qualitative rating/audience embedded for semantic signal; exact numbers stay in metadata)
3. Generates 768-dim embedding with OllamaEmbedder (nomic-embed-text) → stores in ChromaDB: `id=tmdb_id, embedding=vector, document=text, metadata={title, genres, year, vote_average, vote_count}`
4. Ratings and vote counts stay in ChromaDB metadata for post-query filtering/reranking

## Recommendation Engines

### Single user: Taste Clusters
- User's watched movies are clustered into K groups (K=2-4 via weighted k-means, or fallback for <15 movies)
- Each cluster = a centroid vector representing a "mood" or genre preference
- Weights for each movie = rating only (popularity handled in reranking)
- Short history fallback: <5 movies → single centroid, 5-14 → each movie is its own cluster
- Solves the "purée problem": a user who loves both Oppenheimer and Barbie gets TWO clusters instead of one meaningless average

### Group: Consensus Scoring (used by `--user`, `--export`)
- Each user's taste = full watch history (embeddings + ratings), NOT centroids
- For each candidate, predicts each user's rating via **Nadaraya-Watson kernel regression** over their entire history
  - `weight_i = exp(-dist² / sigma²)` where `dist = 1 - cos_sim(emb_candidate, emb_history_i)`
  - `f(user, candidate) = Σ(weight_i * rating_i) / Σ(weight_i)`
- **Consensus** = geometric mean across users: `G = (Π f_i²)^(1/N)` (square reinforces agreement)
- Rewards movies liked by ALL, penalises divisive smoothly (geometric mean is softer than min, harder than arithmetic)
- `--sigma` controls smoothness (default 0.3, lower=spikier, higher=smoother)
- `--auto-sigma` calibrates sigma per user via k-fold CV (tests [0.05-1.0], picks lowest MAE)
- `--max-distance` filters out distant history entries (default 0.3, -1=no limit)
- Re-ranked via Reciprocal Rank Fusion (`_rerank_candidates`) balancing similarity + popularity

### Text query: Embed & Search
- Query text embedded with OllamaEmbedder (nomic-embed-text)
- ChromaDB cosine similarity search on the raw embedding
- Candidates re-ranked with RRF balancing similarity + popularity

### Hybrid: Profile + Text
- Per-user taste cluster centroids averaged, blended with query embedding vector via `--query-weight`
- Query is interpolated: `(1-w) * mean_taste + w * query_emb`
- For each candidate, per-user best-cluster match AND query similarity are both computed, then blended
- `--query-weight` controls blend (0=profile only, 1=text only)

## Taste Similarity (metric between two users)
- Uses **bidirectional top-3 mean nearest-neighbour**:
  - For each movie in A, average similarity of top-3 nearest neighbours in B
  - Same for B→A
  - Final score = mean of both directions
- Why NOT single-direction max: inflates when one user has many more movies (always finds SOME match)
- Top-3 avoids overfitting to a single accidentally-similar movie
- Typical range: 60-80% for genuine overlap, 80%+ is very high

## Re-ranking (`_rerank_candidates`)
ChromaDB uses `hnsw:space: cosine`, so `distance = 1 - cosine_similarity`.
Uses Reciprocal Rank Fusion (RRF) instead of z-score normalization:
`score = alpha / (k + rank_sim) + (1-alpha) / (k + rank_pop)`
where k=60 (standard IR constant), rank_sim = rank by similarity_score,
rank_pop = rank by `vote_avg * log(vote_count+1)`.
Default alpha=0.5.

## Maximum Marginal Relevance (`--diversity`)
Optional diversity via `_mmr_diversify()`:
`MMR = lam * relevance - (1-lam) * max_sim_to_selected`
`--diversity` controls lam (0=relevance, 1=variety, default 0).

## Profile Cache
Scraped Letterboxd profiles are cached to `data/cache/{username}.json`.
Use `--no-cache` to force a fresh scrape.

## Structure
- `main.py` — CLI entry point (also `help` command)
- `collect.py` — TMDB collection runner
- `embed.py` — embedding pipeline runner
- `src/collector.py` — TMDB API fetcher (rate-limited, parallel credits, optional IMDb enrichment)
- `src/imdb_enricher.py` — IMDb API enrichment (batch + individual, rate-limited)
- `src/models.py` — `Movie` dataclass with `enriched_text()` and `metadata()`
- `src/vector_store.py` — ChromaDB client wrapper (cosine space)
- `src/indexer.py` — embeds and indexes into ChromaDB
- `src/embedders.py` — OllamaEmbedder backend (was GeminiEmbedder, removed)
- `src/profile.py` — Letterboxd export parser + TMDB resolver + stats
- `src/recommender.py` — taste vector, group consensus, text query, hybrid, MMR, scraper
- `src/explorer.py` — UMAP + HDBSCAN + Plotly + genre axes
- `web/app.py` — FastAPI web demo
- `tests/test_recommender.py` — 24 unit tests
- `data/chroma_db/` — persistent vector store (gitignored)
- `data/movies.jsonl` — raw TMDB movie data (gitignored)
- `data/cache/` — cached Letterboxd profiles (gitignored)

## Key Lessons Learned
1. **ChromaDB cosine distance**: Collection created with `metadata={"hnsw:space": "cosine"}`. Returned `distances` are `1 - cos_sim` (range [0, 2]). Use `similarity = 1.0 - dist`, NOT `1.0 / (1.0 + dist)` which was for L2.
2. **RRF > z-score normalization**: RRF is distribution-agnostic and rank-based, avoiding noise from small candidate pools or skewed distributions.
3. **nomic-embed-text** (274M params) works well for semantic movie matching — the quality issues stemmed from incorrect distance conversion and aggressive popularity reranking, not the model.
4. **Group consensus > mean vector**: Averaging taste vectors and finding nearest neighbours dilutes individual preferences, especially when tastes diverge. Computing per-user similarity and penalising std.dev finds movies that genuinely satisfy everyone.
5. **Taste weights should be rating-only**: Using inverse popularity in clustering weights gives niche movies disproportionate influence. Keep weights pure (rating) and handle popularity in reranking only.
6. **Taste similarity must be bidirectional + top-k**: Single-direction max-nearest-neighbor inflates scores when one user has a much larger collection. Compute both directions with top-3 mean.
7. **Recency bonus in scraper distorts ratings**: Never multiply ratings by recency factors — weights should be the raw rating (0.1-1.0). Use recency only in clustering weights if needed, not in the rating itself.
8. **Letterboxd URI → data-tmdb-id**: Faster and more reliable than TMDB search for resolving TMDB IDs. Extract slug from `Letterboxd URI` column in CSV exports and scrape the page for `data-tmdb-id` attribute. Use parallel HTTP with ThreadPoolExecutor.
9. **Letterboxd CSV exports include URI column**: Use `f.uri` directly instead of searching TMDB by title+year (which requires fuzzy matching and is rate-limited).
10. **Single source scraper**: `/films/page/N/` is the only source needed — it shows ALL watched films with their current ratings. No need for diary or ratings pages.
11. **Slug resolution must be rate-limited**: Parallel `ThreadPoolExecutor` for TMDB ID resolution needs a `Semaphore(5)` to avoid getting blocked by Letterboxd.
12. **Don't use `logger.info()` for user output**: CLI output should use `print()` — the timestamp prefix from logging breaks table formatting. Reserve logging for internal events.
13. **Sigma calibration via CV**: `_calibrate_sigma_cv()` uses k-fold cross-validation to find optimal sigma per user. Fallback to 0.3 if <10 movies. The `--auto-sigma` flag enables this, overriding a fixed `--sigma`.
14. **Multi-seed discovery > single centroid**: Querying ChromaDB from each user centroid separately (+ random seeds + query vector) finds candidates from ALL taste regions instead of averaging centroids into a meaningless middle point. Dramatically improves result quality.
15. **Numpy batch scoring**: `_predict_rating_batch()` vectorizes kernel regression across M candidates × N history entries in one matrix multiplication. ~60x faster than per-candidate Python loops.

## Rate Limits
- TMDB: keep 0.25s delay between calls (via `_tmdb_get` in collector)
- Letterboxd scraper: 1s delay between `/films/` page fetches, Semaphore(5) on slug→TMDB resolution
