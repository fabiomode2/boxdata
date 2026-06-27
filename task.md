# Task Log — Sesión 25 Jun 2026

## Implementado

### 1. Weighting separado en taste clusters
- **`get_taste_clusters`** y **`get_taste_vector`**: eliminado `inv_pop = 1/ln(vote_count + 10)`.
- El peso ahora es solo `user_weight` (rating puro). La popularidad se maneja exclusivamente en `_rerank_candidates`.
- **Archivo**: `src/recommender.py`

### 2. RRF reemplaza z-scores
- **`_rerank_candidates`**: usa Reciprocal Rank Fusion en vez de z-score normalization.
- `score = alpha / (k + rank_sim) + (1-alpha) / (k + rank_pop)` con k=60.
- **Archivo**: `src/recommender.py`

### 3. Fallback para historiales cortos
- `< 5` películas → un solo centroide (media ponderada).
- `5-14` películas → cada película es su propio cluster (nearest-neighbor scoring).
- `>= 15` → clustering k-means normal.
- **Archivo**: `src/recommender.py`

### 4. Tests unitarios
- 20 tests: cosine_similarity (5), RRF (4), k-means (3), enriched_text (5), consensus scoring (1), alpha blend (1), short history (1).
- **Dependencia**: `pytest>=8.0`
- **Archivos**: `tests/test_recommender.py`

### 5. Gemini/Google eliminado
- `GeminiEmbedder` eliminado de `src/embedders.py`.
- Default embedder en `main.py` cambiado a ollama.
- `GEMINI_API_KEY` eliminado de `src/config.py`.
- `google-genai` eliminado de `pyproject.toml`.
- Branching gemini/ollama simplificado en `src/indexer.py` y `embed.py`.
- **Archivos**: `src/embedders.py`, `main.py`, `src/config.py`, `pyproject.toml`, `src/indexer.py`, `embed.py`

### 6. Logging
- `print()` reemplazado con `logging` en todos los módulos.
- `logging.basicConfig(level=logging.INFO, ...)` en `main.py`.
- **Archivos**: `main.py`, `src/*.py`, `collect.py`, `embed.py`

### 7. Scraper migrado a BeautifulSoup
- `_extract_entries`, `_extract_rating_entries`, `_slug_to_tmdb`: regex → selectores CSS.
- **Dependencia**: `beautifulsoup4>=4.12`
- **Archivo**: `src/recommender.py`

### 8. MMR diversity
- `_mmr_diversify()` implementado: `MMR = lam * relevance - (1-lam) * max_sim_to_selected`.
- Flag `--diversity` (0-1, default 0) en `main.py`.
- Integrado en `recommend_for_group`, `recommend_hybrid`, `recommend_from_query`.
- **Archivos**: `src/recommender.py`, `main.py`

### 9. Dead code eliminado
- `_center()` y `_get_mean_embedding()` removidos de `src/recommender.py`.

### 10. Comando help
- `python main.py help` — vista general de todos los comandos y opciones.
- **Archivo**: `main.py`

### 11. Bugfix: scraper distorsionaba ratings con recency bonus
- **Problema**: El scraper multiplicaba ratings del diario por un factor de recencia (0.5-1.0), distorsionando el peso real. Además `seen_slugs` bloqueaba la sobreescritura desde la página de ratings (que tiene el rating limpio).
- **Solución**: Eliminado el recency bonus. Las entradas del diario ahora usan el rating puro. La página de ratings puede sobreescribir vía `max()`. Todos los ratings son ahora múltiplos limpios de 0.1.
- **Archivo**: `src/recommender.py:78-116`
- **Impacto**: Eyes Wide Shut de narrowd pasó de 0.53 a 1.0 (el rating real 5/5).

### 12. Mejora: taste similarity bidireccional + top-3
- **Problema**: `taste_similarity` usaba max-nearest-neighbor en una sola dirección. Con una colección grande (323 pelis), siempre encontraba algún match, inflando el score (84.8% para dos usuarios con gustos muy distintos).
- **Solución**: Ahora computa ambas direcciones (A→B y B→A) con media de top-3 vecinos, y promedia. La similitud real bajó de 84.8% a 72.0%.
- **Archivo**: `src/recommender.py:455-480`

### 13. Optimización: resolución paralela de TMDB IDs
- **Problema**: `resolve_tmdb_ids` usaba `_tmdb_get` (lock global, 0.25s entre llamadas). 198 pelis → ~50s.
- **Solución**: 
  1. Resuelve vía URI de Letterboxd (columna `Letterboxd URI` del CSV) → `data-tmdb-id` en paralelo (20 workers)
  2. Fallback a TMDB search en paralelo (10 workers, sin lock global)
- Tiempo estimado: de ~50s a ~3-5s para 198 pelis.
- **Archivo**: `src/profile.py`

### 14. Watchlist desactivada en analyze
- `resolve_watchlist=False` — ahorra tiempo resolviendo cientos de pelis para una estadística que apenas se usa.
- **Archivo**: `main.py:185`

## Datos de perfiles recogidos

| Usuario | Pelis scraped | En ChromaDB | Clusters |
|---------|---------------|-------------|----------|
| fabiomode | 0 | 0/0 | — | Username no existe o perfil privado |
| fabiomode2 | 106 | 84/106 | 4 | Mainstream: thrillers, acción, anime |
| narrowd | 233 | 194/233 | 4 | Arthouse: classic europeo, experimental, horror |

- **Taste similarity fabiomode2 ↔ narrowd**: 72.0% (bidi top-3 mean)
  - A→B: 73.6%, B→A: 70.4%
  - Overlap: 65 películas compartidas

## Dependencias actuales
- `requests>=2.31`, `curl-cffi>=0.15`, `python-dotenv>=1.0`, `tqdm>=4.66`
- `chromadb>=0.5.0`, `beautifulsoup4>=4.12`
- Dev: `pytest>=8.0`
- Eliminadas: `google-genai>=1.0.0`
