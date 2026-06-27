# Roadmap — BoxData

## Estado actual: ✅ (Jun 2026)

## Bugs corregidos (Jun 2026)

### 🔴 Crítico

| Bug | Archivo | Solución |
|-----|---------|----------|
| Logger no definido en indexer | `src/indexer.py:50` | Añadido `import logging` + `logger = logging.getLogger(__name__)` |
| ollama no estaba en dependencias | `pyproject.toml` | Añadido `ollama>=0.1` |
| MMR diversity roto para `--query` | `src/recommender.py:734` | `_emb=[]` → se almacena el embedding real de ChromaDB |
| Scraper solo miraba diary+ratings (perdía films sin fecha/rating) | `src/recommender.py` | Migrado a `/{username}/films/page/N/` como fuente única |

### 🟡 Bugs funcionales

| Bug | Archivo | Solución |
|-----|---------|----------|
| `_export_to_weighted` ignoraba films sin rating | `main.py:236` | Incluidos con weight=0.6 (misma lógica que scraper) |
| `_resolve_via_tmdb` sin rate limiting (TMDB 429s) | `src/profile.py:93` | Migrado a usar `_tmdb_get()` que tiene rate limiter |
| `_resolve_slug_parallel` sin rate limiting | `src/recommender.py:88` | Añadido `Semaphore(5)` para máximo 5 concurrentes |
| `_print_recommendations` usaba `logger.info()` | `main.py:227` | Cambiado a `print()` para evitar timestamps en la tabla |

### 🧹 Dead code eliminado

| Archivo | Código | Razón |
|---------|--------|-------|
| `src/recommender.py` | `_slug_to_tmdb()` | Reemplazado por `_resolve_slug_parallel()` |
| `main.py` | `--embedder` argument | No se usaba en runtime |
| `src/config.py` | `API_KEY` | No se usaba (solo `API_TOKEN`) |
| `src/profile.py` | `from src.collector import _tmdb_get` | Import muerto (se re-añadió con uso real) |

### 📚 Documentación actualizada

| Archivo | Cambio |
|---------|--------|
| `AGENTS.md` | Comando `--user` ahora dice "scrapes `/films/page/N/`" |
| `AGENTS.md` | Rate limits actualizados (Semaphore(5) + sin diary/ratings) |
| `AGENTS.md` | Añadidas lecciones 10-12 (scraper single source, rate limiting, print vs logger) |
| `pyproject.toml` | Descripción: "Gemini 2" → "Ollama" |
| `roadmap.md` | Este documento |

## Bugs conocidos (pendientes)

- **Baja prioridad**: `consensus_penalty` no expuesto vía CLI (hardcoded 0.5)
- **Baja prioridad**: tests solo cubren `_weighted_kmeans` — el resto sin testear
- **Baja prioridad**: `_build_chromadb_index()` carga todo ChromaDB en memoria cada vez

## Historial

### ✅ Scraper reescrito (Jun 2026)
- Fuente única: `/{username}/films/page/N/` (antes diary + ratings)
- Resolución de slugs paralela con `ThreadPoolExecutor`
- Rate limiting con `Semaphore(5)` en slug resolution
- Caché persistente en `data/cache/`

### ✅ Bugs anteriores (May 2026)
- Scraper: recency bonus distorsionaba ratings
- Taste similarity: uni-direccional max-NN → bidireccional top-3 mean
- Resolución TMDB: serial → paralela vía URI + TMDB
