# Optimización de Rendimiento — BoxData API

## Línea base (3 usuarios, n=12)

| Fase | Tiempo | % |
|------|--------|---|
| fetch Letterboxd (cacheados) | ~0.00s | 0% |
| get_taste_vector (por usuario) | ~0.02-0.08s | <1% |
| load user histories (ChromaDB) | ~0.10-0.15s | ~1% |
| sigma calibration (CV) | ~0.90-0.99s | ~6% |
| **discovery seeds + ChromaDB query** | **~9.9s** | **~62%** |
| batch scoring + consensus | ~0.02-0.09s | <1% |
| rerank + MMR | ~0.00s | 0% |
| **TOTAL** | **~14-16s** | |

## Bottleneck principal: discovery seeds

El pipeline genera semillas de búsqueda usando `get_taste_clusters()` (2-4 centroides por usuario, vía k-means ponderado) + 5 vectores aleatorios. Cada semilla lanza una query a ChromaDB pidiendo ~40-60 resultados. Con 3 usuarios: ~12-16 queries simultáneas.

### Cambios aplicados (round 1: conservador)

| Parámetro | Antes | Después |
|-----------|-------|---------|
| Random seeds | 5 | 4 |
| `pool_per_seed` | `max(n*5, 15)` | `min(max(n*5, 15), 40)` |

**Resultado**: Sin mejora apreciable — el bottleneck no está en cantidad de resultados por semilla sino en cantidad de semillas.

## Cambios propuestos (round 2)

### 1. Un centroide por usuario (en vez de 2-4)

Se reemplaza `get_taste_clusters()` (k-means, 2-4 clusters) por `get_taste_vector()` (promedio ponderado simple). Esto reduce las semillas de ~12-16 a ~6 (3 usuarios + 3 random aditivos).

Además, el centroide se calcula desde el historial ya cargado en memoria (`user_histories`), eliminando una llamada redundante a ChromaDB.

**Ahorro estimado**: ~6-7s | **Pérdida de calidad**: baja — MMR + random seeds mantienen diversidad

### 2. Sigma cacheado por usuario

Sigma se calibra vía k-fold CV sobre el historial. El resultado es estable para un mismo conjunto de embeddings (no cambia entre requests). Se cachea en un dict global keyeado por `frozenset(watched_ids.items())`.

**Ahorro estimado**: ~0.9s | **Pérdida de calidad**: 0%

## Proyección post-optimización

| Fase | Antes | Después |
|------|-------|---------|
| fetch Letterboxd | 0.00s | 0.00s |
| get_taste_vector | 0.08s | 0.08s |
| load user histories | 0.15s | 0.15s |
| sigma calibration | 0.99s | 0.00s (cache) |
| discovery seeds + query | 9.90s | ~3-4s |
| batch scoring | 0.09s | 0.09s |
| rerank + MMR | 0.00s | 0.00s |
| **TOTAL** | **~15s** | **~4-5s** |
