# Plan de Mejora — BoxData (v2)

Este plan detalla las mejoras a implementar ordenadas por ROI.
Referencia: [`embedding-upgrade-plan.md`](./embedding-upgrade-plan.md) (mejora de embeddings vía LLM).

---

## Quick Wins Aplicados (Jul 2026)

| Cambio | Archivo | Descripción |
|--------|---------|-------------|
| Test `test_best_vote_data_prefers_imdb` corregido | `tests/test_recommender.py:326` | Renombrado a `test_tmdb_vote_data_uses_tmdb_only`, aserciones corregidas |
| `_best_vote_data` renombrado a `_tmdb_vote_data` | `src/recommender.py:868` | Refleja que solo usa TMDB (consistencia con el docstring) |
| Resume bug off-by-one | `src/collect_runner.py:53-58` | `start = last_year + 1` en vez de `start = last_year` |
| HTTP libs unificadas: curl-cffi everywhere | `src/collector.py`, `src/imdbapi_enricher.py`, `src/profile.py` | `requests` → `curl_cffi.requests` con `CurlError` handling |
| Letterboxd scraper con impersonate | `src/profile.py:50-57` | `_lb_session()` usa `Session(impersonate="safari17_0")` como recommender.py |
| Review fetcher creado | `src/review_fetcher.py` | `fetch_top_reviews()` + `enrich_movie_with_reviews()` |
| Error handling mejorado | `src/recommender.py`, `src/profile.py`, `src/imdbapi_enricher.py` | `except Exception: pass` → logging con contexto |
| `taste_similarity()` optimizado | `src/recommender.py:573-574` | Eliminadas 2N llamadas individuales redundantes a ChromaDB |
| `review_fetcher.py` integrado en collector | `src/collector.py` | Flag `--enrich-reviews` (on por defecto, ~0.4s/peli vía TMDB) |
| `filter` + `llm-text` commands | `src/filter_runner.py`, `src/llm_text_generator.py`, `main.py` | Pipeline previo a la Fase 3 (LLM embeddings) |
| Feature reranking implementado | `src/recommender.py:_feature_rerank` | Bonus por director/década/actores (Jaccard), cap 10%, flag `--feature-rerank` |
| TextGenerator abstracto | `src/llm_text_generator.py` | Parámetro `provider` soporta `"ollama"` (local) y `"openai"` (cloud, compatible OpenAI) — swap sin reescribir |

---

## Fase 1: Evaluation Framework ✅ (Jul 2026, implementado)

**Objetivo**: poder medir objetivamente si un cambio (nuevos embeddings, reranking, etc.) mejora o empeora las recomendaciones.

### Estado actual

Implementado en `src/evaluator.py` + comando CLI `evaluate` en `main.py`. Tests en `tests/test_evaluator.py` (15 tests, todos pasando).

### Cómo se usa

```bash
# Evaluar usuarios de Letterboxd (scrapea /films/)
python main.py evaluate --users alice bob

# Evaluar desde export CSV
python main.py evaluate --export ~/Downloads/letterboxd-export-2026/

# Guardar resultados
python main.py evaluate --users alice --output results.json
```

### Qué mide

| Métrica | Descripción |
|---------|-------------|
| **MAE** | Error absoluto medio — más bajo = mejor |
| **RMSE** | Root mean squared error — penaliza errores grandes |
| **Precision@5** | % de top-5 con rating ≥ 4.0 |
| **nDCG@5** | Ranking quality: orden + magnitud |
| **MRR** | Reciprocal rank de primera peli relevante |

### Implementación

- `evaluate_user(weighted_ids, n_folds=5, sigma=0.3, max_distance=-1.0, random_seed=42)` → dict con métricas + std por fold
- `evaluate_users(user_weighted_list, usernames=...)` → lista de resultados
- `_compute_metrics(y_true, y_pred)` → MAE, RMSE, P@5, nDCG@5, MRR
- `_print_evaluation_results(results)` → tabla formateada
- `_average_results(results)` → macro-average entre usuarios
- K-fold CV con Nadaraya-Watson (`_predict_rating_batch`)
- LOO fallback para <10 pelis
- Random shuffle con seed fijo (42) para reproducibilidad
- Reporte de std entre folds
- Tabla de resultados con average global

### Baseline — nomic-embed-text (20 Jul 2026)

Primera ejecución con 4 usuarios reales Letterboxd. Mide rendimiento actual con embeddings por defecto (Ollama nomic-embed-text, 768-dim, texto enriched_text() de `models.py`).

```
User                 #  Folds      MAE    MAEσ     RMSE   RMSEσ     P@5   nDCG@5     MRR
fabiomode2         141      6   0.1122  0.0213   0.1318  0.0172  0.8000   0.8406  1.0000
narrowd            382      6   0.1164  0.0196   0.1480  0.0297  0.4333   0.7716  0.4583
efrx               212      6   0.0964  0.0064   0.1174  0.0106  0.8833   0.9225  1.0000
georgemcnuggets    428      6   0.1751  0.0272   0.2105  0.0352  0.6445   0.7781  0.7222
AVERAGE            291      6   0.1250  0.0186   0.1519  0.0232  0.6903   0.8282  0.7951
```

| Métrica | Valor | Significado |
|---------|-------|-------------|
| MAE | 0.1250 | ~0.625★ error medio en escala 5★ |
| RMSE | 0.1519 | Pocos outliers (cercano a MAE) |
| Precision@5 | 69.0% | 7/10 top-5 tienen rating real ≥ 4.0★ |
| nDCG@5 | 0.828 | Ranking ordena bien relevantes arriba |
| MRR | 0.795 | Primera relevante suele estar en top-1 o top-2 |

Notas:
- ~85-88% de pelis del usuario presentes en ChromaDB → ~12-15% sin embedding
- narrowd MRR bajo (0.46) sugiere gusto heterogéneo o baja cobertura en ChromaDB
- georgemcnuggets MAE alto (0.175) y std alta — dataset grande pero ruidoso
- efrx mejor precision (0.88): gusto más homogéneo o mejor cobertura

Este es el baseline para comparar Fase 3 (LLM embeddings). Si MAE baja de 0.1250, los nuevos embeddings mejoran las predicciones.

---

## Fase 2: Feature Reranking ✅ (Jul 2026, ya implementado)

**Objetivo**: mejorar la calidad percibida de las recomendaciones inyectando conocimiento explícito (mismo director, misma década, mismos actores) sin cambiar los embeddings.

### Estado actual

Implementado en `src/recommender.py:_feature_rerank`. Integrado en `_recommend_pipeline` entre RRF y MMR. Flag `--feature-rerank`/`--no-feature-rerank` (ON por defecto).

**Resultado**: bonus compuesto por director (+10%), década (+5%), actores (Jaccard × 5%), cap 10% total.

---

## Fase 3: Embeddings con LLM (~finde)

Ver plan detallado en [`embedding-upgrade-plan.md`](./embedding-upgrade-plan.md).

### Opciones de generación de texto

| Opción | Modelo | Tiempo ~4,800 pelis | Coste | Calidad texto |
|--------|--------|--------------------|-------|---------------|
| A | llama3.2:3b (local, RTX 2050) | 13-30h | $0 | Baja (modelo 3B) |
| B | phi3:mini (local, RTX 2050) | 8-20h | $0 | Media-baja |
| C | **DeepSeek V4 Flash** (OpenCode Go) | ~30-60 min | $10/mes (incluido) | Alta (modelo grande) |

OpenCode Go incluye ~31,650 peticiones cada 5h con DeepSeek V4 Flash, más que suficiente para 4,800 pelis.

### Estrategia de comparación

1. Seleccionar 100 pelis de prueba representativas (variadas en géneros, décadas, popularidad)
2. Generar `llm_text` con modelo local (phi3:mini o llama3.2:3b) para esas 100
3. Generar `llm_text` con DeepSeek V4 Flash para las mismas 100
4. Embedear ambos conjuntos (`python main.py embed --input data/movies_llm_test.jsonl`)
5. Evaluar con `python main.py evaluate --users alice bob` (Fase 1)
6. Comparar MAE, nDCG@k, MRR → elegir ganador
7. Lanzar batch completo con la opción ganadora

### TextGenerator abstracto (ya implementado)

`src/llm_text_generator.py` soporta múltiples proveedores vía parámetro `provider`:
- `provider="ollama"`: modelo local (llama3.2:3b, phi3:mini, etc.)
- `provider="openai"`: cualquier API compatible OpenAI (DeepSeek V4 Flash vía OpenCode Go, OpenAI, etc.)

El swap entre local/cloud no requiere cambios de código — solo parámetros CLI. Si OpenCode Go cambia precio/disponibilidad, se puede migrar a otro proveedor OpenAI-compatible sin reescribir.

### Modificaciones respecto al plan original

- ❌ **Sin keywords**: no justifican el tiempo de API (~20 min) ni los tokens extra
- ✅ **Incluir `top_reviews` en el prompt del LLM** — `src/review_fetcher.py` ya implementa el fetching
- ✅ **Ejecutar evaluation framework antes/después** para validar la mejora
- ✅ **Cast en metadata** ya disponible (review_fetcher no modifica cast, collector ya lo guarda)

### Tiempo estimado (realista)

Con llama3.2:3b en RTX 2050 (4GB VRAM):

| Escenario | Tiempo por peli | Total ~4,800 pelis |
|-----------|----------------|-------------------|
| Optimista | 5-7s | 7-9h |
| Realista | 10-15s | **13-20h** |

**Recomendación**: probar con 50 pelis primero y extrapolar antes de lanzar el batch completo.

---

## Fase 4: Collaborative Filtering + Hybrid (futuro)

**Trigger para activar**: 50+ usuarios con 100+ ratings cada uno en ChromaDB.
Librería recomendada: [`implicit`](https://github.com/benfred/implicit) (ALS para matrices sparse).

**Idea**: combinar content-based (embeddings actuales) con collaborative filtering (matriz usuario-peli). Hybrid = ponderación ponderada de ambos scores.

**Prerequisito adicional**: benchmark de Fase 1 activo para medir si CF mejora o empeora.

---

## Cold Start (sin historial)

Para usuarios nuevos sin perfil Letterboxd, el sistema actual solo soporta `--query` (búsqueda semántica pura). Estrategia mínima:

1. **Solo query text**: ya funciona (`recommend --query "cyberpunk thriller"`). Embed query + ChromaDB cosine search.
2. **Popularidad como fallback**: si `--query` no se usa, ofrecer top-N por `vote_count * vote_avg` (las más populares/bien valoradas).
3. **Onboarding opcional (futuro)**: pedir 5-10 pelis favoritas del usuario, crear embedding centroid, y usar como perfil sintético. No prioritario.

**Lo que NO hacemos**: no inventar perfiles genéricos ni recomendar al azar. Si no hay data, el sistema lo dice claramente.

---

## Plan de Testing

### Tests existentes (31 tests en `tests/test_recommender.py`)

Cubren: cosine similarity, RRF reranking, weighted k-means, enriched text, geometric mean consensus, sigma calibration, Nadaraya-Watson kernel regression, MMR diversity, year/genre filters, TMDB vote data.

### Tests a añadir

| Módulo | Tests | Prioridad |
|--------|-------|-----------|
| `src/evaluator.py` | `test_evaluate_user_synthetic`, `test_evaluate_user_small_fallback`, `test_metrics_ndcg`, `test_metrics_mrr` | Alta (Fase 1) |
| `src/llm_text_generator.py` | `test_build_prompt_all_fields`, `test_build_prompt_missing_fields`, `test_build_prompt_no_cast` | Media |
| `src/recommender.py:_feature_rerank` | `test_feature_rerank_director_bonus`, `test_feature_rerank_decade_bonus`, `test_feature_rerank_actor_jaccard`, `test_feature_rerank_cap_10pct` (requiere mock de `get_collection`) | Media |

### Estrategia de testing

- **Unit tests puros**: funciones sin dependencias externas (ej. `_build_prompt`, `_compute_feature_bonus` si se extrae)
- **Mock de ChromaDB**: para `_feature_rerank` y `evaluator`, usar `unittest.mock.patch` para mockear `get_collection()`
- **Integration tests opcionales**: para `run_llm_text` con `provider="ollama"` si Ollama está corriendo en CI (skipped si no disponible)
- **CI**: GitHub Actions con pytest, skipped tests que requieran servicios externos

---

## Resumen del plan

| Fase | Qué | Esfuerzo | Impacto | Estado | Depende de |
|------|-----|----------|---------|--------|------------|
| **1** | Evaluation framework | ~2 días | Muy alto (desbloquea métricas) | ✅ Implementado (Jul 2026) | Nada |
| **2** | Feature reranking | ~2h | Alto (perceptible) | ✅ Completado (Jul 2026) | Nada |
| **3** | Embeddings con LLM | ~finde | Medio-Alto | Pendiente | Fase 1 (para medir mejora) |
| **4** | CF + Hybrid | ~1 semana | Alto | Pendiente | Fase 1 + 50 usuarios con 100+ ratings |

### Orden de ejecución recomendado

```
Semana 1:  Fase 1 (evaluation framework + tests)
           Validar con datos reales (métricas baseline)
           Test cualitativo de 50 pelis (Fase 3 preview)

Semana 2:  Fase 3 (embeddings con LLM, validado con Fase 1)
           Comparar local vs cloud (100 pelis test)
           Decidir si Fase 4 merece la pena (contar usuarios)
```
