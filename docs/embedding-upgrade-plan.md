# Plan de Mejora de Embeddings

## Objetivo

Generar embeddings de mayor calidad para las ~4,800 películas más relevantes,
usando textos descriptivos detallados producidos por un LLM local (llama3.2:3b).

## Pipeline

```
movies.jsonl (67k)
    │  filter (rating ≥ 5.0, vote_count ≥ 1000)
    ▼
movies_filtered.jsonl (~4,800)
    │  llm-text (LLM genera texto ~300 palabras)
    ▼
movies_llm_text.jsonl
    │  embed (usa el nuevo texto en vez del enriched_text())
    ▼
ChromaDB (embeddings mejorados)
```

## Estrategia de generación de texto (LLM)

### Opción A: Modelo local (llama3.2:3b en RTX 2050 4GB)

- Tiempo estimado: 13-30h para ~4,800 pelis (según velocidad real de generación)
- Ventaja: sin coste adicional, sin límites de rate
- Desventaja: lento, calidad del texto limitada por modelo pequeño

### Opción B: DeepSeek V4 Flash vía OpenCode Go

OpenCode Go ($10/mes) da acceso a DeepSeek V4 Flash con ~31,650 peticiones cada 5h (~158k/mes).

| Parámetro | Valor |
|-----------|-------|
| Peticiones disponibles | ~31,650 / 5h, ~79k / semana, ~158k / mes |
| Coste | $10/mes (incluido en suscripción Go) |
| Velocidad | Varios cientos de requests/min (API cloud) |
| Endpoint | `https://opencode.ai/zen/go/v1/chat/completions` |
| Model ID | `deepseek-v4-flash` (usar `opencode-go/deepseek-v4-flash` en OpenCode) |

Para 4,800 pelis se necesitan 4,800 peticiones — muy por debajo del límite. El texto generado por DeepSeek V4 Flash será de mucha mayor calidad que el de un modelo local de 3B parámetros.

### Estrategia de comparación

1. Generar texto con **modelo local** (llama3.2:3b o phi3:mini) para 100 pelis de prueba
2. Generar texto con **DeepSeek V4 Flash** para las mismas 100 pelis
3. Embedear ambos conjuntos y medir con el evaluation framework (Fase 2):
   - **MAE** de predicción de ratings
   - **Precision@k** en ranking
4. Elegir la opción con mejor relación calidad/coste/tiempo
5. Lanzar el batch completo con la opción ganadora

### Uso de la API de OpenCode Go desde Python

```python
import requests

def generate_movie_text(movie_data: dict, api_key: str) -> str:
    prompt = f"""Generate a detailed 300-word description of this movie...
    Title: {movie_data['title']}
    Year: {movie_data['year']}
    ..."""
    
    resp = requests.post(
        "https://opencode.ai/zen/go/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 600,
        },
        timeout=60,
    )
    return resp.json()["choices"][0]["message"]["content"]
```

## Comandos nuevos en `main.py`

### 1. `filter` — Filtrar películas

```
python main.py filter [--min-rating 5.0] [--min-votes 1000]
```

- Lee `data/movies.jsonl`
- Filtra por `vote_average >= min_rating` y `vote_count >= min_votes`
- Escribe `data/movies_filtered.jsonl`
- Muestra estadísticas (cuántas pasan, cuántas se descartan)

### 2. `llm-text` — Generar texto descriptivo con LLM

```
python main.py llm-text [--model llama3.2:3b] [--batch 1]
```

- Lee `data/movies_filtered.jsonl`
- Para cada película, arma un prompt con todos los datos estructurados
- Llama a Ollama (llama3.2:3b) para generar texto descriptivo de ~300 palabras
- Guarda el texto generado en el campo `llm_text`
- Escribe `data/movies_llm_text.jsonl`
- Resumible (procesa solo las que no tienen `llm_text` aún)

### 3. `embed` (modificado) — Aceptar `--input`

```
python main.py embed [--input data/movies_llm_text.jsonl]
```

- Si `--input` se especifica, usa ese archivo en vez de `movies.jsonl`
- Cuando detecta campo `llm_text`, lo usa en vez de `enriched_text()`
- El resto del pipeline (ChromaDB, metadatos) igual

## Prompt para el LLM

```text
Generate a detailed 300-word description of this movie using the information below.
Write in natural, flowing English prose — not bullet points. Cover:
1. Basic info (title, year, director, main cast)
2. Genre, style, and tone
3. Plot summary (based on the overview)
4. Themes and what makes it distinctive
5. Critical reception and cultural impact

Title: {title}
Year: {year}
Director: {director}
Cast: {cast}
Genres: {genres}
Overview: {overview}
Rating: {rating}/10 ({votes} votes)
```

## Detalles técnicos

### Filter
- Módulo nuevo: `src/filter_runner.py`
- Sin dependencias externas nuevas
- Filtrado en una pasada

### LLM-Text
- Módulo nuevo: `src/llm_text_generator.py`
- Usa `ollama` (ya instalado) con `llama3.2:3b`
- ~4,800 pelis × 5-7s = ~7-9 horas
- **Resumible**: checkpoint por línea — si una línea ya tiene `llm_text`, se salta
- Sin paralelismo (Ollama maneja una request a la vez con el modelo en GPU)

### Embed (modificado)
- `embed_runner.py`: añadir flag `--input`
- `indexer.py`: aceptar `use_llm_text=True` para usar `llm_text` en vez de `enriched_text()`
- `models.py`: añadir campo opcional `llm_text: str | None = None`
- Al subir a ChromaDB, `documents` usa `llm_text` si existe, sino `enriched_text()`

## Hardware

- CPU: 12 cores
- RAM: 14 GB (2.5 GB libre)
- GPU: NVIDIA RTX 2050 (4 GB VRAM)
- Modelo: llama3.2:3b (~2.5 GB VRAM, cabe sobrado + contexto)
- Tiempo estimado total LLM: ~7-9 horas

## Orden de ejecución en el VPS

Tras implementar y testear localmente con una muestra:

```bash
# 1. Activar venv
source .venv/bin/activate

# 2. Asegurar que el modelo está descargado
ollama pull llama3.2:3b

# 3. Filtrar
python main.py filter

# 4. Generar texto (~8h, dejar overnight)
nohup python main.py llm-text > llm_text.log 2>&1 &

# 5. Respaldar DB actual (por si acaso)
cp -r data/chroma_db data/chroma_db_backup

# 6. Limpiar y re-embedear
python main.py clear-db
python main.py embed --input data/movies_llm_text.jsonl
```
