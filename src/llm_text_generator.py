"""Generate detailed LLM text descriptions for movies (for richer embeddings)."""
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """Generate a detailed 300-word description of this movie using the information below.
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
Rating: {rating}/10 ({votes} votes)"""


def _build_prompt(d: dict) -> str:
    cast = ", ".join(d.get("cast", [])[:5]) if d.get("cast") else ""
    return PROMPT_TEMPLATE.format(
        title=d.get("title", "Unknown"),
        year=d.get("release_date", "0000")[:4] if d.get("release_date") else "?",
        director=d.get("director", "Unknown"),
        cast=cast or "Unknown",
        genres=", ".join(d.get("genres", [])),
        overview=d.get("overview", ""),
        rating=d.get("vote_average", "?"),
        votes=d.get("vote_count", 0),
    )


def _generate_with_ollama(client, model: str, prompt: str, api_key: str | None = None) -> str | None:
    """Generate text via Ollama (local)."""
    try:
        resp = client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp["message"]["content"].strip()
    except Exception as e:
        logger.warning("Ollama generation failed: %s", e)
        return None


def _generate_with_openai(prompt: str, api_base: str, api_key: str, model: str) -> str | None:
    """Generate text via OpenAI-compatible API (e.g. DeepSeek V4 Flash via OpenCode Go)."""
    from curl_cffi import requests, CurlError
    try:
        resp = requests.post(
            api_base,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 600,
            },
            timeout=90,
        )
        if resp.status_code != 200:
            logger.warning("API returned %d: %s", resp.status_code, resp.text[:200])
            return None
        return resp.json()["choices"][0]["message"]["content"].strip()
    except CurlError as e:
        logger.warning("API generation failed: %s", e)
        return None


def run_llm_text(
    input_file: str | None = None,
    output_file: str | None = None,
    model: str = "llama3.2:3b",
    provider: str = "ollama",
    api_base: str | None = None,
    api_key: str | None = None,
    ollama_host: str = "http://localhost:11434",
    delay: float = 0.5,
) -> dict[str, Any]:
    """Generate LLM text descriptions for movies.

    Reads a JSONL file, generates a detailed prose description for each
    movie (skipping those that already have `llm_text` for resumability),
    and writes a new JSONL with the enriched data.

    provider="ollama" uses local Ollama.
    provider="openai" uses any OpenAI-compatible API (requires api_base + api_key).
    """
    in_path = Path(input_file) if input_file else Path("data/movies_filtered.jsonl")
    out_path = Path(output_file) if output_file else Path("data/movies_llm_text.jsonl")

    if not in_path.exists():
        logger.error("%s not found. Run filter first.", in_path)
        return {"error": f"File not found: {in_path}", "generated": 0}

    movies = []
    with open(in_path) as f:
        for line in f:
            movies.append(json.loads(line))

    total = len(movies)
    already_done = sum(1 for m in movies if m.get("llm_text"))
    to_generate = [m for m in movies if not m.get("llm_text")]
    logger.info(
        "Loaded %d movies: %d already have llm_text, %d to generate",
        total, already_done, len(to_generate),
    )

    if not to_generate:
        logger.info("Nothing to do.")
        return {"input": str(in_path), "output": str(out_path), "total": total, "generated": 0}

    client = None
    if provider == "ollama":
        import ollama
        client = ollama.Client(host=ollama_host)

    generated = 0
    start_time = time.perf_counter()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fout:
        for movie in movies:
            if movie.get("llm_text"):
                fout.write(json.dumps(movie, ensure_ascii=False) + "\n")
                continue

            prompt = _build_prompt(movie)
            text = None

            if provider == "ollama":
                text = _generate_with_ollama(client, model, prompt)
            elif provider == "openai":
                if not api_base or not api_key:
                    logger.error("api_base and api_key required for openai provider")
                    return {"error": "api_base and api_key required", "generated": generated}
                text = _generate_with_openai(prompt, api_base, api_key, model)

            if text:
                movie["llm_text"] = text
                generated += 1
                if generated % 10 == 0:
                    elapsed = time.perf_counter() - start_time
                    rate = generated / elapsed if elapsed > 0 else 0
                    logger.info(
                        "  %d/%d generated (%.1f/min)",
                        generated, len(to_generate), rate * 60,
                    )
            else:
                logger.warning("  Failed to generate for '%s', keeping original", movie.get("title"))

            fout.write(json.dumps(movie, ensure_ascii=False) + "\n")
            fout.flush()

            if delay > 0:
                time.sleep(delay)

    elapsed = time.perf_counter() - start_time
    logger.info(
        "Done. %d generated in %.1f min (%.1f s/movie)",
        generated, elapsed / 60, elapsed / max(generated, 1),
    )
    return {
        "input": str(in_path),
        "output": str(out_path),
        "total": total,
        "generated": generated,
        "skipped": already_done,
        "elapsed_min": round(elapsed / 60, 1),
    }
