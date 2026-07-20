"""Generate detailed LLM text descriptions for movies (for richer embeddings)."""
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_OPENAI_API_BASE = "https://opencode.ai/zen/v1/chat/completions"
DEFAULT_OPENAI_MODEL = "mimo-v2.5-free"

PROMPT_TEMPLATE = """Write a compact 80-word description of this movie for a recommendation system.
Focus on factual plot, genre, style, and tone. Be specific — avoid clichés and fluff.

Title: {title} ({year})
Director: {director}
Cast: {cast}
Genres: {genres}
Overview: {overview}
Rating: {rating}/10 ({votes} votes)
{reviews}"""


def _build_prompt(d: dict) -> str:
    cast = ", ".join(d.get("cast", [])[:5]) if d.get("cast") else "Unknown"
    reviews = d.get("top_reviews", [])
    reviews_block = "Reviews: " + (" | ".join(reviews[:3])) if reviews else ""
    return PROMPT_TEMPLATE.format(
        title=d.get("title", "Unknown"),
        year=d.get("release_date", "0000")[:4] if d.get("release_date") else "?",
        director=d.get("director", "Unknown"),
        cast=cast,
        genres=", ".join(d.get("genres", [])),
        overview=d.get("overview", ""),
        rating=d.get("vote_average", "?"),
        votes=d.get("vote_count", 0),
        reviews=reviews_block,
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
                "max_tokens": 8192,
                "reasoning": {"enabled": False},
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
    workers: int = 5,
) -> dict[str, Any]:
    """Generate LLM text descriptions for movies.

    Reads a JSONL file, generates a detailed prose description for each
    movie (skipping those that already have `llm_text` for resumability),
    and writes a new JSONL with the enriched data.

    provider="ollama" uses local Ollama (sequential, GPU-bound).
    provider="openai" uses any OpenAI-compatible API (concurrent via ThreadPoolExecutor).
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

    # Defaults for openai provider
    if provider == "openai":
        if not api_base:
            api_base = DEFAULT_OPENAI_API_BASE
            logger.info("Using default api_base: %s", api_base)
        if not api_key:
            api_key = os.getenv("OPENCODE_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        if not api_key:
            logger.warning("OPENCODE_API_KEY not set. OpenCode Zen may reject unauthenticated requests.")

    generated = 0
    start_time = time.perf_counter()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _gen_one(movie: dict) -> dict | None:
        prompt = _build_prompt(movie)
        if provider == "ollama":
            import ollama
            client = ollama.Client(host=ollama_host)
            text = _generate_with_ollama(client, model, prompt)
        elif provider == "openai":
            if not api_base:
                raise ValueError("api_base required for openai provider")
            text = _generate_with_openai(prompt, api_base, api_key or "", model)
        else:
            text = None

        if text:
            movie["llm_text"] = text
            return movie
        logger.warning("  Failed to generate for '%s', keeping original", movie.get("title"))
        return movie

    if provider == "openai" and workers > 1:
        # concurrent: write incrementally with thread lock
        import threading
        _write_lock = threading.Lock()
        to_gen_indices = [i for i, m in enumerate(movies) if not m.get("llm_text")]

        def _concurrent_gen(idx: int) -> tuple[int, dict]:
            return idx, _gen_one(movies[idx])

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as fout:
            # pre-write already-done movies
            for i, m in enumerate(movies):
                if i not in to_gen_indices:
                    fout.write(json.dumps(m, ensure_ascii=False) + "\n")
            fout.flush()

            with ThreadPoolExecutor(max_workers=workers) as pool:
                fut_to_idx = {pool.submit(_concurrent_gen, i): i for i in to_gen_indices}
                done_count = 0
                for future in as_completed(fut_to_idx):
                    idx, result = future.result()
                    movies[idx] = result
                    done_count += 1
                    if result.get("llm_text"):
                        generated += 1
                    with _write_lock:
                        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                        fout.flush()
                    if done_count % 10 == 0:
                        elapsed = time.perf_counter() - start_time
                        rate = done_count / elapsed if elapsed > 0 else 0
                        logger.info(
                            "  %d/%d generated (%.1f/min)",
                            done_count, len(to_gen_indices), rate * 60,
                        )
    else:
        # sequential (ollama or single-worker openai)
        with open(out_path, "w") as fout:
            for movie in movies:
                if movie.get("llm_text"):
                    fout.write(json.dumps(movie, ensure_ascii=False) + "\n")
                    continue

                movie = _gen_one(movie)
                if movie.get("llm_text"):
                    generated += 1
                    if generated % 10 == 0:
                        elapsed = time.perf_counter() - start_time
                        rate = generated / elapsed if elapsed > 0 else 0
                        logger.info(
                            "  %d/%d generated (%.1f/min)",
                            generated, len(to_generate), rate * 60,
                        )

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
