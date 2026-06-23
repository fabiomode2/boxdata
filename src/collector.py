import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from datetime import date

from src.config import TMDB_BASE, HEADERS
from src.models import Movie

GENRE_CACHE: dict[int, str] = {}
_TMDB_LOCK = threading.Lock()
_TMDB_LAST_CALL: float = 0.0
_TMDB_MIN_INTERVAL = 0.25
_CREDIT_WORKERS = 10


def _tmdb_get(url: str, params: dict | None = None) -> dict:
    global _TMDB_LAST_CALL
    while True:
        with _TMDB_LOCK:
            elapsed = time.time() - _TMDB_LAST_CALL
            if elapsed < _TMDB_MIN_INTERVAL:
                time.sleep(_TMDB_MIN_INTERVAL - elapsed)
            resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
            _TMDB_LAST_CALL = time.time()
        if resp.status_code == 429:
            retry = int(resp.headers.get("Retry-After", 5))
            print(f"  TMDB 429: retrying in {retry}s", flush=True)
            time.sleep(retry)
            continue
        resp.raise_for_status()
        return resp.json()


def _load_genres() -> dict[int, str]:
    if GENRE_CACHE:
        return GENRE_CACHE
    data = _tmdb_get(f"{TMDB_BASE}/genre/movie/list")
    for g in data.get("genres", []):
        GENRE_CACHE[g["id"]] = g["name"]
    return GENRE_CACHE


def _genre_names(genre_ids: list[int]) -> list[str]:
    genres = _load_genres()
    return [genres[gid] for gid in genre_ids if gid in genres]


def _fetch_credits_one(tmdb_id: int) -> tuple[str | None, list[str]]:
    """Fetch credits via `_tmdb_get` (shared rate limiter, no 429s)."""
    url = f"{TMDB_BASE}/movie/{tmdb_id}/credits"
    try:
        data = _tmdb_get(url)
    except requests.RequestException:
        return None, []
    director = None
    for crew in data.get("crew", []):
        if crew["job"] == "Director":
            director = crew["name"]
            break
    cast_names = [c["name"] for c in data.get("cast", [])[:5]]
    return director, cast_names


def _fetch_credits_batch(tmdb_ids: list[int]) -> dict[int, tuple[str | None, list[str]]]:
    if not tmdb_ids:
        return {}
    results = {}
    with ThreadPoolExecutor(max_workers=_CREDIT_WORKERS) as pool:
        futures = {pool.submit(_fetch_credits_one, tid): tid for tid in tmdb_ids}
        for future in as_completed(futures):
            tid = futures[future]
            try:
                results[tid] = future.result()
            except Exception:
                results[tid] = (None, [])
    return results


def _parse_date(d: str | None) -> date | None:
    if not d:
        return None
    try:
        return date.fromisoformat(d)
    except ValueError:
        return None


def fetch_movies_by_year(
    year: int,
    max_per_year: int = 100,
    skip_credits: bool = False,
    seen_ids: set[int] | None = None,
) -> list[Movie]:
    """Fetch top N movies for a year from TMDB, then batch-fetch credits in parallel.

    Fetches 3x the target amount, then re-ranks by vote_count descending.
    TMDB's built-in "popularity" is a rolling black-box metric that can be
    inflated by recent controversy or engagement on a handful of votes, so
    we use raw vote count as the popularity signal instead.
    """
    _load_genres()
    candidates: list[dict] = []
    page = 1

    # Oversample so we have a large enough pool to re-rank meaningfully
    oversample = max_per_year * 3

    while len(candidates) < oversample:
        data = _tmdb_get(f"{TMDB_BASE}/discover/movie", {
            "primary_release_year": year,
            "sort_by": "popularity.desc",
            "page": page,
            "include_adult": "false",
            "vote_count.gte": 10,
        })
        results = data.get("results", [])
        if not results:
            break

        for item in results:
            if len(candidates) >= oversample:
                break
            tmdb_id = item["id"]
            if seen_ids and tmdb_id in seen_ids:
                continue
            overview = item.get("overview", "") or ""
            if not overview.strip():
                continue
            candidates.append(item)

        page += 1
        if page > 500:
            break

    if not candidates:
        return []

    # Re-rank: vote_count dominates (popularity = how many people engaged),
    # but vote_average provides a small quality correction — movies with
    # avg < 2.5 get penalised (truly unwatchable), movies above get a
    # modest linear boost.
    candidates.sort(
        key=lambda m: m["vote_count"] * max(0.1, m["vote_average"] - 2.5),
        reverse=True,
    )
    candidates = candidates[:max_per_year]

    tmdb_ids = [c["id"] for c in candidates]
    credits_map = {} if skip_credits else _fetch_credits_batch(tmdb_ids)

    movies: list[Movie] = []
    for item in candidates:
        tmdb_id = item["id"]
        director, cast = (None, []) if skip_credits else credits_map.get(tmdb_id, (None, []))
        overview = item.get("overview", "") or ""
        movies.append(Movie(
            tmdb_id=tmdb_id,
            title=item["title"],
            overview=overview,
            release_date=_parse_date(item.get("release_date")),
            vote_average=item.get("vote_average", 0),
            vote_count=item.get("vote_count", 0),
            genres=_genre_names(item.get("genre_ids", [])),
            director=director,
            cast=cast,
            poster_path=item.get("poster_path"),
        ))

    return movies
