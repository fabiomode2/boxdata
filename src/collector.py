import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

from curl_cffi import requests
from curl_cffi import CurlError

from src.config import TMDB_BASE, HEADERS
from src.models import Movie

logger = logging.getLogger(__name__)

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
            logger.warning("TMDB 429: retrying in %ds", retry)
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


def _fetch_movie_detail(tmdb_id: int) -> tuple[str | None, list[str], str | None]:
    """Fetch movie detail + credits via `_tmdb_get` (shared rate limiter)."""
    url = f"{TMDB_BASE}/movie/{tmdb_id}"
    try:
        data = _tmdb_get(url, {"append_to_response": "credits"})
    except CurlError:
        return None, [], None
    imdb_id = data.get("imdb_id")
    credits = data.get("credits") or {}
    director = None
    for crew in credits.get("crew", []):
        if crew["job"] == "Director":
            director = crew["name"]
            break
    cast_names = [c["name"] for c in credits.get("cast", [])[:5]]
    return director, cast_names, imdb_id


def _fetch_credits_batch(tmdb_ids: list[int]) -> dict[int, tuple[str | None, list[str], str | None]]:
    if not tmdb_ids:
        return {}
    results = {}
    with ThreadPoolExecutor(max_workers=_CREDIT_WORKERS) as pool:
        futures = {pool.submit(_fetch_movie_detail, tid): tid for tid in tmdb_ids}
        for future in as_completed(futures):
            tid = futures[future]
            try:
                results[tid] = future.result()
            except Exception as e:
                logger.debug("Credit fetch failed for TMDB %d: %s", tid, e)
                results[tid] = (None, [], None)
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
    enrich: bool = True,
    enrich_reviews: bool = True,
) -> list[Movie]:
    """Fetch top N movies for a year from TMDB, then batch-fetch credits + optional IMDb enrichment.

    Fetches 3x the target amount, then re-ranks by vote_count descending.
    TMDB's built-in "popularity" is a rolling black-box metric that can be
    inflated by recent controversy or engagement on a handful of votes, so
    we use raw vote count as the popularity signal instead.

    When enrich=True, also fetches IMDb rating/votes, metacritic, interests
    (subgenres), origin countries, and spoken languages via imdbapi.dev.
    When enrich_reviews=True, fetches top TMDB reviews for each movie
    (used for LLM-enhanced embeddings).
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
    imdb_id_map: dict[int, str] = {}
    for item in candidates:
        tmdb_id = item["id"]
        director, cast, imdb_id = (None, [], None) if skip_credits else credits_map.get(tmdb_id, (None, [], None))
        overview = item.get("overview", "") or ""
        movie = Movie(
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
            imdb_id=imdb_id,
        )
        movies.append(movie)
        if imdb_id:
            imdb_id_map[tmdb_id] = imdb_id

    if enrich and imdb_id_map and not skip_credits:
        _apply_imdb_enrichment(movies, imdb_id_map)

    if enrich_reviews and not skip_credits:
        _apply_review_enrichment(movies)

    return movies


def _apply_imdb_enrichment(movies: list[Movie], imdb_id_map: dict[int, str]) -> None:
    from src.imdbapi_enricher import extract_enrichment, fetch_enrichment

    tmdb_to_imdb = {v: k for k, v in imdb_id_map.items()}
    imdb_ids = list(imdb_id_map.values())
    logger.info("  Enriching %d movies from IMDb API...", len(imdb_ids))

    enriched = fetch_enrichment(imdb_ids)

    found = 0
    for movie in movies:
        imdb_id = imdb_id_map.get(movie.tmdb_id)
        if imdb_id and imdb_id in enriched:
            data = extract_enrichment(enriched[imdb_id])
            movie.imdb_rating = data["imdb_rating"]
            movie.imdb_vote_count = data["imdb_vote_count"]
            movie.metacritic_score = data["metacritic_score"]
            movie.metacritic_review_count = data["metacritic_review_count"]
            movie.interests = data["interests"]
            movie.origin_countries = data["origin_countries"]
            movie.spoken_languages = data["spoken_languages"]
            found += 1

    if found:
        logger.info("  IMDb enrichment: %d/%d movies enriched", found, len(movies))


def _apply_review_enrichment(movies: list[Movie]) -> None:
    """Fetch top reviews from TMDB for each movie in parallel."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from src.review_fetcher import fetch_top_reviews

    logger.info("  Fetching reviews for %d movies...", len(movies))

    movie_map = {m.tmdb_id: m for m in movies}
    found = 0

    with ThreadPoolExecutor(max_workers=_CREDIT_WORKERS) as pool:
        futures = {pool.submit(fetch_top_reviews, m.tmdb_id): m.tmdb_id for m in movies}
        for future in as_completed(futures):
            tid = futures[future]
            try:
                reviews = future.result()
                if reviews:
                    movie_map[tid].top_reviews = reviews
                    found += 1
            except Exception as e:
                logger.debug("Review fetch failed for TMDB %d: %s", tid, e)

    if found:
        logger.info("  Review enrichment: %d/%d movies enriched", found, len(movies))
