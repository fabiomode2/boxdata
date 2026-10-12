from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from curl_cffi import CurlError

from src.session import LB_SEM, LETTERBOXD_BASE, get_curl_session
from src.timers import _timer_start, _timer_stop

logger = logging.getLogger(__name__)

_MAX_PAGES = 10
_SLUG_WORKERS = 15

# Cloudflare serves intermittent "Just a moment..." challenge pages (403, or a
# 200 JS shell without the film grid) to datacenter IPs. Retry with backoff
# rides over the challenge window; a single warmed session then gets real pages.
_RETRY_MAX_ATTEMPTS = 5
_RETRY_BASE_DELAY = 2.5
_PAGE_DELAY = 1.0
_CHALLENGE_MARKER = "Just a moment"

_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
_CACHE_TTL = 86400

_SLUG_CACHE_PATH = _CACHE_DIR / "_slug_cache.json"
_slug_cache: dict[str, int] | None = None


def _load_slug_cache() -> dict[str, int]:
    global _slug_cache
    if _slug_cache is not None:
        return _slug_cache
    if _SLUG_CACHE_PATH.exists():
        raw = json.loads(_SLUG_CACHE_PATH.read_text())
        _slug_cache = {k: v for k, v in raw.items() if v is not None}
    else:
        _slug_cache = {}
    return _slug_cache


def _save_slug_cache() -> None:
    if _slug_cache is None:
        return
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _SLUG_CACHE_PATH.write_text(json.dumps(_slug_cache, indent=2))
    os.chmod(_SLUG_CACHE_PATH, 0o600)


def _cache_path(username: str) -> Path:
    return _CACHE_DIR / f"{username}.json"


def _load_cache(username: str) -> dict[int, float] | None:
    path = _cache_path(username)
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > _CACHE_TTL:
        path.unlink(missing_ok=True)
        return None
    raw = json.loads(path.read_text())
    if not raw:
        path.unlink(missing_ok=True)
        return None
    return {int(k): v for k, v in raw.items()}


def _save_cache(username: str, data: dict[int, float]) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(username)
    path.write_text(json.dumps(data, indent=2))
    path.chmod(0o600)


def _get_with_retry(
    session, url: str, max_attempts: int = _RETRY_MAX_ATTEMPTS,
    base_delay: float = _RETRY_BASE_DELAY,
):
    """GET with backoff. Retries non-200 and Cloudflare challenge pages.

    Returns the last response object (may still be a challenge/non-200), or
    None when every attempt raised a transport error.
    """
    resp = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.get(url, timeout=20)
        except CurlError as e:
            logger.debug("GET %s failed (%s), attempt %d", url, e, attempt)
            resp = None
        if resp is not None and resp.status_code == 200 and _CHALLENGE_MARKER not in resp.text:
            return resp
        if attempt < max_attempts:
            delay = base_delay * attempt
            if resp is not None and resp.status_code == 200:
                logger.debug("Challenge page for %s, retrying in %.1fs", url, delay)
            time.sleep(delay)
    return resp


def fetch_letterboxd_watchlist(
    username: str, max_films: int = 200, no_cache: bool = False,
    progress: dict | None = None,
) -> dict[int, float]:
    _timer_start(f"fetch_letterboxd_watchlist({username})")
    cache_path = _CACHE_DIR / f"{username}_watchlist.json"
    if not no_cache and cache_path.exists():
        raw = json.loads(cache_path.read_text())
        if raw:
            print(f"[BOXSCRAPE] Cache hit: {len(raw)} films in watchlist for {username}", file=sys.stderr, flush=True)
            _timer_stop()
            return {int(k): v for k, v in raw.items()}

    print(f"[BOXSCRAPE] Starting watchlist scrape for {username}", file=sys.stderr, flush=True)
    scraper = get_curl_session()

    _timer_start("  scrape watchlist pages")
    slug_set: set[str] = set()

    for page in range(1, _MAX_PAGES + 1):
        url = f"{LETTERBOXD_BASE}/{username}/watchlist/page/{page}/"
        print(f"[BOXSCRAPE] GET {url}", file=sys.stderr, flush=True)
        resp = _get_with_retry(scraper, url)
        if resp is None or resp.status_code != 200:
            print(f"[BOXSCRAPE] Giving up on watchlist page {page} (HTTP {getattr(resp, 'status_code', None)})", file=sys.stderr, flush=True)
            break
        entries = _extract_film_page_entries(resp.text)
        if not entries:
            break
        for slug, _ in entries:
            if len(slug_set) >= max_films:
                break
            slug_set.add(slug)
        if len(slug_set) >= max_films:
            break
        time.sleep(_PAGE_DELAY)
    _timer_stop()

    if not slug_set:
        _timer_stop()
        return {}

    if progress is not None:
        progress.update(phase="resolving_watchlist", total=len(slug_set), resolved=0)

    _timer_start("  resolve slugs -> TMDB")
    weighted: dict[int, float] = {}
    stop_event = threading.Event()
    with ThreadPoolExecutor(max_workers=_SLUG_WORKERS) as pool:
        fut = {pool.submit(_resolve_slug_parallel, slug, stop_event): slug for slug in slug_set}
        for future in as_completed(fut):
            if stop_event.is_set():
                break
            slug = fut[future]
            try:
                _, tmdb_id = future.result()
                if tmdb_id:
                    weighted[tmdb_id] = 0.5
                    if progress is not None:
                        progress["resolved"] = len(weighted)
            except Exception as e:
                logger.debug("Slug resolution failed for '%s': %s", slug, e)

    for f in fut:
        f.cancel()
    _timer_stop()

    if weighted:
        cache_path.write_text(json.dumps({str(k): v for k, v in weighted.items()}, indent=2))
        cache_path.chmod(0o600)
    _timer_stop()
    return weighted


def fetch_letterboxd_watched(
    username: str, max_films: int = 500, no_cache: bool = False,
    progress: dict | None = None,
) -> dict[int, float]:
    _timer_start(f"fetch_letterboxd_watched({username})")
    if not no_cache:
        cached = _load_cache(username)
        if cached is not None:
            print(f"[BOXSCRAPE] Cache hit: {len(cached)} films for {username}", file=sys.stderr, flush=True)
            _timer_stop()
            return cached

    print(f"[BOXSCRAPE] Starting scrape for {username}", file=sys.stderr, flush=True)
    if progress is not None:
        progress.update(user=username, phase="pages")

    _timer_start("  scrape pages")
    slug_ratings: dict[str, float] = {}
    scraper = get_curl_session()

    for page in range(1, _MAX_PAGES + 1):
        url = f"{LETTERBOXD_BASE}/{username}/films/page/{page}/"
        print(f"[BOXSCRAPE] GET {url}", file=sys.stderr, flush=True)
        resp = _get_with_retry(scraper, url)
        if resp is None or resp.status_code != 200:
            print(f"[BOXSCRAPE] Giving up on page {page} (HTTP {getattr(resp, 'status_code', None)})", file=sys.stderr, flush=True)
            break
        entries = _extract_film_page_entries(resp.text)
        if not entries:
            break
        for slug, rating_norm in entries:
            if len(slug_ratings) >= max_films:
                break
            slug_ratings[slug] = max(slug_ratings.get(slug, 0), rating_norm)
        if len(slug_ratings) >= max_films:
            break
        time.sleep(_PAGE_DELAY)
    _timer_stop()

    if not slug_ratings:
        print(f"[BOXSCRAPE] No films found for {username}", file=sys.stderr, flush=True)
        _timer_stop()
        return {}

    if progress is not None:
        progress.update(phase="resolving", total=len(slug_ratings), resolved=0)

    _timer_start("  resolve slugs -> TMDB")
    weighted: dict[int, float] = {}
    slug_stop = threading.Event()
    with ThreadPoolExecutor(max_workers=_SLUG_WORKERS) as pool:
        fut = {pool.submit(_resolve_slug_parallel, slug, slug_stop): slug for slug in slug_ratings}
        for future in as_completed(fut):
            if slug_stop.is_set():
                break
            slug = fut[future]
            weight = slug_ratings[slug]
            try:
                _, tmdb_id = future.result()
                if tmdb_id:
                    existing = weighted.get(tmdb_id, 0)
                    weighted[tmdb_id] = max(existing, weight)
                    if progress is not None:
                        progress["resolved"] = len(weighted)
            except Exception as e:
                logger.debug("Slug resolution failed for '%s': %s", slug, e)

    for f in fut:
        f.cancel()
    _timer_stop()

    if weighted:
        _save_cache(username, weighted)
    _timer_stop()
    return weighted


def _extract_film_page_entries(html: str) -> list[tuple[str, float]]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    entries: list[tuple[str, float]] = []

    for li in soup.select('li.griditem'):
        rc = li.select_one('div.react-component[data-item-slug]')
        if not rc:
            continue
        slug = rc.get('data-item-slug')
        if not slug:
            continue

        rating_el = li.select_one('.rating')
        class_attr = " ".join(rating_el.get("class", [])) if rating_el else ""
        rating_m = re.search(r'rated-(\d+)', class_attr)
        rating = int(rating_m.group(1)) / 10.0 if rating_m else 0.6

        entries.append((slug, rating))

    return entries


def _resolve_slug_parallel(slug: str, stop_event: threading.Event | None = None) -> tuple[str, int | None]:
    if stop_event and stop_event.is_set():
        return slug, None

    cache = _load_slug_cache()
    cached = cache.get(slug)
    if cached is not None:
        return slug, cached

    with LB_SEM:
        if stop_event and stop_event.is_set():
            return slug, None

        from bs4 import BeautifulSoup

        url = f"{LETTERBOXD_BASE}/film/{slug}/"
        try:
            scraper = get_curl_session()
            resp = _get_with_retry(scraper, url, max_attempts=3, base_delay=2.0)
            if stop_event and stop_event.is_set():
                return slug, None
            if resp is None or resp.status_code != 200:
                return slug, None
            soup = BeautifulSoup(resp.text, "html.parser")
            el = soup.select_one('[data-tmdb-id]')
            if el:
                tid = int(el["data-tmdb-id"])
                cache[slug] = tid
                _save_slug_cache()
                return slug, tid
        except CurlError as e:
            logger.warning("Slug resolution error for '%s': %s", slug, e)
        return slug, None
