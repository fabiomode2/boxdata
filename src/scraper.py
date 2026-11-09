from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
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

# A scraped profile stays fresh for a week. Expired entries are kept on disk
# and served as a fallback when a refresh fails, instead of returning nothing.
_CACHE_TTL = int(os.getenv("BOXSCRAPE_CACHE_TTL", 7 * 86400))
# After a failed scrape, skip retrying that user for a while: the Oracle VPS
# IP is Cloudflare-blocked, so retrying every request only burns 25s per job.
_FAILURE_TTL = int(os.getenv("BOXSCRAPE_FAILURE_TTL", 900))

_SLUG_CACHE_PATH = _CACHE_DIR / "_slug_cache.json"
_SLUG_CACHE_FLUSH_INTERVAL = 5.0
_slug_cache: dict[str, int] | None = None
_slug_cache_lock = threading.Lock()
_slug_cache_dirty = False
_slug_cache_last_flush = 0.0

# Per-user locks: concurrent jobs for the same username would otherwise scrape
# the same pages twice. The waiter re-reads the cache after acquiring the lock.
_USER_LOCKS: dict[str, threading.Lock] = {}
_USER_LOCKS_GUARD = threading.Lock()


@contextmanager
def _user_lock(username: str):
    with _USER_LOCKS_GUARD:
        lock = _USER_LOCKS.setdefault(username, threading.Lock())
    with lock:
        yield


def _load_slug_cache() -> dict[str, int]:
    global _slug_cache
    if _slug_cache is not None:
        return _slug_cache
    with _slug_cache_lock:
        if _slug_cache is not None:  # filled by another thread while waiting
            return _slug_cache
        if _SLUG_CACHE_PATH.exists():
            try:
                raw = json.loads(_SLUG_CACHE_PATH.read_text())
                _slug_cache = {k: v for k, v in raw.items() if v is not None}
            except (OSError, ValueError) as e:
                logger.warning("Corrupt slug cache (%s); rebuilding", e)
                _slug_cache = {}
        else:
            _slug_cache = {}
        return _slug_cache


def _slug_cache_set(slug: str, tmdb_id: int) -> None:
    global _slug_cache_dirty
    cache = _load_slug_cache()
    with _slug_cache_lock:
        cache[slug] = tmdb_id
        _slug_cache_dirty = True
    # Throttled flush: rewriting a 60KB JSON on every resolution would be
    # silly. Callers force a final flush when a scrape ends.
    _save_slug_cache()


def _save_slug_cache(force: bool = False) -> None:
    """Persist slug cache atomically. Throttled unless ``force``."""
    global _slug_cache_dirty, _slug_cache_last_flush
    with _slug_cache_lock:
        if _slug_cache is None or not _slug_cache_dirty:
            return
        now = time.time()
        if not force and now - _slug_cache_last_flush < _SLUG_CACHE_FLUSH_INTERVAL:
            return
        snapshot = dict(_slug_cache)
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = _SLUG_CACHE_PATH.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(snapshot, indent=2))
            os.replace(tmp, _SLUG_CACHE_PATH)
            os.chmod(_SLUG_CACHE_PATH, 0o600)
        except OSError as e:
            logger.warning("Could not persist slug cache: %s", e)
            return
        _slug_cache_dirty = False
        _slug_cache_last_flush = now


def _cache_path(username: str) -> Path:
    return _CACHE_DIR / f"{username}.json"


def _watchlist_cache_path(username: str) -> Path:
    return _CACHE_DIR / f"{username}_watchlist.json"


def _failure_path(key: str) -> Path:
    return _CACHE_DIR / f"{key}.failed"


def _read_cache_file(path: Path) -> tuple[dict[int, float], float] | None:
    """Return ``(data, age_seconds)`` for a cache file, or None if unusable."""
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
        data = {int(k): float(v) for k, v in raw.items()}
    except (OSError, ValueError, AttributeError) as e:
        logger.warning("Discarding unreadable cache %s: %s", path.name, e)
        path.unlink(missing_ok=True)
        return None
    if not data:
        path.unlink(missing_ok=True)
        return None
    return data, time.time() - path.stat().st_mtime


def _load_cache(username: str, ttl: float = _CACHE_TTL) -> dict[int, float] | None:
    """Fresh cached profile, or None. Expired entries are *not* deleted."""
    cached = _read_cache_file(_cache_path(username))
    if cached is None:
        return None
    data, age = cached
    return data if age <= ttl else None


def _save_cache(username: str, data: dict[int, float]) -> None:
    _write_cache(_cache_path(username), data)


def _mark_failure(username: str) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _failure_path(username).write_text(str(int(time.time())))
    except OSError as e:
        logger.debug("Could not write failure marker for %s: %s", username, e)


def _clear_failure(username: str) -> None:
    try:
        _failure_path(username).unlink(missing_ok=True)
    except OSError as e:
        logger.debug("Could not clear failure marker for %s: %s", username, e)


def _recent_failure(username: str) -> bool:
    path = _failure_path(username)
    if not path.exists():
        return False
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return False
    if age > _FAILURE_TTL:
        path.unlink(missing_ok=True)
        return False
    return True


def _cache_hit(username: str, data: dict[int, float], label: str = "films") -> None:
    print(f"[BOXSCRAPE] Cache hit: {len(data)} {label} for {username}",
          file=sys.stderr, flush=True)


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


def _resolve_slugs(
    slug_ratings: dict[str, float], progress: dict | None,
) -> dict[int, float]:
    """Resolve slugs to TMDB IDs in parallel, preserving the best weight."""
    _timer_start("  resolve slugs -> TMDB")
    weighted: dict[int, float] = {}
    stop_event = threading.Event()
    with ThreadPoolExecutor(max_workers=_SLUG_WORKERS) as pool:
        fut = {pool.submit(_resolve_slug_parallel, slug, stop_event): slug
               for slug in slug_ratings}
        for future in as_completed(fut):
            if stop_event.is_set():
                break
            slug = fut[future]
            weight = slug_ratings[slug]
            try:
                _, tmdb_id = future.result()
                if tmdb_id:
                    weighted[tmdb_id] = max(weighted.get(tmdb_id, 0.0), weight)
                    if progress is not None:
                        progress["resolved"] = len(weighted)
            except Exception as e:
                logger.debug("Slug resolution failed for '%s': %s", slug, e)

    for f in fut:
        f.cancel()
    _timer_stop()
    return weighted


def fetch_letterboxd_watchlist(
    username: str, max_films: int = 200, no_cache: bool = False,
    progress: dict | None = None,
) -> dict[int, float]:
    _timer_start(f"fetch_letterboxd_watchlist({username})")
    try:
        return _fetch_watchlist(username, max_films, no_cache, progress)
    finally:
        _timer_stop()


def _fetch_watchlist(
    username: str, max_films: int, no_cache: bool, progress: dict | None,
) -> dict[int, float]:
    path = _watchlist_cache_path(username)
    cached = _read_cache_file(path)
    stale = cached[0] if cached is not None else None
    if cached is not None and not no_cache and cached[1] <= _CACHE_TTL:
        _cache_hit(username, cached[0], "in watchlist")
        return cached[0]

    if not no_cache and _recent_failure(f"{username}_watchlist"):
        print(f"[BOXSCRAPE] Skipping watchlist scrape for {username} "
              f"(recent failure)", file=sys.stderr, flush=True)
        return stale or {}

    with _user_lock(f"watchlist:{username}"):
        if not no_cache:
            again = _read_cache_file(path)
            if again is not None and again[1] <= _CACHE_TTL:
                _cache_hit(username, again[0], "in watchlist")
                return again[0]
            if again is not None:
                stale = again[0]

        weighted, truncated = _scrape_watchlist(username, max_films, progress)
        _save_slug_cache(force=True)

        if not weighted:
            _mark_failure(f"{username}_watchlist")
            if stale:
                print(f"[BOXSCRAPE] Watchlist scrape failed; serving stale cache "
                      f"({len(stale)} films) for {username}", file=sys.stderr, flush=True)
                return stale
            return {}

        _clear_failure(f"{username}_watchlist")
        if truncated and stale and len(stale) >= len(weighted):
            print(f"[BOXSCRAPE] Watchlist scrape truncated ({len(weighted)}/{len(stale)}); "
                  f"keeping larger cached profile for {username}", file=sys.stderr, flush=True)
            return stale

        _write_cache(path, weighted)
        return weighted


def _scrape_watchlist(
    username: str, max_films: int, progress: dict | None,
) -> tuple[dict[int, float], bool]:
    """Return ``(weighted_ids, truncated)``. ``truncated`` marks a partial scrape."""
    print(f"[BOXSCRAPE] Starting watchlist scrape for {username}", file=sys.stderr, flush=True)
    scraper = get_curl_session()

    _timer_start("  scrape watchlist pages")
    slug_set: set[str] = set()
    truncated = False
    try:
        for page in range(1, _MAX_PAGES + 1):
            url = f"{LETTERBOXD_BASE}/{username}/watchlist/page/{page}/"
            print(f"[BOXSCRAPE] GET {url}", file=sys.stderr, flush=True)
            resp = _get_with_retry(scraper, url)
            if resp is None or resp.status_code != 200:
                print(f"[BOXSCRAPE] Giving up on watchlist page {page} "
                      f"(HTTP {getattr(resp, 'status_code', None)})", file=sys.stderr, flush=True)
                truncated = True
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
    finally:
        _timer_stop()

    if not slug_set:
        return {}, truncated

    if progress is not None:
        progress.update(phase="resolving_watchlist", total=len(slug_set), resolved=0)

    weighted = _resolve_slugs({slug: 0.5 for slug in slug_set}, progress)
    return weighted, truncated


def fetch_letterboxd_watched(
    username: str, max_films: int = 500, no_cache: bool = False,
    progress: dict | None = None,
) -> dict[int, float]:
    _timer_start(f"fetch_letterboxd_watched({username})")
    try:
        return _fetch_watched(username, max_films, no_cache, progress)
    finally:
        _timer_stop()


def _fetch_watched(
    username: str, max_films: int, no_cache: bool, progress: dict | None,
) -> dict[int, float]:
    path = _cache_path(username)
    cached = _read_cache_file(path)
    stale = cached[0] if cached is not None else None
    if cached is not None and not no_cache and cached[1] <= _CACHE_TTL:
        _cache_hit(username, cached[0])
        return cached[0]

    if not no_cache and _recent_failure(username):
        print(f"[BOXSCRAPE] Skipping scrape for {username} (recent failure)"
              + (f"; serving stale cache ({len(stale)} films)" if stale else ""),
              file=sys.stderr, flush=True)
        return stale or {}

    with _user_lock(username):
        if not no_cache:
            again = _read_cache_file(path)
            if again is not None and again[1] <= _CACHE_TTL:
                _cache_hit(username, again[0])
                return again[0]
            if again is not None:
                stale = again[0]

        weighted, truncated = _scrape_watched(username, max_films, progress)
        _save_slug_cache(force=True)

        if not weighted:
            _mark_failure(username)
            if stale:
                print(f"[BOXSCRAPE] Scrape failed; serving stale cache "
                      f"({len(stale)} films) for {username}", file=sys.stderr, flush=True)
                return stale
            print(f"[BOXSCRAPE] No films found for {username}", file=sys.stderr, flush=True)
            return {}

        _clear_failure(username)
        if truncated and stale and len(stale) >= len(weighted):
            print(f"[BOXSCRAPE] Scrape truncated ({len(weighted)}/{len(stale)} films); "
                  f"keeping larger cached profile for {username}", file=sys.stderr, flush=True)
            return stale

        _save_cache(username, weighted)
        return weighted


def _scrape_watched(
    username: str, max_films: int, progress: dict | None,
) -> tuple[dict[int, float], bool]:
    """Return ``(weighted_ids, truncated)``. ``truncated`` marks a partial scrape."""
    print(f"[BOXSCRAPE] Starting scrape for {username}", file=sys.stderr, flush=True)
    if progress is not None:
        progress.update(user=username, phase="pages")

    _timer_start("  scrape pages")
    slug_ratings: dict[str, float] = {}
    scraper = get_curl_session()
    truncated = False
    try:
        for page in range(1, _MAX_PAGES + 1):
            url = f"{LETTERBOXD_BASE}/{username}/films/page/{page}/"
            print(f"[BOXSCRAPE] GET {url}", file=sys.stderr, flush=True)
            resp = _get_with_retry(scraper, url)
            if resp is None or resp.status_code != 200:
                print(f"[BOXSCRAPE] Giving up on page {page} "
                      f"(HTTP {getattr(resp, 'status_code', None)})", file=sys.stderr, flush=True)
                truncated = True
                break
            entries = _extract_film_page_entries(resp.text)
            if not entries:
                break
            for slug, rating_norm in entries:
                if len(slug_ratings) >= max_films:
                    break
                slug_ratings[slug] = max(slug_ratings.get(slug, 0.0), rating_norm)
            if len(slug_ratings) >= max_films:
                break
            time.sleep(_PAGE_DELAY)
    finally:
        _timer_stop()

    if not slug_ratings:
        return {}, truncated

    if progress is not None:
        progress.update(phase="resolving", total=len(slug_ratings), resolved=0)

    weighted = _resolve_slugs(slug_ratings, progress)
    return weighted, truncated


def _write_cache(path: Path, data: dict[int, float]) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({str(k): v for k, v in data.items()}, indent=2))
    os.replace(tmp, path)
    path.chmod(0o600)


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
                _slug_cache_set(slug, tid)
                return slug, tid
        except CurlError as e:
            logger.warning("Slug resolution error for '%s': %s", slug, e)
        return slug, None
