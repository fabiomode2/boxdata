from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from curl_cffi import CurlError

from src.session import LB_SEM, LETTERBOXD_BASE, _REQUEST_DELAY, get_curl_session
from src.timers import _timer_start, _timer_stop

logger = logging.getLogger(__name__)

_MAX_PAGES = 10

_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
_CACHE_TTL = 86400


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


def fetch_letterboxd_watched(
    username: str, max_films: int = 500, no_cache: bool = False
) -> dict[int, float]:
    _timer_start(f"fetch_letterboxd_watched({username})")
    if not no_cache:
        cached = _load_cache(username)
        if cached is not None:
            print(f"[BOXSCRAPE] Cache hit: {len(cached)} films for {username}", file=sys.stderr, flush=True)
            _timer_stop()
            return cached

    scraper = get_curl_session()
    slug_ratings: dict[str, float] = {}
    print(f"[BOXSCRAPE] Starting scrape for {username}", file=sys.stderr, flush=True)

    _timer_start("  scrape pages")
    for page in range(1, _MAX_PAGES + 1):
        if len(slug_ratings) >= max_films:
            break
        url = f"{LETTERBOXD_BASE}/{username}/films/page/{page}/"
        print(f"[BOXSCRAPE] GET {url}", file=sys.stderr, flush=True)
        resp = scraper.get(url, timeout=20)
        print(f"[BOXSCRAPE] HTTP {resp.status_code} (len={len(resp.text)})", file=sys.stderr, flush=True)
        if resp.status_code != 200:
            print(f"[BOXSCRAPE] BODY={resp.text[:300]}", file=sys.stderr, flush=True)
            break
        page_entries = _extract_film_page_entries(resp.text)
        if not page_entries:
            break
        for slug, rating_norm in page_entries:
            if len(slug_ratings) >= max_films:
                break
            slug_ratings[slug] = max(slug_ratings.get(slug, 0), rating_norm)
        time.sleep(_REQUEST_DELAY)
    _timer_stop()

    _timer_start("  resolve slugs -> TMDB")
    weighted: dict[int, float] = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        fut = {pool.submit(_resolve_slug_parallel, slug): slug for slug in slug_ratings}
        for future in as_completed(fut):
            slug = fut[future]
            weight = slug_ratings[slug]
            try:
                _, tmdb_id = future.result()
                if tmdb_id:
                    existing = weighted.get(tmdb_id, 0)
                    weighted[tmdb_id] = max(existing, weight)
            except Exception as e:
                logger.debug("Slug resolution failed for '%s': %s", slug, e)
    _timer_stop()

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


def _resolve_slug_parallel(slug: str) -> tuple[str, int | None]:
    with LB_SEM:
        from bs4 import BeautifulSoup

        url = f"{LETTERBOXD_BASE}/film/{slug}/"
        try:
            scraper = get_curl_session()
            resp = scraper.get(url, timeout=20)
            if resp.status_code != 200:
                return slug, None
            soup = BeautifulSoup(resp.text, "html.parser")
            el = soup.select_one('[data-tmdb-id]')
            if el:
                return slug, int(el["data-tmdb-id"])
        except CurlError as e:
            logger.warning("Slug resolution error for '%s': %s", slug, e)
        return slug, None
