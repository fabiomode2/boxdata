from __future__ import annotations

import csv
import logging
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.collector import tmdb_get
from src.config import HEADERS, TMDB_BASE
from src.session import LB_SEM, get_curl_session
from src.vector_store import get_collection

logger = logging.getLogger(__name__)


@dataclass
class WatchedFilm:
    name: str
    year: int
    rating: float | None
    uri: str
    tmdb_id: int | None = None


@dataclass
class UserProfile:
    username: str
    watched: list[WatchedFilm] = field(default_factory=list)
    watchlist: list[WatchedFilm] = field(default_factory=list)


def _parse_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


_SCRAPE_CACHE: dict[str, int | None] = {}
_LETTERBOXD_BASE = "https://letterboxd.com"


def _resolve_from_uri(uri: str) -> int | None:
    """Extract TMDB ID from a Letterboxd film page via data-tmdb-id."""
    from bs4 import BeautifulSoup
    slug_match = re.search(r'/film/([^/]+)/', uri)
    if not slug_match:
        return None
    slug = slug_match.group(1)
    cache_key = f"lb:{slug}"
    cached = _SCRAPE_CACHE.get(cache_key)
    if cached is not None:
        return cached if cached > 0 else None
    with LB_SEM:
        try:
            resp = get_curl_session().get(f"{_LETTERBOXD_BASE}/film/{slug}/", timeout=15)
            if resp.status_code != 200:
                _SCRAPE_CACHE[cache_key] = -1
                return None
            soup = BeautifulSoup(resp.text, "html.parser")
            el = soup.select_one("[data-tmdb-id]")
            if el:
                tid = int(el["data-tmdb-id"])
                _SCRAPE_CACHE[cache_key] = tid
                return tid
            _SCRAPE_CACHE[cache_key] = -1
            return None
        except Exception as e:
            logger.debug("URI resolution failed for '%s': %s", uri, e)
            _SCRAPE_CACHE[cache_key] = -1
            return None


def _resolve_via_tmdb(name: str, year: int) -> int | None:
    """Resolve via TMDB /search/movie."""
    cache_key = f"tmdb:{name}|{year}"
    cached = _SCRAPE_CACHE.get(cache_key)
    if cached is not None:
        return cached if cached > 0 else None
    try:
        data = tmdb_get(
            f"{TMDB_BASE}/search/movie",
            params={"query": name, "year": year},
        )
        results = data.get("results", [])
        if not results:
            _SCRAPE_CACHE[cache_key] = -1
            return None
        for r in results:
            if r.get("title", "").lower().strip() == name.lower().strip():
                _SCRAPE_CACHE[cache_key] = r["id"]
                return r["id"]
        _SCRAPE_CACHE[cache_key] = results[0]["id"]
        return results[0]["id"]
    except Exception as e:
        logger.debug("TMDB search failed for '%s' (%d): %s", name, year, e)
        return None


_CHROMA_INDEX: dict[tuple[str, int], int] | None = None


def _build_chromadb_index() -> dict[tuple[str, int], int]:
    """Build (title_lower, year) → tmdb_id index from ChromaDB (cached)."""
    global _CHROMA_INDEX
    if _CHROMA_INDEX is not None:
        return _CHROMA_INDEX
    col = get_collection()
    total = col.count()
    batch_size = 1000
    index: dict[tuple[str, int], int] = {}
    for offset in range(0, total, batch_size):
        res = col.get(offset=offset, limit=batch_size, include=["metadatas"])
        for i, tid_str in enumerate(res["ids"]):
            meta = res["metadatas"][i]
            title = (meta.get("title") or "").lower().strip()
            year = meta.get("year") or 0
            if title and year:
                index[(title, year)] = int(tid_str)
    _CHROMA_INDEX = index
    return index


def parse_export(export_dir: str) -> UserProfile:
    """Parse Letterboxd export directory into a UserProfile."""
    base = Path(export_dir).expanduser().resolve()
    if not base.is_dir():
        raise NotADirectoryError(f"Export directory not found: {base}")

    username = (base.parent.name
                if base.name.startswith("letterboxd-")
                else base.name)

    ratings_path = base / "ratings.csv"
    watched_path = base / "watched.csv"
    watchlist_path = base / "watchlist.csv"

    profile = UserProfile(username=username)
    seen_in_watched: set[str] = set()

    # Watched + ratings
    if ratings_path.exists():
        for row in _parse_csv(ratings_path):
            name = row.get("Name", "").strip()
            year_s = row.get("Year", "").strip()
            uri = row.get("Letterboxd URI", "").strip()
            rating_s = row.get("Rating", "").strip()
            if not name or not year_s:
                continue
            year = int(year_s)
            rating = float(rating_s) if rating_s else None
            f = WatchedFilm(name=name, year=year, rating=rating, uri=uri)
            profile.watched.append(f)
            seen_in_watched.add(f"{name}|{year}")

    # Pure watches (no rating) from watched.csv not already in watches
    if watched_path.exists():
        for row in _parse_csv(watched_path):
            name = row.get("Name", "").strip()
            year_s = row.get("Year", "").strip()
            uri = row.get("Letterboxd URI", "").strip()
            if not name or not year_s:
                continue
            key = f"{name}|{year_s}"
            if key not in seen_in_watched:
                profile.watched.append(WatchedFilm(
                    name=name, year=int(year_s), rating=None, uri=uri,
                ))
                seen_in_watched.add(key)

    # Watchlist
    if watchlist_path.exists():
        for row in _parse_csv(watchlist_path):
            name = row.get("Name", "").strip()
            year_s = row.get("Year", "").strip()
            uri = row.get("Letterboxd URI", "").strip()
            if not name or not year_s:
                continue
            profile.watchlist.append(WatchedFilm(
                name=name, year=int(year_s), rating=None, uri=uri,
            ))

    return profile


def resolve_tmdb_ids(profile: UserProfile, resolve_watchlist: bool = False) -> UserProfile:
    """Resolve films to TMDB IDs via ChromaDB → URI (parallel) → TMDB search (parallel)."""
    chroma_index = _build_chromadb_index()
    pending: list[WatchedFilm] = []
    for f in profile.watched:
        key = (f.name.lower().strip(), f.year)
        tid = chroma_index.get(key)
        if tid:
            f.tmdb_id = tid
        else:
            pending.append(f)

    if resolve_watchlist:
        for f in profile.watchlist:
            key = (f.name.lower().strip(), f.year)
            tid = chroma_index.get(key)
            if tid:
                f.tmdb_id = tid
            else:
                pending.append(f)

    if not pending:
        return profile

    # Step 1: resolve via URI (Letterboxd page → data-tmdb-id) — no TMDB API
    uri_films = [f for f in pending if f.uri]
    n_uri = len(uri_films)
    if uri_films:
        logger.info("  Resolving %d films via Letterboxd URI...", n_uri)
        done = 0
        with ThreadPoolExecutor(max_workers=20) as pool:
            fut = {pool.submit(_resolve_from_uri, f.uri): f for f in uri_films}
            for future in as_completed(fut):
                f = fut[future]
                try:
                    tid = future.result()
                    if tid:
                        f.tmdb_id = tid
                except Exception as e:
                    logger.debug("URI resolve failed for '%s': %s", f.name, e)
                    logger.info("    resolved %d/%d via URI", done, n_uri)

    # Step 2: TMDB search for remaining (parallel)
    remaining = [f for f in pending if not f.tmdb_id]
    n_rem = len(remaining)
    if remaining:
        logger.info("  Searching %d films via TMDB...", n_rem)
        done = 0
        with ThreadPoolExecutor(max_workers=10) as pool:
            fut = {pool.submit(_resolve_via_tmdb, f.name, f.year): f for f in remaining}
            for future in as_completed(fut):
                f = fut[future]
                try:
                    tid = future.result()
                    if tid:
                        f.tmdb_id = tid
                except Exception as e:
                    logger.debug("TMDB search failed for '%s' (%d): %s", f.name, f.year, e)
                    logger.info("    searched %d/%d via TMDB", done, n_rem)

    return profile


def analyze_profile(profile: UserProfile) -> dict[str, Any]:
    """Compute stats and insights from a resolved profile."""
    col = get_collection()

    # Fetch ChromaDB metadata for all resolved films
    resolved_ids = [f.tmdb_id for f in profile.watched if f.tmdb_id]
    global_meta: dict[int, dict] = {}
    if resolved_ids:
        all_data = col.get(ids=[str(tid) for tid in resolved_ids], include=["metadatas"])
        for i, tid_str in enumerate(all_data["ids"]):
            global_meta[int(tid_str)] = all_data["metadatas"][i]

    rated = [f for f in profile.watched if f.rating is not None]
    avg_rating = sum(f.rating for f in rated) / len(rated) if rated else 0.0

    in_db = sum(1 for f in profile.watched if f.tmdb_id and str(f.tmdb_id) in [str(k) for k in global_meta])

    # Genre distribution (from ChromaDB metadata)
    genre_counter: Counter[str] = Counter()
    genre_ratings: dict[str, list[float]] = defaultdict(list)
    decade_counter: Counter[str] = Counter()
    disagreements: list[dict] = []

    for f in profile.watched:
        meta = global_meta.get(f.tmdb_id) if f.tmdb_id else None
        if not meta:
            continue
        genres_str = meta.get("genres", "")
        for g in genres_str.split(","):
            g = g.strip()
            if g:
                genre_counter[g] += 1
                if f.rating is not None:
                    genre_ratings[g].append(f.rating)

        yr = meta.get("year") or f.year
        decade = f"{yr - (yr % 10)}s"
        decade_counter[decade] += 1

        # Disagreement: user rating vs global TMDB average
        if f.rating is not None:
            user_norm = f.rating / 5.0
            global_norm = (meta.get("vote_average") or 0) / 10.0
            diff = user_norm - global_norm
            if abs(diff) > 0.15:
                disagreements.append({
                    "title": meta.get("title", f.name),
                    "year": yr,
                    "user_rating": f.rating,
                    "global_avg": round(meta.get("vote_average", 0), 1),
                    "diff": round(diff, 2),
                })

    disagreements.sort(key=lambda d: abs(d["diff"]), reverse=True)

    genre_avg = {g: round(sum(v) / len(v), 2) for g, v in genre_ratings.items()}
    top_genres = genre_counter.most_common(10)
    top_decades = decade_counter.most_common()

    in_watchlist_db = sum(
        1 for f in profile.watchlist
        if f.tmdb_id and col.get(str(f.tmdb_id))["ids"]
    )

    return {
        "username": profile.username,
        "total_watched": len(profile.watched),
        "total_rated": len(rated),
        "total_watchlist": len(profile.watchlist),
        "avg_rating": round(avg_rating, 2),
        "in_chromadb": in_db,
        "watchlist_in_chromadb": in_watchlist_db,
        "top_genres": top_genres,
        "genre_avg_rating": genre_avg,
        "decade_distribution": top_decades,
        "top_disagreements": disagreements[:10],
    }


def format_profile_report(stats: dict[str, Any]) -> str:
    lines = []
    lines.append(f"Letterboxd Profile: {stats['username']}")
    lines.append("━" * 50)
    lines.append(
        f" Watched: {stats['total_watched']}  |  "
        f"Rated: {stats['total_rated']}  |  "
        f"Watchlist: {stats['total_watchlist']}"
    )
    lines.append(f" In ChromaDB: {stats['in_chromadb']}/{stats['total_watched']}")
    lines.append(f" Watchlist in ChromaDB: {stats['watchlist_in_chromadb']}/{stats['total_watchlist']}")
    lines.append("")
    lines.append(f" Your average rating: {stats['avg_rating']}★  (Letterboxd scale)")

    lines.append("")
    lines.append("── Top Genres ──")
    for g, count in stats["top_genres"]:
        avg = stats["genre_avg_rating"].get(g, 0)
        lines.append(f"  {g:<20s} {count:3d} films  avg {avg}★")

    lines.append("")
    lines.append("── Decade Breakdown ──")
    for decade, count in stats["decade_distribution"]:
        pct = count / stats["total_watched"] * 100
        lines.append(f"  {decade:<10s} {count:3d} films ({pct:.1f}%)")

    if stats["top_disagreements"]:
        lines.append("")
        lines.append("── Where You Disagree With The Crowd ──")
        for d in stats["top_disagreements"][:8]:
            arrow = "↑" if d["diff"] > 0 else "↓"
            lines.append(
                f"  You: {d['user_rating']}★  |  "
                f"Global: {d['global_avg']}/10  {arrow}  "
                f"{d['title']} ({d['year']})"
            )

    return "\n".join(lines)
