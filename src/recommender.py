from __future__ import annotations

import math
import re
import time
from statistics import mean
from typing import Any

from curl_cffi import requests as curl_requests

from src.vector_store import get_collection

LETTERBOXD_BASE = "https://letterboxd.com"
_REQUEST_DELAY = 1.0
_MAX_PAGES = 10

_session: curl_requests.Session | None = None


def _get_scraper() -> curl_requests.Session:
    global _session
    if _session is None:
        _session = curl_requests.Session(impersonate="chrome120")
    return _session


_mean_embedding: list[float] | None = None


def _get_mean_embedding() -> list[float]:
    """Compute the mean of ALL movie embeddings in ChromaDB (cached)."""
    global _mean_embedding
    if _mean_embedding is not None:
        return _mean_embedding

    col = get_collection()
    res = col.get(include=["embeddings"])
    embs = res["embeddings"]
    dim = len(embs[0])
    mean = [0.0] * dim
    for e in embs:
        arr = e.tolist() if hasattr(e, "tolist") else e
        for i in range(dim):
            mean[i] += arr[i]
    _mean_embedding = [v / len(embs) for v in mean]
    return _mean_embedding


def _center(vec: list[float]) -> list[float]:
    mean = _get_mean_embedding()
    return [vec[i] - mean[i] for i in range(len(vec))]


def fetch_letterboxd_watched(
    username: str, max_films: int = 500
) -> dict[int, float]:
    """Scrape diary + ratings pages → {tmdb_id: weight}.

    Combines diary entries (with recency bonus) and all rated films
    (without recency, since they aren't chronological).
    Weight = (rating/10).
    - rated-7 = 3.5/5 → 0.7
    - no explicit rating → 0.6 (assumed liked)
    - diary entries get a recency bonus (newest ×1.0 → oldest ×0.5)
    """
    scraper = _get_scraper()
    seen_slugs: set[str] = set()
    slug_ratings: dict[str, float] = {}

    # 1) Diary entries (chronological, with recency bonus)
    diary_entries: list[tuple[str, float]] = []
    for page in range(1, _MAX_PAGES + 1):
        if len(diary_entries) >= max_films:
            break
        url = f"{LETTERBOXD_BASE}/{username}/films/diary/page/{page}/"
        resp = scraper.get(url, timeout=20)
        if resp.status_code != 200:
            break
        page_entries = _extract_entries(resp.text)
        if not page_entries:
            break
        diary_entries.extend(page_entries)
        time.sleep(_REQUEST_DELAY)

    n_diary = len(diary_entries)
    for pos, (slug, rating_norm) in enumerate(diary_entries):
        if len(seen_slugs) >= max_films:
            break
        recency = 1.0 - (pos / n_diary) * 0.5 if n_diary > 1 else 1.0
        weight = rating_norm * recency
        if slug not in seen_slugs:
            seen_slugs.add(slug)
            slug_ratings[slug] = max(slug_ratings.get(slug, 0), weight)

    # 2) Ratings page (all rated films, no recency)
    for page in range(1, _MAX_PAGES + 1):
        if len(seen_slugs) >= max_films:
            break
        url = f"{LETTERBOXD_BASE}/{username}/films/by/rating/page/{page}/"
        resp = scraper.get(url, timeout=20)
        if resp.status_code != 200:
            break
        page_entries = _extract_rating_entries(resp.text)
        if not page_entries:
            break
        for slug, rating_norm in page_entries:
            if len(seen_slugs) >= max_films:
                break
            if slug not in seen_slugs:
                seen_slugs.add(slug)
                slug_ratings[slug] = max(slug_ratings.get(slug, 0), rating_norm)
        time.sleep(_REQUEST_DELAY)

    # 3) Resolve slugs → TMDB IDs
    weighted: dict[int, float] = {}
    for slug, weight in slug_ratings.items():
        tmdb_id = _slug_to_tmdb(slug, scraper)
        if tmdb_id:
            existing = weighted.get(tmdb_id, 0)
            weighted[tmdb_id] = max(existing, weight)

    return weighted


def _extract_entries(html: str) -> list[tuple[str, float]]:
    """Extract (slug, normalised_rating) from a diary page (<tr> entries)."""
    entries: list[tuple[str, float]] = []

    tr_pattern = re.compile(
        r'<tr[^>]*class="[^"]*diary-entry-row[^"]*"[^>]*>.*?</tr>',
        re.DOTALL,
    )
    for tr in tr_pattern.finditer(html):
        row = tr.group(0)
        slug = _extract_slug(row)
        if not slug:
            continue

        rating_m = re.search(r'rated-(\d+)', row)
        rating = int(rating_m.group(1)) / 10.0 if rating_m else 0.6

        entries.append((slug, rating))

    return entries


def _extract_rating_entries(html: str) -> list[tuple[str, float]]:
    """Extract (slug, normalised_rating) from a ratings page (<li> grid items)."""
    entries: list[tuple[str, float]] = []

    li_pattern = re.compile(
        r'<li[^>]*class="[^"]*griditem[^"]*"[^>]*>.*?</li>',
        re.DOTALL,
    )
    for li in li_pattern.finditer(html):
        item = li.group(0)
        slug = _extract_slug(item)
        if not slug:
            continue

        rating_m = re.search(r'rated-(\d+)', item)
        if not rating_m:
            continue
        rating = int(rating_m.group(1)) / 10.0

        entries.append((slug, rating))

    return entries


def _extract_slug(html: str) -> str | None:
    m = re.search(r'/film/([^/]+)/"', html)
    return m.group(1) if m else None


def _slug_to_tmdb(slug: str, scraper: curl_requests.Session) -> int | None:
    """Fetch a Letterboxd film page and extract the TMDB ID."""
    url = f"{LETTERBOXD_BASE}/film/{slug}/"
    try:
        resp = scraper.get(url, timeout=20)
        if resp.status_code != 200:
            return None
        m = re.search(r'data-tmdb-id="(\d+)"', resp.text)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _get_embedding(tmdb_id: int) -> list[float] | None:
    col = get_collection()
    res = col.get(str(tmdb_id), include=["embeddings"])
    emb = res["embeddings"]
    if emb is not None and len(emb) > 0 and emb[0] is not None:
        return emb[0].tolist() if hasattr(emb[0], "tolist") else emb[0]
    return None


def get_taste_vector(
    weighted_ids: dict[int, float],
) -> tuple[list[float], list[int], int]:
    """Weighted-average embedding of ALL watched movies.

    Each movie's weight combines:
    - User rating (higher = more weight)
    - Recency (newer = more weight)
    - Inverse popularity (niche movies = more weight)

    Using all movies + inverse-popularity weighting gives stable,
    discriminating taste vectors without needing a top-N cutoff.
    """
    if not weighted_ids:
        return [], [], 0

    # Batch-fetch vote_counts for inverse-popularity weighting
    col = get_collection()
    meta_res = col.get(
        ids=[str(tid) for tid in weighted_ids],
        include=["metadatas", "embeddings"],
    )

    vectors: list[list[float]] = []
    weights: list[float] = []
    found: list[int] = []

    for i, tid_str in enumerate(meta_res["ids"]):
        tid = int(tid_str)
        user_weight = weighted_ids[tid]
        meta = meta_res["metadatas"][i]
        emb = meta_res["embeddings"][i]

        if emb is None:
            continue
        emb_arr = emb.tolist() if hasattr(emb, "tolist") else emb

        vote_count = meta.get("vote_count", 10)
        inv_pop = 1.0 / math.log(vote_count + 10)

        vectors.append(emb_arr)
        weights.append(user_weight * inv_pop)
        found.append(tid)

    if not vectors:
        return [], [], len(weighted_ids)

    dim = len(vectors[0])
    avg = [0.0] * dim
    total_w = sum(weights)
    for vec, w in zip(vectors, weights):
        for i in range(dim):
            avg[i] += vec[i] * (w / total_w)

    return avg, found, len(weighted_ids)


def taste_similarity(
    a_weighted: dict[int, float], b_weighted: dict[int, float]
) -> dict[str, Any]:
    vec_a, found_a, total_a = get_taste_vector(a_weighted)
    vec_b, found_b, total_b = get_taste_vector(b_weighted)

    if not vec_a or not vec_b:
        return {
            "similarity": 0.0,
            "error": "Not enough embeddings found for one or both users",
            "user_a_found": len(found_a),
            "user_a_total": total_a,
            "user_b_found": len(found_b),
            "user_b_total": total_b,
        }

    # Center both vectors for better differentiation
    sim = cosine_similarity(_center(vec_a), _center(vec_b))
    sim = max(-1.0, min(1.0, sim))  # clamp to [-1, 1]
    return {
        "similarity": sim,
        "similarity_pct": round(sim * 100, 1),
        "user_a_found": len(found_a),
        "user_a_total": total_a,
        "user_b_found": len(found_b),
        "user_b_total": total_b,
    }


def recommend_for_group(
    user_weighted_list: list[dict[int, float]],
    n: int = 10,
    min_vote_count: int = 10,
) -> list[dict[str, Any]]:
    exclude: set[int] = set()
    taste_vectors: list[list[float]] = []

    for wdict in user_weighted_list:
        exclude.update(wdict.keys())
        vec, found, _ = get_taste_vector(wdict)
        if vec:
            taste_vectors.append(vec)

    if not taste_vectors:
        return []

    group_vec = [mean(vals) for vals in zip(*taste_vectors)]

    col = get_collection()
    total_db = col.count()
    results = col.query(
        query_embeddings=[group_vec],
        n_results=min(total_db, n * 15),
    )

    candidates: list[dict[str, Any]] = []
    for i, tmdb_id_str in enumerate(results["ids"][0]):
        tid = int(tmdb_id_str)
        if tid in exclude:
            continue
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        similarity = 1.0 / (1.0 + dist)

        vote_avg = meta.get("vote_average", 0)
        vote_cnt = meta.get("vote_count", 0)

        if vote_cnt < min_vote_count:
            continue

        rerank = similarity * (vote_avg * math.log(vote_cnt + 1))

        candidates.append({
            "tmdb_id": tid,
            "title": meta.get("title", "Unknown"),
            "genres": meta.get("genres", ""),
            "year": meta.get("year", 0),
            "vote_average": vote_avg,
            "vote_count": vote_cnt,
            "similarity_score": round(similarity, 4),
            "rerank_score": round(rerank, 4),
            "url": f"https://letterboxd.com/tmdb/{tid}/",
        })

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:n]
