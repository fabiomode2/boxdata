from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from curl_cffi import requests as curl_requests

from src.vector_store import get_collection

logger = logging.getLogger(__name__)

LETTERBOXD_BASE = "https://letterboxd.com"
_REQUEST_DELAY = 1.0
_MAX_PAGES = 10

_session: curl_requests.Session | None = None
_SLUG_SEM = threading.Semaphore(5)


def _get_scraper() -> curl_requests.Session:
    global _session
    if _session is None:
        _session = curl_requests.Session(impersonate="chrome120")
    return _session


_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"


def _cache_path(username: str) -> Path:
    return _CACHE_DIR / f"{username}.json"


def _load_cache(username: str) -> dict[int, float] | None:
    path = _cache_path(username)
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return {int(k): v for k, v in raw.items()}


def _save_cache(username: str, data: dict[int, float]) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(username).write_text(json.dumps(data, indent=2))


def fetch_letterboxd_watched(
    username: str, max_films: int = 500, no_cache: bool = False
) -> dict[int, float]:
    """Scrape /films/page/N/ → {tmdb_id: weight}.

    Single source of truth: the complete "all films" list.
    Weight = (rating/10).
    - rated-7 = 3.5/5 → 0.7
    - no explicit rating → 0.6 (assumed liked)
    """
    if not no_cache:
        cached = _load_cache(username)
        if cached is not None:
            logger.info("  (cached) %d films", len(cached))
            return cached

    scraper = _get_scraper()
    slug_ratings: dict[str, float] = {}

    for page in range(1, _MAX_PAGES + 1):
        if len(slug_ratings) >= max_films:
            break
        url = f"{LETTERBOXD_BASE}/{username}/films/page/{page}/"
        resp = scraper.get(url, timeout=20)
        if resp.status_code != 200:
            break
        page_entries = _extract_film_page_entries(resp.text)
        if not page_entries:
            break
        for slug, rating_norm in page_entries:
            if len(slug_ratings) >= max_films:
                break
            slug_ratings[slug] = max(slug_ratings.get(slug, 0), rating_norm)
        time.sleep(_REQUEST_DELAY)

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
            except Exception:
                pass

    _save_cache(username, weighted)
    return weighted


def _extract_film_page_entries(html: str) -> list[tuple[str, float]]:
    """Extract (slug, normalised_rating) from /films/page/N/ (<li> grid items)."""
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
    """Thread-safe slug→TMDB resolution using curl_cffi for browser impersonation."""
    with _SLUG_SEM:
        from bs4 import BeautifulSoup

        url = f"{LETTERBOXD_BASE}/film/{slug}/"
        try:
            scraper = _get_scraper()
            resp = scraper.get(url, timeout=20)
            if resp.status_code != 200:
                return slug, None
            soup = BeautifulSoup(resp.text, "html.parser")
            el = soup.select_one('[data-tmdb-id]')
            if el:
                return slug, int(el["data-tmdb-id"])
        except Exception:
            pass
        return slug, None


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


def predict_rating(
    emb_candidate: list[float],
    history: list[tuple[list[float], float]],
    sigma: float = 0.3,
    max_distance: float = 0.3,
) -> float:
    """Predict a user's rating for a candidate movie using Nadaraya-Watson kernel regression.

    Each movie in the user's history contributes to the prediction weighted by
    how similar it is to the candidate (Gaussian kernel over cosine distance).

    Parameters
    ----------
    emb_candidate : list[float]
        Embedding of the candidate movie.
    history : list[tuple[list[float], float]]
        List of (embedding, rating) pairs from the user's watched movies.
    sigma : float
        Bandwidth of the Gaussian kernel. Higher = smoother, lower = spikier.
    max_distance : float
        Maximum cosine distance to consider. Movies farther than this are ignored
        (weight ~0). Set to -1 for no limit.
    """
    total_weight = 0.0
    total_value = 0.0

    for emb, rating in history:
        sim = cosine_similarity(emb_candidate, emb)
        dist = 1.0 - sim
        if max_distance >= 0 and dist > max_distance:
            continue
        weight = math.exp(-(dist ** 2) / (sigma ** 2))
        total_weight += weight
        total_value += weight * rating

    return total_value / total_weight if total_weight > 0 else 0.0


def _predict_rating_batch(
    cand_embs: np.ndarray,
    hist_embs: np.ndarray,
    hist_ratings: np.ndarray,
    sigma: float,
    max_distance: float,
) -> np.ndarray:
    """Vectorized Nadaraya-Watson for M candidates × N history entries."""
    cand_norms = np.linalg.norm(cand_embs, axis=1, keepdims=True)
    hist_norms = np.linalg.norm(hist_embs, axis=1, keepdims=True)
    sims = (cand_embs @ hist_embs.T) / (cand_norms @ hist_norms.T + 1e-12)
    dists = np.clip(1.0 - sims, 0.0, 2.0)

    weights = np.exp(-(dists ** 2) / (sigma ** 2 + 1e-12))
    if max_distance >= 0:
        weights[dists > max_distance] = 0.0

    total_weight = weights.sum(axis=1)
    total_value = weights @ hist_ratings
    return np.where(total_weight > 1e-12, total_value / (total_weight + 1e-12), 0.0)


def _calibrate_sigma_cv(
    history: list[tuple[list[float], float]],
    sigma_candidates: list[float] | None = None,
    n_folds: int = 5,
    max_distance: float = 0.3,
) -> float:
    """Find optimal sigma via k-fold cross-validation (vectorized numpy).

    For each sigma candidate, splits history into folds, predicts held-out
    ratings via kernel regression, and picks the sigma with lowest MAE.
    Falls back to 0.3 if not enough data (<10 movies).
    """
    if not sigma_candidates:
        sigma_candidates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 1.0]

    n = len(history)
    if n < 10:
        return 0.3

    all_embs = np.array([h[0] for h in history])
    all_ratings = np.array([h[1] for h in history], dtype=float)
    indices = np.random.default_rng(42).permutation(n)
    fold_size = max(1, n // n_folds)

    best_sigma = 0.3
    best_mae = float("inf")

    for sigma in sigma_candidates:
        mae_total = 0.0
        count = 0

        for fold_start in range(0, n, fold_size):
            fold_end = min(fold_start + fold_size, n)
            test_idx = indices[fold_start:fold_end]
            train_idx = np.setdiff1d(indices, test_idx)
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            train_embs = all_embs[train_idx]
            train_ratings = all_ratings[train_idx]
            test_embs = all_embs[test_idx]
            test_ratings = all_ratings[test_idx]

            predictions = _predict_rating_batch(
                test_embs, train_embs, train_ratings, sigma, max_distance
            )
            mae_total += float(np.sum(np.abs(predictions - test_ratings)))
            count += len(test_ratings)

        if count > 0:
            mae = mae_total / count
            if mae < best_mae:
                best_mae = mae
                best_sigma = sigma

    return best_sigma


def _load_user_history(
    weighted_ids: dict[int, float],
) -> list[tuple[list[float], float]]:
    """Fetch embeddings from ChromaDB and return [(embedding, rating), ...]."""
    if not weighted_ids:
        return []

    col = get_collection()
    meta_res = col.get(
        ids=[str(tid) for tid in weighted_ids],
        include=["embeddings"],
    )

    history: list[tuple[list[float], float]] = []
    for i, tid_str in enumerate(meta_res["ids"]):
        tid = int(tid_str)
        user_rating = weighted_ids[tid]
        emb = meta_res["embeddings"][i]
        if emb is not None:
            emb_arr = emb.tolist() if hasattr(emb, "tolist") else emb
            history.append((emb_arr, user_rating))

    return history


def _fetch_user_embeddings(
    weighted_ids: dict[int, float],
) -> tuple[list[list[float]], list[float], list[int], int]:
    """Fetch embeddings from ChromaDB for user's watched movies.

    Returns (vectors, weights, found_ids, total_requested).
    """
    if not weighted_ids:
        return [], [], [], 0

    col = get_collection()
    meta_res = col.get(
        ids=[str(tid) for tid in weighted_ids],
        include=["embeddings"],
    )

    vectors: list[list[float]] = []
    weights: list[float] = []
    found: list[int] = []

    for i, tid_str in enumerate(meta_res["ids"]):
        tid = int(tid_str)
        user_weight = weighted_ids[tid]
        emb = meta_res["embeddings"][i]
        if emb is None:
            continue
        emb_arr = emb.tolist() if hasattr(emb, "tolist") else emb
        vectors.append(emb_arr)
        weights.append(user_weight)
        found.append(tid)

    return vectors, weights, found, len(weighted_ids)


def get_taste_vector(
    weighted_ids: dict[int, float],
) -> tuple[list[float], list[int], int]:
    """Weighted-average embedding of ALL watched movies (single vector)."""
    vectors, weights, found, total = _fetch_user_embeddings(weighted_ids)
    if not vectors:
        return [], [], total

    dim = len(vectors[0])
    avg = [0.0] * dim
    total_w = sum(weights)
    for vec, w in zip(vectors, weights):
        for i in range(dim):
            avg[i] += vec[i] * (w / total_w)

    return avg, found, total


def _weighted_kmeans(
    vectors: list[list[float]],
    weights: list[float],
    k: int,
    max_iter: int = 30,
) -> list[tuple[list[float], float, list[int]]]:
    """Simple weighted k-means returning (centroid, total_weight, indices) per cluster."""
    n = len(vectors)
    if n <= k:
        k = max(1, n)
    dim = len(vectors[0])

    # Normalize vectors to unit for centroid assignment
    norms = [math.sqrt(sum(x * x for x in v)) or 1.0 for v in vectors]
    unit_vecs = [[v[d] / norms[i] for d in range(dim)] for i, v in enumerate(vectors)]

    # K-means++ initialization
    import random
    random.seed(42)
    centroids_idx = [random.randrange(0, n)]
    for _ in range(1, k):
        dists = []
        for i in range(n):
            min_d = min(
                sum((unit_vecs[i][d] - unit_vecs[c][d]) ** 2 for d in range(dim))
                for c in centroids_idx
            )
            dists.append(min_d)
        total = sum(dists)
        r = random.random() * total
        cum = 0.0
        for i, d in enumerate(dists):
            cum += d
            if cum >= r:
                centroids_idx.append(i)
                break

    centroids = [unit_vecs[idx][:] for idx in centroids_idx]

    for _ in range(max_iter):
        assignments = [[] for _ in range(k)]
        for i in range(n):
            best = 0
            best_d = sum((unit_vecs[i][d] - centroids[0][d]) ** 2 for d in range(dim))
            for c in range(1, k):
                d = sum((unit_vecs[i][d] - centroids[c][d]) ** 2 for d in range(dim))
                if d < best_d:
                    best_d = d
                    best = c
            assignments[best].append(i)

        new_centroids = []
        for c in range(k):
            if not assignments[c]:
                new_centroids.append(centroids[c][:])
                continue
            total_w = sum(weights[i] for i in assignments[c])
            new_c = [0.0] * dim
            for i in assignments[c]:
                w = weights[i] / total_w
                for d in range(dim):
                    new_c[d] += unit_vecs[i][d] * w
            # Normalize centroid back to unit
            nc_norm = math.sqrt(sum(x * x for x in new_c)) or 1.0
            new_centroids.append([x / nc_norm for x in new_c])

        moved = 0.0
        for c in range(k):
            d = sum((new_centroids[c][d] - centroids[c][d]) ** 2 for d in range(dim))
            moved += d
        centroids = new_centroids
        if moved < 1e-8:
            break

    # Final assignment with raw vectors for output centroids
    assignments = [[] for _ in range(k)]
    for i in range(n):
        best = 0
        best_d = sum((unit_vecs[i][d] - centroids[0][d]) ** 2 for d in range(dim))
        for c in range(1, k):
            d = sum((unit_vecs[i][d] - centroids[c][d]) ** 2 for d in range(dim))
            if d < best_d:
                best_d = d
                best = c
        assignments[best].append(i)

    results = []
    for c in range(k):
        if not assignments[c]:
            continue
        total_w = sum(weights[i] for i in assignments[c])
        centroid = [0.0] * dim
        for i in assignments[c]:
            w = weights[i] / total_w
            for d in range(dim):
                centroid[d] += vectors[i][d] * w
        results.append((centroid, total_w, assignments[c]))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def get_taste_clusters(
    weighted_ids: dict[int, float],
    max_k: int = 4,
) -> tuple[list[tuple[list[float], float]], list[int], int]:
    """Cluster a user's watched movies into K taste groups.

    Returns list of (centroid_vector, cluster_weight) per cluster,
    plus list of found TMDB IDs and total count.
    """
    vectors, weights, found, total = _fetch_user_embeddings(weighted_ids)
    if not vectors:
        return [], [], total

    n = len(vectors)
    if n < 5:
        dim = len(vectors[0])
        centroid = [0.0] * dim
        total_w = sum(weights)
        for vec, w in zip(vectors, weights):
            for d in range(dim):
                centroid[d] += vec[d] * (w / total_w)
        return [(centroid, total_w)], found, total
    elif n < 15:
        total_w = sum(weights)
        result = [(vec, w / total_w) for vec, w in zip(vectors, weights)]
        result.sort(key=lambda x: x[1], reverse=True)
        return result, found, total
    else:
        k = min(max_k, max(2, n // 10))
        clusters = _weighted_kmeans(vectors, weights, k)
        result = [(c, w) for c, w, _ in clusters]
        result.sort(key=lambda x: x[1], reverse=True)
        return result, found, total


def _fetch_embeddings(tmdb_ids: list[int]) -> dict[int, list[float]]:
    """Batch fetch embeddings from ChromaDB for a list of TMDB IDs."""
    col = get_collection()
    res = col.get(ids=[str(tid) for tid in tmdb_ids], include=["embeddings"])
    out: dict[int, list[float]] = {}
    for i, tid_str in enumerate(res["ids"]):
        e = res["embeddings"][i]
        if e is not None:
            out[int(tid_str)] = e.tolist() if hasattr(e, "tolist") else e
    return out


def taste_similarity(
    a_weighted: dict[int, float], b_weighted: dict[int, float]
) -> dict[str, Any]:
    """Measure taste similarity via pairwise movie embedding comparison.

    Instead of comparing averaged vectors (which converge to the global mean
    and inflate similarity), for each movie in A finds its nearest neighbour
    in B (by cosine similarity) and averages those scores.
    """
    found_a = [tid for tid in a_weighted if _get_embedding(tid) is not None]
    found_b = [tid for tid in b_weighted if _get_embedding(tid) is not None]

    if not found_a or not found_b:
        return {
            "similarity": 0.0,
            "error": "Not enough embeddings found for one or both users",
            "user_a_found": len(found_a),
            "user_a_total": len(a_weighted),
            "user_b_found": len(found_b),
            "user_b_total": len(b_weighted),
        }

    embs_a = _fetch_embeddings(found_a)
    embs_b = _fetch_embeddings(found_b)
    list_a = list(embs_a.values())
    list_b = list(embs_b.values())

    if not list_a or not list_b:
        return {"similarity": 0.0, "user_a_found": 0, "user_a_total": 0,
                "user_b_found": 0, "user_b_total": 0}

    def _directional_sim(source: list[list[float]], target: list[list[float]]) -> float:
        """For each vector in source, mean of top-3 nearest neighbours in target."""
        total = 0.0
        for emb in source:
            sims = sorted((cosine_similarity(emb, t) for t in target), reverse=True)
            total += sum(sims[:3]) / 3
        return total / len(source)

    sim_a_to_b = _directional_sim(list_a, list_b)
    sim_b_to_a = _directional_sim(list_b, list_a)
    sim = (sim_a_to_b + sim_b_to_a) / 2

    return {
        "similarity": sim,
        "similarity_pct": round(sim * 100, 1),
        "a_to_b_pct": round(sim_a_to_b * 100, 1),
        "b_to_a_pct": round(sim_b_to_a * 100, 1),
        "user_a_found": len(found_a),
        "user_a_total": len(a_weighted),
        "user_b_found": len(found_b),
        "user_b_total": len(b_weighted),
    }


def _recommend_pipeline(
    user_weighted_list: list[dict[int, float]],
    n: int = 10,
    min_vote_count: int = 500,
    sigma: float | None = None,
    max_distance: float = -1.0,
    diversity: float = 0.0,
    genre_filter: list[str] | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    min_rating: float | None = 6.5,
    randomize: bool = False,
    query_text: str | None = None,
    query_weight: float = 0.0,
) -> list[dict[str, Any]]:
    """Unified recommendation pipeline: profile-based, text-based, or hybrid.

    Handles loading user histories, sigma calibration, multi-seed ChromaDB
    discovery, kernel regression scoring, geometric mean consensus, optional
    query blending, reranking, and MMR diversification.
    """
    exclude: set[int] = set()
    user_histories: list[list[tuple[list[float], float]]] = []

    for wdict in user_weighted_list:
        exclude.update(wdict.keys())
        history = _load_user_history(wdict)
        if history:
            user_histories.append(history)

    if not user_histories:
        if query_text:
            return recommend_from_query(
                query_text, n=n, min_vote_count=min_vote_count,
                diversity=diversity, genre_filter=genre_filter,
                min_year=min_year, max_year=max_year,
                min_rating=min_rating, randomize=randomize,
            )
        return []

    # Auto-calibrate sigma per user if not specified
    sigmas: list[float] = []
    for history in user_histories:
        if sigma is None:
            s = _calibrate_sigma_cv(history, max_distance=max_distance)
            sigmas.append(s)
        else:
            sigmas.append(sigma)
    if sigma is None:
        logger.info("  Auto-calibrated sigmas: %s",
                     ", ".join(f"{s:.2f}" for s in sigmas))

    # Embed query text if provided
    query_vec = None
    query_vec_np = None
    w = max(0.0, min(1.0, query_weight))
    if query_text is not None:
        from src.embedders import OllamaEmbedder
        embedder = OllamaEmbedder()
        query_vec = embedder.embed_batch([query_text])[0]
        query_vec_np = np.array([query_vec])

    # Build discovery seeds: centroids + optional query + random
    discovery_seeds: list[list[float]] = []
    for wdict in user_weighted_list:
        clusters, _, _ = get_taste_clusters(wdict)
        for c, _ in clusters:
            discovery_seeds.append(c)
    if query_vec is not None:
        discovery_seeds.append(query_vec)

    if not discovery_seeds:
        discovery_seeds = [user_histories[0][0][0]]

    dim = len(discovery_seeds[0])
    rng = np.random.default_rng(None if randomize else 42)
    for _ in range(5):
        rand_vec = rng.normal(size=dim).tolist()
        discovery_seeds.append(rand_vec)

    # Pre-convert histories to numpy for batch scoring
    hist_np = [
        (
            np.array([h[0] for h in hist]),
            np.array([h[1] for h in hist], dtype=float),
        )
        for hist in user_histories
    ]

    col = get_collection()
    total_db = col.count()
    pool_per_seed = max(n * 5, 15)
    seen: set[int] = set(exclude)
    batch_meta: list[tuple[int, dict, float, int, float, list[float]]] = []

    multi_results = col.query(
        query_embeddings=discovery_seeds,
        n_results=min(total_db, pool_per_seed),
        include=["embeddings", "metadatas"],
    )

    for q_idx in range(len(discovery_seeds)):
        for j, tid_str in enumerate(multi_results["ids"][q_idx]):
            tid = int(tid_str)
            if tid in seen:
                continue
            seen.add(tid)
            meta = multi_results["metadatas"][q_idx][j]
            cand_emb = multi_results["embeddings"][q_idx][j]
            if cand_emb is None:
                continue
            cand_arr = cand_emb.tolist() if hasattr(cand_emb, "tolist") else cand_emb
            vote_avg, vote_cnt, mc_bonus = _best_vote_data(meta)
            if vote_cnt < min_vote_count:
                continue
            if not _matches_genre_filter(meta.get("genres", ""), genre_filter):
                continue
            if not _matches_year_filter(meta.get("year", 0), min_year, max_year):
                continue
            if min_rating is not None and vote_avg < min_rating:
                continue
            batch_meta.append((tid, meta, vote_avg, vote_cnt, mc_bonus, cand_arr))

    if not batch_meta:
        return []

    # Batch scoring: all candidates × all users at once
    cand_embs = np.array([b[5] for b in batch_meta])
    user_preds = np.array([
        _predict_rating_batch(cand_embs, hembs, hrats, sigmas[j], max_distance)
        for j, (hembs, hrats) in enumerate(hist_np)
    ])

    eps = 1e-6
    product = np.prod(np.maximum(user_preds, eps) ** 2, axis=0)
    consensus = product ** (1.0 / user_preds.shape[0])

    # Query similarity (vectorized cosine) if hybrid
    query_sims = None
    if query_vec_np is not None:
        q_norm = np.linalg.norm(query_vec_np)
        c_norms = np.linalg.norm(cand_embs, axis=1, keepdims=True).flatten()
        query_sims = (cand_embs @ query_vec_np.flatten()) / (c_norms * q_norm + 1e-12)

    candidates: list[dict[str, Any]] = []
    for idx, (tid, meta, vote_avg, vote_cnt, mc_bonus, cand_arr) in enumerate(batch_meta):
        if query_sims is not None:
            sim = (1.0 - w) * float(consensus[idx]) + w * float(query_sims[idx])
        else:
            sim = float(consensus[idx])
        pp = meta.get("poster_path", "")
        poster_url = f"https://image.tmdb.org/t/p/w200{pp}" if pp else ""
        candidates.append({
            "tmdb_id": tid,
            "title": meta.get("title", "Unknown"),
            "genres": meta.get("genres", ""),
            "year": meta.get("year", 0),
            "vote_average": vote_avg,
            "vote_count": vote_cnt,
            "metacritic_bonus": mc_bonus,
            "similarity_score": round(sim, 4),
            "rerank_score": 0.0,
            "url": f"https://letterboxd.com/tmdb/{tid}/",
            "poster": poster_url,
            "_emb": cand_arr,
        })

    reranked = _rerank_candidates(candidates, alpha=0.8)
    if diversity > 0.0:
        return _mmr_diversify(reranked, n, lam=1.0 - diversity)
    return reranked[:n]


def recommend_for_group(
    user_weighted_list: list[dict[int, float]],
    n: int = 10,
    min_vote_count: int = 500,
    sigma: float | None = None,
    max_distance: float = -1.0,
    diversity: float = 0.0,
    genre_filter: list[str] | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    min_rating: float | None = 6.5,
    randomize: bool = False,
) -> list[dict[str, Any]]:
    """Recommend movies for a group via kernel regression + geometric mean."""
    return _recommend_pipeline(
        user_weighted_list, n=n, min_vote_count=min_vote_count,
        sigma=sigma, max_distance=max_distance, diversity=diversity,
        genre_filter=genre_filter, min_year=min_year, max_year=max_year,
        min_rating=min_rating, randomize=randomize,
    )


def _matches_genre_filter(meta_genres: str, genre_filter: list[str] | None) -> bool:
    """Check if meta genres contain any of the requested genres (case-insensitive)."""
    if not genre_filter:
        return True
    meta_lower = meta_genres.lower()
    return any(g.lower() in meta_lower for g in genre_filter)


def _matches_year_filter(year: int, min_year: int | None, max_year: int | None) -> bool:
    if min_year is not None and year < min_year:
        return False
    if max_year is not None and year > max_year:
        return False
    return True


def _best_vote_data(meta: dict) -> tuple[float, int, float]:
    """Return (best_avg, best_count, metacritic_bonus) preferring IMDb over TMDB.

    For the popularity multiplier, uses imdb_rating if available (larger sample),
    falls back to tmdb vote_average. Adds a metacritic signal as bonus.
    """
    imdb_avg = meta.get("imdb_rating")
    imdb_cnt = meta.get("imdb_vote_count")
    tmdb_avg = meta.get("vote_average", 0)
    tmdb_cnt = meta.get("vote_count", 0)

    # IMDb fields may be 0 when not enriched — treat 0 as missing
    avg = imdb_avg if (imdb_avg is not None and imdb_avg > 0) else tmdb_avg
    cnt = imdb_cnt if (imdb_cnt is not None and imdb_cnt > 0) else tmdb_cnt

    mc = meta.get("metacritic_score")
    mcb = 1.0 + max(0, (mc or 50) - 50) / 500.0 if mc is not None else 1.0

    return float(avg), int(cnt), mcb


def _rerank_candidates(
    candidates: list[dict[str, Any]],
    alpha: float = 0.5,
    k: int = 60,
) -> list[dict[str, Any]]:
    """Re-rank candidates balancing similarity and popularity via Reciprocal Rank Fusion.

    Each candidate gets two ranks: by similarity_score and by popularity
    (best_vote_avg * log(best_vote_count+1) * metacritic_bonus). The final
    score is:
        alpha / (k + rank_sim) + (1-alpha) / (k + rank_pop)

    alpha=1 → only similarity, alpha=0 → only popularity.
    k is the RRF constant (default 60, standard in IR).
    """
    if len(candidates) < 2:
        return candidates

    ranked_by_sim = sorted(candidates, key=lambda x: x["similarity_score"], reverse=True)
    sim_rank: dict[int, int] = {
        id(c): i for i, c in enumerate(ranked_by_sim)
    }

    ranked_by_pop = sorted(
        candidates,
        key=lambda x: x["vote_average"] * math.log(x["vote_count"] + 1) * x.get("metacritic_bonus", 1.0),
        reverse=True,
    )
    pop_rank: dict[int, int] = {
        id(c): i for i, c in enumerate(ranked_by_pop)
    }

    for c in candidates:
        r_sim = sim_rank[id(c)]
        r_pop = pop_rank[id(c)]
        rrf_sim = 1.0 / (k + r_sim)
        rrf_pop = 1.0 / (k + r_pop)
        c["rerank_score"] = round(alpha * rrf_sim + (1.0 - alpha) * rrf_pop, 4)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates


def recommend_hybrid(
    user_weighted_list: list[dict[int, float]],
    query_text: str,
    n: int = 10,
    min_vote_count: int = 500,
    query_weight: float = 0.3,
    sigma: float | None = None,
    max_distance: float = -1.0,
    diversity: float = 0.0,
    genre_filter: list[str] | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    min_rating: float | None = 6.5,
    randomize: bool = False,
) -> list[dict[str, Any]]:
    """Profile-based + free-text query combined, with kernel regression + geometric mean.

    query_weight=0 → pure profile consensus, query_weight=1 → pure text search.
    If sigma is None, it is auto-calibrated per user via cross-validation.
    """
    return _recommend_pipeline(
        user_weighted_list, n=n, min_vote_count=min_vote_count,
        sigma=sigma, max_distance=max_distance, diversity=diversity,
        genre_filter=genre_filter, min_year=min_year, max_year=max_year,
        min_rating=min_rating, randomize=randomize,
        query_text=query_text, query_weight=query_weight,
    )


def recommend_from_query(
    query_text: str,
    n: int = 10,
    min_vote_count: int = 500,
    diversity: float = 0.0,
    genre_filter: list[str] | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    min_rating: float | None = 6.5,
    randomize: bool = False,
) -> list[dict[str, Any]]:
    """Embed a free-text query and return top-N similar movies from ChromaDB."""
    from src.embedders import OllamaEmbedder
    embedder = OllamaEmbedder()
    vec = embedder.embed_batch([query_text])[0]

    col = get_collection()
    results = col.query(
        query_embeddings=[vec],
        n_results=min(col.count(), n * 30),
        include=["embeddings", "metadatas", "distances"],
    )

    candidates: list[dict[str, Any]] = []
    for i, tmdb_id_str in enumerate(results["ids"][0]):
        tid = int(tmdb_id_str)
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        similarity = 1.0 - dist  # cosine distance → cosine similarity
        cand_emb = results["embeddings"][0][i]
        cand_arr = cand_emb.tolist() if hasattr(cand_emb, "tolist") else cand_emb

        vote_avg, vote_cnt, mc_bonus = _best_vote_data(meta)
        if vote_cnt < min_vote_count:
            continue
        if not _matches_genre_filter(meta.get("genres", ""), genre_filter):
            continue
        if not _matches_year_filter(meta.get("year", 0), min_year, max_year):
            continue
        if min_rating is not None and vote_avg < min_rating:
            continue

        pp = meta.get("poster_path", "")
        poster_url = f"https://image.tmdb.org/t/p/w200{pp}" if pp else ""
        candidates.append({
            "tmdb_id": tid,
            "title": meta.get("title", "Unknown"),
            "genres": meta.get("genres", ""),
            "year": meta.get("year", 0),
            "vote_average": vote_avg,
            "vote_count": vote_cnt,
            "metacritic_bonus": mc_bonus,
            "similarity_score": round(similarity, 4),
            "rerank_score": 0.0,
            "url": f"https://letterboxd.com/tmdb/{tid}/",
            "poster": poster_url,
            "_emb": cand_arr,
        })

    reranked = _rerank_candidates(candidates, alpha=0.9)
    if diversity > 0.0:
        return _mmr_diversify(reranked, n, lam=1.0 - diversity)
    return reranked[:n]


def _mmr_diversify(
    candidates: list[dict[str, Any]],
    n: int,
    lam: float = 0.5,
) -> list[dict[str, Any]]:
    """Maximum Marginal Relevance — trade off relevance for diversity.

    At each step, picks the candidate with highest:
        lam * rerank_score - (1-lam) * max_sim_to_selected

    lam=1 → pure relevance, lam=0 → pure diversity.
    """
    if not candidates or lam >= 1.0:
        return candidates[:n]

    selected: list[dict[str, Any]] = []
    remaining = list(candidates)

    while len(selected) < n and remaining:
        best_idx = 0
        best_score = -float("inf")
        for i, cand in enumerate(remaining):
            relevance = cand.get("rerank_score", cand.get("similarity_score", 0))
            if not selected:
                mmr = relevance
            else:
                max_sim = max(
                    cosine_similarity(
                        cand.get("_emb", [1.0]),
                        sel.get("_emb", [0.0]),
                    )
                    for sel in selected
                )
                mmr = lam * relevance - (1.0 - lam) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_idx = i
        selected.append(remaining.pop(best_idx))

    return selected
