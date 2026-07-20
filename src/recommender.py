from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from src.pipeline import (
    _fetch_user_embeddings,
    _feature_rerank,
    _load_user_history,
    _matches_genre_filter,
    _matches_year_filter,
    _mmr_diversify,
    _recommend_pipeline,
    _rerank_candidates,
    _tmdb_vote_data,
    recommend_from_query,
    rank_watchlist,
)
from src.scraper import (
    _extract_film_page_entries,
    _resolve_slug_parallel,
    fetch_letterboxd_watched,
)
from src.similarity import (
    _calibrate_sigma_cv,
    _cosine_similarity_np,
    _predict_rating_batch,
    cosine_similarity,
    predict_rating,
)
from src.timers import _timer_start, _timer_stop
from src.vector_store import get_collection

logger = logging.getLogger(__name__)

# Re-export from sub-modules for backward compatibility.
__all__ = [
    # scraper
    "fetch_letterboxd_watched", "_extract_film_page_entries", "_resolve_slug_parallel",
    # similarity
    "cosine_similarity", "_cosine_similarity_np", "predict_rating",
    "_predict_rating_batch", "_calibrate_sigma_cv",
    # pipeline
    "_fetch_user_embeddings", "_load_user_history", "_recommend_pipeline",
    "_feature_rerank", "_rerank_candidates", "_mmr_diversify",
    "recommend_from_query", "rank_watchlist",
    "_tmdb_vote_data", "_matches_genre_filter", "_matches_year_filter",
    # local
    "get_taste_vector", "_weighted_kmeans", "get_taste_clusters",
    "_fetch_embeddings", "taste_similarity",
    "recommend_for_group", "recommend_hybrid",
]


def get_taste_vector(
    weighted_ids: dict[int, float],
) -> tuple[list[float], list[int], int]:
    _timer_start(f"  get_taste_vector ({len(weighted_ids)} ids)")
    vectors, weights, found, total, _ = _fetch_user_embeddings(weighted_ids)
    if not vectors:
        _timer_stop()
        return [], [], total

    dim = len(vectors[0])
    avg = [0.0] * dim
    total_w = sum(weights)
    for vec, w in zip(vectors, weights):
        for i in range(dim):
            avg[i] += vec[i] * (w / total_w)
    _timer_stop()

    return avg, found, total


def _weighted_kmeans(
    vectors: list[list[float]],
    weights: list[float],
    k: int,
    max_iter: int = 30,
) -> list[tuple[list[float], float, list[int]]]:
    n = len(vectors)
    if n <= k:
        k = max(1, n)
    dim = len(vectors[0])

    norms = [math.sqrt(sum(x * x for x in v)) or 1.0 for v in vectors]
    unit_vecs = [[v[d] / norms[i] for d in range(dim)] for i, v in enumerate(vectors)]

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
            nc_norm = math.sqrt(sum(x * x for x in new_c)) or 1.0
            new_centroids.append([x / nc_norm for x in new_c])

        moved = 0.0
        for c in range(k):
            d = sum((new_centroids[c][d] - centroids[c][d]) ** 2 for d in range(dim))
            moved += d
        centroids = new_centroids
        if moved < 1e-8:
            break

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
    vectors, weights, found, total, _ = _fetch_user_embeddings(weighted_ids)
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
    embs_a = _fetch_embeddings(list(a_weighted.keys()))
    embs_b = _fetch_embeddings(list(b_weighted.keys()))
    found_a = list(embs_a.keys())
    found_b = list(embs_b.keys())

    if not found_a or not found_b:
        return {
            "similarity": 0.0,
            "error": "Not enough embeddings found for one or both users",
            "user_a_found": len(found_a),
            "user_a_total": len(a_weighted),
            "user_b_found": len(found_b),
            "user_b_total": len(b_weighted),
        }

    list_a = list(embs_a.values())
    list_b = list(embs_b.values())

    if not list_a or not list_b:
        return {"similarity": 0.0, "user_a_found": 0, "user_a_total": 0,
                "user_b_found": 0, "user_b_total": 0}

    def _directional_sim(source: list[list[float]], target: list[list[float]]) -> float:
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


def recommend_for_group(
    user_weighted_list: list[dict[int, float]],
    n: int = 10,
    min_vote_count: int = 2000,
    sigma: float | None = None,
    max_distance: float = -1.0,
    diversity: float = 0.0,
    genre_filter: list[str] | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    min_rating: float | None = 6.5,
    randomize: bool = False,
    feature_rerank: bool = True,
) -> list[dict[str, Any]]:
    return _recommend_pipeline(
        user_weighted_list, n=n, min_vote_count=min_vote_count,
        sigma=sigma, max_distance=max_distance, diversity=diversity,
        genre_filter=genre_filter, min_year=min_year, max_year=max_year,
        min_rating=min_rating, randomize=randomize,
        feature_rerank=feature_rerank,
    )


def recommend_hybrid(
    user_weighted_list: list[dict[int, float]],
    query_text: str,
    n: int = 10,
    min_vote_count: int = 2000,
    query_weight: float = 0.3,
    sigma: float | None = None,
    max_distance: float = -1.0,
    diversity: float = 0.0,
    genre_filter: list[str] | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    min_rating: float | None = 6.5,
    randomize: bool = False,
    feature_rerank: bool = True,
) -> list[dict[str, Any]]:
    return _recommend_pipeline(
        user_weighted_list, n=n, min_vote_count=min_vote_count,
        sigma=sigma, max_distance=max_distance, diversity=diversity,
        genre_filter=genre_filter, min_year=min_year, max_year=max_year,
        min_rating=min_rating, randomize=randomize,
        query_text=query_text, query_weight=query_weight,
        feature_rerank=feature_rerank,
    )
