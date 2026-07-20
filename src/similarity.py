from __future__ import annotations

import math

import numpy as np


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _cosine_similarity_np(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def predict_rating(
    emb_candidate: list[float],
    history: list[tuple[list[float], float]],
    sigma: float = 0.3,
    max_distance: float = 0.3,
) -> float:
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
