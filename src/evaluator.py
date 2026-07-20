from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.recommender import _fetch_user_embeddings, _predict_rating_batch

logger = logging.getLogger(__name__)


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """MAE, RMSE, Precision@5, nDCG@5, MRR from true vs predicted ratings."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    k = min(5, len(y_true))
    if k == 0 or np.isnan(mae):
        return {"mae": 0.0, "rmse": 0.0, "precision@5": 0.0, "ndcg@5": 0.0, "mrr": 0.0}

    top_k = np.argsort(y_pred)[-k:][::-1]
    relevant = y_true[top_k] >= 0.8

    prec = float(np.mean(relevant))

    dcg = sum((2 ** y_true[top_k[i]] - 1) / np.log2(i + 2) for i in range(k))
    ideal = np.sort(y_true)[-k:][::-1]
    idcg = sum((2 ** ideal[i] - 1) / np.log2(i + 2) for i in range(k))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    order = np.argsort(y_pred)[::-1]
    mrr = 0.0
    for rank, idx in enumerate(order, 1):
        if y_true[idx] >= 0.8:
            mrr = 1.0 / rank
            break

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "precision@5": round(prec, 4),
        "ndcg@5": round(ndcg, 4),
        "mrr": round(mrr, 4),
    }


def evaluate_user(
    weighted_ids: dict[int, float],
    n_folds: int = 5,
    sigma: float = 0.3,
    max_distance: float = -1.0,
    random_seed: int = 42,
    embedding_file: str | None = None,
) -> dict[str, Any]:
    """k-fold CV evaluation of rating prediction for one user.

    Splits user history into folds, predicts held-out ratings via
    Nadaraya-Watson kernel regression on training fold.

    Returns MAE, RMSE, Precision@5, nDCG@5, MRR (mean + std across folds).
    """
    vectors, weights, found_ids, total = _fetch_user_embeddings(weighted_ids)

    if len(vectors) < 3:
        return {
            "error": "Too few embeddings found (<3)",
            "found": len(vectors),
            "total": total,
        }

    n = len(vectors)
    effective_folds = min(n_folds, n) if n >= 10 else n  # LOO for <10

    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(n)
    fold_size = max(1, n // effective_folds)

    all_embs = np.array(vectors)
    all_ratings = np.array(weights, dtype=float)

    fold_metrics: list[dict[str, float]] = []

    for fold_start in range(0, n, fold_size):
        fold_end = min(fold_start + fold_size, n)
        test_idx = indices[fold_start:fold_end]
        train_idx = np.setdiff1d(indices, test_idx)

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        predictions = _predict_rating_batch(
            all_embs[test_idx], all_embs[train_idx], all_ratings[train_idx],
            sigma, max_distance,
        )

        fold_metrics.append(_compute_metrics(all_ratings[test_idx], predictions))

    if not fold_metrics:
        return {"error": "No folds could be created", "n_movies": n}

    result: dict[str, Any] = {
        "n_movies": n,
        "n_folds": len(fold_metrics),
        "found": len(vectors),
        "total": total,
    }

    for key in fold_metrics[0]:
        values = [m[key] for m in fold_metrics]
        result[key] = round(float(np.mean(values)), 4)
        result[f"{key}_std"] = round(float(np.std(values)), 4)

    return result


def evaluate_users(
    user_weighted_list: list[dict[int, float]],
    usernames: list[str] | None = None,
    n_folds: int = 5,
    sigma: float = 0.3,
    max_distance: float = -1.0,
    random_seed: int = 42,
    embedding_file: str | None = None,
) -> list[dict[str, Any]]:
    """Evaluate rating prediction for multiple users."""
    results: list[dict[str, Any]] = []
    for i, wdict in enumerate(user_weighted_list):
        label = usernames[i] if usernames and i < len(usernames) else f"user_{i}"
        logger.info("Evaluating user '%s' (%d movies)...", label, len(wdict))
        result = evaluate_user(
            wdict, n_folds=n_folds, sigma=sigma, max_distance=max_distance,
            random_seed=random_seed, embedding_file=embedding_file,
        )
        result["user"] = label
        results.append(result)
        if "error" not in result:
            logger.info(
                "  MAE=%.4f(±%.4f)  RMSE=%.4f(±%.4f)  P@5=%.4f  nDCG@5=%.4f  MRR=%.4f",
                result["mae"], result["mae_std"],
                result["rmse"], result["rmse_std"],
                result.get("precision@5", 0),
                result.get("ndcg@5", 0),
                result.get("mrr", 0),
            )
        else:
            logger.warning("  %s", result["error"])

    return results


def _print_evaluation_results(results: list[dict[str, Any]]) -> None:
    """Print evaluation results table."""
    if not results:
        return

    has_error = any("error" in r for r in results)
    if has_error:
        header = f"{'User':16s}  {'Movies':>6s}  {'Folds':>5s}  {'Status'}"
        sep = "-" * 60
        lines = [header, sep]
        for r in results:
            if "error" in r:
                lines.append(f"{r['user']:16s}  {r.get('n_movies', 0):>6d}  {r.get('n_folds', 0):>5d}  {r['error']}")
            else:
                lines.append(f"{r['user']:16s}  {r['n_movies']:>6d}  {r['n_folds']:>5d}  OK")
        lines.append("")
        print("\n".join(lines))
        return

    header = (
        f"{'User':16s}  {'#':>4s}  {'Folds':>5s}  "
        f"{'MAE':>7s}  {'MAEσ':>6s}  "
        f"{'RMSE':>7s}  {'RMSEσ':>6s}  "
        f"{'P@5':>6s}  {'nDCG@5':>7s}  {'MRR':>6s}"
    )
    sep = "-" * 85
    lines = [header, sep]
    for r in results:
        lines.append(
            f"{r['user']:16s}  {r['n_movies']:>4d}  {r['n_folds']:>5d}  "
            f"{r['mae']:>7.4f}  {r['mae_std']:>6.4f}  "
            f"{r['rmse']:>7.4f}  {r['rmse_std']:>6.4f}  "
            f"{r.get('precision@5', 0):>6.4f}  "
            f"{r.get('ndcg@5', 0):>7.4f}  "
            f"{r.get('mrr', 0):>6.4f}"
        )
    lines.append("")
    print("\n".join(lines))


def _average_results(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Compute macro-average across users (skipping errors)."""
    valid = [r for r in results if "error" not in r]
    if not valid:
        return None

    avg: dict[str, Any] = {"user": "AVERAGE"}
    for key in ("n_movies", "n_folds"):
        avg[key] = int(round(float(np.mean([r[key] for r in valid]))))

    for key in ("mae", "mae_std", "rmse", "rmse_std", "precision@5", "ndcg@5", "mrr"):
        if key in valid[0]:
            avg[key] = round(float(np.mean([r[key] for r in valid])), 4)

    return avg
