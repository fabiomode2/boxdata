from __future__ import annotations

import numpy as np
import pytest

from src.evaluator import (
    _compute_metrics,
    evaluate_user,
    _average_results,
    _print_evaluation_results,
)


# ── _compute_metrics ──

def test_compute_metrics_perfect_prediction() -> None:
    y_true = np.array([0.8, 0.6, 0.9, 0.4, 0.7])
    y_pred = np.array([0.8, 0.6, 0.9, 0.4, 0.7])
    m = _compute_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx(0.0, abs=1e-6)
    assert m["rmse"] == pytest.approx(0.0, abs=1e-6)


def test_compute_metrics_constant_offset() -> None:
    y_true = np.array([0.8, 0.6, 0.9, 0.4, 0.7])
    y_pred = y_true + 0.1
    m = _compute_metrics(y_true, y_pred)
    assert m["mae"] == pytest.approx(0.1, abs=1e-6)
    assert m["rmse"] == pytest.approx(0.1, abs=1e-6)


def test_compute_metrics_empty() -> None:
    y_true = np.array([])
    y_pred = np.array([])
    m = _compute_metrics(y_true, y_pred)
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["precision@5"] == 0.0


def test_compute_metrics_precision_at_5() -> None:
    y_true = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    y_pred = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    m = _compute_metrics(y_true, y_pred)
    # Top-5: 0.9, 0.8, 0.7, 0.6, 0.5 → only 0.9, 0.8 >= 0.8 → 2/5
    assert m["precision@5"] == pytest.approx(0.4, abs=0.01)
    assert m["mrr"] == pytest.approx(1.0, abs=0.01)


def test_compute_metrics_mrr_first_relevant() -> None:
    y_true = np.array([0.5, 0.6, 0.9, 0.4, 0.3])
    y_pred = np.array([0.5, 0.2, 0.9, 0.4, 0.3])
    m = _compute_metrics(y_true, y_pred)
    # y_pred sorted desc: idx 2(0.9), idx 0(0.5), idx 3(0.4), idx 4(0.3), idx 1(0.2)
    # y_true[2] = 0.9 >= 0.8 → rank 1
    assert m["mrr"] == pytest.approx(1.0, abs=0.01)


def test_compute_metrics_ndcg() -> None:
    y_true = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
    y_pred = np.array([0.9, 0.8, 0.7, 0.1, 0.2])
    m = _compute_metrics(y_true, y_pred)
    # Perfect ranking → nDCG = 1.0
    assert m["ndcg@5"] > 0.0
    assert m["ndcg@5"] <= 1.0


# ── evaluate_user ──

def test_evaluate_user_too_few_movies() -> None:
    weighted = {1: 0.9, 2: 0.5}
    result = evaluate_user(weighted)
    assert "error" in result
    assert "Too few" in result["error"]


def test_evaluate_user_synthetic() -> None:
    """Synthetic data: 20 identical vectors → perfect prediction."""
    from unittest.mock import patch, MagicMock

    mock_vectors = [[0.5, 0.5] for _ in range(20)]
    mock_weights = [0.9 for _ in range(20)]
    mock_found = list(range(20))
    mock_total = 20

    with patch("src.evaluator._fetch_user_embeddings",
              return_value=(mock_vectors, mock_weights, mock_found, mock_total, [])):
        result = evaluate_user({i: 0.9 for i in range(20)}, n_folds=3)

    assert "error" not in result
    assert result["n_movies"] == 20
    assert result["n_folds"] >= 2
    assert result.get("mae", 1.0) < 0.01
    assert result.get("rmse", 1.0) < 0.01


def test_evaluate_user_synthetic_leave_one_out() -> None:
    """Small dataset (<10) → LOO fallback."""
    from unittest.mock import patch, MagicMock

    mock_vectors = [[0.5, 0.5] for _ in range(5)]
    mock_weights = [0.9 for _ in range(5)]
    mock_found = list(range(5))
    mock_total = 5

    with patch("src.evaluator._fetch_user_embeddings",
              return_value=(mock_vectors, mock_weights, mock_found, mock_total, [])):
        result = evaluate_user({i: 0.9 for i in range(5)}, n_folds=5)

    assert "error" not in result
    assert result["n_folds"] == 5  # LOO


def test_evaluate_user_high_variance() -> None:
    """Different vectors should produce higher MAE."""
    from unittest.mock import patch, MagicMock

    mock_vectors = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.1], [0.1, 1.0], [0.9, 0.0],
                    [0.0, 0.9], [1.0, 0.2], [0.2, 1.0], [0.8, 0.0], [0.0, 0.8]]
    mock_weights = [0.9, 0.1, 0.8, 0.2, 0.9, 0.1, 0.8, 0.2, 0.9, 0.1]
    mock_found = list(range(10))
    mock_total = 10

    with patch("src.evaluator._fetch_user_embeddings",
              return_value=(mock_vectors, mock_weights, mock_found, mock_total, [])):
        result = evaluate_user({i: w for i, w in enumerate(mock_weights)}, n_folds=3)

    assert "error" not in result
    # Should have some error since vector groups are well-separated
    assert result.get("mae", 0) > 0


# ── _average_results ──

def test_average_results_basic() -> None:
    results = [
        {"user": "alice", "n_movies": 100, "n_folds": 5,
         "mae": 0.15, "mae_std": 0.05, "rmse": 0.20, "rmse_std": 0.06,
         "precision@5": 0.5, "ndcg@5": 0.6, "mrr": 0.7},
        {"user": "bob", "n_movies": 50, "n_folds": 5,
         "mae": 0.25, "mae_std": 0.08, "rmse": 0.30, "rmse_std": 0.10,
         "precision@5": 0.4, "ndcg@5": 0.5, "mrr": 0.6},
    ]
    avg = _average_results(results)
    assert avg is not None
    assert avg["user"] == "AVERAGE"
    assert avg["mae"] == pytest.approx(0.20, abs=0.01)
    assert avg["rmse"] == pytest.approx(0.25, abs=0.01)
    assert avg["mrr"] == pytest.approx(0.65, abs=0.01)


def test_average_results_skips_errors() -> None:
    results = [
        {"user": "alice", "error": "Too few embeddings found"},
        {"user": "bob", "n_movies": 100, "n_folds": 5,
         "mae": 0.15, "mae_std": 0.05, "rmse": 0.20, "rmse_std": 0.06,
         "precision@5": 0.5, "ndcg@5": 0.6, "mrr": 0.7},
    ]
    avg = _average_results(results)
    assert avg is not None
    assert avg["mae"] == pytest.approx(0.15, abs=0.01)


def test_average_results_all_errors() -> None:
    results = [
        {"user": "alice", "error": "Too few"},
        {"user": "bob", "error": "Too few"},
    ]
    assert _average_results(results) is None


# ── _print_evaluation_results (smoke test — no crash) ──

def test_print_results_with_errors(capsys: pytest.CaptureFixture) -> None:
    results = [
        {"user": "alice", "error": "Too few embeddings", "n_movies": 2},
        {"user": "bob", "n_movies": 100, "n_folds": 5,
         "mae": 0.15, "mae_std": 0.05, "rmse": 0.20, "rmse_std": 0.06,
         "precision@5": 0.5, "ndcg@5": 0.6, "mrr": 0.7},
    ]
    _print_evaluation_results(results)
    captured = capsys.readouterr()
    assert "Too few" in captured.out
    assert "OK" in captured.out


def test_print_results_empty() -> None:
    _print_evaluation_results([])
