from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from src.models import Movie
from src.recommender import (
    _predict_rating_batch,
    _rerank_candidates,
    _weighted_kmeans,
    _mmr_diversify,
    cosine_similarity,
    predict_rating,
    _calibrate_sigma_cv,
    _feature_rerank,
)


# ── Cosine Similarity ──

def test_cosine_similarity_parallel() -> None:
    a = [1.0, 0.0, 0.0]
    b = [2.0, 0.0, 0.0]
    assert cosine_similarity(a, b) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal() -> None:
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite() -> None:
    a = [1.0, 0.0, 0.0]
    b = [-1.0, 0.0, 0.0]
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_zero_norm() -> None:
    a = [0.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    assert cosine_similarity(a, b) == 0.0
    assert cosine_similarity(b, a) == 0.0


def test_cosine_similarity_arbitrary() -> None:
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    dot = 4 + 10 + 18
    na = math.sqrt(1 + 4 + 9)
    nb = math.sqrt(16 + 25 + 36)
    expected = dot / (na * nb)
    assert cosine_similarity(a, b) == pytest.approx(expected)


# ── RRF Reranking ──

def test_rerank_rrf_single_candidate() -> None:
    candidates = [{"similarity_score": 0.9, "vote_average": 8.0, "vote_count": 1000}]
    result = _rerank_candidates(candidates)
    assert len(result) == 1


def test_rerank_rrf_two_candidates() -> None:
    candidates = [
        {"similarity_score": 0.9, "vote_average": 8.0, "vote_count": 1000},
        {"similarity_score": 0.5, "vote_average": 6.0, "vote_count": 100},
    ]
    result = _rerank_candidates(candidates, alpha=1.0)
    assert result[0]["similarity_score"] == 0.9


def test_rerank_rrf_popularity_wins() -> None:
    candidates = [
        {"similarity_score": 0.9, "vote_average": 5.0, "vote_count": 10},
        {"similarity_score": 0.5, "vote_average": 9.0, "vote_count": 10000},
    ]
    result = _rerank_candidates(candidates, alpha=0.0)
    assert result[0]["vote_average"] == 9.0


def test_rerank_rrf_alpha_blend() -> None:
    candidates = [
        {"similarity_score": 0.9, "vote_average": 5.0, "vote_count": 10},
        {"similarity_score": 0.5, "vote_average": 9.0, "vote_count": 10000},
    ]
    pure_sim = _rerank_candidates(list(candidates), alpha=1.0)
    pure_pop = _rerank_candidates(list(candidates), alpha=0.0)
    balanced = _rerank_candidates(list(candidates), alpha=0.5)

    assert pure_sim[0]["similarity_score"] == 0.9
    assert pure_pop[0]["vote_average"] == 9.0


# ── Weighted K-Means ──

def test_weighted_kmeans_single_cluster() -> None:
    vectors = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    weights = [1.0, 1.0, 1.0]
    clusters = _weighted_kmeans(vectors, weights, k=1)
    assert len(clusters) == 1
    centroid, total_w, indices = clusters[0]
    assert len(indices) == 3
    assert total_w == pytest.approx(3.0)


def test_weighted_kmeans_k_equals_n() -> None:
    vectors = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    weights = [1.0, 1.0, 1.0]
    clusters = _weighted_kmeans(vectors, weights, k=3)
    assert len(clusters) == 3


def test_weighted_kmeans_k_greater_than_n() -> None:
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    weights = [1.0, 1.0]
    clusters = _weighted_kmeans(vectors, weights, k=10)
    assert len(clusters) == 2


# ── Enriched Text ──

def test_enriched_text_basic() -> None:
    m = Movie(
        tmdb_id=1,
        title="Test Movie",
        overview="A test movie.",
        release_date=date(2020, 6, 15),
        vote_average=7.5,
        vote_count=500,
        genres=["Action", "Drama"],
    )
    text = m.enriched_text()
    assert "Test Movie is a 2020 film." in text
    assert "Genres: Action, Drama." in text
    assert "well-received" in text
    assert "limited audience" in text


def test_enriched_text_no_genres_no_director() -> None:
    m = Movie(
        tmdb_id=2,
        title="Silent Film",
        overview="",
        release_date=None,
        vote_average=9.0,
        vote_count=20000,
    )
    text = m.enriched_text()
    assert "Silent Film is a  film." in text
    assert "widely acclaimed" in text
    assert "large audience" in text


def test_enriched_text_with_cast_and_director() -> None:
    m = Movie(
        tmdb_id=3,
        title="Directed Film",
        overview="A story.",
        release_date=date(1999, 1, 1),
        vote_average=4.5,
        vote_count=50,
        genres=["Comedy"],
        director="Alice Smith",
        cast=["Bob", "Carol", "Dave"],
    )
    text = m.enriched_text()
    assert "Directed by Alice Smith." in text
    assert "Starring Bob, Carol, Dave." in text
    assert "mixed" in text
    assert "limited audience" in text


def test_enriched_text_low_rating() -> None:
    m = Movie(
        tmdb_id=4,
        title="Bad Movie",
        overview="Terrible.",
        release_date=date(2023, 1, 1),
        vote_average=3.0,
        vote_count=100,
    )
    text = m.enriched_text()
    assert "poorly received" in text


def test_enriched_text_no_votes() -> None:
    m = Movie(
        tmdb_id=5,
        title="No Votes",
        overview="Nobody rated this.",
        release_date=date(2024, 1, 1),
        vote_average=0.0,
        vote_count=0,
    )
    text = m.enriched_text()
    assert "No Votes is a 2024 film." in text
    assert "Nobody rated this." in text
    assert "widely acclaimed" not in text
    assert "audience" not in text


# ── Geometric Mean Consensus ──

def test_geometric_mean_consensus() -> None:
    """Test that geometric mean rewards agreement and penalises divergence."""
    # High agreement: both users rate highly
    product_agree = np.prod(np.maximum(np.array([[0.9], [0.8]]), 1e-6) ** 2, axis=0)
    consensus_agree = product_agree ** (1.0 / 2)

    # High disagreement: one loves, one hates
    product_disagree = np.prod(np.maximum(np.array([[0.9], [0.1]]), 1e-6) ** 2, axis=0)
    consensus_disagree = product_disagree ** (1.0 / 2)

    assert consensus_agree[0] > consensus_disagree[0]


# ── Sigma Calibration ──

def test_calibrate_sigma_cv_fallback() -> None:
    # < 10 movies → fallback to 0.3
    result = _calibrate_sigma_cv([([1.0, 0.0], 0.8), ([0.0, 1.0], 0.5)])
    assert result == 0.3


def test_calibrate_sigma_cv_synthetic() -> None:
    # 20 identical vectors with same rating → low sigma should work fine
    history = [([0.5, 0.5], 0.9) for _ in range(20)]
    result = _calibrate_sigma_cv(history, sigma_candidates=[0.1, 0.3, 0.5])
    assert result in (0.1, 0.3, 0.5)
    assert isinstance(result, float)
    assert 0.0 < result <= 0.5


# ── Nadaraya-Watson Kernel Regression ──

def test_predict_rating_identical() -> None:
    emb = [0.5, 0.5]
    history = [([0.5, 0.5], 0.9)]
    pred = predict_rating(emb, history, sigma=0.3, max_distance=0.3)
    assert pred == pytest.approx(0.9, abs=0.01)


def test_predict_rating_distant_ignored() -> None:
    emb = [1.0, 0.0]
    history = [([0.0, 1.0], 0.9)]
    pred = predict_rating(emb, history, sigma=0.3, max_distance=0.3)
    # dist = 1 - 0 = 1.0 > 0.3, should be ignored → fallback to 0.0
    assert pred == 0.0


def test_predict_rating_batch_matches_single() -> None:
    """Batch version should match single version for the same inputs."""
    cand_embs = np.array([[0.5, 0.5]])
    hist_embs = np.array([[0.5, 0.5]])
    hist_ratings = np.array([0.9])
    batch_preds = _predict_rating_batch(cand_embs, hist_embs, hist_ratings, sigma=0.3, max_distance=0.3)

    single_pred = predict_rating([0.5, 0.5], [([0.5, 0.5], 0.9)], sigma=0.3, max_distance=0.3)
    assert float(batch_preds[0]) == pytest.approx(single_pred, abs=0.01)


def test_predict_rating_batch_multi_candidate() -> None:
    """Batch version should produce correct predictions for multiple candidates."""
    cand_embs = np.array([[1.0, 0.0], [0.0, 1.0]])
    hist_embs = np.array([[1.0, 0.0]])
    hist_ratings = np.array([0.8])
    preds = _predict_rating_batch(cand_embs, hist_embs, hist_ratings, sigma=0.3, max_distance=0.3)
    # First candidate is identical → close to 0.8
    assert float(preds[0]) == pytest.approx(0.8, abs=0.02)
    # Second is orthogonal → very low weight → close to 0
    assert float(preds[1]) < 0.1


# ── MMR Diversity ──

def test_mmr_pure_relevance() -> None:
    cands = [
        {"rerank_score": 0.9, "similarity_score": 0.9, "_emb": [1.0, 0.0]},
        {"rerank_score": 0.5, "similarity_score": 0.5, "_emb": [0.0, 1.0]},
    ]
    result = _mmr_diversify(cands, n=2, lam=1.0)
    assert len(result) == 2
    assert result[0]["rerank_score"] == 0.9


def test_mmr_pure_diversity() -> None:
    cands = [
        {"rerank_score": 0.9, "_emb": [1.0, 0.0]},
        {"rerank_score": 0.5, "_emb": [1.0, 0.01]},
        {"rerank_score": 0.3, "_emb": [0.0, 1.0]},
    ]
    result = _mmr_diversify(cands, n=2, lam=0.0)
    # lam=0 means pure diversity: picks most different
    assert len(result) == 2


def test_mmr_empty_candidates() -> None:
    assert _mmr_diversify([], n=5) == []


# ── Matches helpers ──

def test_matches_year_filter() -> None:
    from src.recommender import _matches_year_filter
    assert _matches_year_filter(2000, 1990, 2010) is True
    assert _matches_year_filter(1980, 1990, None) is False
    assert _matches_year_filter(2020, None, 2010) is False
    assert _matches_year_filter(2000, None, None) is True
    # Edge case: min_year=0 should filter correctly
    assert _matches_year_filter(0, 0, None) is True
    assert _matches_year_filter(-1, 0, None) is False


def test_matches_genre_filter() -> None:
    from src.recommender import _matches_genre_filter
    assert _matches_genre_filter("Action, Drama", ["Action"]) is True
    assert _matches_genre_filter("Comedy", ["Action"]) is False
    assert _matches_genre_filter("Action", None) is True
    assert _matches_genre_filter("", ["Action"]) is False
    assert _matches_genre_filter("Science Fiction", ["science fiction"]) is True


def test_tmdb_vote_data_uses_tmdb_only() -> None:
    from src.recommender import _tmdb_vote_data
    meta = {
        "vote_average": 6.0, "vote_count": 100,
        "imdb_rating": 8.0, "imdb_vote_count": 50000,
        "metacritic_score": 80,
    }
    avg, cnt, mc_bonus = _tmdb_vote_data(meta)
    assert avg == 6.0  # uses TMDB, not IMDb
    assert cnt == 100
    assert mc_bonus > 1.0  # metacritic > 50


def test_tmdb_vote_data_handles_empty_meta() -> None:
    from src.recommender import _tmdb_vote_data
    meta: dict = {}
    avg, cnt, mc_bonus = _tmdb_vote_data(meta)
    assert avg == 0.0
    assert cnt == 0
    assert mc_bonus == 1.0


# ── Feature Reranking ──

def test_feature_rerank_director_bonus() -> None:
    """Director match gives +10% bonus."""
    from unittest.mock import patch, MagicMock

    mock_col = MagicMock()
    mock_col.get.return_value = {
        "ids": ["1"],
        "metadatas": [{"director": "christopher nolan", "year": 2010, "cast": ""}],
    }

    candidates = [
        {"tmdb_id": 100, "director": "christopher nolan", "year": 2014, "cast": "",
         "genres": "Sci-Fi", "rerank_score": 1.0},
    ]
    user_weighted_list = [{1: 0.9}]

    with patch("src.recommender.get_collection", return_value=mock_col):
        result = _feature_rerank(candidates, user_weighted_list)

    assert result[0]["rerank_score"] == pytest.approx(1.10, abs=0.01)


def test_feature_rerank_decade_bonus() -> None:
    """Decade match gives +5% bonus."""
    from unittest.mock import patch, MagicMock

    mock_col = MagicMock()
    mock_col.get.return_value = {
        "ids": ["1"],
        "metadatas": [{"director": "someone", "year": 2012, "cast": ""}],
    }

    candidates = [
        {"tmdb_id": 100, "director": "other", "year": 2015, "cast": "",
         "genres": "", "rerank_score": 1.0},
    ]
    user_weighted_list = [{1: 0.8}]

    with patch("src.recommender.get_collection", return_value=mock_col):
        result = _feature_rerank(candidates, user_weighted_list)

    assert result[0]["rerank_score"] == pytest.approx(1.05, abs=0.01)


def test_feature_rerank_actor_jaccard() -> None:
    """Actor overlap uses Jaccard similarity, capped contribution."""
    from unittest.mock import patch, MagicMock

    mock_col = MagicMock()
    mock_col.get.return_value = {
        "ids": ["1"],
        "metadatas": [{"director": "x", "year": 1995, "cast": "alice, bob, charlie"}],
    }

    candidates = [
        {"tmdb_id": 100, "director": "y", "year": 2015, "cast": "alice, bob, dave",
         "genres": "", "rerank_score": 1.0},
    ]
    user_weighted_list = [{1: 0.7}]

    with patch("src.recommender.get_collection", return_value=mock_col):
        result = _feature_rerank(candidates, user_weighted_list)

    # Decades: 1990 vs 2010 → no match
    # Director: x vs y → no match
    # Jaccard({alice, bob, dave}, {alice, bob, charlie}) = 2/4 = 0.5
    # Actor bonus = 0.05 * 0.5 = 0.025
    assert result[0]["rerank_score"] == pytest.approx(1.025, abs=0.01)


def test_feature_rerank_cap_10pct() -> None:
    """Total bonus capped at 10% even when multiple features match."""
    from unittest.mock import patch, MagicMock

    mock_col = MagicMock()
    mock_col.get.return_value = {
        "ids": ["1"],
        "metadatas": [{"director": "nolan", "year": 2010, "cast": "leo, tom"}],
    }

    candidates = [
        {"tmdb_id": 100, "director": "nolan", "year": 2014, "cast": "leo, matt",
         "genres": "", "rerank_score": 1.0},
    ]
    user_weighted_list = [{1: 0.9}]

    with patch("src.recommender.get_collection", return_value=mock_col):
        result = _feature_rerank(candidates, user_weighted_list)

    # director: +0.10, decade: +0.05, actors: +some Jaccard
    # All combined capped at 0.10
    assert result[0]["rerank_score"] == pytest.approx(1.10, abs=0.01)


def test_feature_rerank_empty_inputs() -> None:
    """Empty candidates or empty user list returns unchanged."""
    assert _feature_rerank([], [{1: 0.5}]) == []
    assert _feature_rerank([{"rerank_score": 1.0}], []) == [{"rerank_score": 1.0}]
