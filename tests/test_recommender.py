from __future__ import annotations

import math
from datetime import date

import pytest

from src.models import Movie
from src.recommender import _rerank_candidates, _weighted_kmeans, cosine_similarity


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


def test_consensus_scoring_logic() -> None:
    scores = [0.9, 0.8, 0.91]
    mean_sim = sum(scores) / len(scores)
    std_sim = math.sqrt(sum((s - mean_sim) ** 2 for s in scores) / len(scores))
    penalty = 0.5
    consensus = mean_sim - penalty * std_sim

    scores2 = [0.9, 0.3, 0.2]
    mean_sim2 = sum(scores2) / len(scores2)
    std_sim2 = math.sqrt(sum((s - mean_sim2) ** 2 for s in scores2) / len(scores2))
    consensus2 = mean_sim2 - penalty * std_sim2

    assert consensus > consensus2


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


def test_get_taste_clusters_short_history() -> None:
    from src.recommender import get_taste_clusters
    weighted = {1: 1.0, 2: 0.8, 3: 0.5}
    clusters, found, total = get_taste_clusters(weighted)
    assert len(clusters) == 1


def test_get_taste_clusters_empty() -> None:
    from src.recommender import get_taste_clusters
    clusters, found, total = get_taste_clusters({})
    assert clusters == []
    assert found == []
    assert total == 0


def test_calibrate_sigma_cv_fallback() -> None:
    from src.recommender import _calibrate_sigma_cv
    # < 10 movies → fallback to 0.3
    result = _calibrate_sigma_cv([([1.0, 0.0], 0.8), ([0.0, 1.0], 0.5)])
    assert result == 0.3


def test_calibrate_sigma_cv_synthetic() -> None:
    from src.recommender import _calibrate_sigma_cv
    # 20 identical vectors with same rating → low sigma should work fine
    history = [([0.5, 0.5], 0.9) for _ in range(20)]
    result = _calibrate_sigma_cv(history, sigma_candidates=[0.1, 0.3, 0.5])
    assert result in (0.1, 0.3, 0.5)
    assert isinstance(result, float)
    assert 0.0 < result <= 0.5


def test_predict_rating_identical() -> None:
    from src.recommender import predict_rating
    emb = [0.5, 0.5]
    history = [([0.5, 0.5], 0.9)]
    pred = predict_rating(emb, history, sigma=0.3, max_distance=0.3)
    assert pred == pytest.approx(0.9, abs=0.01)


def test_predict_rating_distant_ignored() -> None:
    from src.recommender import predict_rating
    emb = [1.0, 0.0]
    history = [([0.0, 1.0], 0.9)]
    pred = predict_rating(emb, history, sigma=0.3, max_distance=0.3)
    # dist = 1 - 0 = 1.0 > 0.3, should be ignored → fallback to 0.0
    assert pred == 0.0
