"""Fetch top reviews from TMDB for a movie."""

import logging

from src.collector import tmdb_get
from src.models import Movie

logger = logging.getLogger(__name__)

_MAX_REVIEW_LENGTH = 500


def fetch_top_reviews(tmdb_id: int, max_reviews: int = 3) -> list[str]:
    """Fetch top N reviews from TMDB.

    Each review is truncated to 500 chars. Returns empty list if none found.
    """
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}/reviews"
    try:
        data = tmdb_get(url)
    except Exception as e:
        logger.debug("Review fetch failed for TMDB %d: %s", tmdb_id, e)
        return []
    results = data.get("results", [])
    reviews = []
    for r in results[:max_reviews]:
        content = r.get("content", "").strip()
        if not content:
            continue
        if len(content) > _MAX_REVIEW_LENGTH:
            content = content[:_MAX_REVIEW_LENGTH] + "..."
        reviews.append(content)
    return reviews


def enrich_movie_with_reviews(movie: Movie) -> None:
    """Fetch and attach top reviews to a Movie object in-place."""
    movie.top_reviews = fetch_top_reviews(movie.tmdb_id)
