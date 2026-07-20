from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.indexer import index_movies
from src.models import Movie


def _make_movie(tmdb_id: int) -> Movie:
    return Movie(
        tmdb_id=tmdb_id,
        title=f"Movie {tmdb_id}",
        overview="A film.",
        release_date=None,
        vote_average=7.0, vote_count=500,
    )


@patch("src.embedders.OllamaEmbedder")
@patch("src.indexer.get_collection")
def test_index_movies_overwrite_calls_upsert(mock_get_collection, mock_embedder_cls) -> None:
    mock_col = MagicMock()
    mock_col.get.return_value = {"ids": ["1"]}  # movie 1 already exists
    mock_get_collection.return_value = mock_col

    mock_embedder = MagicMock()
    mock_embedder.embed_batch.return_value = [[0.1] * 4]
    mock_embedder_cls.return_value = mock_embedder

    movies = [_make_movie(1), _make_movie(2)]

    result = index_movies(movies, overwrite=True)

    assert result == 2  # both indexed via upsert
    mock_col.add.assert_not_called()
    assert mock_col.upsert.call_count == 1


@patch("src.embedders.OllamaEmbedder")
@patch("src.indexer.get_collection")
def test_index_movies_skip_existing_when_not_overwrite(mock_get_collection, mock_embedder_cls) -> None:
    mock_col = MagicMock()
    mock_col.get.return_value = {"ids": ["1"]}  # movie 1 exists
    mock_get_collection.return_value = mock_col

    mock_embedder = MagicMock()
    mock_embedder.embed_batch.return_value = [[0.1] * 4]
    mock_embedder_cls.return_value = mock_embedder

    movies = [_make_movie(1), _make_movie(2)]

    result = index_movies(movies, overwrite=False)

    assert result == 1  # only movie 2 indexed
    mock_col.upsert.assert_not_called()
    assert mock_col.add.call_count == 1
