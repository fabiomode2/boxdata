from __future__ import annotations

from datetime import date

from src.models import Movie


def test_enriched_text_builds_default() -> None:
    m = Movie(
        tmdb_id=1, title="Test", overview="A test.",
        release_date=date(2020, 1, 1), vote_average=7.0, vote_count=1000,
        genres=["Drama"], director="Alice", cast=["Bob", "Charlie"],
    )
    text = m.enriched_text()
    assert "Test" in text
    assert "2020" in text
    assert "Alice" in text
    assert "Bob" in text
    assert "Drama" in text


def test_enriched_text_prioritizes_llm_text() -> None:
    m = Movie(
        tmdb_id=1, title="Test", overview="A test.",
        release_date=date(2020, 1, 1), vote_average=7.0, vote_count=1000,
        llm_text="Custom LLM generated description.",
    )
    text = m.enriched_text()
    assert text == "Custom LLM generated description."


def test_enriched_text_llm_text_empty_fallback() -> None:
    m = Movie(
        tmdb_id=1, title="Test", overview="A test.",
        release_date=date(2020, 1, 1), vote_average=7.0, vote_count=1000,
        llm_text="",
    )
    text = m.enriched_text()
    assert text != ""
    assert "Test" in text
    assert "2020" in text
