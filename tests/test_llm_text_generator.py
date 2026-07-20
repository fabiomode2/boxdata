"""Tests for src/llm_text_generator.py"""
from __future__ import annotations

from src.llm_text_generator import _build_prompt


def test_build_prompt_all_fields() -> None:
    movie = {
        "title": "Inception",
        "release_date": "2010-07-16",
        "director": "Christopher Nolan",
        "cast": ["Leonardo DiCaprio", "Joseph Gordon-Levitt", "Tom Hardy"],
        "genres": ["Action", "Science Fiction"],
        "overview": "A thief who steals corporate secrets through dream-sharing technology.",
        "vote_average": 8.4,
        "vote_count": 35000,
    }
    prompt = _build_prompt(movie)
    assert "Inception" in prompt
    assert "2010" in prompt
    assert "Christopher Nolan" in prompt
    assert "Leonardo DiCaprio" in prompt
    assert "Action" in prompt
    assert "Science Fiction" in prompt
    assert "dream-sharing" in prompt
    assert "8.4" in prompt
    assert "35000" in prompt


def test_build_prompt_missing_director() -> None:
    movie = {
        "title": "Unknown Film",
        "release_date": "2020-01-01",
        "director": "",
        "cast": [],
        "genres": ["Drama"],
        "overview": "A mysterious film.",
        "vote_average": 5.0,
        "vote_count": 100,
    }
    prompt = _build_prompt(movie)
    assert "Unknown" in prompt  # director fallback


def test_build_prompt_no_cast() -> None:
    movie = {
        "title": "Solo Film",
        "release_date": "2015-06-01",
        "director": "Jane Doe",
        "genres": ["Comedy"],
        "overview": "A solo performance.",
        "vote_average": 7.0,
        "vote_count": 500,
    }
    prompt = _build_prompt(movie)
    assert "Solo Film" in prompt
    assert "Jane Doe" in prompt
    assert "Unknown" in prompt  # cast fallback


def test_build_prompt_no_release_date() -> None:
    movie = {
        "title": "Mystery Movie",
        "release_date": None,
        "director": "Unknown",
        "genres": [],
        "overview": "When did it come out?",
        "vote_average": 6.0,
        "vote_count": 200,
    }
    prompt = _build_prompt(movie)
    assert "Mystery Movie" in prompt
    assert "?" in prompt  # year fallback


def test_build_prompt_cast_truncated_to_five() -> None:
    movie = {
        "title": "Ensemble",
        "release_date": "2020-01-01",
        "director": "Director",
        "cast": ["A", "B", "C", "D", "E", "F", "G"],
        "genres": ["Drama"],
        "overview": "Big cast.",
        "vote_average": 7.0,
        "vote_count": 1000,
    }
    prompt = _build_prompt(movie)
    assert "A, B, C, D, E" in prompt
    assert "F" not in prompt  # only top 5
