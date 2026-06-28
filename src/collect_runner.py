#!/usr/bin/env python3
"""Collect movies from TMDB and save to JSONL for later embedding."""
import json
import logging
from dataclasses import asdict
from datetime import date
from pathlib import Path

from tqdm import tqdm

from src.collector import fetch_movies_by_year

logger = logging.getLogger(__name__)

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "movies.jsonl"


def load_collected_ids() -> set[int]:
    if not DATA_FILE.exists():
        return set()
    with open(DATA_FILE) as f:
        return {json.loads(line)["tmdb_id"] for line in f}


def run_collect(
    start: int, end: int | None, max_per_year: int, test: bool,
    skip_credits: bool, skip_imdb: bool,
) -> None:
    if end is None:
        end = date.today().year

    years = list(range(start, end + 1))
    if test:
        years = years[:2]

    collected_ids = load_collected_ids()
    logger.info("Known movies in JSONL: %d", len(collected_ids))

    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    total_new = 0

    with open(DATA_FILE, "a") as f:
        for year in tqdm(years, desc="Collecting"):
            movies = fetch_movies_by_year(
                year,
                max_per_year=max_per_year,
                skip_credits=skip_credits,
                seen_ids=collected_ids,
                enrich=not skip_imdb,
            )
            for movie in movies:
                d = asdict(movie)
                d["release_date"] = str(movie.release_date) if movie.release_date else None
                d["director"] = movie.director or ""
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                collected_ids.add(movie.tmdb_id)
            total_new += len(movies)
            if movies:
                tqdm.write(f"  {year}: +{len(movies)}")

    logger.info("Done. +%d new. Total in JSONL: %d", total_new, len(collected_ids))
