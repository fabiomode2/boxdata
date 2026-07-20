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


def _max_year_in_jsonl() -> int | None:
    if not DATA_FILE.exists():
        return None
    max_y = 0
    with open(DATA_FILE) as f:
        for line in f:
            d = json.loads(line)
            rd = d.get("release_date")
            if rd and len(rd) >= 4:
                try:
                    y = int(rd[:4])
                    if y > max_y:
                        max_y = y
                except ValueError:
                    pass
    return max_y if max_y else None


def run_collect(
    start: int, end: int | None, max_per_year: int, test: bool,
    skip_credits: bool, skip_imdb: bool, skip_reviews: bool = False,
) -> None:
    if end is None:
        end = date.today().year

    collected_ids = load_collected_ids()
    logger.info("Known movies in JSONL: %d", len(collected_ids))

    if not test and collected_ids:
        last_year = _max_year_in_jsonl()
        if last_year is not None and last_year >= start:
            logger.info("Resuming from year %d (was %d)", last_year + 1, start)
            start = last_year + 1

    years = list(range(start, end + 1))
    if test:
        years = years[:2]

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
                enrich_reviews=not skip_reviews,
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
