#!/usr/bin/env python3
"""Collect movies from TMDB and save to JSONL for later embedding."""
import argparse
import json
import logging
from dataclasses import asdict
from datetime import date
from pathlib import Path

from tqdm import tqdm

from src.collector import fetch_movies_by_year

logger = logging.getLogger(__name__)

DATA_FILE = Path(__file__).parent / "data" / "movies.jsonl"


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect movies from TMDB to JSONL")
    parser.add_argument("--start", type=int, default=1900)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--max-per-year", type=int, default=100)
    parser.add_argument("--test", action="store_true", help="Only 2 years")
    parser.add_argument("--skip-credits", action="store_true", help="Skip TMDB credits call")
    parser.add_argument("--skip-imdb", action="store_true", help="Skip IMDb API enrichment")
    args = parser.parse_args()
    run_collect(args.start, args.end, args.max_per_year, args.test, args.skip_credits, args.skip_imdb)
