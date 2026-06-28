from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.recommender import (
    fetch_letterboxd_watched,
    get_taste_vector,
    recommend_for_group,
    recommend_from_query,
    recommend_hybrid,
)
from src.profile import parse_export, resolve_tmdb_ids
from src.vector_store import count_movies

logger = logging.getLogger(__name__)

app = FastAPI(title="BoxData API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendRequest(BaseModel):
    users: list[str] = []
    query: str | None = None
    n: int = 10
    diversity: float = 0.0
    genre: list[str] | None = None
    query_weight: float = 0.3
    min_year: int | None = None
    max_year: int | None = None
    min_rating: float = 6.5
    randomize: bool = False

class MovieResult(BaseModel):
    tmdb_id: int
    title: str
    year: int
    genres: str
    vote_average: float
    rerank_score: float
    url: str
    poster: str | None = None

@app.get("/health")
async def health():
    return {"status": "ok", "movies": count_movies()}

@app.post("/api/recommend", response_model=list[MovieResult])
async def recommend(req: RecommendRequest):
    user_weighted: list[dict[int, float]] = []

    if req.users:
        for username in req.users:
            weighted = fetch_letterboxd_watched(username)
            vec, found_ids, total = get_taste_vector(weighted)
            if found_ids:
                user_weighted.append(weighted)

    genre_filter = [g.strip() for g in req.genre if g.strip()] if req.genre else None

    if not user_weighted and req.query:
        recs = recommend_from_query(
            req.query, n=req.n, diversity=req.diversity,
            genre_filter=genre_filter, min_year=req.min_year,
            max_year=req.max_year, min_rating=req.min_rating,
            randomize=req.randomize,
        )
    elif user_weighted and req.query:
        recs = recommend_hybrid(
            user_weighted, req.query, n=req.n,
            query_weight=req.query_weight, diversity=req.diversity,
            genre_filter=genre_filter, min_year=req.min_year,
            max_year=req.max_year, min_rating=req.min_rating,
            randomize=req.randomize,
        )
    elif user_weighted:
        recs = recommend_for_group(
            user_weighted, n=req.n, diversity=req.diversity,
            genre_filter=genre_filter, min_year=req.min_year,
            max_year=req.max_year, min_rating=req.min_rating,
            randomize=req.randomize,
        )
    else:
        return []

    return [
        MovieResult(
            tmdb_id=r["tmdb_id"],
            title=r["title"],
            year=r["year"],
            genres=r.get("genres", ""),
            vote_average=r.get("vote_average", 0),
            rerank_score=r.get("rerank_score", 0),
            url=r.get("url", ""),
            poster=r.get("poster"),
        )
        for r in recs
    ]
