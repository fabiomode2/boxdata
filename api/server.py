from __future__ import annotations

import logging
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import Depends, FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

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

API_KEY = os.getenv("API_KEY")


async def verify_api_key(x_api_key: str | None = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


app = FastAPI(title="BoxData API")

origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "https://boxdata.vercel.app").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendRequest(BaseModel):
    users: list[str] = Field(default=[], max_length=5)
    query: str | None = None
    n: int = Field(default=10, ge=1, le=50)
    min_vote_count: int = Field(default=2000, ge=0, le=500000)
    diversity: float = Field(default=0.0, ge=0.0, le=1.0)
    genre: list[str] | None = None
    query_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    min_year: int | None = None
    max_year: int | None = None
    min_rating: float = Field(default=6.5, ge=0.0, le=10.0)
    randomize: bool = False
    refresh: bool = False

    @field_validator("users")
    @classmethod
    def validate_usernames(cls, users: list[str]) -> list[str]:
        for u in users:
            if not re.match(r"^[a-zA-Z0-9_-]{1,50}$", u):
                raise ValueError(f"Invalid letterboxd username: {u}")
        return users

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
async def recommend(req: RecommendRequest, _auth=Depends(verify_api_key)):
    t0 = time.perf_counter()
    users_str = ",".join(req.users) if req.users else "query-only"
    print(f"\n--- /api/recommend [{users_str}] n={req.n} refresh={req.refresh} ---", file=sys.stderr, flush=True)

    user_weighted: list[dict[int, float]] = []

    if req.users:
        for username in req.users:
            weighted = fetch_letterboxd_watched(username, no_cache=req.refresh)
            vec, found_ids, total = get_taste_vector(weighted)
            if found_ids:
                user_weighted.append(weighted)

    genre_filter = [g.strip() for g in req.genre if g.strip()] if req.genre else None

    if not user_weighted and req.query:
        recs = recommend_from_query(
            req.query, n=req.n, min_vote_count=req.min_vote_count,
            diversity=req.diversity, genre_filter=genre_filter,
            min_year=req.min_year, max_year=req.max_year,
            min_rating=req.min_rating, randomize=req.randomize,
        )
    elif user_weighted and req.query:
        recs = recommend_hybrid(
            user_weighted, req.query, n=req.n,
            min_vote_count=req.min_vote_count,
            query_weight=req.query_weight, diversity=req.diversity,
            genre_filter=genre_filter, min_year=req.min_year,
            max_year=req.max_year, min_rating=req.min_rating,
            randomize=req.randomize,
        )
    elif user_weighted:
        recs = recommend_for_group(
            user_weighted, n=req.n, min_vote_count=req.min_vote_count,
            diversity=req.diversity, genre_filter=genre_filter,
            min_year=req.min_year, max_year=req.max_year,
            min_rating=req.min_rating, randomize=req.randomize,
        )
    else:
        total_t = time.perf_counter() - t0
        print(f"  ⏱  TOTAL: {total_t:.2f}s (no users, no query → empty)", file=sys.stderr, flush=True)
        return []

    total_t = time.perf_counter() - t0
    print(f"  ⏱  TOTAL: {total_t:.2f}s | {len(recs)} movies returned", file=sys.stderr, flush=True)

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
