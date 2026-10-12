from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from curl_cffi import requests
from fastapi import Depends, FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

from src.recommender import (
    fetch_letterboxd_watched,
    get_taste_vector,
    recommend_for_group,
    recommend_from_query,
    recommend_hybrid,
)
from src.scraper import fetch_letterboxd_watchlist
from src.vector_store import count_movies

logger = logging.getLogger(__name__)

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY environment variable is not set")


async def verify_api_key(x_api_key: str | None = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


limiter = Limiter(key_func=get_remote_address)
_DOCS_DISABLED = os.getenv("DISABLE_DOCS", "0") == "1"
app = FastAPI(
    title="BoxData API",
    docs_url=None if _DOCS_DISABLED else "/docs",
    redoc_url=None if _DOCS_DISABLED else "/redoc",
    openapi_url=None if _DOCS_DISABLED else "/openapi.json",
)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "https://boxdata.vercel.app").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Async background job system ──────────────────────────────

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

def _cleanup_old_jobs():
    now = time.time()
    with _jobs_lock:
        expired = [jid for jid, j in _jobs.items() if j.get("status") in ("done", "error") and now - j.get("_ts", 0) > 300]
        for jid in expired:
            del _jobs[jid]

class RecommendRequest(BaseModel):
    users: list[str] = Field(default=[], max_length=5)
    query: str | None = Field(default=None, max_length=500)
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
    include_watchlist: bool = True

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

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: dict = {}
    results: list[MovieResult] | None = None
    error: str | None = None


def _run_recommend_job(job_id: str, req: RecommendRequest):
    """Run recommendation pipeline in background, updating progress."""
    progress: dict = {"status": "queued"}
    prog_lock = threading.Lock()

    def _sync():
        with prog_lock:
            with _jobs_lock:
                if job_id in _jobs:
                    _jobs[job_id] = {**_jobs[job_id], "progress": dict(progress)}

    def _set(**kw):
        with prog_lock:
            progress.update(kw, _ts=time.time())
            new_status = progress.get("status")
        if new_status:
            with _jobs_lock:
                if job_id in _jobs and new_status != _jobs[job_id].get("status"):
                    _jobs[job_id]["status"] = new_status
        _sync()

    try:
        _set(status="scraping")
        user_weighted: list[dict[int, float]] = []

        for username in req.users or []:
            _set(user=username, phase="pages")

            weighted = fetch_letterboxd_watched(
                username, no_cache=req.refresh,
                progress=progress,
            )
            _sync()  # sync scraper's progress.update() calls

            if req.include_watchlist:
                _set(user=username, phase="watchlist")
                watchlist = fetch_letterboxd_watchlist(
                    username, no_cache=req.refresh,
                    progress=progress,
                )
                _sync()

            _set(user=username, phase="taste_vector")
            vec, found_ids, total = get_taste_vector(weighted)
            if found_ids:
                user_weighted.append(weighted)

        _set(user="", phase="recommending")
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
            recs = []

        results = [
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

        _set(status="done")
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id].update(status="done", results=[r.model_dump() for r in results], progress=dict(progress))

        print(f"  ✅ Job {job_id} done: {len(results)} movies", file=sys.stderr, flush=True)

    except Exception as e:
        logger.exception("Job %s failed", job_id)
        with prog_lock:
            progress["status"] = "error"
        _sync()
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id].update(status="error", error=str(e))
        print(f"  ❌ Job {job_id} failed: {e}", file=sys.stderr, flush=True)


@app.get("/")
@limiter.limit("30/minute")
async def root(request: Request):
    return {"service": "BoxData API", "status": "ok", "health": "/health"}


@app.get("/health")
@limiter.limit("30/minute")
async def health(request: Request):
    _cleanup_old_jobs()
    return {"status": "ok", "movies": count_movies()}

OG_RE = re.compile(r'<meta\s+property="og:image"\s+content="([^"]+)"')

@app.get("/api/avatar/{username}")
@limiter.limit("60/minute")
async def avatar(request: Request, username: str, _auth=Depends(verify_api_key)):
    if not re.match(r"^[a-zA-Z0-9_-]{1,50}$", username):
        raise HTTPException(status_code=400, detail="Invalid username")
    url = f"https://letterboxd.com/{username}/"
    # Letterboxd (Cloudflare) intermittently serves challenge pages to
    # datacenter IPs; retry with backoff until a real page comes through.
    for attempt in range(4):
        try:
            resp = await asyncio.to_thread(
                requests.get,
                url,
                impersonate="safari17_0",
                timeout=15,
            )
            if resp.status_code == 200 and "Just a moment" not in resp.text:
                m = OG_RE.search(resp.text)
                return {"avatar_url": m.group(1) if m else None}
        except Exception:
            logger.debug("Avatar fetch attempt %d failed for %s", attempt + 1, username)
        if attempt < 3:
            await asyncio.sleep(2.0 * (attempt + 1))
    return {"avatar_url": None}

@app.post("/api/recommend", status_code=202)
@limiter.limit("30/minute")
async def recommend(request: Request, req: RecommendRequest, _auth=Depends(verify_api_key)):
    _cleanup_old_jobs()
    import asyncio

    users_str = ",".join(req.users) if req.users else "query-only"
    job_id = uuid.uuid4().hex[:12]
    print(f"\n--- /api/recommend [{users_str}] job={job_id} ---", file=sys.stderr, flush=True)

    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": {"status": "queued"},
            "results": None,
            "error": None,
            "_ts": time.time(),
        }

    thread = threading.Thread(target=_run_recommend_job, args=(job_id, req), daemon=True)
    thread.start()

    return {"job_id": job_id, "status": "queued"}

@app.get("/api/job/{job_id}", response_model=JobStatus)
@limiter.limit("120/minute")
async def get_job(request: Request, job_id: str):
    """Public endpoint (no auth) — job_id is a random UUID, safe to expose."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    results = None
    if job["results"]:
        results = [MovieResult(**r) for r in job["results"]]

    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress", {}),
        results=results,
        error=job.get("error"),
    )
