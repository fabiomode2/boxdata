from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import Depends, FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
from src.session import get_curl_session
from src.vector_store import count_movies

logger = logging.getLogger(__name__)

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY environment variable is not set")
if len(API_KEY) < 32:
    logger.warning("API_KEY is shorter than 32 characters; rotate it for a stronger one")


async def verify_api_key(x_api_key: str | None = Header(None)):
    if not x_api_key or not secrets.compare_digest(
        x_api_key.encode("utf-8"), API_KEY.encode("utf-8")
    ):
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
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
    max_age=600,
)

_MAX_BODY_BYTES = 64 * 1024


@app.middleware("http")
async def limit_request_body(request: Request, call_next):
    """Reject oversized payloads before FastAPI buffers them."""
    raw_length = request.headers.get("content-length")
    if raw_length:
        try:
            if int(raw_length) > _MAX_BODY_BYTES:
                return JSONResponse({"detail": "Request body too large"}, status_code=413)
        except ValueError:
            return JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)
    return await call_next(request)

# ── Async background job system ──────────────────────────────

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "4"))
_JOB_RETENTION = 300          # finished jobs stay pollable for 5 min
_STALE_JOB_AGE = 2 * 3600     # never let a wedged job block the queue forever

def _cleanup_old_jobs():
    now = time.time()
    with _jobs_lock:
        expired = [
            jid for jid, j in _jobs.items()
            if (j.get("status") in ("done", "error") and now - j.get("_ts", 0) > _JOB_RETENTION)
            or now - j.get("_ts", 0) > _STALE_JOB_AGE
        ]
        for jid in expired:
            del _jobs[jid]

class RecommendRequest(BaseModel):
    users: list[str] = Field(default=[], max_length=5)
    query: str | None = Field(default=None, max_length=500)
    n: int = Field(default=10, ge=1, le=50)
    min_vote_count: int = Field(default=2000, ge=0, le=500000)
    diversity: float = Field(default=0.0, ge=0.0, le=1.0)
    genre: list[str] | None = Field(default=None, max_length=20)
    query_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    min_year: int | None = Field(default=None, ge=1888, le=2100)
    max_year: int | None = Field(default=None, ge=1888, le=2100)
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

    @field_validator("genre")
    @classmethod
    def validate_genres(cls, genres: list[str] | None) -> list[str] | None:
        if genres is None:
            return None
        for g in genres:
            if len(g) > 50:
                raise ValueError("Genre name too long")
        return genres

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
                _jobs[job_id].update(status="error", error="Recommendation failed. Please try again.")
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

_AVATAR_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
_AVATAR_TTL = int(os.getenv("AVATAR_CACHE_TTL", 7 * 86400))


def _avatar_cache_path(username: str) -> Path:
    return _AVATAR_CACHE_DIR / f"avatar_{username}.json"


def _read_avatar_cache(username: str) -> tuple[str | None, float] | None:
    """Return ``(avatar_url, age_seconds)`` or None when not cached."""
    path = _avatar_cache_path(username)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
        return raw.get("url"), time.time() - path.stat().st_mtime
    except (OSError, ValueError, AttributeError) as e:
        logger.warning("Discarding unreadable avatar cache %s: %s", path.name, e)
        path.unlink(missing_ok=True)
        return None


def _write_avatar_cache(username: str, url: str | None) -> None:
    try:
        _AVATAR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _avatar_cache_path(username)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"url": url}))
        os.replace(tmp, path)
        path.chmod(0o600)
    except OSError as e:
        logger.debug("Could not persist avatar cache for %s: %s", username, e)


def _fetch_avatar_url(username: str) -> str | None:
    """Og:image avatar URL, disk-cached. Retries: Cloudflare challenges datacenter IPs."""
    cached = _read_avatar_cache(username)
    if cached is not None and cached[1] <= _AVATAR_TTL:
        return cached[0]

    session = get_curl_session()
    for attempt in range(6):
        try:
            resp = session.get(f"https://letterboxd.com/{username}/", timeout=15)
            if resp.status_code == 200 and "Just a moment" not in resp.text:
                m = OG_RE.search(resp.text)
                url = m.group(1) if m else None
                _write_avatar_cache(username, url)
                return url
        except Exception:
            logger.debug("Avatar fetch attempt %d failed for %s", attempt + 1, username)
        time.sleep(3.0)
    return cached[0] if cached is not None else None

@app.get("/api/avatar/{username}")
@limiter.limit("60/minute")
async def avatar(request: Request, username: str, _auth=Depends(verify_api_key)):
    if not re.match(r"^[a-zA-Z0-9_-]{1,50}$", username):
        raise HTTPException(status_code=400, detail="Invalid username")
    avatar_url = await asyncio.to_thread(_fetch_avatar_url, username)
    return {"avatar_url": avatar_url}

@app.post("/api/recommend", status_code=202)
@limiter.limit("30/minute")
async def recommend(request: Request, req: RecommendRequest, _auth=Depends(verify_api_key)):
    _cleanup_old_jobs()

    users_str = ",".join(req.users) if req.users else "query-only"
    job_id = uuid.uuid4().hex
    print(f"\n--- /api/recommend [{users_str}] job={job_id} ---", file=sys.stderr, flush=True)

    with _jobs_lock:
        active = sum(
            1 for j in _jobs.values()
            if j.get("status") not in ("done", "error")
        )
        if active >= _MAX_CONCURRENT_JOBS:
            raise HTTPException(status_code=429, detail="Too many active jobs. Try again shortly.")
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
