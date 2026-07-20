from __future__ import annotations

import json
import logging
import math
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from src.similarity import (
    _calibrate_sigma_cv,
    _cosine_similarity_np,
    _predict_rating_batch,
)
from src.timers import _timer_start, _timer_stop
from src.vector_store import get_collection

logger = logging.getLogger(__name__)

_SIGMA_CACHE: dict[frozenset, float] = {}


def _sigma_cache_key(wdict: dict[int, float]) -> frozenset:
    return frozenset(wdict.items())


def _fetch_user_embeddings(
    weighted_ids: dict[int, float],
) -> tuple[list[list[float]], list[float], list[int], int, list[dict]]:
    if not weighted_ids:
        return [], [], [], 0, []

    col = get_collection()
    all_ids = [str(tid) for tid in weighted_ids]

    vectors: list[list[float]] = []
    weights: list[float] = []
    found: list[int] = []
    metadatas: list[dict] = []

    for i in range(0, len(all_ids), 999):
        batch_ids = all_ids[i:i + 999]
        meta_res = col.get(ids=batch_ids, include=["embeddings", "metadatas"])
        for j, tid_str in enumerate(meta_res["ids"]):
            tid = int(tid_str)
            user_weight = weighted_ids[tid]
            emb = meta_res["embeddings"][j]
            if emb is None:
                continue
            emb_arr = emb.tolist() if hasattr(emb, "tolist") else emb
            meta = meta_res["metadatas"][j] if meta_res["metadatas"] else {}
            vectors.append(emb_arr)
            weights.append(user_weight)
            found.append(tid)
            metadatas.append(meta)

    return vectors, weights, found, len(weighted_ids), metadatas


def _load_user_history(
    weighted_ids: dict[int, float],
) -> tuple[list[tuple[list[float], float]], dict[int, dict]]:
    _timer_start("  load_user_history")
    if not weighted_ids:
        _timer_stop()
        return [], {}

    vectors, weights, found, _, metadatas = _fetch_user_embeddings(weighted_ids)
    history = [(vec, w) for vec, w in zip(vectors, weights)]
    meta_cache = dict(zip(found, metadatas))

    _timer_stop()
    return history, meta_cache


def _matches_genre_filter(meta_genres: str, genre_filter: list[str] | None) -> bool:
    if not genre_filter:
        return True
    meta_lower = meta_genres.lower()
    return any(g.lower() in meta_lower for g in genre_filter)


def _matches_year_filter(year: int, min_year: int | None, max_year: int | None) -> bool:
    if min_year is not None and year < min_year:
        return False
    if max_year is not None and year > max_year:
        return False
    return True


def _tmdb_vote_data(meta: dict) -> tuple[float, int, float]:
    avg = meta.get("vote_average", 0)
    cnt = meta.get("vote_count", 0)

    mc = meta.get("metacritic_score")
    mcb = 1.0 + max(0, (mc or 50) - 50) / 500.0 if mc is not None else 1.0

    return float(avg), int(cnt), mcb


def _rerank_candidates(
    candidates: list[dict[str, Any]],
    alpha: float = 0.5,
    k: int = 60,
) -> list[dict[str, Any]]:
    if len(candidates) < 2:
        return candidates

    ranked_by_sim = sorted(candidates, key=lambda x: x["similarity_score"], reverse=True)
    sim_rank: dict[int, int] = {
        id(c): i for i, c in enumerate(ranked_by_sim)
    }

    ranked_by_pop = sorted(
        candidates,
        key=lambda x: x["vote_average"] * math.log(x["vote_count"] + 1) * x.get("metacritic_bonus", 1.0),
        reverse=True,
    )
    pop_rank: dict[int, int] = {
        id(c): i for i, c in enumerate(ranked_by_pop)
    }

    for c in candidates:
        r_sim = sim_rank[id(c)]
        r_pop = pop_rank[id(c)]
        rrf_sim = 1.0 / (k + r_sim)
        rrf_pop = 1.0 / (k + r_pop)
        c["rerank_score"] = round(alpha * rrf_sim + (1.0 - alpha) * rrf_pop, 4)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates


def _mmr_diversify(
    candidates: list[dict[str, Any]],
    n: int,
    lam: float = 0.5,
) -> list[dict[str, Any]]:
    if not candidates or lam >= 1.0:
        return candidates[:n]

    selected: list[dict[str, Any]] = []
    remaining = list(candidates)
    sel_embs: list[np.ndarray] = []

    while len(selected) < n and remaining:
        best_idx = 0
        best_score = -float("inf")
        for i, cand in enumerate(remaining):
            relevance = cand.get("rerank_score", cand.get("similarity_score", 0))
            if not selected:
                mmr = relevance
            else:
                cand_np = np.array(cand.get("_emb", [0.0]))
                max_sim = max(_cosine_similarity_np(cand_np, sel_emb) for sel_emb in sel_embs)
                mmr = lam * relevance - (1.0 - lam) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_idx = i
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        sel_embs.append(np.array(chosen.get("_emb", [0.0])))

    return selected


def _feature_rerank(
    candidates: list[dict[str, Any]],
    user_weighted_list: list[dict[int, float]],
    user_meta_cache: dict[int, dict] | None = None,
) -> list[dict[str, Any]]:
    if not candidates or not user_weighted_list:
        return candidates

    fav_directors: set[str] = set()
    fav_decades: set[int] = set()
    fav_actors: dict[str, float] = {}

    if user_meta_cache is not None:
        for wdict in user_weighted_list:
            for tid in wdict:
                meta = user_meta_cache.get(tid)
                if meta is None:
                    continue
                director = (meta.get("director") or "").lower().strip()
                if director:
                    fav_directors.add(director)
                year = meta.get("year") or 0
                if year:
                    fav_decades.add(year - (year % 10))
                cast_str = meta.get("cast") or ""
                for actor in cast_str.split(","):
                    actor = actor.strip().lower()
                    if actor:
                        fav_actors[actor] = fav_actors.get(actor, 0) + 1
    else:
        col = get_collection()
        for wdict in user_weighted_list:
            user_ids = [str(tid) for tid in wdict]
            for i in range(0, len(user_ids), 999):
                batch = user_ids[i : i + 999]
                res = col.get(ids=batch, include=["metadatas"])
                for j in range(len(res["ids"])):
                    meta = res["metadatas"][j]
                    director = (meta.get("director") or "").lower().strip()
                    if director:
                        fav_directors.add(director)
                    year = meta.get("year") or 0
                    if year:
                        fav_decades.add(year - (year % 10))
                    cast_str = meta.get("cast") or ""
                    for actor in cast_str.split(","):
                        actor = actor.strip().lower()
                        if actor:
                            fav_actors[actor] = fav_actors.get(actor, 0) + 1

    for c in candidates:
        meta_director = (c.get("director") or "").lower().strip()
        meta_year = c.get("year") or 0
        meta_decade = meta_year - (meta_year % 10) if meta_year else None

        bonus = 0.0

        if meta_director and meta_director in fav_directors:
            bonus += 0.10

        if meta_decade is not None and meta_decade in fav_decades:
            bonus += 0.05

        cast_str = c.get("cast", "")
        if cast_str and fav_actors:
            cand_actors = set(a.strip().lower() for a in cast_str.split(",") if a.strip())
            if cand_actors:
                intersection = len(cand_actors & set(fav_actors.keys()))
                union = len(cand_actors | set(fav_actors.keys()))
                jaccard = intersection / union if union > 0 else 0.0
                bonus += 0.05 * jaccard

        capped = min(bonus, 0.10)
        c["rerank_score"] = round(c["rerank_score"] * (1.0 + capped), 4)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates


def recommend_from_query(
    query_text: str,
    n: int = 10,
    min_vote_count: int = 2000,
    diversity: float = 0.0,
    genre_filter: list[str] | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    min_rating: float | None = 6.5,
    randomize: bool = False,
) -> list[dict[str, Any]]:
    _timer_start("recommend_from_query total")
    from src.embedders import OllamaEmbedder
    _timer_start("  embed query")
    embedder = OllamaEmbedder()
    vec = embedder.embed_batch([query_text])[0]
    _timer_stop()

    _timer_start("  ChromaDB query + filter")
    col = get_collection()
    results = col.query(
        query_embeddings=[vec],
        n_results=min(col.count(), n * 30),
        include=["embeddings", "metadatas", "distances"],
    )

    candidates: list[dict[str, Any]] = []
    for i, tmdb_id_str in enumerate(results["ids"][0]):
        tid = int(tmdb_id_str)
        meta = results["metadatas"][0][i]
        dist = results["distances"][0][i]
        similarity = 1.0 - dist
        cand_emb = results["embeddings"][0][i]
        cand_arr = cand_emb.tolist() if hasattr(cand_emb, "tolist") else cand_emb

        vote_avg, vote_cnt, mc_bonus = _tmdb_vote_data(meta)
        if vote_cnt < min_vote_count:
            continue
        if not _matches_genre_filter(meta.get("genres", ""), genre_filter):
            continue
        if not _matches_year_filter(meta.get("year", 0), min_year, max_year):
            continue
        if min_rating is not None and vote_avg < min_rating:
            continue

        pp = meta.get("poster_path", "")
        poster_url = f"https://image.tmdb.org/t/p/w200{pp}" if pp else ""
        candidates.append({
            "tmdb_id": tid,
            "title": meta.get("title", "Unknown"),
            "genres": meta.get("genres", ""),
            "year": meta.get("year", 0),
            "director": meta.get("director", ""),
            "cast": meta.get("cast", ""),
            "vote_average": vote_avg,
            "vote_count": vote_cnt,
            "metacritic_bonus": mc_bonus,
            "similarity_score": round(similarity, 4),
            "rerank_score": 0.0,
            "url": f"https://letterboxd.com/tmdb/{tid}/",
            "poster": poster_url,
            "_emb": cand_arr,
        })
    _timer_stop()

    _timer_start("  rerank")
    reranked = _rerank_candidates(candidates, alpha=0.9)
    _timer_stop()
    _timer_stop()
    if diversity > 0.0:
        return _mmr_diversify(reranked, n, lam=1.0 - diversity)
    return reranked[:n]


def _recommend_pipeline(
    user_weighted_list: list[dict[int, float]],
    n: int = 10,
    min_vote_count: int = 2000,
    sigma: float | None = None,
    max_distance: float = -1.0,
    diversity: float = 0.0,
    genre_filter: list[str] | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    min_rating: float | None = 6.5,
    randomize: bool = False,
    query_text: str | None = None,
    query_weight: float = 0.0,
    feature_rerank: bool = True,
) -> list[dict[str, Any]]:
    _timer_start("_recommend_pipeline total")

    _timer_start("  load user histories")
    exclude: set[int] = set()
    user_histories: list[list[tuple[list[float], float]]] = []
    user_meta_cache: dict[int, dict] = {}

    for wdict in user_weighted_list:
        exclude.update(wdict.keys())
        history, meta = _load_user_history(wdict)
        if history:
            user_histories.append(history)
            user_meta_cache.update(meta)
    _timer_stop()

    if not user_histories:
        if query_text:
            result = recommend_from_query(
                query_text, n=n, min_vote_count=min_vote_count,
                diversity=diversity, genre_filter=genre_filter,
                min_year=min_year, max_year=max_year,
                min_rating=min_rating, randomize=randomize,
            )
            _timer_stop()
            return result
        _timer_stop()
        return []

    _timer_start("  sigma calibration")
    sigmas: list[float] = []
    for h_idx, history in enumerate(user_histories):
        if sigma is not None:
            sigmas.append(sigma)
        else:
            wdict = user_weighted_list[h_idx]
            key = _sigma_cache_key(wdict)
            cached = _SIGMA_CACHE.get(key)
            if cached is not None:
                sigmas.append(cached)
            else:
                s = _calibrate_sigma_cv(history, max_distance=max_distance)
                _SIGMA_CACHE[key] = s
                sigmas.append(s)
    if sigma is None:
        logger.info("  Auto-calibrated sigmas: %s",
                     ", ".join(f"{s:.2f}" for s in sigmas))
    _timer_stop()

    _timer_start("  query embedding")
    query_vec = None
    query_vec_np = None
    w = max(0.0, min(1.0, query_weight))
    if query_text is not None:
        from src.embedders import OllamaEmbedder
        embedder = OllamaEmbedder()
        query_vec = embedder.embed_batch([query_text])[0]
        query_vec_np = np.array([query_vec])
    _timer_stop()

    _timer_start("  discovery seeds + ChromaDB query")
    discovery_seeds: list[list[float]] = []
    for history in user_histories:
        if not history:
            continue
        dim = len(history[0][0])
        total_w = sum(r for _, r in history)
        centroid = [0.0] * dim
        for vec, rating in history:
            w = rating / total_w
            for i in range(dim):
                centroid[i] += vec[i] * w
        discovery_seeds.append(centroid)
        if randomize:
            rng_pert = np.random.default_rng()
            for _ in range(3):
                noise = rng_pert.normal(0, 0.15, size=dim).tolist()
                discovery_seeds.append([c + n for c, n in zip(centroid, noise)])
    if query_vec is not None:
        discovery_seeds.append(query_vec)

    if not discovery_seeds:
        discovery_seeds = [user_histories[0][0][0]]

    dim = len(discovery_seeds[0])
    n_random = 12 if randomize else 4
    rng = np.random.default_rng(None if randomize else 42)
    for _ in range(n_random):
        rand_vec = rng.normal(size=dim).tolist()
        discovery_seeds.append(rand_vec)

    hist_np = [
        (
            np.array([h[0] for h in hist]),
            np.array([h[1] for h in hist], dtype=float),
        )
        for hist in user_histories
    ]

    col = get_collection()
    total_db = col.count()
    pool_per_seed = min(max(n * 5, 15), 40)
    seen: set[int] = set(exclude)
    batch_meta: list[tuple[int, dict, float, int, float, list[float]]] = []

    multi_results = col.query(
        query_embeddings=discovery_seeds,
        n_results=min(total_db, pool_per_seed),
        include=["embeddings", "metadatas"],
    )

    for q_idx in range(len(discovery_seeds)):
        for j, tid_str in enumerate(multi_results["ids"][q_idx]):
            tid = int(tid_str)
            if tid in seen:
                continue
            seen.add(tid)
            meta = multi_results["metadatas"][q_idx][j]
            cand_emb = multi_results["embeddings"][q_idx][j]
            if cand_emb is None:
                continue
            cand_arr = cand_emb.tolist() if hasattr(cand_emb, "tolist") else cand_emb
            vote_avg, vote_cnt, mc_bonus = _tmdb_vote_data(meta)
            if vote_cnt < min_vote_count:
                continue
            if not _matches_genre_filter(meta.get("genres", ""), genre_filter):
                continue
            if not _matches_year_filter(meta.get("year", 0), min_year, max_year):
                continue
            if min_rating is not None and vote_avg < min_rating:
                continue
            batch_meta.append((tid, meta, vote_avg, vote_cnt, mc_bonus, cand_arr))

    if not batch_meta:
        _timer_stop()
        _timer_stop()
        return []
    _timer_stop()

    _timer_start("  batch scoring + consensus")
    cand_embs = np.array([b[5] for b in batch_meta])
    user_preds = np.array([
        _predict_rating_batch(cand_embs, hembs, hrats, sigmas[j], max_distance)
        for j, (hembs, hrats) in enumerate(hist_np)
    ])

    eps = 1e-6
    product = np.prod(np.maximum(user_preds, eps) ** 2, axis=0)
    consensus = product ** (1.0 / user_preds.shape[0])

    query_sims = None
    if query_vec_np is not None:
        q_norm = np.linalg.norm(query_vec_np)
        c_norms = np.linalg.norm(cand_embs, axis=1, keepdims=True).flatten()
        query_sims = (cand_embs @ query_vec_np.flatten()) / (c_norms * q_norm + 1e-12)

    candidates: list[dict[str, Any]] = []
    for idx, (tid, meta, vote_avg, vote_cnt, mc_bonus, cand_arr) in enumerate(batch_meta):
        if query_sims is not None:
            sim = (1.0 - w) * float(consensus[idx]) + w * float(query_sims[idx])
        else:
            sim = float(consensus[idx])
        pp = meta.get("poster_path", "")
        poster_url = f"https://image.tmdb.org/t/p/w200{pp}" if pp else ""
        candidates.append({
            "tmdb_id": tid,
            "title": meta.get("title", "Unknown"),
            "genres": meta.get("genres", ""),
            "year": meta.get("year", 0),
            "director": meta.get("director", ""),
            "cast": meta.get("cast", ""),
            "vote_average": vote_avg,
            "vote_count": vote_cnt,
            "metacritic_bonus": mc_bonus,
            "similarity_score": round(sim, 4),
            "rerank_score": 0.0,
            "url": f"https://letterboxd.com/tmdb/{tid}/",
            "poster": poster_url,
            "_emb": cand_arr,
        })
    _timer_stop()

    _timer_start("  rerank + MMR")
    reranked = _rerank_candidates(candidates, alpha=0.8)
    _timer_stop()

    if feature_rerank and user_histories:
        _timer_start("  feature rerank")
        reranked = _feature_rerank(reranked, user_weighted_list, user_meta_cache)
        _timer_stop()

    _timer_stop()
    if diversity > 0.0:
        return _mmr_diversify(reranked, n, lam=1.0 - diversity)
    return reranked[:n]


def rank_watchlist(
    export_dir: str,
    n: int | None = None,
    min_vote_count: int = 2000,
    sigma: float | None = 0.3,
    max_distance: float = -1.0,
    genre_filter: list[str] | None = None,
    min_year: int | None = None,
    max_year: int | None = None,
    min_rating: float | None = 6.5,
    auto_sigma: bool = False,
) -> list[dict[str, Any]]:
    _timer_start("rank_watchlist total")

    from src.profile import parse_export, resolve_tmdb_ids

    profile = parse_export(export_dir)
    logger.info("Profile: %d watched, %d watchlist", len(profile.watched), len(profile.watchlist))

    _timer_start("  resolve TMDB IDs")
    profile = resolve_tmdb_ids(profile, resolve_watchlist=True)
    _timer_stop()

    watched_ids = {f.tmdb_id for f in profile.watched if f.tmdb_id}
    weighted: dict[int, float] = {}
    for f in profile.watched:
        if f.tmdb_id:
            weighted[f.tmdb_id] = f.rating / 5.0 if f.rating is not None else 0.6
    logger.info("Watched with TMDB IDs: %d", len(weighted))

    if not weighted:
        logger.warning("No watched history found.")
        _timer_stop()
        return []

    history, _ = _load_user_history(weighted)
    if not history:
        logger.warning("No watched films found in ChromaDB.")
        _timer_stop()
        return []

    watchlist_ids = [
        f.tmdb_id for f in profile.watchlist
        if f.tmdb_id and f.tmdb_id not in watched_ids
    ]
    logger.info("Watchlist in ChromaDB (excl. watched): %d", len(watchlist_ids))

    if not watchlist_ids:
        logger.warning("Watchlist is empty or all films already watched.")
        _timer_stop()
        return []

    _timer_start("  fetch watchlist from ChromaDB")
    col = get_collection()
    watch_map: dict[str, tuple[dict, list[float] | None]] = {}
    batch_size = 999
    for i in range(0, len(watchlist_ids), batch_size):
        batch = [str(tid) for tid in watchlist_ids[i:i + batch_size]]
        res = col.get(ids=batch, include=["embeddings", "metadatas"])
        for j, tid_s in enumerate(res["ids"]):
            watch_map[tid_s] = (res["metadatas"][j], res["embeddings"][j])
    _timer_stop()

    _timer_start("  filter candidates")
    cand_embs: list[list[float]] = []
    cand_metas: list[tuple[int, dict]] = []
    not_found = 0
    for tid in watchlist_ids:
        tid_s = str(tid)
        entry = watch_map.get(tid_s)
        if entry is None:
            not_found += 1
            continue
        meta, emb = entry
        if emb is None:
            not_found += 1
            continue

        vote_avg, vote_cnt, _ = _tmdb_vote_data(meta)
        if vote_cnt < min_vote_count:
            continue
        if not _matches_genre_filter(meta.get("genres", ""), genre_filter):
            continue
        if not _matches_year_filter(meta.get("year", 0), min_year, max_year):
            continue
        if min_rating is not None and vote_avg < min_rating:
            continue

        cand_arr = emb.tolist() if hasattr(emb, "tolist") else emb
        cand_embs.append(cand_arr)
        cand_metas.append((tid, meta))
    if not_found:
        logger.info("  %d watchlist films not in ChromaDB", not_found)
    _timer_stop()

    if not cand_embs:
        logger.warning("No watchlist films passed the filters.")
        _timer_stop()
        return []

    _timer_start("  sigma calibration")
    if auto_sigma:
        sigma = _calibrate_sigma_cv(history, max_distance=max_distance)
        print(f"  Auto-sigma: {sigma:.3f}", file=sys.stderr, flush=True)
    _timer_stop()

    _timer_start("  batch predict (%d candidates x %d history)" % (len(cand_embs), len(history)))
    hist_embs_np = np.array([h[0] for h in history])
    hist_ratings_np = np.array([h[1] for h in history], dtype=float)
    cand_np = np.array(cand_embs)
    preds = _predict_rating_batch(cand_np, hist_embs_np, hist_ratings_np, sigma or 0.3, max_distance)
    _timer_stop()

    _timer_start("  sort results")
    results = []
    for idx, (tid, meta) in enumerate(cand_metas):
        pred_raw = float(preds[idx])
        pp = meta.get("poster_path", "")
        poster_url = f"https://image.tmdb.org/t/p/w200{pp}" if pp else ""
        results.append({
            "tmdb_id": tid,
            "title": meta.get("title", "Unknown"),
            "genres": meta.get("genres", ""),
            "year": meta.get("year", 0),
            "vote_average": meta.get("vote_average", 0),
            "vote_count": meta.get("vote_count", 0),
            "predicted_rating": round(pred_raw, 4),
            "predicted_stars": round(pred_raw * 5, 2),
            "url": f"https://letterboxd.com/tmdb/{tid}/",
            "poster": poster_url,
        })

    results.sort(key=lambda x: x["predicted_rating"], reverse=True)
    if n is not None:
        results = results[:n]
    _timer_stop()

    _timer_stop()
    return results
