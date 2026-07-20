import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from curl_cffi import requests
from curl_cffi import CurlError

from src.config import IMDB_BASE

logger = logging.getLogger(__name__)

_BATCH_CHUNK = 20
_MAX_WORKERS = 10


def _fetch_one(imdb_id: str) -> dict | None:
    try:
        resp = requests.get(f"{IMDB_BASE}/titles/{imdb_id}", timeout=15)
        if resp.status_code != 200:
            logger.warning("IMDb API %d for %s", resp.status_code, imdb_id)
            return None
        return resp.json()
    except CurlError as e:
        logger.warning("IMDb API error for %s: %s", imdb_id, e)
        return None


def _fetch_batch(imdb_ids: list[str]) -> dict[str, dict]:
    params = [("title_ids", i) for i in imdb_ids]
    try:
        resp = requests.get(f"{IMDB_BASE}/titles:batchGet", params=params, timeout=30)
        if resp.status_code != 200:
            logger.warning("IMDb batch got %d, falling back to individual", resp.status_code)
            return {}
        data = resp.json()
        titles = data.get("titles", [])
        return {t["id"]: t for t in titles}
    except CurlError as e:
        logger.warning("IMDb batch error: %s", e)
        return {}


def fetch_enrichment(imdb_ids: list[str]) -> dict[str, dict]:
    if not imdb_ids:
        return {}

    deduped = list(dict.fromkeys(imdb_ids))
    result: dict[str, dict] = {}

    for i in range(0, len(deduped), _BATCH_CHUNK):
        chunk = deduped[i : i + _BATCH_CHUNK]
        if len(chunk) == 1:
            data = _fetch_one(chunk[0])
            if data:
                result[chunk[0]] = data
        else:
            batch = _fetch_batch(chunk)
            missing = [iid for iid in chunk if iid not in batch]
            if missing:
                with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
                    futures = {pool.submit(_fetch_one, iid): iid for iid in missing}
                    for future in as_completed(futures):
                        iid = futures[future]
                        try:
                            data = future.result()
                            if data:
                                result[iid] = data
                        except Exception as e:
                            logger.debug("IMDb individual fetch failed for '%s': %s", iid, e)
            result.update(batch)

    return result


def extract_enrichment(data: dict) -> dict:
    rating_data = data.get("rating") or {}
    meta = data.get("metacritic") or {}

    interests = []
    for item in data.get("interests") or []:
        if isinstance(item, dict) and item.get("name"):
            interests.append(item["name"])

    return {
        "imdb_rating": rating_data.get("aggregateRating"),
        "imdb_vote_count": rating_data.get("voteCount"),
        "metacritic_score": meta.get("score"),
        "metacritic_review_count": meta.get("reviewCount"),
        "interests": interests,
        "origin_countries": [
            c.get("name") for c in (data.get("originCountries") or []) if isinstance(c, dict) and c.get("name")
        ],
        "spoken_languages": [
            l.get("name") for l in (data.get("spokenLanguages") or []) if isinstance(l, dict) and l.get("name")
        ],
    }
