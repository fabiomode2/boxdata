"""Select ~1000 movies most shared among eval users for dev embedding comparison."""
import json
import logging
from collections import Counter
from pathlib import Path

from src.profile import parse_export, resolve_tmdb_ids

logger = logging.getLogger(__name__)

EVAL_USERS = ["fabiomode2", "narrowd", "efrx", "georgemcnuggets"]
FILTERED_FILE = Path("data/movies_filtered.jsonl")
OUTPUT_FILE = Path("data/movies_dev_1000.jsonl")


def _load_user_watchlists(usernames: list[str], no_cache: bool = False) -> dict[str, set[int]]:
    from src.recommender import fetch_letterboxd_watched
    result: dict[str, set[int]] = {}
    for username in usernames:
        logger.info("Loading %s watchlist...", username)
        weighted = fetch_letterboxd_watched(username, no_cache=no_cache)
        result[username] = set(weighted.keys())
        logger.info("  %d movies", len(result[username]))
    return result


def _load_filtered_ids() -> set[int]:
    ids: set[int] = set()
    with open(FILTERED_FILE) as f:
        for line in f:
            d = json.loads(line)
            ids.add(d["tmdb_id"])
    logger.info("Filtered movies: %d", len(ids))
    return ids


def run_dev_setup(no_cache: bool = False) -> None:
    if not FILTERED_FILE.exists():
        logger.error("%s not found. Run `python main.py filter` first.", FILTERED_FILE)
        return

    filtered_ids = _load_filtered_ids()
    watchlists = _load_user_watchlists(EVAL_USERS, no_cache=no_cache)

    # Count how many users have each movie
    movie_user_count: Counter[int] = Counter()
    for uid_set in watchlists.values():
        for tmdb_id in uid_set:
            if tmdb_id in filtered_ids:
                movie_user_count[tmdb_id] += 1

    logger.info("Movies shared by at least 1 user in filtered set: %d", len(movie_user_count))

    # Sort by share count desc, then pick top 1000
    top_1000 = [tid for tid, _ in movie_user_count.most_common(1000)]
    logger.info("Top 1000 selected (shared by %d to %d users)",
                sum(1 for tid in top_1000 if movie_user_count[tid] >= 1),
                max(movie_user_count[tid] for tid in top_1000) if top_1000 else 0)

    # Extract full movie records from filtered
    movie_map: dict[int, dict] = {}
    with open(FILTERED_FILE) as f:
        for line in f:
            d = json.loads(line)
            movie_map[d["tmdb_id"]] = d

    selected = [movie_map[tid] for tid in top_1000 if tid in movie_map]
    logger.info("Writing %d movies to %s", len(selected), OUTPUT_FILE)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        for m in selected:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Print share stats
    print(f"\nDev set: {len(selected)} movies → {OUTPUT_FILE}")
    print(f"Shared by 4 users: {sum(1 for tid in top_1000 if movie_user_count.get(tid, 0) == 4)}")
    print(f"Shared by 3 users: {sum(1 for tid in top_1000 if movie_user_count.get(tid, 0) == 3)}")
    print(f"Shared by 2 users: {sum(1 for tid in top_1000 if movie_user_count.get(tid, 0) == 2)}")
    print(f"Shared by 1 user:  {sum(1 for tid in top_1000 if movie_user_count.get(tid, 0) == 1)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_dev_setup()
