#!/usr/bin/env python3
"""
BoxData — Movie Recommendation System

Usage:
  python main.py collect  --start 1950 --end 2026 --max-per-year 10
  python main.py embed
  python main.py count
  python main.py clear-db
  python main.py recommend --ids 550 680
  python main.py recommend --user alice bob  [--n 10]
"""
import argparse

from src.vector_store import count_movies


def main() -> None:
    parser = argparse.ArgumentParser(description="BoxData — movie recommendation system")
    sub = parser.add_subparsers(dest="command", required=True)

    collect_p = sub.add_parser("collect", help="Fetch movies from TMDB → JSONL")
    collect_p.add_argument("--start", type=int, default=1900)
    collect_p.add_argument("--end", type=int, default=None)
    collect_p.add_argument("--max-per-year", type=int, default=100)
    collect_p.add_argument("--test", action="store_true", help="Only 2 years")
    collect_p.add_argument("--skip-credits", action="store_true", help="Skip TMDB credits")

    sub.add_parser("embed", help="Generate embeddings from JSONL → ChromaDB")
    sub.add_parser("count", help="Show movie count in ChromaDB")
    sub.add_parser("clear-db", help="Delete ALL embeddings from ChromaDB")

    analyze_p = sub.add_parser("analyze", help="Analyze your Letterboxd taste profile")
    analyze_p.add_argument("--export", type=str, required=True,
                           help="Path to Letterboxd export directory")

    rec_p = sub.add_parser("recommend", help="Recommend movies for 1+ users")
    rec_ids = rec_p.add_mutually_exclusive_group()
    rec_ids.add_argument("--ids", type=int, nargs="+", metavar="TMDB_ID",
                         help="TMDB IDs of movies one user likes")
    rec_ids.add_argument("--user", type=str, nargs="+", metavar="USERNAME",
                         help="Letterboxd usernames to scrape")
    rec_ids.add_argument("--export", type=str, metavar="DIR",
                         help="Path to Letterboxd export directory")
    rec_p.add_argument("--n", type=int, default=10, help="Number of recommendations")
    rec_p.add_argument("--min-votes", type=int, default=10,
                        help="Minimum vote_count filter")

    args = parser.parse_args()

    if args.command == "collect":
        from collect import run_collect
        run_collect(args.start, args.end, args.max_per_year, args.test, args.skip_credits)
    elif args.command == "embed":
        from embed import run_embed
        run_embed()
    elif args.command == "count":
        print(f"Movies in ChromaDB: {count_movies()}")
    elif args.command == "clear-db":
        from src.vector_store import delete_collection
        delete_collection()
        print("ChromaDB collection deleted. Run `python main.py embed` to rebuild.")
    elif args.command == "analyze":
        _run_analyze(args)
    elif args.command == "recommend":
        _run_recommend(args)




def _run_recommend(args: argparse.Namespace) -> None:
    from src.recommender import fetch_letterboxd_watched, get_taste_vector, \
        recommend_for_group, taste_similarity

    user_weighted: list[dict[int, float]] = []

    if args.export:
        print(f"Loading Letterboxd export: {args.export}...")
        weighted = _export_to_weighted(args.export)
        if weighted:
            print(f"  Found {len(weighted)} rated films.")
            vec, found_ids, total = get_taste_vector(weighted)
            print(f"  → {len(found_ids)}/{total} in ChromaDB")
            if found_ids:
                user_weighted.append(weighted)
    elif args.user:
        for username in args.user:
            print(f"Scraping Letterboxd: {username}...")
            weighted = fetch_letterboxd_watched(username)
            print(f"  Found {len(weighted)} films. Checking embeddings...")
            vec, found_ids, total = get_taste_vector(weighted)
            print(f"  → {len(found_ids)}/{total} in ChromaDB")
            if found_ids:
                user_weighted.append(weighted)

        if len(user_weighted) >= 2:
            sim = taste_similarity(user_weighted[0], user_weighted[1])
            print(f"\nTaste similarity: {sim['similarity_pct']}% "
                  f"(A: {sim['user_a_found']}/{sim['user_a_total']}, "
                  f"B: {sim['user_b_found']}/{sim['user_b_total']})")
    else:
        user_weighted.append({tid: 1.0 for tid in args.ids})

    flattened = [tid for w in user_weighted for tid in w]
    if not flattened:
        print("No user data to recommend from.")
        return

    print(f"\nFinding recommendations for {len(user_weighted)} user(s)...")
    recs = recommend_for_group(user_weighted, n=args.n, min_vote_count=args.min_votes)

    if not recs:
        print("No recommendations found.")
        return

    print(f"\n{'':4s}Title{'':35s}Year  Genres{'':20s}Rating  Score")
    print("-" * 85)
    for i, r in enumerate(recs, 1):
        title = r["title"][:38]
        genres = r["genres"][:22]
        year = r["year"]
        rating = r["vote_average"]
        score = r["rerank_score"]
        print(f"{i:2d}. {title:38s} {year:<5d} {genres:22s} {rating:<5.1f}  {score:.4f}")
    print()
    for r in recs:
        print(f"  {r['url']} — {r['title']}")


def _export_to_weighted(export_dir: str) -> dict[int, float]:
    from src.profile import parse_export, resolve_tmdb_ids
    profile = parse_export(export_dir)
    profile = resolve_tmdb_ids(profile, resolve_watchlist=False)
    weighted: dict[int, float] = {}
    for f in profile.watched:
        if f.tmdb_id and f.rating is not None:
            weighted[f.tmdb_id] = f.rating / 5.0
    return weighted


def _run_analyze(args: argparse.Namespace) -> None:
    from src.profile import parse_export, resolve_tmdb_ids, analyze_profile, format_profile_report

    print("Parsing Letterboxd export...", flush=True)
    profile = parse_export(args.export)
    print(f"  {len(profile.watched)} watched, {len(profile.watchlist)} in watchlist", flush=True)

    print("Resolving TMDB IDs...", flush=True)
    profile = resolve_tmdb_ids(profile, resolve_watchlist=True)
    resolved = sum(1 for f in profile.watched if f.tmdb_id)
    print(f"  Resolved {resolved}/{len(profile.watched)} films", flush=True)

    print("Analyzing profile...", flush=True)
    stats = analyze_profile(profile)

    print()
    print(format_profile_report(stats))



if __name__ == "__main__":
    main()
