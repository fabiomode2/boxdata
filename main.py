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
import json
import logging
from pathlib import Path

from src.recommender import rank_watchlist
from src.vector_store import count_movies

logger = logging.getLogger(__name__)


def _show_help(parser: argparse.ArgumentParser) -> None:
    print("""
 BoxData — Movie Recommendation System

 USOS PRINCIPALES:
   python main.py help                    Mostrar esta ayuda
   python main.py <comando> -h            Ayuda detallada de un comando

 COLECTAR DATOS:
   python main.py collect --start 1900 --end 2026 --max-per-year 100
         --start AÑO       Año inicial (default: 1900)
         --end AÑO         Año final (default: año actual)
         --max-per-year N  Películas por año (default: 100)
         --test            Solo 2 años (quick test)
          --skip-credits    Saltar fetch de créditos
          --skip-imdb       Saltar enrichment de IMDb API (subgéneros, metacritic)

  EMBEDDINGS:
    python main.py embed                    Generar embeddings (Ollama + nomic-embed-text)
    python main.py filter                   Filtrar peliculas por rating y votos

  LLM ENRICHMENT:
    python main.py llm-text                 Generar textos descriptivos con LLM

  EVALUACIÓN:
    python main.py evaluate --users alice bob
          --k N             Número de folds (default: 5)
          --output FILE     Guardar resultados en JSON

  ESTADO:
   python main.py count                    Películas en ChromaDB
   python main.py clear-db                 Borrar toda la base vectorial

 ANALIZAR PERFIL:
   python main.py analyze --export DIR
         --export DIR      Ruta al export .csv de Letterboxd

 RECOMENDACIONES:
   python main.py recommend --ids 550 680
         --ids TMDB_ID     IDs de TMDB que le gustan a un usuario

   python main.py recommend --user alice bob
         --user USER       Usuarios de Letterboxd (scrapea diario + ratings)
         --no-cache        Ignorar caché

   python main.py recommend --export DIR
         --export DIR      Export .csv de Letterboxd

   python main.py recommend --query "gritty noir"
         --query TEXTO     Búsqueda semántica por texto

   python main.py recommend --export DIR --query "noir"
         --query-weight W  Peso de la query vs perfil (0-1, default 0.3)

  OPCIONES COMUNES:
    --n N                 Número de resultados (default: 10)
    --min-votes N         Votaciones mínimas en TMDB (default: 2000)
    --sigma SIGMA         Suavizado del kernel (default: 0.3)
    --max-distance DIST   Distancia máxima para considerar historial
                          (0-1, -1=sin límite, default: 0.3)
    --diversity LAM       Diversidad 0=relevancia, 1=variedad (default: 0)
""".strip())


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="BoxData — movie recommendation system")
    sub = parser.add_subparsers(dest="command", required=True)

    collect_p = sub.add_parser("collect", help="Fetch movies from TMDB → JSONL")
    collect_p.add_argument("--start", type=int, default=1900)
    collect_p.add_argument("--end", type=int, default=None)
    collect_p.add_argument("--max-per-year", type=int, default=100)
    collect_p.add_argument("--test", action="store_true", help="Only 2 years")
    collect_p.add_argument("--skip-credits", action="store_true", help="Skip TMDB credits")
    collect_p.add_argument("--skip-imdb", action="store_true", help="Skip IMDb API enrichment")
    collect_p.add_argument("--skip-reviews", action="store_true", help="Skip fetching TMDB reviews")

    embed_p = sub.add_parser("embed", help="Generate embeddings from JSONL → ChromaDB")
    embed_p.add_argument("--input", type=str, default=None,
                         help="Path to JSONL file (default: data/movies.jsonl)")
    sub.add_parser("count", help="Show movie count in ChromaDB")
    sub.add_parser("clear-db", help="Delete ALL embeddings from ChromaDB")
    sub.add_parser("explore", help="UMAP + HDBSCAN + Plotly visualization of the embedding space")

    filter_p = sub.add_parser("filter", help="Filter movies by rating and vote count thresholds")
    filter_p.add_argument("--min-rating", type=float, default=5.0,
                          help="Minimum TMDB vote_average (default: 5.0)")
    filter_p.add_argument("--min-votes", type=int, default=1000,
                          help="Minimum TMDB vote_count (default: 1000)")
    filter_p.add_argument("--input", type=str, default=None,
                          help="Path to input JSONL (default: data/movies.jsonl)")
    filter_p.add_argument("--output", type=str, default=None,
                          help="Path to output JSONL (default: data/movies_filtered.jsonl)")

    llm_p = sub.add_parser("llm-text", help="Generate rich text descriptions via LLM")
    llm_p.add_argument("--model", type=str, default="llama3.2:3b",
                       help="Model ID (default: llama3.2:3b)")
    llm_p.add_argument("--provider", type=str, default="ollama", choices=["ollama", "openai"],
                       help="LLM provider (default: ollama)")
    llm_p.add_argument("--api-base", type=str, default=None,
                       help="API base URL for openai provider")
    llm_p.add_argument("--api-key", type=str, default=None,
                       help="API key for openai provider")
    llm_p.add_argument("--ollama-host", type=str, default="http://localhost:11434",
                       help="Ollama host URL (default: http://localhost:11434)")
    llm_p.add_argument("--input", type=str, default="data/movies_filtered.jsonl",
                       help="Path to input JSONL (default: data/movies_filtered.jsonl)")
    llm_p.add_argument("--output", type=str, default="data/movies_llm_text.jsonl",
                       help="Path to output JSONL (default: data/movies_llm_text.jsonl)")
    llm_p.add_argument("--delay", type=float, default=0.5,
                       help="Delay between API calls in seconds (default: 0.5)")

    analyze_p = sub.add_parser("analyze", help="Analyze your Letterboxd taste profile")
    analyze_p.add_argument("--export", type=str, required=True,
                           help="Path to Letterboxd export directory")

    eval_p = sub.add_parser("evaluate", help="Evaluate prediction accuracy via k-fold CV")
    eval_p.add_argument("--users", type=str, nargs="+", metavar="USERNAME",
                        help="Letterboxd usernames to evaluate")
    eval_p.add_argument("--export", type=str, metavar="DIR",
                        help="Path to Letterboxd export directory")
    eval_p.add_argument("--k", type=int, default=5,
                        help="Number of folds (default: 5)")
    eval_p.add_argument("--sigma", type=float, default=0.3,
                        help="Kernel bandwidth (default: 0.3)")
    eval_p.add_argument("--output", type=str, default=None,
                        help="Save results to JSON file")
    eval_p.add_argument("--no-cache", action="store_true",
                        help="Ignore cached scraped profiles")

    rec_p = sub.add_parser("recommend", help="Recommend movies for 1+ users")
    rec_p.add_argument("--query", type=str, metavar="TEXT",
                       help="Free-text description of what you want to watch")
    rec_p.add_argument("--query-weight", type=float, default=0.3,
                       help="How much the query influences results (0–1, default 0.3)")
    rec_ids = rec_p.add_mutually_exclusive_group()
    rec_ids.add_argument("--ids", type=int, nargs="+", metavar="TMDB_ID",
                         help="TMDB IDs of movies one user likes")
    rec_ids.add_argument("--user", type=str, nargs="+", metavar="USERNAME",
                         help="Letterboxd usernames to scrape")
    rec_ids.add_argument("--export", type=str, metavar="DIR",
                         help="Path to Letterboxd export directory")
    rec_p.add_argument("--n", type=int, default=10, help="Number of recommendations")
    rec_p.add_argument(    "--min-votes", type=int, default=2000,
                        help="Minimum vote_count filter")
    rec_p.add_argument("--no-cache", action="store_true",
                        help="Ignore cached scraped profiles")
    rec_p.add_argument("--sigma", type=float, default=0.3, metavar="SIGMA",
                       help="Kernel bandwidth for rating prediction (default 0.3)")
    rec_p.add_argument("--auto-sigma", action="store_true",
                       help="Auto-calibrate sigma per user via cross-validation (overrides --sigma)")
    rec_p.add_argument("--max-distance", type=float, default=0.3, metavar="DIST",
                       help="Max cosine distance for history matches (0-1, -1=no limit, default 0.3)")
    rec_p.add_argument("--diversity", type=float, default=0.0, metavar="LAM",
                        help="Diversity in results (0=relevance, 1=variety, default 0)")
    rec_p.add_argument("--genre", type=str, default=None, metavar="GENRES",
                        help="Filter by genres (comma-separated, e.g. 'Drama,Thriller')")
    rec_p.add_argument("--min-year", type=int, default=None, metavar="YEAR",
                        help="Minimum release year (e.g. 2000)")
    rec_p.add_argument("--max-year", type=int, default=None, metavar="YEAR",
                        help="Maximum release year (e.g. 2020)")
    rec_p.add_argument("--min-rating", type=float, default=6.5, metavar="RATING",
                        help="Minimum TMDB vote average (default 6.5)")
    rec_p.add_argument("--randomize", action="store_true",
                         help="Randomize discovery seeds for varied results")
    rec_p.add_argument("--no-feature-rerank", action="store_true",
                         help="Disable feature reranking (director/decade/cast bonus, on by default)")

    rank_p = sub.add_parser("rank-watchlist", help="Rank watchlist by predicted rating")
    rank_p.add_argument("--export", type=str, required=True,
                        help="Path to Letterboxd export directory")
    rank_p.add_argument("--n", type=int, default=None,
                        help="Number of top results (default: all)")
    rank_p.add_argument("--sigma", type=float, default=0.3, metavar="SIGMA",
                        help="Kernel bandwidth (default 0.3)")
    rank_p.add_argument("--auto-sigma", action="store_true",
                        help="Auto-calibrate sigma via cross-validation")
    rank_p.add_argument("--min-votes", type=int, default=2000,
                        help="Minimum TMDB vote_count")
    rank_p.add_argument("--genre", type=str, default=None, metavar="GENRES",
                        help="Filter by genres (comma-separated)")
    rank_p.add_argument("--min-year", type=int, default=None, metavar="YEAR",
                        help="Minimum release year")
    rank_p.add_argument("--max-year", type=int, default=None, metavar="YEAR",
                        help="Maximum release year")
    rank_p.add_argument("--min-rating", type=float, default=6.5, metavar="RATING",
                        help="Minimum TMDB vote average (default 6.5)")

    sub.add_parser("help", help="Show this help message")

    args = parser.parse_args()

    if args.command == "help":
        _show_help(parser)
        return

    if args.command == "collect":
        from src.collect_runner import run_collect
        run_collect(args.start, args.end, args.max_per_year, args.test, args.skip_credits, args.skip_imdb, args.skip_reviews)
    elif args.command == "embed":
        from src.embed_runner import run_embed
        run_embed(input_file=args.input)
    elif args.command == "count":
        logger.info("Movies in ChromaDB: %s", count_movies())
    elif args.command == "clear-db":
        from src.vector_store import delete_collection
        delete_collection()
        logger.info("ChromaDB collection deleted. Run `python main.py embed` to rebuild.")
    elif args.command == "analyze":
        _run_analyze(args)
    elif args.command == "evaluate":
        _run_evaluate(args)
    elif args.command == "recommend":
        _run_recommend(args)
    elif args.command == "explore":
        from src.explorer import run_explore
        run_explore()
    elif args.command == "filter":
        from src.filter_runner import run_filter
        stats = run_filter(
            min_rating=args.min_rating,
            min_votes=args.min_votes,
            input_file=args.input,
            output_file=args.output,
        )
        if "error" not in stats:
            print(f"\nFiltered: {stats['filtered']}/{stats['total']} movies kept "
                  f"(dropped {stats['dropped']})")
            print(f"Output: {stats['output']}")
    elif args.command == "llm-text":
        from src.llm_text_generator import run_llm_text
        api_key = args.api_key
        if args.provider == "openai" and not api_key:
            import os
            api_key = os.getenv("OPENCODE_API_KEY") or os.getenv("OPENAI_API_KEY")
        stats = run_llm_text(
            input_file=args.input,
            output_file=args.output,
            model=args.model,
            provider=args.provider,
            api_base=args.api_base,
            api_key=api_key,
            ollama_host=args.ollama_host,
            delay=args.delay,
        )
        if "error" not in stats:
            print(f"\nGenerated: {stats['generated']}/{stats['total']} movies "
                  f"({stats['elapsed_min']} min)")
            print(f"Output: {stats['output']}")
    elif args.command == "rank-watchlist":
        _run_rank_watchlist(args)



def _run_recommend(args: argparse.Namespace) -> None:
    from src.recommender import fetch_letterboxd_watched, get_taste_vector, \
        recommend_for_group, recommend_from_query, recommend_hybrid, taste_similarity

    user_weighted: list[dict[int, float]] = []

    if args.export:
        logger.info("Loading Letterboxd export: %s...", args.export)
        weighted = _export_to_weighted(args.export)
        if weighted:
            logger.info("  Found %d rated films.", len(weighted))
            vec, found_ids, total = get_taste_vector(weighted)
            logger.info("  → %d/%d in ChromaDB", len(found_ids), total)
            if found_ids:
                user_weighted.append(weighted)
    elif args.user:
        for username in args.user:
            logger.info("Scraping Letterboxd: %s...", username)
            weighted = fetch_letterboxd_watched(username, no_cache=args.no_cache)
            logger.info("  Found %d films. Checking embeddings...", len(weighted))
            vec, found_ids, total = get_taste_vector(weighted)
            logger.info("  → %d/%d in ChromaDB", len(found_ids), total)
            if found_ids:
                user_weighted.append(weighted)

        if len(user_weighted) >= 2:
            sim = taste_similarity(user_weighted[0], user_weighted[1])
            logger.info("\nTaste similarity: %s%% "
                        "(A: %s/%s, "
                        "B: %s/%s)",
                        sim['similarity_pct'],
                        sim['user_a_found'], sim['user_a_total'],
                        sim['user_b_found'], sim['user_b_total'])
    elif args.ids:
        user_weighted.append({tid: 1.0 for tid in args.ids})

    if not user_weighted and not args.query:
        logger.info("No user data to recommend from.")
        return

    genre_filter = [g.strip() for g in args.genre.split(",") if g.strip()] if args.genre else None

    if not user_weighted and args.query:
        recs = recommend_from_query(
            args.query, n=args.n, min_vote_count=args.min_votes,
            diversity=args.diversity, genre_filter=genre_filter,
            min_year=args.min_year, max_year=args.max_year,
            min_rating=args.min_rating,
            randomize=args.randomize,
        )
        _print_recommendations(recs)
        return

    logger.info("\nFinding recommendations for %d user(s)...", len(user_weighted))
    effective_sigma = None if args.auto_sigma else args.sigma
    feature_rerank = not args.no_feature_rerank
    if args.query:
        recs = recommend_hybrid(
            user_weighted, args.query,
            n=args.n, min_vote_count=args.min_votes,
            query_weight=args.query_weight,
            sigma=effective_sigma, max_distance=args.max_distance,
            diversity=args.diversity, genre_filter=genre_filter,
            min_year=args.min_year, max_year=args.max_year,
            min_rating=args.min_rating,
            randomize=args.randomize,
            feature_rerank=feature_rerank,
        )
    else:
        recs = recommend_for_group(
            user_weighted, n=args.n, min_vote_count=args.min_votes,
            sigma=effective_sigma, max_distance=args.max_distance,
            diversity=args.diversity, genre_filter=genre_filter,
            min_year=args.min_year, max_year=args.max_year,
            min_rating=args.min_rating,
            randomize=args.randomize,
            feature_rerank=feature_rerank,
        )

    _print_recommendations(recs)


def _print_recommendations(recs: list[dict]) -> None:
    if not recs:
        return
    out = []
    out.append(f"\n{'':4s}Title{'':35s}Year  Genres{'':20s}Rating  Score")
    out.append("-" * 85)
    for i, r in enumerate(recs, 1):
        title = r["title"][:38]
        genres = r["genres"][:22]
        year = r["year"]
        rating = r["vote_average"]
        score = r["rerank_score"]
        out.append(f"{i:2d}. {title:38s} {year:<5d} {genres:22s} {rating:<5.1f}  {score:.4f}")
    out.append("")
    for r in recs:
        out.append(f"  {r['url']} — {r['title']}")
    print("\n".join(out))


def _export_to_weighted(export_dir: str) -> dict[int, float]:
    from src.profile import parse_export, resolve_tmdb_ids
    profile = parse_export(export_dir)
    profile = resolve_tmdb_ids(profile, resolve_watchlist=False)
    weighted: dict[int, float] = {}
    for f in profile.watched:
        if f.tmdb_id:
            w = f.rating / 5.0 if f.rating is not None else 0.6
            weighted[f.tmdb_id] = w
    return weighted


def _run_analyze(args: argparse.Namespace) -> None:
    from src.profile import parse_export, resolve_tmdb_ids, analyze_profile, format_profile_report

    logger.info("Parsing Letterboxd export...")
    profile = parse_export(args.export)
    logger.info("  %d watched, %d in watchlist", len(profile.watched), len(profile.watchlist))

    logger.info("Resolving TMDB IDs...")
    profile = resolve_tmdb_ids(profile, resolve_watchlist=False)
    resolved = sum(1 for f in profile.watched if f.tmdb_id)
    logger.info("  Resolved %d/%d films", resolved, len(profile.watched))

    logger.info("Analyzing profile...")
    stats = analyze_profile(profile)

    print()
    print(format_profile_report(stats))


def _run_evaluate(args: argparse.Namespace) -> None:
    from src.evaluator import evaluate_users, _print_evaluation_results, _average_results
    from src.recommender import fetch_letterboxd_watched
    from src.profile import parse_export, resolve_tmdb_ids

    user_weighted: list[dict[int, float]] = []
    usernames: list[str] = []

    if args.export:
        logger.info("Loading Letterboxd export: %s...", args.export)
        profile = parse_export(args.export)
        profile = resolve_tmdb_ids(profile, resolve_watchlist=False)
        weighted: dict[int, float] = {}
        for f in profile.watched:
            if f.tmdb_id:
                w = f.rating / 5.0 if f.rating is not None else 0.6
                weighted[f.tmdb_id] = w
        if weighted:
            user_weighted.append(weighted)
            usernames.append(profile.username)
    elif args.users:
        for username in args.users:
            logger.info("Scraping Letterboxd: %s...", username)
            weighted = fetch_letterboxd_watched(username, no_cache=args.no_cache)
            if weighted:
                user_weighted.append(weighted)
                usernames.append(username)

    if not user_weighted:
        logger.info("No user data to evaluate.")
        return

    logger.info("Evaluating %d user(s) with %d-fold CV...", len(user_weighted), args.k)
    results = evaluate_users(
        user_weighted, usernames=usernames, n_folds=args.k, sigma=args.sigma,
    )

    _print_evaluation_results(results)

    avg = _average_results(results)
    if avg:
        print(f"  {'─' * 85}")
        _print_evaluation_results([avg])

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(results, indent=2, default=str))
        logger.info("Results saved to %s", out_path)


def _run_rank_watchlist(args: argparse.Namespace) -> None:
    genre_filter = [g.strip() for g in args.genre.split(",") if g.strip()] if args.genre else None
    sigma = None if args.auto_sigma else args.sigma

    recs = rank_watchlist(
        args.export,
        n=args.n,
        min_vote_count=args.min_votes,
        sigma=sigma,
        genre_filter=genre_filter,
        min_year=args.min_year,
        max_year=args.max_year,
        min_rating=args.min_rating,
        auto_sigma=args.auto_sigma,
    )

    if not recs:
        return

    out = []
    out.append(f"\n{'':4s}Title{'':35s}Year  Genres{'':20s}Predicted★  Score")
    out.append("-" * 90)
    for i, r in enumerate(recs, 1):
        title = r["title"][:38]
        genres = r["genres"][:22]
        year = r["year"]
        stars = r["predicted_stars"]
        score = r["predicted_rating"]
        out.append(f"{i:2d}. {title:38s} {year:<5d} {genres:22s} {stars:<5.2f}★  {score:.4f}")
    out.append("")
    for r in recs:
        out.append(f"  {r['url']} — {r['title']}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
