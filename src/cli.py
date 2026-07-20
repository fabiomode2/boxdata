import argparse


def build_parsers() -> argparse.ArgumentParser:
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
    embed_p.add_argument("--overwrite", action="store_true",
                         help="Re-embed existing movies via upsert")
    embed_p.add_argument("--db-path", type=str, default=None,
                         help="Custom ChromaDB directory (default: data/chroma_db)")
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
    llm_p.add_argument("--workers", type=int, default=5,
                       help="Concurrent workers for OpenAI provider (default: 5, unused for ollama)")

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
    eval_p.add_argument("--db-path", type=str, default=None,
                        help="Custom ChromaDB directory (default: data/chroma_db)")

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
    rec_p.add_argument("--min-votes", type=int, default=2000,
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
    rec_p.add_argument("--no-watchlist", action="store_true",
                       help="Disable watchlist boost (on by default)")

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

    return parser
