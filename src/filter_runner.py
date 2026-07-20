"""Filter movies from JSONL by vote_average and vote_count thresholds."""
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "data" / "movies.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "movies_filtered.jsonl"


def run_filter(
    min_rating: float = 5.0,
    min_votes: int = 1000,
    input_file: str | None = None,
    output_file: str | None = None,
) -> dict[str, Any]:
    """Filter movies from JSONL, writing only those that pass thresholds.

    Returns statistics dict with counts.
    """
    in_path = Path(input_file) if input_file else DEFAULT_INPUT
    out_path = Path(output_file) if output_file else DEFAULT_OUTPUT

    if not in_path.exists():
        logger.error("%s not found. Run collect first.", in_path)
        return {"error": f"File not found: {in_path}", "filtered": 0, "total": 0}

    total = 0
    passed = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            total += 1
            d = json.loads(line)
            vote_avg = d.get("vote_average", 0) or 0
            vote_cnt = d.get("vote_count", 0) or 0
            if vote_avg >= min_rating and vote_cnt >= min_votes:
                fout.write(line)
                passed += 1

    stats = {
        "input": str(in_path),
        "output": str(out_path),
        "total": total,
        "filtered": passed,
        "dropped": total - passed,
        "min_rating": min_rating,
        "min_votes": min_votes,
    }

    logger.info(
        "Filtered %d/%d movies (rating >= %.1f, votes >= %d) → %s",
        passed, total, min_rating, min_votes, out_path,
    )
    return stats
