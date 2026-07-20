import sys
import time

_TIMERS: list[tuple[str, float]] = []


def _timer_start(label: str) -> None:
    _TIMERS.append((label, time.perf_counter()))


def _timer_stop() -> str | None:
    if not _TIMERS:
        return None
    label, start = _TIMERS.pop()
    elapsed = time.perf_counter() - start
    print(f"  ⏱  {label}: {elapsed:.2f}s", file=sys.stderr, flush=True)
    return label
