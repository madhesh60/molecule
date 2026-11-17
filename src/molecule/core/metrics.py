"""metrics.py — in-process latency and error counters per operation.

Kept deliberately tiny: a bounded reservoir of recent durations per
operation, from which `snapshot()` reports count, errors, and p50/p95/max in
milliseconds. No dependency, no background thread, no I/O — cheap enough to
wrap every public call, which is what makes `mol_health` able to answer "is
search getting slow?" from a running server.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager

_RESERVOIR = 512


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._durations: dict[str, deque[float]] = {}
        self._counts: dict[str, int] = {}
        self._errors: dict[str, int] = {}
        self._started = time.time()

    @contextmanager
    def timed(self, operation: str) -> Iterator[None]:
        """Time one call of `operation`; exceptions are counted and re-raised."""
        start = time.perf_counter()
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            self.observe(operation, time.perf_counter() - start, failed)

    def observe(self, operation: str, seconds: float, failed: bool = False) -> None:
        with self._lock:
            self._durations.setdefault(operation, deque(maxlen=_RESERVOIR)).append(seconds)
            self._counts[operation] = self._counts.get(operation, 0) + 1
            if failed:
                self._errors[operation] = self._errors.get(operation, 0) + 1

    def snapshot(self) -> dict:
        with self._lock:
            operations = {}
            for name, durations in self._durations.items():
                ordered = sorted(durations)
                operations[name] = {
                    "count": self._counts[name],
                    "errors": self._errors.get(name, 0),
                    "p50_ms": round(percentile(ordered, 50) * 1000, 2),
                    "p95_ms": round(percentile(ordered, 95) * 1000, 2),
                    "max_ms": round(ordered[-1] * 1000, 2),
                }
            return {"uptime_s": round(time.time() - self._started, 1), "operations": operations}


def percentile(ordered: list[float], pct: float) -> float:
    """Nearest-rank percentile of an already sorted list (0 for an empty one)."""
    if not ordered:
        return 0.0
    rank = max(0, min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1))))
    return ordered[rank]
