from __future__ import annotations

import pytest

from molecule.core.metrics import Metrics, percentile


def test_percentile_nearest_rank():
    data = [float(i) for i in range(1, 101)]
    assert percentile(data, 50) == pytest.approx(50.0, abs=1.0)
    assert percentile(data, 95) == pytest.approx(95.0, abs=1.0)
    assert percentile([], 95) == 0.0
    assert percentile([7.0], 50) == 7.0


def test_timed_records_count_and_latency():
    metrics = Metrics()
    for _ in range(3):
        with metrics.timed("search"):
            pass
    op = metrics.snapshot()["operations"]["search"]
    assert op["count"] == 3 and op["errors"] == 0
    assert 0 <= op["p50_ms"] <= op["p95_ms"] <= op["max_ms"]


def test_errors_are_counted_and_re_raised():
    metrics = Metrics()
    with pytest.raises(ValueError):
        with metrics.timed("ingest"):
            raise ValueError("bad file")
    assert metrics.snapshot()["operations"]["ingest"] == {
        **metrics.snapshot()["operations"]["ingest"], "count": 1, "errors": 1,
    }


def test_reservoir_is_bounded():
    metrics = Metrics()
    for i in range(2000):
        metrics.observe("search", i / 1000)
    assert metrics.snapshot()["operations"]["search"]["count"] == 2000
    assert len(metrics._durations["search"]) <= 512
