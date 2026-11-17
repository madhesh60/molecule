"""eval.py — measure retrieval quality instead of guessing at it.

A query set is a list of dicts with a `query` and a gold label, either

  * `expected_chunk_id` — one id, or a list of ids (any of them is a hit).
    A `§n` part of a split section counts as the section; or
  * `answer` — a phrase that must appear in the returned chunk's body.
    This label survives re-chunking, which makes it the fair way to compare
    two versions of the chunker.

Reported per set: top-k precision (the project's headline target, ≥ 0.80),
Hit@1, MRR, nDCG@k and search latency percentiles. `run_ablation` re-scores
the same set with one ranking signal removed at a time, which shows what
each signal is actually worth.
"""
from __future__ import annotations

import math
import re
import sqlite3
import time
from dataclasses import replace

from molecule.config import MoleculeConfig
from molecule.core.metrics import percentile
from molecule.core.retriever import clamp_k, score_chunks

_WS = re.compile(r"\s+")
TARGET_PRECISION = 0.80


def _matches(returned_id: str, expected_id: str) -> bool:
    """Exact match, or a §-split part of the expected section
    (large sections split into '<heading> §1', '<heading> §2', …)."""
    return returned_id == expected_id or returned_id.startswith(expected_id + " §")


def _normalise_query(item) -> dict:
    """Accept (query, expected_id) tuples as well as dicts."""
    if isinstance(item, dict):
        return item
    query, expected = item
    return {"query": query, "expected_chunk_id": expected}


def _is_gold(conn: sqlite3.Connection, chunk_id: str, query: dict) -> bool:
    if "answer" in query:
        row = conn.execute("SELECT content FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
        return bool(row) and _WS.sub(" ", query["answer"]) in _WS.sub(" ", row["content"])
    expected = query["expected_chunk_id"]
    expected_ids = [expected] if isinstance(expected, str) else list(expected)
    return any(_matches(chunk_id, e) for e in expected_ids)


def run_eval(
    queries: list,
    conn: sqlite3.Connection,
    config: MoleculeConfig,
    k: int = 3,
) -> dict:
    """Score a labeled query set. See the module docstring for label formats."""
    if not queries:
        return {"error": "no queries provided"}
    k = clamp_k(k, config)
    depth = max(k, 10)   # MRR looks a little past k so near-misses are visible

    hits = top1 = 0
    reciprocal = ndcg = 0.0
    misses: list[dict] = []
    per_query: list[dict] = []
    latencies: list[float] = []

    for item in queries:
        query = _normalise_query(item)
        start = time.perf_counter()
        scores = score_chunks(query["query"], conn, config)
        ranked = [scores.corpus.ids[int(row)] for row in scores.order()[:depth]]
        latencies.append(time.perf_counter() - start)

        rank = next(
            (i + 1 for i, cid in enumerate(ranked) if _is_gold(conn, cid, query)), None
        )
        hit = rank is not None and rank <= k
        hits += hit
        top1 += rank == 1
        if rank:
            reciprocal += 1.0 / rank
        if hit:
            ndcg += 1.0 / math.log2(rank + 1)
        else:
            misses.append({
                "query": query["query"],
                "expected": query.get("expected_chunk_id", query.get("answer")),
                "got": ranked[:k],
            })
        per_query.append({
            "query": query["query"],
            "expected": query.get("expected_chunk_id", query.get("answer")),
            "hit": hit,
            "rank": rank if hit else None,
            "top_result": ranked[0] if ranked else None,
        })

    n = len(queries)
    precision = round(hits / n, 3)
    ordered = sorted(latencies)
    return {
        "total": n,
        "hits": hits,
        f"top{k}_precision": precision,
        "hit_at_1": round(top1 / n, 3),
        "mrr": round(reciprocal / n, 3),
        f"ndcg_at_{k}": round(ndcg / n, 3),
        "latency_ms": {
            "p50": round(percentile(ordered, 50) * 1000, 2),
            "p95": round(percentile(ordered, 95) * 1000, 2),
        },
        "target": TARGET_PRECISION,
        "passes": precision >= TARGET_PRECISION,
        "misses": misses,
        "per_query": per_query,
    }


# One ranking signal removed (or isolated) at a time.
ABLATIONS: tuple[tuple[str, dict], ...] = (
    ("full hybrid", {}),
    ("− keyword signal", {"keyword_weight": 0.0}),
    ("− heading boost", {"heading_boost": 0.0}),
    ("− query expansion", {"query_expansion": False}),
    ("dense only", {"keyword_weight": 0.0, "heading_boost": 0.0}),
    ("lexical only", {"emb_weight": 0.0}),
)


def run_ablation(
    queries: list,
    conn: sqlite3.Connection,
    config: MoleculeConfig,
    k: int = 3,
) -> list[dict]:
    """Re-run the eval with each ranking signal switched off in turn."""
    rows = []
    for name, overrides in ABLATIONS:
        result = run_eval(queries, conn, replace(config, **overrides), k)
        rows.append({
            "variant": name,
            f"top{k}_precision": result.get(f"top{k}_precision"),
            "hit_at_1": result.get("hit_at_1"),
            "mrr": result.get("mrr"),
        })
    return rows
