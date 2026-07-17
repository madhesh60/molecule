from __future__ import annotations

import sqlite3

from molecule.config import MoleculeConfig
from molecule.core.retriever import rank_candidates


def _matches(returned_id: str, expected_id: str) -> bool:
    """Exact match, or a §-split part of the expected section
    (large sections split into '<heading> §1', '<heading> §2', …)."""
    return returned_id == expected_id or returned_id.startswith(expected_id + " §")


def run_eval(
    queries: list[tuple[str, str]],
    conn: sqlite3.Connection,
    config: MoleculeConfig,
    k: int = 3,
) -> dict:
    """
    Measure retrieval precision on a labeled query set.

    queries: list of (query_text, expected_chunk_id)
    A hit counts if the expected chunk — or any §-split part of it — is in top-k.
    Returns top-k precision, per-query results, and misses.
    """
    if not queries:
        return {"error": "no queries provided"}

    hits, misses = 0, []
    per_query = []

    for query, expected_id in queries:
        results = rank_candidates(query, conn, k, config)
        returned_ids = [r["id"] for r in results]
        rank = next(
            (i + 1 for i, rid in enumerate(returned_ids) if _matches(rid, expected_id)),
            None,
        )
        hit = rank is not None
        if hit:
            hits += 1
        else:
            misses.append({"query": query, "expected": expected_id, "got": returned_ids})
        per_query.append({
            "query": query,
            "expected": expected_id,
            "hit": hit,
            "rank": rank,
            "top_result": returned_ids[0] if returned_ids else None,
        })

    precision = round(hits / len(queries), 3)
    return {
        "total": len(queries),
        "hits": hits,
        f"top{k}_precision": precision,
        "target": 0.80,
        "passes": precision >= 0.80,
        "misses": misses,
        "per_query": per_query,
    }
