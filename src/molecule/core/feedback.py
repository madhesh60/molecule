"""feedback.py — self-learning router.

Agents report whether a fetched chunk was actually useful for a query.
Each vote nudges the chunk's `learned_weight` via a bounded EMA toward
1.5 (useful) or 0.5 (not useful), so the router genuinely improves with
use while a single vote can never dominate:

    new = clamp((1 - alpha) * current + alpha * target, min, max)

With alpha = 0.2, one positive vote from a fresh chunk moves 1.0 -> 1.10.
The weight multiplies into the ranking score in rank_candidates(); the
always_include guarantee is unaffected (those chunks are appended after
top-k regardless of score).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime

from molecule.config import MoleculeConfig


def record_feedback(
    chunk_id: str,
    useful: bool,
    query: str,
    conn: sqlite3.Connection,
    config: MoleculeConfig,
) -> dict:
    row = conn.execute(
        "SELECT learned_weight FROM chunks WHERE chunk_id=?", (chunk_id,)
    ).fetchone()
    if not row:
        return {"error": f"chunk_id not found: {chunk_id}"}

    conn.execute(
        "INSERT INTO chunk_feedback (chunk_id, useful, query, ts) VALUES (?,?,?,?)",
        (chunk_id, int(useful), query, datetime.utcnow().isoformat()),
    )

    current = float(row["learned_weight"] or 1.0)
    target = config.learned_weight_max if useful else config.learned_weight_min
    alpha = config.feedback_ema_alpha
    new = (1 - alpha) * current + alpha * target
    new = max(config.learned_weight_min, min(config.learned_weight_max, new))

    conn.execute(
        "UPDATE chunks SET learned_weight=? WHERE chunk_id=?", (new, chunk_id)
    )
    conn.commit()
    return {
        "chunk_id": chunk_id,
        "useful": useful,
        "learned_weight": round(new, 4),
        "delta": round(new - current, 4),
    }


def feedback_stats(conn: sqlite3.Connection) -> dict:
    """Aggregate feedback counts + the chunks the router has learned about."""
    total, useful = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(useful), 0) FROM chunk_feedback"
    ).fetchone()
    learned = [
        {"chunk_id": r["chunk_id"], "learned_weight": round(r["learned_weight"], 4)}
        for r in conn.execute(
            "SELECT chunk_id, learned_weight FROM chunks "
            "WHERE learned_weight != 1.0 ORDER BY learned_weight DESC"
        ).fetchall()
    ]
    return {"votes": total, "useful": useful, "not_useful": total - useful, "learned": learned}
