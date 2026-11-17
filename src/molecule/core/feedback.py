"""feedback.py — the self-learning router, made query-aware.

Agents report whether a fetched chunk was actually useful for a query. The
first version of this module kept one global weight per chunk: a chunk that
was useful for *deployment* questions got a boost on *every* question, which
slowly bends unrelated rankings toward whatever was popular.

Now each vote remembers the question it was cast for (as an embedding), and
at search time a vote counts in proportion to how similar that old question
is to the current one. It is the same bounded EMA as before, with the step
scaled by a similarity kernel:

    w    = clip((cos(q_now, q_vote) − floor) / (full − floor), 0, 1)
    step = alpha · max(w, leak)
    L   ←  (1 − step) · L + step · target        target = 1.5 useful / 0.5 not

replayed over the chunk's recent votes, oldest first, starting from 1.0.

  * Same question again  → w = 1 → exactly the old behaviour (1.0 → 1.10).
  * Unrelated question   → w = 0 → only the tiny `leak` share applies.
  * L is a convex blend of values in [0.5, 1.5], so it can never leave that
    range and one vote can never dominate.

Votes cast without a query apply everywhere at full strength. The plain
global EMA is still stored in `chunks.learned_weight` for reporting.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np

from molecule.config import MoleculeConfig
from molecule.core.cache import versioned
from molecule.core.embedder import embed_query, embedder_for, vector_is_current
from molecule.db import utc_now_iso


def _ema_step(current: float, target: float, step: float, config: MoleculeConfig) -> float:
    new = (1.0 - step) * current + step * target
    return max(config.learned_weight_min, min(config.learned_weight_max, new))


def record_feedback(
    chunk_id: str,
    useful: bool,
    query: str,
    conn: sqlite3.Connection,
    config: MoleculeConfig,
) -> dict:
    """Store one vote and update the chunk's global weight."""
    row = conn.execute(
        "SELECT learned_weight FROM chunks WHERE chunk_id=?", (chunk_id,)
    ).fetchone()
    if not row:
        return {"error": f"chunk_id not found: {chunk_id}"}

    query = (query or "").strip()
    vector, model = None, None
    if query:
        embedder = embedder_for(config)
        vector = embed_query(query, embedder, config.chunk_embed_tokens).tobytes()
        model = embedder.name

    conn.execute(
        "INSERT INTO chunk_feedback (chunk_id, useful, query, ts, query_embedding, "
        "embedding_model) VALUES (?,?,?,?,?,?)",
        (chunk_id, int(useful), query, utc_now_iso(), vector, model),
    )

    current = float(row["learned_weight"] or 1.0)
    target = config.learned_weight_max if useful else config.learned_weight_min
    new = _ema_step(current, target, config.feedback_ema_alpha, config)
    conn.execute("UPDATE chunks SET learned_weight=? WHERE chunk_id=?", (new, chunk_id))
    conn.commit()
    return {
        "chunk_id": chunk_id,
        "useful": useful,
        "learned_weight": round(new, 4),
        "delta": round(new - current, 4),
        "scope": "similar queries" if query else "all queries",
    }


@dataclass
class _Votes:
    chunk_ids: list[str]        # one entry per vote, oldest first
    targets: np.ndarray         # 1.5 / 0.5 per vote
    has_query: np.ndarray       # bool per vote
    matrix: np.ndarray          # (votes, d); zero rows for votes without a query


def _load_votes(conn: sqlite3.Connection, config: MoleculeConfig) -> _Votes | None:
    rows = conn.execute(
        "SELECT id, chunk_id, useful, query, query_embedding, embedding_model "
        "FROM chunk_feedback ORDER BY id"
    ).fetchall()
    if not rows:
        return None

    # Keep only the most recent votes per chunk — old opinions age out.
    kept: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        kept.setdefault(row["chunk_id"], []).append(row)
    window = max(1, config.feedback_window)
    recent = sorted(
        (r for votes in kept.values() for r in votes[-window:]), key=lambda r: r["id"]
    )

    embedder = embedder_for(config)
    blobs: dict[int, bytes] = {}
    # Votes stored by an older version (or another model) have text but no
    # usable vector: embed them once and keep the result.
    missing = [
        r for r in recent
        if (r["query"] or "").strip()
        and not vector_is_current(r["query_embedding"], r["embedding_model"], embedder)
    ]
    if missing:
        fresh = embedder.encode([r["query"] for r in missing], config.chunk_embed_tokens,
                                query=True)
        for row, vector in zip(missing, fresh):
            blobs[row["id"]] = vector.tobytes()
            conn.execute(
                "UPDATE chunk_feedback SET query_embedding=?, embedding_model=? WHERE id=?",
                (vector.tobytes(), embedder.name, row["id"]),
            )
        conn.commit()

    matrix = np.zeros((len(recent), embedder.dim), dtype=np.float32)
    has_query = np.zeros(len(recent), dtype=bool)
    for i, row in enumerate(recent):
        blob = blobs.get(row["id"], row["query_embedding"])
        if (row["query"] or "").strip() and blob is not None and len(blob) == embedder.dim * 4:
            matrix[i] = np.frombuffer(blob, dtype=np.float32)
            has_query[i] = True
    return _Votes(
        chunk_ids=[r["chunk_id"] for r in recent],
        targets=np.asarray(
            [config.learned_weight_max if r["useful"] else config.learned_weight_min
             for r in recent],
            dtype=np.float32,
        ),
        has_query=has_query,
        matrix=matrix,
    )


def learned_weights(
    conn: sqlite3.Connection,
    config: MoleculeConfig,
    ids: list[str],
    prior: np.ndarray,
    query_vector: np.ndarray,
) -> np.ndarray:
    """Feedback weight of every chunk *for this query*, aligned with `ids`.

    Chunks without votes keep their stored weight (`prior`, normally 1.0)."""
    weights = prior.astype(np.float32).copy()
    try:
        votes = versioned(
            conn, "feedback_votes", lambda: _load_votes(conn, config), domains=("feedback",)
        )
    except sqlite3.OperationalError:   # store without the feedback table
        return weights
    if votes is None:
        return weights

    span = config.feedback_kernel_full - config.feedback_kernel_floor
    if votes.matrix.shape[1] == query_vector.shape[0]:
        sims = votes.matrix @ query_vector
        kernel = np.clip((sims - config.feedback_kernel_floor) / span, 0.0, 1.0)
    else:
        kernel = np.zeros(len(votes.chunk_ids), dtype=np.float32)
    # A vote with a query applies by similarity (never below the leak);
    # a vote without one is a statement about the chunk itself.
    applies = np.where(votes.has_query, np.maximum(kernel, config.feedback_global_leak), 1.0)

    row_of = {chunk_id: i for i, chunk_id in enumerate(ids)}
    replayed: dict[int, float] = {}
    for chunk_id, target, weight in zip(votes.chunk_ids, votes.targets, applies):
        row = row_of.get(chunk_id)
        if row is None:
            continue
        current = replayed.get(row, 1.0)
        replayed[row] = _ema_step(
            current, float(target), config.feedback_ema_alpha * float(weight), config
        )
    for row, value in replayed.items():
        weights[row] = value
    return weights


def feedback_stats(conn: sqlite3.Connection) -> dict:
    """Aggregate feedback counts + the chunks the router has learned about."""
    total, useful, with_query = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(useful), 0), "
        "COALESCE(SUM(CASE WHEN query IS NOT NULL AND query != '' THEN 1 ELSE 0 END), 0) "
        "FROM chunk_feedback"
    ).fetchone()
    learned = [
        {"chunk_id": r["chunk_id"], "learned_weight": round(r["learned_weight"], 4)}
        for r in conn.execute(
            "SELECT chunk_id, learned_weight FROM chunks "
            "WHERE learned_weight != 1.0 ORDER BY learned_weight DESC"
        ).fetchall()
    ]
    return {
        "votes": total,
        "useful": useful,
        "not_useful": total - useful,
        "query_aware_votes": with_query,
        "learned": learned,
    }
