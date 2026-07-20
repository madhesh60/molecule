"""retriever.py — router index, hybrid ranking, chunk fetch.

Hot-path design: everything about the corpus that doesn't depend on the query
(embedding matrix, metadata rows, BM25 term counts, heading token sets) is
built once and memoized via cache.versioned(), keyed on the connection's
total_changes counter — any ingest/feedback/memory write invalidates it
automatically. A search therefore costs one query embedding + one matrix
multiply + dict lookups.
"""
from __future__ import annotations

import json
import sqlite3

from molecule.config import MoleculeConfig
from molecule.core.cache import versioned
from molecule.core.embedder import cosine_similarity, embed_query, load_embeddings
from molecule.core.scoring import (
    bm25_scores_indexed,
    build_bm25_index,
    heading_match_score_tokens,
    tokenize,
)


def load_router_index(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT chunk_id, heading_path, summary, token_count, always_include, weight FROM chunks"
    ).fetchall()
    return [
        {
            "chunk_id": r["chunk_id"],
            "heading_path": json.loads(r["heading_path"]),
            "summary": r["summary"] or "",
            "token_count": r["token_count"],
            "always_include": bool(r["always_include"]),
            "weight": r["weight"],
        }
        for r in rows
    ]


def _chunk_corpus(conn: sqlite3.Connection) -> dict:
    """Query-independent chunk state: embeddings, metadata, BM25 index, heading tokens."""
    ids, matrix = load_embeddings(conn, "chunks", "chunk_id")
    meta: dict[str, dict] = {}
    docs: dict[str, str] = {}
    if ids:
        for r in conn.execute(
            f"SELECT chunk_id, heading_path, summary, content, weight, learned_weight, "
            f"always_include FROM chunks WHERE chunk_id IN ({','.join('?' * len(ids))})",
            ids,
        ).fetchall():
            heading_path = json.loads(r["heading_path"])
            meta[r["chunk_id"]] = {
                "summary": r["summary"] or "",
                "learned": float(r["learned_weight"] or 1.0),
                "always_include": bool(r["always_include"]),
                "head_tokens": set(tokenize(" ".join(heading_path))),
            }
            # BM25 over heading + summary + content — headings are strong signals.
            docs[r["chunk_id"]] = (
                f"{' '.join(heading_path)} {r['summary'] or ''} {r['content']}"
            )
    return {"ids": ids, "matrix": matrix, "meta": meta, "bm25": build_bm25_index(docs)}


def _memory_corpus(conn: sqlite3.Connection) -> dict:
    """Query-independent active-memory state. Note: expiry uses datetime('now')
    inside the SQL, so a memory crossing its expires_at between writes could be
    served one cache generation late — acceptable for 7-day expiries."""
    ids, matrix = load_embeddings(
        conn, "memories", "memory_id",
        filter_clause="status='active' AND (expires_at IS NULL OR expires_at > datetime('now'))",
    )
    meta: dict[str, dict] = {}
    docs: dict[str, str] = {}
    if ids:
        for r in conn.execute(
            f"SELECT memory_id, text, weight FROM memories "
            f"WHERE memory_id IN ({','.join('?' * len(ids))})",
            ids,
        ).fetchall():
            meta[r["memory_id"]] = {"text": r["text"], "weight": float(r["weight"] or 1.0)}
            docs[r["memory_id"]] = r["text"]
    return {"ids": ids, "matrix": matrix, "meta": meta, "bm25": build_bm25_index(docs)}


def rank_candidates(
    query: str,
    conn: sqlite3.Connection,
    k: int,
    config: MoleculeConfig,
) -> list[dict]:
    # Guard nonsense k from tool callers: 0/negative → config default, huge → cap.
    if k <= 0:
        k = config.default_k
    k = min(k, 100)
    q_vec = embed_query(query, config.embedding_model)
    q_tokens = set(tokenize(query))

    chunks = versioned(conn, "chunk_corpus", lambda: _chunk_corpus(conn))
    mems = versioned(conn, "memory_corpus", lambda: _memory_corpus(conn))

    results: list[dict] = []

    if chunks["ids"]:
        sims = cosine_similarity(q_vec, chunks["matrix"])
        kw = bm25_scores_indexed(query, chunks["bm25"])
        for i, cid in enumerate(chunks["ids"]):
            m = chunks["meta"].get(cid)
            if not m:
                continue
            emb_score = float(sims[i])
            kw_score = kw.get(cid, 0.0)
            head_score = heading_match_score_tokens(q_tokens, m["head_tokens"])
            learned = m["learned"]
            # Never apply always_include weight to ranking score — it causes every
            # query to return the same always_include chunk at rank 1.
            # always_include chunks are appended separately after top-k.
            base = (
                config.emb_weight * emb_score
                + config.keyword_weight * kw_score
                + config.heading_boost * head_score
            )
            combined = base * learned
            reason = f"similarity: {emb_score:.2f}"
            if kw_score > 0:
                reason += f", keyword: {kw_score:.2f}"
            if head_score > 0:
                reason += f", heading match: {config.heading_boost * head_score:.2f}"
            if learned != 1.0:
                reason += f", learned boost: {learned - 1.0:+.2f}"
            results.append({
                "id": cid,
                "type": "chunk",
                "summary": m["summary"],
                "score": combined,
                "always_include": m["always_include"],
                "reason": reason,
            })

    if mems["ids"]:
        sims = cosine_similarity(q_vec, mems["matrix"])
        kw = bm25_scores_indexed(query, mems["bm25"])
        for i, mid in enumerate(mems["ids"]):
            m = mems["meta"].get(mid)
            if not m:
                continue
            emb_score = float(sims[i])
            kw_score = kw.get(mid, 0.0)
            combined = (
                config.emb_weight * emb_score + config.keyword_weight * kw_score
            ) * m["weight"]
            reason = f"similarity: {emb_score:.2f}"
            if kw_score > 0:
                reason += f", keyword: {kw_score:.2f}"
            results.append({
                "id": mid,
                "type": "memory",
                "summary": m["text"],
                "score": combined,
                "always_include": False,
                "reason": reason,
            })

    # Rank purely on merit; then guarantee always_include chunks are present
    # by appending any that didn't make top-k (never dropped, never dominant).
    ranked = sorted(results, key=lambda x: x["score"], reverse=True)
    top_k = ranked[:k]
    seen = {r["id"] for r in top_k}
    always_append = [r for r in results if r["always_include"] and r["id"] not in seen]
    return top_k + always_append


def fetch_chunks(chunk_ids: list[str], conn: sqlite3.Connection) -> list[dict]:
    """Fetch full bodies, preserving the caller's requested order.

    Unknown ids are skipped (a renamed heading changes the chunk_id); callers
    can diff returned chunk_ids against what they asked for."""
    if not chunk_ids:
        return []
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"SELECT chunk_id, heading_path, content, token_count FROM chunks WHERE chunk_id IN ({placeholders})",
        chunk_ids,
    ).fetchall()
    by_id = {
        r["chunk_id"]: {
            "chunk_id": r["chunk_id"],
            "heading_path": json.loads(r["heading_path"]),
            "content": r["content"],
            "token_count": r["token_count"],
        }
        for r in rows
    }
    # dict.fromkeys dedups while keeping first-seen order.
    return [by_id[cid] for cid in dict.fromkeys(chunk_ids) if cid in by_id]
