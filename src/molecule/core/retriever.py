from __future__ import annotations

import json
import sqlite3

from molecule.config import MoleculeConfig
from molecule.core.embedder import cosine_similarity, embed, load_embeddings
from molecule.core.scoring import bm25_scores, heading_match_score


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


def rank_candidates(
    query: str,
    conn: sqlite3.Connection,
    k: int,
    config: MoleculeConfig,
) -> list[dict]:
    q_vec = embed([query], config.embedding_model)[0]

    # --- chunks ---
    chunk_ids, chunk_matrix = load_embeddings(conn, "chunks", "chunk_id")

    # --- memories ---
    mem_ids, mem_matrix = load_embeddings(
        conn, "memories", "memory_id",
        filter_clause="status='active' AND (expires_at IS NULL OR expires_at > datetime('now'))"
    )

    results: list[dict] = []

    if chunk_ids:
        sims = cosine_similarity(q_vec, chunk_matrix)
        meta = {
            r["chunk_id"]: r for r in conn.execute(
                f"SELECT chunk_id, heading_path, summary, content, weight, learned_weight, "
                f"always_include FROM chunks WHERE chunk_id IN ({','.join('?'*len(chunk_ids))})",
                chunk_ids,
            ).fetchall()
        }
        # BM25 over heading + summary + content — headings are strong signals.
        docs = {
            cid: f"{' '.join(json.loads(m['heading_path']))} {m['summary'] or ''} {m['content']}"
            for cid, m in meta.items()
        }
        kw = bm25_scores(query, docs)
        for i, cid in enumerate(chunk_ids):
            m = meta.get(cid)
            if not m:
                continue
            heading_path = json.loads(m["heading_path"])
            emb_score = float(sims[i])
            kw_score = kw.get(cid, 0.0)
            head_score = heading_match_score(query, heading_path)
            learned = float(m["learned_weight"] or 1.0)
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
                "summary": m["summary"] or "",
                "score": combined,
                "always_include": bool(m["always_include"]),
                "reason": reason,
            })

    if mem_ids:
        sims = cosine_similarity(q_vec, mem_matrix)
        meta = {
            r["memory_id"]: r for r in conn.execute(
                f"SELECT memory_id, text, weight FROM memories "
                f"WHERE memory_id IN ({','.join('?'*len(mem_ids))})",
                mem_ids,
            ).fetchall()
        }
        kw = bm25_scores(query, {mid: m["text"] for mid, m in meta.items()})
        for i, mid in enumerate(mem_ids):
            m = meta.get(mid)
            if not m:
                continue
            emb_score = float(sims[i])
            kw_score = kw.get(mid, 0.0)
            combined = (
                config.emb_weight * emb_score + config.keyword_weight * kw_score
            ) * float(m["weight"] or 1.0)
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
    if not chunk_ids:
        return []
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"SELECT chunk_id, heading_path, content, token_count FROM chunks WHERE chunk_id IN ({placeholders})",
        chunk_ids,
    ).fetchall()
    return [
        {
            "chunk_id": r["chunk_id"],
            "heading_path": json.loads(r["heading_path"]),
            "content": r["content"],
            "token_count": r["token_count"],
        }
        for r in rows
    ]
