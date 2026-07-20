"""
memory.py — expose default_llm_fn for merge_duplicates callers
"""
from __future__ import annotations

import re
import sqlite3
import uuid
from datetime import timedelta
from typing import Callable

import numpy as np

from molecule.config import MoleculeConfig
from molecule.core.embedder import cosine_similarity, embed, embed_query, load_embeddings
from molecule.core.scoring import bm25_scores
from molecule.db import utc_now, utc_now_iso

_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_-]{20,}"                      # OpenAI/Anthropic-style keys
    r"|(?:ghp|gho|ghs|ghu|github_pat)_[A-Za-z0-9_]{36,}"  # GitHub tokens
    r"|AKIA[0-9A-Z]{16}"                            # AWS access key id
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"                # Slack tokens
    r"|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}"  # JWT
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"          # PEM private keys
    r"|[A-Za-z0-9+/]{40,}={0,2}"                    # long base64 blobs
    r"|(?:password|passwd|pwd|token|secret|api[_-]?key|authorization)\s*[:=]\s*\S+"
    r"|://[^/\s:@]+:[^/\s:@]+@)",                   # creds in connection URLs
    re.IGNORECASE,
)

_CONFLICT_WORDS = re.compile(
    r"\b(not|no longer|changed|updated|now|instead|actually|wrong|incorrect|never)\b",
    re.IGNORECASE,
)


def _redact(text: str) -> str:
    return _SECRET_RE.sub("[REDACTED]", text)


def _looks_conflicting(new_text: str) -> bool:
    return bool(_CONFLICT_WORDS.search(new_text))


def default_llm_fn(texts: list[str]) -> list[str]:
    """Fallback merge: join texts with a separator, return first sentence of combined."""
    results = []
    for text in texts:
        first = text.split(".")[0].strip().replace("\n", " ")
        results.append(first[:200] if first else text[:200])
    return results


def expire_stale(conn: sqlite3.Connection) -> None:
    conn.execute(
        "UPDATE memories SET status='replaced' "
        "WHERE expires_at IS NOT NULL AND expires_at < datetime('now') AND status='active'"
    )
    conn.commit()


_VALID_MEM_TYPES = {"preference", "task_state"}


def write_memory(
    text: str,
    source_task: str,
    confidence: float,
    conn: sqlite3.Connection,
    config: MoleculeConfig,
    mem_type: str = "preference",
) -> dict:
    if mem_type not in _VALID_MEM_TYPES:
        return {"error": f"invalid mem_type: {mem_type!r} (use 'preference' or 'task_state')"}
    # Clamp instead of erroring — callers (and LLMs) send 90 meaning 0.9.
    confidence = max(0.0, min(1.0, float(confidence)))

    clean = _redact(text.strip())
    if not clean:
        return {"error": "empty text after redaction"}

    vec = embed([clean], config.embedding_model)[0]

    ids, matrix = load_embeddings(conn, "memories", "memory_id", filter_clause="status='active'")

    if ids:
        sims = cosine_similarity(vec, matrix)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        # Dedup: near-identical AND not contradicting → skip.
        # Conflict: same topic (>= conflict_threshold) AND contradicting →
        # retire the old fact (status='replaced', history kept) and store the new one.
        if best_sim >= config.dedup_threshold and not _looks_conflicting(clean):
            return {"skipped": "duplicate", "similar_to": ids[best_idx], "similarity": best_sim}
        if best_sim >= config.conflict_threshold and _looks_conflicting(clean):
            conn.execute(
                "UPDATE memories SET status='replaced' WHERE memory_id=?",
                (ids[best_idx],),
            )

    memory_id = str(uuid.uuid4())
    now = utc_now_iso()
    expires_at = None
    if mem_type == "task_state":
        expires_at = (utc_now() + timedelta(days=config.task_state_expiry_days)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    weight = 1.0 if confidence >= config.low_confidence_threshold else 0.6

    conn.execute(
        """INSERT INTO memories
           (memory_id, text, source_task, confidence, type, status, weight, embedding, created_at, expires_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (memory_id, clean, source_task, confidence, mem_type, "active", weight,
         vec.tobytes(), now, expires_at),
    )
    conn.commit()
    return {"memory_id": memory_id, "stored": True}


def search_memories(
    query: str,
    k: int,
    conn: sqlite3.Connection,
    config: MoleculeConfig,
) -> list[dict]:
    q_vec = embed_query(query, config.embedding_model)
    ids, matrix = load_embeddings(
        conn, "memories", "memory_id",
        filter_clause="status='active' AND (expires_at IS NULL OR expires_at > datetime('now'))"
    )
    if not ids:
        return []

    sims = cosine_similarity(q_vec, matrix)

    meta = {
        r["memory_id"]: r
        for r in conn.execute(
            f"SELECT memory_id, text, confidence, weight FROM memories "
            f"WHERE memory_id IN ({','.join('?'*len(ids))})",
            ids,
        ).fetchall()
    }
    kw_scores = bm25_scores(query, {mid: m["text"] for mid, m in meta.items()})

    scored = []
    for i, mid in enumerate(ids):
        m = meta.get(mid)
        if not m:
            continue
        emb_s = float(sims[i])
        kw_s = kw_scores.get(mid, 0.0)
        score = (config.emb_weight * emb_s + config.keyword_weight * kw_s) * float(m["weight"] or 1.0)
        reason = f"similarity: {emb_s:.2f}"
        if kw_s > 0:
            reason += f", keyword: {kw_s:.2f}"
        scored.append({
            "memory_id": mid,
            "text": m["text"],
            "confidence": m["confidence"],
            "score": score,
            "reason": reason,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]


def merge_duplicates(
    conn: sqlite3.Connection,
    config: MoleculeConfig,
    llm_fn: Callable[[list[str]], list[str]],
) -> int:
    ids, matrix = load_embeddings(conn, "memories", "memory_id", filter_clause="status='active'")
    if len(ids) < 2:
        return 0

    merged_count = 0
    used: set[str] = set()

    for i in range(len(ids)):
        if ids[i] in used:
            continue
        cluster = [i]
        for j in range(i + 1, len(ids)):
            if ids[j] in used:
                continue
            sim = float(np.dot(matrix[i], matrix[j]))
            if sim > config.conflict_threshold:
                cluster.append(j)

        if len(cluster) < 2:
            continue

        cluster_ids = [ids[c] for c in cluster]
        texts = [
            r["text"] for r in conn.execute(
                f"SELECT text FROM memories WHERE memory_id IN ({','.join('?'*len(cluster_ids))})",
                cluster_ids,
            ).fetchall()
        ]
        merged_text = llm_fn(["\n".join(texts)])[0]
        for cid in cluster_ids:
            conn.execute("UPDATE memories SET status='replaced' WHERE memory_id=?", (cid,))
            used.add(cid)

        write_memory(merged_text, "merge_duplicates", 0.9, conn, config)
        merged_count += 1

    conn.commit()
    return merged_count
