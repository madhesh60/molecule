"""memory.py — the small notebook of durable facts.

Every write goes through the same gate, in this order:

  1. **Redact** anything that looks like a secret — nothing sensitive is
     ever persisted.
  2. **Dedup** — a fact that says the same thing as an existing one
     (cosine ≥ dedup_threshold) is skipped.
  3. **Conflict** — a fact on the same topic that contradicts an old one
     retires it (`status='replaced'`, `superseded_by` → the new fact).
     History is kept, so "what did we believe before?" stays answerable.
  4. **Expiry** — `task_state` facts get an `expires_at`; preferences don't.

Low-confidence facts are stored with a lower retrieval weight so a guess
never outranks something the user stated directly. The notebook is bounded:
past `max_active_memories`, the least valuable facts (low confidence, never
recalled, oldest) are evicted first.
"""
from __future__ import annotations

import re
import sqlite3
import uuid
from collections.abc import Callable
from datetime import timedelta

import numpy as np

from molecule.config import MoleculeConfig
from molecule.core.embedder import embedder_for
from molecule.core.retriever import clamp_k, memory_corpus, score_memories
from molecule.db import transaction, utc_now, utc_now_iso

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

_VALID_MEM_TYPES = {"preference", "task_state"}
_LOW_CONFIDENCE_WEIGHT = 0.6
_MAX_MEMORY_CHARS = 2000


def redact(text: str) -> str:
    """Replace anything that looks like a credential with [REDACTED]."""
    return _SECRET_RE.sub("[REDACTED]", text)


_redact = redact  # backwards-compatible private name


def _looks_conflicting(new_text: str) -> bool:
    return bool(_CONFLICT_WORDS.search(new_text))


def default_llm_fn(texts: list[str]) -> list[str]:
    """Fallback merge without an LLM: keep the first sentence of each group."""
    results = []
    for text in texts:
        first = text.split(".")[0].strip().replace("\n", " ")
        results.append(first[:200] if first else text[:200])
    return results


def expire_stale(conn: sqlite3.Connection) -> int:
    """Retire memories whose expiry has passed. Returns how many."""
    cursor = conn.execute(
        "UPDATE memories SET status='replaced' "
        "WHERE expires_at IS NOT NULL AND expires_at < datetime('now') AND status='active'"
    )
    conn.commit()
    return cursor.rowcount


def enforce_capacity(conn: sqlite3.Connection, config: MoleculeConfig) -> int:
    """Evict the least valuable active memories beyond `max_active_memories`.

    Value order: low confidence first, then never-recalled, then oldest —
    the same idea as a cache's eviction policy, applied to facts."""
    active = conn.execute("SELECT COUNT(*) FROM memories WHERE status='active'").fetchone()[0]
    excess = active - config.max_active_memories
    if excess <= 0:
        return 0
    victims = [
        r[0] for r in conn.execute(
            "SELECT m.memory_id FROM memories m "
            "LEFT JOIN memory_stats s ON s.memory_id = m.memory_id "
            "WHERE m.status='active' "
            "ORDER BY m.confidence ASC, COALESCE(s.hits, 0) ASC, m.created_at ASC LIMIT ?",
            (excess,),
        )
    ]
    now = utc_now_iso()
    conn.executemany(
        "UPDATE memories SET status='evicted', updated_at=? WHERE memory_id=?",
        [(now, memory_id) for memory_id in victims],
    )
    conn.commit()
    return len(victims)


def write_memory(
    text: str,
    source_task: str,
    confidence: float,
    conn: sqlite3.Connection,
    config: MoleculeConfig,
    mem_type: str = "preference",
) -> dict:
    """Store one fact (redact → dedup → conflict → insert)."""
    if mem_type not in _VALID_MEM_TYPES:
        return {"error": f"invalid mem_type: {mem_type!r} (use 'preference' or 'task_state')"}
    # Clamp instead of erroring — callers (and LLMs) send 90 meaning 0.9.
    confidence = max(0.0, min(1.0, float(confidence)))

    clean = redact((text or "").strip())[:_MAX_MEMORY_CHARS]
    if not clean:
        return {"error": "empty text after redaction"}

    embedder = embedder_for(config)
    vec = embedder.encode([clean], config.chunk_embed_tokens)[0]

    replaced: str | None = None
    corpus = memory_corpus(conn, config)
    if corpus.ids and corpus.matrix.shape[1] == vec.shape[0]:
        sims = corpus.matrix @ vec
        best = int(np.argmax(sims))
        best_sim = float(sims[best])
        conflicting = _looks_conflicting(clean)
        # Dedup: near-identical AND not contradicting → skip.
        if best_sim >= config.dedup_threshold and not conflicting:
            return {"skipped": "duplicate", "similar_to": corpus.ids[best],
                    "similarity": round(best_sim, 4)}
        # Conflict: same topic AND contradicting → retire the old fact.
        if best_sim >= config.conflict_threshold and conflicting:
            replaced = corpus.ids[best]

    memory_id = str(uuid.uuid4())
    now = utc_now_iso()
    expires_at = None
    if mem_type == "task_state":
        expires_at = (utc_now() + timedelta(days=config.task_state_expiry_days)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    weight = 1.0 if confidence >= config.low_confidence_threshold else _LOW_CONFIDENCE_WEIGHT

    with transaction(conn):
        conn.execute(
            """INSERT INTO memories
               (memory_id, text, source_task, confidence, type, status, weight, embedding,
                embedding_model, created_at, updated_at, expires_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (memory_id, clean, source_task, confidence, mem_type, "active", weight,
             vec.tobytes(), embedder.name, now, now, expires_at),
        )
        if replaced:
            conn.execute(
                "UPDATE memories SET status='replaced', superseded_by=?, updated_at=? "
                "WHERE memory_id=?",
                (memory_id, now, replaced),
            )
    result: dict = {"memory_id": memory_id, "stored": True}
    if replaced:
        result["replaced"] = replaced
    return result


def search_memories(
    query: str,
    k: int,
    conn: sqlite3.Connection,
    config: MoleculeConfig,
) -> list[dict]:
    """Top-k active memories for a query. Each hit is counted, so the
    notebook knows which facts are actually used."""
    hits = score_memories(query, conn, config)[: clamp_k(k, config)]
    if hits:
        now = utc_now_iso()
        conn.executemany(
            "INSERT INTO memory_stats (memory_id, hits, last_hit_at) VALUES (?, 1, ?) "
            "ON CONFLICT(memory_id) DO UPDATE SET hits = hits + 1, last_hit_at = ?",
            [(h["memory_id"], now, now) for h in hits],
        )
        conn.commit()
    return hits


def forget_memory(conn: sqlite3.Connection, memory_id: str) -> dict:
    """Permanently delete one memory."""
    cursor = conn.execute("DELETE FROM memories WHERE memory_id=?", (memory_id,))
    conn.commit()
    if cursor.rowcount == 0:
        return {"error": f"memory_id not found: {memory_id}"}
    return {"deleted": memory_id}


def memory_history(conn: sqlite3.Connection, memory_id: str) -> list[dict]:
    """The chain of facts that replaced each other, oldest first, for the
    topic `memory_id` belongs to."""
    row = conn.execute(
        "SELECT memory_id FROM memories WHERE memory_id=?", (memory_id,)
    ).fetchone()
    if not row:
        return []
    # Walk back to the oldest ancestor, then forward along superseded_by.
    oldest, seen = memory_id, {memory_id}
    while True:
        parent = conn.execute(
            "SELECT memory_id FROM memories WHERE superseded_by=? LIMIT 1", (oldest,)
        ).fetchone()
        if not parent or parent[0] in seen:
            break
        oldest = parent[0]
        seen.add(oldest)
    chain: list[dict] = []
    current: str | None = oldest
    visited: set[str] = set()
    while current and current not in visited:
        visited.add(current)
        item = conn.execute(
            "SELECT memory_id, text, status, confidence, created_at, superseded_by "
            "FROM memories WHERE memory_id=?",
            (current,),
        ).fetchone()
        if not item:
            break
        chain.append(dict(item))
        current = item["superseded_by"]
    return chain


def merge_duplicates(
    conn: sqlite3.Connection,
    config: MoleculeConfig,
    llm_fn: Callable[[list[str]], list[str]],
) -> int:
    """Collapse clusters of near-duplicate memories into one fact each.
    `llm_fn` turns the joined texts of a cluster into a single sentence."""
    corpus = memory_corpus(conn, config)
    if len(corpus.ids) < 2:
        return 0
    similarity = corpus.matrix @ corpus.matrix.T

    merged_count = 0
    used: set[int] = set()
    for i in range(len(corpus.ids)):
        if i in used:
            continue
        cluster = [i] + [
            j for j in range(i + 1, len(corpus.ids))
            if j not in used and similarity[i, j] > config.conflict_threshold
        ]
        if len(cluster) < 2:
            continue
        used.update(cluster)
        merged_text = llm_fn(["\n".join(corpus.texts[c] for c in cluster)])[0]
        cluster_ids = [corpus.ids[c] for c in cluster]
        now = utc_now_iso()
        conn.executemany(
            "UPDATE memories SET status='replaced', updated_at=? WHERE memory_id=?",
            [(now, memory_id) for memory_id in cluster_ids],
        )
        conn.commit()
        result = write_memory(merged_text, "merge_duplicates", 0.9, conn, config)
        if result.get("memory_id"):
            conn.executemany(
                "UPDATE memories SET superseded_by=? WHERE memory_id=?",
                [(result["memory_id"], memory_id) for memory_id in cluster_ids],
            )
            conn.commit()
        merged_count += 1
    return merged_count
