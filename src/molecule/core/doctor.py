"""doctor.py — one command that says whether a store is healthy, and why not.

Each check returns {"check", "ok", "detail"}. Nothing here modifies data;
the `detail` of a failing check names the command that fixes it.
"""
from __future__ import annotations

import sqlite3

from molecule.config import MoleculeConfig
from molecule.core.embedder import EmbeddingUnavailableError, embedder_for, embedder_name
from molecule.core.sync import source_states
from molecule.core.tokens import tokenizer_name
from molecule.db import SCHEMA_VERSION, schema_version


def _check(name: str, ok: bool, detail: str) -> dict:
    return {"check": name, "ok": bool(ok), "detail": detail}


def run_doctor(conn: sqlite3.Connection, config: MoleculeConfig) -> dict:
    """Run every health check and summarise."""
    checks: list[dict] = []

    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    checks.append(_check("database integrity", integrity == "ok", integrity))

    version = schema_version(conn)
    checks.append(_check(
        "schema version", version == SCHEMA_VERSION,
        f"v{version} (current is v{SCHEMA_VERSION})",
    ))

    problems = config.problems()
    checks.append(_check("configuration", not problems, "; ".join(problems) or "valid"))

    try:
        embedder = embedder_for(config)
        checks.append(_check("embedding model", True, f"{embedder.name} ({embedder.dim} dims)"))
    except EmbeddingUnavailableError as exc:
        checks.append(_check("embedding model", False, str(exc)))

    tokenizer = tokenizer_name()
    checks.append(_check(
        "tokenizer", True,
        "exact (cl100k_base)" if tokenizer != "estimate"
        else "estimated — tiktoken data unavailable; token counts are approximate",
    ))

    chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    checks.append(_check(
        "index", chunks > 0,
        f"{chunks} chunks" if chunks else "empty — run `molecule ingest <file>`",
    ))

    stale_vectors = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE embedding IS NULL OR "
        "(embedding_model IS NOT NULL AND embedding_model != ?)",
        (embedder_name(config),),
    ).fetchone()[0]
    checks.append(_check(
        "vectors match the configured model", stale_vectors == 0,
        "all current" if not stale_vectors
        else f"{stale_vectors} chunk(s) will be re-embedded on the next search",
    ))

    states = source_states(conn)
    changed = [s["source"] for s in states if s["state"] == "changed"]
    missing = [s["source"] for s in states if s["state"] == "missing"]
    checks.append(_check(
        "sources are fresh", not changed,
        "all fresh" if not changed else f"edited since ingest: {', '.join(changed)} — "
        "run `molecule sync`",
    ))
    checks.append(_check(
        "source files exist", not missing,
        "all present" if not missing else f"deleted: {', '.join(missing)} — "
        "run `molecule sync --prune`",
    ))

    orphans = conn.execute(
        "SELECT COUNT(*) FROM chunk_feedback WHERE chunk_id NOT IN (SELECT chunk_id FROM chunks)"
    ).fetchone()[0]
    checks.append(_check("feedback references", orphans == 0,
                         "consistent" if not orphans else f"{orphans} orphaned vote(s)"))

    expired = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE status='active' AND expires_at IS NOT NULL "
        "AND expires_at < datetime('now')"
    ).fetchone()[0]
    checks.append(_check("memory expiry", True,
                         f"{expired} expired fact(s) pending cleanup" if expired else "clean"))

    failed = [c["check"] for c in checks if not c["ok"]]
    return {"healthy": not failed, "failed": failed, "checks": checks}
