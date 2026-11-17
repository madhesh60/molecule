"""sync.py — keep the index in step with the files on disk.

An index that silently describes last week's docs is worse than no index.
Each ingested file is tracked in `sources` (hash, size, mtime), which makes
freshness cheap to check:

  * `source_states()` — one `stat()` per file, no reads: fresh / changed /
    missing.
  * `sync_sources()` — re-ingests the changed ones (incrementally: only the
    edited sections are re-embedded) and optionally prunes files that were
    deleted.

The engine runs the stat check automatically, throttled to once every few
seconds, so "edit AGENTS.md, ask a question" just works without a manual
re-ingest step.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from molecule.config import MoleculeConfig
from molecule.core.ingest import ingest_file, remove_source


def source_states(conn: sqlite3.Connection) -> list[dict]:
    """Freshness of every tracked source, from file metadata only."""
    states = []
    for row in conn.execute(
        "SELECT source_file, abs_path, size_bytes, mtime_ns FROM sources ORDER BY source_file"
    ):
        path = Path(row["abs_path"])
        try:
            stat = path.stat()
        except OSError:
            state = "missing"
        else:
            unchanged = (
                stat.st_size == row["size_bytes"] and stat.st_mtime_ns == row["mtime_ns"]
            )
            state = "fresh" if unchanged else "changed"
        states.append({"source": row["source_file"], "path": row["abs_path"], "state": state})
    return states


def sync_sources(
    conn: sqlite3.Connection, config: MoleculeConfig, prune: bool = False
) -> dict:
    """Re-ingest sources whose file changed; optionally drop deleted ones."""
    reingested: list[dict] = []
    missing: list[str] = []
    errors: list[str] = []
    fresh = 0
    for entry in source_states(conn):
        if entry["state"] == "fresh":
            fresh += 1
        elif entry["state"] == "missing":
            missing.append(entry["source"])
            if prune:
                remove_source(conn, entry["source"])
        else:
            result = ingest_file(Path(entry["path"]), conn, config)
            if "error" in result:
                errors.append(result["error"])
            elif result.get("unchanged"):
                fresh += 1          # touched but identical bytes
            else:
                reingested.append({
                    "source": entry["source"], "new": result["new"],
                    "updated": result["updated"], "removed": result["removed"],
                })
    return {
        "checked": fresh + len(reingested) + len(missing) + len(errors),
        "fresh": fresh,
        "reingested": reingested,
        "missing": missing,
        "pruned": missing if prune else [],
        "errors": errors,
    }
