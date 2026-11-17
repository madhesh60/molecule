"""snapshot.py — move what Molecule has *learned* between machines.

Chunks can always be rebuilt from the source files. Memories, feedback
votes and learned bonds cannot — they are the part of the store worth
backing up, reviewing in a pull request, or sharing with a teammate.

  * `export_snapshot()` writes them as plain JSON (no vectors — those are
    recomputed on demand, which also keeps the file model-independent).
  * `import_snapshot()` merges a snapshot into a store. It is idempotent:
    importing the same file twice changes nothing.
  * `backup_database()` copies the whole SQLite file safely while it is in
    use (SQLite's online backup API), for a full-fidelity backup.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from molecule.db import transaction, utc_now_iso

SNAPSHOT_FORMAT = "molecule-snapshot"
SNAPSHOT_VERSION = 1

_MEMORY_COLUMNS = (
    "memory_id", "text", "source_task", "confidence", "type", "status", "weight",
    "created_at", "updated_at", "expires_at", "superseded_by",
)


def export_snapshot(conn: sqlite3.Connection, path: Path) -> dict:
    """Write memories, feedback and learned bonds to a JSON file."""
    memories = [
        dict(r) for r in conn.execute(
            f"SELECT {', '.join(_MEMORY_COLUMNS)} FROM memories ORDER BY created_at, memory_id"
        )
    ]
    feedback = [
        dict(r) for r in conn.execute(
            "SELECT chunk_id, useful, query, ts FROM chunk_feedback ORDER BY id"
        )
    ]
    bonds = [
        dict(r) for r in conn.execute(
            "SELECT src, dst, strength, hits, updated_at FROM chunk_bonds "
            "WHERE kind='cofetch' ORDER BY src, dst"
        )
    ]
    payload = {
        "format": SNAPSHOT_FORMAT,
        "version": SNAPSHOT_VERSION,
        "exported_at": utc_now_iso(),
        "memories": memories,
        "feedback": feedback,
        "bonds": bonds,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "path": str(path), "memories": len(memories),
        "feedback": len(feedback), "bonds": len(bonds),
    }


def import_snapshot(conn: sqlite3.Connection, path: Path) -> dict:
    """Merge a snapshot file into this store (idempotent)."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"error": f"cannot read snapshot {path}: {exc}"}
    if not isinstance(payload, dict) or payload.get("format") != SNAPSHOT_FORMAT:
        return {"error": f"not a Molecule snapshot: {path}"}
    if int(payload.get("version", 0)) > SNAPSHOT_VERSION:
        return {"error": "snapshot was written by a newer Molecule — upgrade first"}

    known_chunks = {r[0] for r in conn.execute("SELECT chunk_id FROM chunks")}
    existing_votes = {
        (r["chunk_id"], r["useful"], r["query"] or "", r["ts"])
        for r in conn.execute("SELECT chunk_id, useful, query, ts FROM chunk_feedback")
    }
    memories = feedback = bonds = skipped = 0

    with transaction(conn):
        for memory in payload.get("memories", []):
            # Vectors are not part of a snapshot: the row is stored without one
            # and embedded by the next search (retriever heals missing vectors).
            cursor = conn.execute(
                f"INSERT OR IGNORE INTO memories ({', '.join(_MEMORY_COLUMNS)}) "
                f"VALUES ({', '.join('?' * len(_MEMORY_COLUMNS))})",
                [memory.get(c) for c in _MEMORY_COLUMNS],
            )
            memories += cursor.rowcount

        for vote in payload.get("feedback", []):
            key = (vote["chunk_id"], int(vote["useful"]), vote.get("query") or "", vote["ts"])
            if vote["chunk_id"] not in known_chunks or key in existing_votes:
                skipped += 1
                continue
            conn.execute(
                "INSERT INTO chunk_feedback (chunk_id, useful, query, ts) VALUES (?,?,?,?)", key
            )
            existing_votes.add(key)
            feedback += 1

        for bond in payload.get("bonds", []):
            if bond["src"] not in known_chunks or bond["dst"] not in known_chunks:
                skipped += 1
                continue
            cursor = conn.execute(
                "INSERT OR IGNORE INTO chunk_bonds (src, dst, kind, strength, hits, updated_at) "
                "VALUES (?,?,'cofetch',?,?,?)",
                (bond["src"], bond["dst"], bond["strength"], bond.get("hits", 1),
                 bond.get("updated_at") or utc_now_iso()),
            )
            bonds += cursor.rowcount

    return {"memories": memories, "feedback": feedback, "bonds": bonds, "skipped": skipped}


def backup_database(conn: sqlite3.Connection, destination: Path) -> dict:
    """Consistent copy of the whole store, safe to take while it is in use."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    conn.commit()
    target = sqlite3.connect(str(destination))
    try:
        conn.backup(target)
    finally:
        target.close()
    return {"path": str(destination), "bytes": destination.stat().st_size}
