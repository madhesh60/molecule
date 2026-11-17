"""db.py — the single SQLite file behind Molecule.

Everything (chunks, vectors, memories, feedback, bonds, telemetry) lives in
`.molecule/store.db`. This module owns three things:

  1. Connections — WAL mode, a busy timeout so a CLI command and the MCP
     server can share the file, and foreign keys on.
  2. Schema migrations — numbered steps tracked with `PRAGMA user_version`.
     Each step runs in one transaction, so a crash can never leave a
     half-upgraded database. Stores created by any earlier version upgrade
     in place, keeping their data.
  3. Generation counters — tiny triggers bump a per-table counter on every
     write. The search cache keys on those counters, so it is impossible to
     serve stale results (even after raw SQL or a write from another
     process), while bookkeeping writes such as telemetry evict nothing.
"""
from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 2

# Tables whose writes can change a search result, and the columns that matter.
# `None` means "any column".
_GENERATION_DOMAINS: dict[str, tuple[str, tuple[str, ...] | None]] = {
    "chunks": ("chunks", None),
    "memories": (
        "memories",
        ("text", "status", "weight", "confidence", "type", "embedding", "expires_at"),
    ),
    "feedback": ("chunk_feedback", None),
    "bonds": ("chunk_bonds", None),
}


class MoleculeConnection(sqlite3.Connection):
    """sqlite3.Connection that can be weak-referenced (plain ones cannot),
    so per-connection caches are released when the connection is."""


def utc_now() -> datetime:
    """Timezone-aware UTC now — datetime.utcnow() is deprecated in 3.12+."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """UTC timestamp in SQLite's own format ('YYYY-MM-DD HH:MM:SS').

    Using the same textual format as datetime('now') makes string comparison
    in SQL exact. (isoformat()'s 'T' separator sorts AFTER a space, which
    silently delayed expiry checks by up to a day.)"""
    return utc_now().strftime("%Y-%m-%d %H:%M:%S")


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def get_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(db_path), check_same_thread=False, timeout=10.0, factory=MoleculeConnection
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """All-or-nothing block: commit on success, roll back on any error."""
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        # A search inside the failed block may have cached uncommitted state.
        from molecule.core import cache

        cache.forget(conn)
        raise


def _column_exists(conn: sqlite3.Connection, table: str, col: str) -> bool:
    return any(r["name"] == col for r in conn.execute(f"PRAGMA table_info({table})"))


def _ensure_column(conn: sqlite3.Connection, table: str, col: str, ddl: str) -> None:
    """Add a column if missing — SQLite has no ADD COLUMN IF NOT EXISTS."""
    if not _column_exists(conn, table, col):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")


# --- Migrations ----------------------------------------------------------------

def _migrate_v1(conn: sqlite3.Connection) -> None:
    """Baseline schema (everything Molecule 0.1 shipped). Idempotent, so it
    also adopts stores created before migrations were tracked."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id     TEXT PRIMARY KEY,
            source_file  TEXT NOT NULL,
            heading_path TEXT NOT NULL,
            summary      TEXT,
            content      TEXT NOT NULL,
            token_count  INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            always_include BOOLEAN DEFAULT 0,
            weight       REAL DEFAULT 1.0,
            embedding    BLOB,
            last_updated TEXT NOT NULL
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            memory_id   TEXT PRIMARY KEY,
            text        TEXT NOT NULL,
            source_task TEXT,
            confidence  REAL NOT NULL DEFAULT 0.8,
            type        TEXT NOT NULL DEFAULT 'preference',
            status      TEXT NOT NULL DEFAULT 'active',
            weight      REAL NOT NULL DEFAULT 1.0,
            embedding   BLOB,
            created_at  TEXT NOT NULL,
            expires_at  TEXT
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_telemetry (
            session_id       TEXT NOT NULL,
            tokens_used      INTEGER NOT NULL,
            tokens_baseline  INTEGER NOT NULL,
            ts               TEXT NOT NULL
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunk_feedback (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT NOT NULL,
            useful   INTEGER NOT NULL,
            query    TEXT,
            ts       TEXT NOT NULL,
            FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
        )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_file)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_chunk ON chunk_feedback(chunk_id)")
    _ensure_column(conn, "chunks", "learned_weight", "REAL NOT NULL DEFAULT 1.0")
    _ensure_column(conn, "chunks", "embedding_text_version", "INTEGER NOT NULL DEFAULT 1")
    # Normalize legacy isoformat timestamps ('T' separator) to SQLite format so
    # string comparisons against datetime('now') are exact.
    conn.execute(
        "UPDATE memories SET expires_at = replace(expires_at, 'T', ' ') "
        "WHERE expires_at LIKE '%T%'"
    )


def _create_generation_triggers(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS generations ("
        "domain TEXT PRIMARY KEY, gen INTEGER NOT NULL DEFAULT 0)"
    )
    for domain, (table, columns) in _GENERATION_DOMAINS.items():
        conn.execute("INSERT OR IGNORE INTO generations (domain, gen) VALUES (?, 0)", (domain,))
        bump = f"UPDATE generations SET gen = gen + 1 WHERE domain = '{domain}'"
        update_of = f" OF {', '.join(columns)}" if columns else ""
        for event, suffix in (("INSERT", "ins"), (f"UPDATE{update_of}", "upd"), ("DELETE", "del")):
            conn.execute(
                f"CREATE TRIGGER IF NOT EXISTS trg_{domain}_{suffix} "
                f"AFTER {event} ON {table} BEGIN {bump}; END"
            )


def _migrate_v2(conn: sqlite3.Connection) -> None:
    """Molecule 0.2: passage vectors, model-tagged embeddings, source
    tracking, bonds, query-aware feedback, memory lineage, honest telemetry."""
    # Which model produced each vector — switching models re-embeds instead of
    # silently comparing vectors from two different spaces.
    _ensure_column(conn, "chunks", "embedding_model", "TEXT")
    _ensure_column(conn, "chunks", "passage_embeddings", "BLOB")
    _ensure_column(conn, "chunks", "passage_count", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "chunks", "position", "INTEGER NOT NULL DEFAULT 0")

    _ensure_column(conn, "memories", "embedding_model", "TEXT")
    _ensure_column(conn, "memories", "superseded_by", "TEXT")
    _ensure_column(conn, "memories", "updated_at", "TEXT")

    _ensure_column(conn, "chunk_feedback", "query_embedding", "BLOB")
    _ensure_column(conn, "chunk_feedback", "embedding_model", "TEXT")

    _ensure_column(conn, "session_telemetry", "kind", "TEXT NOT NULL DEFAULT 'fetch'")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_telemetry_session ON session_telemetry(session_id)"
    )

    conn.execute("""
        CREATE TABLE IF NOT EXISTS sources (
            source_file  TEXT PRIMARY KEY,   -- repo-relative key, also the chunk id prefix
            abs_path     TEXT NOT NULL,
            file_hash    TEXT NOT NULL,
            size_bytes   INTEGER NOT NULL,
            mtime_ns     INTEGER NOT NULL,
            total_tokens INTEGER NOT NULL,
            chunk_count  INTEGER NOT NULL,
            signature    TEXT NOT NULL,      -- chunker + embedder settings used
            ingested_at  TEXT NOT NULL
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunk_bonds (
            src        TEXT NOT NULL,
            dst        TEXT NOT NULL,
            kind       TEXT NOT NULL,        -- ref | continuation | cofetch
            strength   REAL NOT NULL,
            hits       INTEGER NOT NULL DEFAULT 1,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (src, dst, kind),
            FOREIGN KEY (src) REFERENCES chunks(chunk_id) ON DELETE CASCADE,
            FOREIGN KEY (dst) REFERENCES chunks(chunk_id) ON DELETE CASCADE
        )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bonds_dst ON chunk_bonds(dst)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_stats (
            memory_id   TEXT PRIMARY KEY,
            hits        INTEGER NOT NULL DEFAULT 0,
            last_hit_at TEXT,
            FOREIGN KEY (memory_id) REFERENCES memories(memory_id) ON DELETE CASCADE
        )""")
    _create_generation_triggers(conn)


MIGRATIONS: tuple[tuple[int, Callable[[sqlite3.Connection], None]], ...] = (
    (1, _migrate_v1),
    (2, _migrate_v2),
)


def schema_version(conn: sqlite3.Connection) -> int:
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def init_schema(conn: sqlite3.Connection) -> None:
    """Bring the database to the current schema. Safe to call repeatedly."""
    current = schema_version(conn)
    if current > SCHEMA_VERSION:
        raise RuntimeError(
            f"this store was created by a newer Molecule (schema v{current}, "
            f"this version understands up to v{SCHEMA_VERSION}) — upgrade molecule-mcp"
        )
    for version, migrate in MIGRATIONS:
        if version <= current:
            continue
        conn.commit()  # close any implicit transaction before the explicit one
        conn.execute("BEGIN IMMEDIATE")
        try:
            migrate(conn)
            conn.execute(f"PRAGMA user_version = {version}")
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise


def generations(conn: sqlite3.Connection, domains: tuple[str, ...]) -> tuple[int, ...]:
    """Current write counters for `domains` (see _GENERATION_DOMAINS)."""
    marks = ",".join("?" * len(domains))
    found = dict(
        conn.execute(
            f"SELECT domain, gen FROM generations WHERE domain IN ({marks})", domains
        ).fetchall()
    )
    return tuple(int(found.get(d, 0)) for d in domains)
