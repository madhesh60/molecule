from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path


def get_connection(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _column_exists(conn: sqlite3.Connection, table: str, col: str) -> bool:
    return any(r["name"] == col for r in conn.execute(f"PRAGMA table_info({table})"))


def _ensure_column(conn: sqlite3.Connection, table: str, col: str, ddl: str) -> None:
    """Add a column if missing — SQLite has no ADD COLUMN IF NOT EXISTS."""
    if not _column_exists(conn, table, col):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
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
        );

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
        );

        CREATE TABLE IF NOT EXISTS session_telemetry (
            session_id       TEXT NOT NULL,
            tokens_used      INTEGER NOT NULL,
            tokens_baseline  INTEGER NOT NULL,
            ts               TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunk_feedback (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT NOT NULL,
            useful   INTEGER NOT NULL,
            query    TEXT,
            ts       TEXT NOT NULL,
            FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_file);
        CREATE INDEX IF NOT EXISTS idx_chunks_hash   ON chunks(content_hash);
        CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
        CREATE INDEX IF NOT EXISTS idx_feedback_chunk ON chunk_feedback(chunk_id);
    """)
    # In-place upgrades for DBs created before these columns existed.
    _ensure_column(conn, "chunks", "learned_weight", "REAL NOT NULL DEFAULT 1.0")
    _ensure_column(conn, "chunks", "embedding_text_version", "INTEGER NOT NULL DEFAULT 1")
    conn.commit()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
