from __future__ import annotations

import sqlite3

from molecule.db import _column_exists, get_connection, init_schema


def _fresh_conn(tmp_path):
    conn = get_connection(tmp_path / ".molecule" / "store.db")
    init_schema(conn)
    return conn


def test_schema_creates_all_tables(tmp_path):
    conn = _fresh_conn(tmp_path)
    tables = {
        r["name"]
        for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {"chunks", "memories", "session_telemetry", "chunk_feedback"} <= tables


def test_schema_is_idempotent(tmp_path):
    conn = _fresh_conn(tmp_path)
    init_schema(conn)  # second run must not raise
    init_schema(conn)
    assert _column_exists(conn, "chunks", "learned_weight")


def test_old_db_upgrades_in_place(tmp_path):
    """A DB created before learned_weight/embedding_text_version/chunk_feedback
    existed gets upgraded without losing data."""
    db_path = tmp_path / ".molecule" / "store.db"
    db_path.parent.mkdir(parents=True)
    old = sqlite3.connect(str(db_path))
    old.executescript("""
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY, source_file TEXT NOT NULL,
            heading_path TEXT NOT NULL, summary TEXT, content TEXT NOT NULL,
            token_count INTEGER NOT NULL, content_hash TEXT NOT NULL,
            always_include BOOLEAN DEFAULT 0, weight REAL DEFAULT 1.0,
            embedding BLOB, last_updated TEXT NOT NULL
        );
    """)
    old.execute(
        "INSERT INTO chunks (chunk_id, source_file, heading_path, content, "
        "token_count, content_hash, last_updated) VALUES (?,?,?,?,?,?,?)",
        ("f.md::Old", "f.md", '["Old"]', "body", 10, "h", "2026-01-01"),
    )
    old.commit()
    old.close()

    conn = get_connection(db_path)
    init_schema(conn)

    row = conn.execute(
        "SELECT learned_weight, embedding_text_version FROM chunks"
    ).fetchone()
    assert row["learned_weight"] == 1.0
    assert row["embedding_text_version"] == 1
    assert _column_exists(conn, "chunks", "embedding_text_version")
