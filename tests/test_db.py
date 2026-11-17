from __future__ import annotations

import sqlite3

import pytest

from molecule.db import (
    SCHEMA_VERSION,
    _column_exists,
    generations,
    get_connection,
    init_schema,
    schema_version,
    transaction,
)


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
    assert {
        "chunks", "memories", "session_telemetry", "chunk_feedback",
        "sources", "chunk_bonds", "memory_stats", "generations",
    } <= tables


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


def test_fresh_store_is_at_current_schema_version(tmp_path):
    assert schema_version(_fresh_conn(tmp_path)) == SCHEMA_VERSION


def test_v01_store_upgrades_and_keeps_its_data(tmp_path):
    """A complete Molecule 0.1 store (no user_version, no new tables) must
    open, upgrade and still hold every row."""
    db_path = tmp_path / ".molecule" / "store.db"
    db_path.parent.mkdir(parents=True)
    old = sqlite3.connect(str(db_path))
    old.executescript("""
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY, source_file TEXT NOT NULL,
            heading_path TEXT NOT NULL, summary TEXT, content TEXT NOT NULL,
            token_count INTEGER NOT NULL, content_hash TEXT NOT NULL,
            always_include BOOLEAN DEFAULT 0, weight REAL DEFAULT 1.0,
            embedding BLOB, last_updated TEXT NOT NULL,
            learned_weight REAL NOT NULL DEFAULT 1.0,
            embedding_text_version INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE memories (
            memory_id TEXT PRIMARY KEY, text TEXT NOT NULL, source_task TEXT,
            confidence REAL NOT NULL DEFAULT 0.8, type TEXT NOT NULL DEFAULT 'preference',
            status TEXT NOT NULL DEFAULT 'active', weight REAL NOT NULL DEFAULT 1.0,
            embedding BLOB, created_at TEXT NOT NULL, expires_at TEXT
        );
        CREATE TABLE session_telemetry (
            session_id TEXT NOT NULL, tokens_used INTEGER NOT NULL,
            tokens_baseline INTEGER NOT NULL, ts TEXT NOT NULL
        );
        CREATE TABLE chunk_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT, chunk_id TEXT NOT NULL,
            useful INTEGER NOT NULL, query TEXT, ts TEXT NOT NULL
        );
        INSERT INTO chunks (chunk_id, source_file, heading_path, content, token_count,
            content_hash, last_updated, learned_weight)
            VALUES ('f.md::Old', '/abs/f.md', '["Old"]', 'body', 10, 'h', '2026-01-01', 1.2);
        INSERT INTO memories (memory_id, text, created_at) VALUES ('m1', 'a fact', '2026-01-01');
        INSERT INTO session_telemetry VALUES ('s1', 100, 1000, '2026-01-01');
        INSERT INTO chunk_feedback (chunk_id, useful, query, ts)
            VALUES ('f.md::Old', 1, 'a question', '2026-01-01');
    """)
    old.commit()
    old.close()

    conn = get_connection(db_path)
    init_schema(conn)
    assert schema_version(conn) == SCHEMA_VERSION
    assert conn.execute("SELECT learned_weight FROM chunks").fetchone()[0] == 1.2
    assert conn.execute("SELECT text FROM memories").fetchone()[0] == "a fact"
    assert conn.execute("SELECT kind FROM session_telemetry").fetchone()[0] == "fetch"
    assert conn.execute("SELECT query FROM chunk_feedback").fetchone()[0] == "a question"
    assert _column_exists(conn, "chunks", "embedding_model")
    assert _column_exists(conn, "memories", "superseded_by")


def test_newer_store_is_refused_with_a_clear_message(tmp_path):
    conn = _fresh_conn(tmp_path)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    conn.commit()
    with pytest.raises(RuntimeError, match="newer Molecule"):
        init_schema(conn)


def test_generation_counters_track_their_tables(tmp_path):
    conn = _fresh_conn(tmp_path)
    before = generations(conn, ("chunks", "memories", "feedback", "bonds"))
    conn.execute(
        "INSERT INTO chunks (chunk_id, source_file, heading_path, content, token_count, "
        "content_hash, last_updated) VALUES ('a', 'f', '[]', 'x', 1, 'h', 'now')"
    )
    conn.execute("INSERT INTO memories (memory_id, text, created_at) VALUES ('m', 'x', 'now')")
    conn.commit()
    after = generations(conn, ("chunks", "memories", "feedback", "bonds"))
    assert after[0] == before[0] + 1 and after[1] == before[1] + 1
    assert after[2:] == before[2:]


def test_transaction_rolls_back_on_error(tmp_path):
    conn = _fresh_conn(tmp_path)
    with pytest.raises(ValueError):
        with transaction(conn):
            conn.execute("INSERT INTO memories (memory_id, text, created_at) "
                         "VALUES ('m', 'x', 'now')")
            raise ValueError("abort")
    assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0


def test_deleting_a_chunk_cascades_to_feedback_and_bonds(tmp_path):
    conn = _fresh_conn(tmp_path)
    for chunk_id in ("a", "b"):
        conn.execute(
            "INSERT INTO chunks (chunk_id, source_file, heading_path, content, token_count, "
            "content_hash, last_updated) VALUES (?, 'f', '[]', 'x', 1, 'h', 'now')", (chunk_id,)
        )
    conn.execute("INSERT INTO chunk_feedback (chunk_id, useful, ts) VALUES ('a', 1, 'now')")
    conn.execute("INSERT INTO chunk_bonds (src, dst, kind, strength, updated_at) "
                 "VALUES ('a', 'b', 'ref', 0.5, 'now')")
    conn.commit()
    conn.execute("DELETE FROM chunks WHERE chunk_id='a'")
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM chunk_feedback").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM chunk_bonds").fetchone()[0] == 0
