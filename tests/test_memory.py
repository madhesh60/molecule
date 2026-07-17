from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from molecule.config import MoleculeConfig
from molecule.core.memory import expire_stale, search_memories, write_memory
from molecule.db import get_connection, init_schema


@pytest.fixture
def mem_db(tmp_path):
    cfg = MoleculeConfig(db_path=tmp_path / ".molecule" / "store.db", embedding_model="all-MiniLM-L6-v2")
    conn = get_connection(cfg.db_path)
    init_schema(conn)
    return conn, cfg


def test_write_and_search(mem_db):
    conn, cfg = mem_db
    result = write_memory("User prefers dark mode", "task1", 0.9, conn, cfg)
    assert result.get("stored") is True
    hits = search_memories("dark mode preference", 5, conn, cfg)
    assert any("dark mode" in h["text"] for h in hits)


def test_dedup_skips_similar(mem_db):
    conn, cfg = mem_db
    write_memory("User prefers dark mode", "task1", 0.9, conn, cfg)
    result = write_memory("User prefers dark mode", "task2", 0.9, conn, cfg)
    assert result.get("skipped") == "duplicate"


def test_conflict_replaces(mem_db):
    conn, cfg = mem_db
    r1 = write_memory("User prefers dark mode", "task1", 0.9, conn, cfg)
    mid = r1["memory_id"]
    write_memory("User no longer prefers dark mode, changed to light mode", "task2", 0.9, conn, cfg)
    row = conn.execute("SELECT status FROM memories WHERE memory_id=?", (mid,)).fetchone()
    assert row["status"] == "replaced"


def test_secret_redaction(mem_db):
    conn, cfg = mem_db
    write_memory("API key is sk-abc123xyz456def789ghi012jkl345mno", "task1", 0.9, conn, cfg)
    rows = conn.execute("SELECT text FROM memories").fetchall()
    assert all("sk-" not in r["text"] for r in rows)


def test_expiry(mem_db):
    conn, cfg = mem_db
    write_memory("Debugging issue X", "task1", 0.8, conn, cfg, mem_type="task_state")
    # Manually expire it
    conn.execute("UPDATE memories SET expires_at=? WHERE type='task_state'",
                 ((datetime.utcnow() - timedelta(days=1)).isoformat(),))
    conn.commit()
    expire_stale(conn)
    rows = conn.execute("SELECT status FROM memories WHERE type='task_state'").fetchall()
    assert all(r["status"] == "replaced" for r in rows)
