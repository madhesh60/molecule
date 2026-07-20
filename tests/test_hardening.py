"""Hardening regression tests: input validation, edge cases, ordering."""
from __future__ import annotations

import pytest

from molecule.config import MoleculeConfig
from molecule.core.ingest import do_ingest
from molecule.core.memory import write_memory
from molecule.core.retriever import fetch_chunks
from molecule.db import get_connection, init_schema


@pytest.fixture
def db(tmp_path):
    cfg = MoleculeConfig(db_path=tmp_path / ".molecule" / "store.db")
    conn = get_connection(cfg.db_path)
    init_schema(conn)
    return conn, cfg, tmp_path


def test_ingest_empty_file_errors(db):
    conn, cfg, tmp = db
    p = tmp / "empty.md"
    p.write_text("", encoding="utf-8")
    result = do_ingest(p, conn, cfg)
    assert "error" in result


def test_ingest_whitespace_only_errors(db):
    conn, cfg, tmp = db
    p = tmp / "ws.md"
    p.write_text("\n\n   \n", encoding="utf-8")
    result = do_ingest(p, conn, cfg)
    assert "error" in result


def test_ingest_binary_file_errors_not_raises(db):
    conn, cfg, tmp = db
    p = tmp / "bin.md"
    p.write_bytes(b"\xff\xfe\x00\x01 binary junk")
    result = do_ingest(p, conn, cfg)
    assert "error" in result


def test_write_memory_invalid_type_errors(db):
    conn, cfg, _ = db
    result = write_memory("some fact", "t", 0.8, conn, cfg, mem_type="bogus")
    assert "error" in result


def test_write_memory_clamps_confidence(db):
    conn, cfg, _ = db
    result = write_memory("a distinctive test fact", "t", 5.0, conn, cfg)
    assert result.get("stored")
    row = conn.execute(
        "SELECT confidence FROM memories WHERE memory_id=?", (result["memory_id"],)
    ).fetchone()
    assert row["confidence"] == 1.0


def test_fetch_preserves_request_order_and_dedups(db):
    conn, cfg, _ = db
    for cid, txt in [("f::A", "alpha content"), ("f::B", "beta content")]:
        conn.execute(
            "INSERT INTO chunks (chunk_id, source_file, heading_path, summary, content, "
            "token_count, content_hash, last_updated) VALUES (?,?,?,?,?,?,?,?)",
            (cid, "f", '["X"]', "", txt, 2, cid, "now"),
        )
    conn.commit()
    got = fetch_chunks(["f::B", "f::A", "f::B", "f::missing"], conn)
    assert [c["chunk_id"] for c in got] == ["f::B", "f::A"]
