"""Hardening regression tests: input validation, edge cases, ordering."""
from __future__ import annotations

from molecule.core.ingest import do_ingest
from molecule.core.memory import write_memory
from molecule.core.retriever import fetch_chunks


def test_ingest_missing_file_errors(conn, cfg, tmp_path):
    assert "error" in do_ingest(tmp_path / "nope.md", conn, cfg)


def test_ingest_empty_file_errors(conn, cfg, tmp_path):
    p = tmp_path / "empty.md"
    p.write_text("", encoding="utf-8")
    result = do_ingest(p, conn, cfg)
    assert "error" in result


def test_ingest_whitespace_only_errors(conn, cfg, tmp_path):
    p = tmp_path / "ws.md"
    p.write_text("\n\n   \n", encoding="utf-8")
    result = do_ingest(p, conn, cfg)
    assert "error" in result


def test_ingest_binary_file_errors_not_raises(conn, cfg, tmp_path):
    p = tmp_path / "bin.md"
    p.write_bytes(b"\xff\xfe\x00\x01 binary junk")
    result = do_ingest(p, conn, cfg)
    assert "error" in result


def test_ingest_heading_only_file_errors(conn, cfg, tmp_path):
    p = tmp_path / "titles.md"
    p.write_text("# One\n\n## Two\n\n### Three\n", encoding="utf-8")
    assert "error" in do_ingest(p, conn, cfg)


def test_ingest_handles_windows_line_endings(conn, cfg, tmp_path):
    p = tmp_path / "crlf.md"
    p.write_bytes(b"# Title\r\n\r\nBody text here.\r\n\r\n# Other\r\n\r\nMore text.\r\n")
    result = do_ingest(p, conn, cfg)
    assert result["total"] == 2
    ids = {r[0] for r in conn.execute("SELECT chunk_id FROM chunks")}
    assert ids == {"crlf.md::Title", "crlf.md::Other"}   # no stray \r in ids


def test_write_memory_invalid_type_errors(conn, cfg):
    result = write_memory("some fact", "t", 0.8, conn, cfg, mem_type="bogus")
    assert "error" in result


def test_write_memory_empty_text_errors(conn, cfg):
    assert "error" in write_memory("   ", "t", 0.8, conn, cfg)
    assert "error" in write_memory(None, "t", 0.8, conn, cfg)


def test_write_memory_clamps_confidence(conn, cfg):
    result = write_memory("a distinctive test fact", "t", 5.0, conn, cfg)
    assert result.get("stored")
    row = conn.execute(
        "SELECT confidence FROM memories WHERE memory_id=?", (result["memory_id"],)
    ).fetchone()
    assert row["confidence"] == 1.0


def test_fetch_preserves_request_order_and_dedups(conn):
    for cid, txt in [("f::A", "alpha content"), ("f::B", "beta content")]:
        conn.execute(
            "INSERT INTO chunks (chunk_id, source_file, heading_path, summary, content, "
            "token_count, content_hash, last_updated) VALUES (?,?,?,?,?,?,?,?)",
            (cid, "f", '["X"]', "", txt, 2, cid, "now"),
        )
    conn.commit()
    got = fetch_chunks(["f::B", "f::A", "f::B", "f::missing"], conn)
    assert [c["chunk_id"] for c in got] == ["f::B", "f::A"]
    assert fetch_chunks([], conn) == []
