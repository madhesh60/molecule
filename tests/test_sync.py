"""Freshness: detect edited/deleted source files and bring the index back in step."""
from __future__ import annotations

import os

from molecule import Molecule
from molecule.core.ingest import ingest_file
from molecule.core.sync import source_states, sync_sources
from tests.conftest import make_config

PAD = "This sentence pads the section so that it stays a chunk of its own. " * 5


def _write(path, body: str = "original body"):
    path.write_text(f"# Doc\n\n## Part\n\n{body}. {PAD}\n", encoding="utf-8")
    # Make sure the edit is visible to a metadata-only check on coarse clocks.
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000))
    return path


def test_states_fresh_changed_missing(conn, cfg, tmp_path):
    path = _write(tmp_path / "a.md")
    ingest_file(path, conn, cfg)
    assert [s["state"] for s in source_states(conn)] == ["fresh"]
    _write(path, "edited body")
    assert [s["state"] for s in source_states(conn)] == ["changed"]
    path.unlink()
    assert [s["state"] for s in source_states(conn)] == ["missing"]


def test_sync_reingests_only_changed_sources(conn, cfg, tmp_path):
    a, b = _write(tmp_path / "a.md"), _write(tmp_path / "b.md")
    ingest_file(a, conn, cfg)
    ingest_file(b, conn, cfg)
    _write(a, "edited body")
    report = sync_sources(conn, cfg)
    assert [r["source"] for r in report["reingested"]] == ["a.md"]
    assert report["fresh"] == 1 and report["missing"] == [] and report["errors"] == []
    content = conn.execute("SELECT content FROM chunks WHERE source_file='a.md'").fetchone()[0]
    assert "edited body" in content


def test_touched_but_identical_file_is_not_reingested(conn, cfg, tmp_path):
    path = _write(tmp_path / "a.md")
    ingest_file(path, conn, cfg)
    _write(path)                               # same bytes, newer mtime
    report = sync_sources(conn, cfg)
    assert report["reingested"] == [] and report["fresh"] == 1
    assert [s["state"] for s in source_states(conn)] == ["fresh"]   # mtime was refreshed


def test_missing_source_is_reported_and_pruned_on_request(conn, cfg, tmp_path):
    path = _write(tmp_path / "a.md")
    ingest_file(path, conn, cfg)
    path.unlink()
    assert sync_sources(conn, cfg)["missing"] == ["a.md"]
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1   # kept by default
    assert sync_sources(conn, cfg, prune=True)["pruned"] == ["a.md"]
    assert conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    assert source_states(conn) == []


def test_engine_picks_up_an_edit_without_a_manual_reingest(tmp_path):
    """Edit the file, ask again: the answer already reflects the edit."""
    cfg = make_config(tmp_path, auto_sync=True, auto_sync_interval_s=0.0)
    path = _write(tmp_path / "a.md", "the deploy window is on Tuesday")
    with Molecule(config=cfg) as mol:
        mol.ingest(path)
        assert "Tuesday" in mol.context("when is the deploy window")["context"]
        _write(path, "the deploy window is on Thursday")
        context = mol.context("when is the deploy window")["context"]
        assert "Thursday" in context and "Tuesday" not in context


def test_engine_auto_sync_can_be_disabled(tmp_path):
    cfg = make_config(tmp_path, auto_sync=False)
    path = _write(tmp_path / "a.md", "the deploy window is on Tuesday")
    with Molecule(config=cfg) as mol:
        mol.ingest(path)
        _write(path, "the deploy window is on Thursday")
        assert "Tuesday" in mol.context("when is the deploy window")["context"]
        assert mol.health()["stale_sources"] == ["a.md"]
        assert [r["source"] for r in mol.sync()["reingested"]] == ["a.md"]
        assert "Thursday" in mol.context("when is the deploy window")["context"]
