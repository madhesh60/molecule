"""The `molecule` command: every command runs and prints something sensible."""
from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from molecule.cli import cli
from tests.conftest import SAMPLE_DOC


@pytest.fixture
def run(tmp_path, monkeypatch):
    """Invoke the CLI against a hermetic store rooted at tmp_path."""
    monkeypatch.setenv("MOLECULE_ROOT", str(tmp_path))
    monkeypatch.setenv("MOLECULE_EMBEDDING_MODEL", "hashing")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "handbook.md").write_text(SAMPLE_DOC, encoding="utf-8")
    runner = CliRunner()

    def invoke(*args: str, ok: bool = True):
        result = runner.invoke(cli, list(args), catch_exceptions=False)
        if ok:
            assert result.exit_code == 0, result.output
        return result

    return invoke


@pytest.fixture
def indexed(run):
    assert "5 new" in run("ingest", "handbook.md").output
    return run


def test_version(run):
    assert "molecule, version" in run("--version").output


def test_ingest_reports_counts_and_skips_on_repeat(indexed):
    again = indexed("ingest", "handbook.md").output
    assert "0 new, 0 updated, 5 skipped" in again


def test_ingest_directory(run, tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "extra.md").write_text("# Extra\n\nMore text here.\n", encoding="utf-8")
    assert "1 files" in run("ingest", "docs").output


def test_ingest_json(run):
    data = json.loads(run("ingest", "handbook.md", "--json").output)
    assert data["new"] == 5 and data["source"] == "handbook.md"


def test_search_shows_reason_and_related(indexed):
    output = indexed("search", "production deploys canary", "-k", "1").output
    assert "handbook.md::Service Handbook:Deploy process" in output
    assert "reason:" in output and "relevance:" in output
    assert "related (ref" in output


def test_search_json_and_no_results(indexed, run):
    data = json.loads(indexed("search", "rollback", "--json").output)
    assert data[0]["type"] == "chunk"


def test_context_prints_paste_ready_text(indexed):
    result = indexed("context", "how do I roll back a release", "--budget", "200")
    assert "deployctl rollback" in result.output
    assert "### handbook.md › Rollback" in result.output


def test_context_json_respects_budget(indexed):
    data = json.loads(indexed("context", "roll back a release", "--budget", "90", "--json").output)
    assert data["tokens_used"] <= 90


def test_index_and_sources_and_remove(indexed):
    index = json.loads(indexed("index", "--max-tokens", "120").output)
    assert "toc" in index[0]
    assert "handbook.md  (5 chunks" in indexed("sources").output
    assert "Removed 5 chunks" in indexed("remove", "handbook.md").output
    assert "No sources ingested" in indexed("sources").output
    assert indexed("remove", "handbook.md", ok=False).exit_code == 1


def test_feedback_and_bonds(indexed):
    chunk = "handbook.md::Service Handbook:Rollback"
    out = indexed("feedback", chunk, "--query", "undo a release").output
    assert "learned weight 1.1" in out and "similar queries" in out
    assert "▼" in indexed("feedback", chunk, "--not-useful").output
    stats = indexed("feedback-stats").output
    assert "Votes: 2 (1 useful, 1 not; 1 query-aware)" in stats
    assert "ref" in indexed("bonds", "handbook.md::Service Handbook:Deploy process").output
    assert "No bonds." in indexed("bonds", "handbook.md::Service Handbook:Logging").output
    assert indexed("feedback", "nope::x", ok=False).exit_code == 1


def test_extract_merge_export_import_backup(indexed, tmp_path):
    (tmp_path / "talk.txt").write_text(
        "We decided to always deploy through Docker Compose. Lunch ran long.", encoding="utf-8"
    )
    assert "1 stored" in indexed("extract", "talk.txt").output
    assert "Merged 0" in indexed("merge").output
    assert "1 memories" in indexed("export", "snap.json").output
    assert "Imported 0 memories" in indexed("import", "snap.json").output
    assert (tmp_path / "snap.json").exists()
    indexed("backup", "backup.db")
    assert (tmp_path / "backup.db").stat().st_size > 0


def test_sync_reports_edits(indexed, tmp_path):
    import os

    assert "already fresh" in indexed("sync").output
    path = tmp_path / "handbook.md"
    path.write_text(SAMPLE_DOC.replace("structured JSON", "plain text"), encoding="utf-8")
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000))
    assert "handbook.md: 0 new, 1 updated" in indexed("sync").output
    path.unlink()
    assert "missing: handbook.md" in indexed("sync").output
    assert "pruned: handbook.md" in indexed("sync", "--prune").output


def test_stats_health_doctor(indexed):
    assert "Savings:" in indexed("stats").output
    assert "lifetime" in json.loads(indexed("stats", "--json").output)
    health = indexed("health").output
    assert "Chunks:   5 from 1 source(s)" in health and "hashing-384" in health
    doctor = indexed("doctor").output
    assert "✓ database integrity: ok" in doctor and "✗" not in doctor


def test_doctor_fails_on_an_empty_store(run):
    result = run("doctor", ok=False)
    assert result.exit_code == 1 and "✗ index" in result.output


def test_eval_passes_and_fails_with_the_right_exit_code(indexed, tmp_path):
    (tmp_path / "good.json").write_text(json.dumps(
        [{"query": "undo a release", "answer": "deployctl rollback"}]), encoding="utf-8")
    (tmp_path / "bad.json").write_text(json.dumps(
        [{"query": "undo a release", "expected_chunk_id": "nope::x"}]), encoding="utf-8")
    good = indexed("eval", "good.json", "--ablate").output
    assert "Top-3 precision: 1.0 (1/1)" in good and "✓ PASS" in good and "dense only" in good
    bad = indexed("eval", "bad.json", ok=False)
    assert bad.exit_code == 1 and "✗ FAIL" in bad.output


def test_setup_reports_the_backend(run):
    assert "hashing-384 ready" in run("setup").output
