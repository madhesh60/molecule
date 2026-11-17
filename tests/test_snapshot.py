"""Export / import of what the store has learned, and whole-store backups."""
from __future__ import annotations

import json
import sqlite3

from molecule import Molecule
from tests.conftest import SAMPLE_DOC, make_config


def _teach(mol: Molecule) -> None:
    mol.remember("Deploys never go out on Fridays", confidence=0.9)
    mol.remember("Currently debugging the flaky webhook test", mem_type="task_state")
    ids = [e["chunk_id"] for e in mol.index()]
    mol.feedback(ids[0], True, "how do deploys work")
    mol.fetch(ids[:2])
    mol.fetch(ids[:2])


def test_export_writes_plain_json_without_vectors(ingested, tmp_path):
    _teach(ingested)
    report = ingested.export_snapshot(tmp_path / "snap.json")
    assert (report["memories"], report["feedback"], report["bonds"]) == (2, 1, 1)
    payload = json.loads((tmp_path / "snap.json").read_text(encoding="utf-8"))
    assert payload["format"] == "molecule-snapshot"
    assert "embedding" not in json.dumps(payload)


def test_import_into_a_fresh_store_restores_learning(ingested, tmp_path):
    _teach(ingested)
    ingested.export_snapshot(tmp_path / "snap.json")

    other_root = tmp_path / "other"
    other_root.mkdir()
    (other_root / "handbook.md").write_text(SAMPLE_DOC, encoding="utf-8")
    with Molecule(config=make_config(other_root)) as other:
        other.ingest(other_root / "handbook.md")
        report = other.import_snapshot(tmp_path / "snap.json")
        assert (report["memories"], report["feedback"], report["bonds"]) == (2, 1, 1)
        # Imported memories have no vector yet; the first recall embeds them.
        assert "Fridays" in other.recall("when do deploys go out")[0]["text"]
        stats = other.feedback_stats()
        assert stats["votes"] == 1 and stats["bonds"]["cofetch"] == 1


def test_import_is_idempotent(ingested, tmp_path):
    _teach(ingested)
    ingested.export_snapshot(tmp_path / "snap.json")
    again = ingested.import_snapshot(tmp_path / "snap.json")
    assert (again["memories"], again["feedback"], again["bonds"]) == (0, 0, 0)
    assert ingested.feedback_stats()["votes"] == 1


def test_import_skips_learning_about_unknown_chunks(mol, ingested, tmp_path):
    _teach(ingested)
    ingested.export_snapshot(tmp_path / "snap.json")
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    with Molecule(config=make_config(empty_root)) as empty:
        report = empty.import_snapshot(tmp_path / "snap.json")
    assert report["memories"] == 2 and report["feedback"] == 0 and report["skipped"] == 2


def test_import_rejects_files_that_are_not_snapshots(mol, tmp_path):
    (tmp_path / "bad.json").write_text('{"hello": "world"}', encoding="utf-8")
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert "error" in mol.import_snapshot(tmp_path / "bad.json")
    assert "error" in mol.import_snapshot(tmp_path / "broken.json")
    assert "error" in mol.import_snapshot(tmp_path / "missing.json")


def test_backup_is_a_complete_usable_database(ingested, tmp_path):
    _teach(ingested)
    report = ingested.backup(tmp_path / "backups" / "store.db")
    assert report["bytes"] > 0
    copy = sqlite3.connect(report["path"])
    assert copy.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert copy.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == len(ingested.index())
    assert copy.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 2
    copy.close()
