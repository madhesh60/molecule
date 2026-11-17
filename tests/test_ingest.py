"""Ingest: incremental, atomic, self-describing, and kind to what was learned."""
from __future__ import annotations

from pathlib import Path

import pytest

from molecule.core.feedback import record_feedback
from molecule.core.ingest import (
    do_ingest,
    ingest_file,
    iter_markdown_files,
    list_sources,
    remove_source,
    source_key,
)
from tests.conftest import make_config

DOC = "# Guide\n\n## Alpha\n\n{alpha}\n\n## Beta\n\n{beta}\n"
LONG = "This sentence pads the section so it is not merged into a sibling. " * 6


def _write(path: Path, alpha: str = "alpha body", beta: str = "beta body") -> Path:
    path.write_text(DOC.format(alpha=f"{alpha}. {LONG}", beta=f"{beta}. {LONG}"),
                    encoding="utf-8")
    return path


def _ids(conn) -> set[str]:
    return {r[0] for r in conn.execute("SELECT chunk_id FROM chunks")}


def test_first_ingest_creates_chunks_and_a_source_row(conn, cfg, tmp_path):
    result = ingest_file(_write(tmp_path / "guide.md"), conn, cfg)
    assert (result["new"], result["updated"], result["skipped"]) == (2, 0, 0)
    assert result["source"] == "guide.md"
    assert _ids(conn) == {"guide.md::Guide:Alpha", "guide.md::Guide:Beta"}
    source = conn.execute("SELECT * FROM sources").fetchone()
    assert source["chunk_count"] == 2 and source["total_tokens"] > 0


def test_unchanged_file_takes_the_fast_path(conn, cfg, tmp_path):
    path = _write(tmp_path / "guide.md")
    ingest_file(path, conn, cfg)
    again = ingest_file(path, conn, cfg)
    assert again["unchanged"] is True
    assert (again["new"], again["updated"], again["skipped"]) == (0, 0, 2)
    assert again["timing"]["embed_s"] == 0.0


def test_only_the_edited_section_is_re_embedded(conn, cfg, tmp_path):
    path = _write(tmp_path / "guide.md")
    ingest_file(path, conn, cfg)
    before = dict(conn.execute("SELECT chunk_id, embedding FROM chunks").fetchall())
    result = ingest_file(_write(path, beta="beta body rewritten entirely"), conn, cfg)
    assert (result["new"], result["updated"], result["skipped"]) == (0, 1, 1)
    after = dict(conn.execute("SELECT chunk_id, embedding FROM chunks").fetchall())
    assert after["guide.md::Guide:Alpha"] == before["guide.md::Guide:Alpha"]
    assert after["guide.md::Guide:Beta"] != before["guide.md::Guide:Beta"]


def test_force_re_embeds_everything(conn, cfg, tmp_path):
    path = _write(tmp_path / "guide.md")
    ingest_file(path, conn, cfg)
    assert ingest_file(path, conn, cfg, force=True)["updated"] == 2


def test_removed_section_is_deleted_from_the_index(conn, cfg, tmp_path):
    path = _write(tmp_path / "guide.md")
    ingest_file(path, conn, cfg)
    path.write_text(f"# Guide\n\n## Alpha\n\nalpha body. {LONG}\n", encoding="utf-8")
    result = ingest_file(path, conn, cfg)
    assert result["removed"] == 1
    assert _ids(conn) == {"guide.md::Guide:Alpha"}


def test_renamed_heading_keeps_what_the_router_learned(conn, cfg, tmp_path):
    path = _write(tmp_path / "guide.md")
    ingest_file(path, conn, cfg)
    record_feedback("guide.md::Guide:Beta", True, "a question about beta", conn, cfg)
    path.write_text(path.read_text(encoding="utf-8").replace("## Beta", "## Beta renamed"),
                    encoding="utf-8")
    ingest_file(path, conn, cfg)
    new_id = "guide.md::Guide:Beta renamed"
    assert new_id in _ids(conn) and "guide.md::Guide:Beta" not in _ids(conn)
    row = conn.execute("SELECT learned_weight FROM chunks WHERE chunk_id=?", (new_id,)).fetchone()
    assert row["learned_weight"] == pytest.approx(1.10)
    assert conn.execute(
        "SELECT COUNT(*) FROM chunk_feedback WHERE chunk_id=?", (new_id,)
    ).fetchone()[0] == 1


def test_changing_chunk_settings_rebuilds_the_source(conn, cfg, tmp_path):
    path = _write(tmp_path / "guide.md")
    ingest_file(path, conn, cfg)
    smaller = make_config(tmp_path, max_chunk_tokens=40, min_chunk_tokens=5)
    result = ingest_file(path, conn, smaller)
    assert result["unchanged"] is False
    assert result["total"] > 2   # the same text is now cut into more chunks


def test_failed_ingest_leaves_the_previous_index_intact(conn, cfg, tmp_path, monkeypatch):
    path = _write(tmp_path / "guide.md")
    ingest_file(path, conn, cfg)
    snapshot = conn.execute("SELECT chunk_id, content FROM chunks ORDER BY chunk_id").fetchall()

    from molecule.core import ingest as ingest_module

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(ingest_module, "rebuild_structural_bonds", boom)
    _write(path, alpha="completely new alpha", beta="completely new beta")
    with pytest.raises(RuntimeError):
        ingest_file(path, conn, cfg)
    assert conn.execute(
        "SELECT chunk_id, content FROM chunks ORDER BY chunk_id"
    ).fetchall() == snapshot


def test_source_key_is_repo_relative(tmp_path):
    (tmp_path / "docs").mkdir()
    assert source_key(tmp_path / "docs" / "a.md", tmp_path) == "docs/a.md"
    assert source_key(Path("/elsewhere/b.md"), tmp_path) == "b.md"


def test_same_file_name_in_two_folders_does_not_collide(conn, cfg, tmp_path):
    for folder in ("api", "web"):
        (tmp_path / folder).mkdir()
        _write(tmp_path / folder / "README.md", alpha=f"{folder} alpha")
        ingest_file(tmp_path / folder / "README.md", conn, cfg)
    assert {r["source_file"] for r in list_sources(conn)} == {"api/README.md", "web/README.md"}
    assert len(_ids(conn)) == 4


def test_directory_ingest_skips_hidden_and_vendored_folders(conn, cfg, tmp_path):
    for folder in ("docs", ".git", "node_modules"):
        (tmp_path / folder).mkdir()
        _write(tmp_path / folder / "x.md")
    (tmp_path / "docs" / "notes.txt").write_text("not markdown", encoding="utf-8")
    assert [p.relative_to(tmp_path).as_posix() for p in iter_markdown_files(tmp_path, 100)] == [
        "docs/x.md"
    ]
    result = do_ingest(tmp_path, conn, cfg)
    assert result["files"] == 1 and result["new"] == 2 and result["errors"] == []


def test_directory_without_markdown_is_an_error(conn, cfg, tmp_path):
    (tmp_path / "empty").mkdir()
    assert "error" in do_ingest(tmp_path / "empty", conn, cfg)


def test_oversized_file_is_refused(conn, tmp_path):
    cfg = make_config(tmp_path, max_file_bytes=100)
    assert "too large" in ingest_file(_write(tmp_path / "guide.md"), conn, cfg)["error"]


def test_remove_source_by_key_path_or_bare_name(conn, cfg, tmp_path):
    (tmp_path / "docs").mkdir()
    path = _write(tmp_path / "docs" / "guide.md")
    for name in ("docs/guide.md", str(path), "guide.md"):
        ingest_file(path, conn, cfg, force=True)
        assert remove_source(conn, name)["chunks_deleted"] == 2
        assert not _ids(conn) and not list_sources(conn)
    assert "error" in remove_source(conn, "missing.md")


def test_ingest_tags_vectors_with_the_model(conn, cfg, tmp_path):
    ingest_file(_write(tmp_path / "guide.md"), conn, cfg)
    assert {r[0] for r in conn.execute("SELECT embedding_model FROM chunks")} == {"hashing-384"}


def test_passage_vectors_are_stored_only_when_enabled(conn, tmp_path):
    path = _write(tmp_path / "guide.md")
    assert ingest_file(path, conn, make_config(tmp_path))["passages"] == 0
    deep = ingest_file(path, conn, make_config(tmp_path, passage_vectors=True))
    assert deep["passages"] > 0
    row = conn.execute("SELECT passage_embeddings, passage_count FROM chunks").fetchone()
    assert len(row["passage_embeddings"]) == row["passage_count"] * 384 * 4
