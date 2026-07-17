from __future__ import annotations

import pytest

from molecule.config import MoleculeConfig
from molecule.core.chunker import chunk_file, parse_markdown
from molecule.db import get_connection, init_schema


@pytest.fixture
def tmp_db(tmp_path):
    cfg = MoleculeConfig(db_path=tmp_path / ".molecule" / "store.db")
    conn = get_connection(cfg.db_path)
    init_schema(conn)
    return conn, cfg


@pytest.fixture
def sample_md(tmp_path):
    md = tmp_path / "sample.md"
    md.write_text(
        "# Section One\n\nThis is the first section with some content.\n\n"
        "## Sub Section\n\nSub content here.\n\n"
        "# Section Two\n\n<!-- always_include -->\nThis must always load.\n",
        encoding="utf-8",
    )
    return md


def test_parse_markdown_headings():
    text = "# H1\n\nBody one.\n\n## H2\n\nBody two.\n"
    sections = parse_markdown(text)
    assert len(sections) == 2
    assert sections[0].heading_path == ["H1"]
    assert sections[1].heading_path == ["H1", "H2"]


def test_always_include_detected(sample_md, tmp_db):
    _, cfg = tmp_db
    sections = chunk_file(sample_md, cfg)
    always = [s for s in sections if s.always_include]
    assert len(always) >= 1


def test_chunk_token_bounds(tmp_path):
    cfg = MoleculeConfig(max_chunk_tokens=50, min_chunk_tokens=5)
    big = tmp_path / "big.md"
    # Create a section that exceeds max_chunk_tokens
    big.write_text("# Big\n\n" + ("word " * 200) + "\n", encoding="utf-8")
    sections = chunk_file(big, cfg)
    for s in sections:
        assert s.token_count <= cfg.max_chunk_tokens + 20  # small overshoot allowed at sentence boundary


def test_content_hash_stable(sample_md, tmp_db):
    _, cfg = tmp_db
    s1 = chunk_file(sample_md, cfg)
    s2 = chunk_file(sample_md, cfg)
    assert [s.hash for s in s1] == [s.hash for s in s2]
