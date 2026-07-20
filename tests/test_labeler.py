"""Tests for structure-aware default labels (no-LLM table-of-contents lines)."""
from __future__ import annotations

from molecule.config import MoleculeConfig
from molecule.core.embed_text import EMBEDDING_TEXT_VERSION
from molecule.core.labeler import default_llm_fn, generate_labels
from molecule.db import content_hash, get_connection, init_schema


def _label(text: str) -> str:
    return default_llm_fn([text])[0]


def test_prose_lead_sentence():
    label = _label("The router loads only summaries. Bodies come later.")
    assert label.startswith("The router loads only summaries")


def test_table_lists_first_column():
    text = (
        "| Tool | What it solves |\n"
        "|---|---|\n"
        "| Mem0 | Fact memory |\n"
        "| Letta | Tiered memory |\n"
    )
    label = _label(text)
    assert "Mem0" in label and "Letta" in label
    assert "---" not in label  # separator rows skipped


def test_code_lists_defined_names():
    text = "```python\ndef mol_search(q):\n    pass\n\nclass Router:\n    pass\n```"
    label = _label(text)
    assert "mol_search" in label and "Router" in label


def test_sql_lists_table_names():
    text = "```sql\nCREATE TABLE chunks (id TEXT);\nCREATE TABLE memories (id TEXT);\n```"
    label = _label(text)
    assert "chunks" in label and "memories" in label


def test_lead_sentence_skips_code_fences():
    text = "```mermaid\nflowchart TD\n  A --> B\n```\n\nEverything is local-first."
    label = _label(text)
    assert label.startswith("Everything is local-first")
    assert "flowchart" not in label


def test_bold_phrases_surface():
    label = _label("Some intro sentence here. **live token-savings counter** matters most.")
    assert "live token-savings counter" in label


def test_label_cache_ignores_older_embedding_version(tmp_path):
    """Labels cached under an older embedding_text_version must regenerate."""
    cfg = MoleculeConfig(db_path=tmp_path / "store.db")
    conn = get_connection(cfg.db_path)
    init_schema(conn)

    content = "The router loads only summaries. Bodies come later."
    conn.execute(
        "INSERT INTO chunks (chunk_id, source_file, heading_path, summary, content, "
        "token_count, content_hash, embedding_text_version, last_updated) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        ("f::A", "f", '["A"]', "OLD STALE LABEL", content, 10,
         content_hash(content), EMBEDDING_TEXT_VERSION - 1, "now"),
    )
    conn.commit()

    class Sec:
        pass

    sec = Sec()
    sec.content = content
    labels = generate_labels([sec], conn, default_llm_fn)
    assert labels[0] != "OLD STALE LABEL"

    # Same content stored at the CURRENT version → cache hit.
    conn.execute(
        "UPDATE chunks SET summary='fresh label', embedding_text_version=? WHERE chunk_id='f::A'",
        (EMBEDDING_TEXT_VERSION,),
    )
    conn.commit()
    labels = generate_labels([sec], conn, default_llm_fn)
    assert labels[0] == "fresh label"
