from __future__ import annotations

import json
from datetime import datetime

import pytest

from molecule.config import MoleculeConfig
from molecule.core.chunker import chunk_file
from molecule.core.embedder import embed_and_store
from molecule.core.labeler import default_llm_fn, generate_labels
from molecule.core.retriever import fetch_chunks, load_router_index, rank_candidates
from molecule.db import get_connection, init_schema


@pytest.fixture
def ingested_db(tmp_path):
    cfg = MoleculeConfig(db_path=tmp_path / ".molecule" / "store.db")
    conn = get_connection(cfg.db_path)
    init_schema(conn)

    md = tmp_path / "doc.md"
    md.write_text(
        "# Authentication\n\nUse JWT tokens for all API calls.\n\n"
        "# Database\n\nPostgres is the primary store. Never use raw SQL.\n\n"
        "# Deployment\n\nDeploy via Docker. Use multi-stage builds.\n",
        encoding="utf-8",
    )

    sections = chunk_file(md, cfg)
    labels = generate_labels(sections, conn, default_llm_fn)
    now = datetime.utcnow().isoformat()

    for sec, label in zip(sections, labels):
        chunk_id = f"doc::{':'.join(sec.heading_path)}"
        conn.execute(
            """INSERT OR REPLACE INTO chunks
               (chunk_id, source_file, heading_path, summary, content, token_count,
                content_hash, always_include, weight, last_updated)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (chunk_id, str(md), json.dumps(sec.heading_path), label, sec.content,
             sec.token_count, sec.hash, int(sec.always_include), 1.0, now),
        )
        embed_and_store(conn, "chunks", "chunk_id", chunk_id, sec.content, cfg.embedding_model)

    conn.commit()
    return conn, cfg


def test_router_index_no_content(ingested_db):
    conn, _ = ingested_db
    index = load_router_index(conn)
    assert len(index) >= 3
    for item in index:
        assert "content" not in item
        assert "embedding" not in item


def test_search_returns_reason(ingested_db):
    conn, cfg = ingested_db
    results = rank_candidates("JWT authentication tokens", conn, 3, cfg)
    assert results
    assert all("reason" in r for r in results)


def test_fetch_returns_content(ingested_db):
    conn, cfg = ingested_db
    results = rank_candidates("database postgres", conn, 1, cfg)
    assert results
    chunks = fetch_chunks([results[0]["id"]], conn)
    assert chunks
    assert "content" in chunks[0]
    assert len(chunks[0]["content"]) > 0
