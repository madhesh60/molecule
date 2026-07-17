from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from molecule.config import MoleculeConfig
from molecule.core.chunker import chunk_file
from molecule.core.embedder import embed_and_store
from molecule.core.eval import run_eval
from molecule.core.labeler import default_llm_fn, generate_labels
from molecule.db import get_connection, init_schema


@pytest.fixture
def eval_db(tmp_path):
    cfg = MoleculeConfig(db_path=tmp_path / ".molecule" / "store.db")
    conn = get_connection(cfg.db_path)
    init_schema(conn)

    md = tmp_path / "eval_doc.md"
    md.write_text(
        "# Authentication\n\nUse JWT tokens. Never store passwords in plain text.\n\n"
        "# Database\n\nPostgres is the primary store. Use connection pooling.\n\n"
        "# Deployment\n\nDeploy via Docker. Use multi-stage builds for smaller images.\n\n"
        "# Testing\n\nRun pytest before every commit. Coverage must be above 80%.\n\n"
        "# Logging\n\nUse structured JSON logs. Never log secrets or PII.\n",
        encoding="utf-8",
    )

    sections = chunk_file(md, cfg)
    labels = generate_labels(sections, conn, default_llm_fn)
    now = datetime.utcnow().isoformat()

    for sec, label in zip(sections, labels):
        chunk_id = f"{md.name}::{':'.join(sec.heading_path)}"
        conn.execute(
            """INSERT OR REPLACE INTO chunks
               (chunk_id, source_file, heading_path, summary, content, token_count,
                content_hash, always_include, weight, last_updated)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (chunk_id, str(md), json.dumps(sec.heading_path), label, sec.content,
             sec.token_count, sec.hash, 0, 1.0, now),
        )
        embed_and_store(conn, "chunks", "chunk_id", chunk_id, sec.content, cfg.embedding_model)

    conn.commit()
    return conn, cfg, md.name


def test_eval_precision_target(eval_db):
    conn, cfg, fname = eval_db
    queries = [
        {"query": "JWT authentication tokens", "expected_chunk_id": f"{fname}::Authentication"},
        {"query": "postgres database connection", "expected_chunk_id": f"{fname}::Database"},
        {"query": "docker deployment build", "expected_chunk_id": f"{fname}::Deployment"},
        {"query": "pytest coverage testing", "expected_chunk_id": f"{fname}::Testing"},
        {"query": "structured logs no secrets", "expected_chunk_id": f"{fname}::Logging"},
    ]
    pairs = [(q["query"], q["expected_chunk_id"]) for q in queries]
    result = run_eval(pairs, conn, cfg, k=3)
    assert result["top3_precision"] >= 0.80, (
        f"Precision {result['top3_precision']} below 0.80 target. Misses: {result['misses']}"
    )


def test_eval_all_hits_returns_pass(eval_db):
    conn, cfg, fname = eval_db
    # Single query we know will hit
    pairs = [("JWT authentication tokens", f"{fname}::Authentication")]
    result = run_eval(pairs, conn, cfg, k=3)
    assert result["hits"] >= 1


def test_eval_empty_queries():
    result = run_eval([], None, None)
    assert "error" in result


def test_eval_miss_recorded(eval_db):
    conn, cfg, fname = eval_db
    pairs = [("JWT authentication tokens", "nonexistent::chunk")]
    result = run_eval(pairs, conn, cfg, k=3)
    assert result["hits"] == 0
    assert len(result["misses"]) == 1
    assert result["passes"] is False


@pytest.mark.slow
def test_real_agents_md_hits_precision_target(tmp_path):
    """End-to-end: ingest the repo's real AGENTS.md, run the shipped
    eval_queries.json, assert the 80% top-3 target genuinely holds."""
    import json as _json

    from molecule.core.ingest import do_ingest

    repo_root = Path(__file__).parent.parent
    agents_md = repo_root / "AGENTS.md"
    eval_file = repo_root / "eval_queries.json"

    cfg = MoleculeConfig(db_path=tmp_path / ".molecule" / "store.db")
    conn = get_connection(cfg.db_path)
    init_schema(conn)
    do_ingest(agents_md, conn, cfg)

    queries = _json.loads(eval_file.read_text(encoding="utf-8"))
    pairs = [(q["query"], q["expected_chunk_id"]) for q in queries]
    result = run_eval(pairs, conn, cfg, k=3)
    assert result["top3_precision"] >= 0.80, (
        f"top-3 precision {result['top3_precision']} < 0.80. Misses: {result['misses']}"
    )
