from __future__ import annotations

import json
from pathlib import Path

import pytest

from molecule.core.eval import ABLATIONS, run_ablation, run_eval
from molecule.core.ingest import do_ingest, ingest_file
from molecule.db import get_connection, init_schema

DOC = (
    "# Authentication\n\nUse JWT tokens. Never store passwords in plain text.\n\n"
    "# Database\n\nPostgres is the primary store. Use connection pooling.\n\n"
    "# Deployment\n\nDeploy via Docker. Use multi-stage builds for smaller images.\n\n"
    "# Testing\n\nRun pytest before every commit. Coverage must be above 80%.\n\n"
    "# Logging\n\nUse structured JSON logs. Never log secrets or PII.\n"
)


@pytest.fixture
def eval_db(conn, cfg, tmp_path):
    md = tmp_path / "eval_doc.md"
    md.write_text(DOC, encoding="utf-8")
    assert "error" not in ingest_file(md, conn, cfg)
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
    result = run_eval(queries, conn, cfg, k=3)
    assert result["top3_precision"] >= 0.80, (
        f"Precision {result['top3_precision']} below 0.80 target. Misses: {result['misses']}"
    )
    assert result["passes"] is True
    assert 0.0 < result["mrr"] <= 1.0 and 0.0 < result["ndcg_at_3"] <= 1.0
    assert result["latency_ms"]["p50"] <= result["latency_ms"]["p95"]


def test_eval_accepts_tuples(eval_db):
    conn, cfg, fname = eval_db
    result = run_eval([("JWT authentication tokens", f"{fname}::Authentication")], conn, cfg, k=3)
    assert result["hits"] == 1 and result["hit_at_1"] == 1.0 and result["mrr"] == 1.0


def test_eval_accepts_answer_labels(eval_db):
    """An `answer` phrase is a gold label that survives re-chunking."""
    conn, cfg, _ = eval_db
    result = run_eval(
        [{"query": "how do we deploy", "answer": "multi-stage builds"},
         {"query": "how do we deploy", "answer": "this phrase is nowhere"}],
        conn, cfg, k=3,
    )
    assert result["hits"] == 1 and len(result["misses"]) == 1


def test_eval_accepts_several_gold_ids(eval_db):
    conn, cfg, fname = eval_db
    result = run_eval(
        [{"query": "postgres pooling", "expected_chunk_id": ["nope::x", f"{fname}::Database"]}],
        conn, cfg, k=1,
    )
    assert result["hits"] == 1


def test_split_parts_count_as_their_section(conn, tmp_path):
    from tests.conftest import make_config

    cfg = make_config(tmp_path, max_chunk_tokens=60, min_chunk_tokens=5)
    body = "\n\n".join(f"Paragraph {i} is about the kiwi orchard harvest." for i in range(20))
    (tmp_path / "big.md").write_text(f"# Orchard\n\n{body}\n", encoding="utf-8")
    ingest_file(tmp_path / "big.md", conn, cfg)
    result = run_eval([("kiwi orchard harvest", "big.md::Orchard")], conn, cfg, k=3)
    assert result["hits"] == 1


def test_eval_empty_queries():
    result = run_eval([], None, None)
    assert "error" in result


def test_eval_miss_recorded(eval_db):
    conn, cfg, fname = eval_db
    result = run_eval([("JWT authentication tokens", "nonexistent::chunk")], conn, cfg, k=3)
    assert result["hits"] == 0
    assert len(result["misses"]) == 1
    assert result["passes"] is False
    assert result["mrr"] == 0.0


def test_ablation_reports_every_variant(eval_db):
    conn, cfg, fname = eval_db
    queries = [("JWT authentication tokens", f"{fname}::Authentication"),
               ("postgres database connection", f"{fname}::Database")]
    rows = run_ablation(queries, conn, cfg, k=3)
    assert [r["variant"] for r in rows] == [name for name, _ in ABLATIONS]
    assert rows[0]["top3_precision"] == 1.0
    assert all(0.0 <= r["mrr"] <= 1.0 for r in rows)


# --- The real model on the real documents (skipped when not cached) ----------------

REPO_ROOT = Path(__file__).parent.parent


def _real_eval(tmp_path, model_cfg, doc: Path, query_file: Path) -> dict:
    conn = get_connection(model_cfg.db_path)
    init_schema(conn)
    assert "error" not in do_ingest(doc, conn, model_cfg)
    queries = json.loads(query_file.read_text(encoding="utf-8"))
    return run_eval(queries, conn, model_cfg, k=3)


@pytest.mark.model
@pytest.mark.slow
@pytest.mark.parametrize("query_file", ["eval_queries.json", "eval_queries_adversarial.json"])
def test_real_agents_md_hits_precision_target(tmp_path, model_cfg, query_file):
    """End-to-end: ingest the repo's real AGENTS.md, run the shipped query
    sets, assert the 80% top-3 target genuinely holds."""
    result = _real_eval(tmp_path, model_cfg, REPO_ROOT / "AGENTS.md", REPO_ROOT / query_file)
    assert result["top3_precision"] >= 0.80, (
        f"top-3 precision {result['top3_precision']} < 0.80. Misses: {result['misses']}"
    )


@pytest.mark.model
@pytest.mark.slow
@pytest.mark.parametrize("name,floor", [("handbook_direct", 0.95), ("handbook_paraphrase", 0.80)])
def test_handbook_benchmark_holds(tmp_path, model_cfg, name, floor):
    """The held-out handbook: direct questions must be near-perfect and
    paraphrased ones must clear the 80% target."""
    bench = REPO_ROOT / "benchmarks"
    result = _real_eval(tmp_path, model_cfg, bench / "corpus" / "handbook.md",
                        bench / "queries" / f"{name}.json")
    assert result["top3_precision"] >= floor, result["misses"]
