"""Router index, hybrid ranking, relevance labels, guardrails, fetch."""
from __future__ import annotations

import json

import numpy as np
import pytest

from molecule.core.ingest import ingest_file
from molecule.core.retriever import (
    chunk_corpus,
    fetch_chunks,
    load_router_index,
    rank_candidates,
    relevance_label,
    score_chunks,
)
from molecule.core.tokens import count_tokens
from tests.conftest import make_config

DOC = (
    "# Authentication\n\nUse JWT tokens for all API calls. Tokens expire after 15 minutes.\n\n"
    "# Database\n\nPostgres is the primary store. Never use raw SQL in handlers.\n\n"
    "# Deployment\n\nDeploy via Docker. Use multi-stage builds for smaller images.\n\n"
    "# Guardrails\n\n<!-- always_include -->\nNever touch the production database by hand.\n"
)


@pytest.fixture
def ingested_db(conn, cfg, tmp_path):
    md = tmp_path / "doc.md"
    md.write_text(DOC, encoding="utf-8")
    assert "error" not in ingest_file(md, conn, cfg)
    return conn, cfg


def test_router_index_no_content(ingested_db):
    conn, _ = ingested_db
    index = load_router_index(conn)
    assert len(index) >= 3
    for item in index:
        assert "content" not in item
        assert "embedding" not in item


def test_router_index_is_in_document_order(ingested_db):
    conn, _ = ingested_db
    assert [e["chunk_id"].split("::")[1] for e in load_router_index(conn)] == [
        "Authentication", "Database", "Deployment", "Guardrails",
    ]


@pytest.mark.parametrize("budget", [40, 80, 150, 400, 5000])
def test_router_index_always_fits_its_budget(ingested_db, budget):
    conn, _ = ingested_db
    index = load_router_index(conn, max_tokens=budget)
    assert count_tokens(json.dumps(index, ensure_ascii=False)) <= budget


def test_router_index_degrades_to_a_table_of_contents(ingested_db):
    conn, _ = ingested_db
    toc = load_router_index(conn, max_tokens=70)
    assert toc and "toc" in toc[0]
    assert "Authentication" in toc[0]["toc"]            # heading → token count
    assert isinstance(toc[0]["toc"]["Authentication"], int)


def test_router_index_prefix_drills_down(ingested_db):
    conn, _ = ingested_db
    only = load_router_index(conn, prefix="doc.md::Data")
    assert [e["chunk_id"] for e in only] == ["doc.md::Database"]
    assert load_router_index(conn, prefix="100%_literal") == []   # LIKE wildcards are escaped


def test_search_returns_reason(ingested_db):
    conn, cfg = ingested_db
    results = rank_candidates("JWT authentication tokens", conn, 3, cfg)
    assert results
    assert all("reason" in r for r in results)
    assert results[0]["id"] == "doc.md::Authentication"
    assert "keyword" in results[0]["reason"]


def test_results_carry_relevance_and_fetch_cost(ingested_db):
    conn, cfg = ingested_db
    top = rank_candidates("postgres database store", conn, 1, cfg)[0]
    assert top["relevance"] in {"high", "medium"}
    assert top["tokens"] > 0


def test_off_topic_query_is_labelled_low(ingested_db):
    conn, cfg = ingested_db
    results = rank_candidates("recipe for sourdough bread", conn, 2, cfg)
    assert all(r["relevance"] == "low" for r in results if r["type"] == "chunk")


def test_relevance_label_thresholds(cfg):
    assert relevance_label(cfg.relevance_high, cfg) == "high"
    assert relevance_label(cfg.relevance_low, cfg) == "medium"
    assert relevance_label(cfg.relevance_low - 0.01, cfg) == "low"


def test_guardrail_is_appended_but_never_ranked_up(ingested_db):
    conn, cfg = ingested_db
    results = rank_candidates("docker deployment builds", conn, 1, cfg)
    assert results[0]["id"] == "doc.md::Deployment"          # merit decides rank 1
    assert results[-1]["id"] == "doc.md::Guardrails"         # guardrail is still delivered
    assert results[-1]["always_include"] is True
    assert len(results) == 2


def test_k_is_clamped(ingested_db):
    conn, cfg = ingested_db
    assert len(rank_candidates("database", conn, 0, cfg)) >= 1       # 0 → default k
    assert len(rank_candidates("database", conn, 10_000, cfg)) <= 100


def test_long_summaries_are_shortened_in_results(conn, cfg, tmp_path):
    md = tmp_path / "long.md"
    md.write_text("# Long\n\n" + "A very long lead sentence " * 30 + ".\n", encoding="utf-8")
    ingest_file(md, conn, cfg)
    summary = rank_candidates("long lead sentence", conn, 1, cfg)[0]["summary"]
    assert len(summary) <= 160 and summary.endswith("…")


def test_fetch_returns_content(ingested_db):
    conn, cfg = ingested_db
    results = rank_candidates("database postgres", conn, 1, cfg)
    assert results
    chunks = fetch_chunks([results[0]["id"]], conn)
    assert chunks
    assert "content" in chunks[0]
    assert len(chunks[0]["content"]) > 0


def test_score_is_explained_by_its_parts(ingested_db):
    """final = (w_e·dense + w_k·keyword + w_h·heading) × learned — exactly."""
    conn, cfg = ingested_db
    s = score_chunks("JWT tokens for the API", conn, cfg)
    expected = (
        cfg.emb_weight * s.dense + cfg.keyword_weight * s.keyword + cfg.heading_boost * s.heading
    ) * s.learned
    assert np.allclose(s.final, expected, atol=1e-6)


def test_vectors_are_re_embedded_when_the_model_changes(ingested_db, tmp_path):
    """Self-healing: vectors tagged with another model are never compared
    against the current model's query vectors."""
    conn, cfg = ingested_db
    conn.execute("UPDATE chunks SET embedding_model='some-old-model', embedding=?",
                 (np.ones(8, dtype=np.float32).tobytes(),))
    conn.commit()
    results = rank_candidates("JWT authentication tokens", conn, 1, cfg)
    assert results[0]["id"] == "doc.md::Authentication"
    assert {r[0] for r in conn.execute("SELECT embedding_model FROM chunks")} == {"hashing-384"}
    assert chunk_corpus(conn, cfg).matrix.shape[1] == 384


def test_search_on_empty_store_returns_nothing(conn, cfg):
    assert rank_candidates("anything", conn, 5, cfg) == []


def test_passage_vectors_lift_the_matching_chunk(conn, tmp_path):
    """With the deep index on, one strongly matching sentence raises its chunk."""
    cfg = make_config(tmp_path, passage_vectors=True)
    md = tmp_path / "doc.md"
    md.write_text(DOC, encoding="utf-8")
    ingest_file(md, conn, cfg)
    with_passages = score_chunks("tokens expire after 15 minutes", conn, cfg)
    without = score_chunks("tokens expire after 15 minutes", conn,
                           make_config(tmp_path, passage_vectors=True, passage_weight=0.0))
    row = with_passages.corpus.row_of["doc.md::Authentication"]
    assert with_passages.dense[row] > without.dense[row]
