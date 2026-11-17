"""The budgeted context compiler: fits the budget, stays relevant, explains itself."""
from __future__ import annotations

import numpy as np
import pytest

from molecule.core.context import build_context, compress_chunk
from molecule.core.embedder import embed_query, embedder_for
from molecule.core.ingest import ingest_file
from molecule.core.memory import write_memory
from molecule.core.tokens import count_tokens
from tests.conftest import make_config

DOC = """\
# Operations

## Rollback

To roll back a service run `deployctl rollback`. The command reverts the service to
the previous healthy image and prints the version that is now serving.
A rollback completes in under five minutes and does not need approval. Anyone on
the owning team may start one at any hour.
Database changes are never rolled back; fix them forward instead. Schema steps are
written so the previous code keeps working against the current schema.

## Deploy process

Production deploys start from chat with the deploy command. The bot posts progress
in the channel so everyone can follow along.
They go out as a canary first, five percent of traffic for fifteen minutes while
error rate and latency are watched.
If the canary looks wrong, roll back — see [Rollback](#rollback). You stay online
until the change is fully out.

## Logging

Logs are structured JSON, one event per line, and every line carries a correlation
identifier so a request can be followed across components.
Never log passwords, tokens or personal data. Debug output is off in production and
can be enabled per tenant for one hour.

## Safety rules

<!-- always_include -->
Never run destructive commands against production without a second person.
"""


@pytest.fixture
def store(conn, cfg, tmp_path):
    path = tmp_path / "ops.md"
    path.write_text(DOC, encoding="utf-8")
    assert "error" not in ingest_file(path, conn, cfg)
    return conn, cfg


@pytest.mark.parametrize("budget", [64, 90, 130, 200, 350, 600, 2000])
def test_context_never_exceeds_its_budget(store, budget):
    conn, cfg = store
    for query in ("how do I roll back a service", "what must never be logged",
                  "production deploy canary"):
        result = build_context(query, conn, cfg, budget=budget)
        assert result["tokens_used"] == count_tokens(result["context"])
        assert result["tokens_used"] <= budget, (budget, query)


def test_large_budget_delivers_the_whole_relevant_chunk(store):
    conn, cfg = store
    result = build_context("how do I roll back a service", conn, cfg, budget=2000)
    first_chunk = next(i for i in result["items"] if i["type"] == "chunk")
    assert first_chunk["id"] == "ops.md::Operations:Rollback"
    assert "compressed" not in first_chunk
    assert "deployctl rollback" in result["context"]
    assert result["confidence"] in {"high", "medium"}


def test_budget_is_a_ceiling_not_a_target(store):
    """Room in the budget must not be filled with unrelated sections."""
    conn, cfg = store
    result = build_context("how do I roll back a service", conn, cfg, budget=5000)
    ids = {i["id"] for i in result["items"]}
    assert "ops.md::Operations:Logging" not in ids
    assert result["tokens_used"] < 1500


def test_guardrail_is_always_included_first(store):
    conn, cfg = store
    result = build_context("what must never be logged", conn, cfg, budget=600)
    assert result["items"][0] == {**result["items"][0], "type": "guardrail",
                                  "id": "ops.md::Operations:Safety rules"}
    assert "second person" in result["context"]


def test_tight_budget_compresses_and_keeps_the_relevant_sentence(store):
    conn, cfg = store
    result = build_context("how long does a rollback take to complete", conn, cfg, budget=100)
    item = next(i for i in result["items"] if i["id"] == "ops.md::Operations:Rollback")
    assert item["compressed"] is True and item["full_tokens"] > item["tokens"]
    assert "under five minutes" in result["context"]
    assert "(excerpt)" in result["context"]


def test_compression_can_be_disabled(store):
    conn, cfg = store
    result = build_context("how long does a rollback take", conn, cfg, budget=100, compress=False)
    assert all(not i.get("compressed") for i in result["items"])
    assert any(o["id"] == "ops.md::Operations:Rollback" for o in result["omitted"])


def test_off_topic_query_gets_no_chunks(store):
    conn, cfg = store
    result = build_context("recipe for sourdough bread", conn, cfg, budget=800)
    assert result["confidence"] == "low"
    assert [i["type"] for i in result["items"]] == ["guardrail"]   # guardrails only
    assert "note" in result


def _chunks_for(root, doc_text: str, query: str) -> list[str]:
    """Ingest `doc_text` into a fresh store under `root`; return the chunk
    headings the compiler includes for `query`."""
    from molecule.db import get_connection, init_schema

    root.mkdir()
    cfg = make_config(root)
    conn = get_connection(cfg.db_path)
    init_schema(conn)
    (root / "ops.md").write_text(doc_text, encoding="utf-8")
    ingest_file(root / "ops.md", conn, cfg)
    result = build_context(query, conn, cfg, budget=2000)
    conn.close()
    return [i["id"].split("::", 1)[1] for i in result["items"] if i["type"] == "chunk"]


def test_bonded_neighbour_gets_a_lower_bar(tmp_path):
    """A chunk that is only half-relevant on its own is included when a
    selected chunk links to it — and left out when nothing links to it."""
    query = "deploy command canary traffic rollback"
    linked = _chunks_for(tmp_path / "linked", DOC, query)
    assert linked == ["Operations:Deploy process", "Operations:Rollback"]

    without_link = DOC.replace(" — see [Rollback](#rollback)", "")
    assert _chunks_for(tmp_path / "unlinked", without_link, query) == [
        "Operations:Deploy process"
    ]


def test_relevant_memories_are_appended(store):
    conn, cfg = store
    write_memory("Always roll back through deployctl, never by hand", "t", 0.9, conn, cfg)
    write_memory("The office coffee machine is on the third floor", "t", 0.9, conn, cfg)
    result = build_context("how do I roll back a service", conn, cfg, budget=1200)
    memories = [i for i in result["items"] if i["type"] == "memory"]
    assert len(memories) == 1
    assert "Remembered facts:" in result["context"]
    assert "coffee" not in result["context"]
    assert build_context("how do I roll back a service", conn, cfg, budget=1200,
                         include_memories=False)["context"].count("Remembered facts:") == 0


def test_items_explain_themselves(store):
    conn, cfg = store
    result = build_context("how do I roll back a service", conn, cfg, budget=1200)
    for item in result["items"]:
        assert item["reason"] and item["tokens"] > 0
    assert result["sources"] == ["ops.md"]


def test_empty_store_returns_an_empty_context(conn, cfg):
    result = build_context("anything", conn, cfg, budget=500)
    assert result["context"] == "" and result["items"] == [] and result["tokens_used"] == 0


def test_near_duplicate_chunks_do_not_both_take_the_budget(conn, tmp_path):
    cfg = make_config(tmp_path, context_mmr_lambda=0.9, context_min_score_ratio=0.2)
    body = (
        "To roll back a service run deployctl rollback. The command reverts the service "
        "to the previous healthy image and prints the version that is now serving."
    )
    other = (
        "A rollback of a service never needs approval. Anyone on the owning team may "
        "start one at any hour, and the chat channel is told automatically."
    )
    (tmp_path / "dup.md").write_text(
        f"# Ops\n\n## Rollback\n\n{body}\n\n## Rollback again\n\n{body}\n\n"
        f"## Rollback approvals\n\n{other}\n",
        encoding="utf-8",
    )
    ingest_file(tmp_path / "dup.md", conn, make_config(tmp_path, min_chunk_tokens=5))
    result = build_context("roll back a service", conn, cfg, budget=130, compress=False)
    ids = [i["id"] for i in result["items"]]
    # After the first copy, the *different* section is worth more than the twin.
    assert ids[0] == "dup.md::Ops:Rollback"
    assert ids[1] == "dup.md::Ops:Rollback approvals"
    assert "dup.md::Ops:Rollback again" not in ids


def test_compress_chunk_keeps_document_order_and_fits(cfg):
    content = (
        "Rollbacks are simple and fast for everyone. A rollback completes in under five minutes. "
        "The canary watches error rate and latency. Translations ship with the mobile train."
    )
    query = "how many minutes does a rollback take"
    vector = embed_query(query, embedder_for(cfg), cfg.chunk_embed_tokens)
    text, tokens = compress_chunk(content, "Rollback", query, vector, 30, cfg)
    assert tokens == count_tokens(text) <= 30
    assert "under five minutes" in text
    assert "Translations" not in text


def test_compress_chunk_handles_a_passage_larger_than_the_room(cfg):
    content = "word " * 400
    vector = np.zeros(384, dtype=np.float32)
    text, tokens = compress_chunk(content, "Big", "word", vector, 20, cfg)
    assert 0 < tokens <= 24 and text.endswith("…")
