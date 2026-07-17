from __future__ import annotations

import pytest

from molecule.config import MoleculeConfig
from molecule.core.extractor import _classify, extract_memories
from molecule.db import get_connection, init_schema


@pytest.fixture
def conn(tmp_path):
    c = get_connection(tmp_path / ".molecule" / "store.db")
    init_schema(c)
    return c


@pytest.fixture
def config(tmp_path):
    return MoleculeConfig(db_path=tmp_path / ".molecule" / "store.db")


def test_direct_statement_high_confidence():
    assert _classify("We decided to use PostgreSQL for the main database.") == (0.9, "preference")
    assert _classify("Always run the linter before committing changes.") == (0.9, "preference")
    assert _classify("I prefer tabs over spaces in this codebase.") == (0.9, "preference")


def test_hedged_statement_low_confidence():
    assert _classify("I think we should probably cache the results somewhere.") == (0.6, "preference")
    assert _classify("Maybe the timeout seems too aggressive here.") == (0.6, "preference")


def test_ephemeral_routes_to_task_state():
    conf, mem_type = _classify("We are currently debugging the payment webhook handler.")
    assert mem_type == "task_state"
    assert conf <= 0.6


def test_no_cue_dropped():
    assert _classify("The weather was nice on Tuesday afternoon.") is None


def test_extract_stores_facts(conn, config):
    transcript = (
        "We decided to deploy everything via Docker Compose. "
        "I think maybe Redis could work for the queue. "
        "We are currently debugging the router scoring bug. "
        "The sky is blue and the tests took a while."
    )
    result = extract_memories(transcript, "test_task", conn, config)
    assert result["extracted"] == 3
    assert result["stored"] == 3

    by_conf = {f["text"][:15]: f for f in result["facts"]}
    assert by_conf["We decided to d"]["confidence"] == 0.9
    assert by_conf["I think maybe R"]["confidence"] == 0.6
    assert by_conf["We are currentl"]["type"] == "task_state"

    # task_state fact got an expiry in the DB
    row = conn.execute(
        "SELECT expires_at FROM memories WHERE type='task_state'"
    ).fetchone()
    assert row["expires_at"] is not None


def test_caps_at_max_facts(conn, config):
    transcript = " ".join(
        f"We decided to always use library number {i} for module {i}." for i in range(10)
    )
    result = extract_memories(transcript, "t", conn, config)
    assert result["extracted"] == config.extract_max_facts == 5


def test_secret_redacted_via_write_memory(conn, config):
    transcript = "We decided the deploy key is token: sk-abc123456789012345678901 for CI."
    extract_memories(transcript, "t", conn, config)
    row = conn.execute("SELECT text FROM memories").fetchone()
    assert "sk-abc" not in row["text"]
    assert "[REDACTED]" in row["text"]


def test_duplicate_sentences_deduped_before_store(conn, config):
    transcript = (
        "We decided to use Docker for deployment. "
        "We decided to use Docker for deployment."
    )
    result = extract_memories(transcript, "t", conn, config)
    assert result["extracted"] == 1


def test_llm_fn_rewrites(conn, config):
    transcript = "We decided to always use pytest for every test in the repo."
    result = extract_memories(
        transcript, "t", conn, config, llm_fn=lambda texts: ["CLEAN: " + t for t in texts]
    )
    assert result["facts"][0]["text"].startswith("CLEAN: ")


def test_empty_transcript(conn, config):
    result = extract_memories("", "t", conn, config)
    assert result["extracted"] == 0
    assert result["stored"] == 0
