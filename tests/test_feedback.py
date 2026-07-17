from __future__ import annotations

import numpy as np
import pytest

from molecule.config import MoleculeConfig
from molecule.core.feedback import feedback_stats, record_feedback
from molecule.db import get_connection, init_schema


@pytest.fixture
def conn(tmp_path):
    c = get_connection(tmp_path / ".molecule" / "store.db")
    init_schema(c)
    return c


@pytest.fixture
def config(tmp_path):
    return MoleculeConfig(db_path=tmp_path / ".molecule" / "store.db")


def _add_chunk(conn, chunk_id: str, learned_weight: float = 1.0):
    vec = np.zeros(4, dtype=np.float32)
    conn.execute(
        "INSERT INTO chunks (chunk_id, source_file, heading_path, content, "
        "token_count, content_hash, learned_weight, embedding, last_updated) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (chunk_id, "f.md", '["H"]', "body", 10, "h", learned_weight,
         vec.tobytes(), "2026-01-01"),
    )
    conn.commit()


def test_positive_vote_moves_up(conn, config):
    _add_chunk(conn, "f.md::A")
    result = record_feedback("f.md::A", True, "q", conn, config)
    # (1-0.2)*1.0 + 0.2*1.5 = 1.10
    assert result["learned_weight"] == pytest.approx(1.10)
    assert result["delta"] == pytest.approx(0.10)


def test_negative_vote_moves_down(conn, config):
    _add_chunk(conn, "f.md::A")
    result = record_feedback("f.md::A", False, "q", conn, config)
    # (1-0.2)*1.0 + 0.2*0.5 = 0.90
    assert result["learned_weight"] == pytest.approx(0.90)


def test_clamps_at_max(conn, config):
    _add_chunk(conn, "f.md::A")
    for _ in range(50):
        result = record_feedback("f.md::A", True, "q", conn, config)
    assert result["learned_weight"] <= config.learned_weight_max
    assert result["learned_weight"] == pytest.approx(1.5, abs=0.01)


def test_clamps_at_min(conn, config):
    _add_chunk(conn, "f.md::A")
    for _ in range(50):
        result = record_feedback("f.md::A", False, "q", conn, config)
    assert result["learned_weight"] >= config.learned_weight_min
    assert result["learned_weight"] == pytest.approx(0.5, abs=0.01)


def test_one_bad_vote_cannot_tank(conn, config):
    _add_chunk(conn, "f.md::A")
    result = record_feedback("f.md::A", False, "q", conn, config)
    assert result["learned_weight"] >= 0.9  # only a nudge, not a cliff


def test_unknown_chunk_errors(conn, config):
    result = record_feedback("nope::X", True, "q", conn, config)
    assert "error" in result


def test_feedback_row_persisted(conn, config):
    _add_chunk(conn, "f.md::A")
    record_feedback("f.md::A", True, "my query", conn, config)
    row = conn.execute("SELECT * FROM chunk_feedback").fetchone()
    assert row["chunk_id"] == "f.md::A"
    assert row["useful"] == 1
    assert row["query"] == "my query"


def test_weight_persists_across_connections(tmp_path, config):
    conn = get_connection(tmp_path / ".molecule" / "store.db")
    init_schema(conn)
    _add_chunk(conn, "f.md::A")
    record_feedback("f.md::A", True, "q", conn, config)
    conn.close()

    conn2 = get_connection(tmp_path / ".molecule" / "store.db")
    init_schema(conn2)
    row = conn2.execute(
        "SELECT learned_weight FROM chunks WHERE chunk_id='f.md::A'"
    ).fetchone()
    assert row["learned_weight"] == pytest.approx(1.10)


def test_feedback_stats(conn, config):
    _add_chunk(conn, "f.md::A")
    _add_chunk(conn, "f.md::B")
    record_feedback("f.md::A", True, "q1", conn, config)
    record_feedback("f.md::B", False, "q2", conn, config)
    s = feedback_stats(conn)
    assert s["votes"] == 2
    assert s["useful"] == 1
    assert s["not_useful"] == 1
    assert {e["chunk_id"] for e in s["learned"]} == {"f.md::A", "f.md::B"}


def test_feedback_changes_ranking(conn, config):
    """Two chunks with identical embeddings — positive feedback breaks the tie."""
    from molecule.core.retriever import rank_candidates

    vec = np.ones(384, dtype=np.float32)
    vec /= np.linalg.norm(vec)
    for cid in ("f.md::A", "f.md::B"):
        conn.execute(
            "INSERT INTO chunks (chunk_id, source_file, heading_path, content, "
            "token_count, content_hash, embedding, last_updated) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (cid, "f.md", '["Same heading"]', "identical body text", 10,
             f"h-{cid}", vec.tobytes(), "2026-01-01"),
        )
    conn.commit()

    for _ in range(3):
        record_feedback("f.md::B", True, "identical", conn, config)

    results = rank_candidates("identical body text", conn, 2, config)
    assert results[0]["id"] == "f.md::B"
    assert "learned boost: +" in results[0]["reason"]
