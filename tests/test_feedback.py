from __future__ import annotations

import numpy as np
import pytest

from molecule.core.embedder import embed_query, embedder_for
from molecule.core.feedback import feedback_stats, learned_weights, record_feedback
from molecule.db import get_connection, init_schema


def _add_chunk(conn, chunk_id: str, learned_weight: float = 1.0):
    vec = np.zeros(384, dtype=np.float32)
    conn.execute(
        "INSERT INTO chunks (chunk_id, source_file, heading_path, content, "
        "token_count, content_hash, learned_weight, embedding, last_updated) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (chunk_id, "f.md", '["H"]', "body", 10, "h", learned_weight,
         vec.tobytes(), "2026-01-01"),
    )
    conn.commit()


def _weight_for(conn, cfg, chunk_id: str, query: str) -> float:
    vector = embed_query(query, embedder_for(cfg), cfg.chunk_embed_tokens)
    return float(learned_weights(conn, cfg, [chunk_id], np.ones(1, dtype=np.float32), vector)[0])


def test_positive_vote_moves_up(conn, cfg):
    _add_chunk(conn, "f.md::A")
    result = record_feedback("f.md::A", True, "q", conn, cfg)
    # (1-0.2)*1.0 + 0.2*1.5 = 1.10
    assert result["learned_weight"] == pytest.approx(1.10)
    assert result["delta"] == pytest.approx(0.10)


def test_negative_vote_moves_down(conn, cfg):
    _add_chunk(conn, "f.md::A")
    result = record_feedback("f.md::A", False, "q", conn, cfg)
    # (1-0.2)*1.0 + 0.2*0.5 = 0.90
    assert result["learned_weight"] == pytest.approx(0.90)


def test_clamps_at_max(conn, cfg):
    _add_chunk(conn, "f.md::A")
    for _ in range(50):
        result = record_feedback("f.md::A", True, "q", conn, cfg)
    assert result["learned_weight"] <= cfg.learned_weight_max
    assert result["learned_weight"] == pytest.approx(1.5, abs=0.01)


def test_clamps_at_min(conn, cfg):
    _add_chunk(conn, "f.md::A")
    for _ in range(50):
        result = record_feedback("f.md::A", False, "q", conn, cfg)
    assert result["learned_weight"] >= cfg.learned_weight_min
    assert result["learned_weight"] == pytest.approx(0.5, abs=0.01)


def test_one_bad_vote_cannot_tank(conn, cfg):
    _add_chunk(conn, "f.md::A")
    result = record_feedback("f.md::A", False, "q", conn, cfg)
    assert result["learned_weight"] >= 0.9  # only a nudge, not a cliff


def test_unknown_chunk_errors(conn, cfg):
    result = record_feedback("nope::X", True, "q", conn, cfg)
    assert "error" in result


def test_feedback_row_persisted(conn, cfg):
    _add_chunk(conn, "f.md::A")
    record_feedback("f.md::A", True, "my query", conn, cfg)
    row = conn.execute("SELECT * FROM chunk_feedback").fetchone()
    assert row["chunk_id"] == "f.md::A"
    assert row["useful"] == 1
    assert row["query"] == "my query"
    assert row["query_embedding"] is not None and row["embedding_model"] == "hashing-384"


def test_weight_persists_across_connections(cfg):
    conn = get_connection(cfg.db_path)
    init_schema(conn)
    _add_chunk(conn, "f.md::A")
    record_feedback("f.md::A", True, "q", conn, cfg)
    conn.close()

    conn2 = get_connection(cfg.db_path)
    init_schema(conn2)
    row = conn2.execute(
        "SELECT learned_weight FROM chunks WHERE chunk_id='f.md::A'"
    ).fetchone()
    assert row["learned_weight"] == pytest.approx(1.10)


def test_feedback_stats(conn, cfg):
    _add_chunk(conn, "f.md::A")
    _add_chunk(conn, "f.md::B")
    record_feedback("f.md::A", True, "q1", conn, cfg)
    record_feedback("f.md::B", False, "", conn, cfg)
    s = feedback_stats(conn)
    assert s["votes"] == 2
    assert s["useful"] == 1
    assert s["not_useful"] == 1
    assert s["query_aware_votes"] == 1
    assert {e["chunk_id"] for e in s["learned"]} == {"f.md::A", "f.md::B"}


def test_feedback_changes_ranking(conn, cfg):
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
        record_feedback("f.md::B", True, "identical body text", conn, cfg)

    results = rank_candidates("identical body text", conn, 2, cfg)
    assert results[0]["id"] == "f.md::B"
    assert "learned boost: +" in results[0]["reason"]


# --- Query-aware learning -----------------------------------------------------------

def test_same_query_reproduces_the_plain_ema(conn, cfg):
    """Kernel weight 1 → exactly the classic bounded EMA: 1.0 → 1.10 → 1.18."""
    _add_chunk(conn, "f.md::A")
    record_feedback("f.md::A", True, "how do deploys work", conn, cfg)
    assert _weight_for(conn, cfg, "f.md::A", "how do deploys work") == pytest.approx(1.10)
    record_feedback("f.md::A", True, "how do deploys work", conn, cfg)
    assert _weight_for(conn, cfg, "f.md::A", "how do deploys work") == pytest.approx(1.18)


def test_unrelated_query_is_barely_affected(conn, cfg):
    """The whole point: a vote for one topic must not bend another topic."""
    _add_chunk(conn, "f.md::A")
    for _ in range(5):
        record_feedback("f.md::A", True, "how do deploys work", conn, cfg)
    related = _weight_for(conn, cfg, "f.md::A", "how do deploys work")
    unrelated = _weight_for(conn, cfg, "f.md::A", "bundle size budget for the dashboard")
    assert related > 1.3
    assert 1.0 <= unrelated < 1.06            # only the small global leak applies


def test_similar_query_gets_a_partial_boost(conn, cfg):
    _add_chunk(conn, "f.md::A")
    for _ in range(3):
        record_feedback("f.md::A", True, "how do I roll back a deploy", conn, cfg)
    exact = _weight_for(conn, cfg, "f.md::A", "how do I roll back a deploy")
    similar = _weight_for(conn, cfg, "f.md::A", "roll back the deploy quickly")
    unrelated = _weight_for(conn, cfg, "f.md::A", "translation of user facing strings")
    assert exact > similar > unrelated


def test_vote_without_query_applies_everywhere(conn, cfg):
    _add_chunk(conn, "f.md::A")
    record_feedback("f.md::A", False, "", conn, cfg)
    for query in ("how do deploys work", "bundle size budget"):
        assert _weight_for(conn, cfg, "f.md::A", query) == pytest.approx(0.90)


def test_weights_never_leave_the_clamp_range(conn, cfg):
    _add_chunk(conn, "f.md::A")
    for i in range(40):
        record_feedback("f.md::A", i % 3 != 0, f"question number {i % 4}", conn, cfg)
    for query in ("question number 1", "something else entirely"):
        weight = _weight_for(conn, cfg, "f.md::A", query)
        assert cfg.learned_weight_min <= weight <= cfg.learned_weight_max


def test_old_votes_age_out_of_the_window(conn, tmp_path):
    from tests.conftest import make_config

    cfg = make_config(tmp_path, feedback_window=3)
    _add_chunk(conn, "f.md::A")
    for _ in range(10):
        record_feedback("f.md::A", False, "the question", conn, cfg)
    for _ in range(3):
        record_feedback("f.md::A", True, "the question", conn, cfg)
    # Only the 3 most recent (positive) votes are replayed.
    assert _weight_for(conn, cfg, "f.md::A", "the question") > 1.2


def test_votes_from_an_older_store_are_embedded_on_first_use(conn, cfg):
    """Votes written before query vectors existed have text but no vector."""
    _add_chunk(conn, "f.md::A")
    conn.execute(
        "INSERT INTO chunk_feedback (chunk_id, useful, query, ts) "
        "VALUES ('f.md::A', 1, 'how do deploys work', '2026-01-01')"
    )
    conn.commit()
    assert _weight_for(conn, cfg, "f.md::A", "how do deploys work") == pytest.approx(1.10)
    assert conn.execute("SELECT query_embedding FROM chunk_feedback").fetchone()[0] is not None


def test_chunk_without_votes_keeps_its_stored_weight(conn, cfg):
    _add_chunk(conn, "f.md::A", learned_weight=1.3)
    vector = embed_query("anything", embedder_for(cfg), cfg.chunk_embed_tokens)
    prior = np.asarray([1.3], dtype=np.float32)
    assert learned_weights(conn, cfg, ["f.md::A"], prior, vector)[0] == pytest.approx(1.3)


def test_vote_cache_respects_settings(conn, cfg, tmp_path):
    """The replay window is a setting: changing it must change the result
    even though no vote was written in between."""
    from tests.conftest import make_config

    _add_chunk(conn, "f.md::A")
    for _ in range(6):
        record_feedback("f.md::A", True, "the question", conn, cfg)
    full = _weight_for(conn, cfg, "f.md::A", "the question")
    short = _weight_for(conn, make_config(tmp_path, feedback_window=1), "f.md::A", "the question")
    assert short == pytest.approx(1.10)
    assert full > short
    assert _weight_for(conn, cfg, "f.md::A", "the question") == pytest.approx(full)
