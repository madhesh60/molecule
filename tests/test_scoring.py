from __future__ import annotations

from molecule.core.scoring import bm25_scores, heading_match_score, tokenize


def test_tokenize_drops_short_words():
    assert tokenize("go to the database") == ["the", "database"]


def test_bm25_exact_term_wins():
    docs = {
        "a": "deployment docker compose production release",
        "b": "testing pytest fixtures unit tests",
        "c": "authentication login oauth session tokens",
    }
    scores = bm25_scores("docker deployment", docs)
    assert scores["a"] == 1.0  # max-normalized
    assert scores["b"] == 0.0
    assert scores["c"] == 0.0


def test_bm25_rare_term_outweighs_common():
    docs = {
        "a": "common common common rareword",
        "b": "common common common common",
        "c": "common other things here",
    }
    scores = bm25_scores("rareword common", docs)
    assert scores["a"] > scores["b"]


def test_bm25_empty_query():
    assert bm25_scores("", {"a": "text"}) == {"a": 0.0}


def test_bm25_no_docs():
    assert bm25_scores("query", {}) == {}


def test_bm25_no_hits_all_zero():
    scores = bm25_scores("zebra", {"a": "docker deploy", "b": "pytest tests"})
    assert all(v == 0.0 for v in scores.values())


def test_heading_match_full():
    assert heading_match_score("core modules", ["6. Core modules"]) == 1.0


def test_heading_match_partial():
    assert heading_match_score("core banana", ["6. Core modules"]) == 0.5


def test_heading_match_none():
    assert heading_match_score("banana", ["6. Core modules"]) == 0.0
