"""Tests for concept-cluster query expansion (expansion.py + scoring tiers)."""
from __future__ import annotations

from molecule.core.expansion import SYNONYM_GROUPS, expand_terms
from molecule.core.scoring import (
    bm25_scores,
    heading_match_score_tokens,
    tokenize,
)


def test_expand_returns_cluster_siblings():
    exp = expand_terms({"password"})
    assert "secret" in exp and "token" in exp
    assert "password" not in exp  # original tokens excluded


def test_expand_unknown_tokens_empty():
    assert expand_terms({"zebra", "xylophone"}) == set()


def test_expand_mixed_known_unknown():
    exp = expand_terms({"remember", "zebra"})
    assert "memory" in exp
    assert "zebra" not in exp


def test_clusters_are_disjoint():
    """A word in two clusters would make expansion order-dependent."""
    seen: set[str] = set()
    for group in SYNONYM_GROUPS:
        overlap = seen & group
        assert not overlap, f"words in multiple clusters: {overlap}"
        seen |= group


def test_bm25_paraphrase_reaches_synonym_doc():
    """'saving my API key' shares no tokens with the secrets doc — the
    secrets cluster must still route it there."""
    docs = {
        "secrets": "never persist raw secrets redaction pass strips them",
        "chunking": "split markdown sections on heading boundaries",
    }
    scores = bm25_scores("stop saving my API key", docs)
    assert scores["secrets"] > scores["chunking"]


def test_bm25_exact_match_beats_expansion():
    """A doc matching the exact query token must outrank a doc that only
    matches via cluster siblings."""
    docs = {
        "exact": "rotate your password regularly",
        "sibling": "keep the secret in a vault",
    }
    scores = bm25_scores("password", docs)
    assert scores["exact"] > scores["sibling"]


def test_heading_expansion_counts_reduced():
    q = set(tokenize("does it remember decisions"))
    head_direct = set(tokenize("Remember Decisions"))
    head_sibling = set(tokenize("Session Memory Store"))
    direct = heading_match_score_tokens(q, head_direct)
    via_sibling = heading_match_score_tokens(q, head_sibling)
    assert direct > via_sibling > 0.0


def test_heading_score_capped_at_one():
    q = set(tokenize("memory"))
    head = set(tokenize("memory memories remember recall notebook"))
    assert heading_match_score_tokens(q, head) <= 1.0
