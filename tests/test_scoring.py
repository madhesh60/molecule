from __future__ import annotations

import pytest

from molecule.core.scoring import (
    bm25_scores,
    build_lexical_index,
    heading_match_score,
    heading_scores,
    lexical_scores,
    tokenize,
)


def test_tokenize_drops_stopwords_and_stems():
    assert tokenize("go to the databases") == ["go", "databas"]


def test_bm25_exact_term_wins():
    docs = {
        "a": "deployment docker compose production release",
        "b": "testing pytest fixtures unit tests",
        "c": "authentication login oauth session tokens",
    }
    scores = bm25_scores("docker deployment", docs)
    assert scores["a"] > 0.3
    assert scores["b"] == 0.0
    assert scores["c"] == 0.0


def test_scores_are_coverage_not_max_normalised():
    """A document matching one stray word of a long query must score low —
    the top document does NOT automatically get 1.0."""
    docs = {
        "weak": "the pager rotation is weekly and customers are happy",
        "other": "bundle size budgets for the dashboard",
    }
    scores = bm25_scores("zebra xylophone quantum customers", docs)
    assert 0.0 < scores["weak"] < 0.2
    assert scores["other"] == 0.0


def test_full_match_scores_higher_than_partial():
    docs = {
        "full": "rollback the service with deployctl",
        "partial": "the service is healthy",
        "none": "bundle budgets",
    }
    scores = bm25_scores("rollback service deployctl", docs)
    assert scores["full"] > scores["partial"] > scores["none"] == 0.0


def test_scores_bounded_in_unit_interval():
    docs = {"a": "deploy deploy deploy deploy deploy", "b": "deploy once"}
    scores = bm25_scores("deploy", docs)
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_stemming_matches_word_forms():
    docs = {"a": "carrier polling runs every minute", "b": "invoices are archived"}
    scores = bm25_scores("where is the code that polls the carriers", docs)
    assert scores["a"] > scores["b"]


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


def test_stopword_only_query_scores_nothing():
    assert bm25_scores("how do I do this", {"a": "how to do things"}) == {"a": 0.0}


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


def test_vectorised_heading_scores_match_scalar():
    docs = {"a": "body one", "b": "body two", "c": "body three"}
    headings = {"a": "Rollback procedure", "b": "Session Memory Store", "c": "Logging"}
    index = build_lexical_index(docs, headings)
    query = "how do I remember a rollback"
    vector = heading_scores(tokenize(query), index)
    for row, doc_id in enumerate(index.ids):
        assert vector[row] == pytest.approx(heading_match_score(query, [headings[doc_id]]))


def test_expansion_can_be_switched_off():
    index = build_lexical_index({"secrets": "never persist raw secrets"})
    terms = tokenize("stop saving my API key")
    assert lexical_scores(terms, index, expansion=True)[0] > 0.0
    assert lexical_scores(terms, index, expansion=False)[0] == 0.0


def test_compound_words_match_their_spaced_form():
    """'roll back' in the query finds 'rollback' in the corpus: the compound
    is credited for both of its words."""
    from molecule.core.scoring import query_terms

    index = build_lexical_index(
        {"a": "run the rollback command", "b": "roll the dice and step away"},
        {"a": "Rollback", "b": "Games"},
    )
    terms, aliases = query_terms("how do I roll back a release", index)
    assert aliases == {"roll": ["rollback"], "back": ["rollback"]}
    with_aliases = lexical_scores(terms, index, aliases=aliases)
    without = lexical_scores(terms, index)
    assert with_aliases[0] > with_aliases[1]      # the compound document wins
    assert with_aliases[0] > without[0] == 0.0    # and only thanks to the alias
    assert heading_scores(terms, index, aliases=aliases)[0] > 0


def test_compound_of_two_stopwords_becomes_its_own_term():
    from molecule.core.scoring import query_terms

    index = build_lexical_index({"a": "the oncall engineer acknowledges the page"})
    terms, aliases = query_terms("who is on call tonight", index)
    assert "oncall" in aliases.get("call", []) or "oncall" in terms


def test_compounds_are_only_added_when_the_corpus_uses_them():
    from molecule.core.scoring import query_terms

    index = build_lexical_index({"a": "nothing relevant here"})
    terms, aliases = query_terms("roll back a release", index)
    assert terms == tokenize("roll back a release") and aliases == {}
