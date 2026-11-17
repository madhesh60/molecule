"""Tests for the tokenizer: stopwords, short-term rules and the light stemmer."""
from __future__ import annotations

import pytest

from molecule.core.normalize import STOPWORDS, stem, tokenize


@pytest.mark.parametrize("forms", [
    ["deploy", "deploys", "deployed", "deploying", "deployment", "deployments"],
    ["cache", "caches", "cached", "caching"],
    ["policy", "policies"],
    ["retry", "retries", "retried"],
    ["log", "logs", "logged", "logging"],
    ["poll", "polls", "polling"],
    ["release", "releases", "released", "releasing"],
    ["index", "indexes", "indexing"],
    ["configure", "configuration", "configurations"],
])
def test_word_forms_share_one_stem(forms):
    assert len({stem(word) for word in forms}) == 1, {w: stem(w) for w in forms}


@pytest.mark.parametrize("a,b", [("role", "roll"), ("test", "text"), ("cache", "catch")])
def test_different_words_stay_different(a, b):
    assert stem(a) != stem(b)


def test_short_words_and_numbers_are_untouched():
    assert stem("api") == "api"
    assert stem("2024") == "2024"


def test_words_ending_in_ss_us_is_keep_their_s():
    assert stem("access") == "access"
    assert stem("status") == "status"
    assert stem("analysis") == "analysis"


def test_stopwords_are_removed():
    assert tokenize("how do I do this with the thing") == ["thing"]
    assert "how" in STOPWORDS and "the" in STOPWORDS


def test_meaningful_two_letter_terms_survive():
    assert tokenize("the CI and DB and UI") == ["ci", "db", "ui"]
    assert tokenize("go to it") == ["go"]


def test_alphanumeric_identifiers_survive():
    assert "p95" in tokenize("p95 latency under 250 ms")
    assert "v2" in tokenize("the /v2 endpoints")


def test_identifiers_split_on_punctuation():
    assert tokenize("mol_search") == ["mol", "search"]


def test_tokenize_is_case_insensitive():
    assert tokenize("DEPLOY Deploy deploy") == [stem("deploy")] * 3


def test_empty_text():
    assert tokenize("") == []
    assert tokenize("   \n ") == []
