"""Token counting must work with and without tiktoken's downloaded vocabulary."""
from __future__ import annotations

import pytest

from molecule.core import tokens


@pytest.fixture
def estimator(monkeypatch):
    monkeypatch.setenv("MOLECULE_TOKENIZER", "estimate")
    tokens.reset_tokenizer()
    yield
    monkeypatch.delenv("MOLECULE_TOKENIZER")
    tokens.reset_tokenizer()


def test_empty_text_is_zero_tokens():
    assert tokens.count_tokens("") == 0


def test_count_is_positive_and_monotonic():
    short = tokens.count_tokens("hello world")
    long = tokens.count_tokens("hello world " * 20)
    assert 0 < short < long


def test_estimator_is_used_when_forced(estimator):
    assert tokens.tokenizer_name() == "estimate"
    assert tokens.count_tokens("hello world") == 2


def test_estimator_counts_punctuation_and_long_words(estimator):
    assert tokens.estimate_tokens("a, b.") == 4
    assert tokens.estimate_tokens("internationalization") >= 3


def test_estimator_is_close_to_real_tokenizer():
    """Within 25% of the exact count on ordinary prose (when tiktoken works)."""
    if tokens.tokenizer_name() == "estimate":
        pytest.skip("tiktoken vocabulary not available")
    text = (
        "Production deploys are started from chat and go out as a canary first: "
        "5% of traffic for 15 minutes while the bot watches error rate and latency."
    ) * 5
    exact = tokens.count_tokens(text)
    assert abs(tokens.estimate_tokens(text) - exact) / exact < 0.25


def test_split_by_tokens_loses_nothing():
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 10
    pieces = tokens.split_by_tokens(text, 12)
    assert "".join(pieces) == text
    assert len(pieces) > 1
    assert all(tokens.count_tokens(p) <= 12 for p in pieces)


def test_split_by_tokens_with_estimator(estimator):
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 10
    pieces = tokens.split_by_tokens(text, 12)
    assert "".join(pieces) == text
    assert all(tokens.count_tokens(p) <= 12 for p in pieces)


def test_split_by_tokens_rejects_nonpositive_limit():
    with pytest.raises(ValueError):
        tokens.split_by_tokens("text", 0)


def test_import_never_touches_the_network():
    """The encoder is resolved lazily: importing the module must not load it."""
    import importlib

    tokens.reset_tokenizer()
    importlib.reload(tokens)
    assert tokens._resolved is False
