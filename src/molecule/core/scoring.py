"""scoring.py — shared lexical scoring: tokenizer + hand-rolled BM25-lite.

Kept dependency-free on purpose (no rank_bm25): the corpus is tiny (<10k rows),
so an in-memory pass per query is fast and keeps the project lightweight.
Scores are normalized per-query to [0, 1] so they share a band with cosine
similarity and the config weights stay meaningful.
"""
from __future__ import annotations

import math
import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [w for w in _TOKEN_RE.findall(text.lower()) if len(w) > 2]


def bm25_scores(
    query: str,
    docs: dict[str, str],
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, float]:
    """BM25 over `docs` (id -> searchable text), max-normalized to [0, 1]."""
    q_terms = set(tokenize(query))
    if not q_terms or not docs:
        return {i: 0.0 for i in docs}

    tokens = {i: tokenize(t) for i, t in docs.items()}
    n = len(tokens)
    avgdl = sum(len(v) for v in tokens.values()) / n
    df = {t: sum(1 for v in tokens.values() if t in v) for t in q_terms}
    idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in q_terms}

    raw: dict[str, float] = {}
    max_score = 0.0
    for i, toks in tokens.items():
        dl = len(toks) or 1
        score = 0.0
        for t in q_terms:
            f = toks.count(t)
            if f:
                score += idf[t] * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        raw[i] = score
        max_score = max(max_score, score)

    if max_score == 0.0:
        return {i: 0.0 for i in docs}
    return {i: s / max_score for i, s in raw.items()}


def heading_match_score(query: str, heading_path: list[str]) -> float:
    """Fraction of query tokens that appear in the heading path, in [0, 1]."""
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return 0.0
    head_tokens = set(tokenize(" ".join(heading_path)))
    return len(q_tokens & head_tokens) / len(q_tokens)
