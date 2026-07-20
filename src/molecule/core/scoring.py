"""scoring.py — shared lexical scoring: tokenizer + hand-rolled BM25-lite.

Kept dependency-free on purpose (no rank_bm25): the corpus is tiny (<10k rows),
so an in-memory pass per query is fast and keeps the project lightweight.
Scores are normalized per-query to [0, 1] so they share a band with cosine
similarity and the config weights stay meaningful.

For the hot search path, `build_bm25_index()` precomputes per-doc term counts
once (cached across queries by the retriever); `bm25_scores_indexed()` then
scores a query in O(docs × query_terms) dict lookups. `bm25_scores()` keeps
the original build-per-call signature for one-off callers.

Query terms come in two tiers: the user's exact tokens at full weight, and
their concept-cluster siblings (see expansion.py) at a reduced weight — so
"stop it saving my API key" still reaches the section that says "secrets"
and "redaction", but never outranks an exact match.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from molecule.core.expansion import expand_terms

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Weight for expanded (cluster-sibling) terms relative to exact query terms.
# Measured on the AGENTS.md eval sets: main 20/20 unchanged at any value here;
# adversarial paraphrase set goes 8/12 -> 12/12 at 0.7 (BM25) / 0.5 (heading).
BM25_EXPANSION_WEIGHT = 0.7
HEADING_EXPANSION_WEIGHT = 0.5


def tokenize(text: str) -> list[str]:
    return [w for w in _TOKEN_RE.findall(text.lower()) if len(w) > 2]


@dataclass
class Bm25Index:
    counts: dict[str, Counter]   # doc id -> term counts
    lengths: dict[str, int]      # doc id -> token count (min 1)
    avgdl: float
    n: int
    doc_freq: Counter            # term -> number of docs containing it


def build_bm25_index(docs: dict[str, str]) -> Bm25Index:
    """Precompute everything about `docs` that doesn't depend on the query."""
    counts = {i: Counter(tokenize(t)) for i, t in docs.items()}
    lengths = {i: max(1, sum(c.values())) for i, c in counts.items()}
    n = len(counts)
    avgdl = (sum(lengths.values()) / n) if n else 1.0
    doc_freq = Counter(t for c in counts.values() for t in c.keys())
    return Bm25Index(counts=counts, lengths=lengths, avgdl=avgdl, n=n, doc_freq=doc_freq)


def bm25_scores_indexed(
    query: str,
    index: Bm25Index,
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, float]:
    """BM25 against a prebuilt index, max-normalized to [0, 1].

    Exact query terms score at full weight; concept-cluster siblings
    (expansion.py) score at BM25_EXPANSION_WEIGHT so paraphrased queries
    still land while exact wording always dominates."""
    q_terms = set(tokenize(query))
    if not q_terms or not index.n:
        return {i: 0.0 for i in index.counts}

    exp_terms = expand_terms(q_terms)
    tiers: tuple[tuple[set[str], float], ...] = (
        (q_terms, 1.0),
        (exp_terms, BM25_EXPANSION_WEIGHT),
    )

    idf = {}
    for t in q_terms | exp_terms:
        df = index.doc_freq.get(t, 0)
        idf[t] = math.log(1 + (index.n - df + 0.5) / (df + 0.5))

    raw: dict[str, float] = {}
    max_score = 0.0
    for i, counts in index.counts.items():
        dl = index.lengths[i]
        score = 0.0
        for terms, tier_w in tiers:
            for t in terms:
                f = counts.get(t, 0)
                if f:
                    score += (
                        tier_w * idf[t] * (f * (k1 + 1))
                        / (f + k1 * (1 - b + b * dl / index.avgdl))
                    )
        raw[i] = score
        max_score = max(max_score, score)

    if max_score == 0.0:
        return {i: 0.0 for i in index.counts}
    return {i: s / max_score for i, s in raw.items()}


def bm25_scores(
    query: str,
    docs: dict[str, str],
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, float]:
    """BM25 over `docs` (id -> searchable text), max-normalized to [0, 1]."""
    return bm25_scores_indexed(query, build_bm25_index(docs), k1, b)


def heading_match_score(query: str, heading_path: list[str]) -> float:
    """Fraction of query tokens that appear in the heading path, in [0, 1]."""
    return heading_match_score_tokens(set(tokenize(query)), set(tokenize(" ".join(heading_path))))


def heading_match_score_tokens(q_tokens: set[str], head_tokens: set[str]) -> float:
    """Heading match on pretokenized sets (hot path).

    Cluster-sibling hits count at HEADING_EXPANSION_WEIGHT; capped at 1.0 so
    the band still matches cosine/BM25."""
    if not q_tokens:
        return 0.0
    direct = len(q_tokens & head_tokens)
    expanded = len(expand_terms(q_tokens) & head_tokens)
    return min(1.0, (direct + HEADING_EXPANSION_WEIGHT * expanded) / len(q_tokens))
