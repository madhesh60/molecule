"""scoring.py — the lexical half of hybrid retrieval.

A hand-rolled BM25 over an inverted index (numpy arrays, no extra
dependency), with one deliberate change from textbook BM25:

**scores are coverage-normalised, not max-normalised.** The raw score of a
document is divided by the best score the *query* could possibly get, so the
result answers "what share of this query's information does the document
explain?" in [0, 1]. Max-normalisation (score ÷ best document) always hands
the top document a 1.0 — even when it only matched one stray common word —
which made weak keyword matches as loud as strong ones and dragged
paraphrased queries to the wrong section. With coverage, an exact-wording
query gets a strong lexical signal and a paraphrase gets a faint one, so the
embedding signal decides. (Measured on a held-out handbook: top-3 accuracy
on paraphrased questions 54% → 77% from this change plus stemming and
stopwords — see benchmarks/.)

Query terms come in two tiers: the user's own terms at full weight, and
their concept-cluster siblings (expansion.py) at a reduced weight that also
never exceeds the sibling's own rarity. Exact wording always wins.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from molecule.core.expansion import siblings
from molecule.core.normalize import stem, tokenize

__all__ = [
    "BM25_EXPANSION_WEIGHT", "HEADING_EXPANSION_WEIGHT", "LexicalIndex",
    "bm25_scores", "bm25_scores_indexed", "build_bm25_index", "build_lexical_index",
    "heading_match_score", "heading_match_score_tokens", "heading_scores",
    "lexical_scores", "query_terms", "tokenize",
]

# Weight of a cluster-sibling match relative to an exact term match.
BM25_EXPANSION_WEIGHT = 0.7
HEADING_EXPANSION_WEIGHT = 0.5

_K1 = 1.5
_B = 0.75
_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class LexicalIndex:
    """Inverted index over a small corpus (query-independent, cacheable)."""

    ids: list[str]
    postings: dict[str, tuple[np.ndarray, np.ndarray]]   # term -> (doc rows, term freqs)
    length_norm: np.ndarray                               # k1·(1 − b + b·len/avg_len) per doc
    heading_postings: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.ids)

    def idf(self, term: str) -> float:
        """Rarity of a term; a term absent from the corpus is the rarest."""
        entry = self.postings.get(term)
        df = len(entry[0]) if entry is not None else 0
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))


def build_lexical_index(
    docs: Mapping[str, str], headings: Mapping[str, str] | None = None
) -> LexicalIndex:
    """Index `docs` (id -> searchable text) and optionally their headings."""
    ids = list(docs)
    term_rows: dict[str, list[int]] = {}
    term_freqs: dict[str, list[int]] = {}
    lengths = np.ones(len(ids), dtype=np.float32)
    for row, doc_id in enumerate(ids):
        counts = Counter(tokenize(docs[doc_id]))
        lengths[row] = max(1, sum(counts.values()))
        for term, freq in counts.items():
            term_rows.setdefault(term, []).append(row)
            term_freqs.setdefault(term, []).append(freq)

    average = float(lengths.mean()) if len(ids) else 1.0
    heading_rows: dict[str, list[int]] = {}
    if headings:
        for row, doc_id in enumerate(ids):
            for term in set(tokenize(headings.get(doc_id, ""))):
                heading_rows.setdefault(term, []).append(row)

    return LexicalIndex(
        ids=ids,
        postings={
            term: (np.asarray(rows, dtype=np.int32),
                   np.asarray(term_freqs[term], dtype=np.float32))
            for term, rows in term_rows.items()
        },
        length_norm=_K1 * (1.0 - _B + _B * lengths / average),
        heading_postings={t: np.asarray(r, dtype=np.int32) for t, r in heading_rows.items()},
    )


def _unique(terms: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(terms))


Aliases = Mapping[str, Sequence[str]]


def query_terms(query: str, index: LexicalIndex) -> tuple[list[str], dict[str, list[str]]]:
    """Search terms of a query, plus the compounds the corpus itself uses.

    Technical writing flips freely between "roll back" and "rollback",
    "set up" and "setup", "log in" and "login". For every pair of adjacent
    query words whose joined form occurs in the indexed corpus, that joined
    form becomes an *alias* of both words: a document that says "rollback"
    is credited for "roll" and for "back". Only compounds the corpus really
    contains are considered, so this can never add noise.

    Returns (terms, aliases) where aliases maps a term to its compounds."""
    terms = _unique(tokenize(query))
    aliases: dict[str, list[str]] = {}
    words = _WORD_RE.findall(query.lower())
    for first, second in zip(words, words[1:]):
        joined = stem(first + second)
        if joined in terms or not (
            joined in index.postings or joined in index.heading_postings
        ):
            continue
        parts = [t for t in (stem(first), stem(second)) if t in terms]
        if not parts:              # both halves were stopwords: the compound stands alone
            terms.append(joined)
        for part in parts:
            aliases.setdefault(part, []).append(joined)
    return terms, aliases


def _saturation(index: LexicalIndex, rows: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """BM25 term-frequency saturation in [0, 1), length-normalised."""
    return freqs / (freqs + index.length_norm[rows])


def lexical_scores(
    query_terms: Sequence[str],
    index: LexicalIndex,
    expansion: bool = True,
    aliases: Aliases | None = None,
) -> np.ndarray:
    """Coverage-normalised BM25 of every document, in [0, 1].

    For each query term a document gets the best of (a) its saturated
    frequency of the exact term, (b) the same for a compound alias of the
    term, and (c) a discounted saturated frequency of a cluster sibling.
    Terms are weighted by rarity (IDF); the total is divided by the IDF mass
    of the whole query."""
    scores = np.zeros(index.n, dtype=np.float32)
    terms = _unique(query_terms)
    if not terms or not index.n:
        return scores

    in_query = set(terms)
    total_idf = 0.0
    for term in terms:
        term_idf = index.idf(term)
        total_idf += term_idf
        candidates = [(term, 1.0)]
        candidates += [(alias, 1.0) for alias in (aliases or {}).get(term, ())]
        if expansion:
            candidates += [(s, BM25_EXPANSION_WEIGHT) for s in siblings(term) - in_query]

        best: np.ndarray | None = None
        for candidate, weight in candidates:
            entry = index.postings.get(candidate)
            if entry is None:
                continue
            rows, freqs = entry
            # A common stand-in is weak evidence for a rare query term.
            weight *= min(1.0, index.idf(candidate) / term_idf)
            if best is None:
                best = np.zeros(index.n, dtype=np.float32)
            best[rows] = np.maximum(best[rows], weight * _saturation(index, rows, freqs))
        if best is not None:
            scores += term_idf * best

    return scores / total_idf if total_idf > 0 else scores


def heading_scores(
    query_terms: Sequence[str],
    index: LexicalIndex,
    expansion: bool = True,
    aliases: Aliases | None = None,
) -> np.ndarray:
    """Share of the query's terms found in each document's heading, in [0, 1].
    A term matched only through a cluster sibling counts at reduced weight."""
    scores = np.zeros(index.n, dtype=np.float32)
    terms = _unique(query_terms)
    if not terms or not index.heading_postings:
        return scores
    for term in terms:
        direct = np.zeros(index.n, dtype=bool)
        for candidate in (term, *(aliases or {}).get(term, ())):
            rows = index.heading_postings.get(candidate)
            if rows is not None:
                direct[rows] = True
        via_sibling = np.zeros(index.n, dtype=bool)
        if expansion:
            for sibling in siblings(term):
                rows = index.heading_postings.get(sibling)
                if rows is not None:
                    via_sibling[rows] = True
        scores += direct + HEADING_EXPANSION_WEIGHT * (via_sibling & ~direct)
    return np.minimum(1.0, scores / len(terms))


# --- Small convenience wrappers (one-off callers and tests) ---------------------

def build_bm25_index(docs: Mapping[str, str]) -> LexicalIndex:
    """Index `docs` (id -> searchable text). Alias of build_lexical_index."""
    return build_lexical_index(docs)


def bm25_scores_indexed(query: str, index: LexicalIndex) -> dict[str, float]:
    """Coverage-normalised BM25 of `query` against a prebuilt index."""
    terms, aliases = query_terms(query, index)
    scores = lexical_scores(terms, index, aliases=aliases)
    return {doc_id: float(scores[row]) for row, doc_id in enumerate(index.ids)}


def bm25_scores(query: str, docs: Mapping[str, str]) -> dict[str, float]:
    """Coverage-normalised BM25 of `query` over `docs` (id -> text), in [0, 1]."""
    return bm25_scores_indexed(query, build_lexical_index(docs))


def heading_match_score_tokens(q_tokens: set[str], head_tokens: set[str]) -> float:
    """Heading match on pre-tokenised sets: share of query terms present in
    the heading, sibling matches at HEADING_EXPANSION_WEIGHT, capped at 1."""
    if not q_tokens:
        return 0.0
    total = 0.0
    for term in q_tokens:
        if term in head_tokens:
            total += 1.0
        elif siblings(term) & head_tokens:
            total += HEADING_EXPANSION_WEIGHT
    return min(1.0, total / len(q_tokens))


def heading_match_score(query: str, heading_path: list[str]) -> float:
    """Fraction of query terms that appear in the heading path, in [0, 1]."""
    return heading_match_score_tokens(
        set(tokenize(query)), set(tokenize(" ".join(heading_path)))
    )
