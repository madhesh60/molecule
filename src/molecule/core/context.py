"""context.py — the budgeted context compiler behind `mol_context`.

One call replaces the index → search → fetch round-trips: give it a question
and a token budget, get back ready-to-paste context that is guaranteed to
fit, with provenance for every piece.

    rank → gate → pack in rank order → excerpt the one that overflows

  1. **Rank** every chunk with the hybrid scorer (retriever.py).
  2. **Gate.** The budget is a ceiling, not a target: a chunk is only
     eligible if it scores within `context_min_score_ratio` of the best
     one. (Measured: the same answer recall as filling the budget, for
     roughly half the tokens.) When even the best chunk has low absolute
     evidence, only that one best guess is included, clearly flagged.
  3. **Pack** eligible chunks whole, best first. A near-copy of something
     already packed is skipped — it adds no information. A chunk bonded to
     a packed one (bonds.py) is eligible at half the usual bar, so "see
     §7.3" can pull §7.3 in when there is room.
  4. **Excerpt** the first chunk that does not fit, so the remaining room
     is not wasted. Two strategies (`context_excerpt`):
       * "head"  — keep the beginning of the chunk. The default: on the
         held-out benchmark it kept the answer more often than anything
         cleverer, because sections lead with their key fact.
       * "focus" — keep the passages most similar to the query, in document
         order. Better when answers sit deep inside long sections.
     Whatever still does not fit is listed under `omitted`, so the agent
     can fetch it deliberately.

`always_include` guardrail chunks are packed first, unconditionally, and
relevant remembered facts are appended at the end.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from molecule.config import MoleculeConfig
from molecule.core.bonds import neighbors
from molecule.core.embedder import Embedder, embedder_for
from molecule.core.markdown import FenceTracker
from molecule.core.passages import join_passages, passage_embedding_texts, split_passages
from molecule.core.retriever import (
    ChunkScores,
    _chunk_reason,
    relevance_evidence,
    relevance_label,
    score_chunks,
    score_memories,
)
from molecule.core.scoring import tokenize
from molecule.core.tokens import count_tokens, split_by_tokens

_POOL_SIZE = 12                 # how far down the ranking the compiler looks
_MEMORY_BUDGET_SHARE = 0.15     # at most this share of the budget goes to memories
_SEPARATOR = "\n\n"
_SEPARATOR_TOKENS = 2
_MIN_BUDGET = 64
_ELLIPSIS = " …"


@dataclass
class _Item:
    id: str
    kind: str                   # "guardrail" | "chunk" | "memory"
    text: str
    tokens: int
    score: float
    reason: str
    compressed: bool = False
    full_tokens: int = 0

    def public(self) -> dict:
        out = {
            "id": self.id, "type": self.kind, "tokens": self.tokens,
            "score": round(self.score, 4), "reason": self.reason,
        }
        if self.compressed:
            out["compressed"] = True
            out["full_tokens"] = self.full_tokens
        return out


@dataclass
class _Pack:
    budget: int
    used: int = 0
    items: list[_Item] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return self.budget - self.used

    def add(self, item: _Item) -> None:
        self.items.append(item)
        self.used += item.tokens + _SEPARATOR_TOKENS


def _header(chunk_id: str, title: str, excerpt: bool = False, guardrail: bool = False) -> str:
    """Short, readable provenance line: `### file.md › Section title`.

    The full chunk id is long (it repeats every parent heading) and would eat
    a real share of a small budget; it is returned in `items`, in the same
    order, for fetch and feedback calls."""
    source = chunk_id.partition("::")[0]
    tags = [t for t, on in (("guardrail", guardrail), ("excerpt", excerpt)) if on]
    return f"### {source} › {title}" + (f" ({', '.join(tags)})" if tags else "")


def _chunk_row(conn: sqlite3.Connection, chunk_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT chunk_id, heading_path, content, token_count, passage_embeddings, "
        "passage_count FROM chunks WHERE chunk_id=?",
        (chunk_id,),
    ).fetchone()


# --- Excerpts ----------------------------------------------------------------------

def head_excerpt(content: str, max_tokens: int) -> tuple[str, int]:
    """The beginning of a chunk in at most `max_tokens` tokens.

    Cut on a word boundary, and a code fence left open by the cut is closed
    so the excerpt is still valid markdown."""
    reserve = count_tokens(_ELLIPSIS) + 3            # the marker and a possible closing fence
    if max_tokens <= reserve:
        return "", 0
    piece = split_by_tokens(content, max_tokens - reserve)[0]
    cut_mid_word = len(piece) < len(content) and not content[len(piece)].isspace()
    if cut_mid_word and len(piece.split()) > 1:
        piece = piece.rsplit(None, 1)[0]             # do not end in half a word
    piece = piece.rstrip()

    fence = FenceTracker()
    for line in piece.split("\n"):
        fence.feed(line)
    text = piece + ("\n```" if fence.open else "") + _ELLIPSIS
    return text, count_tokens(text)


@lru_cache(maxsize=256)
def _passage_matrix(
    embedder: Embedder, heading: str, content: str, max_passages: int, max_tokens: int
) -> np.ndarray:
    """Vectors for a chunk's passages, embedded on demand and memoised —
    only chunks that actually need a focused excerpt ever pay for this."""
    passages = split_passages(content, max_passages)
    return embedder.encode(passage_embedding_texts(heading, passages), max_tokens)


def focus_excerpt(
    content: str,
    heading: str,
    query: str,
    query_vector: np.ndarray,
    max_tokens: int,
    config: MoleculeConfig,
    passage_blob: bytes | None = None,
    passage_count: int = 0,
) -> tuple[str, int]:
    """Query-focused extract of a chunk in at most `max_tokens` tokens.

    Passages are scored by similarity to the query plus the share of query
    terms they contain; the best ones are kept, in their original order,
    with gaps marked. Passage vectors come from the store when the deep
    index is on, otherwise they are embedded here once and cached."""
    passages = split_passages(content, config.max_passages_per_chunk)
    if not passages:
        return "", 0

    dim = query_vector.shape[0]
    if passage_blob is not None and passage_count == len(passages) and dim and (
        len(passage_blob) == passage_count * dim * 4
    ):
        matrix = np.frombuffer(passage_blob, dtype=np.float32).reshape(passage_count, dim)
    else:
        matrix = _passage_matrix(
            embedder_for(config), heading, content,
            config.max_passages_per_chunk, config.passage_embed_tokens,
        )
    dense = (
        matrix @ query_vector if matrix.shape == (len(passages), dim)
        else np.zeros(len(passages), dtype=np.float32)
    )

    query_terms = set(tokenize(query))
    overlap = np.zeros(len(passages), dtype=np.float32)
    if query_terms:
        for i, passage in enumerate(passages):
            overlap[i] = len(query_terms & set(tokenize(passage.text))) / len(query_terms)
    relevance = 0.7 * dense + 0.3 * overlap
    relevance[0] += 0.05   # the lead sentence usually says what the section is about

    costs = [count_tokens(p.text) + 2 for p in passages]
    selected: list[int] = []
    used = 0
    for i in np.argsort(-relevance, kind="stable"):
        i = int(i)
        header_cost = (
            count_tokens(passages[i].header) + 1
            if passages[i].header and not any(passages[j].block == passages[i].block
                                              for j in selected)
            else 0
        )
        if used + costs[i] + header_cost <= max_tokens:
            selected.append(i)
            used += costs[i] + header_cost
    if not selected:
        return head_excerpt(passages[int(np.argmax(relevance))].text, max_tokens)

    text = join_passages(passages, selected)
    return text, count_tokens(text)


def _render_chunk(
    conn: sqlite3.Connection,
    scores: ChunkScores,
    row: int,
    query: str,
    config: MoleculeConfig,
    room: int,
    allow_excerpt: bool,
    guardrail: bool = False,
    extra_reason: str = "",
) -> _Item | None:
    """A chunk as a context item that fits in `room` tokens, or None."""
    chunk_id = scores.corpus.ids[row]
    stored = _chunk_row(conn, chunk_id)
    if stored is None:
        return None
    reason = _chunk_reason(scores, row, config) + extra_reason
    score = float(scores.final[row])
    kind = "guardrail" if guardrail else "chunk"
    title = json.loads(stored["heading_path"])[-1]

    whole = f"{_header(chunk_id, title, guardrail=guardrail)}\n{stored['content']}"
    whole_tokens = count_tokens(whole)
    if whole_tokens <= room:
        return _Item(chunk_id, kind, whole, whole_tokens, score, reason)

    header = _header(chunk_id, title, excerpt=True, guardrail=guardrail)
    body_room = room - count_tokens(header) - 2
    # An excerpt must be worth reading: at least `context_min_excerpt_tokens`,
    # or 40% of a chunk that is itself small.
    full_tokens = int(stored["token_count"])
    smallest_useful = min(config.context_min_excerpt_tokens, max(16, int(0.4 * full_tokens)))
    if not allow_excerpt or body_room < smallest_useful:
        return None
    if config.context_excerpt == "focus":
        body, _ = focus_excerpt(
            stored["content"], title, query, scores.query_vector, body_room, config,
            stored["passage_embeddings"], stored["passage_count"] or 0,
        )
    else:
        body, _ = head_excerpt(stored["content"], body_room)
    if not body:
        return None
    text = f"{header}\n{body}"
    tokens = count_tokens(text)
    if tokens > room:
        return None
    return _Item(chunk_id, kind, text, tokens, score, reason,
                 compressed=True, full_tokens=full_tokens)


def _near_duplicate_of(
    scores: ChunkScores, row: int, packed: list[int], threshold: float
) -> int | None:
    """The already-packed chunk that `row` is a near-copy of, if any."""
    matrix = scores.corpus.matrix
    if not packed or not matrix.size:
        return None
    sims = matrix[packed] @ matrix[row]
    best = int(np.argmax(sims))
    return packed[best] if float(sims[best]) >= threshold else None


def _memory_items(
    query: str, conn: sqlite3.Connection, config: MoleculeConfig, budget: int
) -> list[_Item]:
    """Remembered facts that bear on the query — cheap, capped, never 'low'."""
    items: list[_Item] = []
    room = int(budget * _MEMORY_BUDGET_SHARE)
    for memory in score_memories(query, conn, config)[: config.context_max_memories]:
        if memory["relevance"] == "low":
            continue
        line = f"- {memory['text']}"
        tokens = count_tokens(line) + 1
        if tokens > room:
            continue
        room -= tokens
        items.append(_Item(memory["memory_id"], "memory", line, tokens,
                           memory["score"], memory["reason"]))
    return items


def _assemble(pack: _Pack, memories: list[_Item]) -> str:
    blocks = [item.text for item in pack.items]
    if memories:
        blocks.append("Remembered facts:\n" + "\n".join(m.text for m in memories))
    return _SEPARATOR.join(blocks)


def build_context(
    query: str,
    conn: sqlite3.Connection,
    config: MoleculeConfig,
    budget: int | None = None,
    compress: bool = True,
    include_memories: bool = True,
) -> dict:
    """Compile the best context for `query` that fits in `budget` tokens.

    `compress=False` only ever includes whole chunks (no excerpts)."""
    budget = max(_MIN_BUDGET, int(budget or config.context_budget))
    scores = score_chunks(query, conn, config)
    corpus = scores.corpus
    order = [int(r) for r in scores.order()]
    pack = _Pack(budget)
    omitted: list[dict] = []
    packed: list[int] = []

    def skip(row: int, why: str) -> None:
        omitted.append({"id": corpus.ids[row], "tokens": int(corpus.token_counts[row]),
                        "why": why})

    def try_pack(row: int, note: str = "", guardrail: bool = False) -> bool:
        item = _render_chunk(conn, scores, row, query, config, pack.remaining,
                             allow_excerpt=compress or guardrail, guardrail=guardrail,
                             extra_reason=note)
        if item is None:
            return False
        pack.add(item)
        packed.append(row)
        return True

    # 1. Guardrails: unconditional, first.
    for row in np.flatnonzero(corpus.always_include):
        if not try_pack(int(row), guardrail=True):
            skip(int(row), "guardrail does not fit the budget")

    # 2. Gate: which chunks have earned a place?
    confidence = "low"
    eligible: list[int] = []
    top_score = 0.0
    if order:
        top = order[0]
        top_score = float(scores.final[top])
        confidence = relevance_label(
            relevance_evidence(scores.dense[top], scores.keyword[top]), config
        )
        if confidence == "low":
            eligible = [top]                 # one flagged best guess, nothing more
        else:
            bar = config.context_min_score_ratio * top_score
            eligible = [r for r in order[:_POOL_SIZE] if scores.final[r] >= bar]
    eligible = [r for r in eligible if r not in packed]

    # Room for remembered facts is set aside before chunks are packed.
    memories = (
        _memory_items(query, conn, config, budget)
        if include_memories and config.context_max_memories > 0 else []
    )
    if memories:
        pack.used += sum(m.tokens for m in memories) + count_tokens("Remembered facts:") + 4

    # 3–4. Pack in rank order; the first chunk that overflows is excerpted.
    for row in eligible:
        twin = _near_duplicate_of(scores, row, packed, config.dedup_threshold)
        if twin is not None:
            skip(row, f"near-duplicate of {corpus.ids[twin]}")
        elif not try_pack(row):
            skip(row, "over budget")

    # A packed chunk vouches for the chunks it is bonded to: they get in at
    # half the usual bar, strongest bond first, while room remains.
    if confidence != "low":
        bonded = neighbors(conn, config)
        half_bar = 0.5 * config.context_min_score_ratio * top_score
        seen = set(packed) | set(eligible)
        extras: list[tuple[float, int, str]] = []
        for row in list(packed):
            for other_id, strength, kind in bonded.get(corpus.ids[row], []):
                other = corpus.row_of.get(other_id)
                if other is None or other in seen or scores.final[other] < half_bar:
                    continue
                seen.add(other)
                extras.append((strength, other, f", bonded to {corpus.ids[row]} ({kind})"))
        for _, row, note in sorted(extras, key=lambda e: -e[0]):
            if _near_duplicate_of(scores, row, packed, config.dedup_threshold) is None:
                if not try_pack(row, note):
                    skip(row, "over budget")

    context = _assemble(pack, memories)
    tokens_used = count_tokens(context)
    # Token counts of joined text can differ slightly from the sum of parts;
    # the budget is a hard promise, so trim from the end until it holds.
    while tokens_used > budget and pack.items:
        dropped = pack.items.pop()
        omitted.append({"id": dropped.id, "tokens": dropped.full_tokens or dropped.tokens,
                        "why": "over budget"})
        context = _assemble(pack, memories)
        tokens_used = count_tokens(context)

    chunk_ids = [item.id for item in pack.items]
    result: dict = {
        "query": query,
        "context": context,
        "confidence": confidence,
        "tokens_used": tokens_used,
        "budget": budget,
        "items": [item.public() for item in pack.items] + [m.public() for m in memories],
        "omitted": omitted,
        "sources": sorted({cid.split("::")[0] for cid in chunk_ids}),
    }
    if confidence == "low" and order:
        result["note"] = (
            "Low confidence: no section clearly covers this query. The single closest "
            "match is included as a best guess; treat it with care."
        )
    return result
