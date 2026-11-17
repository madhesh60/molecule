"""context.py — the budgeted context compiler behind `mol_context`.

One call replaces the index → search → fetch round-trips: give it a question
and a token budget, get back ready-to-paste context that is guaranteed to
fit, with provenance for every piece.

    rank → gate → select (diverse, bonded) → pack → compress if needed

  1. **Rank** every chunk with the hybrid scorer (retriever.py).
  2. **Gate.** The budget is a ceiling, not a target: a chunk is only
     eligible if it scores within `context_min_score_ratio` of the best
     one. If even the best one has low absolute evidence, nothing is
     included — an honest "the docs do not cover this" costs zero tokens.
  3. **Select** greedily by marginal value: relevance discounted by
     similarity to what is already selected (maximal marginal relevance),
     so two near-duplicate sections do not both eat the budget. A chunk
     bonded to a selected one (bonds.py) gets a lower bar — "see §7.3"
     pulls §7.3 in when there is room.
  4. **Pack** whole chunks while they fit.
  5. **Compress** the first chunk that does not fit: keep only its passages
     most relevant to the query (stored passage vectors + term overlap),
     in document order, with gaps marked. Whatever still does not fit is
     listed under `omitted` so the agent can fetch it deliberately.

`always_include` guardrail chunks are packed first, unconditionally.
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

_POOL_SIZE = 12
_MEMORY_BUDGET_SHARE = 0.15
_SEPARATOR = "\n\n"
_SEPARATOR_TOKENS = 2
_MIN_BUDGET = 64
_PASSAGE_KEEP_RATIO = 0.5   # passages below this share of the best one are dropped


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


def _header(chunk_id: str, excerpt: bool = False, guardrail: bool = False) -> str:
    tags = [t for t, on in (("guardrail", guardrail), ("excerpt", excerpt)) if on]
    return f"### {chunk_id}" + (f" ({', '.join(tags)})" if tags else "")


def _chunk_row(conn: sqlite3.Connection, chunk_id: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT chunk_id, heading_path, content, token_count, passage_embeddings, "
        "passage_count FROM chunks WHERE chunk_id=?",
        (chunk_id,),
    ).fetchone()


@lru_cache(maxsize=256)
def _passage_matrix(
    embedder: Embedder, heading: str, content: str, max_passages: int, max_tokens: int
) -> np.ndarray:
    """Vectors for a chunk's passages, embedded on demand and memoised —
    only chunks that actually need compressing ever pay for this."""
    passages = split_passages(content, max_passages)
    return embedder.encode(passage_embedding_texts(heading, passages), max_tokens)


def compress_chunk(
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
    terms they contain; the best ones are kept, in their original order.
    Passage vectors come from the store when the deep index is enabled,
    otherwise they are embedded here once and cached."""
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
    # Room is a ceiling here too: a passage far less relevant than the best
    # one is left out even when it would fit.
    floor = _PASSAGE_KEEP_RATIO * float(relevance.max())
    for i in np.argsort(-relevance, kind="stable"):
        i = int(i)
        if relevance[i] < floor:
            break
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
        # Not even the best passage fits: keep its beginning.
        best = int(np.argmax(relevance))
        text = split_by_tokens(passages[best].text, max(8, max_tokens - 2))[0].rstrip() + " …"
        return text, count_tokens(text)

    text = join_passages(passages, selected)
    return text, count_tokens(text)


def _render_chunk(
    conn: sqlite3.Connection,
    scores: ChunkScores,
    row: int,
    query: str,
    config: MoleculeConfig,
    room: int,
    allow_compress: bool,
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

    whole = f"{_header(chunk_id, guardrail=guardrail)}\n{stored['content']}"
    whole_tokens = count_tokens(whole)
    if whole_tokens <= room:
        return _Item(chunk_id, kind, whole, whole_tokens, score, reason)

    header = _header(chunk_id, excerpt=True, guardrail=guardrail)
    body_room = room - count_tokens(header) - 2
    # An excerpt must be worth reading: at least `context_min_excerpt_tokens`,
    # or 40% of a chunk that is itself small.
    full_tokens = int(stored["token_count"])
    smallest_useful = min(config.context_min_excerpt_tokens, max(16, int(0.4 * full_tokens)))
    if not allow_compress or body_room < smallest_useful:
        return None
    body, _ = compress_chunk(
        stored["content"], json.loads(stored["heading_path"])[-1], query,
        scores.query_vector, body_room, config,
        stored["passage_embeddings"], stored["passage_count"] or 0,
    )
    if not body:
        return None
    text = f"{header}\n{body}"
    tokens = count_tokens(text)
    if tokens > room:
        return None
    return _Item(chunk_id, kind, text, tokens, score, reason,
                 compressed=True, full_tokens=full_tokens)


def _pick_next(
    scores: ChunkScores, queue: list[int], selected: list[int],
    bonus: dict[int, float], lam: float,
) -> int:
    """The queued chunk with the highest marginal relevance: its score,
    discounted by how similar it is to anything already selected."""
    matrix = scores.corpus.matrix
    best_row, best_value = queue[0], float("-inf")
    for row in queue:
        value = float(scores.final[row]) * (1.0 + bonus.get(row, 0.0))
        if selected and matrix.size:
            redundancy = float(np.max(matrix[selected] @ matrix[row]))
            value *= 1.0 - lam * max(0.0, redundancy)
        if value > best_value:
            best_row, best_value = row, value
    return best_row


def _near_duplicate_of(
    scores: ChunkScores, row: int, selected: list[int], threshold: float
) -> int | None:
    """The already-selected chunk that `row` is a near-copy of, if any.
    A copy adds no information, so it never gets budget."""
    matrix = scores.corpus.matrix
    if not selected or not matrix.size:
        return None
    sims = matrix[selected] @ matrix[row]
    best = int(np.argmax(sims))
    return selected[best] if float(sims[best]) >= threshold else None


def build_context(
    query: str,
    conn: sqlite3.Connection,
    config: MoleculeConfig,
    budget: int | None = None,
    compress: bool = True,
    include_memories: bool = True,
) -> dict:
    """Compile the best context for `query` that fits in `budget` tokens."""
    budget = max(_MIN_BUDGET, int(budget or config.context_budget))
    scores = score_chunks(query, conn, config)
    corpus = scores.corpus
    order = [int(r) for r in scores.order()]
    pack = _Pack(budget)
    omitted: list[dict] = []
    included: set[int] = set()

    # 1. Guardrails: unconditional, first.
    for row in np.flatnonzero(corpus.always_include):
        row = int(row)
        item = _render_chunk(conn, scores, row, query, config, pack.remaining,
                             allow_compress=True, guardrail=True)
        if item:
            pack.add(item)
            included.add(row)
        else:
            omitted.append({"id": corpus.ids[row], "tokens": int(corpus.token_counts[row]),
                            "why": "guardrail does not fit the budget"})

    # 2. Gate: is there real evidence that anything answers this query?
    confidence = "low"
    eligible: list[int] = []
    if order:
        top = order[0]
        confidence = relevance_label(
            relevance_evidence(scores.dense[top], scores.keyword[top]), config
        )
        if confidence != "low":
            bar = config.context_min_score_ratio * float(scores.final[top])
            eligible = [r for r in order[:_POOL_SIZE] if scores.final[r] >= bar]

    # 3. Remembered facts that bear on the query (cheap, capped).
    memory_items: list[_Item] = []
    if include_memories and config.context_max_memories > 0:
        room = int(budget * _MEMORY_BUDGET_SHARE)
        for memory in score_memories(query, conn, config)[: config.context_max_memories]:
            if memory["relevance"] == "low":
                continue
            line = f"- {memory['text']}"
            tokens = count_tokens(line) + 1
            if tokens > room:
                continue
            room -= tokens
            memory_items.append(_Item(memory["memory_id"], "memory", line, tokens,
                                      memory["score"], memory["reason"]))
    memory_reserve = (
        sum(m.tokens for m in memory_items) + count_tokens("Remembered facts:") + 4
        if memory_items else 0
    )
    pack.used += memory_reserve

    # 4–5. Select by marginal relevance; pack whole, compress the one that overflows.
    bonded = neighbors(conn, config)
    pool = [r for r in eligible if r not in included]
    bonus: dict[int, float] = {}
    bond_note: dict[int, str] = {}
    queue = list(pool)
    half_bar = (
        0.5 * config.context_min_score_ratio * float(scores.final[order[0]]) if order else 0.0
    )
    done: set[int] = set()
    selected: list[int] = []
    while queue:
        row = _pick_next(scores, queue, selected, bonus, config.context_mmr_lambda)
        queue.remove(row)
        done.add(row)
        twin = _near_duplicate_of(scores, row, selected, config.dedup_threshold)
        if twin is not None:
            omitted.append({"id": corpus.ids[row], "tokens": int(corpus.token_counts[row]),
                            "why": f"near-duplicate of {corpus.ids[twin]}"})
            continue
        item = _render_chunk(conn, scores, row, query, config, pack.remaining,
                             allow_compress=compress, extra_reason=bond_note.get(row, ""))
        if item is None:
            omitted.append({"id": corpus.ids[row], "tokens": int(corpus.token_counts[row]),
                            "why": "over budget"})
            continue
        pack.add(item)
        included.add(row)
        selected.append(row)
        # A selected chunk vouches for the chunks it is bonded to.
        for other_id, strength, kind in bonded.get(corpus.ids[row], []):
            other = corpus.row_of.get(other_id)
            if other is None or other in done or other in included or other in queue:
                continue
            if scores.final[other] >= half_bar:
                queue.append(other)
                bonus[other] = strength
                bond_note[other] = f", bonded to {corpus.ids[row]} ({kind})"

    # Assemble: guardrails and chunks in selection order, then memories.
    blocks = [item.text for item in pack.items]
    if memory_items:
        blocks.append("Remembered facts:\n" + "\n".join(m.text for m in memory_items))
    context = _SEPARATOR.join(blocks)
    tokens_used = count_tokens(context)
    # Token counts of joined text can differ slightly from the sum of parts;
    # the budget is a hard promise, so trim from the end until it holds.
    while tokens_used > budget and pack.items:
        dropped = pack.items.pop()
        omitted.append({"id": dropped.id, "tokens": dropped.full_tokens or dropped.tokens,
                        "why": "over budget"})
        blocks = [item.text for item in pack.items]
        if memory_items:
            blocks.append("Remembered facts:\n" + "\n".join(m.text for m in memory_items))
        context = _SEPARATOR.join(blocks)
        tokens_used = count_tokens(context)

    items = [item.public() for item in pack.items] + [m.public() for m in memory_items]
    chunk_ids = [item.id for item in pack.items]
    result: dict = {
        "query": query,
        "context": context,
        "confidence": confidence,
        "tokens_used": tokens_used,
        "budget": budget,
        "items": items,
        "omitted": omitted,
        "sources": sorted({cid.split("::")[0] for cid in chunk_ids}),
    }
    if confidence == "low" and order:
        result["note"] = (
            "No section clearly covers this query. Closest matches (not included): "
            + json.dumps([corpus.ids[r] for r in order[:3]], ensure_ascii=False)
        )
    return result
