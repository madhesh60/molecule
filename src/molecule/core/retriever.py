"""retriever.py — router index, hybrid ranking, chunk fetch.

Ranking formula (every term is explained in each result's `reason`):

    dense  = (1 − p)·cos(query, chunk) + p·max cos(query, passage)
    score  = (w_e·dense + w_k·coverage_bm25 + w_h·heading_match) × learned

  * `dense` blends the whole-chunk vector with the chunk's best sentence —
    a question usually matches one sentence, not the section's average.
  * `coverage_bm25` is the share of the query's rare terms the chunk
    explains (scoring.py). Strong for exact wording, faint for paraphrases.
  * `learned` is the query-aware feedback weight (feedback.py), 0.5–1.5.

Hot path: everything that does not depend on the query — vectors, passage
vectors, the inverted index, vote history — is built once and memoised under
generation counters (cache.py), so a search is one query embedding, two
matrix products and a few array operations.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

import numpy as np

from molecule.config import MoleculeConfig
from molecule.core import feedback as feedback_mod
from molecule.core.bonds import related_for
from molecule.core.cache import versioned
from molecule.core.embed_text import build_embedding_text
from molecule.core.embedder import embed_query, embedder_for, vector_is_current
from molecule.core.passages import passage_embedding_texts, split_passages
from molecule.core.scoring import (
    LexicalIndex,
    build_lexical_index,
    heading_scores,
    lexical_scores,
    query_terms,
)
from molecule.core.tokens import count_tokens
from molecule.db import utc_now

_MAX_K = 100
_SUMMARY_CHARS = 160   # search results carry a short label; the body is one fetch away
_ACTIVE_MEMORY = "status='active' AND (expires_at IS NULL OR expires_at > datetime('now'))"


# --- Router index (the table of contents) ----------------------------------------

_BODY_KEY = "(body)"   # a heading that has its own text *and* sub-sections


def _toc(rows: list[sqlite3.Row], depth: int | None) -> list[dict]:
    """One nested table of contents per source file.

    A heading maps to its token count (a chunk) or to its sub-headings. Each
    heading is written once, so this is far cheaper than a flat list of ids
    that repeat the same parents — and it still tells the agent every chunk
    id: `<source>::<heading>:<sub-heading>`. Below `depth`, a sub-tree is
    collapsed to its chunk and token totals."""
    trees: dict[str, dict] = {}
    for r in rows:
        source = r["chunk_id"].partition("::")[0]
        path = json.loads(r["heading_path"])
        visible = path if depth is None else path[:depth]
        folded = len(path) > len(visible)      # the chunk lies below the visible depth
        node = trees.setdefault(source, {})
        for heading in (visible if folded else visible[:-1]):
            child = node.get(heading)
            if not isinstance(child, dict):
                child = {} if child is None else {_BODY_KEY: child}
                node[heading] = child
            node = child
        if folded:
            rest = node.setdefault("…", {"chunks": 0, "tokens": 0})
            rest["chunks"] += 1
            rest["tokens"] += r["token_count"]
        elif isinstance(node.get(visible[-1]), dict):
            node[visible[-1]][_BODY_KEY] = r["token_count"]
        else:
            node[visible[-1]] = r["token_count"]

    entries = [{"source": source, "toc": tree} for source, tree in trees.items()]
    if entries:
        entries[0]["chunk_id_format"] = "<source>::<heading>:<sub-heading>"
    return entries


def _index_levels(rows: list[sqlite3.Row]) -> list[list[dict]]:
    """The same index at decreasing levels of detail, most detailed first."""
    full = [
        {
            "chunk_id": r["chunk_id"],
            "heading_path": json.loads(r["heading_path"]),
            "summary": r["summary"] or "",
            "token_count": r["token_count"],
            "always_include": bool(r["always_include"]),
            "weight": r["weight"],
        }
        for r in rows
    ]

    compact = []
    for r in rows:
        entry: dict = {"chunk_id": r["chunk_id"], "token_count": r["token_count"]}
        if r["summary"]:
            entry["summary"] = r["summary"][:120]
        if r["always_include"]:
            entry["always_include"] = True
        compact.append(entry)

    totals: dict[str, dict] = {}
    for r in rows:
        source = r["chunk_id"].partition("::")[0]
        total = totals.setdefault(
            source, {"source": source, "prefix": f"{source}::", "chunks": 0, "token_count": 0}
        )
        total["chunks"] += 1
        total["token_count"] += r["token_count"]

    return [full, compact, _toc(rows, None), _toc(rows, 2), _toc(rows, 1),
            list(totals.values())]


def load_router_index(
    conn: sqlite3.Connection, max_tokens: int | None = None, prefix: str | None = None
) -> list[dict]:
    """The cheap table of contents: ids, summaries, token counts — never bodies.

    With `max_tokens`, the index is returned at the most detailed level that
    fits the budget: full summaries → short summaries → a nested table of
    contents (headings + token counts) → a shallower one → per-file totals.
    Pass `prefix` to drill into one file or heading. It therefore stays cheap
    on a 50-chunk file and on a 5,000-chunk docs folder alike.
    """
    sql = (
        "SELECT chunk_id, heading_path, summary, token_count, always_include, weight "
        "FROM chunks"
    )
    args: tuple = ()
    if prefix:
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        sql += " WHERE chunk_id LIKE ? ESCAPE '\\'"
        args = (escaped + "%",)
    rows = conn.execute(sql + " ORDER BY source_file, position, rowid", args).fetchall()
    levels = _index_levels(rows)
    if not max_tokens or max_tokens <= 0:
        return levels[0]
    for level in levels:
        if count_tokens(json.dumps(level, ensure_ascii=False)) <= max_tokens:
            return level
    # Even the per-file outline is too big: keep as many groups as fit.
    coarse = levels[-1]
    while coarse and count_tokens(json.dumps(coarse, ensure_ascii=False)) > max_tokens:
        coarse = coarse[: max(1, len(coarse) // 2)] if len(coarse) > 1 else []
    return coarse


# --- Query-independent corpus state (cached) ----------------------------------------

@dataclass
class ChunkCorpus:
    ids: list[str]
    row_of: dict[str, int]
    matrix: np.ndarray            # (n, d) chunk vectors; zero rows where none is usable
    passage_matrix: np.ndarray    # (p, d) every passage vector, grouped by chunk
    passage_starts: np.ndarray    # first passage row of each chunk that has passages
    passage_owner: np.ndarray     # chunk row for each entry of passage_starts
    lexical: LexicalIndex
    summaries: list[str]
    token_counts: np.ndarray
    always_include: np.ndarray    # bool
    prior_weight: np.ndarray      # stored learned_weight (used when a chunk has no votes)


def _heal_chunk_vectors(conn: sqlite3.Connection, config: MoleculeConfig) -> None:
    """Re-embed chunks whose vectors came from a different model.

    Runs only when such rows exist (after `embedding_model` was changed).
    Comparing vectors from two models would silently return garbage."""
    embedder = embedder_for(config)
    rows = [
        r for r in conn.execute(
            "SELECT chunk_id, heading_path, summary, content, embedding, embedding_model "
            "FROM chunks"
        )
        if not vector_is_current(r["embedding"], r["embedding_model"], embedder)
    ]
    if not rows:
        return
    texts = [
        build_embedding_text(json.loads(r["heading_path"]), r["summary"] or "", r["content"])
        for r in rows
    ]
    vectors = embedder.encode(texts, config.chunk_embed_tokens)
    for row, vector in zip(rows, vectors):
        blob, count = None, 0
        if config.passage_vectors:
            passages = split_passages(row["content"], config.max_passages_per_chunk)
            if passages:
                heading = json.loads(row["heading_path"])[-1]
                matrix = embedder.encode(
                    passage_embedding_texts(heading, passages), config.passage_embed_tokens
                )
                blob, count = matrix.tobytes(), len(passages)
        conn.execute(
            "UPDATE chunks SET embedding=?, embedding_model=?, passage_embeddings=?, "
            "passage_count=? WHERE chunk_id=?",
            (vector.tobytes(), embedder.name, blob, count, row["chunk_id"]),
        )
    conn.commit()


def _chunk_corpus(conn: sqlite3.Connection, config: MoleculeConfig | None = None) -> ChunkCorpus:
    """Everything about the chunks that does not depend on the query."""
    dim = 0
    if config is not None and conn.execute("SELECT 1 FROM chunks LIMIT 1").fetchone():
        embedder = embedder_for(config)
        stale = conn.execute(
            "SELECT 1 FROM chunks WHERE embedding IS NULL OR length(embedding) != ? OR "
            "(embedding_model IS NOT NULL AND embedding_model != ?) LIMIT 1",
            (embedder.dim * 4, embedder.name),
        ).fetchone()
        if stale:
            _heal_chunk_vectors(conn, config)

    rows = conn.execute(
        "SELECT chunk_id, heading_path, summary, content, token_count, always_include, "
        "learned_weight, embedding, passage_embeddings, passage_count FROM chunks "
        "ORDER BY source_file, position, rowid"
    ).fetchall()
    for r in rows:
        if r["embedding"] is not None:
            dim = len(r["embedding"]) // 4
            break

    ids = [r["chunk_id"] for r in rows]
    matrix = np.zeros((len(rows), dim), dtype=np.float32)
    passage_blobs: list[bytes] = []
    starts: list[int] = []
    owners: list[int] = []
    offset = 0
    docs: dict[str, str] = {}
    headings: dict[str, str] = {}
    for i, r in enumerate(rows):
        if r["embedding"] is not None and len(r["embedding"]) == dim * 4:
            matrix[i] = np.frombuffer(r["embedding"], dtype=np.float32)
        blob, count = r["passage_embeddings"], r["passage_count"] or 0
        if blob is not None and count and len(blob) == count * dim * 4:
            passage_blobs.append(blob)
            starts.append(offset)
            owners.append(i)
            offset += count
        heading_path = json.loads(r["heading_path"])
        heading = " ".join(heading_path)
        headings[r["chunk_id"]] = heading
        # Lexical search covers heading + summary + content — headings are strong signals.
        docs[r["chunk_id"]] = f"{heading} {r['summary'] or ''} {r['content']}"

    passage_matrix = (
        np.frombuffer(b"".join(passage_blobs), dtype=np.float32).reshape(-1, dim)
        if passage_blobs else np.zeros((0, dim), dtype=np.float32)
    )
    return ChunkCorpus(
        ids=ids,
        row_of={cid: i for i, cid in enumerate(ids)},
        matrix=matrix,
        passage_matrix=passage_matrix,
        passage_starts=np.asarray(starts, dtype=np.int64),
        passage_owner=np.asarray(owners, dtype=np.int64),
        lexical=build_lexical_index(docs, headings),
        summaries=[r["summary"] or "" for r in rows],
        token_counts=np.asarray([r["token_count"] for r in rows], dtype=np.int64),
        always_include=np.asarray([bool(r["always_include"]) for r in rows], dtype=bool),
        prior_weight=np.asarray(
            [float(r["learned_weight"] or 1.0) for r in rows], dtype=np.float32
        ),
    )


def _expected_model(config: MoleculeConfig) -> str:
    from molecule.core.embedder import embedder_name

    return embedder_name(config)


@dataclass
class MemoryCorpus:
    ids: list[str]
    matrix: np.ndarray
    lexical: LexicalIndex
    texts: list[str]
    weights: np.ndarray
    confidences: list[float]


def _memory_corpus(conn: sqlite3.Connection, config: MoleculeConfig) -> MemoryCorpus:
    """Active, unexpired memories. Vectors from another model are re-embedded."""
    rows = conn.execute(
        f"SELECT memory_id, text, weight, confidence, embedding, embedding_model "
        f"FROM memories WHERE {_ACTIVE_MEMORY} ORDER BY created_at, rowid"
    ).fetchall()
    vectors: dict[str, bytes] = {}
    width = 0
    if rows:
        embedder = embedder_for(config)
        width = embedder.dim * 4
        stale = [
            r for r in rows
            if not vector_is_current(r["embedding"], r["embedding_model"], embedder)
        ]
        if stale:
            fresh = embedder.encode([r["text"] for r in stale], config.chunk_embed_tokens)
            for row, vector in zip(stale, fresh):
                vectors[row["memory_id"]] = vector.tobytes()
                conn.execute(
                    "UPDATE memories SET embedding=?, embedding_model=? WHERE memory_id=?",
                    (vector.tobytes(), embedder.name, row["memory_id"]),
                )
            conn.commit()
    blobs = [vectors.get(r["memory_id"], r["embedding"]) for r in rows]
    usable = [(r, b) for r, b in zip(rows, blobs) if b is not None and len(b) == width]
    matrix = (
        np.frombuffer(b"".join(b for _, b in usable), dtype=np.float32).reshape(len(usable), -1)
        if usable else np.zeros((0, 0), dtype=np.float32)
    )
    ids = [r["memory_id"] for r, _ in usable]
    texts = [r["text"] for r, _ in usable]
    return MemoryCorpus(
        ids=ids,
        matrix=matrix,
        lexical=build_lexical_index(dict(zip(ids, texts))),
        texts=texts,
        weights=np.asarray([float(r["weight"] or 1.0) for r, _ in usable], dtype=np.float32),
        confidences=[float(r["confidence"]) for r, _ in usable],
    )


def chunk_corpus(conn: sqlite3.Connection, config: MoleculeConfig) -> ChunkCorpus:
    return versioned(
        conn, "chunk_corpus", lambda: _chunk_corpus(conn, config), domains=("chunks",),
        salt=_expected_model(config),
    )


def memory_corpus(conn: sqlite3.Connection, config: MoleculeConfig) -> MemoryCorpus:
    # Memories can expire on their own: re-check the clock every five minutes.
    bucket = int(utc_now().timestamp() // 300)
    return versioned(
        conn, "memory_corpus", lambda: _memory_corpus(conn, config), domains=("memories",),
        salt=(bucket, _expected_model(config)),
    )


# --- Scoring -----------------------------------------------------------------------

@dataclass
class ChunkScores:
    corpus: ChunkCorpus
    query_vector: np.ndarray
    dense: np.ndarray        # blended chunk + best-passage similarity
    keyword: np.ndarray      # coverage-normalised BM25
    heading: np.ndarray      # share of query terms in the heading
    learned: np.ndarray      # query-aware feedback weight
    final: np.ndarray

    def order(self) -> np.ndarray:
        """Chunk rows, best first (stable: ties keep document order)."""
        return np.argsort(-self.final, kind="stable")


def _dense_scores(corpus: ChunkCorpus, query_vector: np.ndarray, passage_weight: float):
    dense = corpus.matrix @ query_vector if corpus.matrix.size else np.zeros(len(corpus.ids))
    dense = dense.astype(np.float32)
    if passage_weight > 0 and len(corpus.passage_starts):
        passage_sims = corpus.passage_matrix @ query_vector
        best = np.maximum.reduceat(passage_sims, corpus.passage_starts)
        owner = corpus.passage_owner
        dense[owner] = (1.0 - passage_weight) * dense[owner] + passage_weight * best
    return dense


def score_chunks(query: str, conn: sqlite3.Connection, config: MoleculeConfig) -> ChunkScores:
    """Score every chunk against a query. The single source of ranking truth —
    search, the context compiler and the eval harness all go through here."""
    corpus = chunk_corpus(conn, config)
    query_vector = embed_query(query, embedder_for(config), config.chunk_embed_tokens)
    terms, aliases = query_terms(query, corpus.lexical)
    n = len(corpus.ids)

    if n and corpus.matrix.shape[1] == query_vector.shape[0]:
        dense = _dense_scores(corpus, query_vector, config.passage_weight)
    else:
        dense = np.zeros(n, dtype=np.float32)
    keyword = lexical_scores(terms, corpus.lexical, config.query_expansion, aliases)
    heading = heading_scores(terms, corpus.lexical, config.query_expansion, aliases)
    learned = feedback_mod.learned_weights(conn, config, corpus.ids, corpus.prior_weight,
                                           query_vector)
    base = (
        config.emb_weight * dense
        + config.keyword_weight * keyword
        + config.heading_boost * heading
    )
    return ChunkScores(corpus, query_vector, dense, keyword, heading, learned, base * learned)


def relevance_evidence(dense: float, keyword: float) -> float:
    """Absolute evidence that a result really answers the query, in ~[0, 1].

    Unlike `score` (only meaningful relative to other results of the same
    query), this is comparable across queries: it is what lets Molecule say
    "nothing here covers that" instead of returning its least-bad guess."""
    return max(0.0, float(dense)) + 0.5 * float(keyword)


def relevance_label(evidence: float, config: MoleculeConfig) -> str:
    if evidence >= config.relevance_high:
        return "high"
    if evidence >= config.relevance_low:
        return "medium"
    return "low"


def _chunk_reason(scores: ChunkScores, row: int, config: MoleculeConfig) -> str:
    reason = f"similarity: {scores.dense[row]:.2f}"
    if scores.keyword[row] > 0:
        reason += f", keyword: {scores.keyword[row]:.2f}"
    if scores.heading[row] > 0:
        reason += f", heading match: {config.heading_boost * scores.heading[row]:.2f}"
    learned = float(scores.learned[row])
    if abs(learned - 1.0) >= 0.005:
        reason += f", learned boost: {learned - 1.0:+.2f}"
    return reason


def _chunk_result(scores: ChunkScores, row: int, config: MoleculeConfig) -> dict:
    corpus = scores.corpus
    evidence = relevance_evidence(scores.dense[row], scores.keyword[row])
    summary = corpus.summaries[row]
    if len(summary) > _SUMMARY_CHARS:
        summary = summary[: _SUMMARY_CHARS - 1].rstrip() + "…"
    return {
        "id": corpus.ids[row],
        "type": "chunk",
        "summary": summary,
        "score": round(float(scores.final[row]), 4),
        "relevance": relevance_label(evidence, config),
        "tokens": int(corpus.token_counts[row]),
        "always_include": bool(corpus.always_include[row]),
        "reason": _chunk_reason(scores, row, config),
    }


def score_memories(
    query: str, conn: sqlite3.Connection, config: MoleculeConfig
) -> list[dict]:
    """Every active memory scored against a query, best first."""
    corpus = memory_corpus(conn, config)
    if not corpus.ids:
        return []
    query_vector = embed_query(query, embedder_for(config), config.chunk_embed_tokens)
    if corpus.matrix.shape[1] == query_vector.shape[0]:
        sims = (corpus.matrix @ query_vector).astype(np.float32)
    else:
        sims = np.zeros(len(corpus.ids), dtype=np.float32)
    terms, aliases = query_terms(query, corpus.lexical)
    keyword = lexical_scores(terms, corpus.lexical, config.query_expansion, aliases)
    final = (config.emb_weight * sims + config.keyword_weight * keyword) * corpus.weights

    results = []
    for row in np.argsort(-final, kind="stable"):
        reason = f"similarity: {sims[row]:.2f}"
        if keyword[row] > 0:
            reason += f", keyword: {keyword[row]:.2f}"
        evidence = relevance_evidence(sims[row], keyword[row])
        results.append({
            "memory_id": corpus.ids[row],
            "text": corpus.texts[row],
            "confidence": corpus.confidences[row],
            "score": round(float(final[row]), 4),
            "relevance": relevance_label(evidence, config),
            "reason": reason,
        })
    return results


def clamp_k(k: int | None, config: MoleculeConfig) -> int:
    """Guard nonsense k from tool callers: 0/negative → default, huge → cap."""
    if not k or k <= 0:
        k = config.default_k
    return min(int(k), _MAX_K)


def rank_candidates(
    query: str,
    conn: sqlite3.Connection,
    k: int,
    config: MoleculeConfig,
) -> list[dict]:
    """Top-k chunks and memories for a query: ids + summaries + why, no bodies.

    Ranked purely on merit. `always_include` guardrail chunks that did not
    make the top-k are appended afterwards — never dropped, never dominant.
    """
    k = clamp_k(k, config)
    scores = score_chunks(query, conn, config)
    order = scores.order()

    results = [_chunk_result(scores, int(row), config) for row in order[:k]]
    for memory in score_memories(query, conn, config)[:k]:
        results.append({
            "id": memory["memory_id"],
            "type": "memory",
            "summary": memory["text"],
            "score": memory["score"],
            "relevance": memory["relevance"],
            "always_include": False,
            "reason": memory["reason"],
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    top = results[:k]

    seen = {r["id"] for r in top}
    for result in top:
        if result["type"] == "chunk":
            related = related_for(conn, config, result["id"], exclude=seen)
            if related:
                result["related"] = related

    guardrails = [
        _chunk_result(scores, int(row), config)
        for row in np.flatnonzero(scores.corpus.always_include)
        if scores.corpus.ids[int(row)] not in seen
    ]
    return top + guardrails


def fetch_chunks(chunk_ids: list[str], conn: sqlite3.Connection) -> list[dict]:
    """Fetch full bodies, preserving the caller's requested order.

    Unknown ids are skipped (a renamed heading changes the chunk_id); callers
    can diff returned chunk_ids against what they asked for."""
    if not chunk_ids:
        return []
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"SELECT chunk_id, heading_path, content, token_count FROM chunks "
        f"WHERE chunk_id IN ({placeholders})",
        chunk_ids,
    ).fetchall()
    by_id = {
        r["chunk_id"]: {
            "chunk_id": r["chunk_id"],
            "heading_path": json.loads(r["heading_path"]),
            "content": r["content"],
            "token_count": r["token_count"],
        }
        for r in rows
    }
    # dict.fromkeys dedups while keeping first-seen order.
    return [by_id[cid] for cid in dict.fromkeys(chunk_ids) if cid in by_id]
