from __future__ import annotations

import os
import sqlite3
from functools import lru_cache

import numpy as np

from molecule.db import content_hash as _hash

# Wordpiece truncation for embedding. MiniLM's native limit is 256; 96 cuts
# encode time ~2.4x with no precision loss on either eval set (heading +
# summary lead the embedding text, and BM25 still scores the full content —
# measured 20/20 main, 8/12 adversarial at 96 vs 7/12 at 256).
# Override with MOLECULE_EMBED_SEQ (0 disables truncation).
_MAX_SEQ_TOKENS = int(os.environ.get("MOLECULE_EMBED_SEQ", "96"))
_BATCH_SIZE = 128


@lru_cache(maxsize=1)
def _get_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    if _MAX_SEQ_TOKENS:
        model.max_seq_length = min(model.max_seq_length, _MAX_SEQ_TOKENS)
    # torch defaults to 4 intra-op threads here; on batch encode, using all
    # but two cores is measurably faster (4.9s -> 4.5s on a 227-chunk doc)
    # while leaving headroom for the host agent process.
    try:
        import torch
        torch.set_num_threads(max(4, (os.cpu_count() or 4) - 2))
    except Exception:
        pass
    return model


def get_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load (and cache) the embedding model — call once to warm the cache."""
    return _get_model(model_name)


def embed(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    model = _get_model(model_name)
    return model.encode(
        texts, batch_size=_BATCH_SIZE, convert_to_numpy=True, normalize_embeddings=True
    ).astype(np.float32)


@lru_cache(maxsize=256)
def _embed_query_cached(text: str, model_name: str) -> bytes:
    # bytes are hashable/immutable — safe to share from an lru_cache.
    return embed([text], model_name)[0].tobytes()


def embed_query(text: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Embed a single query string with an LRU cache.

    Model inference dominates search latency (~55ms of ~100ms); agents often
    retry or paraphrase-loop the same query, so repeats become near-free.
    Returns a fresh array view over cached bytes (read-only)."""
    buf = _embed_query_cached(text, model_name)
    vec = np.frombuffer(buf, dtype=np.float32)
    return vec


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    # Both are already L2-normalized from encode(normalize_embeddings=True)
    return (matrix @ query).astype(np.float32)


def embed_and_store(
    conn: sqlite3.Connection,
    table: str,
    id_col: str,
    row_id: str,
    text: str,
    model_name: str,
) -> None:
    h = _hash(text)
    existing = conn.execute(
        f"SELECT content_hash, embedding FROM {table} WHERE {id_col} = ?", (row_id,)
    ).fetchone()
    # Skip only when hash matches AND embedding already exists
    if existing and existing["content_hash"] == h and existing["embedding"] is not None:
        return
    vec = embed([text], model_name)[0]
    conn.execute(
        f"UPDATE {table} SET embedding = ?, content_hash = ? WHERE {id_col} = ?",
        (vec.tobytes(), h, row_id),
    )
    conn.commit()


def load_embeddings(
    conn: sqlite3.Connection, table: str, id_col: str, filter_clause: str = ""
) -> tuple[list[str], np.ndarray]:
    sql = f"SELECT {id_col}, embedding FROM {table} WHERE embedding IS NOT NULL"
    if filter_clause:
        sql += f" AND {filter_clause}"
    rows = conn.execute(sql).fetchall()
    if not rows:
        return [], np.empty((0,), dtype=np.float32)
    ids = [r[0] for r in rows]
    dim = len(np.frombuffer(rows[0][1], dtype=np.float32))
    matrix = np.zeros((len(rows), dim), dtype=np.float32)
    for i, r in enumerate(rows):
        matrix[i] = np.frombuffer(r[1], dtype=np.float32)
    return ids, matrix
