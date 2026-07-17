from __future__ import annotations

import sqlite3
from functools import lru_cache

import numpy as np

from molecule.db import content_hash as _hash


@lru_cache(maxsize=1)
def _get_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def get_model(model_name: str = "all-MiniLM-L6-v2"):
    """Load (and cache) the embedding model — call once to warm the cache."""
    return _get_model(model_name)


def embed(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    model = _get_model(model_name)
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)


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
