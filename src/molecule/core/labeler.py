from __future__ import annotations

import sqlite3
from typing import Callable

from molecule.db import content_hash


def generate_labels(
    chunks: list,  # list[RawSection]
    conn: sqlite3.Connection,
    llm_fn: Callable[[list[str]], list[str]],
) -> list[str]:
    to_generate: list[int] = []
    results: list[str | None] = [None] * len(chunks)

    for i, chunk in enumerate(chunks):
        h = content_hash(chunk.content)
        row = conn.execute(
            "SELECT summary FROM chunks WHERE content_hash = ?", (h,)
        ).fetchone()
        if row and row["summary"]:
            results[i] = row["summary"]
        else:
            to_generate.append(i)

    if to_generate:
        texts = [chunks[i].content[:800] for i in to_generate]
        labels = llm_fn(texts)
        for idx, label in zip(to_generate, labels):
            results[idx] = label

    return [r or "" for r in results]


def default_llm_fn(texts: list[str]) -> list[str]:
    """Fallback: heading path + first sentence (no LLM call required)."""
    summaries = []
    for text in texts:
        first = text.split(".")[0].strip().replace("\n", " ")
        summaries.append(first[:150] if first else text[:150])
    return summaries
