"""labeler.py — one-line labels ("table of contents" entries) for chunks.

Labels are cached by content_hash so unchanged sections are never
re-summarized (matters when llm_fn is a real, paid LLM). The cache is
version-gated on EMBEDDING_TEXT_VERSION: labels feed the embedding text,
so a format bump regenerates both together on the next ingest.

default_llm_fn needs no LLM: it reads the *structure* of the section —
lead sentence for prose, first-column entries for tables, defined names
for code, bolded phrases anywhere — because that structure is what a
human skims to decide "is this the section I need?".
"""
from __future__ import annotations

import re
import sqlite3
from typing import Callable

from molecule.core.embed_text import EMBEDDING_TEXT_VERSION
from molecule.db import content_hash

_MAX_LABEL_CHARS = 250

_SENTENCE_END = re.compile(r"(?<=[.!?])\s")
_BOLD_RE = re.compile(r"\*\*([^*\n]+)\*\*")
_DEF_RE = re.compile(r"^\s*(?:def|class)\s+(\w+)", re.MULTILINE)
_TABLE_CELL_RE = re.compile(r"^\s*\|\s*([^|\n]+?)\s*\|", re.MULTILINE)
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+(.+)$", re.MULTILINE)
_MARKUP_RE = re.compile(r"[*_`>\[\]]")
_TABLE_DDL_RE = re.compile(r"CREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+(\w+)", re.IGNORECASE)


def _clean(text: str) -> str:
    return _MARKUP_RE.sub("", text).strip()


def _lead_sentence(text: str) -> str:
    """First real prose sentence — skips markup lines AND fenced code blocks."""
    in_fence = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not s or s.startswith(("|", "#", "<!--", "---")):
            continue
        s = _LIST_ITEM_RE.match(s).group(1) if _LIST_ITEM_RE.match(s) else s
        return _clean(_SENTENCE_END.split(s)[0])
    return ""


def _summarize(text: str) -> str:
    parts: list[str] = []
    lead = _lead_sentence(text)
    if lead:
        parts.append(lead)

    # Tables: the first column is what each row is *about* — list it.
    cells = [
        _clean(c) for c in _TABLE_CELL_RE.findall(text)
        if not set(c) <= set("-: ")  # skip |---|---| separator rows
    ]
    if len(cells) >= 2:
        parts.append("Covers: " + "; ".join(cells[:8]))

    # Code: the defined names are the interface — list them.
    names = list(dict.fromkeys(_DEF_RE.findall(text)))
    if names:
        parts.append("Defines functions: " + ", ".join(names))

    # SQL DDL: table names say what the schema stores.
    tables = list(dict.fromkeys(_TABLE_DDL_RE.findall(text)))
    if tables:
        parts.append("Defines tables: " + ", ".join(tables))

    # Bold phrases: authors bold what matters.
    bolds = list(dict.fromkeys(_clean(b) for b in _BOLD_RE.findall(text)))
    if bolds:
        parts.append("Key: " + "; ".join(bolds)[:120])

    # Lists: first items carry the section's main claims.
    if len(parts) < 2:
        items = [_clean(i) for i in _LIST_ITEM_RE.findall(text)[:3]]
        if items:
            parts.append("Includes: " + "; ".join(_SENTENCE_END.split(i)[0] for i in items))

    label = " ".join(dict.fromkeys(p for p in parts if p)).replace("\n", " ")
    return label[:_MAX_LABEL_CHARS] or _clean(text)[:150]


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
            "SELECT summary FROM chunks "
            "WHERE content_hash = ? AND embedding_text_version >= ?",
            (h, EMBEDDING_TEXT_VERSION),
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
    """Structure-aware fallback labels — no LLM call required."""
    return [_summarize(t) for t in texts]
