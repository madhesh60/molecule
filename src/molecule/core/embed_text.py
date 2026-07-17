"""embed_text.py — build the text that gets embedded for each chunk.

Embedding heading path + summary + content together (instead of raw content
only) gives the router strong signals for queries phrased like section titles.
Bump EMBEDDING_TEXT_VERSION whenever this format changes so existing DBs
re-embed on the next ingest.
"""
from __future__ import annotations

EMBEDDING_TEXT_VERSION = 2


def build_embedding_text(heading_path: list[str], summary: str, content: str) -> str:
    head = " > ".join(heading_path)
    return "\n".join(part for part in (head, summary, content) if part)
