"""embed_text.py — build the text that gets embedded for each chunk.

Embedding heading path + summary + content together (instead of raw content
only) gives the router strong signals for queries phrased like section titles.
Bump EMBEDDING_TEXT_VERSION whenever this format — or how it is encoded —
changes, so existing stores re-embed on the next ingest.

v4: chunk vectors are joined by per-passage vectors (see passages.py) and
the truncation length is configurable (`chunk_embed_tokens`).
"""
from __future__ import annotations

EMBEDDING_TEXT_VERSION = 4


def build_embedding_text(heading_path: list[str], summary: str, content: str) -> str:
    head = " > ".join(heading_path)
    return "\n".join(part for part in (head, summary, content) if part)
