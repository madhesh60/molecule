"""passages.py — cut a chunk into its smallest self-contained units.

A *passage* is one sentence, one list item, one table row or one code block.
Passages serve two features with a single deterministic split:

  * **Retrieval** — each passage gets its own vector at ingest. A question
    usually matches one sentence of a section, not its average; scoring a
    chunk by its best passage is what lifts paraphrased queries.
  * **Compression** — `mol_context` keeps only the passages that matter for
    the query when a whole chunk does not fit the token budget. Because the
    split here is the same one used at ingest, the stored passage vectors
    are reused and compression costs no model call.
"""
from __future__ import annotations

from dataclasses import dataclass

from molecule.core.markdown import split_blocks, split_list_items, split_sentences, split_table

_MIN_SENTENCE_CHARS = 30


@dataclass(frozen=True)
class Passage:
    text: str
    kind: str          # "prose" | "list" | "table" | "code"
    block: int         # index of the markdown block it came from
    header: str = ""   # table header to repeat when a row is shown on its own


def _merge_short(sentences: list[str]) -> list[str]:
    """Attach fragments ("See below.") to the sentence before them."""
    merged: list[str] = []
    for sentence in sentences:
        if merged and len(sentence) < _MIN_SENTENCE_CHARS:
            merged[-1] = f"{merged[-1]} {sentence}"
        else:
            merged.append(sentence)
    if len(merged) > 1 and len(merged[0]) < _MIN_SENTENCE_CHARS:
        merged[1] = f"{merged[0]} {merged[1]}"
        merged.pop(0)
    return merged


def _cap(passages: list[Passage], limit: int) -> list[Passage]:
    """Merge the shortest neighbouring prose passages until at most `limit`."""
    passages = list(passages)
    while len(passages) > limit:
        best = -1
        best_len = 0
        for i in range(len(passages) - 1):
            a, b = passages[i], passages[i + 1]
            if a.kind == b.kind == "prose" and a.block == b.block:
                size = len(a.text) + len(b.text)
                if best < 0 or size < best_len:
                    best, best_len = i, size
        if best < 0:
            return passages[:limit]    # nothing mergeable — keep the head
        a, b = passages[best], passages[best + 1]
        passages[best : best + 2] = [Passage(f"{a.text} {b.text}", "prose", a.block)]
    return passages


def split_passages(content: str, max_passages: int = 32) -> list[Passage]:
    """Passages of a chunk body, in document order. Deterministic."""
    passages: list[Passage] = []
    for index, block in enumerate(split_blocks(content)):
        if block.kind == "code":
            passages.append(Passage(block.text, "code", index))
        elif block.kind == "table":
            header, rows = split_table(block.text)
            passages.extend(Passage(row, "table", index, header) for row in rows if row.strip())
        elif block.kind == "list":
            passages.extend(Passage(item, "list", index) for item in split_list_items(block.text))
        else:
            passages.extend(
                Passage(sentence, "prose", index)
                for sentence in _merge_short(split_sentences(block.text))
            )
    return _cap(passages, max_passages)


def passage_embedding_texts(heading: str, passages: list[Passage]) -> list[str]:
    """Text embedded for each passage: the section title gives a lone
    sentence the context it needs ("Rollback: completes in under five minutes")."""
    return [f"{heading}: {p.text}" for p in passages]


def join_passages(passages: list[Passage], selected: list[int]) -> str:
    """Rebuild readable markdown from a subset of passages (by index).

    Document order is preserved; a table keeps its header; a gap between
    kept passages is marked with an ellipsis so the reader knows text was
    left out."""
    out: list[str] = []
    previous: int | None = None
    shown_headers: set[int] = set()
    for i in sorted(selected):
        passage = passages[i]
        gap = previous is not None and i != previous + 1
        same_block = previous is not None and passages[previous].block == passage.block

        if passage.kind == "table" and passage.header and passage.block not in shown_headers:
            shown_headers.add(passage.block)
            out.append(("\n\n" if out else "") + passage.header + "\n" + passage.text)
        elif not out:
            out.append(("… " if i > 0 else "") + passage.text)
        elif same_block and not gap:
            out.append((" " if passage.kind == "prose" else "\n") + passage.text)
        elif same_block and passage.kind in ("list", "table"):
            out.append("\n" + passage.text)
        elif same_block:
            out.append(" … " + passage.text)
        else:
            out.append("\n\n" + ("… " if gap and passage.kind == "prose" else "") + passage.text)
        previous = i
    text = "".join(out)
    if selected and max(selected) < len(passages) - 1 and passages[max(selected)].kind == "prose":
        text += " …"
    return text
