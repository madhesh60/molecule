"""markdown.py — just enough markdown structure to cut documents safely.

Not a full parser. It answers the questions the chunker and the passage
splitter need answered correctly:

  * is this line inside a fenced code block? (a `# comment` in a bash
    snippet is not a heading)
  * where do the blocks — paragraphs, lists, tables, code fences — begin
    and end, so a cut never lands in the middle of one?
  * where do the sentences of a paragraph end?
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_LIST_ITEM_RE = re.compile(r"^(\s*)(?:[-*+]|\d+[.)])\s+\S")
_TABLE_LINE_RE = re.compile(r"^\s*\|")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$")
# A sentence ends at . ! or ? (optionally followed by a closing quote/bracket)
# when the next thing looks like the start of a new sentence.
_SENTENCE_END_RE = re.compile(
    r"(?:(?<=[.!?])|(?<=[.!?][\"')\]]))\s+(?=[A-Z0-9\"'`(\[*_])"
)
_SOFT_WRAP_RE = re.compile(r"\s*\n\s*")


class FenceTracker:
    """Feed lines in order; `feed()` says whether each is part of a fenced
    code block (the opening and closing fence lines count as inside)."""

    def __init__(self) -> None:
        self._fence: str | None = None

    @property
    def open(self) -> bool:
        return self._fence is not None

    def feed(self, line: str) -> bool:
        match = _FENCE_RE.match(line)
        if self._fence is None:
            if match:
                self._fence = match.group(1)
                return True
            return False
        if (
            match
            and match.group(1)[0] == self._fence[0]
            and len(match.group(1)) >= len(self._fence)
            and not match.group(2).strip()
        ):
            self._fence = None
        return True


@dataclass(frozen=True)
class Block:
    kind: str   # "code" | "table" | "list" | "para"
    text: str


def split_blocks(text: str) -> list[Block]:
    """Split a section body into its top-level blocks, in order."""
    blocks: list[Block] = []
    kind: str | None = None
    buffer: list[str] = []
    fence = FenceTracker()

    def flush() -> None:
        nonlocal kind, buffer
        if buffer and kind:
            blocks.append(Block(kind, "\n".join(buffer)))
        kind, buffer = None, []

    for line in text.split("\n"):
        was_open = fence.open
        if fence.feed(line):
            if not was_open:            # opening fence starts a new code block
                flush()
                kind = "code"
            buffer.append(line)
            if not fence.open:          # closing fence ends it
                flush()
            continue

        if not line.strip():
            flush()
        elif _TABLE_LINE_RE.match(line):
            if kind != "table":
                flush()
                kind = "table"
            buffer.append(line)
        elif _LIST_ITEM_RE.match(line):
            if kind != "list":
                flush()
                kind = "list"
            buffer.append(line)
        elif kind == "list":
            buffer.append(line)         # continuation line of the current item
        else:
            if kind != "para":
                flush()
                kind = "para"
            buffer.append(line)
    flush()
    return blocks


def split_sentences(text: str) -> list[str]:
    """Sentences of one paragraph. Soft line wraps are joined first."""
    flat = _SOFT_WRAP_RE.sub(" ", text.strip())
    return [s for s in _SENTENCE_END_RE.split(flat) if s]


def split_list_items(text: str) -> list[str]:
    """Top-level items of a list block (each keeps its continuation lines)."""
    lines = text.split("\n")
    indents = [len(m.group(1)) for m in map(_LIST_ITEM_RE.match, lines) if m]
    top = min(indents) if indents else 0
    items: list[list[str]] = []
    for line in lines:
        match = _LIST_ITEM_RE.match(line)
        if match and len(match.group(1)) == top:
            items.append([line])
        elif items:
            items[-1].append(line)
        else:
            items.append([line])
    return ["\n".join(item) for item in items]


def split_table(text: str) -> tuple[str, list[str]]:
    """(header, rows) of a table block. The header is the first row plus the
    `|---|` separator when present, otherwise empty."""
    lines = text.split("\n")
    if len(lines) >= 2 and _TABLE_SEPARATOR_RE.match(lines[1]):
        return "\n".join(lines[:2]), lines[2:]
    return "", lines


def code_fence_parts(text: str) -> tuple[str, list[str], str]:
    """(opening line, body lines, closing line) of a fenced code block."""
    lines = text.split("\n")
    opening = lines[0]
    match = _FENCE_RE.match(opening)
    marker = match.group(1) if match else "```"
    if len(lines) > 1 and _FENCE_RE.match(lines[-1]) and not lines[-1].strip().strip("`~"):
        return opening, lines[1:-1], lines[-1]
    return opening, lines[1:], marker   # unterminated fence — close it ourselves
