"""chunker.py — turn one markdown document into labelled, bounded chunks.

Pipeline:  headings → sections → split the big ones → merge the tiny ones.

  * **Headings** (H1..H3 by default) are the natural boundaries. Lines
    inside fenced code blocks are never headings, so a `# comment` in a
    bash snippet cannot tear a section apart.
  * **Big sections** (> max_chunk_tokens) are split on *block* boundaries —
    paragraph, list, table, code fence — and only inside a block when one
    block alone is too large. A split table repeats its header row, a split
    code block is re-fenced, a split paragraph breaks on sentences with one
    sentence of overlap. Every part is still valid markdown.
  * **Tiny sections** (< min_chunk_tokens) are merged into the previous
    sibling under the same parent heading.

Chunk bodies keep the author's formatting byte-for-byte unless a block had
to be cut. Heading paths are unique within a file, so chunk ids never
collide (a repeated "Example" heading becomes "Example (2)").
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from molecule.config import MoleculeConfig
from molecule.core.markdown import (
    Block,
    FenceTracker,
    code_fence_parts,
    split_blocks,
    split_list_items,
    split_sentences,
    split_table,
)
from molecule.core.tokens import count_tokens, split_by_tokens
from molecule.db import content_hash

# Bump when chunk boundaries change for the same input, so stores re-chunk.
CHUNKER_VERSION = 2

_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_ALWAYS_COMMENT_RE = re.compile(r"<!--\s*always_include\s*-->", re.IGNORECASE)
_FRONT_MATTER_FLAG_RE = re.compile(r"^always_include:\s*true\s*$", re.IGNORECASE | re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`[^`]*`")
_BLOCK_JOIN = "\n\n"

ROOT_HEADING = "(root)"     # a file with no headings at all
INTRO_HEADING = "(intro)"   # text before the first heading


def _count_tokens(text: str) -> int:
    return count_tokens(text)


@dataclass
class RawSection:
    heading_path: list[str]
    content: str
    always_include: bool = False
    token_count: int = 0
    hash: str = ""
    part: int = 0        # 0 = whole section, n = n-th part of a split section
    position: int = 0    # order of the chunk in the document

    def __post_init__(self) -> None:
        self.token_count = _count_tokens(self.content)
        self.hash = content_hash(self.content)


def _strip_front_matter(text: str) -> tuple[str, bool]:
    """Remove a leading YAML front-matter block; report its always_include flag."""
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return text, False
    for i in range(1, len(lines)):
        if lines[i].strip() in ("---", "..."):
            front = "\n".join(lines[1:i])
            return "\n".join(lines[i + 1 :]), bool(_FRONT_MATTER_FLAG_RE.search(front))
    return text, False


def parse_markdown(text: str, split_depth: int = 3) -> list[RawSection]:
    """Split a document into sections at headings of level <= split_depth."""
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    body, file_always = _strip_front_matter(text)

    sections: list[RawSection] = []
    stack: list[str] = []
    path: list[str] | None = None
    buffer: list[str] = []
    marked = False
    saw_heading = False
    fence = FenceTracker()

    def flush() -> None:
        content = "\n".join(buffer).strip()
        if content:
            sections.append(RawSection(
                heading_path=list(path) if path else [INTRO_HEADING],
                content=content,
                always_include=file_always or marked,
            ))

    for line in body.split("\n"):
        if not fence.feed(line):
            heading = _HEADING_RE.match(line)
            if heading and len(heading.group(1)) <= split_depth:
                flush()
                buffer, marked, saw_heading = [], False, True
                level = len(heading.group(1))
                stack = stack[: level - 1] + [heading.group(2).strip()]
                path = list(stack)
                continue
            # The marker must be real markup, not a mention inside `inline code`.
            if _ALWAYS_COMMENT_RE.search(_INLINE_CODE_RE.sub("", line)):
                marked = True
                # It is an instruction to Molecule, not content for the agent.
                line = _ALWAYS_COMMENT_RE.sub("", line)
                if not line.strip():
                    continue
        buffer.append(line)
    flush()

    if not saw_heading:
        for section in sections:
            section.heading_path = [ROOT_HEADING]
    return sections


# --- Splitting sections that are too large ---------------------------------------

def _split_prose(text: str, max_tokens: int) -> list[str]:
    """Sentence groups of at most max_tokens, with one small sentence of overlap."""
    sentences: list[str] = []
    for sentence in split_sentences(text):
        if _count_tokens(sentence) > max_tokens:
            sentences.extend(split_by_tokens(sentence, max_tokens))
        else:
            sentences.append(sentence)

    groups: list[str] = []
    current: list[str] = []
    used = 0
    for sentence in sentences:
        cost = _count_tokens(sentence)
        if current and used + cost > max_tokens:
            groups.append(" ".join(current))
            # ~1 sentence of overlap so meaning isn't cut mid-thought — but only
            # when that sentence is small; a big one would blow the bound.
            last = current[-1]
            carry = [last] if _count_tokens(last) <= max_tokens // 4 else []
            current, used = list(carry), sum(_count_tokens(s) for s in carry)
        current.append(sentence)
        used += cost
    if current:
        groups.append(" ".join(current))
    return groups


def _group_lines(lines: list[str], budget: int) -> list[list[str]]:
    """Consecutive line groups of at most `budget` tokens each."""
    groups: list[list[str]] = []
    current: list[str] = []
    used = 0
    for line in lines:
        pieces = split_by_tokens(line, budget) if _count_tokens(line) > budget else [line]
        for piece in pieces:
            cost = _count_tokens(piece) + 1
            if current and used + cost > budget:
                groups.append(current)
                current, used = [], 0
            current.append(piece)
            used += cost
    if current:
        groups.append(current)
    return groups


def _split_code(text: str, max_tokens: int) -> list[str]:
    opening, body, closing = code_fence_parts(text)
    budget = max(16, max_tokens - _count_tokens(opening) - _count_tokens(closing) - 2)
    return ["\n".join([opening, *group, closing]) for group in _group_lines(body, budget)]


def _split_table(text: str, max_tokens: int) -> list[str]:
    header, rows = split_table(text)
    budget = max(16, max_tokens - _count_tokens(header) - 1)
    parts = []
    for group in _group_lines(rows, budget):
        parts.append("\n".join(([header] if header else []) + group))
    return parts


def _split_block(block: Block, max_tokens: int) -> list[str]:
    """Cut one oversized block into pieces that each fit, keeping its shape."""
    if block.kind == "code":
        return _split_code(block.text, max_tokens)
    if block.kind == "table":
        return _split_table(block.text, max_tokens)
    if block.kind == "list":
        pieces: list[str] = []
        for item in split_list_items(block.text):
            if _count_tokens(item) > max_tokens:
                pieces.extend(_split_prose(item, max_tokens))
            else:
                pieces.append(item)
        return pieces
    return _split_prose(block.text, max_tokens)


def _split_large(section: RawSection, max_tokens: int) -> list[RawSection]:
    """Split an oversized section on block boundaries into numbered parts."""
    units: list[tuple[str, str]] = []   # (kind, text)
    for block in split_blocks(section.content):
        if _count_tokens(block.text) > max_tokens:
            units.extend((block.kind, piece) for piece in _split_block(block, max_tokens))
        else:
            units.append((block.kind, block.text))

    parts: list[str] = []
    current: list[str] = []
    current_kind = ""
    used = 0
    for kind, text in units:
        cost = _count_tokens(text) + 1
        if current and used + cost > max_tokens:
            parts.append("".join(current))
            current, used = [], 0
        if current:
            # Consecutive list items stay one list; everything else is a new block.
            current.append("\n" if kind == "list" and current_kind == "list" else _BLOCK_JOIN)
        current.append(text)
        current_kind = kind
        used += cost
    if current:
        parts.append("".join(current))

    if len(parts) <= 1:
        return [section]
    # Distinct heading paths per part — otherwise all parts would share one
    # chunk id and silently overwrite each other in the DB.
    *parents, last = section.heading_path
    return [
        RawSection(
            heading_path=[*parents, f"{last} §{n}"],
            content=part,
            always_include=section.always_include,
            part=n,
        )
        for n, part in enumerate(parts, start=1)
    ]


def _merge_small(sections: list[RawSection], min_tokens: int) -> list[RawSection]:
    merged: list[RawSection] = []
    for sec in sections:
        # Merge a tiny section into its previous sibling ONLY when they share a
        # real (non-root) parent heading. Top-level H1 sections are distinct
        # topics — merging them would poison routing (one chunk id, three topics).
        # A guardrail (always_include) is short by nature and must stay exactly
        # what the author marked: it is never merged with an ordinary section.
        parent = sec.heading_path[:-1]
        if (
            merged
            and sec.token_count < min_tokens
            and parent
            and merged[-1].heading_path[:-1] == parent
            and merged[-1].always_include == sec.always_include
        ):
            prev = merged[-1]
            merged[-1] = RawSection(
                heading_path=prev.heading_path,
                content=prev.content + _BLOCK_JOIN + sec.content,
                always_include=prev.always_include or sec.always_include,
                part=prev.part,
            )
        else:
            merged.append(sec)
    return merged


def _make_paths_unique(sections: list[RawSection]) -> None:
    """Give repeated heading paths a numeric suffix so chunk ids are unique."""
    seen: dict[tuple[str, ...], int] = {}
    for section in sections:
        key = tuple(section.heading_path)
        count = seen.get(key, 0) + 1
        seen[key] = count
        if count > 1:
            section.heading_path = [*section.heading_path[:-1],
                                    f"{section.heading_path[-1]} ({count})"]
            seen[tuple(section.heading_path)] = 1


def chunk_text(text: str, config: MoleculeConfig) -> list[RawSection]:
    """Chunk a markdown string. See the module docstring for the rules."""
    split: list[RawSection] = []
    for section in parse_markdown(text, config.heading_split_depth):
        if section.token_count > config.max_chunk_tokens:
            split.extend(_split_large(section, config.max_chunk_tokens))
        else:
            split.append(section)

    chunks = _merge_small(split, config.min_chunk_tokens)
    _make_paths_unique(chunks)
    for position, chunk in enumerate(chunks):
        chunk.position = position
    return chunks


def chunk_file(file_path: Path, config: MoleculeConfig) -> list[RawSection]:
    return chunk_text(file_path.read_text(encoding="utf-8"), config)
