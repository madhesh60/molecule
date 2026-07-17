from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import tiktoken

from molecule.config import MoleculeConfig
from molecule.db import content_hash

_ENC = tiktoken.get_encoding("cl100k_base")

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
_ALWAYS_INCLUDE_RE = re.compile(
    r"always_include:\s*true|<!--\s*always_include\s*-->", re.IGNORECASE
)


def _count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def _split_sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", text.strip())


@dataclass
class RawSection:
    heading_path: list[str]
    content: str
    always_include: bool = False
    token_count: int = 0
    hash: str = ""

    def __post_init__(self) -> None:
        self.token_count = _count_tokens(self.content)
        self.hash = content_hash(self.content)


def _detect_always_include(text: str) -> bool:
    return bool(_ALWAYS_INCLUDE_RE.search(text))


def parse_markdown(text: str) -> list[RawSection]:
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [RawSection(heading_path=["(root)"], content=text.strip())]

    sections: list[RawSection] = []
    heading_stack: list[str] = []

    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        heading_stack = heading_stack[: level - 1] + [title]

        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        if body:
            sections.append(
                RawSection(
                    heading_path=list(heading_stack),
                    content=body,
                    always_include=_detect_always_include(body),
                )
            )

    return sections


def _hard_split(text: str, max_tokens: int) -> list[str]:
    """Last-resort split for a single 'sentence' longer than max_tokens
    (e.g. a long list, table, or code block with no sentence punctuation).
    Splits on a token window so no chunk can ever exceed the bound."""
    tokens = _ENC.encode(text)
    return [
        _ENC.decode(tokens[i : i + max_tokens])
        for i in range(0, len(tokens), max_tokens)
    ]


def _split_large(section: RawSection, max_tokens: int) -> list[RawSection]:
    sentences = _split_sentences(section.content)
    # Any single "sentence" over the bound gets hard-split by token window first.
    expanded: list[str] = []
    for sent in sentences:
        if _count_tokens(sent) > max_tokens:
            expanded.extend(_hard_split(sent, max_tokens))
        else:
            expanded.append(sent)
    sentences = expanded
    chunks: list[RawSection] = []
    current: list[str] = []
    current_tokens = 0
    overlap: list[str] = []

    for sent in sentences:
        t = _count_tokens(sent)
        if current_tokens + t > max_tokens and current:
            body = " ".join(current)
            chunks.append(
                RawSection(
                    heading_path=section.heading_path,
                    content=body,
                    always_include=section.always_include,
                )
            )
            # ~1 sentence of overlap so meaning isn't cut mid-thought — but only
            # if the overlap sentence is small; a big one would blow the bound.
            overlap = current[-1:] if _count_tokens(current[-1]) <= max_tokens // 4 else []
            current = list(overlap)
            current_tokens = _count_tokens(" ".join(current))
        current.append(sent)
        current_tokens += t

    if current:
        chunks.append(
            RawSection(
                heading_path=section.heading_path,
                content=" ".join(current),
                always_include=section.always_include,
            )
        )

    # Distinct heading paths per part — otherwise all parts share one chunk_id
    # and silently overwrite each other in the DB.
    if len(chunks) > 1:
        for n, chunk in enumerate(chunks, start=1):
            chunk.heading_path = [*section.heading_path[:-1], f"{section.heading_path[-1]} §{n}"]
    return chunks


def _merge_small(sections: list[RawSection], min_tokens: int) -> list[RawSection]:
    merged: list[RawSection] = []
    for sec in sections:
        # Merge a tiny section into its previous sibling ONLY when they share a
        # real (non-root) parent heading. Top-level H1 sections are distinct
        # topics — merging them would poison routing (one chunk id, three topics).
        parent = sec.heading_path[:-1]
        if (
            merged
            and sec.token_count < min_tokens
            and parent
            and merged[-1].heading_path[:-1] == parent
        ):
            prev = merged[-1]
            merged[-1] = RawSection(
                heading_path=prev.heading_path,
                content=prev.content + "\n\n" + sec.content,
                always_include=prev.always_include or sec.always_include,
            )
        else:
            merged.append(sec)
    return merged


def chunk_file(file_path: Path, config: MoleculeConfig) -> list[RawSection]:
    text = file_path.read_text(encoding="utf-8")
    raw = parse_markdown(text)

    split: list[RawSection] = []
    for sec in raw:
        if sec.token_count > config.max_chunk_tokens:
            split.extend(_split_large(sec, config.max_chunk_tokens))
        else:
            split.append(sec)

    return _merge_small(split, config.min_chunk_tokens)
