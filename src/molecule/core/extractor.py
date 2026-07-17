"""extractor.py — heuristic memory extraction from session transcripts.

Pulls up to `extract_max_facts` durable facts out of raw transcript text
with zero LLM calls (works offline, no API key). Cue-based confidence
follows AGENTS.md §16.2: direct statements 0.9, hedged/inferred 0.6.
Ephemeral facts route to `task_state` (auto 7-day expiry).

Every extracted fact goes through write_memory(), which already handles
secret redaction, dedup and conflict replacement — nothing reimplemented.
An optional llm_fn (same pattern as labeler.py) can rewrite facts into
cleaner one-liners when an LLM is available.
"""
from __future__ import annotations

import re
import sqlite3
from typing import Callable

from molecule.config import MoleculeConfig
from molecule.core.memory import write_memory

# Direct decision/preference/rule statements → confidence 0.9
_DIRECT_CUES = re.compile(
    r"\b(we (decided|agreed)|from now on|always|never|must|do not|don't"
    r"|prefer(s|red)?|instead of|the rule is|going forward|make sure"
    r"|remember to|use \S+ (not|over|rather than))\b",
    re.IGNORECASE,
)

# First-person preference statements → confidence 0.9
_FIRST_PERSON_CUES = re.compile(
    r"\b(i|we)\s+(prefer|like|want|hate|avoid|use|love|need)\b",
    re.IGNORECASE,
)

# Hedged / inferred statements → confidence 0.6
_HEDGED_CUES = re.compile(
    r"\b(i think|probably|maybe|seems?|might|it looks like|perhaps)\b",
    re.IGNORECASE,
)

# Ephemeral markers → task_state (expires), confidence capped at 0.6
_EPHEMERAL_CUES = re.compile(
    r"\b(currently|for now|right now|today|debugging|this week|temporarily)\b",
    re.IGNORECASE,
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

_MIN_SENTENCE_LEN = 15


def _classify(sentence: str) -> tuple[float, str] | None:
    """Return (confidence, mem_type) for a sentence, or None if not a fact."""
    direct = bool(_DIRECT_CUES.search(sentence) or _FIRST_PERSON_CUES.search(sentence))
    hedged = bool(_HEDGED_CUES.search(sentence))
    ephemeral = bool(_EPHEMERAL_CUES.search(sentence))

    if not (direct or hedged or ephemeral):
        return None

    # Hedging caps confidence even when a direct cue is also present.
    confidence = 0.9 if direct and not hedged else 0.6
    mem_type = "preference"
    if ephemeral:
        confidence = min(confidence, 0.6)
        mem_type = "task_state"
    return confidence, mem_type


def extract_memories(
    transcript: str,
    source_task: str,
    conn: sqlite3.Connection,
    config: MoleculeConfig,
    llm_fn: Callable[[list[str]], list[str]] | None = None,
) -> dict:
    """Extract durable facts from a transcript and store them via write_memory."""
    sentences = [
        s.strip() for s in _SENTENCE_SPLIT.split(transcript)
        if len(s.strip()) > _MIN_SENTENCE_LEN
    ]

    candidates: list[tuple[str, float, str]] = []
    seen: set[str] = set()
    for sentence in sentences:
        verdict = _classify(sentence)
        if verdict is None:
            continue
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        confidence, mem_type = verdict
        candidates.append((sentence, confidence, mem_type))

    # Highest confidence first; shorter facts win ties (more likely atomic).
    candidates.sort(key=lambda c: (-c[1], len(c[0])))
    candidates = candidates[: config.extract_max_facts]

    if llm_fn and candidates:
        rewritten = llm_fn([c[0] for c in candidates])
        candidates = [
            (new_text, conf, mtype)
            for new_text, (_, conf, mtype) in zip(rewritten, candidates)
        ]

    facts = []
    stored = skipped = 0
    for text, confidence, mem_type in candidates:
        result = write_memory(text, source_task, confidence, conn, config, mem_type)
        outcome = "stored" if result.get("stored") else "skipped"
        if result.get("stored"):
            stored += 1
        else:
            skipped += 1
        facts.append({
            "text": text,
            "confidence": confidence,
            "type": mem_type,
            "outcome": outcome,
        })

    return {
        "extracted": len(candidates),
        "stored": stored,
        "skipped": skipped,
        "facts": facts,
    }
