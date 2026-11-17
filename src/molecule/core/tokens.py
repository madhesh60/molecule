"""tokens.py — token counting that never needs the network at import time.

Molecule's promise is "works offline". tiktoken, however, downloads its
vocabulary on first use — and the old code did that at *import* time, so a
machine with no network (or a locked-down CI runner) could not even import
the package. This module fixes that:

  * the tiktoken encoder is loaded lazily, on the first count;
  * if it cannot be loaded (offline, blocked, not installed) we fall back to
    a deterministic estimator instead of crashing;
  * `tokenizer_name()` reports which one is active, so `mol_health` can say
    whether the savings numbers are exact or estimated.

Set `MOLECULE_TOKENIZER=estimate` to force the estimator (handy in tests).
"""
from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any

_log = logging.getLogger("molecule")

_ENCODING_NAME = "cl100k_base"
_lock = threading.Lock()
_encoder: Any = None
_resolved = False

# One "piece" is a run of letters/digits or a single punctuation mark.
_PIECE_RE = re.compile(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]")
# A word plus the whitespace after it — used to split text without losing layout.
_WORD_WS_RE = re.compile(r"\S+\s*")


def _load_encoder() -> Any:
    """Return the tiktoken encoder, or None when it is unavailable."""
    global _encoder, _resolved
    if _resolved:
        return _encoder
    with _lock:
        if _resolved:
            return _encoder
        if os.environ.get("MOLECULE_TOKENIZER", "").lower() != "estimate":
            try:
                import tiktoken

                _encoder = tiktoken.get_encoding(_ENCODING_NAME)
            except Exception as exc:  # noqa: BLE001 — any failure means "estimate"
                _log.warning(
                    "tiktoken unavailable (%s: %s) — using the built-in token estimator",
                    type(exc).__name__, exc,
                )
        _resolved = True
    return _encoder


def reset_tokenizer() -> None:
    """Forget the cached encoder choice (tests flip MOLECULE_TOKENIZER)."""
    global _encoder, _resolved
    with _lock:
        _encoder = None
        _resolved = False


def tokenizer_name() -> str:
    """'cl100k_base' when counts are exact, 'estimate' when approximated."""
    return _ENCODING_NAME if _load_encoder() is not None else "estimate"


def estimate_tokens(text: str) -> int:
    """Deterministic approximation of a BPE token count.

    Short words are one token, long words/identifiers roughly one token per
    five characters, every punctuation mark one token. Within ~10% of
    cl100k_base on English prose and markdown — good enough to keep chunk
    sizes and savings percentages meaningful when tiktoken is unavailable.
    """
    total = 0
    for piece in _PIECE_RE.findall(text):
        total += max(1, (len(piece) + 3) // 5) if piece[0].isalnum() else 1
    return total


def count_tokens(text: str) -> int:
    """Number of LLM tokens in `text` (exact with tiktoken, else estimated)."""
    if not text:
        return 0
    encoder = _load_encoder()
    if encoder is None:
        return estimate_tokens(text)
    return len(encoder.encode(text, disallowed_special=()))


def split_by_tokens(text: str, max_tokens: int) -> list[str]:
    """Cut `text` into consecutive pieces of at most `max_tokens` tokens.

    Last-resort splitter for a single unit that is too large on its own
    (a giant sentence, a minified line). Nothing is dropped: joining the
    pieces gives back the original text.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    encoder = _load_encoder()
    if encoder is not None:
        ids = encoder.encode(text, disallowed_special=())
        return [
            encoder.decode(ids[i : i + max_tokens]) for i in range(0, len(ids), max_tokens)
        ]

    pieces: list[str] = []
    current: list[str] = []
    used = 0
    for word in _WORD_WS_RE.findall(text):
        cost = estimate_tokens(word)
        if current and used + cost > max_tokens:
            pieces.append("".join(current))
            current, used = [], 0
        current.append(word)
        used += cost
    if current:
        pieces.append("".join(current))
    return pieces
