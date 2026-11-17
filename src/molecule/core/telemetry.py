"""telemetry.py — honest token accounting.

`mol_stats` answers one question: how many tokens did Molecule put into the
agent's context, compared with loading the source files whole?

Rules that keep the number honest (not cherry-picked):

  * **Everything returned counts as used** — the index, search summaries,
    recalled memories, fetched bodies and compiled contexts — not just the
    fetched chunks.
  * **The baseline is conservative**: each source file is counted once per
    session, the first time one of its chunks is actually delivered. A file
    that was only searched is never counted as "would have been loaded".
  * The true size of each source file is measured at ingest, so the
    baseline is the real file, not a sum of chunks.

Every row carries a `kind` (index | search | fetch | context | memory) so
the breakdown shows where the tokens went.
"""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

from molecule.core.tokens import count_tokens, tokenizer_name
from molecule.db import utc_now_iso

_SESSION_ID = str(uuid.uuid4())

KINDS = ("index", "search", "fetch", "context", "memory")


def session_id() -> str:
    return _SESSION_ID


def baseline_tokens(file_path: Path) -> int:
    """Token size of a file on disk (0 when unreadable)."""
    try:
        return count_tokens(file_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return 0


def source_tokens(conn: sqlite3.Connection, source: str) -> int:
    """Full token size of one ingested source file."""
    row = conn.execute(
        "SELECT total_tokens FROM sources WHERE source_file=?", (source,)
    ).fetchone()
    if row:
        return int(row[0])
    # Store from before sources were tracked: fall back to the sum of chunks.
    row = conn.execute(
        "SELECT SUM(token_count) FROM chunks WHERE chunk_id LIKE ?", (f"{source}::%",)
    ).fetchone()
    return int(row[0] or 0)


def record_usage(
    conn: sqlite3.Connection, tokens_used: int, tokens_baseline: int, kind: str = "fetch"
) -> None:
    conn.execute(
        "INSERT INTO session_telemetry (session_id, tokens_used, tokens_baseline, ts, kind) "
        "VALUES (?,?,?,?,?)",
        (_SESSION_ID, int(tokens_used), int(tokens_baseline), utc_now_iso(), kind),
    )
    conn.commit()


def _savings_pct(used: int, baseline: int) -> float:
    return round((1 - used / baseline) * 100, 1) if baseline > 0 else 0.0


def get_stats(conn: sqlite3.Connection, cost_per_mtok_usd: float = 3.0) -> dict:
    """Session and lifetime token savings, with a per-kind breakdown."""
    by_kind = {
        r["kind"]: {"calls": r["calls"], "tokens": r["used"] or 0}
        for r in conn.execute(
            "SELECT kind, COUNT(*) AS calls, SUM(tokens_used) AS used "
            "FROM session_telemetry WHERE session_id=? GROUP BY kind",
            (_SESSION_ID,),
        )
    }
    row = conn.execute(
        "SELECT SUM(tokens_used) AS used, SUM(tokens_baseline) AS baseline "
        "FROM session_telemetry WHERE session_id=?",
        (_SESSION_ID,),
    ).fetchone()
    used, baseline = row["used"] or 0, row["baseline"] or 0

    life = conn.execute(
        "SELECT SUM(tokens_used) AS used, SUM(tokens_baseline) AS baseline, "
        "COUNT(DISTINCT session_id) AS sessions FROM session_telemetry"
    ).fetchone()
    life_used, life_baseline = life["used"] or 0, life["baseline"] or 0
    saved = max(0, life_baseline - life_used)

    return {
        "session_id": _SESSION_ID,
        "tokens_used_this_session": used,
        "tokens_if_full_file_loaded": baseline,
        "savings_pct": _savings_pct(used, baseline),
        "by_kind": by_kind,
        "lifetime": {
            "sessions": life["sessions"] or 0,
            "tokens_used": life_used,
            "tokens_if_full_files_loaded": life_baseline,
            "savings_pct": _savings_pct(life_used, life_baseline),
            "tokens_saved": saved,
            "est_usd_saved": round(saved / 1_000_000 * cost_per_mtok_usd, 4),
        },
        "tokenizer": tokenizer_name(),
    }
