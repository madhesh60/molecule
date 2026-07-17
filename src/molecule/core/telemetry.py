from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import tiktoken

_SESSION_ID = str(uuid.uuid4())
_ENC = tiktoken.get_encoding("cl100k_base")


def session_id() -> str:
    return _SESSION_ID


def baseline_tokens(file_path: Path) -> int:
    try:
        return len(_ENC.encode(file_path.read_text(encoding="utf-8")))
    except Exception:
        return 0


def record_usage(conn: sqlite3.Connection, tokens_used: int, tokens_baseline: int) -> None:
    conn.execute(
        "INSERT INTO session_telemetry (session_id, tokens_used, tokens_baseline, ts) VALUES (?,?,?,?)",
        (_SESSION_ID, tokens_used, tokens_baseline, datetime.utcnow().isoformat()),
    )
    conn.commit()


def get_stats(conn: sqlite3.Connection) -> dict:
    row = conn.execute(
        "SELECT SUM(tokens_used) as used, SUM(tokens_baseline) as baseline FROM session_telemetry WHERE session_id=?",
        (_SESSION_ID,),
    ).fetchone()
    used = row["used"] or 0
    baseline = row["baseline"] or 0
    savings_pct = round((1 - used / baseline) * 100, 1) if baseline > 0 else 0.0
    return {
        "session_id": _SESSION_ID,
        "tokens_used_this_session": used,
        "tokens_if_full_file_loaded": baseline,
        "savings_pct": savings_pct,
    }
