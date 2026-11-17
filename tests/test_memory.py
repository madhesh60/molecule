from __future__ import annotations

from datetime import timedelta

import pytest

from molecule.core.memory import (
    enforce_capacity,
    expire_stale,
    forget_memory,
    memory_history,
    merge_duplicates,
    redact,
    search_memories,
    write_memory,
)
from molecule.db import utc_now
from tests.conftest import make_config


def _status(conn, memory_id: str) -> str:
    return conn.execute(
        "SELECT status FROM memories WHERE memory_id=?", (memory_id,)
    ).fetchone()["status"]


def test_write_and_search(conn, cfg):
    result = write_memory("User prefers dark mode", "task1", 0.9, conn, cfg)
    assert result.get("stored") is True
    hits = search_memories("dark mode preference", 5, conn, cfg)
    assert any("dark mode" in h["text"] for h in hits)
    assert hits[0]["reason"] and hits[0]["relevance"] in {"high", "medium"}


def test_dedup_skips_similar(conn, cfg):
    write_memory("User prefers dark mode", "task1", 0.9, conn, cfg)
    result = write_memory("User prefers dark mode", "task2", 0.9, conn, cfg)
    assert result.get("skipped") == "duplicate"
    assert conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1


@pytest.mark.model
def test_conflict_replaces(conn, model_cfg):
    r1 = write_memory("User prefers dark mode", "task1", 0.9, conn, model_cfg)
    r2 = write_memory("User no longer prefers dark mode, changed to light mode", "task2", 0.9,
                      conn, model_cfg)
    assert _status(conn, r1["memory_id"]) == "replaced"
    assert r2["replaced"] == r1["memory_id"]


def test_conflict_keeps_a_lineage(conn, tmp_path):
    """The retired fact points at the fact that replaced it, so the history
    of a decision can be read back."""
    cfg = make_config(tmp_path, conflict_threshold=0.5)
    first = write_memory("Deploys go out on Fridays", "t", 0.9, conn, cfg)
    second = write_memory("Deploys no longer go out on Fridays", "t", 0.9, conn, cfg)
    assert second["replaced"] == first["memory_id"]
    assert _status(conn, first["memory_id"]) == "replaced"

    history = memory_history(conn, second["memory_id"])
    assert [h["memory_id"] for h in history] == [first["memory_id"], second["memory_id"]]
    assert history[0]["superseded_by"] == second["memory_id"]
    assert memory_history(conn, first["memory_id"]) == history   # same chain from either end
    assert memory_history(conn, "nope") == []

    # Only the current fact is searchable.
    assert [h["memory_id"] for h in search_memories("deploys on fridays", 5, conn, cfg)] == [
        second["memory_id"]
    ]


def test_secret_redaction(conn, cfg):
    write_memory("API key is sk-abc123xyz456def789ghi012jkl345mno", "task1", 0.9, conn, cfg)
    rows = conn.execute("SELECT text FROM memories").fetchall()
    assert all("sk-" not in r["text"] for r in rows)


@pytest.mark.parametrize("secret", [
    "ghp_" + "a" * 36,
    "AKIA" + "ABCDEFGHIJKLMNOP",
    "xoxb-1234567890-abcdefghij",
    "password: hunter2",
    "postgres://admin:s3cret@db.internal/prod",
    "-----BEGIN RSA PRIVATE KEY-----",
])
def test_redact_covers_common_secret_shapes(secret):
    redacted = redact(f"note {secret} end")
    assert "[REDACTED]" in redacted
    assert secret not in redacted


def test_expiry(conn, cfg):
    write_memory("Debugging issue X", "task1", 0.8, conn, cfg, mem_type="task_state")
    yesterday = (utc_now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("UPDATE memories SET expires_at=? WHERE type='task_state'", (yesterday,))
    conn.commit()
    assert expire_stale(conn) == 1
    rows = conn.execute("SELECT status FROM memories WHERE type='task_state'").fetchall()
    assert all(r["status"] == "replaced" for r in rows)


def test_task_state_gets_an_expiry_and_preferences_do_not(conn, cfg):
    write_memory("Currently debugging the webhook test", "t", 0.8, conn, cfg, "task_state")
    write_memory("Always deploy through Docker Compose", "t", 0.9, conn, cfg)
    rows = {r["type"]: r["expires_at"] for r in conn.execute("SELECT type, expires_at FROM memories")}
    assert rows["task_state"] is not None and rows["preference"] is None


def test_expired_memory_is_not_returned(conn, cfg):
    write_memory("Currently debugging the webhook test", "t", 0.8, conn, cfg, "task_state")
    yesterday = (utc_now() - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("UPDATE memories SET expires_at=?", (yesterday,))
    conn.commit()
    assert search_memories("debugging webhook test", 5, conn, cfg) == []


def test_low_confidence_fact_ranks_below_a_stated_one(conn, cfg):
    write_memory("Maybe the team uses Redis for the queue", "t", 0.5, conn, cfg)
    write_memory("The team uses SQS for the job queue", "t", 0.9, conn, cfg)
    hits = search_memories("what does the team use for the queue", 2, conn, cfg)
    assert "SQS" in hits[0]["text"]
    weights = {r["confidence"]: r["weight"] for r in conn.execute("SELECT confidence, weight FROM memories")}
    assert weights[0.5] < weights[0.9]


def test_recall_counts_hits_without_touching_the_search_cache(conn, cfg):
    from molecule.db import generations

    result = write_memory("User prefers dark mode", "t", 0.9, conn, cfg)
    before = generations(conn, ("memories",))
    for _ in range(3):
        search_memories("dark mode", 5, conn, cfg)
    hits = conn.execute(
        "SELECT hits FROM memory_stats WHERE memory_id=?", (result["memory_id"],)
    ).fetchone()[0]
    assert hits == 3
    assert generations(conn, ("memories",)) == before   # bookkeeping only


def test_capacity_evicts_the_least_valuable_first(conn, tmp_path):
    cfg = make_config(tmp_path, max_active_memories=2)
    guess = write_memory("Maybe we use tabs somewhere", "t", 0.4, conn, cfg)
    stated = write_memory("The API listens on port 8420", "t", 0.9, conn, cfg)
    used = write_memory("Invoices are kept for seven years", "t", 0.9, conn, cfg)
    search_memories("how long are invoices kept", 1, conn, cfg)
    assert enforce_capacity(conn, cfg) == 1
    assert _status(conn, guess["memory_id"]) == "evicted"
    assert _status(conn, stated["memory_id"]) == _status(conn, used["memory_id"]) == "active"
    assert enforce_capacity(conn, cfg) == 0


def test_forget(conn, cfg):
    result = write_memory("User prefers dark mode", "t", 0.9, conn, cfg)
    assert forget_memory(conn, result["memory_id"]) == {"deleted": result["memory_id"]}
    assert "error" in forget_memory(conn, result["memory_id"])
    assert search_memories("dark mode", 5, conn, cfg) == []


def test_merge_duplicates_collapses_a_cluster(conn, tmp_path):
    cfg = make_config(tmp_path, dedup_threshold=0.999, conflict_threshold=0.5)
    for text in ("Keep pull requests under 300 lines",
                 "Keep pull requests under 300 lines of code please",
                 "The API listens on port 8420"):
        assert write_memory(text, "t", 0.9, conn, cfg).get("stored")
    merged = merge_duplicates(conn, cfg, lambda groups: [g.split("\n")[0] for g in groups])
    assert merged == 1
    active = [r["text"] for r in conn.execute("SELECT text FROM memories WHERE status='active'")]
    assert sorted(active) == ["Keep pull requests under 300 lines", "The API listens on port 8420"]
    replaced = conn.execute(
        "SELECT superseded_by FROM memories WHERE status='replaced'"
    ).fetchall()
    assert len(replaced) == 2 and all(r["superseded_by"] for r in replaced)


def test_overlong_memory_is_truncated(conn, cfg):
    write_memory("fact " * 2000, "t", 0.9, conn, cfg)
    assert len(conn.execute("SELECT text FROM memories").fetchone()["text"]) <= 2000
