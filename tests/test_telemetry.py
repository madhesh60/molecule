from __future__ import annotations

from molecule.core.telemetry import get_stats, record_usage, source_tokens


def test_savings_pct(conn):
    record_usage(conn, 300, 1000)
    stats = get_stats(conn)
    assert stats["tokens_used_this_session"] == 300
    assert stats["tokens_if_full_file_loaded"] == 1000
    assert stats["savings_pct"] == 70.0


def test_zero_baseline(conn):
    stats = get_stats(conn)
    assert stats["savings_pct"] == 0.0


def test_accumulates(conn):
    record_usage(conn, 100, 500)
    record_usage(conn, 200, 500)
    stats = get_stats(conn)
    assert stats["tokens_used_this_session"] == 300
    assert stats["tokens_if_full_file_loaded"] == 1000


def test_breakdown_by_kind(conn):
    record_usage(conn, 40, 0, "index")
    record_usage(conn, 60, 0, "search")
    record_usage(conn, 60, 0, "search")
    record_usage(conn, 300, 2000, "fetch")
    stats = get_stats(conn)
    assert stats["by_kind"] == {
        "index": {"calls": 1, "tokens": 40},
        "search": {"calls": 2, "tokens": 120},
        "fetch": {"calls": 1, "tokens": 300},
    }
    # Overhead (index + search) counts against the savings — nothing is hidden.
    assert stats["tokens_used_this_session"] == 460
    assert stats["savings_pct"] == 77.0


def test_lifetime_spans_sessions_and_estimates_cost(conn):
    conn.execute(
        "INSERT INTO session_telemetry (session_id, tokens_used, tokens_baseline, ts, kind) "
        "VALUES ('an-older-session', 1000, 501000, '2026-01-01 00:00:00', 'fetch')"
    )
    conn.commit()
    record_usage(conn, 1000, 501000)
    life = get_stats(conn, cost_per_mtok_usd=2.0)["lifetime"]
    assert life["sessions"] == 2
    assert life["tokens_saved"] == 1_000_000
    assert life["est_usd_saved"] == 2.0


def test_source_tokens_prefers_the_true_file_size(conn):
    conn.execute(
        "INSERT INTO sources (source_file, abs_path, file_hash, size_bytes, mtime_ns, "
        "total_tokens, chunk_count, signature, ingested_at) "
        "VALUES ('a.md', '/a.md', 'h', 1, 1, 4321, 2, 's', 'now')"
    )
    conn.execute(
        "INSERT INTO chunks (chunk_id, source_file, heading_path, content, token_count, "
        "content_hash, last_updated) VALUES ('b.md::X', 'b.md', '[]', 'x', 77, 'h', 'now')"
    )
    conn.commit()
    assert source_tokens(conn, "a.md") == 4321
    assert source_tokens(conn, "b.md") == 77      # pre-0.2 store: sum of chunks
    assert source_tokens(conn, "missing.md") == 0
