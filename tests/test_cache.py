"""Tests for the versioned per-connection cache powering the hot search path."""
from __future__ import annotations

import sqlite3

import pytest

from molecule.core import cache


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE t (x INTEGER)")
    return c


def setup_function(_fn) -> None:
    cache.clear()


def test_returns_build_result():
    conn = _conn()
    assert cache.versioned(conn, "k", lambda: 42) == 42


def test_caches_between_reads():
    conn = _conn()
    calls = []
    cache.versioned(conn, "k", lambda: calls.append(1))
    cache.versioned(conn, "k", lambda: calls.append(1))
    assert len(calls) == 1  # second call served from cache


def test_write_invalidates():
    conn = _conn()
    cache.versioned(conn, "k", lambda: "old")
    conn.execute("INSERT INTO t VALUES (1)")  # bumps total_changes
    assert cache.versioned(conn, "k", lambda: "new") == "new"


def test_keys_are_independent():
    conn = _conn()
    cache.versioned(conn, "a", lambda: 1)
    assert cache.versioned(conn, "b", lambda: 2) == 2
    assert cache.versioned(conn, "a", lambda: 99) == 1  # still cached


def test_connections_are_independent():
    c1, c2 = _conn(), _conn()
    cache.versioned(c1, "k", lambda: "one")
    assert cache.versioned(c2, "k", lambda: "two") == "two"


def test_cap_clears_instead_of_growing():
    conn = _conn()
    for i in range(cache._MAX_ENTRIES + 5):
        cache.versioned(conn, f"k{i}", lambda: i)
    assert len(cache._store) <= cache._MAX_ENTRIES + 1


def test_search_uses_cache_and_write_busts_it(tmp_path):
    """End-to-end: rank_candidates hits the corpus cache; feedback invalidates it."""
    from molecule.config import MoleculeConfig
    from molecule.core import retriever
    from molecule.db import get_connection, init_schema

    cfg = MoleculeConfig(db_path=tmp_path / "store.db")
    conn = get_connection(cfg.db_path)
    init_schema(conn)

    import numpy as np
    vec = np.ones(4, dtype=np.float32) / 2.0
    conn.execute(
        "INSERT INTO chunks (chunk_id, source_file, heading_path, summary, content, "
        "token_count, content_hash, always_include, weight, embedding, last_updated) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("f::A", "f", '["A"]', "about apples", "apples are red", 4, "h1", 0, 1.0,
         vec.tobytes(), "now"),
    )
    conn.commit()

    corpus1 = cache.versioned(conn, "chunk_corpus", lambda: retriever._chunk_corpus(conn))
    corpus2 = cache.versioned(conn, "chunk_corpus", lambda: retriever._chunk_corpus(conn))
    assert corpus1 is corpus2  # same object — cache hit

    conn.execute("UPDATE chunks SET learned_weight=1.2 WHERE chunk_id='f::A'")
    conn.commit()
    corpus3 = cache.versioned(conn, "chunk_corpus", lambda: retriever._chunk_corpus(conn))
    assert corpus3 is not corpus1  # write invalidated
    assert corpus3.prior_weight[corpus3.row_of["f::A"]] == pytest.approx(1.2)


def test_telemetry_writes_do_not_invalidate(tmp_path):
    """The normal search→fetch→search loop logs telemetry between searches;
    that bookkeeping must not evict the corpus cache."""
    from molecule.config import MoleculeConfig
    from molecule.core.telemetry import record_usage
    from molecule.db import get_connection, init_schema

    cfg = MoleculeConfig(db_path=tmp_path / "store.db")
    conn = get_connection(cfg.db_path)
    init_schema(conn)

    v1 = cache.versioned(conn, "k", lambda: object())
    record_usage(conn, 100, 1000)
    v2 = cache.versioned(conn, "k", lambda: object())
    assert v1 is v2  # telemetry insert did not bust the cache


def test_other_connection_write_invalidates(tmp_path):
    """A second process (e.g. CLI beside the MCP server) writing to the same
    DB file must invalidate this connection's cache (via PRAGMA data_version)."""
    from molecule.config import MoleculeConfig
    from molecule.db import get_connection, init_schema

    cfg = MoleculeConfig(db_path=tmp_path / "store.db")
    conn_a = get_connection(cfg.db_path)
    init_schema(conn_a)

    v1 = cache.versioned(conn_a, "k", lambda: object())

    conn_b = get_connection(cfg.db_path)
    conn_b.execute(
        "INSERT INTO memories (memory_id, text, created_at) VALUES ('m1', 'x', 'now')"
    )
    conn_b.commit()
    conn_b.close()

    v2 = cache.versioned(conn_a, "k", lambda: object())
    assert v1 is not v2  # external commit invalidated


# --- Precise invalidation through generation counters ---------------------------

def _chunk(conn, chunk_id: str) -> None:
    conn.execute(
        "INSERT INTO chunks (chunk_id, source_file, heading_path, content, token_count, "
        "content_hash, last_updated) VALUES (?,?,?,?,?,?,?)",
        (chunk_id, "f", '["H"]', "body", 1, chunk_id, "now"),
    )
    conn.commit()


def test_domain_cache_ignores_unrelated_tables(conn):
    """A value derived from chunks must survive writes to telemetry, memory
    stats and bonds — only chunk writes may evict it."""
    from molecule.core.telemetry import record_usage

    _chunk(conn, "f::A")
    _chunk(conn, "f::B")
    first = cache.versioned(conn, "k", object, domains=("chunks",))
    record_usage(conn, 10, 100)
    conn.execute(
        "INSERT INTO chunk_bonds (src, dst, kind, strength, updated_at) "
        "VALUES ('f::A', 'f::B', 'cofetch', 0.2, 'now')"
    )
    conn.commit()
    assert cache.versioned(conn, "k", object, domains=("chunks",)) is first


def test_domain_cache_is_evicted_by_raw_sql(conn):
    _chunk(conn, "f::A")
    first = cache.versioned(conn, "k", object, domains=("chunks",))
    conn.execute("UPDATE chunks SET summary='changed' WHERE chunk_id='f::A'")
    conn.commit()
    assert cache.versioned(conn, "k", object, domains=("chunks",)) is not first


def test_domain_cache_sees_writes_from_another_connection(cfg, conn):
    from molecule.db import get_connection

    first = cache.versioned(conn, "k", object, domains=("chunks",))
    other = get_connection(cfg.db_path)
    _chunk(other, "f::Z")
    other.close()
    assert cache.versioned(conn, "k", object, domains=("chunks",)) is not first


def test_memory_bookkeeping_does_not_evict_memory_cache(conn):
    conn.execute("INSERT INTO memories (memory_id, text, created_at) VALUES ('m1', 'x', 'now')")
    conn.commit()
    first = cache.versioned(conn, "k", object, domains=("memories",))
    conn.execute("UPDATE memories SET updated_at='later' WHERE memory_id='m1'")
    conn.execute("INSERT INTO memory_stats (memory_id, hits) VALUES ('m1', 3)")
    conn.commit()
    assert cache.versioned(conn, "k", object, domains=("memories",)) is first
    conn.execute("UPDATE memories SET status='replaced' WHERE memory_id='m1'")
    conn.commit()
    assert cache.versioned(conn, "k", object, domains=("memories",)) is not first


def test_salt_change_invalidates(conn):
    first = cache.versioned(conn, "k", object, domains=("chunks",), salt="model-a")
    assert cache.versioned(conn, "k", object, domains=("chunks",), salt="model-a") is first
    assert cache.versioned(conn, "k", object, domains=("chunks",), salt="model-b") is not first


def test_rollback_forgets_cached_state(conn):
    """A value cached inside a failed transaction must not outlive it."""
    from molecule.db import transaction

    with pytest.raises(RuntimeError):
        with transaction(conn):
            _chunk_no_commit(conn, "f::tmp")
            cache.versioned(conn, "k", lambda: "uncommitted", domains=("chunks",))
            raise RuntimeError("boom")
    assert cache.versioned(conn, "k", lambda: "clean", domains=("chunks",)) == "clean"


def _chunk_no_commit(conn, chunk_id: str) -> None:
    conn.execute(
        "INSERT INTO chunks (chunk_id, source_file, heading_path, content, token_count, "
        "content_hash, last_updated) VALUES (?,?,?,?,?,?,?)",
        (chunk_id, "f", '["H"]', "body", 1, chunk_id, "now"),
    )


def test_connections_are_released_with_their_cache_entries(cfg):
    """Molecule connections are weak-referenceable, so a cache entry does not
    keep a closed connection alive."""
    import gc
    import weakref

    from molecule.db import get_connection, init_schema

    connection = get_connection(cfg.db_path)
    init_schema(connection)
    cache.versioned(connection, "k", object, domains=("chunks",))
    ref = weakref.ref(connection)
    connection.close()
    del connection
    gc.collect()
    assert ref() is None
