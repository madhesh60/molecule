"""cache.py — tiny per-connection, data-versioned caches for the hot search path.

Search re-reads embeddings, metadata and BM25 doc tokens from SQLite on every
call even though they only change on a write (ingest / feedback / mem_write).
`versioned()` memoizes any build step keyed on a version tuple that changes on
every write that could affect ranking, so a stale result can never be served:

  - `conn.total_changes` bumps on every INSERT/UPDATE/DELETE through this
    connection. Telemetry inserts (one per fetch — the normal search→fetch→
    search loop) are subtracted out via session_telemetry's MAX(rowid) so
    pure bookkeeping never evicts the corpus.
  - `PRAGMA data_version` bumps when ANOTHER connection commits to the same
    DB file, so a second process (CLI next to the MCP server) invalidates too.

A weakref identity check guards against id() reuse: if a connection is
garbage-collected and a new one lands at the same address, the old entry is
ignored. Kept dependency-free and deliberately blunt: one flat dict, cleared
wholesale if it ever grows past a small cap (tests open many short-lived
connections).
"""
from __future__ import annotations

import sqlite3
import weakref
from typing import Any, Callable

_MAX_ENTRIES = 64

_store: dict[tuple[int, str], tuple[Any, tuple[int, int], Any]] = {}


def _version(conn: sqlite3.Connection) -> tuple[int, int]:
    """Version tuple that changes iff a ranking-relevant write happened.

    Telemetry rows are append-only, so subtracting MAX(rowid) cancels their
    contribution to total_changes exactly; any other write shifts the diff.
    External commits (other connections) are caught by data_version.
    """
    data_version = conn.execute("PRAGMA data_version").fetchone()[0]
    try:
        telemetry_rows = conn.execute(
            "SELECT COALESCE(MAX(rowid), 0) FROM session_telemetry"
        ).fetchone()[0]
    except sqlite3.OperationalError:  # schema not initialized (bare test conns)
        telemetry_rows = 0
    return (data_version, conn.total_changes - telemetry_rows)


def versioned(conn: sqlite3.Connection, key: str, build: Callable[[], Any]) -> Any:
    """Return build()'s result, cached until a ranking-relevant write happens
    on `conn` or any other connection to the same DB file.

    `key` namespaces independent caches on the same connection. Reads don't
    change the version, so pure-read builds are always safe to memoize.
    """
    version = _version(conn)
    k = (id(conn), key)
    hit = _store.get(k)
    if hit is not None and hit[0]() is conn and hit[1] == version:
        return hit[2]
    value = build()
    if len(_store) >= _MAX_ENTRIES:
        _store.clear()
    try:
        ref = weakref.ref(conn)
    except TypeError:  # plain sqlite3.Connection isn't weakref-able
        ref = _StrongRef(conn)
    _store[k] = (ref, version, value)
    return value


class _StrongRef:
    """weakref.ref-compatible fallback for objects that can't be weak-referenced."""

    __slots__ = ("_obj",)

    def __init__(self, obj: Any) -> None:
        self._obj = obj

    def __call__(self) -> Any:
        return self._obj


def clear() -> None:
    """Drop everything — used by tests."""
    _store.clear()
