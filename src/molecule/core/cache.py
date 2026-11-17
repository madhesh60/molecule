"""cache.py — per-connection caches for the hot search path that cannot go stale.

Search needs the embedding matrix, the lexical index and chunk metadata on
every call, but those only change on a write. `versioned()` memoizes any
build step under a *version* that changes exactly when the underlying data
does. Two ways to get that version:

  * `domains=("chunks", "feedback", ...)` — precise mode. SQLite triggers
    (see db.py) bump a per-table generation counter on every write, so the
    version is just those counters. Raw SQL and writes from another process
    are caught too, and bookkeeping tables (telemetry, memory stats) evict
    nothing.
  * no domains — generic mode for any connection, including ones without
    Molecule's schema: `PRAGMA data_version` (commits by other connections)
    plus `total_changes` (writes on this one), minus append-only telemetry.

A weakref identity check guards against id() reuse: if a connection is
garbage-collected and a new one lands at the same address, the old entry is
ignored. Kept deliberately blunt: one flat dict, cleared wholesale if it
ever grows past a small cap.
"""
from __future__ import annotations

import sqlite3
import weakref
from collections.abc import Callable
from typing import Any

from molecule.db import generations

_MAX_ENTRIES = 64

_store: dict[tuple[int, str], tuple[Any, tuple, Any]] = {}


def _generic_version(conn: sqlite3.Connection) -> tuple[int, int]:
    """Version tuple that changes on any non-telemetry write.

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


def versioned(
    conn: sqlite3.Connection,
    key: str,
    build: Callable[[], Any],
    domains: tuple[str, ...] | None = None,
    salt: Any = None,
) -> Any:
    """Return build()'s result, cached until the data it depends on changes.

    `key` namespaces independent caches on the same connection. `domains`
    names the tables the value is derived from (precise invalidation);
    `salt` is any extra value that should also invalidate (e.g. a time
    bucket for data that expires on its own).
    """
    version: tuple = generations(conn, domains) if domains else _generic_version(conn)
    version = (*version, salt)
    k = (id(conn), key)
    hit = _store.get(k)
    if hit is not None and hit[0]() is conn and hit[1] == version:
        return hit[2]
    value = build()
    if len(_store) >= _MAX_ENTRIES:
        _store.clear()
    try:
        ref: Any = weakref.ref(conn)
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


def forget(conn: sqlite3.Connection) -> None:
    """Drop every cached value for one connection (used after a rollback)."""
    for k in [k for k in _store if k[0] == id(conn)]:
        del _store[k]


def clear() -> None:
    """Drop everything — used by tests."""
    _store.clear()
