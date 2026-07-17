from __future__ import annotations

import pytest

from molecule.config import MoleculeConfig
from molecule.core.telemetry import get_stats, record_usage
from molecule.db import get_connection, init_schema


@pytest.fixture
def tel_db(tmp_path):
    cfg = MoleculeConfig(db_path=tmp_path / ".molecule" / "store.db")
    conn = get_connection(cfg.db_path)
    init_schema(conn)
    return conn


def test_savings_pct(tel_db):
    record_usage(tel_db, 300, 1000)
    stats = get_stats(tel_db)
    assert stats["tokens_used_this_session"] == 300
    assert stats["tokens_if_full_file_loaded"] == 1000
    assert stats["savings_pct"] == 70.0


def test_zero_baseline(tel_db):
    stats = get_stats(tel_db)
    assert stats["savings_pct"] == 0.0


def test_accumulates(tel_db):
    record_usage(tel_db, 100, 500)
    record_usage(tel_db, 200, 500)
    stats = get_stats(tel_db)
    assert stats["tokens_used_this_session"] == 300
    assert stats["tokens_if_full_file_loaded"] == 1000
