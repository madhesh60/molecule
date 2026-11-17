"""Shared fixtures.

Unit tests run on the built-in `hashing` embedder: no model download, no
network, fully deterministic — so the suite is hermetic and fast. Tests that
check *semantic* behaviour ask for the `model_cfg` fixture instead; they use
the real embedding model and are skipped (not failed) when it is not cached.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from molecule import Molecule, MoleculeConfig  # noqa: E402
from molecule.core import cache  # noqa: E402
from molecule.db import get_connection, init_schema  # noqa: E402

REPO_ROOT = Path(__file__).parent.parent

SAMPLE_DOC = """\
# Service Handbook

Welcome to the handbook. It explains how the payments service is built, shipped
and operated day to day. Every section is meant to be read on its own, so an
agent or a new hire can answer one question without reading everything else
first. When this file disagrees with an old wiki page, this file wins.

## Deploy process

Production deploys are started with the `deployctl ship` command and roll out
as a canary first: five percent of traffic for fifteen minutes while the bot
watches error rate and latency. If the canary looks unhealthy, roll back first
and investigate afterwards — see [Rollback](#rollback). You deploy your own
change and stay online until it is fully out.

## Rollback

To undo a release run `deployctl rollback`. The command reverts the service to
the previous healthy image and prints the version that is now serving. A
rollback completes in under five minutes and never needs approval; anyone on
the owning team may start one at any hour of the day or night.

## Database migrations

Schema changes use the expand and contract pattern: add the new column first,
write to both shapes, backfill in batches, and remove the old column in a
later release. Never drop a column in the same deploy that stops using it.
Every migration sets a lock timeout so a blocked statement fails fast.

## Logging

Logs are structured JSON, one event per line, and every line carries a
correlation identifier so one request can be followed across components.
Never log passwords, tokens or personal data. Debug output is switched off in
production and can be enabled for a single tenant for one hour.
"""


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "model: needs the real embedding model (cached)")


@pytest.fixture(autouse=True)
def _fresh_caches():
    cache.clear()
    yield
    cache.clear()


def make_config(tmp_path: Path, **overrides) -> MoleculeConfig:
    """Hermetic config: hashing embedder, no background freshness checks."""
    settings = {"embedding_model": "hashing", "auto_sync": False, **overrides}
    return MoleculeConfig(db_path=tmp_path / ".molecule" / "store.db", **settings)


@pytest.fixture
def cfg(tmp_path: Path) -> MoleculeConfig:
    return make_config(tmp_path)


@pytest.fixture
def conn(cfg: MoleculeConfig):
    connection = get_connection(cfg.db_path)
    init_schema(connection)
    yield connection
    connection.close()


@pytest.fixture
def mol(cfg: MoleculeConfig):
    engine = Molecule(config=cfg)
    yield engine
    engine.close()


@pytest.fixture
def sample_doc(tmp_path: Path) -> Path:
    path = tmp_path / "handbook.md"
    path.write_text(SAMPLE_DOC, encoding="utf-8")
    return path


@pytest.fixture
def ingested(mol: Molecule, sample_doc: Path) -> Molecule:
    result = mol.ingest(sample_doc)
    assert "error" not in result, result
    return mol


@pytest.fixture(scope="session")
def real_model():
    """The real embedding model, or skip the test when it is unavailable."""
    from molecule.core.embedder import DEFAULT_MODEL, EmbeddingUnavailableError, get_embedder

    try:
        return get_embedder(DEFAULT_MODEL)
    except EmbeddingUnavailableError as exc:  # offline and never set up
        pytest.skip(f"embedding model not available: {exc}")


@pytest.fixture
def model_cfg(tmp_path: Path, real_model) -> MoleculeConfig:
    return MoleculeConfig(db_path=tmp_path / ".molecule" / "store.db", auto_sync=False)
