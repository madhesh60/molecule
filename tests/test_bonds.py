"""Bonds: cross-references, continuations and Hebbian co-fetch learning."""
from __future__ import annotations

from datetime import timedelta

import pytest

from molecule.core.bonds import (
    bond_stats,
    neighbors,
    related_for,
    slugify,
    strengthen_cofetch,
)
from molecule.core.ingest import ingest_file
from molecule.db import utc_now
from tests.conftest import make_config

PAD = "This sentence pads the section so that it stays a chunk of its own. " * 5

DOC = f"""\
# Handbook

## 1. Deploys

### 1.1 Deploy process

Ship with the deploy command. If it looks wrong, roll back first (see §1.2). {PAD}

### 1.2 Rollback

Run the rollback command. Migrations are expand/contract, see section 2.1. {PAD}

## 2. Data

### 2.1 Migrations

Use expand and contract. Details are in [the rollback notes](#12-rollback). {PAD}

### 2.2 Backups

Nightly snapshots. Nothing here refers to anything else. {PAD}
"""


@pytest.fixture
def store(conn, cfg, tmp_path):
    path = tmp_path / "book.md"
    path.write_text(DOC, encoding="utf-8")
    result = ingest_file(path, conn, cfg)
    assert "error" not in result
    return conn, cfg, result


def _id(suffix: str) -> str:
    parents = {"1": "1. Deploys", "2": "2. Data"}[suffix[0]]
    return f"book.md::Handbook:{parents}:{suffix}"


DEPLOY, ROLLBACK, MIGRATIONS, BACKUPS = (
    _id("1.1 Deploy process"), _id("1.2 Rollback"), _id("2.1 Migrations"), _id("2.2 Backups"),
)


def _bonds(conn) -> set[tuple[str, str, str]]:
    return {(r["src"], r["dst"], r["kind"]) for r in conn.execute("SELECT * FROM chunk_bonds")}


def test_slugify_matches_github_anchors():
    assert slugify("7.3 Rollback") == "73-rollback"
    assert slugify("Who to ask?") == "who-to-ask"


def test_section_number_references_become_bonds(store):
    conn, _, result = store
    assert result["bonds"] == 3
    assert (DEPLOY, ROLLBACK, "ref") in _bonds(conn)        # "see §1.2"
    assert (ROLLBACK, MIGRATIONS, "ref") in _bonds(conn)    # "see section 2.1"


def test_anchor_links_become_bonds(store):
    conn, _, _ = store
    assert (MIGRATIONS, ROLLBACK, "ref") in _bonds(conn)    # [..](#12-rollback)


def test_unreferenced_chunk_has_no_bonds(store):
    conn, cfg, _ = store
    assert related_for(conn, cfg, BACKUPS) == []


def test_reference_bonds_are_rebuilt_not_duplicated(store, tmp_path):
    conn, cfg, _ = store
    ingest_file(tmp_path / "book.md", conn, cfg, force=True)
    assert len(_bonds(conn)) == 3


def test_removed_reference_removes_its_bond(store, tmp_path):
    conn, cfg, _ = store
    path = tmp_path / "book.md"
    path.write_text(DOC.replace("(see §1.2)", "(ask the team)"), encoding="utf-8")
    ingest_file(path, conn, cfg)
    assert (DEPLOY, ROLLBACK, "ref") not in _bonds(conn)


def test_links_to_another_file_resolve_to_its_chunks(conn, cfg, tmp_path):
    (tmp_path / "setup.md").write_text(f"# Setup\n\nInstall things. {PAD}\n", encoding="utf-8")
    (tmp_path / "readme.md").write_text(
        f"# Readme\n\nStart with [the setup guide](setup.md). {PAD}\n", encoding="utf-8"
    )
    ingest_file(tmp_path / "setup.md", conn, cfg)
    ingest_file(tmp_path / "readme.md", conn, cfg)
    assert ("readme.md::Readme", "setup.md::Setup", "ref") in _bonds(conn)


def test_external_links_are_ignored(conn, cfg, tmp_path):
    (tmp_path / "a.md").write_text(
        f"# A\n\nSee [the docs](https://example.com/x.md#y) and [mail](mailto:a@b.c). {PAD}\n",
        encoding="utf-8",
    )
    assert ingest_file(tmp_path / "a.md", conn, cfg)["bonds"] == 0


def test_parts_of_a_split_section_are_bonded_both_ways(conn, tmp_path):
    cfg = make_config(tmp_path, max_chunk_tokens=60, min_chunk_tokens=5)
    body = "\n\n".join(f"Paragraph {i} says something specific about topic {i}." for i in range(24))
    (tmp_path / "big.md").write_text(f"# Big\n\n{body}\n", encoding="utf-8")
    ingest_file(tmp_path / "big.md", conn, cfg)
    bonds = _bonds(conn)
    assert ("big.md::Big §1", "big.md::Big §2", "continuation") in bonds
    assert ("big.md::Big §2", "big.md::Big §1", "continuation") in bonds


def test_cofetch_strength_follows_the_hebbian_rule(store):
    conn, cfg, _ = store
    for expected in (0.2, 0.36, 0.488):       # s ← s + 0.2·(1 − s)
        strengthen_cofetch(conn, [DEPLOY, BACKUPS], cfg)
        conn.commit()
        row = conn.execute("SELECT strength FROM chunk_bonds WHERE kind='cofetch'").fetchone()
        assert row["strength"] == pytest.approx(expected)
    assert conn.execute("SELECT hits FROM chunk_bonds WHERE kind='cofetch'").fetchone()[0] == 3


def test_cofetch_strength_saturates_below_one(store):
    conn, cfg, _ = store
    for _ in range(100):
        strengthen_cofetch(conn, [DEPLOY, BACKUPS], cfg)
    strength = conn.execute(
        "SELECT strength FROM chunk_bonds WHERE kind='cofetch'"
    ).fetchone()[0]
    assert 0.99 < strength <= 1.0


def test_cofetch_ignores_single_and_unknown_chunks(store):
    conn, cfg, _ = store
    assert strengthen_cofetch(conn, [DEPLOY], cfg) == 0
    assert strengthen_cofetch(conn, [DEPLOY, "book.md::nope"], cfg) == 0


def test_weak_learned_bond_is_hidden_until_reinforced(store):
    conn, cfg, _ = store
    strengthen_cofetch(conn, [DEPLOY, BACKUPS], cfg)        # 0.20 < bond_min_strength
    conn.commit()
    assert all(r["id"] != BACKUPS for r in related_for(conn, cfg, DEPLOY))
    strengthen_cofetch(conn, [DEPLOY, BACKUPS], cfg)        # 0.36
    conn.commit()
    learned = [r for r in related_for(conn, cfg, DEPLOY) if r["id"] == BACKUPS]
    assert learned and learned[0]["bond"] == "cofetch"
    # A learned bond works in both directions.
    assert any(r["id"] == DEPLOY for r in related_for(conn, cfg, BACKUPS))


def test_learned_bonds_fade_when_not_reinforced(store):
    conn, cfg, _ = store
    for _ in range(4):
        strengthen_cofetch(conn, [DEPLOY, BACKUPS], cfg)
    conn.commit()
    assert BACKUPS in {n[0] for n in neighbors(conn, cfg)[DEPLOY]}
    long_ago = (utc_now() - timedelta(days=120)).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("UPDATE chunk_bonds SET updated_at=? WHERE kind='cofetch'", (long_ago,))
    conn.commit()
    # 0.59 × 0.5^(120/30) ≈ 0.04 — below the visibility threshold.
    assert BACKUPS not in {n[0] for n in neighbors(conn, cfg).get(DEPLOY, [])}


def test_two_reasons_to_bond_reinforce_each_other(store):
    conn, cfg, _ = store
    before = next(n for n in neighbors(conn, cfg)[DEPLOY] if n[0] == ROLLBACK)[1]
    for _ in range(3):
        strengthen_cofetch(conn, [DEPLOY, ROLLBACK], cfg)
    conn.commit()
    after = next(n for n in neighbors(conn, cfg)[DEPLOY] if n[0] == ROLLBACK)[1]
    assert before == pytest.approx(0.5) and after > before


def test_related_for_respects_exclude_and_limit(store):
    conn, cfg, _ = store
    assert related_for(conn, cfg, DEPLOY, exclude={ROLLBACK}) == []
    assert len(related_for(conn, cfg, DEPLOY, limit=1)) == 1


def test_bond_stats(store):
    conn, cfg, _ = store
    for _ in range(2):
        strengthen_cofetch(conn, [DEPLOY, BACKUPS], cfg)
    conn.commit()
    stats = bond_stats(conn)
    assert stats["bonds"] == {"cofetch": 1, "ref": 3}
    assert stats["strongest_learned"][0]["hits"] == 2
