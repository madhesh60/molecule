"""The `Molecule` engine: the public API, honest accounting, learning hooks."""
from __future__ import annotations

import threading

import pytest

from molecule import Molecule, MoleculeConfig, __version__
from tests.conftest import make_config


def test_round_trip(ingested):
    results = ingested.search("how do I roll back a release", k=3)
    assert results[0]["id"] == "handbook.md::Service Handbook:Rollback"
    body = ingested.fetch([results[0]["id"]])[0]
    assert "deployctl rollback" in body["content"]


def test_relative_paths_resolve_against_the_repo_root(mol, sample_doc):
    assert mol.ingest("handbook.md")["source"] == "handbook.md"
    assert "error" in mol.ingest("does-not-exist.md")


def test_invalid_explicit_config_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        Molecule(config=MoleculeConfig(db_path=tmp_path / "x.db", default_k=0))


def test_empty_queries_are_harmless(ingested):
    assert ingested.search("") == []
    assert ingested.search("   ") == []
    assert ingested.recall("") == []
    assert "error" in ingested.context("")


def test_context_manager_closes_the_store(cfg):
    with Molecule(config=cfg) as engine:
        assert engine.health()["status"] == "ok"
    with pytest.raises(Exception):
        engine.health()
    engine.close()   # closing twice is fine


# --- Honest token accounting ---------------------------------------------------------

def test_every_response_is_counted(ingested):
    ingested.index(max_tokens=200)
    results = ingested.search("rollback", k=2)
    ingested.fetch([results[0]["id"]])
    ingested.remember("Deploys never go out on Fridays")
    ingested.recall("when do deploys go out")
    by_kind = ingested.stats()["by_kind"]
    assert set(by_kind) == {"index", "search", "fetch", "memory"}
    assert all(entry["tokens"] > 0 for entry in by_kind.values())


def test_baseline_counts_each_file_once_and_only_when_delivered(ingested):
    stats = ingested.stats
    ingested.search("rollback")
    assert stats()["tokens_if_full_file_loaded"] == 0       # searching delivers no body
    first = ingested.search("rollback", k=1)[0]["id"]
    ingested.fetch([first])
    full_file = stats()["tokens_if_full_file_loaded"]
    assert full_file == ingested.sources()[0]["total_tokens"]
    ingested.fetch([first])
    ingested.context("how do deploys work")
    assert stats()["tokens_if_full_file_loaded"] == full_file   # never double-counted


def test_savings_include_the_overhead(ingested):
    results = ingested.search("rollback", k=1)
    chunk = ingested.fetch([results[0]["id"]])[0]
    stats = ingested.stats()
    assert stats["tokens_used_this_session"] > chunk["token_count"]   # search counted too
    expected = round((1 - stats["tokens_used_this_session"]
                      / stats["tokens_if_full_file_loaded"]) * 100, 1)
    assert stats["savings_pct"] == expected


def test_context_records_exactly_what_it_returned(ingested):
    result = ingested.context("how do I roll back a release", budget=300)
    assert ingested.stats()["by_kind"]["context"] == {"calls": 1, "tokens": result["tokens_used"]}


# --- Learning hooks ------------------------------------------------------------------

def test_chunks_fetched_together_become_bonded(ingested):
    logging_id = "handbook.md::Service Handbook:Logging"
    migrations_id = "handbook.md::Service Handbook:Database migrations"
    assert all(r["id"] != migrations_id for r in ingested.bonds(logging_id))
    ingested.fetch([logging_id, migrations_id])
    ingested.fetch([logging_id])
    ingested.fetch([migrations_id])            # within the window → counts as together
    related = {r["id"]: r for r in ingested.bonds(logging_id)}
    assert related[migrations_id]["bond"] == "cofetch"
    # …and the next fetch of one points at the other.
    assert ingested.fetch([logging_id])[0]["related"][0]["id"] == migrations_id


def test_search_results_point_at_referenced_sections(ingested):
    top = ingested.search("production deploys canary", k=1)[0]
    assert top["id"] == "handbook.md::Service Handbook:Deploy process"
    assert top["related"][0] == {
        "id": "handbook.md::Service Handbook:Rollback", "bond": "ref", "strength": 0.5,
    }


def test_feedback_then_search_shows_the_boost(ingested):
    query = "how do I roll back a release"
    second = ingested.search(query, k=3)[1]["id"]
    for _ in range(3):
        assert "error" not in ingested.feedback(second, True, query)
    after = next(r for r in ingested.search(query, k=3) if r["id"] == second)
    assert "learned boost: +" in after["reason"]
    assert ingested.feedback_stats()["votes"] == 3
    assert "error" in ingested.feedback("nope::chunk", True, query)


# --- Memory through the engine -------------------------------------------------------

def test_memory_survives_a_restart(cfg, sample_doc):
    with Molecule(config=cfg) as first:
        first.ingest(sample_doc)
        first.remember("Always deploy through Docker Compose", confidence=0.9)
        first.feedback("handbook.md::Service Handbook:Rollback", True, "undo a release")
    with Molecule(config=cfg) as second:
        assert "Docker Compose" in second.recall("how should we deploy")[0]["text"]
        assert second.feedback_stats()["votes"] == 1
        assert len(second.index()) == 5


def test_extract_and_history(mol):
    result = mol.extract("We decided to always deploy through Docker Compose. Lunch was late.")
    assert result["stored"] == 1
    memory_id = mol.recall("deploy")[0]["memory_id"]
    assert [h["memory_id"] for h in mol.memory_history(memory_id)] == [memory_id]
    assert mol.forget(memory_id) == {"deleted": memory_id}


def test_capacity_is_enforced_on_write(tmp_path):
    cfg = make_config(tmp_path, max_active_memories=2)
    with Molecule(config=cfg) as engine:
        for text in ("The API listens on port 8420", "Invoices are kept for seven years",
                     "The dashboard is served on port 5173"):
            engine.remember(text, confidence=0.9)
        assert engine.health()["active_memories"] == 2


# --- Operations ----------------------------------------------------------------------

def test_health_reports_version_counts_and_latency(ingested):
    ingested.search("rollback")
    health = ingested.health()
    assert health["version"] == __version__
    assert health["chunks_indexed"] == 5 and health["sources"] == 1
    assert health["stale_sources"] == []
    assert health["embedding_id"] == "hashing-384"
    assert health["metrics"]["operations"]["search"]["count"] == 1


def test_doctor_is_healthy_on_a_good_store_and_explains_problems(ingested, sample_doc):
    report = ingested.doctor()
    assert report["healthy"], report["failed"]
    sample_doc.unlink()
    report = ingested.doctor()
    assert report["failed"] == ["source files exist"]
    detail = next(c["detail"] for c in report["checks"] if c["check"] == "source files exist")
    assert "molecule sync --prune" in detail


def test_doctor_flags_an_empty_store(mol):
    assert "index" in mol.doctor()["failed"]


def test_evaluate_with_ablation(ingested):
    result = ingested.evaluate(
        [{"query": "undo a release", "answer": "deployctl rollback"}], k=3, ablate=True
    )
    assert result["hits"] == 1 and len(result["ablation"]) == 6


def test_sources_and_removal(ingested):
    assert [s["source_file"] for s in ingested.sources()] == ["handbook.md"]
    assert ingested.remove_source("handbook.md")["chunks_deleted"] == 5
    assert ingested.sources() == [] and ingested.search("rollback") == []


def test_one_engine_is_safe_to_share_between_threads(ingested):
    """Hammer one instance from several threads: no errors, consistent results."""
    errors: list[Exception] = []
    tops: list[str] = []
    stored: list[bool] = []

    def worker(n: int) -> None:
        try:
            for i in range(15):
                tops.append(ingested.search("how do I roll back a release", k=2)[0]["id"])
                result = ingested.remember(f"thread {n} fact number {i} about topic {n * 100 + i}")
                stored.append(bool(result.get("stored")))
                ingested.context("structured logging rules", budget=200)
                ingested.stats()
        except Exception as exc:  # noqa: BLE001 — collected and asserted below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert set(tops) == {"handbook.md::Service Handbook:Rollback"}
    # Every write either landed or was (correctly) skipped as a near-duplicate;
    # none was lost or double-counted.
    assert len(stored) == 90
    assert ingested.health()["active_memories"] == sum(stored) > 0
    assert ingested.metrics()["operations"]["search"]["count"] == 90
