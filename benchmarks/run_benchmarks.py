"""run_benchmarks.py — measure everything the README claims, reproducibly.

    python benchmarks/run_benchmarks.py            # all experiments
    python benchmarks/run_benchmarks.py routing context   # just some

Writes `benchmarks/results/latest.json` and regenerates `benchmarks/RESULTS.md`.
Every experiment runs against a fresh temporary store with the default
configuration (real embedding model), so the numbers are what a new user
gets. Nothing here is mocked.

Experiments
  routing     Hit@1 / Hit@3 / MRR on every labeled query set, + v0.1 baseline
  ablation    the same sets with one ranking signal removed at a time
  abstention  does "relevance: low" separate off-topic from on-topic questions?
  context     answer recall vs token budget: compiler vs naive truncation
  feedback    does a vote help similar questions without bending unrelated ones?
  bonds       are both halves of a two-part question delivered together?
  savings     tokens spent per workflow on a scripted 10-question session
  speed       ingest, incremental re-ingest, search and context latency
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import statistics
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from molecule import Molecule, MoleculeConfig, __version__  # noqa: E402
from molecule.core.context import build_context  # noqa: E402
from molecule.core.eval import run_ablation  # noqa: E402
from molecule.core.retriever import score_chunks  # noqa: E402
from molecule.core.tokens import count_tokens, split_by_tokens  # noqa: E402

HANDBOOK = BENCH / "corpus" / "handbook.md"
AGENTS = ROOT / "AGENTS.md"
_WS = re.compile(r"\s+")


def norm(text: str) -> str:
    return _WS.sub(" ", text)


def load(name: str) -> list[dict]:
    path = BENCH / "queries" / f"{name}.json"
    if not path.exists():
        path = ROOT / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


QUERY_SETS: dict[str, tuple[Path, str]] = {
    "agents_main": (AGENTS, "eval_queries"),
    "agents_adversarial": (AGENTS, "eval_queries_adversarial"),
    "handbook_direct": (HANDBOOK, "handbook_direct"),
    "handbook_paraphrase": (HANDBOOK, "handbook_paraphrase"),
}


@contextmanager
def store(*docs: Path, **overrides):
    """A fresh engine with `docs` ingested, in a temp dir that is cleaned up."""
    tmp = Path(tempfile.mkdtemp(prefix="molecule_bench_"))
    cfg = MoleculeConfig(db_path=tmp / ".molecule" / "store.db", auto_sync=False, **overrides)
    mol = Molecule(config=cfg)
    try:
        for doc in docs:
            result = mol.ingest(doc)
            if "error" in result:
                raise SystemExit(f"ingest failed: {result['error']}")
        yield mol
    finally:
        mol.close()
        shutil.rmtree(tmp, ignore_errors=True)


def pct(value: float) -> float:
    return round(100.0 * value, 1)


# --- routing --------------------------------------------------------------------------

def bench_routing() -> dict:
    out: dict = {}
    for name, (doc, query_file) in QUERY_SETS.items():
        with store(doc) as mol:
            result = mol.evaluate(load(query_file), k=3)
        out[name] = {
            "n": result["total"],
            "hit_at_1": result["hit_at_1"],
            "hit_at_3": result["top3_precision"],
            "mrr": result["mrr"],
            "misses": [m["query"] for m in result["misses"]],
        }
    baseline_path = BENCH / "results" / "baseline_v0.1.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        for name, row in baseline.items():
            if name in out:
                out[name]["v01"] = {
                    "hit_at_1": row["hit_at_1"], "hit_at_3": row["hit_at_3"],
                    "mrr": row["mrr_at_10"],
                }
    return out


# --- ablation -------------------------------------------------------------------------

def bench_ablation() -> dict:
    out: dict = {}
    for name, (doc, query_file) in QUERY_SETS.items():
        with store(doc) as mol:
            rows = run_ablation(load(query_file), mol._conn, mol.config, k=3)
        out[name] = {
            r["variant"]: {"hit_at_3": r["top3_precision"], "hit_at_1": r["hit_at_1"]}
            for r in rows
        }
    return out


# --- abstention -----------------------------------------------------------------------

def _answered(mol: Molecule, query: str) -> bool:
    """True when Molecule would hand the agent a section for this query."""
    chunks = [r for r in mol.search(query, k=1) if r["type"] == "chunk"]
    return bool(chunks) and chunks[0]["relevance"] != "low"


def bench_abstention() -> dict:
    out: dict = {}
    cases = {
        "agents (thresholds were tuned here)": (
            AGENTS, load("eval_queries") + load("eval_queries_adversarial"), load("oos_dev")),
        "handbook (held out)": (
            HANDBOOK, load("handbook_direct") + load("handbook_paraphrase"),
            load("handbook_oos")),
    }
    for name, (doc, in_scope, off_topic) in cases.items():
        with store(doc) as mol:
            kept = sum(_answered(mol, q["query"]) for q in in_scope)
            wrongly_answered = [q["query"] for q in off_topic if _answered(mol, q["query"])]
        out[name] = {
            "in_scope": len(in_scope), "in_scope_answered": kept,
            "off_topic": len(off_topic),
            "off_topic_refused": len(off_topic) - len(wrongly_answered),
            "off_topic_wrongly_answered": wrongly_answered,
        }
    return out


# --- context compiler -----------------------------------------------------------------

def _naive_context(mol: Molecule, query: str, budget: int) -> str:
    """The "fill the budget" baseline: ranked chunks in order, as many as fit,
    the text cut off wherever the budget ends."""
    scores = score_chunks(query, mol._conn, mol.config)
    parts: list[str] = []
    used = 0
    for row in scores.order():
        chunk_id = scores.corpus.ids[int(row)]
        body = mol._conn.execute(
            "SELECT content FROM chunks WHERE chunk_id=?", (chunk_id,)
        ).fetchone()[0]
        text = f"### {chunk_id.partition('::')[0]} › {chunk_id.rsplit(':', 1)[-1]}\n{body}"
        cost = count_tokens(text)
        if used + cost > budget:
            room = budget - used
            if room > 8:
                parts.append(split_by_tokens(text, room)[0])
            break
        parts.append(text)
        used += cost + 2
    return "\n\n".join(parts)


def bench_context() -> dict:
    """Answer recall vs tokens spent, for three ways of filling a budget."""
    # The paraphrase set is used because there the labeled phrase is exactly
    # what the question asks for. (In the direct set the phrase only identifies
    # the right section, so a correct excerpt may legitimately leave it out.)
    queries = load("handbook_paraphrase")
    off_topic = load("handbook_oos")
    budgets = (100, 150, 200, 300, 600, 1200)
    out: dict = {"n": len(queries), "budgets": {}}
    with store(HANDBOOK) as mol:
        focus = replace(mol.config, context_excerpt="focus")
        for budget in budgets:
            row: dict = {}
            for label, cfg in (("compiler", mol.config), ("compiler_focus", focus)):
                hits, spent, over = 0, [], 0
                for q in queries:
                    result = build_context(q["query"], mol._conn, cfg, budget=budget)
                    hits += norm(q["answer"]) in norm(result["context"])
                    spent.append(result["tokens_used"])
                    over += result["tokens_used"] > budget
                row[f"{label}_answer_recall"] = round(hits / len(queries), 3)
                row[f"{label}_mean_tokens"] = round(statistics.mean(spent))
                row[f"{label}_over_budget"] = over
            hits, spent = 0, []
            for q in queries:
                naive = _naive_context(mol, q["query"], budget)
                hits += norm(q["answer"]) in norm(naive)
                spent.append(count_tokens(naive))
            row["fill_answer_recall"] = round(hits / len(queries), 3)
            row["fill_mean_tokens"] = round(statistics.mean(spent))
            row["compiler_off_topic_mean_tokens"] = round(statistics.mean(
                build_context(q["query"], mol._conn, mol.config, budget=budget)["tokens_used"]
                for q in off_topic
            ))
            out["budgets"][budget] = row
        out["full_file_tokens"] = mol.sources()[0]["total_tokens"]
    return out


# --- feedback -------------------------------------------------------------------------

def _gold_rank(mol: Molecule, query: dict, cfg: MoleculeConfig) -> tuple[int | None, list[str]]:
    scores = score_chunks(query["query"], mol._conn, cfg)
    ids = [scores.corpus.ids[int(r)] for r in scores.order()[:10]]
    needle = norm(query["answer"])
    for rank, chunk_id in enumerate(ids, start=1):
        body = mol._conn.execute(
            "SELECT content FROM chunks WHERE chunk_id=?", (chunk_id,)
        ).fetchone()[0]
        if needle in norm(body):
            return rank, ids
    return None, ids


def _hit1_mrr(mol: Molecule, queries: list[dict], cfg: MoleculeConfig) -> dict:
    ranks = [_gold_rank(mol, q, cfg)[0] for q in queries]
    return {
        "hit_at_1": round(sum(r == 1 for r in ranks) / len(ranks), 3),
        "mrr": round(sum(1 / r for r in ranks if r) / len(ranks), 3),
    }


def bench_feedback() -> dict:
    """An agent struggles with half of the (hard, paraphrased) questions, fetches
    around, and votes. Then three things are measured:

      same      the very same questions, asked again
      reworded  the same needs in new words (never seen while voting)
      other     every question about the *other* half of the topics — these
                must not get worse

    …under query-aware feedback (the default) and under a plain global
    weight per chunk (what 0.1 did), from the same votes.
    """
    direct, paraphrase = load("handbook_direct"), load("handbook_paraphrase")
    reworded = load("handbook_reworded")
    taught = list(range(0, len(paraphrase), 2))
    untouched = list(range(1, len(paraphrase), 2))
    sets = {
        "same_question_again": [paraphrase[i] for i in taught],
        "reworded_question": reworded,
        "other_topics": [direct[i] for i in untouched] + [paraphrase[i] for i in untouched],
    }
    out: dict = {"n": {name: len(qs) for name, qs in sets.items()}}

    with store(HANDBOOK) as mol:
        aware = mol.config
        # leak = 1 makes every vote apply to every query: a global chunk weight.
        global_only = replace(mol.config, feedback_global_leak=1.0)
        out["before"] = {name: _hit1_mrr(mol, qs, aware) for name, qs in sets.items()}

        votes = 0
        for i in taught:
            question = paraphrase[i]
            rank, ids = _gold_rank(mol, question, aware)
            if rank is None:
                continue                      # the agent never found the right section
            for _ in range(3):
                mol.feedback(ids[rank - 1], True, question["query"])
                votes += 1
            if rank != 1:                     # the section that wrongly ranked first
                mol.feedback(ids[0], False, question["query"])
                votes += 1
        out["votes_cast"] = votes
        out["query_aware"] = {name: _hit1_mrr(mol, qs, aware) for name, qs in sets.items()}
        out["global_weight"] = {name: _hit1_mrr(mol, qs, global_only) for name, qs in sets.items()}
    return out


# --- bonds ----------------------------------------------------------------------------

def _complete(context: str, answers: list[str]) -> bool:
    return all(norm(a) in norm(context) for a in answers)


def bench_bonds() -> dict:
    """Two-part questions: is the material for *both* parts delivered?"""
    queries = load("handbook_multihop")
    out: dict = {"n": len(queries), "budgets": {}}

    def incomplete(contexts: list[str]) -> list[str]:
        return [q["query"] for q, text in zip(queries, contexts)
                if not _complete(text, q["answers"])]

    for budget in (400, 600, 1200):
        row: dict = {}
        with store(HANDBOOK) as mol:
            no_bonds = replace(mol.config, bond_min_strength=1.01)   # nothing clears the bar
            row["no_bonds"] = incomplete([
                build_context(q["query"], mol._conn, no_bonds, budget)["context"]
                for q in queries
            ])
            row["reference_bonds"] = incomplete(
                [mol.context(q["query"], budget)["context"] for q in queries]
            )
            # An agent that needed both sections fetches them together a few times.
            bodies = mol._conn.execute("SELECT chunk_id, content FROM chunks").fetchall()
            for q in queries:
                pair = [r["chunk_id"] for answer in q["answers"] for r in bodies
                        if norm(answer) in norm(r["content"])]
                for _ in range(3):
                    mol.fetch(pair)
            row["after_cofetch_learning"] = incomplete(
                [mol.context(q["query"], budget)["context"] for q in queries]
            )
        out["budgets"][budget] = {
            name: {"complete": len(queries) - len(missing), "incomplete": missing}
            for name, missing in row.items()
        }
    return out


# --- token savings --------------------------------------------------------------------

def bench_savings() -> dict:
    """Ten questions in one session, three ways of getting the context."""
    queries = (load("handbook_direct")[::10] + load("handbook_paraphrase")[5::10])[:10]
    out: dict = {"questions": len(queries), "workflows": {}}

    def tokens(payload) -> int:
        return count_tokens(json.dumps(payload, ensure_ascii=False))

    with store(HANDBOOK) as mol:
        full_file = mol.sources()[0]["total_tokens"]
        out["full_file_tokens"] = full_file

        # A. Step by step, the way 0.1 worked: full index, search top 5, fetch top 1.
        used = tokens(mol.index())
        answered = 0
        for q in queries:
            results = mol.search(q["query"], k=5)
            used += tokens(results)
            top = next((r for r in results if r["type"] == "chunk"), None)
            if top:
                chunk = mol.fetch([top["id"]])[0]
                used += chunk["token_count"]
                answered += norm(q["answer"]) in norm(chunk["content"])
        out["workflows"]["index + search + fetch top-1 (full index)"] = {
            "tokens": used, "answered": answered}

        # B. Same steps with the budget-fit index.
        used = tokens(mol.index(max_tokens=mol.config.index_token_budget))
        answered = 0
        for q in queries:
            results = mol.search(q["query"], k=3)
            used += tokens(results)
            top = next((r for r in results if r["type"] == "chunk"), None)
            if top:
                chunk = mol.fetch([top["id"]])[0]
                used += chunk["token_count"]
                answered += norm(q["answer"]) in norm(chunk["content"])
        out["workflows"]["budget-fit index + search + fetch top-1"] = {
            "tokens": used, "answered": answered}

        # C. One call per question.
        for budget in (300, 600):
            used = answered = 0
            for q in queries:
                result = mol.context(q["query"], budget)
                used += result["tokens_used"]
                answered += norm(q["answer"]) in norm(result["context"])
            out["workflows"][f"mol_context(budget={budget})"] = {
                "tokens": used, "answered": answered}

    for row in out["workflows"].values():
        row["vs_full_file_pct"] = pct(1 - row["tokens"] / full_file)
    return out


# --- speed ----------------------------------------------------------------------------

def _percentiles(samples: list[float]) -> dict:
    ordered = sorted(samples)
    pick = lambda p: ordered[min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))]  # noqa: E731
    return {"p50_ms": round(pick(50) * 1000, 2), "p95_ms": round(pick(95) * 1000, 2)}


def bench_speed() -> dict:
    import benchmark_5k

    out: dict = {}
    rng = random.Random(benchmark_5k.SEED)
    doc = benchmark_5k.synth_doc(rng, benchmark_5k.TARGET_LINES)
    tmp = Path(tempfile.mkdtemp(prefix="molecule_speed_"))
    try:
        path = tmp / "big.md"
        path.write_text(doc, encoding="utf-8")
        cfg = MoleculeConfig(db_path=tmp / ".molecule" / "store.db", auto_sync=False)
        with Molecule(config=cfg) as mol:
            mol.ingest(HANDBOOK)            # warm the model so its load is not measured
            start = time.perf_counter()
            result = mol.ingest(path)
            wall = time.perf_counter() - start
            out["ingest_5k_lines"] = {
                "lines": doc.count("\n") + 1, "chunks": result["total"],
                "wall_s": round(wall, 2), "embed_s": result["timing"]["embed_s"],
            }
            path.write_text(doc.replace("# 1. Top-Level Area", "# 1. Top-Level Area (edited)", 1),
                            encoding="utf-8")
            start = time.perf_counter()
            again = mol.ingest(path)
            out["incremental_reingest"] = {
                "wall_s": round(time.perf_counter() - start, 3),
                "re_embedded": again["new"] + again["updated"], "skipped": again["skipped"],
            }
            start = time.perf_counter()
            mol.ingest(path)
            out["unchanged_reingest_s"] = round(time.perf_counter() - start, 4)

            queries = [q["query"] for name in ("handbook_direct", "handbook_paraphrase",
                                               "eval_queries", "eval_queries_adversarial")
                       for q in load(name)]
            mol.search("warm up the caches")
            search, context = [], []
            for q in queries:                   # every query is new: embedding included
                start = time.perf_counter()
                mol.search(q, k=5)
                search.append(time.perf_counter() - start)
            for q in queries:                   # query vectors now cached
                start = time.perf_counter()
                mol.context(q + " please", budget=600)
                context.append(time.perf_counter() - start)
            chunks = mol.health()["chunks_indexed"]
            out["search"] = {"chunks": chunks, "queries": len(queries), **_percentiles(search)}
            out["context"] = {"chunks": chunks, "queries": len(queries), **_percentiles(context)}
            out["store_bytes"] = cfg.db_path.stat().st_size
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


EXPERIMENTS = {
    "routing": bench_routing,
    "ablation": bench_ablation,
    "abstention": bench_abstention,
    "context": bench_context,
    "feedback": bench_feedback,
    "bonds": bench_bonds,
    "savings": bench_savings,
    "speed": bench_speed,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Molecule's benchmarks.")
    parser.add_argument("experiments", nargs="*",
                        help=f"which to run (default: all): {', '.join(EXPERIMENTS)}")
    args = parser.parse_args()
    unknown = [name for name in args.experiments if name not in EXPERIMENTS]
    if unknown:
        parser.error(f"unknown experiment(s): {', '.join(unknown)}")
    selected = args.experiments or list(EXPERIMENTS)

    results_path = BENCH / "results" / "latest.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
    results["version"] = __version__
    for name in selected:
        start = time.perf_counter()
        print(f"[{name}] running…", flush=True)
        results[name] = EXPERIMENTS[name]()
        print(f"[{name}] done in {time.perf_counter() - start:.1f}s", flush=True)
        print(json.dumps(results[name], indent=2, ensure_ascii=False)[:3000], flush=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")

    from report import write_report

    write_report(results, BENCH / "RESULTS.md")
    print(f"\nwrote {results_path} and {BENCH / 'RESULTS.md'}")


if __name__ == "__main__":
    sys.path.insert(0, str(BENCH))
    main()
