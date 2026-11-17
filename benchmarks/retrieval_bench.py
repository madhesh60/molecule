"""retrieval_bench.py — routing accuracy on every labeled query set.

Deliberately uses only the stable public API (`Molecule.ingest/search/fetch`)
so the *same script* can score any version of Molecule — that is how the
v0.1 baseline in `benchmarks/results/` was produced (run from a worktree of
the old commit).

Two kinds of gold label:

  * `expected_chunk_id` — hand-picked chunk id (AGENTS.md sets).
  * `answer`            — a phrase that occurs in exactly one section of the
                          corpus (handbook sets). A result is a hit when the
                          returned chunk's body contains the phrase. This is
                          chunker-agnostic: it stays fair when the chunker
                          changes how a document is cut.

Metrics: Hit@1, Hit@3 and MRR@10.

Run:  python benchmarks/retrieval_bench.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks"

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text)


def _id_matches(returned_id: str, expected_id: str) -> bool:
    return returned_id == expected_id or returned_id.startswith(expected_id + " §")


def _rank_of_gold(mol, query: dict, k: int) -> int | None:
    """1-based rank of the first gold chunk in the top-k, or None."""
    results = [r for r in mol.search(query["query"], k=k) if r.get("type") == "chunk"]
    ids = [r["id"] for r in results]
    if "answer" in query:
        bodies = {c["chunk_id"]: _norm(c["content"]) for c in mol.fetch(ids)}
        needle = _norm(query["answer"])
        for rank, cid in enumerate(ids, start=1):
            if needle in bodies.get(cid, ""):
                return rank
        return None
    for rank, cid in enumerate(ids, start=1):
        if _id_matches(cid, query["expected_chunk_id"]):
            return rank
    return None


def score_set(mol, queries: list[dict], k: int = 10) -> dict:
    ranks = [_rank_of_gold(mol, q, k) for q in queries]
    n = len(queries)
    return {
        "n": n,
        "hit_at_1": round(sum(1 for r in ranks if r == 1) / n, 4),
        "hit_at_3": round(sum(1 for r in ranks if r and r <= 3) / n, 4),
        "mrr_at_10": round(sum(1 / r for r in ranks if r) / n, 4),
        "misses_at_3": [q["query"] for q, r in zip(queries, ranks) if not r or r > 3],
    }


def _load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


SETS = {
    # name: (document to ingest, query file)
    "agents_main": (ROOT / "AGENTS.md", ROOT / "eval_queries.json"),
    "agents_adversarial": (ROOT / "AGENTS.md", ROOT / "eval_queries_adversarial.json"),
    "handbook_direct": (
        BENCH / "corpus" / "handbook.md", BENCH / "queries" / "handbook_direct.json"),
    "handbook_paraphrase": (
        BENCH / "corpus" / "handbook.md", BENCH / "queries" / "handbook_paraphrase.json"),
}


def run(src_override: str | None = None) -> dict:
    if src_override:
        sys.path.insert(0, src_override)
    from molecule import Molecule

    out: dict[str, dict] = {}
    by_doc: dict[Path, list[str]] = {}
    for name, (doc, _) in SETS.items():
        by_doc.setdefault(doc, []).append(name)

    for doc, names in by_doc.items():
        tmp = Path(tempfile.mkdtemp(prefix="molecule_bench_"))
        try:
            mol = Molecule(root=tmp)
            result = mol.ingest(doc)
            if "error" in result:
                raise SystemExit(f"ingest failed: {result['error']}")
            for name in names:
                out[name] = score_set(mol, _load(SETS[name][1]))
                out[name]["chunks"] = result["total"]
            del mol
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--json", help="write results to this file")
    parser.add_argument("--src", help="score a different checkout's src/ directory")
    args = parser.parse_args()

    results = run(args.src)
    print(f"{'set':<22}{'n':>4}{'Hit@1':>8}{'Hit@3':>8}{'MRR@10':>8}")
    for name, r in results.items():
        print(f"{name:<22}{r['n']:>4}{r['hit_at_1']:>8.3f}{r['hit_at_3']:>8.3f}{r['mrr_at_10']:>8.3f}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
