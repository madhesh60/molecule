"""benchmark_5k.py — synthesize a realistic ~5,000-line markdown doc, ingest it
into a FRESH temp Molecule DB, and measure cold-ingest + incremental re-ingest
timing against the AGENTS.md §12 targets (full embed <5s, incremental <1s).

Run:  PYTHONUTF8=1 python scripts/benchmark_5k.py

Does not add dependencies. Deterministic (fixed seed) so runs are comparable.
"""
from __future__ import annotations

import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from molecule import Molecule  # noqa: E402

SEED = 1337
TARGET_LINES = 5000

WORDS = (
    "router chunk memory embedding vector index token session agent context "
    "retrieval ranking similarity heading summary confidence dedup redaction "
    "telemetry savings baseline pipeline latency offline sqlite fallback merge "
    "conflict expiry preference guardrail feedback weight schema ingest cache "
    "boundary sentence overlap threshold cosine metadata payload query cursor "
    "transaction commit rollback batch encode float array blob persist scale"
).split()

CODE_SNIPPETS = [
    'def rank(query: str, k: int = 5) -> list[dict]:\n    """Rank candidate chunks by hybrid score."""\n    scored = [(c, score(query, c)) for c in candidates]\n    return [c for c, _ in sorted(scored, key=lambda x: -x[1])[:k]]',
    'SELECT chunk_id, summary, token_count\nFROM chunks\nWHERE source_file = ?\nORDER BY last_updated DESC\nLIMIT 10;',
    'class Store:\n    def __init__(self, path: Path) -> None:\n        self.conn = sqlite3.connect(path)\n\n    def upsert(self, row: tuple) -> None:\n        self.conn.execute(INSERT_SQL, row)',
    'mol = Molecule(root=".")\nmol.ingest("AGENTS.md")\nhits = mol.search("how does routing work")\nbody = mol.fetch([hits[0]["id"]])',
]


def _sentence(rng: random.Random, n: int) -> str:
    ws = [rng.choice(WORDS) for _ in range(n)]
    ws[0] = ws[0].capitalize()
    return " ".join(ws) + "."


def _paragraph(rng: random.Random) -> str:
    # Shortish paragraphs — a real markdown doc is not wall-to-wall dense prose;
    # it's interleaved with headings, blank lines, lists and code, so tokens/line
    # stays modest (~8 tok/line). This keeps a genuine 5k-line doc near the
    # realistic ~40k-token, ~100-chunk range the §12 <5s target assumes.
    return " ".join(_sentence(rng, rng.randint(6, 11)) for _ in range(rng.randint(2, 3)))


def _section_body(rng: random.Random, lines: list[str]) -> None:
    # Substantial sections (~250-400 tokens each) so a genuine 5k-line doc maps
    # to a realistic ~100-chunk index rather than hundreds of tiny fragments.
    # Chunk count then tracks tokens/~300, matching what the §12 <5s embed
    # target assumes for a real document.
    for _ in range(rng.randint(6, 10)):
        lines.append(_paragraph(rng))
        lines.append("")
    roll = rng.random()
    if roll < 0.3:
        lines.append("```python")
        lines.extend(rng.choice(CODE_SNIPPETS).splitlines())
        lines.append("```")
        lines.append("")
    elif roll < 0.6:
        for _ in range(rng.randint(4, 8)):
            lines.append(f"- {rng.choice(WORDS)}: {_sentence(rng, rng.randint(4, 9))}")
        lines.append("")


def synth_doc(rng: random.Random, target_lines: int) -> str:
    lines: list[str] = ["# Synthetic Benchmark Document", ""]
    h1 = 0
    while len(lines) < target_lines:
        h1 += 1
        lines.append(f"# {h1}. Top-Level Area {rng.choice(WORDS).capitalize()}")
        lines.append("")
        _section_body(rng, lines)
        for h2 in range(1, rng.randint(2, 3)):
            lines.append(f"## {h1}.{h2} {rng.choice(WORDS).capitalize()} {rng.choice(WORDS)}")
            lines.append("")
            _section_body(rng, lines)
    return "\n".join(lines)


def _tokenish_count(text: str) -> int:
    # Rough token estimate without pulling in a tokenizer dependency.
    return len(text.split())


def main() -> None:
    rng = random.Random(SEED)
    tmpdir = Path(tempfile.mkdtemp(prefix="molecule_bench_"))
    try:
        doc = synth_doc(rng, TARGET_LINES)
        doc_path = tmpdir / "big.md"
        doc_path.write_text(doc, encoding="utf-8")
        n_lines = doc.count("\n") + 1
        n_tokens = _tokenish_count(doc)

        mol = Molecule(root=tmpdir)

        # (c) Cold full ingest. model_load_s is separated out by the ingest result.
        t0 = time.perf_counter()
        res = mol.ingest(doc_path)
        full_wall = time.perf_counter() - t0
        embed_s = res["timing"]["embed_s"]
        model_load_s = res["timing"]["model_load_s"]
        chunk_count = res["total"]

        # (d) Modify ONE section, then incremental re-ingest (unchanged sections skipped).
        modified = doc.replace(
            "# 1. Top-Level Area",
            "# 1. Top-Level Area (edited " + _sentence(rng, 8) + ")",
            1,
        )
        doc_path.write_text(modified, encoding="utf-8")
        t0 = time.perf_counter()
        inc_res = mol.ingest(doc_path)
        inc_wall = time.perf_counter() - t0
        inc_embed_s = inc_res["timing"]["embed_s"]

        # (e) Report.
        print("=" * 60)
        print("Molecule 5k-line ingest benchmark")
        print("=" * 60)
        print(f"lines                     : {n_lines}")
        print(f"tokens (whitespace approx): {n_tokens}")
        print(f"chunks                    : {chunk_count}")
        print("-" * 60)
        print(f"full ingest wall time     : {full_wall:.3f}s")
        print(f"  model load (excluded)   : {model_load_s:.3f}s")
        print(f"  embed only              : {embed_s:.3f}s   (target <5s)")
        print(f"full ingest minus load    : {full_wall - model_load_s:.3f}s")
        print("-" * 60)
        print(f"incremental re-ingest     : {inc_wall:.3f}s   (target <1s)")
        print(f"  incremental embed only  : {inc_embed_s:.3f}s")
        print(f"  changed: {inc_res['new']} new, {inc_res['updated']} updated, "
              f"{inc_res['skipped']} skipped")
        print("-" * 60)
        embed_ok = "PASS" if embed_s < 5 else "FAIL"
        inc_ok = "PASS" if inc_wall < 1 else "FAIL"
        print(f"embed <5s      : {embed_ok}")
        print(f"incremental <1s: {inc_ok}")
        print("=" * 60)
    finally:
        del mol
        # Windows holds the sqlite handle briefly; retry cleanup.
        for _ in range(5):
            try:
                shutil.rmtree(tmpdir)
                break
            except PermissionError:
                time.sleep(0.3)


if __name__ == "__main__":
    main()
