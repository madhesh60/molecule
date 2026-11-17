"""verify_efficiency.py — check AGENTS.md §15 "Definition of Done", item by item.

    python verify_efficiency.py        (exit code 0 = every item passes)

Each check runs for real (real model, real files, real processes) and prints
PASS or FAIL with the measured value. For the full benchmark tables see
benchmarks/RESULTS.md (python benchmarks/run_benchmarks.py).
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(__file__).parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

from molecule import Molecule, MoleculeConfig  # noqa: E402

GREEN, RED, BOLD, DIM, RESET = "\033[92m", "\033[91m", "\033[1m", "\033[2m", "\033[0m"
results: list[bool] = []


def check(name: str, passed: bool, detail: str) -> None:
    results.append(passed)
    mark = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    print(f"  [{mark}] {name}\n         {DIM}{detail}{RESET}")


def fresh_store(tmp: Path) -> Molecule:
    return Molecule(config=MoleculeConfig(db_path=tmp / ".molecule" / "store.db",
                                          auto_sync=False))


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


work = Path(tempfile.mkdtemp(prefix="molecule_verify_"))
env = {**os.environ, "PYTHONPATH": str(SRC), "MOLECULE_ROOT": str(work)}
print(f"\n{BOLD}Molecule — Definition of Done (AGENTS.md §15){RESET}\n")

try:
    # 1. A real 5,000-line markdown file → sane index in under 5 seconds.
    import benchmark_5k

    doc = benchmark_5k.synth_doc(random.Random(benchmark_5k.SEED), benchmark_5k.TARGET_LINES)
    big = work / "big.md"
    big.write_text(doc, encoding="utf-8")
    with fresh_store(work / "ingest") as mol:
        mol.ingest(ROOT / "AGENTS.md")                 # load the model outside the timing
        start = time.perf_counter()
        result = mol.ingest(big)
        seconds = time.perf_counter() - start
    check("Ingest a 5,000-line file in under 5 s", seconds < 5.0,
          f"{doc.count(chr(10)) + 1} lines → {result['total']} chunks in {seconds:.2f} s")

    # 2. Router ≥ 80 % top-3 on the labeled sets.
    sets = [
        ("AGENTS.md, 20-query set", ROOT / "AGENTS.md", ROOT / "eval_queries.json"),
        ("AGENTS.md, paraphrased", ROOT / "AGENTS.md", ROOT / "eval_queries_adversarial.json"),
        ("held-out handbook, paraphrased", ROOT / "benchmarks/corpus/handbook.md",
         ROOT / "benchmarks/queries/handbook_paraphrase.json"),
        ("blind firmware notes, paraphrased", ROOT / "benchmarks/corpus/blind_firmware.md",
         ROOT / "benchmarks/queries/blind_paraphrase.json"),
    ]
    for label, document, queries in sets:
        with fresh_store(work / label.replace(" ", "_").replace(",", "")) as mol:
            mol.ingest(document)
            ev = mol.evaluate(load(queries), k=3)
        check(f"Top-3 routing ≥ 80 % — {label}", ev["passes"],
              f"{ev['hits']}/{ev['total']} = {100 * ev['top3_precision']:.1f} %")

    # 3. mol_stats is live, accurate and not cherry-picked.
    with fresh_store(work / "stats") as mol:
        mol.ingest(ROOT / "AGENTS.md")
        index = mol.index(max_tokens=300)
        hits = mol.search("how are memories deduplicated", k=3)
        body = mol.fetch([hits[0]["id"]])
        ctx = mol.context("what are the known risks", budget=400)
        stats = mol.stats()
    from molecule.core.tokens import count_tokens

    returned = (count_tokens(json.dumps(index, ensure_ascii=False))
                + count_tokens(json.dumps(hits, ensure_ascii=False))
                + body[0]["token_count"] + ctx["tokens_used"])
    expected = round((1 - returned / stats["tokens_if_full_file_loaded"]) * 100, 1)
    check("mol_stats counts every returned token",
          stats["tokens_used_this_session"] == returned and stats["savings_pct"] == expected,
          f"index + search + fetch + context = {returned} tokens vs "
          f"{stats['tokens_if_full_file_loaded']} for the file → {stats['savings_pct']} % saved")

    # 4. The server runs standalone and speaks MCP to a real client.
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    shutil.copy(ROOT / "AGENTS.md", work / "AGENTS.md")

    async def over_the_wire() -> tuple[int, str]:
        params = StdioServerParameters(command=sys.executable,
                                       args=["-m", "molecule", "serve"], env=env, cwd=str(work))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                await session.call_tool("mol_ingest", {"file_path": "AGENTS.md"})
                reply = await session.call_tool(
                    "mol_context", {"query": "how is token savings tracked", "budget": 300})
                return len(tools), json.loads(reply.content[0].text)["context"]

    tool_count, context = asyncio.run(over_the_wire())
    check("MCP server works over stdio with the official client",
          tool_count == 18 and "Telemetry" in context,
          f"{tool_count} tools listed; mol_context answered from '6.5 Token-Savings Telemetry'")

    # 5. Memory survives a full process restart.
    write = ("from molecule import Molecule; "
             "Molecule().remember('Release trains leave on Thursdays', confidence=0.9)")
    read = ("from molecule import Molecule; "
            "print(Molecule().recall('when do releases ship')[0]['text'])")
    subprocess.run([sys.executable, "-c", write], env=env, cwd=work, check=True)
    recalled = subprocess.run([sys.executable, "-c", read], env=env, cwd=work, check=True,
                              capture_output=True, text=True).stdout.strip()
    check("Memory persists across a process restart",
          recalled == "Release trains leave on Thursdays",
          f"written by one process, recalled by another: {recalled!r}")

    # 6. The walkthrough (demo.py) runs end to end.
    demo = subprocess.run([sys.executable, str(ROOT / "demo.py")], env=env, cwd=ROOT,
                          capture_output=True, text=True)
    check("Reference walkthrough (demo.py) runs", demo.returncode == 0,
          "all eight steps completed" if demo.returncode == 0 else demo.stderr[-300:])
finally:
    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(ROOT / ".molecule_demo_tmp", ignore_errors=True)

passed = sum(results)
colour = GREEN if passed == len(results) else RED
print(f"\n  {colour}{BOLD}{passed}/{len(results)} checks passed{RESET}\n")
sys.exit(0 if passed == len(results) else 1)
