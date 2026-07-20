"""demo.py — Molecule hackathon demo.

A terminal narrative that walks through everything Molecule does:

  1. Ingest a big messy doc  →  labeled chunks in SQLite
  2. Cheap router index      →  table of contents, not the textbook
  3. Search + selective fetch →  live token savings
  4. Self-learning router    →  feedback visibly changes ranking
  5. Auto memory extraction  →  memory writes itself from a transcript
  6. Cross-session recall    →  a brand-new process still remembers

Run:  python demo.py   (or: make demo)
"""
from __future__ import annotations

import io
import shutil
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BOLD = "\033[1m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"

ROOT = Path(__file__).parent
DEMO_ROOT = ROOT / ".molecule_demo_tmp"
DOC = ROOT / "AGENTS.md"


def banner(step: str, title: str) -> None:
    print(f"\n{CYAN}{BOLD}{'═' * 62}{RESET}")
    print(f"{CYAN}{BOLD}  {step} — {title}{RESET}")
    print(f"{CYAN}{BOLD}{'═' * 62}{RESET}")


def say(msg: str) -> None:
    print(f"  {msg}")


def win(msg: str) -> None:
    print(f"  {GREEN}★ {msg}{RESET}")


def pause() -> None:
    time.sleep(0.4)


if DEMO_ROOT.exists():
    shutil.rmtree(DEMO_ROOT)

from molecule import Molecule  # noqa: E402  (import after path setup)

banner("STEP 1", "Point Molecule at a messy 300-line doc")
mol = Molecule(root=DEMO_ROOT)
t0 = time.perf_counter()
result = mol.ingest(DOC)
say(f"Ingested {DOC.name}: {result['total']} chunks "
    f"({result['new']} new) in {time.perf_counter() - t0:.1f}s")
timing = result.get("timing", {})
say(f"{DIM}model load {timing.get('model_load_s', 0)}s (one-time) · "
    f"embedding {timing.get('embed_s', 0)}s{RESET}")
win("No hand-written SKILL.md files. Chunking, labeling, embedding — automatic.")
pause()

banner("STEP 2", "The cheap router index (table of contents)")
index = mol.index()
say(f"{len(index)} entries, summaries only — no full bodies loaded")
for entry in index[:5]:
    say(f"{DIM}{entry['chunk_id'][:58]:<58} {entry['token_count']:>5} tok{RESET}")
say(f"{DIM}… and {len(index) - 5} more{RESET}")
win("The agent reads the table of contents, not the whole textbook.")
pause()

banner("STEP 3", "Ask a question → fetch ONE chunk, not the whole file")
query = "How does the chunker split large sections?"
say(f"{BOLD}Q: {query}{RESET}")
results = mol.search(query, k=3)
top = results[0]
say(f"→ {top['id']}")
say(f"{DIM}   why: {top['reason']}{RESET}")
fetched = mol.fetch([top["id"]])
say(f"→ fetched {fetched[0]['token_count']} tokens (the full file is ~4,300)")
stats = mol.stats()
win(f"Live savings so far: {stats['savings_pct']}% of tokens NOT spent.")
pause()

banner("STEP 4", "The router LEARNS — tell it what was useful")
say("Suppose the 2nd result was actually the one that answered the question…")
second = results[1]["id"]
before = {r["id"]: r["score"] for r in mol.search(query, k=3)}
for _ in range(3):
    fb = mol.feedback(second, useful=True, query=query)
say(f"3 × 'useful' votes on {second}")
say(f"learned weight: 1.0 → {fb['learned_weight']}")
after = mol.search(query, k=3)
after_entry = next(r for r in after if r["id"] == second)
say(f"score: {before[second]:.4f} → {after_entry['score']:.4f}")
say(f"{DIM}   reason now: {after_entry['reason']}{RESET}")
win("Feedback is bounded (EMA, clamps 0.5–1.5) — one vote can never dominate.")
pause()

banner("STEP 5", "Memory writes ITSELF — extract facts from a transcript")
transcript = (
    "Long session. We decided to always deploy through Docker Compose. "
    "I prefer smaller pull requests, please keep diffs under 300 lines. "
    "I think maybe we should try Redis for the job queue at some point. "
    "We are currently debugging the flaky webhook test. "
    "Anyway the weather was great and lunch ran long."
)
say(f"{DIM}transcript: \"{transcript[:70]}…\"{RESET}")
er = mol.extract(transcript, source_task="demo_session")
say(f"extracted {er['extracted']} durable facts (no LLM call, no API key):")
for f in er["facts"]:
    tag = "7-day expiry" if f["type"] == "task_state" else "permanent"
    say(f"  [{f['confidence']}] ({tag}) {f['text'][:62]}")
win("Decisions 0.9 confidence · hedges 0.6 · ephemeral facts auto-expire.")
pause()

banner("STEP 6", "Kill the process. Start fresh. It still remembers.")
del mol
mol2 = Molecule(root=DEMO_ROOT)  # brand-new instance, same local DB
recalled = mol2.recall("how should we deploy")
say(f"{BOLD}Q (new session): how should we deploy?{RESET}")
say(f"→ {recalled[0]['text']}")
say(f"{DIM}   why: {recalled[0]['reason']}{RESET}")
learned = mol2.feedback_stats()
say(f"router memory also survived: {learned['votes']} votes, "
    f"{len(learned['learned'])} learned weight(s)")
win("Everything lives in one local SQLite file. No cloud. No API keys.")

banner("FINALE", "The number that matters")
final = mol2.stats()
print(f"""
  {BOLD}Token savings this demo: {GREEN}{final['savings_pct']}%{RESET}
  {DIM}used {final['tokens_used_this_session']} tokens instead of {final['tokens_if_full_file_loaded']}{RESET}

  One MCP server · 16 tools · works in Claude Code, Cursor, LangChain
  {DIM}(temp demo DB at {DEMO_ROOT} — safe to delete){RESET}
""")
