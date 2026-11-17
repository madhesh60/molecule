"""demo.py — a two-minute tour of Molecule, using this repo's own AGENTS.md.

    python demo.py          (or: make demo)

  1. Ingest a messy doc            → labelled chunks + cross-reference bonds
  2. A table of contents that fits → any budget, any corpus size
  3. One call, ready-to-paste      → mol_context never exceeds its budget
  4. Honest "I don't know"         → off-topic questions are flagged
  5. Feedback that stays on topic  → helps similar questions, leaves others alone
  6. Memory across sessions        → extracted from a transcript, recalled by a new process
  7. Always fresh                  → edit the file, the next answer already reflects it
  8. The number that matters       → every returned token counted

Everything runs locally in a throw-away folder; nothing leaves the machine.
"""
from __future__ import annotations

import io
import os
import shutil
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from molecule import Molecule  # noqa: E402
from molecule.core.retriever import load_router_index, score_chunks  # noqa: E402

BOLD, GREEN, CYAN, DIM, RESET = "\033[1m", "\033[92m", "\033[96m", "\033[2m", "\033[0m"
DEMO_ROOT = ROOT / ".molecule_demo_tmp"


def banner(step: str, title: str) -> None:
    print(f"\n{CYAN}{BOLD}{'━' * 66}\n  {step} — {title}\n{'━' * 66}{RESET}")


def say(text: str = "") -> None:
    print(f"  {text}")


def win(text: str) -> None:
    print(f"  {GREEN}★ {text}{RESET}")


def short(chunk_id: str) -> str:
    return chunk_id.rsplit(":", 1)[-1]


def learned_boost(mol: Molecule, query: str, chunk_id: str) -> str:
    """Peek at the router's learned weight for one chunk and one question.
    (A diagnostic read, so it does not go through the token-counted API.)"""
    scores = score_chunks(query, mol._conn, mol.config)
    return f"{scores.learned[scores.corpus.row_of[chunk_id]] - 1.0:+.2f}"


if DEMO_ROOT.exists():
    shutil.rmtree(DEMO_ROOT)
DEMO_ROOT.mkdir()
doc = DEMO_ROOT / "AGENTS.md"
shutil.copy(ROOT / "AGENTS.md", doc)          # a copy, so step 7 can edit it safely
mol = Molecule(root=DEMO_ROOT)

# ── 1 ───────────────────────────────────────────────────────────────────────
banner("STEP 1", "Point Molecule at a messy document")
start = time.perf_counter()
result = mol.ingest(doc)
file_tokens = mol.sources()[0]["total_tokens"]
say(f"AGENTS.md ({file_tokens} tokens) → {result['total']} chunks, "
    f"{result['bonds']} cross-reference bonds, in {time.perf_counter() - start:.1f}s")
say(f"{DIM}(includes loading the embedding model once: {result['timing']['model_load_s']}s){RESET}")
win("No hand-written skill files: chunking, labelling and linking are automatic.")

# ── 2 ───────────────────────────────────────────────────────────────────────
banner("STEP 2", "A table of contents that always fits")
roomy = load_router_index(mol._conn, max_tokens=2000)      # just to show the shape
say(f"budget 2000 tokens → {'summaries' if 'summary' in roomy[0] else 'headings'}")
index = mol.index(max_tokens=300)                          # what an agent would load
say(f"budget  300 tokens → {'summaries' if 'summary' in index[0] else 'headings + sizes'}")
win("Same call on a 50-chunk file or a 5,000-chunk docs folder: it degrades, never overflows.")

# ── 3 ───────────────────────────────────────────────────────────────────────
banner("STEP 3", "Ask once, get exactly enough context")
query = "how does the chunker deal with a section that is too big?"
ctx = mol.context(query, budget=350)
say(f"{BOLD}Q: {query}{RESET}")
for line in ctx["context"].splitlines()[:6]:
    say(f"{DIM}│ {line[:72]}{RESET}")
say(f"{DIM}│ …{RESET}")
say(f"{ctx['tokens_used']} tokens used of a 350 budget · confidence: {ctx['confidence']}")
for item in ctx["items"]:
    say(f"  • {short(item['id'])}  ({item['tokens']} tok)  {DIM}{item['reason']}{RESET}")
win(f"{ctx['tokens_used']} tokens instead of {file_tokens}: one call replaces index → search → fetch.")

# ── 4 ───────────────────────────────────────────────────────────────────────
banner("STEP 4", "Knowing when the docs do not cover something")
for question in ("what is the recipe for banana bread?", "how are duplicate memories avoided?"):
    top = mol.search(question, k=1)[0]
    say(f"{question:<42} → relevance: {BOLD}{top['relevance']}{RESET}")
win("A 'low' label means: do not trust this match — no made-up context.")

# ── 5 ───────────────────────────────────────────────────────────────────────
banner("STEP 5", "Feedback that helps the right questions only")
asked = "which tools can an agent call over MCP?"
target = next(r["id"] for r in mol.search(asked, k=5) if "MCP Tool Interface" in r["id"])
for _ in range(3):
    mol.feedback(target, useful=True, query=asked)
say(f"3 × useful on '{short(target)}' for: {asked!r}")
for label, question in (("same need, new words", "what functions does the MCP server expose?"),
                        ("unrelated question", "what are the known risks of the project?")):
    say(f"  {label:<22} learned boost {BOLD}{learned_boost(mol, question, target)}{RESET}")
win("Each vote remembers its question, so one topic's lessons never bend another's ranking.")

# ── 6 ───────────────────────────────────────────────────────────────────────
banner("STEP 6", "Memory that writes itself and survives a restart")
transcript = (
    "Long session today. We decided to always deploy through Docker Compose. "
    "I prefer small pull requests, keep diffs under 300 lines. "
    "We are currently debugging the flaky webhook test. "
    "Anyway, lunch ran long."
)
extracted = mol.extract(transcript, source_task="demo")
for fact in extracted["facts"]:
    kind = "expires in 7 days" if fact["type"] == "task_state" else "kept"
    say(f"[{fact['confidence']}] ({kind}) {fact['text']}")
mol.close()
mol = Molecule(root=DEMO_ROOT)                  # a brand-new instance, same local file
answer = mol.recall("how should we ship to production?")[0]
say(f"{BOLD}New session — how should we ship?{RESET} → {answer['text']}")
win("No LLM call to extract, no server to store: one SQLite file.")

# ── 7 ───────────────────────────────────────────────────────────────────────
banner("STEP 7", "Edit the document — the next answer already knows")
text = doc.read_text(encoding="utf-8")
doc.write_text(text.replace("(default 400)", "(default 900)"), encoding="utf-8")
stat = doc.stat()
os.utime(doc, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000))
mol._last_sync_check = 0.0                      # skip the 5-second throttle for the demo
fresh = mol.context("what is the default MAX_CHUNK_TOKENS?", budget=300)["context"]
say("changed '(default 400)' to '(default 900)' in the file, asked again…")
say(f"context now says 900: {BOLD}{'(default 900)' in fresh}{RESET}")
win("Only the edited section was re-embedded; no manual re-ingest.")

# ── 8 ───────────────────────────────────────────────────────────────────────
banner("FINALE", "The number that matters — counted honestly")
stats = mol.stats()
say(f"tokens returned this session: {stats['tokens_used_this_session']} "
    f"(index, searches, contexts, memories — all of it)")
say(f"loading the file instead:     {stats['tokens_if_full_file_loaded']}")
say(f"{BOLD}saved: {GREEN}{stats['savings_pct']}%{RESET}")
say(f"{DIM}breakdown: { {k: v['tokens'] for k, v in stats['by_kind'].items()} }{RESET}")
mol.close()
print(f"\n  {DIM}(demo store in {DEMO_ROOT} — safe to delete){RESET}\n")
