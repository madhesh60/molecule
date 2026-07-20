"""
verify_efficiency.py — End-to-end Molecule verification script.

Exercises the full pipeline using the project's own AGENTS.md:
  1. Ingest & chunking
  2. Router index (cheap table-of-contents)
  3. Search → selective fetch (token savings)
  4. Memory write / recall
  5. Cross-session memory persistence
  6. Token-savings telemetry
"""
from __future__ import annotations

import io
import shutil
import sys
import time
from pathlib import Path

import tiktoken

from molecule import Molecule

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

DIVIDER = "=" * 60
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"

def header(title: str) -> None:
    print(f"\n{CYAN}{BOLD}{DIVIDER}{RESET}")
    print(f"{CYAN}{BOLD}  {title}{RESET}")
    print(f"{CYAN}{BOLD}{DIVIDER}{RESET}")

def ok(msg: str) -> None:
    print(f"  {GREEN}[OK]{RESET} {msg}")

def warn(msg: str) -> None:
    print(f"  {YELLOW}[WARN]{RESET} {msg}")

def fail(msg: str) -> None:
    print(f"  {RED}[FAIL]{RESET} {msg}")


# ── setup ────────────────────────────────────────────────────────────────────
AGENTS_MD = Path(__file__).parent / "AGENTS.md"
TEST_ROOT = Path(__file__).parent / ".molecule_verify_tmp"

# Clean slate
if TEST_ROOT.exists():
    shutil.rmtree(TEST_ROOT)

header("Step 0 — Setup")
mol = Molecule(root=TEST_ROOT)
ok(f"Molecule initialised at {mol.health()['db_path']}")

# ── 1. Ingest ────────────────────────────────────────────────────────────────
header("Step 1 — Ingest AGENTS.md")
t0 = time.perf_counter()
result = mol.ingest(AGENTS_MD)
elapsed = time.perf_counter() - t0

if "error" in result:
    fail(f"Ingest failed: {result['error']}")
    raise SystemExit(1)

ok(f"File: {Path(result['file']).name}")
ok(f"Chunks created: {result['new']} new, {result['updated']} updated, {result['skipped']} skipped")
ok(f"Total chunks: {result['total']}")
timing = result.get("timing", {})
model_load_s = timing.get("model_load_s", 0.0)
embed_s = timing.get("embed_s", 0.0)
ok(f"Wall time: {elapsed:.2f}s (model load: {model_load_s:.2f}s — one-time download/warm-up)")
if embed_s < 5:
    ok(f"Embed time: {embed_s:.2f}s (< 5s target ✓)")
else:
    fail(f"Embed time: {embed_s:.2f}s (> 5s target ✗)")

# ── 2. Router index ─────────────────────────────────────────────────────────
header("Step 2 — Load Router Index (cheap table of contents)")
index = mol.index()

# Count total tokens in index summaries vs full file
enc = tiktoken.get_encoding("cl100k_base")
full_file_tokens = len(enc.encode(AGENTS_MD.read_text(encoding="utf-8")))
index_tokens = sum(len(enc.encode(c["summary"])) for c in index)

ok(f"Index entries: {len(index)}")
ok(f"Index cost: ~{index_tokens} tokens  (full file = {full_file_tokens} tokens)")
ok(f"Index is {(index_tokens / full_file_tokens * 100):.1f}% of the full file")

print(f"\n  {'ID':<50} {'Tokens':>6}  Summary")
print(f"  {'—'*50} {'—'*6}  {'—'*40}")
for c in index[:8]:
    cid = c['chunk_id'][:50]
    print(f"  {cid:<50} {c['token_count']:>6}  {c['summary'][:40]}…")
if len(index) > 8:
    print(f"  … and {len(index) - 8} more")

# ── 3. Search + Fetch ────────────────────────────────────────────────────────
header("Step 3 — Search & Selective Fetch")

# (query, substring the top-hit chunk id should contain)
queries = [
    ("How does the chunker split large sections?", "Auto-Chunker"),
    ("What MCP tools does Molecule expose?", "MCP Tool Interface"),
    ("What is the token savings telemetry?", "Telemetry"),
    ("What are the non-goals of this project?", "Non-goals"),
]

routing_hits = 0
for q, expected_sub in queries:
    print(f"\n  {BOLD}Q: {q}{RESET}")
    results = mol.search(q, k=3)
    if not results:
        warn("  No results")
        continue
    top = results[0]
    hit = any(expected_sub in r["id"] for r in results[:3])
    routing_hits += hit
    fetched = mol.fetch([top["id"]])
    fetched_tokens = fetched[0]["token_count"] if fetched else 0
    print(f"  → Top hit: {top['id']}  {'✓' if hit else '✗ (expected …' + expected_sub + '…)'}")
    print(f"    reason: {top['reason']}")
    print(f"    tokens fetched: {fetched_tokens}  (vs {full_file_tokens} full file)")

print()
if routing_hits == len(queries):
    ok(f"Routing: {routing_hits}/{len(queries)} queries hit the right chunk in top-3")
else:
    warn(f"Routing: {routing_hits}/{len(queries)} queries hit the right chunk in top-3")

# ── 3b. Self-learning router ────────────────────────────────────────────────
header("Step 3b — Self-Learning Router (mol_feedback)")

fb_query = "How does the chunker split large sections?"
before = mol.search(fb_query, k=3)
target_id = before[0]["id"]
before_score = before[0]["score"]

r = mol.feedback(target_id, useful=True, query=fb_query)
ok(f"Voted useful on {target_id}")
ok(f"Learned weight: 1.0 → {r['learned_weight']} (delta {r['delta']:+.4f})")

after = mol.search(fb_query, k=3)
after_top = next((x for x in after if x["id"] == target_id), None)
if after_top and after_top["score"] > before_score:
    ok(f"Score rose: {before_score:.4f} → {after_top['score']:.4f}")
    ok(f"Reason now says: {after_top['reason']}")
else:
    fail("Feedback did not raise the chunk's score")

# ── 3c. Auto memory extraction ──────────────────────────────────────────────
header("Step 3c — Auto Memory Extraction (mem_extract)")

transcript = (
    "We looked at the retriever bug together. "
    "We decided to always use BM25 for keyword scoring. "
    "I think maybe the heading boost could go higher. "
    "We are currently debugging the eval query ids. "
    "The meeting ran long and pizza arrived at noon."
)
er = mol.extract(transcript, source_task="verify_script")
ok(f"Extracted {er['extracted']} fact(s) from a 5-sentence transcript")
for f in er["facts"]:
    ok(f"  [{f['outcome']}] conf={f['confidence']} type={f['type']}: {f['text'][:60]}")
if er["extracted"] != 3:
    warn(f"Expected 3 facts (decision, hedged, ephemeral), got {er['extracted']}")


# ── 4. Memory write + recall ────────────────────────────────────────────────
header("Step 4 — Memory: Write & Recall")

memories_to_store = [
    ("User prefers dark mode for all UI demos", "preference"),
    ("Currently debugging the embedder hash check", "task_state"),
    ("Never deploy to production on Fridays", "preference"),
]

for text, mtype in memories_to_store:
    r = mol.remember(text, source_task="verify_script", confidence=0.9, mem_type=mtype)
    if r.get("stored"):
        ok(f"Stored ({mtype}): {text}")
    else:
        warn(f"Skipped: {r}")

# Recall
print()
recall_results = mol.recall("UI preferences")
if recall_results:
    ok(f"Recall 'UI preferences' → {recall_results[0]['text']}")
    ok(f"  reason: {recall_results[0]['reason']}")
else:
    fail("Recall returned nothing")

recall_results2 = mol.recall("deployment rules")
if recall_results2:
    ok(f"Recall 'deployment rules' → {recall_results2[0]['text']}")
else:
    fail("Recall returned nothing for 'deployment rules'")

# Dedup test
print()
dup = mol.remember("User prefers dark mode for all UI demos", source_task="verify", confidence=0.9)
if dup.get("skipped"):
    ok(f"Dedup works: duplicate detected (similar_to: {dup['similar_to'][:20]}…)")
else:
    warn(f"Dedup: expected skip, got {dup}")

# ── 5. Cross-session persistence ────────────────────────────────────────────
header("Step 5 — Cross-Session Persistence")

# Create a brand-new Molecule instance pointing to same DB
mol2 = Molecule(root=TEST_ROOT)
recall2 = mol2.recall("dark mode")
if recall2:
    ok(f"New session recalled: '{recall2[0]['text']}'")
else:
    fail("Memory did NOT persist across sessions!")

sources2 = mol2.sources()
if sources2:
    ok(f"Chunk index persists: {sources2[0]['chunks']} chunks from {Path(sources2[0]['source_file']).name}")
else:
    fail("Chunk index did NOT persist!")

# ── 6. Token savings ────────────────────────────────────────────────────────
header("Step 6 — Token Savings Telemetry")

stats = mol.stats()
ok(f"Tokens actually used this session: {stats['tokens_used_this_session']}")
ok(f"Tokens if full file loaded each time: {stats['tokens_if_full_file_loaded']}")
pct = stats["savings_pct"]
color = GREEN if pct > 50 else YELLOW if pct > 0 else RED
ok(f"Savings: {color}{BOLD}{pct}%{RESET}")

# ── 7. Health check ──────────────────────────────────────────────────────────
header("Step 7 — Health Check")
h = mol.health()
for k, v in h.items():
    ok(f"{k}: {v}")

# ── Summary ──────────────────────────────────────────────────────────────────
header("Summary")
print(f"""
  Full AGENTS.md tokens:       {full_file_tokens}
  Tokens used (selective):     {stats['tokens_used_this_session']}
  {color}{BOLD}Token savings:               {pct}%{RESET}
  Chunks indexed:              {result['total']}
  Routing accuracy (top-3):    {routing_hits}/{len(queries)}
  Memories stored:             {len(memories_to_store)} + {er['stored']} auto-extracted
  Cross-session recall:        {'✓ works' if recall2 else '✗ broken'}
  Embed time:                  {embed_s:.2f}s (model load {model_load_s:.2f}s one-time)
""")

# Cleanup
print(f"  Temp DB at: {TEST_ROOT / '.molecule' / 'store.db'}")
print(f"  (delete {TEST_ROOT} when done)\n")
