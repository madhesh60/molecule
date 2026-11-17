# CLAUDE.md — Molecule Working Workflow

> **This file is the operational context for developing Molecule using Claude Code.**
> Read `AGENTS.md` for the full spec. This file focuses on *how the system works in practice*
> so you (Claude) can reason about Molecule while helping me build it.

---

## How Molecule Works — End-to-End Workflow

```
┌───────────────────────┐      ┌──────────────────────────────┐      ┌─────────────────────────────────┐
│  DEVELOPER'S PROJECT  │      │  AGENT (Claude Code / CLI /  │      │         MOLECULE                │
│  (client-side folder) │─────▶│  LangChain / CrewAI)         │─────▶│  (context layer + memory)       │
│                       │      │                              │      │                                 │
│  • AGENTS.md          │      │  Uses Molecule as its        │      │  • Auto-chunks AGENTS.md        │
│  • CRAWL.md           │      │    context layer module       │      │  • Routes to relevant chunks    │
│  • docs/*.md          │      │  • reads only what's needed  │      │  • Saves facts across sessions  │
│  • preferences/rules  │      │  • remembers across sessions │      │  • Tracks token savings         │
│  • code files         │      │  • learns from feedback      │      │  • Self-learns from feedback    │
└───────────────────────┘      └──────────────────────────────┘      └─────────────────────────────────┘
```

### The Three Actors

1. **Developer's project folder** — a normal repo. Contains `AGENTS.md` (or any big messy
   context file). The developer writes their rules, preferences, and project knowledge here.
   Think of this like a local Antigravity-style workspace where the dev codes with their
   context docs and tells the agent what they care about.

2. **The Agent** — Claude Code, Cursor, a LangChain agent, a CrewAI agent, or a simple
   CLI script. The agent talks to the developer and needs project context to do its job.
   Instead of reading the entire `AGENTS.md` every turn, it calls Molecule
   (usually one `mol_context` call).

3. **Molecule** — the context layer between the agent and the docs. It auto-chunks
   `AGENTS.md` into labeled sections, routes the agent to only the relevant chunk(s) for
   the current task, and remembers small facts across sessions. All local, all SQLite,
   no cloud.

---

## The Fast Path (one call)

```
Agent calls:  mol_context("how does the chunker handle large sections?", budget=600)
Returns:      ready-to-paste text of only the sections that answer it, never over
              600 tokens, plus items[] (id, tokens, reason), omitted[] and a
              confidence label ("low" = the docs probably do not cover this)
```

This replaces Steps 1–3 below for most agents. The step-by-step tools stay for
agents that want control.

## The Actual Flow (Step by Step)

### Step 0: One-Time Setup

```python
# Developer installs molecule, then:
from molecule import Molecule
mol = Molecule()              # creates .molecule/store.db in the project root
mol.ingest("AGENTS.md")       # auto-chunks, labels, embeds — done once
```

Or via MCP server for Claude Code / Cursor:
```json
{
  "mcpServers": {
    "molecule": {
      "command": "molecule",
      "args": ["serve"],
      "cwd": "/path/to/repo"
    }
  }
}
```
Then the agent calls `mol_ingest("AGENTS.md")` once.

### Step 1: Agent Starts a Session → Loads the Cheap Index

The agent does NOT read the full file. Instead:

```
Agent calls: mol_load_index()            # default budget: 400 tokens
Returns:     chunk IDs + one-line summaries, or — when that would not fit — a
             nested table of contents (headings + token counts) that does
             (the "table of contents", not the "textbook")
```

Near-instant, and it always fits its budget, however large the docs.

### Step 2: Developer Asks a Question → Agent Searches

```
Developer:   "How does the chunker handle large sections?"
Agent calls:  mol_search("how does chunker handle large sections", k=3)
Returns:      Top 3 candidate chunk IDs + summaries + reason why each matched
              (hybrid: (0.6·dense + 0.6·coverage-BM25 + 0.15·heading) × learned_weight)
```

Still cheap — returns summaries only, not full content.

### Step 3: Agent Fetches Only What It Needs

```
Agent calls:  mol_fetch(["AGENTS.md::AGENTS.md — Project: Molecule:6. Core modules:6.1 Auto-Chunker"])
Returns:      Full body text of just that one chunk (~200 tokens) + `related` chunks
              Instead of the entire AGENTS.md (~4,700 tokens)
```

**This is the key savings moment.** Telemetry counts everything returned —
the index and the search results too — against the whole file, once.

### Step 4: Agent Does Its Work, Then Teaches the Router

```
Agent calls:  mol_feedback("AGENTS.md::Auto-Chunker", useful=True, query="how does chunker...")
Effect:       the vote is stored with the question's vector (bounded EMA, 0.5–1.5)
              Similar questions rank this chunk higher; unrelated ones are unaffected
```

### Step 5: Agent Remembers Facts for Next Session

```
Agent calls:  mem_write("User prefers smaller PRs, keep diffs under 300 lines",
                        source_task="code_review", confidence=0.9)
Or auto:      mem_extract(transcript_text, source_task="session")
              → Heuristically pulls up to 5 durable facts, no LLM call
              → Direct statements = 0.9 confidence, hedges = 0.6, ephemeral = 7-day expiry
```

### Step 6: Next Session → Agent Recalls Without Being Re-Told

```
# New process, new session, same .molecule/store.db
mol2 = Molecule()
mol2.recall("deployment preferences")
→ "Always deploy through Docker Compose" (confidence: 0.9, reason: "similarity: 0.62, keyword: 0.41")
```

### Step 7: Check the Savings

```
Agent calls:  mol_stats()
Returns:      {"tokens_used_this_session": 713, "tokens_if_full_file_loaded": 4747,
               "savings_pct": 85.0, "by_kind": {"index": …, "search": …, "fetch": …, "context": …},
               "lifetime": {…}}
              (measured by verify_efficiency.py: index + search + fetch + one context)
```

---

## Key Principle: Read the Table of Contents, Not the Textbook

```
WITHOUT Molecule:  Agent reads ~4,700 tokens of AGENTS.md every single turn
                   Most of it is irrelevant to the current task
                   No memory across sessions

WITH Molecule:     Agent makes one mol_context call: ~200–400 tokens of exactly
                   the sections that answer the question (~90–96% fewer tokens per question,
                   every returned token counted)
                   Says "low confidence" instead of guessing when docs don't cover it
                   Remembers decisions, preferences, corrections across sessions
                   Router gets smarter over time via feedback — on the right questions
```

---

## Project Structure (for development reference)

```
molecule/
├── AGENTS.md                    # Full project spec (the "bible")
├── CLAUDE.md                    # This file — working workflow context
├── INTEGRATION.md               # Integration examples (LangChain, CrewAI, etc.)
├── README.md                    # Public-facing readme
├── CHANGELOG.md                 # What changed per release
├── docs/ARCHITECTURE.md         # Design decisions + the experiments behind them
├── demo.py                      # Two-minute terminal tour
├── verify_efficiency.py         # Definition of Done (AGENTS.md §15), pass/fail
├── eval_queries*.json           # Labeled queries for AGENTS.md
├── benchmarks/                  # corpora, labeled queries, run_benchmarks.py, RESULTS.md
├── src/molecule/
│   ├── engine.py                # Molecule — the public API (lock, freshness, accounting)
│   ├── server.py                # MCP server (18 thin tools; mcp 1.x and 2.x)
│   ├── cli.py                   # CLI: molecule ingest/context/search/doctor/...
│   ├── config.py                # MoleculeConfig: defaults < config.json < MOLECULE_* env
│   ├── db.py                    # SQLite connections, migrations, generation triggers
│   └── core/
│       ├── markdown.py          # Fence-aware block / sentence / list / table splitting
│       ├── chunker.py           # Headings → sections → block-aware split → merge
│       ├── labeler.py           # One-line labels from structure (no LLM)
│       ├── normalize.py         # Tokenizer: stopwords + light stemmer
│       ├── expansion.py         # Concept-cluster synonyms
│       ├── scoring.py           # Coverage-normalised BM25, heading match, compounds
│       ├── embedder.py          # ONNX / torch / hashing backends, model-tagged vectors
│       ├── ingest.py            # Atomic incremental ingest, sources, directories
│       ├── retriever.py         # Hybrid ranking, relevance labels, budget-fit index
│       ├── context.py           # Budgeted context compiler (mol_context)
│       ├── bonds.py             # Cross-reference + Hebbian co-fetch links
│       ├── feedback.py          # Query-aware bounded-EMA learning
│       ├── memory.py            # Redact → dedup → supersede → expire → evict
│       ├── extractor.py         # mem_extract — heuristic fact extraction
│       ├── telemetry.py         # Honest token accounting
│       ├── sync.py              # Freshness: re-ingest edited files
│       ├── snapshot.py          # Export / import / backup
│       ├── doctor.py            # Health checks with fixes
│       ├── metrics.py           # Latency percentiles per operation
│       ├── cache.py             # Generation-keyed caches for the hot path
│       ├── tokens.py            # Offline-safe token counting
│       └── eval.py              # Precision, Hit@1, MRR, nDCG, ablations
├── tests/                       # 391 tests (hermetic by default)
└── .molecule/                   # Local DB store (like .git — invisible until you look)
    └── store.db                 # SQLite: chunks, memories, feedback, bonds, telemetry
```

---

## MCP Tools (the agent's interface)

| Tool | Purpose | Cost |
|---|---|---|
| `mol_context(query, budget)` | **The fast path**: compiled context that fits the budget | One call |
| `mol_load_index(max_tokens, prefix)` | Table of contents that always fits the budget | ~60–400 tokens |
| `mol_search(query, k)` | Ranked ids + summaries + reason + relevance + related | Summaries only |
| `mol_fetch(chunk_ids)` | Full body text for specific chunks | Targeted |
| `mol_ingest(path)` | Chunk + index a file or a folder | One-time, incremental |
| `mol_sync(prune)` | Re-index sources edited on disk (also automatic) | Incremental |
| `mol_feedback(chunk_id, useful, query)` | Teach the router (applies to similar queries) | Free |
| `mol_feedback_stats()` | Learned weights + strongest learned bonds | Free |
| `mem_write(text, ...)` | Store a durable fact | Free |
| `mem_search(query, k)` | Retrieve relevant memories | Cheap |
| `mem_extract(transcript)` | Auto-extract facts from a conversation | No LLM call |
| `mol_forget(memory_id)` | Delete a memory | Free |
| `mem_merge_duplicates()` | Merge near-duplicate memories | Periodic cleanup |
| `mol_list_sources()` / `mol_remove_source(file)` | Manage indexed files | Free |
| `mol_eval(queries, k)` | Precision / Hit@1 / MRR on labeled queries | Dev-time only |
| `mol_health()` | Status, counts, stale sources, latency | Free |
| `mol_stats()` | Token savings — session, lifetime, by kind | Free |

---

## Retrieval Ranking Formula

```
final_score = (0.6 × dense_similarity
             + 0.6 × coverage_BM25          # share of the query's rare words explained
             + 0.15 × heading_match)
             × learned_weight               # query-aware, 0.5–1.5
```

- `coverage_BM25`: BM25 divided by what the *query* could score, with stemming,
  stopwords, compound words and synonym clusters at reduced weight
- `learned_weight`: each vote counts by similarity of its question to this one
- `relevance` label from absolute evidence `max(dense,0) + 0.5·keyword`:
  high ≥ 0.45, low < 0.25
- Guardrails (`always_include`) are appended to search and packed first in
  `mol_context`, never ranked up
- Every result includes a `reason` field explaining why it was picked

---

## Memory System

- **Confidence**: 0.9 (direct statement) → 0.6 (hedged); below 0.7 gets lower retrieval weight
- **Types**: `preference` (permanent) vs `task_state` (7-day expiry)
- **Dedup**: cosine similarity ≥ 0.92 → skipped
- **Conflicts**: new fact contradicts old → old marked `replaced`, `superseded_by` → new;
  `memory_history()` shows the chain
- **Redaction**: strips common secret patterns before persisting
- **Extraction**: `mem_extract` pulls facts heuristically — no LLM call; a time word alone
  ("today") is not a fact
- **Capacity**: past `max_active_memories`, lowest-confidence / never-recalled / oldest go first

---

## Development Commands

```bash
make check                          # ruff + all tests (incl. real-model accuracy gates)
python verify_efficiency.py         # Definition of Done, 9 pass/fail checks
python benchmarks/run_benchmarks.py # regenerates benchmarks/RESULTS.md
python demo.py                      # two-minute tour

molecule ingest AGENTS.md           # or a folder
molecule context "how does chunking work" --budget 400
molecule search "how does chunking work"
molecule stats | health | doctor
molecule eval eval_queries.json --ablate
molecule serve                      # MCP server (stdio)
```

Tests use the built-in `hashing` embedder (no download); tests marked `model`
use the real model and are skipped when it is not cached.

---

## Key Rules for Development

1. Python 3.10+, type hints on all public functions
2. `pytest` + `ruff` must pass before any change is done
3. MCP tool functions stay thin — real logic lives in `src/molecule/core/`
4. Local DB at `.molecule/store.db` — like `.git`, invisible until you look
5. Never persist raw secrets or full file contents in memory — summaries and facts only
6. Default to local embedding model (all-MiniLM-L6-v2) — no cloud dependency
7. Never re-embed text you've already embedded — check `content_hash` first
8. Router index load must stay near-instant and fit its token budget — never load full chunk bodies
9. If a feature adds a network call or big dependency just for scale we'll never hit, leave it out
10. A ranking or compiler change ships only with a benchmark run that shows it helps

---

## Reference

1. @AGENTS.md — full project specification
2. @INTEGRATION.md — integration examples for LangChain, CrewAI, Claude Code, etc.
3. @README.md — public-facing documentation