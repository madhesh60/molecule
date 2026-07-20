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
   Instead of reading the entire `AGENTS.md` every turn, it calls Molecule.

3. **Molecule** — the context layer between the agent and the docs. It auto-chunks
   `AGENTS.md` into labeled sections, routes the agent to only the relevant chunk(s) for
   the current task, and remembers small facts across sessions. All local, all SQLite,
   no cloud.

---

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
Agent calls: mol_load_index()
Returns:     ~100 tokens of chunk IDs + one-line summaries
             (the "table of contents", not the "textbook")
```

This is near-instant and costs almost nothing.

### Step 2: Developer Asks a Question → Agent Searches

```
Developer:   "How does the chunker handle large sections?"
Agent calls:  mol_search("how does chunker handle large sections", k=3)
Returns:      Top 3 candidate chunk IDs + summaries + reason why each matched
              (hybrid: 0.6·embedding + 0.3·BM25 + 0.15·heading-match × weight × learned_weight)
```

Still cheap — returns summaries only, not full content.

### Step 3: Agent Fetches Only What It Needs

```
Agent calls:  mol_fetch(["AGENTS.md::Auto-Chunker"])
Returns:      Full body text of just that one chunk (~300-400 tokens)
              Instead of the entire AGENTS.md (~4,300 tokens)
```

**This is the key savings moment.** Token telemetry records: used 350, would have been 4,300.

### Step 4: Agent Does Its Work, Then Teaches the Router

```
Agent calls:  mol_feedback("AGENTS.md::Auto-Chunker", useful=True, query="how does chunker...")
Effect:       learned_weight for that chunk nudges up (bounded EMA, clamped 0.5–1.5)
              Next time a similar query comes in, this chunk ranks higher
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
→ "Always deploy through Docker Compose" (confidence: 0.9, reason: "keyword+embed match")
```

### Step 7: Check the Savings

```
Agent calls:  mol_stats()
Returns:      {"tokens_used_this_session": 350, "tokens_if_full_file_loaded": 4300, "savings_pct": 91.9}
```

---

## Key Principle: Read the Table of Contents, Not the Textbook

```
WITHOUT Molecule:  Agent reads 4,300 tokens of AGENTS.md every single turn
                   Most of it is irrelevant to the current task
                   No memory across sessions

WITH Molecule:     Agent reads ~100 tokens (index) + ~350 tokens (one chunk)
                   90%+ token savings
                   Remembers decisions, preferences, corrections across sessions
                   Router gets smarter over time via feedback
```

---

## Project Structure (for development reference)

```
molecule/
├── AGENTS.md                    # Full project spec (the "bible")
├── CLAUDE.md                    # This file — working workflow context
├── INTEGRATION.md               # Integration examples (LangChain, CrewAI, etc.)
├── README.md                    # Public-facing readme
├── pyproject.toml               # Package config
├── demo.py                      # Terminal demo showing all 6 steps
├── eval_queries.json            # 20-query labeled test set for retrieval eval
├── eval_queries_adversarial.json # Adversarial eval queries
├── src/molecule/
│   ├── __init__.py              # Molecule class — the public Python API
│   ├── server.py                # MCP server (16 tools via FastMCP)
│   ├── cli.py                   # CLI: molecule ingest/search/serve/stats/...
│   ├── config.py                # MoleculeConfig + load_config()
│   ├── db.py                    # SQLite connection + schema init
│   ├── core/
│   │   ├── chunker.py           # Heading-based markdown splitter
│   │   ├── labeler.py           # One-line summary generator for each chunk
│   │   ├── embedder.py          # Local embedding (all-MiniLM-L6-v2, ~80MB)
│   │   ├── ingest.py            # Orchestrates chunk → label → embed → store
│   │   ├── retriever.py         # Hybrid ranking: embed + BM25 + heading match
│   │   ├── scoring.py           # Score combination logic
│   │   ├── expansion.py         # Query expansion helpers
│   │   ├── memory.py            # mem_write, mem_search, dedup, conflict detection
│   │   ├── extractor.py         # mem_extract — heuristic fact extraction
│   │   ├── feedback.py          # Self-learning router (EMA weights)
│   │   ├── telemetry.py         # Token savings tracking
│   │   ├── cache.py             # Content-hash based caching
│   │   └── eval.py              # Retrieval precision evaluation
│   └── logging_setup.py
├── tests/                       # pytest test suite
└── .molecule/                   # Local DB store (like .git — invisible until you look)
    └── store.db                 # SQLite: chunks + memories + telemetry + feedback
```

---

## MCP Tools (the agent's interface)

| Tool | Purpose | Cost |
|---|---|---|
| `mol_ingest(file)` | Chunk + index a markdown file | One-time, ~2-5s |
| `mol_load_index()` | Get table of contents (IDs + summaries) | ~100 tokens, instant |
| `mol_search(query, k)` | Find relevant chunks + memories | Summaries only, cheap |
| `mol_fetch(chunk_ids)` | Get full body text for specific chunks | The "expensive" step — but targeted |
| `mol_feedback(chunk_id, useful)` | Teach the router what was useful | Free |
| `mol_feedback_stats()` | See what the router has learned | Free |
| `mem_write(text, ...)` | Store a durable fact | Free |
| `mem_search(query, k)` | Retrieve relevant memories | Cheap |
| `mem_extract(transcript)` | Auto-extract facts from a conversation | No LLM call |
| `mol_forget(memory_id)` | Delete a memory | Free |
| `mol_list_sources()` | List ingested files | Free |
| `mol_remove_source(file)` | Remove a file from the index | Free |
| `mem_merge_duplicates()` | Merge near-duplicate memories | Periodic cleanup |
| `mol_eval(queries, k)` | Run retrieval precision eval | Dev-time only |
| `mol_health()` | DB status + counts | Free |
| `mol_stats()` | Live token savings percentage | Free |

---

## Retrieval Ranking Formula

```
final_score = (0.6 × embedding_similarity
             + 0.3 × BM25_keyword_score
             + 0.15 × heading_match)
             × weight
             × learned_weight
```

- `weight`: default 1.0, guardrails (`always_include`) = 2.0+
- `learned_weight`: starts at 1.0, adjusted by `mol_feedback` votes (bounded EMA, clamped 0.5–1.5)
- Every result includes a `reason` field explaining why it was picked

---

## Memory System

- **Confidence**: 0.9 (direct statement) → 0.6 (hedged) → auto-trust ≥ 0.7 without asking
- **Types**: `preference` (permanent) vs `task_state` (7-day expiry)
- **Dedup**: cosine similarity > 0.92 → skip or merge
- **Conflicts**: new memory contradicts old → old marked `status='replaced'`
- **Redaction**: strips common secret patterns before persisting
- **Extraction**: `mem_extract` pulls facts heuristically from a transcript — no LLM call

---

## Development Commands

```bash
# Run tests
pytest

# Lint
ruff check src/ tests/

# Demo (shows all 6 steps in terminal)
python demo.py

# Start MCP server
molecule serve

# Quick CLI usage
molecule ingest AGENTS.md
molecule search "how does chunking work"
molecule stats
molecule health

# Run retrieval eval
molecule eval eval_queries.json
```

---

## Key Rules for Development

1. Python 3.10+, type hints on all public functions
2. `pytest` + `ruff` must pass before any change is done
3. MCP tool functions stay thin — real logic lives in `src/molecule/core/`
4. Local DB at `.molecule/store.db` — like `.git`, invisible until you look
5. Never persist raw secrets or full file contents in memory — summaries and facts only
6. Default to local embedding model (all-MiniLM-L6-v2) — no cloud dependency
7. Never re-embed text you've already embedded — check `content_hash` first
8. Router index load must stay near-instant — never load full chunk bodies
9. If a feature adds a network call or big dependency just for scale we'll never hit, leave it out

---

## Reference

1. @AGENTS.md — full project specification
2. @INTEGRATION.md — integration examples for LangChain, CrewAI, Claude Code, etc.
3. @README.md — public-facing documentation