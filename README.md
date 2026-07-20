# Molecule

**Give any AI agent a memory and a brain — without reading the whole manual every time.**

Molecule is a small MCP server you drop into any agent (Claude Code, Cursor, LangChain, CrewAI). It does three things:

1. **Splits your big docs** (`AGENTS.md`, `CRAWL.md`, any markdown) into labeled chunks and only gives the agent the one chunk it actually needs — instead of dumping the whole file every time.
2. **Remembers facts across sessions** — decisions, preferences, corrections — so the agent never asks the same question twice. It can even **extract facts from a transcript automatically** (`mem_extract`, no LLM call needed).
3. **Learns from feedback** — agents report which chunks were actually useful (`mol_feedback`), and the router ranks those higher next time. The routing genuinely improves with use.

It proves the savings with a live number: `mol_stats()` tells you exactly how many tokens were used vs. what would have been sent without Molecule.

---

## The problem in plain English

| Without Molecule | With Molecule |
|---|---|
| Agent reads all 5,000 lines of `AGENTS.md` every session | Agent reads a ~100-token table of contents, then fetches only the 1 relevant chunk (~300 tokens) |
| User re-explains preferences every new chat | Facts stored locally, retrieved by meaning — or auto-extracted from the transcript |
| No idea why a chunk was picked | Every result has a `reason` field: `"similarity: 0.71, keyword: 0.44, heading match: 0.15, learned boost: +0.10"` |
| Retrieval quality is fixed forever | The router learns from `mol_feedback` votes and gets better with use |

---

## How it works (technically simple)

```
Your docs (AGENTS.md, etc.)
        │
        ▼
   Chunker          splits on headings (H1/H2/H3), then sentence boundaries
        │            merges tiny sections, splits huge ones
        ▼
   SQLite DB        .molecule/store.db  ← single local file, no internet needed
   ┌──────────┐
   │  chunks  │  id · heading · summary · content · embedding · learned_weight
   │ memories │  text · confidence · type · expires_at · embedding
   │ feedback │  chunk_id · useful · query · ts       ← the router's training data
   │ telemetry│  tokens_used · tokens_baseline · session_id
   └──────────┘
        │
        ▼
   MCP Server       16 tools via FastMCP
        │
        ▼
   Any MCP client   Claude Code · Cursor · LangChain · CrewAI
```

Retrieval uses **hybrid search**:

```
score = (0.6 × embedding + 0.3 × BM25 keyword + 0.15 × heading match) × learned_weight
```

- **Embeddings**: local `all-MiniLM-L6-v2` (~80MB, CPU-only, no API key). Each chunk embeds its *heading path + summary + content* together — much stronger routing signal than raw content alone.
- **BM25-lite keyword scoring**: hand-rolled IDF-weighted BM25 (~30 lines, numpy-free), normalized per-query so it shares a [0,1] band with cosine similarity.
- **Query expansion**: each query token pulls in its concept-cluster siblings (key→secret/token/credential, forget→memory/recall, …) as reduced-weight terms — so a paraphrased question still finds the section that answers it, while exact wording always dominates. Generic developer vocabulary, works on any corpus, no LLM call.
- **Heading match boost**: query terms that appear in the section heading count extra.
- **Learned weight**: adjusted by `mol_feedback` votes via a bounded EMA (clamped 0.5–1.5) — useful chunks rise, useless ones sink, and one vote can never dominate.
- **Hot-path caching**: corpus state (embedding matrix, BM25 index) is memoized per connection and invalidated automatically on any write; repeat query embeddings are LRU-cached. A search costs one query embedding + one matrix multiply.

Measured on this repo's own `AGENTS.md`: **20/20** on the labeled eval set and **12/12** on an adversarial paraphrase set (queries sharing almost no vocabulary with their target section). A 5,000-line document embeds in ~4s; editing one section re-ingests in ~0.13s.

No external vector DB. No internet required.

---

## Install from zero

**Requirements:** Python 3.10+

```bash
# 1. Clone
git clone https://github.com/rajmadhesh/molecule
cd molecule

# 2. Install (editable, with dev tools)
pip install -e ".[dev]"

# 3. One-time: download and cache the embedding model (~80MB)
molecule setup
```

That's it. After `molecule setup`, every ingest is embed-only (a full AGENTS.md re-ingest takes well under 5 seconds).

---

## Quick demo

```bash
python demo.py     # or: make demo
```

Walks through the full story in one terminal run: ingest → cheap index → selective fetch with live savings → feedback visibly changing the ranking → auto memory extraction → cross-session recall from a fresh process.

---

## Run the tests

```bash
pytest            # all tests
pytest -v         # with detail
ruff check src tests
make check        # lint + tests + eval — the full gate
```

**What the tests cover:**
- `test_chunker.py` — heading parsing, token boundary splits, `always_include` detection, hash stability
- `test_scoring.py` — BM25 term weighting, normalization, heading-match scores
- `test_expansion.py` — concept-cluster expansion: paraphrases reach synonym docs, exact matches always win
- `test_retriever.py` — router index never loads full content, search returns `reason` field, fetch returns content
- `test_cache.py` — hot-path cache hits, write invalidation (own + external connections), telemetry writes don't evict
- `test_labeler.py` — structure-aware labels: tables, code, SQL DDL, bold phrases, fence-skipping
- `test_feedback.py` — EMA math, clamping, persistence, feedback flipping a ranking
- `test_extractor.py` — cue classification, confidence assignment, expiry routing, secret redaction, dedup
- `test_memory.py` — dedup, conflict replacement, secret redaction, expiry
- `test_hardening.py` — empty/binary/missing files, invalid inputs, fetch ordering
- `test_telemetry.py` — savings % arithmetic, accumulation across calls
- `test_db.py` / `test_config.py` — schema migration in place, config.json + env precedence
- `test_eval.py` — precision harness + a slow integration test against the real AGENTS.md

---

## Connect to Claude Code

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "molecule": {
      "command": "python",
      "args": ["-m", "molecule"],
      "cwd": "/absolute/path/to/your/repo"
    }
  }
}
```

Restart Claude Code. You'll see the 16 tools available.

## Connect to Cursor

Add to `.cursor/mcp.json` in your project:

```json
{
  "mcpServers": {
    "molecule": {
      "command": "python",
      "args": ["-m", "molecule"]
    }
  }
}
```

## Connect to a LangChain / custom agent

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(command="python", args=["-m", "molecule"])
async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        result = await session.call_tool("mol_ingest", {"file_path": "AGENTS.md"})
```

---

## The tools — and when to call them

| Tool | When to call | Cost |
|---|---|---|
| `mol_ingest(file_path)` | Once, when you add/change a doc | Embed-only after setup (<5s) |
| `mol_load_index()` | Start of every session | Cheap — summaries only, no full text |
| `mol_search(query, k=5)` | When you need to find something | Cheap — returns ids + summaries + reason |
| `mol_fetch(chunk_ids)` | After search, to read the actual content | Only what you need |
| `mol_feedback(chunk_id, useful)` | After fetch, once you know if it helped | Instant — teaches the router |
| `mol_feedback_stats()` | To see what the router has learned | Instant |
| `mem_write(text, source_task)` | To save one key fact | Fast |
| `mem_extract(transcript)` | End of a task — auto-extracts up to 5 facts | Fast, no LLM call |
| `mem_search(query, k=5)` | When you need to recall something | Fast |
| `mol_forget(memory_id)` | To delete a stored memory | Instant |
| `mem_merge_duplicates()` | Periodically, to keep memory lean | Fast |
| `mol_list_sources()` / `mol_remove_source()` | Manage ingested files | Instant |
| `mol_eval(queries)` | Measure retrieval precision | Fast |
| `mol_stats()` | Anytime — shows live token savings | Instant |
| `mol_health()` | Health check | Instant |

### Recommended agent workflow

```
Session start:
  mol_load_index()          → get the table of contents (~100 tokens)

When you need info:
  mol_search("your query")  → find relevant chunk ids + summaries
  mol_fetch([chunk_id])     → read only that chunk (~300-400 tokens)
  mol_feedback(chunk_id, useful=True)   → teach the router

After finishing a task:
  mem_extract(transcript)   → memory writes itself
  (or mem_write("key fact", "task_name") for a single fact)

Check savings:
  mol_stats()               → see "saved 78% of tokens this session"
```

---

## The self-learning router

Every `mol_feedback(chunk_id, useful)` vote nudges that chunk's `learned_weight` with a bounded EMA:

```
new = clamp(0.8 × current + 0.2 × target, 0.5, 1.5)     target = 1.5 if useful else 0.5
```

- One positive vote from fresh: `1.0 → 1.10`. It takes repeated agreement to reach the cap.
- One negative vote: `1.0 → 0.90` — a nudge, never a cliff.
- Weights survive re-ingest and process restarts (they live in SQLite).
- `always_include` guardrail chunks can drop in rank but are **never** dropped from results.
- The `reason` field shows it working: `"…, learned boost: +0.10"`.

## Auto memory extraction

`mem_extract(transcript)` pulls durable facts out of raw conversation text with zero LLM calls:

- **Direct statements** ("we decided…", "always…", "I prefer…") → confidence **0.9**
- **Hedged statements** ("I think maybe…") → confidence **0.6**
- **Ephemeral facts** ("currently debugging…") → stored as `task_state`, auto-expire in 7 days
- Everything else is dropped. Max 5 facts per call.
- Each fact flows through the same pipeline as `mem_write`: secret redaction, dedup, conflict replacement — for free.
- Optionally pass an `llm_fn` to rewrite facts into cleaner one-liners when an LLM is available.

## Memory features

**Deduplication** — if a new fact is too similar to an existing one (cosine similarity > 0.92), it's skipped automatically.

**Conflict detection** — if a new fact contradicts an old one ("no longer", "changed", "actually"), the old one is marked `replaced` (not deleted — history is preserved).

**Expiry** — `task_state` memories (e.g. "currently debugging X") expire after 7 days. `preference` memories live forever.

**Secret redaction** — API keys, tokens, passwords are stripped before anything is stored.

**Confidence scores** — facts stated directly get 0.9+. Low-confidence facts (< 0.7) get lower retrieval weight so they don't override things you said explicitly.

**Reason field** — every search result explains why it was picked. No black box.

---

## Mark critical rules as always-load

Add this comment anywhere in your markdown to make a section load every session regardless of the query:

```markdown
<!-- always_include -->
Never modify the production database directly.
```

Or in YAML front-matter:
```yaml
---
always_include: true
---
```

These chunks are **appended** to every search result if they didn't rank on merit — they never dominate the ranking, and feedback can never remove them.

---

## Data & privacy

Everything lives in `.molecule/store.db` relative to your repo root — same idea as `.git/`. Nothing leaves your machine. No API keys needed. Errors are logged to `.molecule/molecule.log` (the MCP stdio channel stays clean).

---

## Configuration

Three layers, later wins: **defaults < `.molecule/config.json` < environment variables**.

`.molecule/config.json` example:

```json
{
  "max_chunk_tokens": 300,
  "default_k": 3,
  "heading_boost": 0.2
}
```

| Setting | Default | What it does |
|---|---|---|
| `max_chunk_tokens` | 400 | Split sections larger than this |
| `min_chunk_tokens` | 50 | Merge sections smaller than this |
| `default_k` | 5 | Results returned by search |
| `emb_weight` | 0.6 | Embedding similarity weight in ranking |
| `keyword_weight` | 0.3 | BM25 keyword weight in ranking |
| `heading_boost` | 0.15 | Max boost for query terms in headings |
| `feedback_ema_alpha` | 0.2 | How fast feedback moves learned weights |
| `learned_weight_min/max` | 0.5 / 1.5 | Clamp range for learned weights |
| `extract_max_facts` | 5 | Max facts per mem_extract call |
| `dedup_threshold` | 0.92 | Similarity above which a memory is a duplicate |
| `embedding_model` | `all-MiniLM-L6-v2` | Local model, no API key needed |
| `task_state_expiry_days` | 7 | How long task_state memories live |
| `low_confidence_threshold` | 0.7 | Below this, memory gets lower retrieval weight |

Env vars (`MOLECULE_MAX_CHUNK_TOKENS`, `MOLECULE_DEFAULT_K`, `MOLECULE_EMBEDDING_MODEL`, …) override everything — handy for Docker/CI.

> **Migration note:** when the embedding format changes (e.g. this version embeds heading+summary+content instead of raw content), existing DBs re-embed automatically on the next `mol_ingest` — no wipe needed, and learned feedback weights are preserved.

---

## Project structure

```
src/molecule/
├── config.py          MoleculeConfig + config.json/env loading
├── db.py              SQLite schema + in-place migrations
├── server.py          16 MCP tools (thin wrappers, @_safe error handling)
├── logging_setup.py   file logger (.molecule/molecule.log)
├── cli.py             ingest/search/setup/feedback/extract/eval/serve/…
├── __main__.py        python -m molecule entry point
└── core/
    ├── chunker.py     heading split → sentence overflow → merge small
    ├── embed_text.py  heading+summary+content embedding text (versioned)
    ├── embedder.py    local sentence-transformers, lazy singleton, hash-skip
    ├── scoring.py     BM25-lite + heading-match scoring (two-tier terms)
    ├── expansion.py   concept-cluster query expansion (no LLM)
    ├── cache.py       versioned per-connection corpus cache (hot path)
    ├── labeler.py     structure-aware hash-cached labels, swappable LLM fn
    ├── retriever.py   hybrid ranking, learned weights, reason field
    ├── feedback.py    self-learning router (bounded EMA)
    ├── extractor.py   heuristic fact extraction from transcripts
    ├── memory.py      dedup, conflict replace, expiry, secret redaction
    └── telemetry.py   per-session token savings counter

tests/                 test_chunker · test_scoring · test_expansion · test_retriever
                       test_cache · test_labeler · test_feedback · test_extractor
                       test_memory · test_hardening · test_telemetry · test_eval
                       test_db · test_config
```

---

## Verify it works end-to-end

```bash
python verify_efficiency.py
```

Checks all of it against this repo's own `AGENTS.md`: ingest speed (embed <5s), routing accuracy (top-3), token savings %, feedback learning, auto-extraction, dedup, and cross-session persistence.

For MCP-level inspection:

```bash
npx @modelcontextprotocol/inspector python -m molecule
```

---

## Non-goals

- Not a full agent framework
- No GUI, no cloud sync, no multi-user features
- No external database — SQLite only
- No FAISS/ANN index — brute-force is fast enough under ~10k rows
