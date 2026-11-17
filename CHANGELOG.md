# Changelog

## 0.2.0

Measured with `python benchmarks/run_benchmarks.py`; full tables in
`benchmarks/RESULTS.md`, design notes in `docs/ARCHITECTURE.md`.

### Retrieval
- Keyword scoring is coverage-normalised BM25 with stemming, stopwords and
  compound words ("roll back" ↔ "rollback"). Reworded questions, top-3:
  42.3% → 88.5% on a blind document, 54.2% → 87.5% on a held-out handbook.
- Chunk vectors read 256 wordpieces instead of 96.
- Every result has a `relevance` label (high / medium / low).

### New
- `mol_context(query, budget)`: one call returns ready-to-paste context that
  never exceeds the budget — same answers as filling the budget, up to 47%
  fewer tokens.
- Bonds between chunks: parsed cross-references (`§7.3`, `[x](#anchor)`, links
  to other files) and Hebbian co-fetch learning with a 30-day half-life.
  Shown as `related` on search and fetch results; used by `mol_context`.
- Query-aware feedback: each vote counts in proportion to how similar the
  current question is to the one it was cast for.
- `mol_load_index(max_tokens, prefix)` always fits its budget.
- `mol_sync`, automatic freshness before reads, directory ingest.
- `molecule doctor`, `export`, `import`, `backup`, `context`, `bonds`, `sync`;
  `--json` on most commands; latency percentiles in `mol_health`.
- Memory: supersession chain (`superseded_by`, `memory_history`), capacity
  limit with value-based eviction, recall counts.

### Production
- Embeddings run on ONNX Runtime: no PyTorch, ~0.3 s model load. PyTorch is an
  optional backend; a built-in hashing embedder needs no download.
- Works offline: the model is read from the local cache first, and token
  counting no longer downloads anything at import time.
- Versioned schema migrations; 0.1 stores upgrade in place.
- Atomic incremental ingest, per-source build signatures, model-tagged vectors
  that re-embed themselves when the model changes.
- Exact cache invalidation via SQLite triggers and generation counters.
- One thread-safe `Molecule` engine; the MCP server and CLI are thin wrappers.
- The MCP server works with both mcp 1.x and 2.x (0.1 failed to import on 2.x).
- Every config field can be set with `MOLECULE_<NAME>`; bad values never crash.

### Fixes
- `# comment` lines inside code blocks were treated as headings.
- Splits could break tables, lists and code blocks; text before the first
  heading was dropped; duplicate headings overwrote each other.
- Telemetry ignored the cost of the index and search results, and counted a
  file twice when a store was opened twice in one process.
- A guardrail chunk could be merged into an ordinary one; the
  `always_include` marker was shown to the agent.
- Co-fetch learning reinforced the same pair repeatedly and could drop the
  chunk just fetched when trimming a large group.
- Caches for learned weights and bonds ignored settings they depend on.
- "Long session today." was stored as a memory.

### Tests
88 → 391 tests, hermetic by default, plus real-model accuracy gates, an
end-to-end MCP stdio test and `verify_efficiency.py` (Definition of Done).

## 0.1.0

First release: heading chunker, hybrid retrieval, bounded-EMA feedback,
heuristic memory extraction, token telemetry, 16 MCP tools.
