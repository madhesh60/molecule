# Molecule — architecture and design decisions

This document explains *why* Molecule is built the way it is. Every decision
below was taken by measuring, and the measurements are reproducible with
`python benchmarks/run_benchmarks.py`. Where an idea was tried and dropped,
it says so and why.

- [1. The shape of the system](#1-the-shape-of-the-system)
- [2. Chunking](#2-chunking)
- [3. Ranking](#3-ranking)
- [4. The context compiler](#4-the-context-compiler)
- [5. Learning: query-aware feedback](#5-learning-query-aware-feedback)
- [6. Bonds](#6-bonds)
- [7. Knowing what it does not know](#7-knowing-what-it-does-not-know)
- [8. Memory](#8-memory)
- [9. Storage, caching and freshness](#9-storage-caching-and-freshness)
- [10. Token accounting](#10-token-accounting)
- [11. How it was evaluated](#11-how-it-was-evaluated)
- [12. Experiments that did not make it](#12-experiments-that-did-not-make-it)

---

## 1. The shape of the system

```
            MCP server (server.py)        CLI (cli.py)        your code
                       \                      |                  /
                        \_____________ Molecule (engine.py) ____/
                                          |
     lock · freshness check · token accounting · learning hooks · latency metrics
                                          |
   core/: chunker → labeler → embedder → ingest      retriever → context
          bonds · feedback · memory · extractor · telemetry · sync · snapshot · doctor
                                          |
                         SQLite (.molecule/store.db, WAL)
```

One rule held the design together: **the tool layers contain no logic.** The
MCP server and the CLI are one-line delegations to the `Molecule` engine, and
the engine is a thin coordinator over pure functions in `core/` that take a
connection and a config. Everything is unit-testable without a server, and
the server and CLI cannot drift apart.

The engine adds the cross-cutting concerns once: a re-entrant lock (one
instance can be shared by threads and by the async server), the freshness
check before reads, recording every returned token, turning "fetched
together" into a learning signal, and per-operation latency.

## 2. Chunking

`core/markdown.py` + `core/chunker.py`

A chunk should be the smallest piece that still answers a question on its own.
Headings are the author's own answer to "where does one topic end", so they
are the primary boundary (H1–H3 by default; deeper headings stay inside).

Problems found in 0.1 and fixed:

| Problem | Fix |
|---|---|
| `# comment` lines in a bash block became H1 headings and tore sections apart (it happened in this repo's own `CLAUDE.md`) | a fence tracker; lines inside fenced code are never headings |
| Large sections were split on `. ` across code, tables and lists, producing broken markdown | split on *blocks* (paragraph, list, table, code); only an oversized block is split inside — tables repeat their header, code is re-fenced, lists split on items, prose on sentences with one sentence of overlap |
| Two sections with the same heading path got the same id and overwrote each other | duplicate paths get `(2)`, `(3)` |
| Text before the first heading was silently dropped | kept as `(intro)` |
| A short guardrail was merged into its neighbour, which then became a guardrail too | guardrails never merge |
| The `<!-- always_include -->` marker was shown to the agent | stripped; a mention in `inline code` is not a marker |

Labels (the one-line "table of contents" entry) are derived from structure
with no LLM: lead sentence, first column of a table, defined function and
table names, bolded phrases. They are cached by content hash.

## 3. Ranking

`core/scoring.py`, `core/normalize.py`, `core/expansion.py`, `core/retriever.py`

```
score = (0.6 · dense + 0.6 · keyword + 0.15 · heading) × learned
```

### Dense

`all-MiniLM-L6-v2`, reading the first 256 wordpieces of
`heading path + label + content`. Embedding the heading and label with the
body gives the vector the author's own words for the topic.

### Keyword: coverage-normalised BM25

0.1 normalised BM25 by the best document: `bm25(d) / max(bm25)`. The top
document always scores 1.0, however weak the match. On a reworded question
the best keyword match is often one incidental common word, and it was
reported as a perfect keyword hit, outvoting the embedding.

Molecule now divides by what the *question* could score:

```
keyword(d) = Σ_t idf(t) · sat(t, d)  /  Σ_t idf(t)       sat = tf / (tf + k1·(1 − b + b·len/avglen))
```

`sat` is BM25's saturated term frequency (in [0, 1)), so the score is "the
share of the question's information (by rarity) that this document explains".
A query whose rare words all appear scores high; a paraphrase that shares one
common word scores near zero and the dense signal decides.

The tokenizer drops English stopwords (on a 30-section document IDF is too
noisy to do that job) and applies a light stemmer (plurals, -ing/-ed, a few
noun endings), so *deploy, deployed, deploying, deployments* are one term.
Adjacent query words are also tried joined (`roll back` → `rollback`), but
only when the joined form exists in the corpus, so this never adds noise.

Concept-cluster synonyms (`key ~ secret ~ token ~ credential`, …) count at
0.7 weight and never more than the synonym's own rarity allows, so an exact
match always beats a synonym match.

Measured on the held-out handbook's reworded questions: top-3 54% → 77% from
coverage + stemming + stopwords alone; 88% with the rest of this section.

### Heading

The share of question terms that appear in the section's heading path
(synonyms at half weight). Small weight, but it breaks ties on questions
phrased like section titles.

### Ablation (top-3, from RESULTS.md)

| | AGENTS.md reworded | handbook reworded |
|---|---|---|
| full | 91.7% | 87.5% |
| − keyword | 50.0% | 81.2% |
| dense only | 50.0% | 81.2% |
| lexical only | 75.0% | 58.3% |

Neither signal is enough alone; which one matters more depends on the
document. Query expansion helps on AGENTS.md (the document its clusters were
first written against) and is neutral on the handbook, so it is kept as a
small helper, not relied on.

### Hot path

Everything that does not depend on the question — the vector matrix, the
inverted index, labels, token counts, the vote history — is built once and
cached (section 9). A search is one query embedding (~2–3 ms on CPU), one
matrix–vector product and a few array operations: ~3 ms at p50 on 282 chunks,
embedding included.

## 4. The context compiler

`core/context.py`

The step-by-step workflow (index → search → fetch) costs tokens at every
step, and on a small file it costs more than loading the file (section 10).
`mol_context(query, budget)` does the whole thing in one call:

1. **Rank** all chunks.
2. **Gate.** Keep only chunks scoring at least 60% of the best. The budget is a
   ceiling, not a target. Measured: the same answer recall (within 2 points) as
   filling the budget with the next best results, for up to 47% fewer tokens.
3. **Pack** whole chunks, best first. A chunk whose vector is ≥ 0.92 similar to
   one already packed is skipped and reported as a near-duplicate.
4. **Bonds.** Chunks linked to a packed chunk (section 6) are eligible at half
   the bar, strongest link first.
5. **Excerpt** the first chunk that overflows. Default `head`: keep its
   beginning, cut on a word boundary, close an open code fence. Optional
   `focus`: keep the passages most similar to the question, in document order,
   with gaps marked.
6. **Guarantee the budget.** Joined text is re-counted; items are dropped from
   the end until it fits. No benchmark run has ever exceeded a budget.

Guardrails are packed first; relevant memories are appended in a capped share
of the budget. Each item reports its id, tokens, score and reason; skipped
chunks are listed under `omitted` with the reason.

## 5. Learning: query-aware feedback

`core/feedback.py`

0.1 kept one learned weight per chunk, moved by a bounded moving average on
each vote. The flaw: the weight applied to *every* question. A chunk that
helped with deploy questions got boosted for logging questions too.

Now each vote stores its question's embedding. At search time the chunk's
weight is computed for *this* question by replaying its recent votes (up to
64), oldest first, from 1.0:

```
w     = clip((cos(q_now, q_vote) − 0.3) / (0.85 − 0.3), 0, 1)     # how much this vote applies
step  = 0.2 · max(w, 0.1)                                          # 0.1 = small global leak
L    ← (1 − step) · L + step · target                              # target 1.5 useful / 0.5 not
```

Properties, each covered by a test:

- The same question again → `w = 1` → exactly the 0.1 behaviour (1.0 → 1.10).
- An unrelated question → only the 0.1 leak applies.
- `L` is a convex combination of values in [0.5, 1.5], so it can never leave
  that range; one vote can never dominate.
- Votes stored by 0.1 (text, no vector) are embedded on first use.

Measured (RESULTS.md §6): after 81 votes on 24 hard questions, top-1 on
*rewordings* of those questions went 62.5% → 91.7% (a single global weight
reached 83.3%), and questions about other topics stayed at 87.5% (the global
weight pulled them down to 83.3%).

A caveat measured during design: rewordings of one question are very similar
(cosine ≈ 0.93), but *different* questions about the same topic often are not
(median ≈ 0.23). The kernel transfers a lesson to rewordings, not to the
whole topic. That is the conservative choice.

## 6. Bonds

`core/bonds.py`

Chunks are the atoms; bonds say "whoever needs this one probably needs that
one too". Three kinds, merged with a noisy-or when several apply:

- **ref** (0.5) — parsed at ingest: `§7.3`, `section 2.1`, `[text](#anchor)`,
  `[text](other.md#anchor)`. Numbers resolve against numbered headings; links
  against GitHub-style slugs.
- **continuation** (0.6) — consecutive parts of a section that was split.
- **cofetch** — learned. When the agent fetches chunks within the same
  3-minute window (or marks both useful), each pair moves toward 1:
  `s ← s + 0.2·(1 − s)`. Each co-occurrence is counted once: a new fetch is
  paired with what came before it, but earlier pairs are not reinforced
  again. Links fade with a 30-day half-life and links below 0.25 are
  ignored, so one coincidence never becomes permanent.

Bonds never change rank. They appear as `related` hints on search and fetch
results (saving the agent a search round-trip) and lower the bar in the
context compiler. Measured on 8 two-part questions: answered completely 5/8
without bonds, 7/8 with cross-reference bonds, and 8/8 at a 1,200-token
budget once co-fetch learning has seen each pair used together three times.
At 400–600 tokens the eighth question needs one more section than fits.

Two flaws in the first co-fetch rule were found by that benchmark and fixed:
the same pair was reinforced again on every later fetch inside the window
(inflating links), and a large group was trimmed alphabetically, which could
drop the chunk that had just been fetched.

## 7. Knowing what it does not know

The final score is only meaningful *within* one question. To say "nothing
covers this", Molecule uses a separate absolute evidence score:

```
evidence = max(dense, 0) + 0.5 · keyword          high ≥ 0.45 · low < 0.25
```

Thresholds were set on AGENTS.md with a separate set of off-topic questions
(`oos_dev.json`), then checked on the two other documents without changes:

| | real questions answered | off-topic flagged low |
|---|---|---|
| AGENTS.md (tuned here) | 31/32 | 12/12 |
| handbook | 93/96 | 10/16 |
| firmware notes (blind) | 49/52 | 6/10 |

Unrelated questions (recipes, sports, geography) are always caught. Questions
that sound like the document but are not answered by it ("parental leave
policy" against an engineering handbook, "price of the station hardware"
against firmware notes) mostly are not — a small embedding model cannot tell
"about the same company" from "answered here". When confidence is low,
`mol_context` returns only the single best guess, labelled as such, instead
of filling the budget with weak matches.

## 8. Memory

`core/memory.py`, `core/extractor.py`

Every write passes one pipeline: redact secrets (API keys, tokens, JWTs, PEM
headers, credentials in URLs, `password: x`) → embed → skip if a near-copy
(cosine ≥ 0.92) exists → if it is on the same topic (≥ 0.85) and contains a
contradiction cue ("no longer", "instead", "actually", …), retire the old fact
with `status='replaced'` and `superseded_by` pointing at the new one → insert.
`memory_history()` walks that chain in both directions.

Extraction from a transcript is heuristic (no LLM): direct statements ("we
decided", "always", "I prefer") get confidence 0.9, hedged ones 0.6, ongoing
states ("we are currently debugging…") become 7-day `task_state` facts. A
sentence that only mentions time ("long session today") is not a fact.

The notebook is bounded (`max_active_memories`). Recalls are counted in a
separate table (so counting does not invalidate the search cache), and when
the cap is reached the least valuable facts go first: lowest confidence,
then never recalled, then oldest.

## 9. Storage, caching and freshness

`db.py`, `core/cache.py`, `core/ingest.py`, `core/sync.py`

**One file.** SQLite in WAL mode with a busy timeout, so the MCP server and a
CLI command can use the same store at once. Vectors are float32 blobs in the
same file; brute-force similarity is ~0.1 ms per thousand chunks, so no ANN
index is needed at this scale.

**Migrations.** `PRAGMA user_version` + numbered steps, each in one
transaction. A store from 0.1 (no version) is adopted by step 1 and upgraded
by step 2 with its data, votes and weights. A store from a *newer* Molecule
is refused with a clear message instead of being damaged.

**Cache invalidation without guesswork.** Triggers on `chunks`, `memories`
(only the columns that affect ranking), `chunk_feedback` and `chunk_bonds`
increment a per-table counter in a `generations` table. Each cached value is
stored with the counters of the tables it was built from (plus any settings
it depends on). Raw SQL and writes from another process change the counters,
so a stale result is impossible; recording telemetry or memory hits touches
no counted column, so it evicts nothing.

**Ingest is atomic and incremental.** All writes for a file happen in one
transaction. A file whose bytes are unchanged returns after one hash (0.6 ms).
In a changed file only chunks whose content hash changed are re-embedded
(editing one section of a 5,000-line file: 0.09–0.14 s across runs). Each source records a
*signature* — chunker version, embedding format, model, chunk sizes — and a
change to any of them rebuilds that source. Every vector records the model
that produced it; vectors from another model are re-embedded on first use,
never compared. A renamed heading whose body did not change keeps its votes,
weight and learned bonds.

**Freshness.** Before a read, the engine compares each source's size and
modification time with the stored values (one `stat()` per file, throttled to
once per 5 s) and re-ingests what changed.

**Portable.** Chunk ids use the repo-relative path (`docs/setup.md::…`), so
two files with the same name in different folders do not collide and moving
the repo does not change ids.

## 10. Token accounting

`core/telemetry.py`

0.1 counted only fetched chunk bodies as "used", and counted the whole file as
the baseline once per engine instance. Both flattered the result: the table of
contents and search results were free, and opening the store twice counted
the file twice.

Now every response is counted by kind (index, search, fetch, context, memory)
and the baseline counts each source file once per *session*, process-wide,
only when one of its chunks is actually delivered. The true file size is
measured at ingest. Lifetime totals and an estimated dollar saving are kept
across sessions.

The consequence is visible in RESULTS.md §8: on a 5.6k-token file, the
step-by-step workflow costs 555–1,124 tokens per question, while `mol_context`
costs 227 (budget 300) with every question answered. Loading the whole file
costs 5,631. The honest pitch is therefore: use `mol_context`; and the larger
the docs, the larger the saving, because its cost per question does not grow
with the corpus.

## 11. How it was evaluated

- **Three documents.** AGENTS.md (32 questions), an invented engineering
  handbook (48 direct + 48 reworded + 24 rewordings-of-rewordings + 16
  off-topic + 8 two-part), and invented firmware notes (26 + 26 + 10).
- **Answer-phrase labels.** A handbook or firmware question is labelled with a
  phrase that appears in exactly one section (checked by script). A result is a
  hit when the returned text contains it. This stays fair when the chunker
  changes, which hand-picked chunk ids do not.
- **A blind set.** The firmware notes and their questions were written after
  every setting was frozen and scored once. Development sets were used for
  tuning; the blind numbers are the ones to quote.
- **Same script for both versions.** `benchmarks/retrieval_bench.py --src`
  scores any checkout through the stable public API; the 0.1 baseline was
  produced that way from the previous commit.

## 12. Experiments that did not make it

| Idea | Result | Decision |
|---|---|---|
| Newer small embedding models (bge-small, e5-small, gte-small, arctic-embed-s) | 4–10 points *lower* top-3 than MiniLM-L6 on reworded questions (dense only, 256 tokens) | keep MiniLM-L6; any model is one setting away |
| A vector per sentence ("passage vectors"), chunk scored by its best sentence | ≈ +1 point top-3 for ≈ 5× ingest time | off by default (`passage_vectors`) |
| Two-stage search: rerank the top 5/10/20 by sentence vectors | no gain over one stage | dropped |
| Diversity re-ordering (MMR) in the compiler | no gain; slightly lower recall at mid budgets | dropped; near-duplicate skip kept |
| Query-focused excerpts | lost to keeping the head of the section at tight budgets (58% vs 73% at 100 tokens): sections lead with their key fact | opt-in (`context_excerpt="focus"`) |
| int8-quantised ONNX model | 2× faster encoding, but vectors drift (cosine 0.98–0.99 to the reference) | available via `onnx_model_file`; tagged as a different model so it is never mixed |
| Abstaining entirely on low confidence | cost hard reworded questions more than it saved | one flagged best guess instead |
| Embedding fewer wordpieces (96 / 128) | faster ingest (≈ 2 s vs 4.5 s for 5k lines), 1–4 points lower on reworded questions | 256 by default; `chunk_embed_tokens` |
