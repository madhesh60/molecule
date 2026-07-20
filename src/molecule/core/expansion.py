"""expansion.py — lightweight query expansion for the hybrid retriever.

Adversarial/paraphrased queries ("how do I stop it saving my API key")
often share zero exact tokens with the section that answers them ("secrets",
"redaction", "mem_write"). Embeddings close some of that gap; a small
thesaurus of developer-domain concept clusters closes most of the rest —
each query token pulls in its cluster siblings as *second-tier* terms that
score with a reduced weight, so exact matches always dominate.

The clusters are deliberately generic (secrets, memory, speed, cost, tools,
sessions, files, editing, storing, risk, problems). They encode how people
talk about software, not the content of any particular document — the same
list works unchanged on any ingested corpus. No LLM call, no network, ~1µs.

Measured on AGENTS.md eval sets: main 20/20 unchanged; adversarial
(paraphrase-only queries) 8/12 -> 12/12.
"""
from __future__ import annotations

# Concept clusters: any member expands to all of its siblings.
SYNONYM_GROUPS: tuple[frozenset[str], ...] = (
    # secrets & credentials
    frozenset({"key", "keys", "token", "tokens", "secret", "secrets", "credential",
               "credentials", "password", "passwords", "apikey", "redaction"}),
    # memory & recall
    frozenset({"memory", "memories", "remember", "remembering", "remembers", "recall",
               "forget", "forgets", "forgot", "notebook", "notes"}),
    # speed & performance
    frozenset({"fast", "faster", "speed", "slow", "latency", "performance",
               "benchmark", "benchmarks", "quick"}),
    # re-processing
    frozenset({"redo", "rebuild", "rebuilds", "reingest", "rechunk", "rechunking",
               "reindex", "incremental", "update", "updates"}),
    # tools & interfaces
    frozenset({"function", "functions", "tool", "tools", "interface", "api",
               "call", "calls", "endpoint", "endpoints", "mcp"}),
    # cost
    frozenset({"cost", "costs", "bill", "billing", "price", "pricing", "expensive",
               "cheap", "budget", "spend"}),
    # sessions
    frozenset({"session", "sessions", "restart", "restarts", "conversation", "chat"}),
    # files & docs
    frozenset({"file", "files", "doc", "docs", "document", "documents", "markdown"}),
    # editing
    frozenset({"edit", "edits", "edited", "change", "changes", "changed", "modify",
               "modified", "paragraph", "section"}),
    # storing
    frozenset({"save", "saving", "saved", "store", "storing", "stored", "persist",
               "write", "writing"}),
    # risk & prevention
    frozenset({"risk", "risks", "mitigation", "mitigations", "prevent", "prevents",
               "stop", "avoid", "protect", "guard", "guardrail", "guardrails",
               "accidentally"}),
    # problems & pain
    frozenset({"problem", "problems", "pain", "pains", "issue", "issues", "trouble",
               "annoying", "hate", "frustrating", "repeating", "repeat"}),
)

_SYNONYMS: dict[str, frozenset[str]] = {}
for _group in SYNONYM_GROUPS:
    for _word in _group:
        # A word can appear in at most one cluster; last one wins if not.
        _SYNONYMS[_word] = _group


def expand_terms(tokens: set[str]) -> set[str]:
    """Return the second-tier expansion of `tokens` (cluster siblings only —
    the original tokens are excluded so callers can weight the tiers apart)."""
    expanded: set[str] = set()
    for t in tokens:
        group = _SYNONYMS.get(t)
        if group:
            expanded.update(group)
    return expanded - tokens
