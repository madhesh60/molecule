"""expansion.py — lightweight query expansion for the hybrid retriever.

A paraphrased query ("how do I stop it saving my API key") can share no
exact term with the section that answers it ("secrets", "redaction").
Embeddings close part of that gap; a small thesaurus of engineering concept
clusters closes more of it. Each query term pulls in its cluster siblings as
*second-tier* evidence that scores at a reduced weight, so an exact match
always beats a synonym match.

The clusters describe how engineers talk about software in general — they
are not derived from, or specific to, any one document. No LLM call, no
network, a dict lookup per term.
"""
from __future__ import annotations

from molecule.core.normalize import stem

# Concept clusters: any member expands to all of its siblings.
# Written as plain words for readability; matched on their stems.
SYNONYM_GROUPS: tuple[frozenset[str], ...] = (
    # secrets & credentials
    frozenset({"key", "keys", "token", "tokens", "secret", "secrets", "credential",
               "credentials", "password", "passwords", "apikey", "redaction"}),
    # memory & recall
    frozenset({"memory", "memories", "remember", "remembering", "remembers", "recall",
               "forget", "forgets", "forgot", "notebook", "notes"}),
    # speed & performance
    frozenset({"fast", "faster", "speed", "slow", "slower", "latency", "performance",
               "benchmark", "benchmarks", "quick", "quickly"}),
    # re-processing
    frozenset({"redo", "rebuild", "rebuilds", "reingest", "rechunk", "rechunking",
               "reindex", "incremental", "update", "updates", "upgrade", "refresh"}),
    # tools & interfaces
    frozenset({"function", "functions", "tool", "tools", "interface", "api",
               "call", "calls", "endpoint", "endpoints", "mcp"}),
    # cost
    frozenset({"cost", "costs", "bill", "billing", "price", "pricing", "expensive",
               "cheap", "budget", "spend", "pricey"}),
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
    # shipping code
    frozenset({"deploy", "deploys", "deployment", "release", "releases", "ship",
               "shipping", "rollout", "launch", "production", "prod"}),
    # undoing a change
    frozenset({"rollback", "revert", "undo", "restore", "recover", "recovery"}),
    # things going wrong in production
    frozenset({"outage", "outages", "incident", "incidents", "downtime", "broken",
               "failure", "failures", "crash", "crashes", "crashing"}),
    # signing in
    frozenset({"login", "logins", "signin", "authentication", "authenticate", "auth",
               "sso", "oauth"}),
    # permissions
    frozenset({"permission", "permissions", "role", "roles", "authorization",
               "access", "allowed", "rbac"}),
    # automated checks
    frozenset({"test", "tests", "testing", "spec", "specs", "check", "checks",
               "coverage", "pytest"}),
    # removing data
    frozenset({"delete", "deletes", "deleted", "remove", "removes", "removed",
               "purge", "drop", "erase", "wipe"}),
    # background work
    frozenset({"job", "jobs", "queue", "queues", "worker", "workers", "task",
               "tasks", "background", "async"}),
    # watching the system
    frozenset({"monitor", "monitoring", "alert", "alerts", "alerting", "pager",
               "oncall"}),
    # limits
    frozenset({"limit", "limits", "quota", "throttle", "throttling", "cap",
               "ceiling", "maximum", "max"}),
    # third-party code
    frozenset({"dependency", "dependencies", "library", "libraries", "package",
               "packages", "vendor"}),
    # settings
    frozenset({"config", "configuration", "configure", "setting", "settings",
               "option", "options", "flag", "flags"}),
    # people using the product
    frozenset({"user", "users", "customer", "customers", "client", "clients",
               "tenant", "tenants", "account", "accounts"}),
)

# stem -> (cluster index, stems of every sibling)
_CLUSTERS: dict[str, tuple[int, frozenset[str]]] = {}
for _index, _group in enumerate(SYNONYM_GROUPS):
    _stems = frozenset(stem(word) for word in _group)
    for _stem in _stems:
        # A stem belongs to one cluster; the first definition wins.
        _CLUSTERS.setdefault(_stem, (_index, _stems))


def siblings(term: str) -> frozenset[str]:
    """Cluster siblings of one stemmed term (without the term itself)."""
    entry = _CLUSTERS.get(term)
    return entry[1] - {term} if entry else frozenset()


def cluster_id(term: str) -> int | None:
    """Index of the concept cluster a stemmed term belongs to, if any."""
    entry = _CLUSTERS.get(term)
    return entry[0] if entry else None


def expand_terms(tokens: set[str]) -> set[str]:
    """Second-tier expansion of stemmed `tokens`: every cluster sibling,
    minus the tokens themselves (so callers can weight the tiers apart)."""
    expanded: set[str] = set()
    for token in tokens:
        expanded |= siblings(token)
    return expanded - tokens
