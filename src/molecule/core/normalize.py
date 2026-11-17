"""normalize.py — turn raw text into comparable search terms.

Three small, dependency-free steps that every lexical signal shares:

  1. lower-case and split on anything that is not a letter or digit;
  2. drop stopwords ("the", "how", "do") — on a 30-section document IDF is
     too noisy to do that job, and a stray "how" must not route a query;
  3. reduce each word to a light stem so "deploy", "deployed", "deploying"
     and "deployments" are one term.

The stemmer is intentionally light (plurals, -ing/-ed, a few noun endings).
It only has to map word *forms* together; meaning-level matches are the
embedding model's job. A stem is not idempotent in general ("database" →
"databas"), so text is stemmed exactly once, here, and everything downstream
works on stems.
"""
from __future__ import annotations

import re
from functools import lru_cache

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_VOWELS = frozenset("aeiou")

STOPWORDS: frozenset[str] = frozenset("""
a about above after again against all also am an and any are as at be because been
before being below between both but by can could did do does doing down during each
etc ever few for from further get gets got had has have having he her here hers him
his how i if in into is it its just like may me might more most much must my need
needs no nor not of off on once one only or other our ours out over own per please
really same shall she should so some still such than that the their theirs them then
there these they this those through to too two under until up us use used uses using
very via want wants was we were what when where which while who whom why will with
would yet you your yours
""".split())

# Two-letter tokens that carry real meaning in engineering docs.
_SHORT_TERMS: frozenset[str] = frozenset({
    "ai", "ci", "cd", "db", "go", "id", "io", "ip", "js", "ml", "os", "pr", "qa",
    "ts", "ui", "ux", "vm",
})


@lru_cache(maxsize=65536)
def stem(word: str) -> str:
    """Light suffix-stripping stem of one lower-case word."""
    if len(word) <= 3 or word.isdigit():
        return word

    # Plurals and third-person -s.
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith(("ss", "us", "is")):
        pass
    elif word.endswith("s"):
        word = word[:-1]

    # -ing / -ed, when what is left still looks like a word.
    for suffix in ("ing", "ed"):
        base = word[: -len(suffix)]
        if word.endswith(suffix) and len(base) >= 3 and _VOWELS & set(base):
            word = base
            if word.endswith(("at", "bl", "iz")):
                word += "e"                      # creat → create, enabl → enable
            elif len(word) > 2 and word[-1] == word[-2] and word[-1] not in "lsz":
                word = word[:-1]                 # logg → log, runn → run
            break

    # Common noun endings: deployment → deploy, configuration → configur(e).
    for suffix in ("ations", "ation", "ments", "ment"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            word = word[: -len(suffix)]
            break

    if word.endswith("e") and len(word) > 3:
        word = word[:-1]                         # cache/caching → cach
    if word.endswith("y") and len(word) > 3:
        word = word[:-1] + "i"                   # policy/policies → polici
    return word


def _keep(word: str) -> bool:
    if len(word) >= 3:
        return True
    if len(word) == 2:
        return word in _SHORT_TERMS or any(c.isdigit() for c in word)
    return False


def tokenize(text: str) -> list[str]:
    """Search terms of `text`: lower-cased, stopwords removed, stemmed."""
    return [
        stem(word)
        for word in _TOKEN_RE.findall(text.lower())
        if _keep(word) and word not in STOPWORDS
    ]
