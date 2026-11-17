"""Tests for the passage splitter and the extract re-assembly."""
from __future__ import annotations

from molecule.core.passages import (
    join_passages,
    passage_embedding_texts,
    split_passages,
)

CONTENT = (
    "The rollback command reverts a service. It completes in under five minutes.\n\n"
    "- first item\n- second item\n\n"
    "| Level | Meaning |\n|---|---|\n| SEV1 | outage |\n| SEV2 | degraded |\n\n"
    "```bash\ndeployctl rollback api\n```"
)


def test_passage_kinds_in_document_order():
    kinds = [p.kind for p in split_passages(CONTENT)]
    assert kinds == ["prose", "prose", "list", "list", "table", "table", "code"]


def test_table_rows_carry_their_header():
    rows = [p for p in split_passages(CONTENT) if p.kind == "table"]
    assert [r.text for r in rows] == ["| SEV1 | outage |", "| SEV2 | degraded |"]
    assert all(r.header == "| Level | Meaning |\n|---|---|" for r in rows)


def test_split_is_deterministic():
    assert split_passages(CONTENT) == split_passages(CONTENT)


def test_short_fragments_attach_to_a_neighbour():
    passages = split_passages("See below. The canary runs for fifteen minutes before rollout.")
    assert len(passages) == 1


def test_cap_merges_instead_of_dropping_text():
    text = " ".join(f"This is sentence number {i} of the long paragraph." for i in range(40))
    passages = split_passages(text, max_passages=8)
    assert len(passages) <= 8
    assert all(f"number {i} " in " ".join(p.text for p in passages) for i in range(40))


def test_embedding_text_is_prefixed_with_the_heading():
    passages = split_passages("It completes in under five minutes.")
    assert passage_embedding_texts("Rollback", passages) == [
        "Rollback: It completes in under five minutes."
    ]


def test_join_marks_gaps_and_keeps_order():
    passages = split_passages(
        "First sentence is right here for you now. Second sentence is right here for you now. "
        "Third sentence is right here for you now."
    )
    text = join_passages(passages, [2, 0])
    assert text.index("First") < text.index("Third")
    assert "Second" not in text
    assert "…" in text


def test_join_table_row_brings_its_header():
    passages = split_passages(CONTENT)
    sev2 = next(i for i, p in enumerate(passages) if "SEV2" in p.text)
    text = join_passages(passages, [sev2])
    assert "| Level | Meaning |\n|---|---|\n| SEV2 | degraded |" in text
    assert "SEV1" not in text


def test_join_everything_reads_like_the_original():
    passages = split_passages(CONTENT)
    text = join_passages(passages, list(range(len(passages))))
    for needle in ("rollback command", "- first item\n- second item", "| SEV1 | outage |",
                   "deployctl rollback api"):
        assert needle in text
    assert "…" not in text
