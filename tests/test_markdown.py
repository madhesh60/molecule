"""Tests for the small markdown structure helpers."""
from __future__ import annotations

from molecule.core.markdown import (
    FenceTracker,
    code_fence_parts,
    split_blocks,
    split_list_items,
    split_sentences,
    split_table,
)


def test_fence_tracker_marks_code_lines():
    tracker = FenceTracker()
    lines = ["text", "```bash", "# not a heading", "```", "after"]
    assert [tracker.feed(line) for line in lines] == [False, True, True, True, False]


def test_fence_needs_matching_marker_to_close():
    tracker = FenceTracker()
    flags = [tracker.feed(line) for line in ["````md", "```", "still code", "````", "out"]]
    assert flags == [True, True, True, True, False]


def test_tilde_fence_is_not_closed_by_backticks():
    tracker = FenceTracker()
    flags = [tracker.feed(line) for line in ["~~~", "```", "~~~", "out"]]
    assert flags == [True, True, True, False]


def test_split_blocks_kinds():
    text = (
        "A paragraph\nwrapped over two lines.\n\n"
        "- item one\n- item two\n  continued\n\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n\n"
        "```python\nx = 1\n\ny = 2\n```\n\n"
        "Closing paragraph."
    )
    blocks = split_blocks(text)
    assert [b.kind for b in blocks] == ["para", "list", "table", "code", "para"]
    assert blocks[3].text == "```python\nx = 1\n\ny = 2\n```"  # blank line kept inside code
    assert "continued" in blocks[1].text


def test_list_directly_after_paragraph_starts_a_new_block():
    blocks = split_blocks("Top-level folders:\n- api\n- worker")
    assert [b.kind for b in blocks] == ["para", "list"]


def test_split_sentences_joins_soft_wraps():
    text = "The first sentence is\nwrapped. The second one follows! And a third?"
    assert split_sentences(text) == [
        "The first sentence is wrapped.",
        "The second one follows!",
        "And a third?",
    ]


def test_split_sentences_keeps_abbreviations_together():
    assert split_sentences("Use a marker, e.g. an HTML comment. Then stop.") == [
        "Use a marker, e.g. an HTML comment.",
        "Then stop.",
    ]


def test_split_list_items_keeps_continuations():
    items = split_list_items("- one\n  more of one\n- two\n  - nested\n- three")
    assert items == ["- one\n  more of one", "- two\n  - nested", "- three"]


def test_split_table_separates_header():
    header, rows = split_table("| a | b |\n|---|:-:|\n| 1 | 2 |\n| 3 | 4 |")
    assert header == "| a | b |\n|---|:-:|"
    assert rows == ["| 1 | 2 |", "| 3 | 4 |"]


def test_split_table_without_separator_has_no_header():
    header, rows = split_table("| 1 | 2 |\n| 3 | 4 |")
    assert header == ""
    assert len(rows) == 2


def test_code_fence_parts():
    opening, body, closing = code_fence_parts("```sql\nSELECT 1;\nSELECT 2;\n```")
    assert (opening, body, closing) == ("```sql", ["SELECT 1;", "SELECT 2;"], "```")


def test_unterminated_code_fence_gets_a_closing_marker():
    opening, body, closing = code_fence_parts("```py\nx = 1")
    assert opening == "```py" and body == ["x = 1"] and closing == "```"
