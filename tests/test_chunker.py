from __future__ import annotations

import pytest

from molecule.config import MoleculeConfig
from molecule.core.chunker import chunk_file, chunk_text, parse_markdown
from molecule.core.markdown import split_blocks


@pytest.fixture
def sample_md(tmp_path):
    md = tmp_path / "sample.md"
    md.write_text(
        "# Section One\n\nThis is the first section with some content.\n\n"
        "## Sub Section\n\nSub content here.\n\n"
        "# Section Two\n\n<!-- always_include -->\nThis must always load.\n",
        encoding="utf-8",
    )
    return md


def _small(**overrides) -> MoleculeConfig:
    return MoleculeConfig(**{"max_chunk_tokens": 60, "min_chunk_tokens": 5, **overrides})


def test_parse_markdown_headings():
    text = "# H1\n\nBody one.\n\n## H2\n\nBody two.\n"
    sections = parse_markdown(text)
    assert len(sections) == 2
    assert sections[0].heading_path == ["H1"]
    assert sections[1].heading_path == ["H1", "H2"]


def test_always_include_detected(sample_md):
    sections = chunk_file(sample_md, MoleculeConfig())
    always = [s for s in sections if s.always_include]
    assert [s.heading_path for s in always] == [["Section Two"]]


def test_always_include_mentioned_in_inline_code_is_not_a_marker():
    """Documentation *about* the marker must not turn itself into a guardrail."""
    text = "# Docs\n\nMark a section with `<!-- always_include -->` or `always_include: true`.\n"
    assert not parse_markdown(text)[0].always_include


def test_always_include_inside_code_fence_is_not_a_marker():
    text = "# Docs\n\n```markdown\n<!-- always_include -->\nNever touch prod.\n```\n"
    assert not parse_markdown(text)[0].always_include


def test_front_matter_flag_marks_every_section():
    text = "---\nalways_include: true\n---\n# A\n\nrule a\n\n# B\n\nrule b\n"
    sections = parse_markdown(text)
    assert [s.heading_path for s in sections] == [["A"], ["B"]]
    assert all(s.always_include for s in sections)
    assert "always_include" not in sections[0].content  # front matter is not content


def test_code_comments_are_not_headings():
    """Regression: `# comment` lines in a bash block used to become H1s and
    tear the section apart."""
    text = (
        "# Setup\n\nRun this:\n\n```bash\n# Clone and bootstrap\nmake bootstrap\n\n"
        "# Start the stack\ndocker compose up\n```\n\nThen open the app.\n\n"
        "# Usage\n\nUse it.\n"
    )
    sections = parse_markdown(text)
    assert [s.heading_path for s in sections] == [["Setup"], ["Usage"]]
    assert "docker compose up" in sections[0].content
    assert "Then open the app." in sections[0].content


def test_text_before_first_heading_is_kept():
    sections = parse_markdown("Intro paragraph.\n\n# First\n\nBody.\n")
    assert sections[0].heading_path == ["(intro)"]
    assert sections[0].content == "Intro paragraph."


def test_document_without_headings_is_one_root_section():
    sections = parse_markdown("Just some text.\nMore text.")
    assert len(sections) == 1 and sections[0].heading_path == ["(root)"]


def test_headings_deeper_than_split_depth_stay_inline():
    text = "# A\n\nintro\n\n#### Detail\n\ndeep text\n"
    sections = parse_markdown(text, split_depth=3)
    assert len(sections) == 1
    assert "#### Detail" in sections[0].content


def test_repeated_headings_get_unique_paths():
    text = "# A\n\n## Example\n\n" + "first body " * 20 + "\n\n# B\n\n" + "text " * 20 + \
           "\n\n# A\n\n## Example\n\n" + "second body " * 20 + "\n"
    paths = [tuple(s.heading_path) for s in chunk_text(text, MoleculeConfig(min_chunk_tokens=5))]
    assert len(paths) == len(set(paths))
    assert ("A", "Example") in paths and ("A", "Example (2)") in paths


def test_chunk_token_bounds(tmp_path):
    cfg = MoleculeConfig(max_chunk_tokens=50, min_chunk_tokens=5)
    big = tmp_path / "big.md"
    # Create a section that exceeds max_chunk_tokens
    big.write_text("# Big\n\n" + ("word " * 200) + "\n", encoding="utf-8")
    sections = chunk_file(big, cfg)
    assert len(sections) > 1
    for s in sections:
        assert s.token_count <= cfg.max_chunk_tokens + 20  # small overshoot allowed at boundaries


def test_split_parts_are_numbered_and_ordered():
    text = "# Big\n\n" + "\n\n".join(f"Paragraph number {i} has a few words in it." for i in range(30))
    sections = chunk_text(text, _small())
    assert [s.heading_path[-1] for s in sections] == [f"Big §{n}" for n in range(1, len(sections) + 1)]
    assert [s.part for s in sections] == list(range(1, len(sections) + 1))
    assert [s.position for s in sections] == list(range(len(sections)))


def test_split_never_cuts_inside_a_small_code_block():
    code = "```python\ndef f():\n    return 1\n```"
    text = "# S\n\n" + "\n\n".join(["Some prose sentence here to fill space."] * 6 + [code] +
                                   ["More prose after the code block ends here."] * 6)
    for section in chunk_text(text, _small()):
        assert section.content.count("```") % 2 == 0  # fences always balanced
    assert any(code in s.content for s in chunk_text(text, _small()))


def test_oversized_code_block_is_refenced_in_every_part():
    code = "```python\n" + "\n".join(f"value_{i} = compute({i})" for i in range(40)) + "\n```"
    sections = chunk_text("# Code\n\n" + code, _small())
    assert len(sections) > 1
    for section in sections:
        assert section.content.startswith("```python")
        assert section.content.rstrip().endswith("```")


def test_oversized_table_repeats_its_header():
    rows = "\n".join(f"| name_{i} | value number {i} |" for i in range(40))
    sections = chunk_text("# T\n\n| Name | Value |\n|---|---|\n" + rows, _small())
    assert len(sections) > 1
    for section in sections:
        assert section.content.startswith("| Name | Value |\n|---|---|")
    joined = "\n".join(s.content for s in sections)
    assert all(f"name_{i} " in joined for i in range(40))  # no row lost


def test_split_keeps_list_structure():
    items = "\n".join(f"- item {i} ends with a period." for i in range(40))
    sections = chunk_text("# L\n\n" + items, _small())
    assert len(sections) > 1
    for section in sections:
        assert all(line.startswith("- item") for line in section.content.split("\n"))


def test_unsplit_section_keeps_formatting_byte_for_byte():
    body = "Intro line.\n\n- a\n- b\n\n| x | y |\n|---|---|\n| 1 | 2 |\n\n```sh\necho hi\n```"
    sections = chunk_text("# Keep\n\n" + body + "\n", MoleculeConfig())
    assert sections[0].content == body


def test_split_content_is_still_parseable_markdown():
    text = "# Mixed\n\n" + "\n\n".join(
        ["A sentence of prose. Another sentence follows it."] * 8
        + ["- one\n- two\n- three"] * 4
        + ["```\ncode line\n```"] * 4
    )
    for section in chunk_text(text, _small()):
        assert split_blocks(section.content)  # every part has at least one block


def test_tiny_sections_merge_into_previous_sibling():
    text = "# Top\n\n## A\n\n" + "long enough body " * 30 + "\n\n## B\n\ntiny\n"
    sections = chunk_text(text, MoleculeConfig(min_chunk_tokens=50))
    assert [s.heading_path for s in sections] == [["Top", "A"]]
    assert sections[0].content.endswith("tiny")


def test_top_level_sections_never_merge():
    sections = chunk_text("# A\n\ntiny a\n\n# B\n\ntiny b\n", MoleculeConfig())
    assert [s.heading_path for s in sections] == [["A"], ["B"]]


def test_content_hash_stable(sample_md):
    cfg = MoleculeConfig()
    s1 = chunk_file(sample_md, cfg)
    s2 = chunk_file(sample_md, cfg)
    assert [s.hash for s in s1] == [s.hash for s in s2]


def test_utf8_bom_is_ignored():
    sections = parse_markdown("\ufeff# Title\n\nBody.\n")
    assert sections[0].heading_path == ["Title"]
