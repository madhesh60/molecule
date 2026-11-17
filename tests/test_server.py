"""The MCP layer: every tool is registered, thin, and survives failures."""
from __future__ import annotations

import asyncio

import pytest

from molecule import server


@pytest.fixture
def tools(ingested):
    """Server tools bound to a hermetic engine."""
    server.set_engine(ingested)
    yield server
    server.set_engine(None)


def run(coro):
    return asyncio.run(coro)


EXPECTED_TOOLS = {
    "mol_context", "mol_load_index", "mol_search", "mol_fetch", "mol_ingest", "mol_sync",
    "mol_list_sources", "mol_remove_source", "mol_feedback", "mol_feedback_stats",
    "mem_write", "mem_search", "mem_extract", "mol_forget", "mem_merge_duplicates",
    "mol_eval", "mol_health", "mol_stats",
}


def test_all_tools_are_registered_with_descriptions():
    registered = run(server.mcp.list_tools())
    assert {t.name for t in registered} == EXPECTED_TOOLS
    assert all(t.description and len(t.description) > 30 for t in registered)


def test_importing_the_server_creates_no_store(tmp_path, monkeypatch):
    """The engine is built lazily — importing must not touch the filesystem."""
    import importlib

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MOLECULE_ROOT", raising=False)
    server.set_engine(None)
    importlib.reload(server)
    assert not (tmp_path / ".molecule").exists()


def test_search_fetch_feedback_flow(tools):
    results = run(tools.mol_search("how do I roll back a release", 3))
    top = results[0]["id"]
    assert top == "handbook.md::Service Handbook:Rollback"
    assert "content" not in results[0]
    chunks = run(tools.mol_fetch([top]))
    assert "deployctl rollback" in chunks[0]["content"]
    vote = run(tools.mol_feedback(top, True, "how do I roll back a release"))
    assert vote["learned_weight"] > 1.0
    assert run(tools.mol_feedback_stats())["votes"] == 1


def test_context_tool_uses_the_default_budget_when_zero(tools):
    result = run(tools.mol_context("how do I roll back a release"))
    assert result["budget"] == tools.get_engine().config.context_budget
    assert "deployctl rollback" in result["context"]
    assert run(tools.mol_context("how do I roll back a release", 90))["tokens_used"] <= 90


def test_index_tool_fits_the_default_budget(tools):
    import json

    from molecule.core.tokens import count_tokens

    index = run(tools.mol_load_index())
    assert count_tokens(json.dumps(index)) <= tools.get_engine().config.index_token_budget
    drilled = run(tools.mol_load_index(2000, "handbook.md::Service Handbook:Roll"))
    assert [e["chunk_id"] for e in drilled] == ["handbook.md::Service Handbook:Rollback"]


def test_memory_tools(tools):
    stored = run(tools.mem_write("User prefers small pull requests", "review", 0.9))
    assert stored["stored"] is True
    hits = run(tools.mem_search("pull request size", 3))
    assert hits[0]["memory_id"] == stored["memory_id"]
    extracted = run(tools.mem_extract("We decided to always squash merge. The sky is blue."))
    assert extracted["stored"] == 1
    assert run(tools.mem_merge_duplicates()) == {"merged_groups": 0}
    assert run(tools.mol_forget(stored["memory_id"])) == {"deleted": stored["memory_id"]}
    assert "error" in run(tools.mol_forget("nope"))


def test_source_and_ops_tools(tools, sample_doc):
    assert run(tools.mol_list_sources())[0]["source_file"] == "handbook.md"
    assert run(tools.mol_ingest(str(sample_doc)))["unchanged"] is True
    assert run(tools.mol_sync())["fresh"] == 1
    assert run(tools.mol_health())["chunks_indexed"] == 5
    assert "savings_pct" in run(tools.mol_stats())
    evaluation = run(tools.mol_eval([{"query": "undo a release", "answer": "deployctl rollback"}]))
    assert evaluation["hits"] == 1
    assert run(tools.mol_remove_source("handbook.md"))["chunks_deleted"] == 5


def test_tool_errors_are_returned_not_raised(tools, monkeypatch):
    """A failing tool must hand back a structured error; the server survives."""
    def boom(*args, **kwargs):
        raise RuntimeError("database is on fire")

    monkeypatch.setattr(tools.get_engine(), "search", boom)
    assert run(tools.mol_search("anything")) == {"error": "RuntimeError: database is on fire"}
    # The next call on a healthy tool still works.
    assert run(tools.mol_health())["status"] == "ok"


def test_bad_arguments_are_structured_errors(tools):
    assert "error" in run(tools.mol_ingest("/no/such/file.md"))
    assert "error" in run(tools.mol_feedback("nope::chunk", True))
    assert "error" in run(tools.mol_eval([]))
    assert "error" in run(tools.mem_write("x", "t", 0.9, "not-a-type"))


# --- Over the wire: a real MCP client talking to `molecule serve` -------------------

def test_stdio_server_end_to_end(tmp_path):
    """Spawn the server as a subprocess and drive it with the official MCP
    client over stdio — the same protocol Claude Code and Cursor speak."""
    import json
    import os
    import sys

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    from tests.conftest import SAMPLE_DOC, SRC

    (tmp_path / "handbook.md").write_text(SAMPLE_DOC, encoding="utf-8")
    env = {
        **os.environ,
        "MOLECULE_ROOT": str(tmp_path),
        "MOLECULE_EMBEDDING_MODEL": "hashing",
        "PYTHONPATH": str(SRC),
    }
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "molecule", "serve"], env=env, cwd=str(tmp_path)
    )

    async def session_flow() -> dict:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                names = {t.name for t in (await session.list_tools()).tools}
                ingest = await session.call_tool("mol_ingest", {"file_path": "handbook.md"})
                context = await session.call_tool(
                    "mol_context", {"query": "how do I roll back a release", "budget": 300}
                )
                search = await session.call_tool("mol_search", {"query": "rollback", "k": 2})
                stats = await session.call_tool("mol_stats", {})
                return {
                    "names": names,
                    "ingest": json.loads(ingest.content[0].text),
                    "context": json.loads(context.content[0].text),
                    "search_blocks": len(search.content),
                    "stats": json.loads(stats.content[0].text),
                }

    out = run(session_flow())
    assert out["names"] == EXPECTED_TOOLS
    assert out["ingest"]["new"] == 5
    assert "deployctl rollback" in out["context"]["context"]
    assert out["context"]["tokens_used"] <= 300
    assert out["search_blocks"] >= 1
    assert out["stats"]["by_kind"]["context"]["calls"] == 1
    # stdout carried only protocol frames; the store was created in the repo root.
    assert (tmp_path / ".molecule" / "store.db").exists()
