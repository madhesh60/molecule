# Molecule — Integration Guide

## Install

```bash
pip install -e ".[dev]"     # from a clone; Python 3.10+
molecule setup              # one-time: cache the embedding model (~90 MB) for offline use
```

No API keys, no external services, no PyTorch. After `molecule setup` it works
with the network unplugged. To skip the model download entirely (weaker on
reworded questions), set `MOLECULE_EMBEDDING_MODEL=hashing`.

---

## The one call most agents need

```python
from molecule import Molecule

mol = Molecule()               # ./.molecule/store.db, created on first use
mol.ingest("AGENTS.md")        # once; a folder works too. Edits are picked up automatically.

ctx = mol.context("how does auth work?", budget=600)
prompt_context = ctx["context"]        # ready to paste — never more than 600 tokens
if ctx["confidence"] == "low":
    ...                                # the docs probably do not cover this question
```

`ctx` also holds `items` (id, tokens, reason for each piece), `omitted` (what
did not fit, and why) and `sources`.

## Step by step, when you want control

```python
hits = mol.search("how does auth work", k=3)      # ids + summaries + reason + relevance
best = [h for h in hits if h["type"] == "chunk" and h["relevance"] != "low"]
body = mol.fetch([best[0]["id"]])[0]["content"]   # one section, in full
related = mol.fetch([r["id"] for r in best[0].get("related", [])])  # what it links to

mol.feedback(best[0]["id"], useful=True, query="how does auth work")   # teach the router
```

---

## LangChain agent

```python
from langchain.agents import tool
from molecule import Molecule

mol = Molecule()
mol.ingest("docs/")

@tool
def project_docs(question: str) -> str:
    """Look up how this project does things (rules, architecture, conventions)."""
    ctx = mol.context(question, budget=800)
    if ctx["confidence"] == "low":
        return "The project docs do not seem to cover this."
    return ctx["context"]

@tool
def remember_fact(fact: str) -> str:
    """Store a decision or preference to remember across sessions."""
    result = mol.remember(fact, confidence=0.9)
    return "Remembered." if result.get("stored") else f"Not stored: {result}"

@tool
def recall_facts(question: str) -> str:
    """Recall decisions and preferences from earlier sessions."""
    memories = mol.recall(question, k=3)
    return "\n".join(m["text"] for m in memories if m["relevance"] != "low") or "Nothing recalled."
```

---

## CrewAI agent

```python
from crewai import Agent
from crewai.tools import tool
from molecule import Molecule

mol = Molecule()
mol.ingest("AGENTS.md")

@tool("Search project docs")
def search_docs(question: str) -> str:
    """Answer material from the project docs for one question."""
    return mol.context(question, budget=800)["context"] or "No relevant docs."

researcher = Agent(
    role="Researcher",
    goal="Answer questions using project documentation",
    backstory="Knows where everything in the project docs lives.",
    tools=[search_docs],
)
```

---

## Claude Code / Cursor (MCP server)

`~/.claude/settings.json` (Claude Code) or `.cursor/mcp.json` (Cursor):

```json
{
  "mcpServers": {
    "molecule": {
      "command": "molecule",
      "args": ["serve"],
      "cwd": "/path/to/your/repo"
    }
  }
}
```

The server sends usage instructions to the client when it connects. In practice:

```
mol_ingest("AGENTS.md")                       # once
mol_context("how does chunking work?", 600)   # most questions
mol_search(...) → mol_fetch(...)              # when you want to pick sections yourself
mol_feedback(chunk_id, true, "the question")  # after a section helped
mem_extract("<the conversation>")             # at the end of a task
mol_stats()                                   # savings so far
```

Works with the MCP Python SDK 1.x and 2.x.

## Talking to the server from your own code

```python
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

params = StdioServerParameters(command="molecule", args=["serve"])
async with stdio_client(params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        await session.call_tool("mol_ingest", {"file_path": "AGENTS.md"})
        reply = await session.call_tool("mol_context", {"query": "how is auth done?", "budget": 600})
        print(json.loads(reply.content[0].text)["context"])
```

---

## Custom agent loop

```python
from molecule import Molecule

class MyAgent:
    def __init__(self) -> None:
        self.mol = Molecule()
        self.mol.ingest("docs/")              # incremental: unchanged files are skipped

    def run(self, task: str) -> str:
        ctx = self.mol.context(task, budget=800)          # docs + relevant memories
        prompt = f"Task: {task}\n\nProject context:\n{ctx['context']}"
        result = your_llm(prompt)

        for item in ctx["items"]:                          # say what helped
            if item["type"] == "chunk":
                self.mol.feedback(item["id"], useful=True, query=task)
        self.mol.extract(f"{task}\n{result}", source_task="agent")   # remember decisions
        return result
```

---

## Configuration

```python
from pathlib import Path
from molecule import Molecule, MoleculeConfig

mol = Molecule(root="/path/to/repo")          # store at /path/to/repo/.molecule/store.db

mol = Molecule(config=MoleculeConfig(         # explicit config (validated)
    db_path=Path("/path/to/repo/.molecule/store.db"),
    context_budget=800,
    max_chunk_tokens=300,
))
```

Or `.molecule/config.json` in the repo, or environment variables — **every**
setting is available as `MOLECULE_<NAME>`:

```bash
MOLECULE_ROOT=/path/to/repo
MOLECULE_CONTEXT_BUDGET=800
MOLECULE_EMBEDDING_MODEL=hashing        # zero-download mode
MOLECULE_AUTO_SYNC=false
```

Precedence: defaults < `config.json` < environment. Bad values are reported on
stderr and ignored.

---

## Full API reference

```python
mol = Molecule(root=None, config=None)          # also a context manager: with Molecule() as mol:

# Context
mol.ingest(path, force=False)                   # file or folder
mol.sync(prune=False)                           # re-index edited files (also automatic)
mol.context(query, budget=None, compress=True, include_memories=True)
mol.index(max_tokens=None, prefix=None)         # table of contents that fits
mol.search(query, k=None)                       # ids + summaries + reason + relevance + related
mol.fetch(chunk_ids)                            # full content (+ related)
mol.bonds(chunk_id)                             # linked chunks
mol.sources(); mol.remove_source(name)

# Memory
mol.remember(text, source_task="agent", confidence=0.8, mem_type="preference")
mol.recall(query, k=None)
mol.forget(memory_id)
mol.extract(transcript, source_task="session")  # up to 5 facts, no LLM call
mol.memory_history(memory_id)                   # supersession chain
mol.merge_duplicates()

# Learning
mol.feedback(chunk_id, useful, query="")        # applies to similar questions
mol.feedback_stats()

# Operations
mol.stats()                                     # session + lifetime savings, by kind
mol.health(); mol.metrics(); mol.doctor()
mol.evaluate(queries, k=3, ablate=False)
mol.export_snapshot(path); mol.import_snapshot(path); mol.backup(path)
mol.close()
```
