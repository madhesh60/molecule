# Molecule — Integration Guide

## Install

```bash
pip install molecule-mcp
```

That's it. No API keys. No external services. Works offline.

---

## 3-line integration (any Python agent)

```python
from molecule import Molecule

mol = Molecule()
mol.ingest("AGENTS.md")  # do this once

# Every session:
chunks = mol.search("how does auth work")
content = mol.fetch([chunks[0]["id"]])
print(content[0]["content"])  # only the relevant section
```

---

## LangChain agent

```python
from langchain.agents import tool
from molecule import Molecule

mol = Molecule()
mol.ingest("AGENTS.md")

@tool
def search_docs(query: str) -> str:
    """Search project documentation for relevant context."""
    results = mol.search(query, k=3)
    if not results:
        return "No relevant docs found."
    chunks = mol.fetch([results[0]["id"]])
    return chunks[0]["content"]

@tool
def remember_fact(fact: str) -> str:
    """Store a fact to remember across sessions."""
    mol.remember(fact)
    return "Remembered."

@tool
def recall_facts(query: str) -> str:
    """Recall relevant facts from memory."""
    memories = mol.recall(query, k=3)
    return "\n".join(m["text"] for m in memories) or "Nothing recalled."
```

---

## CrewAI agent

```python
from crewai import Agent, Task, Crew
from crewai_tools import tool
from molecule import Molecule

mol = Molecule()
mol.ingest("AGENTS.md")

@tool("Search project docs")
def search_docs(query: str) -> str:
    results = mol.search(query, k=3)
    if not results:
        return "No results."
    return mol.fetch([results[0]["id"]])[0]["content"]

researcher = Agent(
    role="Researcher",
    goal="Answer questions using project documentation",
    tools=[search_docs],
    verbose=True,
)
```

---

## Claude Code / Cursor (MCP server)

Add to `~/.claude/settings.json`:

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

Then in Claude Code:
```
mol_ingest("AGENTS.md")
mol_search("how does chunking work")
mol_fetch(["AGENTS.md::Auto-Chunker"])
mol_stats()
```

---

## Custom agent (raw Python)

```python
from molecule import Molecule

class MyAgent:
    def __init__(self):
        self.mol = Molecule()
        # Ingest your docs once
        self.mol.ingest("AGENTS.md")
        self.mol.ingest("CRAWL.md")

    def run(self, task: str) -> str:
        # 1. Find relevant context (cheap)
        candidates = self.mol.search(task, k=3)

        # 2. Fetch only what's needed (targeted)
        chunk_ids = [c["id"] for c in candidates if c["type"] == "chunk"]
        context = self.mol.fetch(chunk_ids) if chunk_ids else []

        # 3. Recall relevant memories (cross-session)
        memories = self.mol.recall(task, k=3)

        # 4. Build prompt with only relevant context
        prompt = self._build_prompt(task, context, memories)

        # 5. Call your LLM here ...
        result = your_llm(prompt)

        # 6. Remember key facts for next session
        self.mol.remember(f"Task '{task}' completed: {result[:100]}", source_task=task)

        return result

    def _build_prompt(self, task, context, memories):
        parts = [f"Task: {task}"]
        if context:
            parts.append("Relevant docs:\n" + "\n---\n".join(c["content"] for c in context))
        if memories:
            parts.append("From memory:\n" + "\n".join(m["text"] for m in memories))
        return "\n\n".join(parts)
```

---

## Configuration

```python
from molecule import Molecule, MoleculeConfig

# Custom config
mol = Molecule(config=MoleculeConfig(
    max_chunk_tokens=300,       # smaller chunks
    default_k=3,                # fewer results
    embedding_model="all-MiniLM-L6-v2",  # default, CPU-only
))

# Or point at a specific repo root
mol = Molecule(root="/path/to/repo")

# Or via env vars (useful for Docker/CI)
# MOLECULE_ROOT=/path/to/repo
# MOLECULE_MAX_CHUNK_TOKENS=300
# MOLECULE_DEFAULT_K=3
# MOLECULE_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

---

## Full API reference

```python
mol = Molecule(root=None, config=None)

# Context
mol.ingest("file.md", force=False)   # chunk + index a file
mol.index()                           # cheap table of contents
mol.search("query", k=5)             # find relevant chunks + memories
mol.fetch(["chunk_id"])              # get full content
mol.sources()                         # list ingested files
mol.remove_source("file.md")         # remove a file from index

# Memory
mol.remember("fact", source_task="task", confidence=0.8, mem_type="preference")
mol.recall("query", k=5)             # retrieve relevant memories
mol.forget("memory_id")              # delete a memory
mol.extract(transcript, source_task="session")  # auto-extract up to 5 facts, no LLM call

# Self-learning router
mol.feedback("chunk_id", useful=True, query="the query")  # teach the router
mol.feedback_stats()                 # what the router has learned

# Telemetry
mol.stats()    # {"tokens_used": 300, "tokens_baseline": 5000, "savings_pct": 94.0}
mol.health()   # {"status": "ok", "chunks_indexed": 42, "active_memories": 7}
```

---

## Publish to PyPI (make it installable)

```bash
pip install build twine
python -m build
twine upload dist/*
# then anyone can: pip install molecule-mcp
```
