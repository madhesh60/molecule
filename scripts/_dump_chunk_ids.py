"""_dump_chunk_ids.py — dev helper: ingest AGENTS.md into a temp DB and print
the real chunk ids, so eval_queries.json can be authored against reality.

Run:  python scripts/_dump_chunk_ids.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

TMP = ROOT / ".molecule_idtmp"
if TMP.exists():
    shutil.rmtree(TMP)

from molecule import Molecule  # noqa: E402

mol = Molecule(root=TMP)
result = mol.ingest(ROOT / "AGENTS.md")
print(f"# {result['total']} chunks\n")
for entry in mol.index():
    print(f"{entry['chunk_id']}  [{entry['token_count']} tok]")

shutil.rmtree(TMP)
