"""Molecule — a local-first context router and memory layer for AI agents.

    from molecule import Molecule

    mol = Molecule()
    mol.ingest("AGENTS.md")
    print(mol.context("how does auth work?", budget=800)["context"])

See `molecule.engine.Molecule` for the full API.
"""
from __future__ import annotations

from molecule.config import MoleculeConfig, load_config
from molecule.engine import Molecule

__version__ = "0.2.0"

__all__ = ["Molecule", "MoleculeConfig", "__version__", "load_config"]
