"""report.py — placeholder, replaced below once the result shapes are final."""
from pathlib import Path


def write_report(results: dict, path: Path) -> None:
    path.write_text("# Benchmark results\n\n(see results/latest.json)\n", encoding="utf-8")
