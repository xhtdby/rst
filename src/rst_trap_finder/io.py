"""IO utilities for loading word graphs."""
from __future__ import annotations

from csv import DictReader
from pathlib import Path
from typing import Dict

Graph = Dict[str, Dict[str, float]]


def load_csv(path: str | Path) -> Graph:
    """Load edges from a CSV file.

    The file must have columns ``src``, ``dst`` and ``weight``. Tokens are
    lower-cased and empty rows ignored.
    """
    graph: Graph = {}
    with Path(path).open() as fh:
        reader = DictReader(fh)
        for row in reader:
            src = (row.get("src") or "").strip().lower()
            dst = (row.get("dst") or "").strip().lower()
            if not src or not dst:
                continue
            try:
                w = float(row.get("weight", "0"))
            except ValueError:
                continue
            if w <= 0:
                continue
            outs = graph.setdefault(src, {})
            outs[dst] = outs.get(dst, 0.0) + w
    return graph
