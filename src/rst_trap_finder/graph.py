"""Graph utilities."""
from __future__ import annotations

from typing import Dict, Mapping

from . import TRAP_LETTERS

Graph = Dict[str, Dict[str, float]]
OutSums = Dict[str, float]


def out_sums(graph: Mapping[str, Mapping[str, float]]) -> OutSums:
    """Compute outgoing weight sums for each node."""
    return {u: sum(ws.values()) for u, ws in graph.items()}


def normalize(u: str, graph: Mapping[str, Mapping[str, float]], sums: Mapping[str, float], bias_alpha: float = 1.0):
    """Return biased probability distribution over neighbors.

    Parameters
    ----------
    u: node
    graph: adjacency weights
    sums: precomputed outgoing sums
    bias_alpha: boost for neighbors starting with letters in ``TRAP_LETTERS``.
    """
    outs = graph.get(u, {})
    if not outs:
        return {}
    weights: Dict[str, float] = {}
    for v, w in outs.items():
        boost = bias_alpha if v and v[0] in TRAP_LETTERS else 1.0
        weights[v] = w * boost
    total = sum(weights.values())
    return {v: w / total for v, w in weights.items()}
