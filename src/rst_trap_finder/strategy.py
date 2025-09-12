"""Strategy helpers for choosing next words."""
from __future__ import annotations

import math
from typing import Dict, Mapping, FrozenSet, Tuple, List

from . import TRAP_LETTERS
from .scores import (
    composite,
    one_step_rst_prob,
    escape_hardness,
)

Graph = Mapping[str, Mapping[str, float]]


def basin_strength(y: str, W_out: Graph, trap: FrozenSet[str] = TRAP_LETTERS, k: int = 5) -> float:
    """Mean one-step probability over ``y``'s top-k neighbors."""
    outs = sorted(W_out.get(y, {}).items(), key=lambda x: x[1], reverse=True)[:k]
    if not outs:
        return 0.0
    return sum(one_step_rst_prob(v, W_out, trap) for v, _ in outs) / len(outs)


def recommend_next(current: str, W_out: Graph, trap: FrozenSet[str], pr: Mapping[str, float], lambdas: Tuple[float, float, float, float, float], k: int = 5, m: int = 3) -> Dict[str, object]:
    """Recommend next word from ``current`` based on composite score."""
    outs = W_out.get(current)
    if not outs:
        raise ValueError(f"Unknown or terminal word: {current}")
    candidates: List[Dict[str, object]] = []
    for y, _w in outs.items():
        comp = composite(y, W_out, trap, pr, lambdas)
        basin = basin_strength(y, W_out, trap, k)
        outs_y = W_out.get(y, {})
        if outs_y:
            mx = max(outs_y.values())
            strong = [v for v, w in outs_y.items() if w >= 0.05 * mx and v and v[0] not in trap]
        else:
            strong = []
        non_rst_strong = len(strong)
        # simple lookahead using softmax on weights favoring non-trap
        outs_sorted = sorted(outs_y.items(), key=lambda x: x[1], reverse=True)[:m]
        if outs_sorted:
            ws = [w * (2.0 if v and v[0] not in trap else 1.0) for v, w in outs_sorted]
            exps = [math.exp(w) for w in ws]
            Z = sum(exps)
            expected = 0.0
            for (v, _), e in zip(outs_sorted, exps):
                expected += e / Z * composite(v, W_out, trap, pr, lambdas)
        else:
            expected = comp
        candidates.append({
            "word": y,
            "composite": comp,
            "basin": basin,
            "non_rst_strong_exits": non_rst_strong,
            "expected": expected,
        })
    candidates.sort(key=lambda c: (-c["composite"], c["non_rst_strong_exits"], -c["basin"]))
    best = candidates[0]
    return {"best": best, "candidates": candidates}
