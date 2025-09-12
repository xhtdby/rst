"""Scoring functions for trap analysis."""
from __future__ import annotations

import math
from typing import FrozenSet, Mapping

from . import TRAP_LETTERS
from .graph import out_sums, normalize, Graph


def one_step_rst_prob(u: str, W_out: Mapping[str, Mapping[str, float]], trap: FrozenSet[str] = TRAP_LETTERS) -> float:
    """Probability that a one-step reply starts with a trap letter.

    S1(u) = sum_{v: v[0] in L} w(u,v) / sum_{v} w(u,v)
    """
    outs = W_out.get(u, {})
    if not outs:
        return 0.0
    total = sum(outs.values())
    trap_w = sum(w for v, w in outs.items() if v and v[0] in trap)
    return trap_w / total


def escape_hardness(u: str, W_out: Mapping[str, Mapping[str, float]], trap: FrozenSet[str] = TRAP_LETTERS, min_w_frac: float = 0.05) -> float:
    """Fraction of strong edges that lead to the trap set."""
    outs = W_out.get(u, {})
    if not outs:
        return 0.0
    mx = max(outs.values())
    thresh = min_w_frac * mx
    strong = [v for v, w in outs.items() if w >= thresh]
    if not strong:
        return 0.0
    trap_count = sum(1 for v in strong if v and v[0] in trap)
    return trap_count / len(strong)


def biased_pagerank(W_out: Mapping[str, Mapping[str, float]], trap: FrozenSet[str] = TRAP_LETTERS, alpha: float = 1.5, d: float = 0.85, iters: int = 40, tol: float = 1e-10) -> dict[str, float]:
    """Compute PageRank with biased transitions to trap letters."""
    nodes = set(W_out)
    for outs in W_out.values():
        nodes.update(outs)
    nodes_list = list(nodes)
    sums = out_sums(W_out)
    N = len(nodes_list)
    pr = {u: 1.0 / N for u in nodes_list}
    dangling = [u for u in nodes_list if not W_out.get(u)]
    for _ in range(iters):
        prev = pr.copy()
        sink_mass = d * sum(prev[u] for u in dangling) / N
        for u in nodes_list:
            pr[u] = (1 - d) / N + sink_mass
        for u in nodes_list:
            outs = normalize(u, W_out, sums, bias_alpha=alpha)
            for v, p in outs.items():
                pr[v] += d * prev[u] * p
        diff = sum(abs(pr[u] - prev[u]) for u in nodes_list)
        if diff < tol:
            break
    total = sum(pr.values())
    return {u: v / total for u, v in pr.items()}


def k_step_rst_prob(u: str, W_out: Mapping[str, Mapping[str, float]], trap: FrozenSet[str] = TRAP_LETTERS, k: int = 2) -> float:
    """Approximate probability of reaching trap within ``k`` steps."""
    if k <= 0:
        return 0.0
    p1 = one_step_rst_prob(u, W_out, trap)
    if k == 1:
        return p1
    outs = W_out.get(u, {})
    if not outs:
        return p1
    total = sum(outs.values())
    avg_next = 0.0
    for v, w in outs.items():
        avg_next += (w / total) * one_step_rst_prob(v, W_out, trap)
    return 1 - (1 - p1) * (1 - avg_next) ** (k - 1)


def minimax_topm(u: str, W_out: Mapping[str, Mapping[str, float]], trap: FrozenSet[str] = TRAP_LETTERS, m: int = 3, beta: float = 6.0) -> float:
    """Minimax with softmax mixture among top-m edges."""
    outs = sorted(W_out.get(u, {}).items(), key=lambda x: x[1], reverse=True)[:m]
    if not outs:
        return 0.0
    worst = min(1.0 if v and v[0] in trap else 0.0 for v, _ in outs)
    ws = [w for _, w in outs]
    max_w = max(ws)
    exps = [math.exp(beta * (w / max_w)) for w in ws]
    Z = sum(exps)
    soft = 0.0
    for (v, w), e in zip(outs, exps):
        soft += e / Z * (1.0 if v and v[0] in trap else 0.0)
    return 0.5 * worst + 0.5 * soft


def composite(u: str, W_out: Mapping[str, Mapping[str, float]], trap: FrozenSet[str], pr: Mapping[str, float], lambdas: tuple[float, float, float, float, float] = (0.35, 0.2, 0.25, 0.1, 0.1)) -> float:
    """Composite score combining multiple metrics."""
    l1, l2, l3, l4, l5 = lambdas
    s1 = one_step_rst_prob(u, W_out, trap)
    h = escape_hardness(u, W_out, trap)
    k2 = k_step_rst_prob(u, W_out, trap, k=2)
    mm3 = minimax_topm(u, W_out, trap, m=3)
    return l1 * s1 + l2 * h + l3 * pr.get(u, 0.0) + l4 * k2 + l5 * mm3
