"""Command line interface using argparse."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from . import TRAP_LETTERS
from .io import load_csv
from .scores import (
    biased_pagerank,
    composite,
    one_step_rst_prob,
    escape_hardness,
    k_step_rst_prob,
    minimax_topm,
)
from .strategy import recommend_next


def parse_lambdas(s: str) -> Tuple[float, float, float, float, float]:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 5:
        raise ValueError("need five comma-separated numbers")
    return tuple(parts)  # type: ignore[return-value]


def format_table(headers: List[str], rows: List[List[object]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(f"{cell}"))
    fmt = " | ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*headers), "-+-".join("-" * w for w in widths)]
    for row in rows:
        lines.append(fmt.format(*row))
    return "\n".join(lines)


def cmd_rank(args: argparse.Namespace) -> None:
    graph = load_csv(args.csv)
    lamb = parse_lambdas(args.lambdas)
    pr = biased_pagerank(graph, TRAP_LETTERS, args.alpha)
    rows = []
    for u in graph:
        s1 = one_step_rst_prob(u, graph, TRAP_LETTERS)
        h = escape_hardness(u, graph, TRAP_LETTERS, args.min_w)
        k2 = k_step_rst_prob(u, graph, TRAP_LETTERS, args.k)
        mm = minimax_topm(u, graph, TRAP_LETTERS, args.m)
        comp = composite(u, graph, TRAP_LETTERS, pr, lamb)
        outs = graph.get(u, {})
        strong_non = sum(
            1 for v, w in outs.items() if v and v[0] not in TRAP_LETTERS and w >= args.min_w * max(outs.values())
        )
        rows.append(
            [
                u,
                f"{comp:.3f}",
                f"{s1:.3f}",
                f"{h:.3f}",
                f"{pr.get(u,0.0):.3e}",
                f"{k2:.3f}",
                f"{mm:.3f}",
                len(outs),
                strong_non,
            ]
        )
    rows.sort(key=lambda r: float(r[1]), reverse=True)
    headers = ["word", "comp", "S1", "H", "PR", "K2", "MM3", "outdeg", "strong_nonRST_exits"]
    print(format_table(headers, rows[: args.top]))


def cmd_next(args: argparse.Namespace) -> None:
    graph = load_csv(args.csv)
    lamb = parse_lambdas(args.lambdas)
    pr = biased_pagerank(graph)
    res = recommend_next(args.word, graph, TRAP_LETTERS, pr, lamb)
    best = res["best"]
    print(f"Best: {best['word']}")
    headers = ["word", "comp", "basin", "non_rst_strong_exits", "expected"]
    rows = [
        [
            c["word"],
            f"{c['composite']:.3f}",
            f"{c['basin']:.3f}",
            c["non_rst_strong_exits"],
            f"{c['expected']:.3f}",
        ]
        for c in res["candidates"]
    ]
    print(format_table(headers, rows))


def cmd_info(args: argparse.Namespace) -> None:
    graph = load_csv(args.csv)
    if args.word not in graph:
        raise ValueError("word not in graph")
    pr = biased_pagerank(graph)
    s1 = one_step_rst_prob(args.word, graph, TRAP_LETTERS)
    h = escape_hardness(args.word, graph, TRAP_LETTERS)
    k2 = k_step_rst_prob(args.word, graph, TRAP_LETTERS)
    mm = minimax_topm(args.word, graph, TRAP_LETTERS)
    comp = composite(args.word, graph, TRAP_LETTERS, pr)
    print(
        f"word: {args.word}\nS1={s1:.3f} H={h:.3f} PR={pr.get(args.word,0.0):.3e} K2={k2:.3f} MM3={mm:.3f} comp={comp:.3f}"
    )
    outs = sorted(graph[args.word].items(), key=lambda x: x[1], reverse=True)[:10]
    headers = ["dst", "weight", "RST"]
    rows = [[v, f"{w:.3f}", str(v[0] in TRAP_LETTERS)] for v, w in outs]
    print(format_table(headers, rows))


def cmd_tune(args: argparse.Namespace) -> None:
    graph = load_csv(args.csv)
    lamb = parse_lambdas(args.lambdas)
    pr = biased_pagerank(graph, TRAP_LETTERS, args.alpha)
    rows = []
    for u in graph:
        comp = composite(u, graph, TRAP_LETTERS, pr, lamb)
        rows.append([u, f"{comp:.3f}"])
    rows.sort(key=lambda r: float(r[1]), reverse=True)
    print(format_table(["word", "comp"], rows[:20]))


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rst_trap_finder")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rank = sub.add_parser("rank")
    p_rank.add_argument("--csv", type=Path, required=True)
    p_rank.add_argument("--top", type=int, default=10)
    p_rank.add_argument("--alpha", type=float, default=1.5)
    p_rank.add_argument("--min-w", dest="min_w", type=float, default=0.05)
    p_rank.add_argument("--k", type=int, default=2)
    p_rank.add_argument("--m", type=int, default=3)
    p_rank.add_argument("--lambdas", type=str, default="0.35,0.2,0.25,0.1,0.1")
    p_rank.set_defaults(func=cmd_rank)

    p_next = sub.add_parser("next")
    p_next.add_argument("--word", required=True)
    p_next.add_argument("--csv", type=Path, required=True)
    p_next.add_argument("--lambdas", type=str, default="0.35,0.2,0.25,0.1,0.1")
    p_next.set_defaults(func=cmd_next)

    p_info = sub.add_parser("info")
    p_info.add_argument("--word", required=True)
    p_info.add_argument("--csv", type=Path, required=True)
    p_info.set_defaults(func=cmd_info)

    p_tune = sub.add_parser("tune")
    p_tune.add_argument("--csv", type=Path, required=True)
    p_tune.add_argument("--alpha", type=float, default=1.5)
    p_tune.add_argument("--lambdas", type=str, default="0.35,0.2,0.25,0.1,0.1")
    p_tune.set_defaults(func=cmd_tune)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
