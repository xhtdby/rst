import pytest
from rst_trap_finder import TRAP_LETTERS
from rst_trap_finder.scores import (
    one_step_rst_prob,
    escape_hardness,
    k_step_rst_prob,
    minimax_topm,
    biased_pagerank,
)


def make_graph():
    return {
        "a": {"red": 1.0, "blue": 1.0},
        "red": {"stop": 1.0},
        "blue": {"go": 1.0},
    }


def test_basic_scores():
    g = make_graph()
    assert one_step_rst_prob("a", g, TRAP_LETTERS) == 0.5
    assert escape_hardness("a", g, TRAP_LETTERS, 0.05) == 0.5
    assert k_step_rst_prob("a", g, TRAP_LETTERS, 2) == 0.75
    mm = minimax_topm("a", g, TRAP_LETTERS, m=2, beta=1.0)
    assert pytest.approx(mm, rel=1e-5) == 0.25


def test_pagerank_normalized():
    g = make_graph()
    pr = biased_pagerank(g, TRAP_LETTERS, alpha=1.5, iters=10)
    assert all(v >= 0 for v in pr.values())
    assert pytest.approx(sum(pr.values()), rel=1e-6) == 1.0
