"""
RST Trap Finder - A clean, consolidated toolkit for word association graph analysis.

This package provides tools for analyzing word association graphs to identify
"trap words" that effectively funnel opponents toward words starting with 
specific letters (R, S, T by default).

Usage:
    from rst_trap_finder import load_graph
    
    # Load graph from CSV
    graph = load_graph("data/edges.csv")
    
    # Analyze words
    top_words = graph.rank_words(top_k=10)
    analysis = graph.get_word_analysis("color")
    
    # Get recommendations
    recommendations = graph.recommend_next_word("start")
"""
from __future__ import annotations

from .core import (
    WordAssociationGraph,
    load_graph,
    TRAP_LETTERS,
    Graph,
    ScoreDict
)

__version__ = "1.0.0"

__all__ = [
    "WordAssociationGraph",
    "load_graph", 
    "TRAP_LETTERS",
    "Graph",
    "ScoreDict",
    "__version__"
]
