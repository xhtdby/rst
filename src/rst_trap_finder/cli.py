"""
Simplified command line interface for RST Trap Finder.

This CLI provides easy access to the core functionality for analyzing
word association graphs and finding trap words.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .core import load_graph


def format_table(headers: List[str], rows: List[List[str]]) -> str:
    """Format data as a simple aligned table."""
    if not rows:
        return ""
    
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(cell)))
    
    # Format table
    format_str = " | ".join(f"{{:<{w}}}" for w in widths)
    separator = "-+-".join("-" * w for w in widths)
    
    lines = [
        format_str.format(*headers),
        separator
    ]
    
    for row in rows:
        lines.append(format_str.format(*row))
    
    return "\n".join(lines)


def cmd_analyze(args: argparse.Namespace) -> None:
    """Analyze the graph and show top trap words."""
    try:
        graph = load_graph(args.csv)
    except Exception as e:
        print(f"Error loading graph: {e}")
        sys.exit(1)
    
    print("Loading graph and computing scores...")
    graph.print_summary()
    print()
    
    # Get top words
    top_words = graph.rank_words(top_k=args.top)
    
    if not top_words:
        print("No words found in graph.")
        return
    
    print(f"Top {len(top_words)} trap words:")
    headers = ["Rank", "Word", "Score", "One-Step", "Hardness", "PageRank"]
    rows = []
    
    for i, (word, score) in enumerate(top_words, 1):
        analysis = graph.get_word_analysis(word)
        rows.append([
            str(i),
            word,
            f"{score:.4f}",
            f"{analysis['one_step_probability']:.4f}",
            f"{analysis['escape_hardness']:.4f}",
            f"{analysis['pagerank_score']:.4f}"
        ])
    
    print(format_table(headers, rows))


def cmd_word(args: argparse.Namespace) -> None:
    """Analyze a specific word in detail."""
    try:
        graph = load_graph(args.csv)
    except Exception as e:
        print(f"Error loading graph: {e}")
        sys.exit(1)
    
    word = args.word.lower()
    
    if not graph.has_word(word):
        print(f"Word '{word}' not found in graph.")
        return
    
    try:
        analysis = graph.get_word_analysis(word)
    except Exception as e:
        print(f"Error analyzing word: {e}")
        return
    
    print(f"Analysis for '{word}':")
    print(f"  Composite Score: {analysis['composite_score']:.4f}")
    print(f"  One-Step Probability: {analysis['one_step_probability']:.4f}")
    print(f"  Escape Hardness: {analysis['escape_hardness']:.4f}")
    print(f"  PageRank Score: {analysis['pagerank_score']:.4f}")
    print(f"  Neighbor Count: {analysis['neighbor_count']}")
    print(f"  Total Outgoing Weight: {analysis['total_outgoing_weight']:.2f}")
    
    if analysis['neighbors']:
        print(f"\nTop neighbors:")
        headers = ["Word", "Weight", "Is Trap"]
        rows = []
        for neighbor, weight, is_trap in analysis['neighbors']:
            rows.append([
                neighbor,
                f"{weight:.2f}",
                "Yes" if is_trap else "No"
            ])
        print(format_table(headers, rows))


def cmd_recommend(args: argparse.Namespace) -> None:
    """Recommend next words from a given word."""
    try:
        graph = load_graph(args.csv)
    except Exception as e:
        print(f"Error loading graph: {e}")
        sys.exit(1)
    
    word = args.word.lower()
    
    if not graph.has_word(word):
        print(f"Word '{word}' not found in graph.")
        return
    
    recommendations = graph.recommend_next_word(word, top_k=args.top)
    
    if not recommendations:
        print(f"No recommendations available from '{word}'.")
        return
    
    best = recommendations[0]
    print(f"Best next word from '{word}': {best['word']} (score: {best['score']:.4f})")
    print()
    
    print(f"Top {len(recommendations)} recommendations:")
    headers = ["Rank", "Word", "Score", "One-Step", "Hardness", "Weight"]
    rows = []
    
    for i, rec in enumerate(recommendations, 1):
        rows.append([
            str(i),
            rec['word'],
            f"{rec['score']:.4f}",
            f"{rec['one_step_prob']:.4f}",
            f"{rec['hardness']:.4f}",
            f"{rec['edge_weight']:.2f}"
        ])
    
    print(format_table(headers, rows))


def cmd_export(args: argparse.Namespace) -> None:
    """Export word scores to a CSV file."""
    try:
        graph = load_graph(args.csv)
    except Exception as e:
        print(f"Error loading graph: {e}")
        sys.exit(1)
    
    output_path = Path(args.output)
    
    try:
        graph.export_scores(output_path)
        print(f"Scores exported to {output_path}")
    except Exception as e:
        print(f"Error exporting scores: {e}")
        sys.exit(1)


def main(argv: Optional[List[str]] = None) -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RST Trap Finder - Analyze word association graphs for trap words",
        prog="rst-find"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.required = True
    
    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", 
        help="Analyze graph and show top trap words"
    )
    analyze_parser.add_argument(
        "csv", 
        type=Path, 
        help="Path to CSV file containing word associations"
    )
    analyze_parser.add_argument(
        "--top", 
        type=int, 
        default=20, 
        help="Number of top words to show (default: 20)"
    )
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # Word command
    word_parser = subparsers.add_parser(
        "word", 
        help="Analyze a specific word in detail"
    )
    word_parser.add_argument(
        "word", 
        help="Word to analyze"
    )
    word_parser.add_argument(
        "csv", 
        type=Path, 
        help="Path to CSV file containing word associations"
    )
    word_parser.set_defaults(func=cmd_word)
    
    # Recommend command
    recommend_parser = subparsers.add_parser(
        "recommend", 
        help="Get recommendations for next word"
    )
    recommend_parser.add_argument(
        "word", 
        help="Current word"
    )
    recommend_parser.add_argument(
        "csv", 
        type=Path, 
        help="Path to CSV file containing word associations"
    )
    recommend_parser.add_argument(
        "--top", 
        type=int, 
        default=10, 
        help="Number of recommendations to show (default: 10)"
    )
    recommend_parser.set_defaults(func=cmd_recommend)
    
    # Export command
    export_parser = subparsers.add_parser(
        "export", 
        help="Export word scores to CSV"
    )
    export_parser.add_argument(
        "csv", 
        type=Path, 
        help="Path to input CSV file containing word associations"
    )
    export_parser.add_argument(
        "output", 
        type=Path, 
        help="Path to output CSV file for scores"
    )
    export_parser.set_defaults(func=cmd_export)
    
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()