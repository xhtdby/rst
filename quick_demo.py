#!/usr/bin/env python3
"""
RST Trap Finder - Quick Demo

This script demonstrates the core functionality of the RST Trap Finder toolkit.
Run this to see the tool in action with the sample data.
"""

from rst_trap_finder import load_graph

def main():
    print("=" * 50)
    print("RST Trap Finder - Quick Demo")
    print("=" * 50)
    
    # Load the sample graph
    print("\n1. Loading sample data...")
    graph = load_graph("data/edges.sample.csv")
    graph.print_summary()
    
    # Find top trap words
    print("\n2. Top 5 trap words:")
    top_words = graph.rank_words(top_k=5)
    for i, (word, score) in enumerate(top_words, 1):
        print(f"  {i}. {word}: {score:.4f}")
    
    # Analyze a specific word
    print("\n3. Analyzing 'color':")
    analysis = graph.get_word_analysis("color")
    print(f"  Composite score: {analysis['composite_score']:.4f}")
    print(f"  One-step prob: {analysis['one_step_probability']:.4f}")
    print(f"  Escape hardness: {analysis['escape_hardness']:.4f}")
    
    # Get recommendations
    print("\n4. Recommendations from 'start':")
    recommendations = graph.recommend_next_word("start", top_k=3)
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec['word']}: {rec['score']:.4f}")
    
    print("\n" + "=" * 50)
    print("Demo complete! Try the CLI:")
    print("  rst-find analyze data/edges.sample.csv")
    print("  rst-find recommend start data/edges.sample.csv")
    print("=" * 50)

if __name__ == "__main__":
    main()