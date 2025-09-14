#!/usr/bin/env python3
"""
Multi-Step Analysis Demo - Advanced RST Framework

This script demonstrates the new multi-step analysis capabilities including:
- k-step probability analysis with optimization
- Information flow algorithms with entropy measures
- Path optimization using A* search and game theory
- Strategic depth evaluation for multi-step planning
"""

import sys
import time
from pathlib import Path
from typing import List, Dict

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from rst_trap_finder.core import WordAssociationGraph
from rst_trap_finder.multistep import MultiStepAnalyzer, PathInfo, MultiStepResult

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


def find_available_dataset():
    """Find the best available dataset for demonstration."""
    # Try reduced datasets first for faster demo
    reduced_files = list(Path("data/reduced").glob("reduced_rst_dataset_*.csv"))
    if reduced_files:
        return max(reduced_files, key=lambda f: f.stat().st_mtime)
    
    # Try complete dataset
    complete_path = Path("data/merged/complete_rst_dataset.csv")
    if complete_path.exists():
        return complete_path
    
    # Fallback to processed datasets
    processed_files = [
        Path("data/processed/edges_conceptnet.csv"),
        Path("data/processed/edges_usf.csv"),
    ]
    
    for file_path in processed_files:
        if file_path.exists():
            return file_path
    
    return None


def demo_k_step_analysis(analyzer: MultiStepAnalyzer, test_words: List[str]):
    """Demonstrate k-step probability analysis."""
    print("🔢 K-Step Probability Analysis")
    print("=" * 50)
    
    for word in test_words:
        if not analyzer.graph.has_word(word):
            continue
            
        print(f"\n📊 Analysis for '{word}':")
        
        # Compare different k values
        print("   Step | Exact Prob | Cumulative | Simple Method")
        print("   -----|------------|------------|-------------")
        
        for k in range(1, 6):
            # Exact probability for exactly k steps
            exact_prob = analyzer.k_step_probability_exact(word, k)
            
            # Cumulative probability within k steps
            cumulative_prob = analyzer.k_step_probability_cumulative(word, k)
            
            # Simple method for comparison
            simple_prob = analyzer.graph._k_step_rst_probability_simple(word, k)
            
            print(f"   {k:4d} | {exact_prob:10.4f} | {cumulative_prob:10.4f} | {simple_prob:11.4f}")
        
        # Show neighborhood info
        neighbors = len(analyzer.graph.get_neighbors(word))
        one_step = analyzer.graph.one_step_rst_probability(word)
        print(f"   ℹ️  {neighbors} neighbors, {one_step:.3f} one-step RST prob")


def demo_information_flow(analyzer: MultiStepAnalyzer, test_words: List[str]):
    """Demonstrate information flow and entropy calculations."""
    print(f"\n📊 Information Flow Analysis")
    print("=" * 50)
    
    for word in test_words[:3]:  # Limit for demo
        if not analyzer.graph.has_word(word):
            continue
            
        print(f"\n🔍 Information analysis for '{word}':")
        
        neighbors = analyzer.graph.get_neighbors(word)
        if not neighbors:
            print("   No neighbors found")
            continue
        
        # Calculate information gain for each neighbor
        info_gains = []
        for neighbor in list(neighbors.keys())[:5]:  # Top 5 neighbors
            info_gain = analyzer.information_gain(word, neighbor)
            info_gains.append((neighbor, info_gain))
        
        # Sort by information gain
        info_gains.sort(key=lambda x: x[1], reverse=True)
        
        print("   🎯 Top neighbors by information gain:")
        print("      Neighbor        | Info Gain | RST Prob")
        print("      ----------------|-----------|----------")
        
        for neighbor, gain in info_gains:
            rst_prob = analyzer.graph.one_step_rst_probability(neighbor)
            print(f"      {neighbor:15s} | {gain:9.4f} | {rst_prob:8.3f}")


def demo_path_optimization(analyzer: MultiStepAnalyzer, test_words: List[str]):
    """Demonstrate optimal path finding with A* search."""
    print(f"\n🛤️  Path Optimization Analysis")
    print("=" * 50)
    
    for word in test_words[:2]:  # Limit for demo
        if not analyzer.graph.has_word(word):
            continue
            
        print(f"\n🎯 Optimal paths from '{word}':")
        
        start_time = time.time()
        optimal_paths = analyzer.find_optimal_paths(word, max_steps=4, max_paths=5)
        analysis_time = time.time() - start_time
        
        if not optimal_paths:
            print("   No paths to RST targets found")
            continue
        
        print(f"   Found {len(optimal_paths)} paths in {analysis_time:.3f}s")
        print()
        
        for i, path_info in enumerate(optimal_paths, 1):
            print(f"   Path {i}: {' → '.join(path_info.path)}")
            print(f"      📈 Probability: {path_info.probability:.4f}")
            print(f"      📊 Entropy: {path_info.entropy:.4f}")
            print(f"      💡 Info Gain: {path_info.information_gain:.4f}")
            print(f"      🎯 Endpoint: {path_info.rst_endpoint}")
            print()


def demo_comprehensive_analysis(analyzer: MultiStepAnalyzer, test_words: List[str]):
    """Demonstrate comprehensive multi-step analysis."""
    print(f"\n🎯 Comprehensive Multi-Step Analysis")
    print("=" * 50)
    
    # Analyze strategic depth for multiple words
    print("Analyzing strategic depth...")
    results = analyzer.compare_strategic_depth(test_words[:5], max_k=4)
    
    if not results:
        print("No results obtained")
        return
    
    # Sort by strategic score
    sorted_results = sorted(results.items(), key=lambda x: x[1].strategic_score, reverse=True)
    
    print(f"\n📊 Strategic Rankings:")
    print("Rank | Word          | Strategic | K-Step | Paths | Avg Entropy | Max Info")
    print("-----|---------------|-----------|--------|-------|-------------|----------")
    
    for i, (word, result) in enumerate(sorted_results, 1):
        print(f"{i:4d} | {word:13s} | {result.strategic_score:9.4f} | "
              f"{result.total_probability:6.3f} | {result.path_count:5d} | "
              f"{result.average_entropy:11.4f} | {result.max_information_gain:8.4f}")
    
    # Show detailed analysis for top word
    if sorted_results:
        top_word, top_result = sorted_results[0]
        print(f"\n🏆 Detailed Analysis - Top Word: '{top_word}'")
        print(f"   Strategic Score: {top_result.strategic_score:.4f}")
        print(f"   {top_result.k_steps}-Step Probability: {top_result.total_probability:.4f}")
        print(f"   Path Count: {top_result.path_count}")
        print(f"   Average Entropy: {top_result.average_entropy:.4f}")
        print(f"   Max Information Gain: {top_result.max_information_gain:.4f}")
        
        if top_result.optimal_paths:
            print(f"\n   🎯 Top Strategic Paths:")
            for i, path in enumerate(top_result.optimal_paths[:3], 1):
                print(f"      {i}. {' → '.join(path.path)} (prob: {path.probability:.4f})")


def demo_performance_comparison():
    """Demonstrate performance improvements of the multi-step framework."""
    print(f"\n⚡ Performance Comparison")
    print("=" * 50)
    
    dataset_path = find_available_dataset()
    if not dataset_path:
        print("No dataset available for performance test")
        return
    
    print(f"Loading dataset: {dataset_path.name}")
    graph = WordAssociationGraph.from_csv(dataset_path)
    analyzer = MultiStepAnalyzer(graph)
    
    # Get test words
    all_words = list(graph.get_all_words())
    test_words = [w for w in all_words[:100] if w[0].lower() not in {'r', 's', 't'}][:20]
    
    print(f"Testing with {len(test_words)} words...")
    
    # Test simple method
    start_time = time.time()
    simple_results = []
    for word in test_words:
        prob = graph._k_step_rst_probability_simple(word, 3)
        simple_results.append(prob)
    simple_time = time.time() - start_time
    
    # Test optimized method
    start_time = time.time()
    optimized_results = []
    for word in test_words:
        prob = analyzer.k_step_probability_cumulative(word, 3)
        optimized_results.append(prob)
    optimized_time = time.time() - start_time
    
    print(f"   Simple method:    {simple_time:.3f}s ({len(test_words)/simple_time:.1f} words/sec)")
    print(f"   Optimized method: {optimized_time:.3f}s ({len(test_words)/optimized_time:.1f} words/sec)")
    
    # Check accuracy
    differences = [abs(s - o) for s, o in zip(simple_results, optimized_results)]
    max_diff = max(differences) if differences else 0
    avg_diff = sum(differences) / len(differences) if differences else 0
    
    print(f"   Accuracy: max diff {max_diff:.6f}, avg diff {avg_diff:.6f}")
    
    speedup = simple_time / optimized_time if optimized_time > 0 else 1
    print(f"   Speedup: {speedup:.1f}x")


def main():
    """Run comprehensive multi-step analysis demonstration."""
    print("🚀 Multi-Step RST Analysis Framework Demo")
    print("=" * 60)
    
    # Find and load dataset
    dataset_path = find_available_dataset()
    if not dataset_path:
        print("❌ No suitable dataset found!")
        print("💡 Run dataset_integration.py to download datasets")
        return
    
    print(f"📊 Loading dataset: {dataset_path.name}")
    start_time = time.time()
    graph = WordAssociationGraph.from_csv(dataset_path)
    load_time = time.time() - start_time
    
    total_words = len(graph.get_all_words())
    total_edges = sum(len(neighbors) for neighbors in graph.graph.values())
    print(f"✅ Loaded: {total_words:,} words, {total_edges:,} edges ({load_time:.2f}s)")
    
    # Initialize multi-step analyzer
    analyzer = MultiStepAnalyzer(graph)
    
    # Test words for demonstration
    test_words = ["word", "color", "think", "big", "fast", "love", "time"]
    available_test_words = [w for w in test_words if graph.has_word(w)]
    
    if not available_test_words:
        # Fallback to any available words
        all_words = list(graph.get_all_words())
        available_test_words = [w for w in all_words[:10] if w[0].lower() not in {'r', 's', 't'}][:7]
    
    print(f"🎯 Test words: {', '.join(available_test_words)}")
    
    # Run demonstrations
    print(f"\n" + "="*60)
    print("🔬 MULTI-STEP ANALYSIS DEMONSTRATIONS")
    print("="*60)
    
    # 1. K-step probability analysis
    demo_k_step_analysis(analyzer, available_test_words)
    
    # 2. Information flow analysis
    demo_information_flow(analyzer, available_test_words)
    
    # 3. Path optimization
    demo_path_optimization(analyzer, available_test_words)
    
    # 4. Comprehensive analysis
    demo_comprehensive_analysis(analyzer, available_test_words)
    
    # 5. Performance comparison
    demo_performance_comparison()
    
    print(f"\n🎯 Multi-Step Analysis Demo Complete!")
    print("=" * 60)
    print("✅ The framework now supports:")
    print("   • K-step probability analysis with dynamic programming")
    print("   • Information flow algorithms with entropy measures")
    print("   • Path optimization using A* search and game theory")
    print("   • Strategic depth evaluation for multi-step planning")
    print("   • Performance optimizations with memoization")
    print("\n🚀 Ready for advanced RST strategic analysis!")


if __name__ == "__main__":
    main()