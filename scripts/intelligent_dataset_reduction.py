#!/usr/bin/env python3
"""
Intelligent Dataset Reduction - Smart pruning for RST analysis

This module implements intelligent pruning algorithms that:
1. Maintain long chains for debugging and path analysis
2. Preserve high-value trap words and pathways
3. Remove low-value edges while keeping graph connectivity
4. Create smaller datasets optimized for testing speed

Strategy:
- Keep all edges with high RST probability
- Preserve paths that lead to multiple R/S/T words
- Maintain graph connectivity using centrality measures
- Remove redundant low-weight edges
"""

import pandas as pd
import json
import networkx as nx
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import time
import math
from collections import defaultdict, deque

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


class IntelligentDatasetReducer:
    """Smart dataset reduction maintaining quality and connectivity."""
    
    def __init__(self, input_path: Path, target_size_factor: float = 0.3):
        """
        Initialize the reducer.
        
        Args:
            input_path: Path to the complete dataset CSV
            target_size_factor: Target size as fraction of original (0.3 = 30%)
        """
        self.input_path = input_path
        self.target_size_factor = target_size_factor
        self.output_dir = Path("data/reduced")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # RST letters for trap analysis
        self.trap_letters = {'r', 's', 't'}
        
        # Load and analyze the dataset
        self.df = pd.read_csv(input_path)
        self.graph = self._build_networkx_graph()
        
        print(f"📊 Original dataset: {len(self.df):,} edges, {len(self.graph.nodes):,} words")
        print(f"🎯 Target reduction: {target_size_factor:.1%} → ~{int(len(self.df) * target_size_factor):,} edges")
    
    def _build_networkx_graph(self) -> nx.DiGraph:
        """Build NetworkX graph for centrality analysis."""
        G = nx.DiGraph()
        
        for _, row in self.df.iterrows():
            src, dst, weight = row['src'], row['dst'], row['weight']
            G.add_edge(src, dst, weight=weight)
        
        return G
    
    def _calculate_rst_value(self, word: str) -> float:
        """Calculate RST trap value for a word."""
        if not word or not isinstance(word, str) or len(word) == 0:
            return 0.0
        
        # Direct RST value
        if word[0].lower() in self.trap_letters:
            return 1.0
        
        return 0.0
    
    def _calculate_edge_importance(self) -> Dict[Tuple[str, str], float]:
        """Calculate importance score for each edge."""
        print("🧠 Calculating edge importance scores...")
        
        edge_scores = {}
        
        # Get centrality measures
        print("   📊 Computing centrality measures...")
        try:
            pagerank = nx.pagerank(self.graph, weight='weight', max_iter=50)
            betweenness = nx.betweenness_centrality(self.graph, weight='weight', k=min(1000, len(self.graph.nodes)))
        except:
            # Fallback for large graphs
            pagerank = {node: 1.0 / len(self.graph.nodes) for node in self.graph.nodes}
            betweenness = {node: 0.0 for node in self.graph.nodes}
        
        print("   🎯 Scoring edges...")
        for src, dst, data in tqdm(self.graph.edges(data=True), desc="Scoring edges"):
            weight = data['weight']
            
            # Component scores
            weight_score = min(weight / 5.0, 1.0)  # Normalize high weights
            src_centrality = pagerank.get(src, 0) * 10
            dst_centrality = pagerank.get(dst, 0) * 10
            betweenness_score = (betweenness.get(src, 0) + betweenness.get(dst, 0)) * 5
            
            # RST trap value
            dst_rst_value = self._calculate_rst_value(dst)
            src_rst_value = self._calculate_rst_value(src)
            rst_score = dst_rst_value * 2.0 + src_rst_value * 0.5
            
            # Path length preservation (favor edges that maintain connectivity)
            connectivity_bonus = 0.0
            if len(list(self.graph.predecessors(dst))) > 1 and len(list(self.graph.successors(src))) > 1:
                connectivity_bonus = 0.3
            
            # Combine scores
            importance = (
                weight_score * 0.25 +
                (src_centrality + dst_centrality) * 0.25 +
                betweenness_score * 0.15 +
                rst_score * 0.30 +
                connectivity_bonus * 0.05
            )
            
            edge_scores[(src, dst)] = importance
        
        return edge_scores
    
    def _find_long_chains(self, min_length: int = 3) -> Set[Tuple[str, str]]:
        """Find edges that are part of long chains for debugging."""
        print("🔗 Identifying long chains for preservation...")
        
        chain_edges = set()
        
        # Find paths that lead to RST words
        rst_words = [word for word in self.graph.nodes if self._calculate_rst_value(word) > 0]
        
        print(f"   🎯 Found {len(rst_words)} RST target words")
        
        # Sample a subset for performance
        sample_nodes = list(self.graph.nodes)[:min(2000, len(self.graph.nodes))]
        
        for start_node in tqdm(sample_nodes, desc="Finding chains"):
            try:
                # Find paths to RST words within reasonable length
                for rst_word in rst_words[:50]:  # Limit targets for performance
                    if start_node == rst_word:
                        continue
                    
                    try:
                        # Find shortest path
                        if nx.has_path(self.graph, start_node, rst_word):
                            path = nx.shortest_path(self.graph, start_node, rst_word, weight=None)
                            
                            if len(path) >= min_length:
                                # Add all edges in this path to preservation set
                                for i in range(len(path) - 1):
                                    chain_edges.add((path[i], path[i + 1]))
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
                        
            except Exception:
                continue
        
        print(f"   ✅ Preserved {len(chain_edges):,} chain edges")
        return chain_edges
    
    def _preserve_high_value_neighborhoods(self) -> Set[Tuple[str, str]]:
        """Preserve neighborhoods around high-value trap words."""
        print("🏆 Preserving high-value word neighborhoods...")
        
        preserved_edges = set()
        
        # Find words with high RST probability
        high_value_words = []
        
        for word in self.graph.nodes:
            if self._calculate_rst_value(word) > 0:
                # Count RST neighbors
                rst_neighbors = sum(1 for neighbor in self.graph.successors(word) 
                                  if self._calculate_rst_value(neighbor) > 0)
                
                out_degree = self.graph.out_degree(word)
                if out_degree > 0:
                    rst_ratio = rst_neighbors / out_degree
                    if rst_ratio > 0.3 or rst_neighbors >= 2:  # High RST potential
                        high_value_words.append(word)
        
        print(f"   🎯 Found {len(high_value_words)} high-value words")
        
        # Preserve all edges from/to high-value words
        for word in high_value_words:
            # Outgoing edges
            for successor in self.graph.successors(word):
                preserved_edges.add((word, successor))
            
            # Incoming edges (up to 3 highest weight)
            incoming = [(pred, word, self.graph[pred][word]['weight']) 
                       for pred in self.graph.predecessors(word)]
            incoming.sort(key=lambda x: x[2], reverse=True)
            
            for pred, word, weight in incoming[:3]:
                preserved_edges.add((pred, word))
        
        print(f"   ✅ Preserved {len(preserved_edges):,} high-value edges")
        return preserved_edges
    
    def reduce_dataset(self) -> Tuple[Path, Dict]:
        """Apply intelligent reduction algorithm."""
        print("🧠 Intelligent Dataset Reduction")
        print("=" * 50)
        
        # Calculate edge importance
        edge_scores = self._calculate_edge_importance()
        
        # Find preservation sets
        chain_edges = self._find_long_chains()
        high_value_edges = self._preserve_high_value_neighborhoods()
        
        # Combine preservation sets
        must_keep = chain_edges | high_value_edges
        print(f"🔒 Must preserve: {len(must_keep):,} edges")
        
        # Sort all edges by importance
        print("📊 Ranking edges by importance...")
        sorted_edges = sorted(edge_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate target size
        target_edges = int(len(self.df) * self.target_size_factor)
        
        # Select edges to keep
        selected_edges = set()
        
        # First, add all must-keep edges
        for edge in must_keep:
            if edge in edge_scores:  # Ensure edge exists
                selected_edges.add(edge)
        
        # Add highest-scoring edges until we reach target
        for (src, dst), score in sorted_edges:
            if len(selected_edges) >= target_edges:
                break
            
            selected_edges.add((src, dst))
        
        print(f"✅ Selected {len(selected_edges):,} edges for reduced dataset")
        
        # Create reduced dataframe
        reduced_rows = []
        for _, row in self.df.iterrows():
            if (row['src'], row['dst']) in selected_edges:
                reduced_rows.append(row)
        
        reduced_df = pd.DataFrame(reduced_rows)
        
        # Save reduced dataset
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"reduced_rst_dataset_{timestamp}.csv"
        reduced_df.to_csv(output_path, index=False)
        
        # Calculate statistics
        original_words = set(self.df['src']) | set(self.df['dst'])
        reduced_words = set(reduced_df['src']) | set(reduced_df['dst'])
        
        # Analyze preservation effectiveness
        preserved_chains = len(chain_edges & selected_edges)
        preserved_high_value = len(high_value_edges & selected_edges)
        
        stats = {
            "original_edges": len(self.df),
            "reduced_edges": len(reduced_df),
            "reduction_ratio": len(reduced_df) / len(self.df),
            "original_words": len(original_words),
            "reduced_words": len(reduced_words),
            "word_retention": len(reduced_words) / len(original_words),
            "preserved_chains": preserved_chains,
            "preserved_high_value": preserved_high_value,
            "chain_preservation_rate": preserved_chains / len(chain_edges) if chain_edges else 0,
            "high_value_preservation_rate": preserved_high_value / len(high_value_edges) if high_value_edges else 0
        }
        
        # Save metadata
        metadata = {
            "created_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_dataset": str(self.input_path),
            "reduction_strategy": "intelligent_preservation",
            "target_size_factor": self.target_size_factor,
            "algorithm_components": [
                "Edge importance scoring (weight + centrality + RST value)",
                "Long chain preservation for debugging",
                "High-value neighborhood preservation",
                "Connectivity maintenance"
            ],
            "statistics": stats,
            "quality_metrics": {
                "chain_preservation": f"{stats['chain_preservation_rate']:.1%}",
                "high_value_preservation": f"{stats['high_value_preservation_rate']:.1%}",
                "word_retention": f"{stats['word_retention']:.1%}",
                "size_reduction": f"{(1 - stats['reduction_ratio']):.1%}"
            }
        }
        
        metadata_path = self.output_dir / f"reduced_rst_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Print results
        print(f"\n✅ Intelligent Reduction Complete:")
        print(f"   📁 Output: {output_path}")
        print(f"   📊 Size: {len(reduced_df):,} edges ({stats['reduction_ratio']:.1%} of original)")
        print(f"   📝 Words: {len(reduced_words):,} ({stats['word_retention']:.1%} retention)")
        print(f"   🔗 Chain preservation: {stats['chain_preservation_rate']:.1%}")
        print(f"   🏆 High-value preservation: {stats['high_value_preservation_rate']:.1%}")
        
        return output_path, stats


def create_multiple_reductions(input_path: Path):
    """Create datasets of different sizes for various use cases."""
    print("🎯 Creating Multiple Reduced Datasets")
    print("=" * 45)
    
    sizes = [
        (0.1, "tiny", "Ultra-fast testing"),
        (0.3, "small", "Quick development"),
        (0.5, "medium", "Balanced analysis"),
        (0.7, "large", "Comprehensive testing")
    ]
    
    results = {}
    
    for factor, name, description in sizes:
        print(f"\n📊 Creating {name} dataset ({factor:.0%})...")
        reducer = IntelligentDatasetReducer(input_path, factor)
        output_path, stats = reducer.reduce_dataset()
        
        results[name] = {
            "path": output_path,
            "stats": stats,
            "description": description
        }
        
        print(f"   ✅ {name}: {stats['reduced_edges']:,} edges")
    
    # Summary
    print(f"\n📈 Reduction Summary:")
    for name, data in results.items():
        stats = data['stats']
        print(f"   • {name:6s}: {stats['reduced_edges']:6,} edges ({stats['reduction_ratio']:4.1%}) - {data['description']}")
    
    return results


def test_reduced_dataset(dataset_path: Path):
    """Test a reduced dataset with the RST framework."""
    print(f"\n🧪 Testing Reduced Dataset: {dataset_path.name}")
    print("=" * 40)
    
    try:
        import sys
        sys.path.append('.')
        from rst_trap_finder.core import WordAssociationGraph
        
        # Load and test
        graph = WordAssociationGraph.from_csv(dataset_path)
        words = graph.get_all_words()
        edges = sum(len(neighbors) for neighbors in graph.graph.values())
        
        print(f"✅ Dataset loaded successfully:")
        print(f"   📊 {len(words):,} words, {edges:,} edges")
        print(f"   📈 {edges / len(words):.1f} average edges per word")
        
        # Test sample words
        test_words = ['start', 'color', 'run', 'think', 'red', 'big']
        available_words = [w for w in test_words if graph.has_word(w)]
        
        if available_words:
            print(f"\n   🎯 Sample analysis:")
            for word in available_words[:4]:
                rst_prob = graph.one_step_rst_probability(word)
                neighbors = len(graph.get_neighbors(word))
                print(f"      {word:8s}: {rst_prob:.3f} RST prob, {neighbors:3d} neighbors")
        
        # Quick ranking test
        print(f"\n   🏆 Quick ranking test (top 5)...")
        start_time = time.time()
        top_words = graph.rank_words(top_k=5)
        ranking_time = time.time() - start_time
        
        print(f"   ⚡ Ranking time: {ranking_time:.2f}s")
        for i, (word, score) in enumerate(top_words[:3]):
            print(f"      {i+1}. {word:12s}: {score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Main reduction pipeline."""
    input_path = Path("data/merged/complete_rst_dataset.csv")
    
    if not input_path.exists():
        print(f"❌ Complete dataset not found: {input_path}")
        print("   Run complete_dataset_merger.py first")
        return
    
    # Create multiple sized reductions
    results = create_multiple_reductions(input_path)
    
    # Test the small dataset
    if "small" in results:
        test_reduced_dataset(results["small"]["path"])
    
    print(f"\n🎯 Intelligent Dataset Reduction Complete!")
    print(f"   Created {len(results)} optimized datasets for different use cases")
    print(f"   All datasets preserve long chains and high-value trap pathways")
    print(f"   Ready for fast testing and development!")


if __name__ == "__main__":
    main()