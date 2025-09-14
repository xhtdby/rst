#!/usr/bin/env python3
"""
Selective Pruning Tool - Remove articles and conjunctions from word association graph
"""

import sys
from pathlib import Path
import csv

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from rst_trap_finder.core import WordAssociationGraph

class SelectivePruner:
    """Selectively prune specific word categories from the graph"""
    
    def __init__(self, graph_path):
        print(f"🔄 Loading word association graph from {graph_path}")
        self.graph = WordAssociationGraph.from_csv(graph_path)
        
        words = len(self.graph.get_all_words())
        edges = sum(len(neighbors) for neighbors in self.graph.graph.values())
        print(f"📊 Original graph: {words:,} words, {edges:,} edges")
        
        # Define categories to prune
        self.articles = {"the", "a", "an"}
        self.conjunctions = {"and", "but", "or", "nor", "for", "so", "yet", "because", "since", "while", "although", "though", "unless", "until", "before", "after", "when", "where", "if", "whether"}
        
        # Filter to only words that exist in our graph
        self.articles_in_graph = {word for word in self.articles if word in self.graph.graph}
        self.conjunctions_in_graph = {word for word in self.conjunctions if word in self.graph.graph}
        
        self.words_to_prune = self.articles_in_graph | self.conjunctions_in_graph
        
        print(f"🎯 Articles found in graph: {len(self.articles_in_graph)} - {sorted(self.articles_in_graph)}")
        print(f"🎯 Conjunctions found in graph: {len(self.conjunctions_in_graph)} - {sorted(self.conjunctions_in_graph)}")
        print(f"📝 Total words to prune: {len(self.words_to_prune)}")
    
    def analyze_pruning_impact(self):
        """Analyze what would be lost by pruning these words"""
        print(f"\n🔍 PRUNING IMPACT ANALYSIS")
        print("=" * 80)
        
        total_connections_lost = 0
        total_edges_lost = 0
        
        print(f"{'Word':<15} {'Type':<12} {'Outgoing':<10} {'Incoming':<10} {'Total Impact'}")
        print("-" * 65)
        
        for word in sorted(self.words_to_prune):
            # Count outgoing connections
            outgoing = len(self.graph.get_neighbors(word))
            
            # Count incoming connections (how many words point to this word)
            incoming = 0
            for other_word in self.graph.get_all_words():
                if word in self.graph.get_neighbors(other_word):
                    incoming += 1
            
            # Determine type
            word_type = "Article" if word in self.articles_in_graph else "Conjunction"
            
            total_impact = outgoing + incoming
            total_connections_lost += total_impact
            total_edges_lost += outgoing
            
            print(f"{word:<15} {word_type:<12} {outgoing:<10} {incoming:<10} {total_impact}")
        
        print(f"\n📊 PRUNING SUMMARY:")
        print(f"   🔗 Total edges to be removed: {total_edges_lost:,}")
        print(f"   💔 Total connection disruptions: {total_connections_lost:,}")
        
        original_edges = sum(len(neighbors) for neighbors in self.graph.graph.values())
        remaining_edges = original_edges - total_edges_lost
        pruning_percentage = (total_edges_lost / original_edges) * 100
        
        print(f"   📈 Edges after pruning: {remaining_edges:,}")
        print(f"   📉 Reduction: {pruning_percentage:.2f}%")
        
        return total_edges_lost, total_connections_lost
    
    def create_pruned_graph(self, output_path="data/merged/pruned_association_graph.csv"):
        """Create a new graph with articles and conjunctions removed"""
        print(f"\n✂️  CREATING PRUNED GRAPH")
        print("=" * 80)
        
        pruned_edges = []
        original_edge_count = 0
        pruned_edge_count = 0
        
        print(f"🔄 Processing edges...")
        
        for word, neighbors in self.graph.graph.items():
            original_edge_count += len(neighbors)
            
            # Skip if the source word should be pruned
            if word in self.words_to_prune:
                continue
            
            # Filter out target words that should be pruned
            for target, weight in neighbors.items():
                if target not in self.words_to_prune:
                    pruned_edges.append([word, target, weight])
                    pruned_edge_count += 1
        
        # Write pruned graph to CSV
        print(f"💾 Writing pruned graph to {output_path}")
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['src', 'dst', 'weight'])  # Use correct column names
            writer.writerows(pruned_edges)
        
        print(f"✅ Pruned graph created!")
        print(f"   📊 Original edges: {original_edge_count:,}")
        print(f"   📊 Pruned edges: {pruned_edge_count:,}")
        print(f"   📉 Removed: {original_edge_count - pruned_edge_count:,} edges ({((original_edge_count - pruned_edge_count) / original_edge_count * 100):.2f}%)")
        
        return output_path, pruned_edge_count
    
    def analyze_pruned_graph(self, pruned_path):
        """Load and analyze the pruned graph"""
        print(f"\n📊 ANALYZING PRUNED GRAPH")
        print("=" * 80)
        
        print(f"🔄 Loading pruned graph from {pruned_path}")
        pruned_graph = WordAssociationGraph.from_csv(pruned_path)
        
        pruned_words = len(pruned_graph.get_all_words())
        pruned_edges = sum(len(neighbors) for neighbors in pruned_graph.graph.values())
        
        original_words = len(self.graph.get_all_words())
        original_edges = sum(len(neighbors) for neighbors in self.graph.graph.values())
        
        print(f"📈 COMPARISON:")
        print(f"   Words: {original_words:,} → {pruned_words:,} ({original_words - pruned_words:,} removed)")
        print(f"   Edges: {original_edges:,} → {pruned_edges:,} ({original_edges - pruned_edges:,} removed)")
        
        if original_words > 0 and original_edges > 0:
            word_reduction = ((original_words - pruned_words) / original_words) * 100
            edge_reduction = ((original_edges - pruned_edges) / original_edges) * 100
            
            print(f"   📉 Word reduction: {word_reduction:.2f}%")
            print(f"   📉 Edge reduction: {edge_reduction:.2f}%")
        
        # Check remaining function words
        remaining_function_words = []
        function_word_candidates = {"is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", 
                                  "will", "would", "can", "could", "should", "may", "might", "must",
                                  "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
                                  "my", "your", "his", "her", "its", "our", "their",
                                  "in", "on", "at", "by", "with", "from", "to", "of", "about", "into", "through"}
        
        for word in function_word_candidates:
            if word in pruned_graph.graph:
                remaining_function_words.append(word)
        
        print(f"\n🎯 REMAINING FUNCTION WORDS: {len(remaining_function_words)}")
        if remaining_function_words:
            print(f"   {', '.join(sorted(remaining_function_words)[:15])}...")
        
        return pruned_graph

def main():
    # Use the large merged dataset
    dataset_path = "data/merged/merged_association_graph.csv"
    
    pruner = SelectivePruner(dataset_path)
    
    print(f"\n{'='*20} SELECTIVE PRUNING: ARTICLES & CONJUNCTIONS {'='*20}")
    
    # Analyze impact before pruning
    edges_lost, connections_lost = pruner.analyze_pruning_impact()
    
    # Create pruned graph
    pruned_path, pruned_count = pruner.create_pruned_graph()
    
    # Analyze the result
    pruned_graph = pruner.analyze_pruned_graph(pruned_path)
    
    print(f"\n🎯 PRUNING COMPLETE!")
    print(f"   📂 Pruned dataset: {pruned_path}")
    print(f"   🧹 Removed articles: {', '.join(sorted(pruner.articles_in_graph))}")
    print(f"   🧹 Removed conjunctions: {', '.join(sorted(pruner.conjunctions_in_graph))}")

if __name__ == "__main__":
    main()