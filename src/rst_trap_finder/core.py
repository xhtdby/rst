"""
RST Trap Finder - Core functionality for analyzing word association graphs.

This module provides a complete toolkit for analyzing word association graphs
to identify "trap words" - words that tend to lead opponents toward words
starting with the letters R, S, or T.

Key concepts:
- Word association graph: directed graph where edges represent word associations
- Trap letters: R, S, T (configurable)
- Trap words: words that effectively funnel opponents toward trap letters
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, FrozenSet, List, Mapping, Optional, Tuple, Union

# Type aliases for clarity
Graph = Dict[str, Dict[str, float]]
ScoreDict = Dict[str, float]

# Default trap letters (R, S, T)
TRAP_LETTERS: FrozenSet[str] = frozenset({"r", "s", "t"})


class WordAssociationGraph:
    """
    A word association graph for analyzing trap words.
    
    This class encapsulates all functionality for loading, analyzing, and
    scoring word association graphs to find effective trap words.
    """
    
    def __init__(self, graph: Optional[Graph] = None, trap_letters: Optional[FrozenSet[str]] = None):
        """
        Initialize the word association graph.
        
        Args:
            graph: Dictionary representing the graph structure
            trap_letters: Set of letters to consider as traps (default: R, S, T)
        """
        self.graph: Graph = graph or {}
        self.trap_letters = trap_letters or TRAP_LETTERS
        self._out_sums: Optional[Dict[str, float]] = None
    
    @classmethod
    def from_csv(cls, path: Union[str, Path], trap_letters: Optional[FrozenSet[str]] = None) -> 'WordAssociationGraph':
        """
        Load a word association graph from a CSV file.
        
        Expected CSV format:
        - src: source word
        - dst: destination word  
        - weight: association strength (positive number)
        
        Args:
            path: Path to CSV file
            trap_letters: Set of trap letters (default: R, S, T)
            
        Returns:
            WordAssociationGraph instance
        """
        graph: Graph = {}
        path = Path(path)
        
        with path.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Skip comment lines
                if row.get('src', '').startswith('#'):
                    continue
                    
                src = (row.get('src') or '').strip().lower()
                dst = (row.get('dst') or '').strip().lower() 
                
                if not src or not dst:
                    continue
                    
                try:
                    weight = float(row.get('weight', '0'))
                except ValueError:
                    continue
                    
                if weight <= 0:
                    continue
                    
                if src not in graph:
                    graph[src] = {}
                    
                # Accumulate weights for duplicate edges
                graph[src][dst] = graph[src].get(dst, 0.0) + weight
        
        return cls(graph, trap_letters)
    
    @property
    def out_sums(self) -> Dict[str, float]:
        """Get cached outgoing weight sums for each node."""
        if self._out_sums is None:
            self._out_sums = {
                node: sum(weights.values()) 
                for node, weights in self.graph.items()
            }
        return self._out_sums
    
    def get_neighbors(self, word: str) -> Dict[str, float]:
        """Get all neighbors of a word with their weights."""
        return self.graph.get(word, {})
    
    def has_word(self, word: str) -> bool:
        """Check if a word exists in the graph."""
        return word in self.graph
    
    def get_all_words(self) -> set[str]:
        """Get all words (nodes) in the graph."""
        all_words = set(self.graph.keys())
        for neighbors in self.graph.values():
            all_words.update(neighbors.keys())
        return all_words
    
    def one_step_rst_probability(self, word: str) -> float:
        """
        Calculate probability that a one-step reply starts with a trap letter.
        
        Formula: S1(u) = sum_{v: v[0] in trap_letters} w(u,v) / sum_{v} w(u,v)
        
        Args:
            word: Word to analyze
            
        Returns:
            Probability (0.0 to 1.0)
        """
        neighbors = self.get_neighbors(word)
        if not neighbors:
            return 0.0
            
        total_weight = sum(neighbors.values())
        trap_weight = sum(
            weight for neighbor, weight in neighbors.items()
            if neighbor and neighbor[0] in self.trap_letters
        )
        
        return trap_weight / total_weight
    
    def escape_hardness(self, word: str, min_weight_fraction: float = 0.05) -> float:
        """
        Calculate fraction of strong edges that lead to trap words.
        
        Strong edges are those with weight >= min_weight_fraction * max_weight.
        
        Args:
            word: Word to analyze
            min_weight_fraction: Minimum weight threshold as fraction of max weight
            
        Returns:
            Fraction of strong edges leading to traps (0.0 to 1.0)
        """
        neighbors = self.get_neighbors(word)
        if not neighbors:
            return 0.0
            
        max_weight = max(neighbors.values())
        threshold = min_weight_fraction * max_weight
        
        strong_edges = [
            neighbor for neighbor, weight in neighbors.items()
            if weight >= threshold
        ]
        
        if not strong_edges:
            return 0.0
            
        trap_edges = [
            neighbor for neighbor in strong_edges
            if neighbor and neighbor[0] in self.trap_letters
        ]
        
        return len(trap_edges) / len(strong_edges)
    
    def biased_pagerank(self, trap_bias: float = 1.5, damping: float = 0.85, 
                       max_iterations: int = 100, tolerance: float = 1e-10) -> ScoreDict:
        """
        Compute PageRank with bias toward trap letters.
        
        Args:
            trap_bias: Multiplier for edges leading to trap words
            damping: PageRank damping factor
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Dictionary mapping words to PageRank scores
        """
        all_words = list(self.get_all_words())
        n_words = len(all_words)
        
        if n_words == 0:
            return {}
        
        # Initialize PageRank values uniformly
        pagerank = {word: 1.0 / n_words for word in all_words}
        
        # Identify dangling nodes (no outgoing edges)
        dangling_nodes = [word for word in all_words if word not in self.graph]
        
        for iteration in range(max_iterations):
            prev_pagerank = pagerank.copy()
            
            # Handle dangling nodes
            dangling_sum = sum(prev_pagerank[word] for word in dangling_nodes)
            base_value = (1 - damping) / n_words + damping * dangling_sum / n_words
            
            # Reset all values to base
            for word in all_words:
                pagerank[word] = base_value
            
            # Add contributions from non-dangling nodes
            for source_word in all_words:
                if source_word not in self.graph:
                    continue
                    
                neighbors = self.graph[source_word]
                if not neighbors:
                    continue
                
                # Calculate biased transition probabilities
                biased_weights = {}
                for neighbor, weight in neighbors.items():
                    bias = trap_bias if neighbor and neighbor[0] in self.trap_letters else 1.0
                    biased_weights[neighbor] = weight * bias
                
                total_biased_weight = sum(biased_weights.values())
                
                # Distribute PageRank according to biased probabilities
                source_contribution = damping * prev_pagerank[source_word]
                for neighbor, biased_weight in biased_weights.items():
                    probability = biased_weight / total_biased_weight
                    pagerank[neighbor] += source_contribution * probability
            
            # Check for convergence
            total_diff = sum(abs(pagerank[word] - prev_pagerank[word]) for word in all_words)
            if total_diff < tolerance:
                break
        
        # Normalize to ensure sum = 1
        total = sum(pagerank.values())
        if total > 0:
            pagerank = {word: score / total for word, score in pagerank.items()}
        
        return pagerank
    
    def k_step_rst_probability(self, word: str, k: int = 2) -> float:
        """
        Calculate probability of reaching a trap word within k steps.
        
        Args:
            word: Starting word
            k: Number of steps to look ahead
            
        Returns:
            Probability of reaching trap within k steps
        """
        if k == 0:
            return 1.0 if word and word[0] in self.trap_letters else 0.0
        
        # If current word is already a trap, we've reached it
        if word and word[0] in self.trap_letters:
            return 1.0
        
        neighbors = self.get_neighbors(word)
        if not neighbors:
            return 0.0
        
        total_weight = sum(neighbors.values())
        probability = 0.0
        
        for neighbor, weight in neighbors.items():
            transition_prob = weight / total_weight
            future_prob = self.k_step_rst_probability(neighbor, k - 1)
            probability += transition_prob * future_prob
            
        return probability
    
    def composite_score(self, word: str, pagerank_scores: Optional[ScoreDict] = None,
                       weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)) -> float:
        """
        Calculate a composite trap score combining multiple metrics.
        
        Args:
            word: Word to score
            pagerank_scores: Pre-computed PageRank scores (computed if None)
            weights: Weights for (one_step, hardness, pagerank) components
            
        Returns:
            Composite score (higher = better trap word)
        """
        if pagerank_scores is None:
            pagerank_scores = self.biased_pagerank()
        
        one_step = self.one_step_rst_probability(word)
        hardness = self.escape_hardness(word)
        pagerank = pagerank_scores.get(word, 0.0)
        
        w1, w2, w3 = weights
        return w1 * one_step + w2 * hardness + w3 * pagerank
    
    def rank_words(self, top_k: Optional[int] = None, 
                   score_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)) -> List[Tuple[str, float]]:
        """
        Rank all words by their trap effectiveness.
        
        Args:
            top_k: Number of top words to return (None for all)
            score_weights: Weights for composite scoring
            
        Returns:
            List of (word, score) tuples sorted by score (highest first)
        """
        pagerank_scores = self.biased_pagerank()
        
        word_scores = []
        for word in self.graph:
            score = self.composite_score(word, pagerank_scores, score_weights)
            word_scores.append((word, score))
        
        word_scores.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            word_scores = word_scores[:top_k]
            
        return word_scores
    
    def recommend_next_word(self, current_word: str, top_k: int = 10) -> List[Dict[str, Union[str, float]]]:
        """
        Recommend next words from the current word based on trap effectiveness.
        
        Args:
            current_word: Current word in the game
            top_k: Number of recommendations to return
            
        Returns:
            List of recommendation dictionaries with word, score, and metrics
        """
        neighbors = self.get_neighbors(current_word)
        if not neighbors:
            return []
        
        pagerank_scores = self.biased_pagerank()
        recommendations = []
        
        for neighbor, weight in neighbors.items():
            one_step = self.one_step_rst_probability(neighbor)
            hardness = self.escape_hardness(neighbor)
            pagerank = pagerank_scores.get(neighbor, 0.0)
            composite = self.composite_score(neighbor, pagerank_scores)
            
            recommendations.append({
                'word': neighbor,
                'score': composite,
                'one_step_prob': one_step,
                'hardness': hardness,
                'pagerank': pagerank,
                'edge_weight': weight
            })
        
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_k]
    
    def get_word_analysis(self, word: str) -> Dict[str, Union[str, float, List[Tuple[str, float]]]]:
        """
        Get comprehensive analysis of a specific word.
        
        Args:
            word: Word to analyze
            
        Returns:
            Dictionary with detailed metrics and neighbor analysis
        """
        if not self.has_word(word):
            raise ValueError(f"Word '{word}' not found in graph")
        
        pagerank_scores = self.biased_pagerank()
        neighbors = self.get_neighbors(word)
        
        # Analyze neighbors
        neighbor_analysis = []
        for neighbor, weight in sorted(neighbors.items(), key=lambda x: x[1], reverse=True):
            is_trap = neighbor and neighbor[0] in self.trap_letters
            neighbor_analysis.append((neighbor, weight, is_trap))
        
        return {
            'word': word,
            'one_step_probability': self.one_step_rst_probability(word),
            'escape_hardness': self.escape_hardness(word),
            'pagerank_score': pagerank_scores.get(word, 0.0),
            'composite_score': self.composite_score(word, pagerank_scores),
            'neighbor_count': len(neighbors),
            'total_outgoing_weight': self.out_sums.get(word, 0.0),
            'neighbors': neighbor_analysis[:10]  # Top 10 neighbors
        }
    
    def export_scores(self, output_path: Union[str, Path], 
                     score_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)) -> None:
        """
        Export word scores to a CSV file.
        
        Args:
            output_path: Path for output CSV file
            score_weights: Weights for composite scoring
        """
        pagerank_scores = self.biased_pagerank()
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'word', 'composite_score', 'one_step_prob', 
                'escape_hardness', 'pagerank_score', 'neighbor_count'
            ])
            
            for word in self.graph:
                one_step = self.one_step_rst_probability(word)
                hardness = self.escape_hardness(word)
                pagerank = pagerank_scores.get(word, 0.0)
                composite = self.composite_score(word, pagerank_scores, score_weights)
                neighbor_count = len(self.get_neighbors(word))
                
                writer.writerow([
                    word, f'{composite:.6f}', f'{one_step:.6f}',
                    f'{hardness:.6f}', f'{pagerank:.6f}', neighbor_count
                ])
    
    def print_summary(self) -> None:
        """Print a summary of the graph statistics."""
        all_words = self.get_all_words()
        total_edges = sum(len(neighbors) for neighbors in self.graph.values())
        total_weight = sum(sum(neighbors.values()) for neighbors in self.graph.values())
        
        trap_words = [word for word in all_words if word and word[0] in self.trap_letters]
        
        print(f"Graph Summary:")
        print(f"  Total words: {len(all_words)}")
        print(f"  Words with outgoing edges: {len(self.graph)}")
        print(f"  Total edges: {total_edges}")
        print(f"  Total weight: {total_weight:.2f}")
        print(f"  Trap words ({', '.join(sorted(self.trap_letters))}): {len(trap_words)}")
        print(f"  Trap word percentage: {100 * len(trap_words) / len(all_words):.1f}%")


def load_graph(csv_path: Union[str, Path], trap_letters: Optional[FrozenSet[str]] = None) -> WordAssociationGraph:
    """
    Convenience function to load a word association graph from CSV.
    
    Args:
        csv_path: Path to CSV file
        trap_letters: Set of trap letters (default: R, S, T)
        
    Returns:
        WordAssociationGraph instance
    """
    return WordAssociationGraph.from_csv(csv_path, trap_letters)


# Export the main classes and functions
__all__ = [
    'WordAssociationGraph', 
    'load_graph', 
    'TRAP_LETTERS',
    'Graph',
    'ScoreDict'
]