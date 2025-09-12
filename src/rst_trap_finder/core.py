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

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback for when tqdm is not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

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
            
            # Get total number of lines for progress bar
            f.seek(0)
            total_lines = sum(1 for _ in f) - 1  # Subtract header
            f.seek(0)
            next(reader)  # Skip header
            
            try:
                pbar = tqdm(reader, total=total_lines, desc=f"Loading {path.name}", unit="edges")
            except NameError:
                pbar = reader
                print(f"Loading word associations from {path.name}...")
            
            for row in pbar:
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
            
            # Close progress bar if it exists
            if hasattr(pbar, 'close'):
                pbar.close()
        
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
        
        try:
            iterator = tqdm(range(max_iterations), desc="Computing PageRank", unit="iter")
        except NameError:
            iterator = range(max_iterations)
            print("Computing biased PageRank...")
        
        for iteration in iterator:
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
                if hasattr(iterator, 'set_description'):
                    iterator.set_description(f"PageRank converged (iter {iteration+1})")
                break
                
        # Close progress bar if it exists
        if hasattr(iterator, 'close'):
            iterator.close()
        
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
        words = list(self.graph.keys())
        
        try:
            word_iterator = tqdm(words, desc="Computing word scores", unit="words")
        except NameError:
            word_iterator = words
            print(f"Computing scores for {len(words)} words...")
        
        for word in word_iterator:
            score = self.composite_score(word, pagerank_scores, score_weights)
            word_scores.append((word, score))
        
        # Close progress bar if it exists
        if hasattr(word_iterator, 'close'):
            word_iterator.close()
        
        print("Sorting words by score...")
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
    
    def analyze_trap_pathways(self, word: str, max_steps: int = 3, min_weight: float = 0.01) -> Dict[str, Union[str, List, Dict]]:
        """
        Analyze the pathways from a word to RST trap words.
        
        Shows what specific R/S/T words this trap word actually leads to,
        and the paths taken to get there.
        
        Args:
            word: Word to analyze pathways from
            max_steps: Maximum path length to explore (default: 3)
            min_weight: Minimum edge weight to consider (default: 0.01)
            
        Returns:
            Dictionary containing pathway analysis with direct RST targets,
            multi-step paths, and statistics
        """
        if not self.has_word(word):
            raise ValueError(f"Word '{word}' not found in graph")
        
        # Track all paths to RST words
        rst_targets = {'R': [], 'S': [], 'T': []}
        all_paths = []
        
        def explore_paths(current_word, path, remaining_steps):
            """Recursively explore paths to RST words."""
            if remaining_steps <= 0:
                return
                
            neighbors = self.get_neighbors(current_word)
            if not neighbors:
                return
            
            for neighbor, weight in neighbors.items():
                if weight < min_weight:
                    continue
                
                # Skip if neighbor starts with RST (can't be used in game)
                if neighbor and neighbor[0] in self.trap_letters:
                    # Only count as endpoint, don't continue path through RST words
                    new_path = path + [(neighbor, weight)]
                    letter = neighbor[0].upper()
                    rst_targets[letter].append({
                        'target_word': neighbor,
                        'path': new_path,
                        'total_weight': sum(w for _, w in new_path),
                        'path_length': len(new_path)
                    })
                    all_paths.append(new_path)
                    continue  # Don't explore further through RST words
                    
                new_path = path + [(neighbor, weight)]
                
                # Continue exploring only through non-RST words
                if neighbor not in [w for w, _ in path]:
                    explore_paths(neighbor, new_path, remaining_steps - 1)
        
        # Start exploration
        explore_paths(word, [], max_steps)
        
        # Organize results
        direct_rst = self.get_neighbors(word)
        direct_targets = {}
        for neighbor, weight in direct_rst.items():
            if neighbor and neighbor[0] in self.trap_letters:
                letter = neighbor[0].upper()
                if letter not in direct_targets:
                    direct_targets[letter] = []
                direct_targets[letter].append((neighbor, weight))
        
        # Sort targets by weight
        for letter in direct_targets:
            direct_targets[letter].sort(key=lambda x: x[1], reverse=True)
        
        # Get top pathways for each letter
        top_pathways = {}
        for letter in ['R', 'S', 'T']:
            if rst_targets[letter]:
                # Sort by total weight and take top paths
                sorted_paths = sorted(rst_targets[letter], 
                                    key=lambda x: x['total_weight'], reverse=True)
                top_pathways[letter] = sorted_paths[:5]  # Top 5 paths per letter
            else:
                top_pathways[letter] = []
        
        # Calculate statistics
        total_rst_neighbors = sum(len(targets) for targets in direct_targets.values())
        total_neighbors = len(direct_rst)
        direct_rst_probability = total_rst_neighbors / total_neighbors if total_neighbors > 0 else 0
        
        return {
            'source_word': word,
            'direct_rst_targets': direct_targets,
            'top_pathways': top_pathways,
            'statistics': {
                'total_neighbors': total_neighbors,
                'direct_rst_neighbors': total_rst_neighbors,
                'direct_rst_probability': direct_rst_probability,
                'total_paths_found': len(all_paths),
                'max_steps_explored': max_steps
            }
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
            
            words = list(self.graph.keys())
            
            try:
                word_iterator = tqdm(words, desc="Exporting scores", unit="words")
            except NameError:
                word_iterator = words
                print(f"Exporting scores for {len(words)} words...")
            
            for word in word_iterator:
                one_step = self.one_step_rst_probability(word)
                hardness = self.escape_hardness(word)
                pagerank = pagerank_scores.get(word, 0.0)
                composite = self.composite_score(word, pagerank_scores, score_weights)
                neighbor_count = len(self.get_neighbors(word))
                
                writer.writerow([
                    word, f'{composite:.6f}', f'{one_step:.6f}',
                    f'{hardness:.6f}', f'{pagerank:.6f}', neighbor_count
                ])
                
            # Close progress bar if it exists
            if hasattr(word_iterator, 'close'):
                word_iterator.close()
    
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