#!/usr/bin/env python3
"""
RST Trap Finder - Multi-Step Analysis Framework

This module implements advanced multi-step analysis algorithms including:
- k-step probability analysis with dynamic programming optimization
- Information flow algorithms with entropy measures
- Path optimization using A* search and game theory
- Strategic depth evaluation for multi-step planning

Key Features:
- Information Theory: Entropy, information gain, uncertainty reduction
- Path Optimization: A* search, dynamic programming, minimax
- Strategic Planning: Multi-step lookahead, game-theoretic analysis
- Performance: Memoization, efficient algorithms for large graphs
"""

import math
import heapq
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from collections import defaultdict, deque
from dataclasses import dataclass
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


@dataclass
class PathInfo:
    """Information about a specific path through the word association graph."""
    path: List[str]
    probability: float
    entropy: float
    information_gain: float
    steps: int
    rst_endpoint: Optional[str] = None


@dataclass
class MultiStepResult:
    """Comprehensive results from multi-step analysis."""
    word: str
    k_steps: int
    total_probability: float
    path_count: int
    average_entropy: float
    max_information_gain: float
    optimal_paths: List[PathInfo]
    strategic_score: float


class MultiStepAnalyzer:
    """
    Advanced multi-step analysis for RST trap finding.
    
    This class provides sophisticated algorithms for analyzing word association
    graphs beyond simple one-step probabilities, incorporating information theory
    and game-theoretic concepts for strategic planning.
    """
    
    def __init__(self, graph):
        """
        Initialize multi-step analyzer.
        
        Args:
            graph: WordAssociationGraph instance
        """
        self.graph = graph
        self.rst_letters = {'r', 's', 't'}
        
        # Caching for performance
        self._k_step_cache = {}
        self._entropy_cache = {}
        self._path_cache = {}
        
    def clear_cache(self):
        """Clear all memoization caches."""
        self._k_step_cache.clear()
        self._entropy_cache.clear()
        self._path_cache.clear()
    
    def k_step_probability_exact(self, word: str, k: int, 
                               memo: Optional[Dict] = None) -> float:
        """
        Calculate probability of reaching RST word in exactly k steps.
        
        Enhanced version with dynamic programming optimization.
        
        Args:
            word: Starting word
            k: Exact number of steps
            memo: Memoization dictionary (internal use)
            
        Returns:
            Probability of reaching RST word in exactly k steps
        """
        if memo is None:
            memo = {}
            
        cache_key = (word, k)
        if cache_key in memo:
            return memo[cache_key]
        
        # Base cases
        if k == 0:
            result = 1.0 if word and word[0].lower() in self.rst_letters else 0.0
            memo[cache_key] = result
            return result
        
        if k < 0:
            return 0.0
            
        # If already RST word, can't reach RST in k > 0 steps
        if word and word[0].lower() in self.rst_letters:
            memo[cache_key] = 0.0
            return 0.0
        
        neighbors = self.graph.get_neighbors(word)
        if not neighbors:
            memo[cache_key] = 0.0
            return 0.0
        
        total_weight = sum(neighbors.values())
        probability = 0.0
        
        # Filter out RST neighbors for intermediate steps
        safe_neighbors = {w: weight for w, weight in neighbors.items() 
                         if w[0].lower() not in self.rst_letters}
        
        for neighbor, weight in safe_neighbors.items():
            transition_prob = weight / total_weight
            future_prob = self.k_step_probability_exact(neighbor, k - 1, memo)
            probability += transition_prob * future_prob
        
        # Add direct RST probability only for k=1
        if k == 1:
            rst_neighbors = {w: weight for w, weight in neighbors.items() 
                           if w[0].lower() in self.rst_letters}
            rst_weight = sum(rst_neighbors.values())
            probability += rst_weight / total_weight
        
        memo[cache_key] = probability
        return probability
    
    def k_step_probability_cumulative(self, word: str, k: int) -> float:
        """
        Calculate probability of reaching RST word within k steps.
        
        Args:
            word: Starting word
            k: Maximum number of steps
            
        Returns:
            Cumulative probability of reaching RST within k steps
        """
        cache_key = (word, k, 'cumulative')
        if cache_key in self._k_step_cache:
            return self._k_step_cache[cache_key]
        
        total_prob = 0.0
        memo = {}
        
        for step in range(1, k + 1):
            step_prob = self.k_step_probability_exact(word, step, memo)
            total_prob += step_prob
        
        self._k_step_cache[cache_key] = total_prob
        return total_prob
    
    def path_entropy(self, path: List[str]) -> float:
        """
        Calculate information entropy of a path through the graph.
        
        Args:
            path: Sequence of words forming a path
            
        Returns:
            Entropy of the path in bits
        """
        if len(path) < 2:
            return 0.0
        
        cache_key = tuple(path)
        if cache_key in self._entropy_cache:
            return self._entropy_cache[cache_key]
        
        entropy = 0.0
        
        for i in range(len(path) - 1):
            current_word = path[i]
            next_word = path[i + 1]
            
            neighbors = self.graph.get_neighbors(current_word)
            if not neighbors:
                continue
                
            total_weight = sum(neighbors.values())
            if next_word in neighbors:
                prob = neighbors[next_word] / total_weight
                if prob > 0:
                    entropy -= prob * math.log2(prob)
        
        self._entropy_cache[cache_key] = entropy
        return entropy
    
    def information_gain(self, word: str, target_word: str) -> float:
        """
        Calculate information gain from word to target_word.
        
        Args:
            word: Starting word
            target_word: Target word
            
        Returns:
            Information gain in bits
        """
        # Prior entropy (uncertainty before knowing the transition)
        neighbors = self.graph.get_neighbors(word)
        if not neighbors:
            return 0.0
        
        total_weight = sum(neighbors.values())
        prior_entropy = 0.0
        
        for neighbor, weight in neighbors.items():
            prob = weight / total_weight
            if prob > 0:
                prior_entropy -= prob * math.log2(prob)
        
        # Posterior entropy (uncertainty after observing the transition)
        if target_word not in neighbors:
            return 0.0
        
        target_prob = neighbors[target_word] / total_weight
        if target_prob <= 0:
            return 0.0
        
        posterior_entropy = -math.log2(target_prob)
        
        # Information gain = reduction in uncertainty
        return max(0.0, prior_entropy - posterior_entropy)
    
    def find_optimal_paths(self, start_word: str, max_steps: int = 5,
                          max_paths: int = 10) -> List[PathInfo]:
        """
        Find optimal paths from start_word to RST targets.
        
        Uses A* search with information-theoretic heuristics.
        
        Args:
            start_word: Starting word
            max_steps: Maximum path length
            max_paths: Maximum number of paths to return
            
        Returns:
            List of optimal PathInfo objects
        """
        # Priority queue: (negative_score, steps, path, current_word)
        heap = [(0.0, 0, [start_word], start_word)]
        visited_paths = set()
        optimal_paths = []
        
        while heap and len(optimal_paths) < max_paths:
            neg_score, steps, path, current_word = heapq.heappop(heap)
            
            # Skip if we've seen this path state before
            path_state = (current_word, steps)
            if path_state in visited_paths:
                continue
            visited_paths.add(path_state)
            
            # Check if we've reached an RST word
            if current_word[0].lower() in self.rst_letters:
                path_info = PathInfo(
                    path=path,
                    probability=self._calculate_path_probability(path),
                    entropy=self.path_entropy(path),
                    information_gain=self._calculate_total_info_gain(path),
                    steps=steps,
                    rst_endpoint=current_word
                )
                optimal_paths.append(path_info)
                continue
            
            # Stop if maximum steps reached
            if steps >= max_steps:
                continue
            
            # Expand neighbors
            neighbors = self.graph.get_neighbors(current_word)
            if not neighbors:
                continue
            
            total_weight = sum(neighbors.values())
            
            for neighbor, weight in neighbors.items():
                # Skip if neighbor would create a cycle
                if neighbor in path:
                    continue
                
                new_path = path + [neighbor]
                new_steps = steps + 1
                
                # Calculate heuristic score (higher is better)
                transition_prob = weight / total_weight
                path_entropy = self.path_entropy(new_path)
                info_gain = self.information_gain(current_word, neighbor)
                
                # Heuristic: favor high probability, low entropy, high info gain
                heuristic_score = (
                    0.4 * transition_prob +
                    0.3 * info_gain +
                    0.2 * (1.0 / (1.0 + path_entropy)) +  # Prefer lower entropy
                    0.1 * self.graph.one_step_rst_probability(neighbor)
                )
                
                # Push to heap (negate score for min-heap)
                heapq.heappush(heap, (-heuristic_score, new_steps, new_path, neighbor))
        
        # Sort by strategic value
        optimal_paths.sort(key=lambda p: self._strategic_path_score(p), reverse=True)
        return optimal_paths
    
    def _calculate_path_probability(self, path: List[str]) -> float:
        """Calculate the probability of following a specific path."""
        if len(path) < 2:
            return 1.0
        
        probability = 1.0
        
        for i in range(len(path) - 1):
            current_word = path[i]
            next_word = path[i + 1]
            
            neighbors = self.graph.get_neighbors(current_word)
            if not neighbors or next_word not in neighbors:
                return 0.0
            
            total_weight = sum(neighbors.values())
            transition_prob = neighbors[next_word] / total_weight
            probability *= transition_prob
        
        return probability
    
    def _calculate_total_info_gain(self, path: List[str]) -> float:
        """Calculate total information gain along a path."""
        if len(path) < 2:
            return 0.0
        
        total_gain = 0.0
        
        for i in range(len(path) - 1):
            current_word = path[i]
            next_word = path[i + 1]
            total_gain += self.information_gain(current_word, next_word)
        
        return total_gain
    
    def _strategic_path_score(self, path_info: PathInfo) -> float:
        """Calculate strategic score for a path."""
        # Balanced scoring considering multiple factors
        prob_score = path_info.probability
        efficiency_score = 1.0 / (1.0 + path_info.steps)  # Prefer shorter paths
        info_score = path_info.information_gain
        certainty_score = 1.0 / (1.0 + path_info.entropy)  # Prefer lower entropy
        
        return (0.3 * prob_score + 
                0.25 * efficiency_score + 
                0.25 * info_score + 
                0.2 * certainty_score)
    
    def analyze_multi_step(self, word: str, max_k: int = 5) -> MultiStepResult:
        """
        Comprehensive multi-step analysis for a word.
        
        Args:
            word: Word to analyze
            max_k: Maximum number of steps to analyze
            
        Returns:
            MultiStepResult with comprehensive analysis
        """
        # Calculate k-step probabilities
        k_step_probs = []
        for k in range(1, max_k + 1):
            prob = self.k_step_probability_cumulative(word, k)
            k_step_probs.append(prob)
        
        # Find optimal paths
        optimal_paths = self.find_optimal_paths(word, max_steps=max_k)
        
        # Calculate aggregate metrics
        total_probability = k_step_probs[-1] if k_step_probs else 0.0
        path_count = len(optimal_paths)
        
        if optimal_paths:
            average_entropy = sum(p.entropy for p in optimal_paths) / len(optimal_paths)
            max_information_gain = max(p.information_gain for p in optimal_paths)
        else:
            average_entropy = 0.0
            max_information_gain = 0.0
        
        # Calculate strategic score
        strategic_score = self._calculate_strategic_score(word, optimal_paths, k_step_probs)
        
        return MultiStepResult(
            word=word,
            k_steps=max_k,
            total_probability=total_probability,
            path_count=path_count,
            average_entropy=average_entropy,
            max_information_gain=max_information_gain,
            optimal_paths=optimal_paths[:5],  # Top 5 paths
            strategic_score=strategic_score
        )
    
    def _calculate_strategic_score(self, word: str, paths: List[PathInfo], 
                                 k_probs: List[float]) -> float:
        """Calculate overall strategic score incorporating multi-step analysis."""
        if not paths or not k_probs:
            return 0.0
        
        # Multi-step probability component
        prob_component = sum(k_probs) / len(k_probs)
        
        # Path diversity component
        diversity_component = min(1.0, len(paths) / 10.0)
        
        # Information efficiency component
        if paths:
            avg_efficiency = sum(p.information_gain / max(1, p.steps) for p in paths) / len(paths)
            efficiency_component = min(1.0, avg_efficiency)
        else:
            efficiency_component = 0.0
        
        # Combine components
        return (0.5 * prob_component + 
                0.3 * efficiency_component + 
                0.2 * diversity_component)
    
    def compare_strategic_depth(self, words: List[str], max_k: int = 5) -> Dict[str, MultiStepResult]:
        """
        Compare strategic depth of multiple words.
        
        Args:
            words: List of words to compare
            max_k: Maximum analysis depth
            
        Returns:
            Dictionary mapping words to their MultiStepResult
        """
        results = {}
        
        try:
            word_iterator = tqdm(words, desc="Multi-step analysis", unit="words")
        except NameError:
            word_iterator = words
            print(f"Analyzing strategic depth for {len(words)} words...")
        
        for word in word_iterator:
            if self.graph.has_word(word):
                results[word] = self.analyze_multi_step(word, max_k)
        
        return results