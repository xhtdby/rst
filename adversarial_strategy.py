#!/usr/bin/env python3
"""
Enhanced Adversarial RST Avoidance Strategy

This module implements sophisticated adversarial strategies for avoiding RST traps
while simultaneously trying to lead opponents into them. Uses minimax-style 
evaluation and multi-step lookahead for competitive gameplay.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import math

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from rst_trap_finder.core import WordAssociationGraph
from rst_trap_finder.multistep import MultiStepAnalyzer


@dataclass
class MoveEvaluation:
    """Evaluation of a potential move in adversarial play"""
    word: str
    self_safety: float  # How safe this move is for current player
    opponent_danger: float  # How dangerous this puts opponent
    strategic_value: float  # Overall strategic assessment
    lookahead_depth: int  # How many steps ahead were evaluated
    escape_routes: int  # Number of non-RST continuations available
    trap_paths: List[str]  # Paths that lead opponent to traps


class AdversarialRSTStrategy:
    """
    Advanced adversarial strategy for RST word games.
    
    This strategy implements:
    1. Multi-step safety analysis (avoid leading self into traps)
    2. Opponent trap analysis (create traps for opponent)
    3. Minimax-style evaluation with lookahead
    4. Dynamic difficulty adjustment
    5. Escape route preservation
    """
    
    def __init__(self, graph: WordAssociationGraph, difficulty: str = "medium"):
        """
        Initialize adversarial strategy.
        
        Args:
            graph: Word association graph
            difficulty: "easy", "medium", "hard", or "expert"
        """
        self.graph = graph
        self.multistep = MultiStepAnalyzer(graph)
        self.difficulty = difficulty
        
        # Configure strategy parameters based on difficulty
        self.config = self._get_difficulty_config(difficulty)
        
        # Game state tracking
        self.move_history: List[str] = []
        self.opponent_patterns: Dict[str, int] = {}
        self.known_safe_words: Set[str] = set()
        self.known_trap_words: Set[str] = set()
    
    def _get_difficulty_config(self, difficulty: str) -> Dict:
        """Get configuration parameters for difficulty level"""
        configs = {
            "easy": {
                "lookahead_depth": 2,
                "safety_weight": 0.8,
                "aggression_weight": 0.2,
                "min_escape_routes": 2,
                "analysis_time_limit": 0.1,
                "consider_opponent_patterns": False
            },
            "medium": {
                "lookahead_depth": 3,
                "safety_weight": 0.65,
                "aggression_weight": 0.35,
                "min_escape_routes": 1,
                "analysis_time_limit": 0.5,
                "consider_opponent_patterns": True
            },
            "hard": {
                "lookahead_depth": 4,
                "safety_weight": 0.5,
                "aggression_weight": 0.5,
                "min_escape_routes": 1,
                "analysis_time_limit": 1.0,
                "consider_opponent_patterns": True
            },
            "expert": {
                "lookahead_depth": 5,
                "safety_weight": 0.4,
                "aggression_weight": 0.6,
                "min_escape_routes": 0,
                "analysis_time_limit": 2.0,
                "consider_opponent_patterns": True
            }
        }
        return configs.get(difficulty, configs["medium"])
    
    def choose_move(self, current_word: str, available_words: List[str]) -> str:
        """
        Choose the best adversarial move given current game state.
        
        Args:
            current_word: The current word in the game
            available_words: List of valid next words
            
        Returns:
            Best word choice for adversarial play
        """
        if not available_words:
            return ""
        
        # Update game state
        self.move_history.append(current_word)
        
        # Evaluate all possible moves
        evaluations = []
        for word in available_words:
            evaluation = self._evaluate_move(current_word, word, available_words)
            evaluations.append(evaluation)
        
        # Sort by strategic value
        evaluations.sort(key=lambda e: e.strategic_value, reverse=True)
        
        # Apply final selection strategy
        best_move = self._select_final_move(evaluations)
        
        # Update learned patterns
        self._update_learned_patterns(current_word, best_move.word)
        
        return best_move.word
    
    def _evaluate_move(self, current_word: str, candidate_word: str, 
                      available_words: List[str]) -> MoveEvaluation:
        """Comprehensively evaluate a potential move"""
        
        # Calculate self-safety (how safe is this move for us)
        self_safety = self._calculate_self_safety(candidate_word)
        
        # Calculate opponent danger (how dangerous does this make it for opponent)
        opponent_danger = self._calculate_opponent_danger(candidate_word)
        
        # Count escape routes
        escape_routes = self._count_escape_routes(candidate_word)
        
        # Find trap paths this creates for opponent
        trap_paths = self._find_opponent_trap_paths(candidate_word)
        
        # Calculate overall strategic value
        strategic_value = self._calculate_strategic_value(
            self_safety, opponent_danger, escape_routes, len(trap_paths)
        )
        
        return MoveEvaluation(
            word=candidate_word,
            self_safety=self_safety,
            opponent_danger=opponent_danger,
            strategic_value=strategic_value,
            lookahead_depth=self.config["lookahead_depth"],
            escape_routes=escape_routes,
            trap_paths=trap_paths
        )
    
    def _calculate_self_safety(self, word: str) -> float:
        """Calculate how safe a word choice is for the current player"""
        
        # Check if word is immediately RST (very dangerous)
        if word[0].lower() in self.graph.trap_letters:
            return 0.0
        
        # Calculate multi-step trap probability
        lookahead = self.config["lookahead_depth"]
        trap_probability = self.multistep.k_step_probability_cumulative(word, lookahead)
        
        # Safety is inverse of trap probability
        base_safety = 1.0 - trap_probability
        
        # Bonus for known safe words
        if word in self.known_safe_words:
            base_safety = min(1.0, base_safety + 0.1)
        
        # Penalty for known trap words
        if word in self.known_trap_words:
            base_safety = max(0.0, base_safety - 0.2)
        
        # Consider escape routes
        neighbors = self.graph.get_neighbors(word)
        if neighbors:
            safe_neighbors = sum(1 for w in neighbors.keys() 
                               if w[0].lower() not in self.graph.trap_letters)
            safety_ratio = safe_neighbors / len(neighbors)
            base_safety = 0.7 * base_safety + 0.3 * safety_ratio
        
        return base_safety
    
    def _calculate_opponent_danger(self, word: str) -> float:
        """Calculate how dangerous this word makes it for the opponent"""
        
        # Get opponent's possible responses
        neighbors = self.graph.get_neighbors(word)
        if not neighbors:
            return 0.0
        
        # Calculate average trap probability of opponent's options
        total_danger = 0.0
        total_weight = sum(neighbors.values())
        
        for next_word, weight in neighbors.items():
            # Skip immediate RST words (opponent can't play them)
            if next_word[0].lower() in self.graph.trap_letters:
                continue
            
            # Calculate trap probability for opponent from this word
            lookahead = max(1, self.config["lookahead_depth"] - 1)
            trap_prob = self.multistep.k_step_probability_cumulative(next_word, lookahead)
            
            # Weight by transition probability
            transition_prob = weight / total_weight
            total_danger += transition_prob * trap_prob
        
        return total_danger
    
    def _count_escape_routes(self, word: str) -> int:
        """Count number of safe continuation paths from a word"""
        neighbors = self.graph.get_neighbors(word)
        if not neighbors:
            return 0
        
        safe_count = 0
        for neighbor in neighbors.keys():
            if neighbor[0].lower() not in self.graph.trap_letters:
                # Check if this neighbor has its own safe continuations
                sub_neighbors = self.graph.get_neighbors(neighbor)
                if sub_neighbors:
                    sub_safe = sum(1 for w in sub_neighbors.keys() 
                                 if w[0].lower() not in self.graph.trap_letters)
                    if sub_safe > 0:
                        safe_count += 1
        
        return safe_count
    
    def _find_opponent_trap_paths(self, word: str) -> List[str]:
        """Find paths that would lead opponent into traps"""
        trap_paths = []
        
        # Get opponent's immediate options
        neighbors = self.graph.get_neighbors(word)
        if not neighbors:
            return trap_paths
        
        for next_word in neighbors.keys():
            if next_word[0].lower() in self.graph.trap_letters:
                continue
                
            # Find paths from this word to traps
            paths = self.multistep.find_optimal_paths(
                next_word, 
                max_steps=self.config["lookahead_depth"],
                max_paths=3
            )
            
            for path in paths:
                if path.rst_endpoint and path.probability > 0.3:
                    path_str = " → ".join(path.path)
                    trap_paths.append(path_str)
        
        return trap_paths
    
    def _calculate_strategic_value(self, self_safety: float, opponent_danger: float,
                                 escape_routes: int, trap_path_count: int) -> float:
        """Calculate overall strategic value of a move"""
        
        # Base score from safety vs aggression
        safety_weight = self.config["safety_weight"]
        aggression_weight = self.config["aggression_weight"]
        
        base_score = (safety_weight * self_safety + 
                     aggression_weight * opponent_danger)
        
        # Escape route bonus
        min_routes = self.config["min_escape_routes"]
        if escape_routes >= min_routes:
            route_bonus = 0.1 * min(escape_routes - min_routes, 3)
        else:
            route_bonus = -0.2 * (min_routes - escape_routes)
        
        # Trap creation bonus
        trap_bonus = 0.1 * min(trap_path_count, 3)
        
        # Opponent pattern exploitation (for higher difficulties)
        pattern_bonus = 0.0
        if self.config["consider_opponent_patterns"]:
            pattern_bonus = self._calculate_pattern_bonus()
        
        total_score = base_score + route_bonus + trap_bonus + pattern_bonus
        return max(0.0, min(1.0, total_score))
    
    def _calculate_pattern_bonus(self) -> float:
        """Calculate bonus for exploiting learned opponent patterns"""
        if len(self.opponent_patterns) < 3:
            return 0.0
        
        # Simple pattern recognition - could be much more sophisticated
        most_common = max(self.opponent_patterns.values())
        if most_common >= 2:
            return 0.05  # Small bonus for pattern recognition
        
        return 0.0
    
    def _select_final_move(self, evaluations: List[MoveEvaluation]) -> MoveEvaluation:
        """Apply final selection logic to choose from top evaluations"""
        
        if not evaluations:
            return MoveEvaluation("", 0, 0, 0, 0, 0, [])
        
        # For expert difficulty, sometimes take calculated risks
        if self.difficulty == "expert" and len(evaluations) > 1:
            top_two = evaluations[:2]
            if (top_two[1].opponent_danger > top_two[0].opponent_danger and 
                top_two[1].self_safety > 0.6):
                return top_two[1]  # Take the aggressive option
        
        # For easy difficulty, be more conservative
        if self.difficulty == "easy":
            safe_moves = [e for e in evaluations if e.self_safety > 0.7]
            if safe_moves:
                return safe_moves[0]
        
        return evaluations[0]
    
    def _update_learned_patterns(self, from_word: str, to_word: str):
        """Update learned patterns about opponent play"""
        pattern = f"{from_word}->{to_word}"
        self.opponent_patterns[pattern] = self.opponent_patterns.get(pattern, 0) + 1
        
        # Update word safety knowledge based on outcomes
        # This would be enhanced with actual game outcome feedback
        
    def get_strategy_explanation(self, word: str, evaluation: MoveEvaluation) -> str:
        """Provide human-readable explanation of strategy choice"""
        explanation = f"🎯 **Adversarial Strategy Analysis for '{word}'**\n\n"
        
        explanation += f"**Safety Assessment:** {evaluation.self_safety:.2f}/1.00\n"
        explanation += f"**Opponent Pressure:** {evaluation.opponent_danger:.2f}/1.00\n"
        explanation += f"**Escape Routes:** {evaluation.escape_routes}\n"
        explanation += f"**Strategic Value:** {evaluation.strategic_value:.2f}/1.00\n\n"
        
        if evaluation.trap_paths:
            explanation += "**Opponent Trap Paths Created:**\n"
            for path in evaluation.trap_paths[:3]:
                explanation += f"  • {path}\n"
        
        # Strategy reasoning
        if evaluation.self_safety > 0.8:
            explanation += "\n✅ **Very safe choice** - low trap risk for you"
        elif evaluation.self_safety < 0.3:
            explanation += "\n⚠️ **Risky choice** - high trap probability"
        
        if evaluation.opponent_danger > 0.6:
            explanation += "\n🎯 **Aggressive positioning** - puts opponent in danger"
        
        return explanation
    
    def reset_game_state(self):
        """Reset strategy state for new game"""
        self.move_history.clear()
        # Keep learned patterns across games for adaptation


def create_adversarial_computer_player(graph: WordAssociationGraph, 
                                     difficulty: str = "medium"):
    """
    Create a computer player with adversarial RST avoidance strategy.
    
    Args:
        graph: Word association graph
        difficulty: "easy", "medium", "hard", or "expert"
        
    Returns:
        Computer player function compatible with game CLI
    """
    strategy = AdversarialRSTStrategy(graph, difficulty)
    
    def adversarial_player(current_word: str, game_state: dict) -> str:
        """Adversarial computer player function"""
        
        # Get available words (would normally come from game state)
        neighbors = graph.get_neighbors(current_word)
        if not neighbors:
            return ""
        
        # Filter out immediate RST words
        available = [word for word in neighbors.keys() 
                    if word[0].lower() not in graph.trap_letters]
        
        if not available:
            return ""  # No safe moves available
        
        # Choose best adversarial move
        chosen_word = strategy.choose_move(current_word, available)
        
        # Store explanation for debugging/learning
        if chosen_word:
            evaluations = []
            for word in available:
                eval_result = strategy._evaluate_move(current_word, word, available)
                evaluations.append(eval_result)
            
            chosen_eval = next((e for e in evaluations if e.word == chosen_word), None)
            if chosen_eval:
                explanation = strategy.get_strategy_explanation(chosen_word, chosen_eval)
                # Store explanation in game state for optional display
                game_state['last_strategy_explanation'] = explanation
        
        return chosen_word
    
    # Attach strategy object for external access
    adversarial_player.strategy = strategy
    return adversarial_player


def demonstrate_adversarial_strategy():
    """Demonstrate the adversarial strategy with examples"""
    print("🎯 ADVERSARIAL RST AVOIDANCE STRATEGY DEMONSTRATION")
    print("=" * 60)
    
    # Create a simple test graph
    test_graph = {
        "cat": {"dog": 0.4, "mouse": 0.3, "rat": 0.3},
        "dog": {"bone": 0.5, "tail": 0.3, "bark": 0.2},
        "mouse": {"cheese": 0.6, "trap": 0.4},
        "bone": {"dig": 0.7, "bury": 0.3},
        "cheese": {"yellow": 0.8, "smell": 0.2},
        "bark": {"tree": 0.6, "sound": 0.4},
        "tree": {"green": 0.5, "tall": 0.3, "root": 0.2},
        "yellow": {"sun": 0.7, "color": 0.3},
        "dig": {"hole": 0.6, "deep": 0.4},
        "bury": {"ground": 0.8, "hide": 0.2}
    }
    
    trap_letters = frozenset(['r', 's', 't'])
    wag = WordAssociationGraph(test_graph, trap_letters)
    
    # Test different difficulty levels
    for difficulty in ["easy", "medium", "hard", "expert"]:
        print(f"\n🎮 **{difficulty.upper()} DIFFICULTY**")
        print("-" * 40)
        
        strategy = AdversarialRSTStrategy(wag, difficulty)
        
        # Test from "cat"
        current_word = "cat"
        neighbors = wag.get_neighbors(current_word)
        available = [w for w in neighbors.keys() if w[0].lower() not in trap_letters]
        
        chosen = strategy.choose_move(current_word, available)
        
        # Get evaluation for explanation
        evaluation = strategy._evaluate_move(current_word, chosen, available)
        
        print(f"From '{current_word}', chose: **{chosen}**")
        print(f"Safety: {evaluation.self_safety:.2f}, Aggression: {evaluation.opponent_danger:.2f}")
        print(f"Strategic Value: {evaluation.strategic_value:.2f}")
        
        if evaluation.trap_paths:
            print(f"Trap paths created: {len(evaluation.trap_paths)}")


if __name__ == "__main__":
    demonstrate_adversarial_strategy()