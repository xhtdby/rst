#!/usr/bin/env python3
"""
Verbose RST Trap Simulation - Shows trap words and strategic intent

Note: Example/demo script. Not part of the core package API.
"""

import sys
import random
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from rst_trap_finder.core import WordAssociationGraph
from rst_trap_finder.multistep import MultiStepAnalyzer

class VerboseTrapSimulator:
    """Highly verbose RST trap simulation showing all strategic decisions"""
    
    def __init__(self, graph_path):
        print(f"🔄 Loading word association graph from {graph_path}")
        self.graph = WordAssociationGraph.from_csv(graph_path)
        self.analyzer = MultiStepAnalyzer(self.graph)
        
        words = len(self.graph.get_all_words())
        edges = sum(len(neighbors) for neighbors in self.graph.graph.values())
        print(f"📊 Loaded graph: {words:,} words, {edges:,} edges")
        
        # Sample some RST words for analysis
        self.rst_words = ["the", "and", "of", "to", "a", "in", "is", "it", "you", "that", 
                         "he", "was", "for", "on", "are", "as", "with", "his", "they", "i"]
        
        # Filter to only RST words that exist in our graph
        self.available_rst = [word for word in self.rst_words if word in self.graph.graph]
        print(f"🎯 Available RST trap words in graph: {len(self.available_rst)}")
        print(f"   RST words: {', '.join(self.available_rst[:10])}...")
    
    def find_simple_path(self, start_word, target_word, max_depth=3):
        """Find a simple path between two words using BFS"""
        if start_word == target_word:
            return [start_word], 1.0
        
        from collections import deque
        
        queue = deque([(start_word, [start_word], 1.0)])
        visited = {start_word}
        
        for _ in range(max_depth):
            if not queue:
                break
                
            for _ in range(len(queue)):
                current_word, path, prob = queue.popleft()
                
                neighbors = self.graph.get_neighbors(current_word)
                for next_word, weight in neighbors.items():
                    if next_word == target_word:
                        return path + [next_word], prob * weight
                    
                    if next_word not in visited and len(path) < max_depth:
                        visited.add(next_word)
                        queue.append((next_word, path + [next_word], prob * weight))
        
        return None, 0.0

    def analyze_trap_potential(self, word, max_depth=3):
        """Analyze why a word is a good trap"""
        print(f"\n🔍 ANALYZING TRAP POTENTIAL: '{word}'")
        print("=" * 60)
        
        # Check if it's an RST word
        is_rst = word in self.available_rst
        print(f"📋 RST Classification: {'✅ CONFIRMED RST TRAP' if is_rst else '❌ Not RST'}")
        
        # Get direct associations
        neighbors = self.graph.get_neighbors(word)
        print(f"🔗 Direct associations: {len(neighbors)} words")
        print(f"   Top connections: {', '.join(list(neighbors.keys())[:8])}...")
        
        # Analyze multi-step paths from this word
        print(f"\n🎲 STRATEGIC ANALYSIS - Multi-step trap potential:")
        
        # Sample some target words to see paths
        sample_targets = random.sample(list(self.graph.get_all_words()), min(5, len(self.graph.get_all_words())))
        
        for target in sample_targets:
            if target == word or target in self.available_rst:
                continue
                
            path, probability = self.find_simple_path(word, target, max_depth=max_depth)
            
            if path:
                print(f"   🎯 Path to '{target}': {' → '.join(path)} (prob: {probability:.4f})")
                print(f"      💡 Trap insight: Player starting at '{word}' has {probability:.1%} chance to reach '{target}'")
                
                if probability < 0.1:
                    print(f"      ⚠️  LOW ESCAPE PROBABILITY - This is a TRAP SCENARIO!")
                elif probability > 0.3:
                    print(f"      ✅ High escape probability - Easy path available")
                else:
                    print(f"      ⚖️  Medium difficulty - Moderate trap potential")
            else:
                print(f"   🚫 No path to '{target}' found within {max_depth} steps - PERFECT TRAP!")
        
        return is_rst, len(neighbors), neighbors

    def simulate_verbose_game(self, starting_word=None, target_word=None, max_turns=15):
        """Run a highly verbose RST game simulation"""
        print("\n" + "🎮" * 20 + " VERBOSE RST GAME SIMULATION " + "🎮" * 20)
        
        # Choose words
        if not starting_word:
            starting_word = random.choice(self.available_rst)
        
        if not target_word:
            all_words = list(self.graph.get_all_words())
            # Avoid RST words as targets
            non_rst_words = [w for w in all_words if w not in self.available_rst]
            target_word = random.choice(non_rst_words) if non_rst_words else random.choice(all_words)
        
        print(f"🎯 GAME SETUP:")
        print(f"   Starting word: '{starting_word}' {'(🎯 RST TRAP WORD!)' if starting_word in self.available_rst else ''}")
        print(f"   Target word: '{target_word}'")
        print(f"   Maximum turns: {max_turns}")
        
        # Analyze the overall challenge
        overall_path, overall_prob = self.find_simple_path(starting_word, target_word, max_depth=max_turns)
        print(f"\n📊 STRATEGIC OVERVIEW:")
        if overall_path:
            print(f"   🛤️  Optimal path exists: {' → '.join(overall_path)}")
            print(f"   📈 Path probability: {overall_prob:.6f} ({overall_prob:.3%})")
            print(f"   📏 Path length: {len(overall_path) - 1} steps")
            
            if overall_prob < 0.01:
                print(f"   🚨 EXTREME TRAP: Less than 1% success probability!")
            elif overall_prob < 0.1:
                print(f"   ⚠️  STRONG TRAP: Less than 10% success probability")
            elif overall_prob < 0.3:
                print(f"   ⚖️  MODERATE TRAP: 10-30% success probability")
            else:
                print(f"   ✅ MANAGEABLE: >30% success probability")
        else:
            print(f"   🚫 NO PATH FOUND - ULTIMATE TRAP SCENARIO!")
            print(f"   💀 Player will likely get stuck in association loops")
        
        # Simulate actual gameplay
        print(f"\n🎭 GAMEPLAY SIMULATION:")
        print("=" * 80)
        
        current_word = starting_word
        turn = 0
        path_taken = [starting_word]
        
        while turn < max_turns and current_word != target_word:
            turn += 1
            print(f"\n🔄 TURN {turn}: Currently at '{current_word}'")
            
            # Get available moves
            neighbors = self.graph.get_neighbors(current_word)
            if not neighbors:
                print(f"   💀 DEAD END! No associations from '{current_word}' - TRAPPED!")
                break
            
            print(f"   🔗 Available associations ({len(neighbors)}): {', '.join(list(neighbors.keys())[:10])}...")
            
            # Analyze each option strategically
            print(f"   🧠 STRATEGIC ANALYSIS of top options:")
            
            sorted_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for next_word, weight in sorted_neighbors:
                # Check if this would be a trap move
                is_rst_move = next_word in self.available_rst
                
                # Check probability to target from this word
                future_path, future_prob = self.find_simple_path(next_word, target_word, max_depth=max_turns-turn)
                
                print(f"      → '{next_word}' (weight: {weight:.3f})", end="")
                
                if is_rst_move:
                    print(f" 🎯 RST TRAP! ", end="")
                
                if future_path:
                    print(f" Success prob: {future_prob:.4f} ({future_prob:.2%})", end="")
                    
                    if future_prob < 0.05:
                        print(f" ⚠️ TRAP MOVE!")
                    elif future_prob > 0.2:
                        print(f" ✅ GOOD MOVE!")
                    else:
                        print(f" ⚖️ RISKY")
                else:
                    print(f" 🚫 NO PATH - TRAP!")
            
            # Choose next word (simulate human-like choice with some randomness but strategy awareness)
            # Weight by both association strength and strategic value
            strategic_weights = {}
            
            for next_word, weight in neighbors.items():
                future_path, future_prob = self.find_simple_path(next_word, target_word, max_depth=max_turns-turn)
                
                # Penalty for RST words (traps)
                rst_penalty = 0.5 if next_word in self.available_rst else 1.0
                
                # Bonus for words that can reach target
                path_bonus = future_prob if future_path else 0.001
                
                strategic_weights[next_word] = weight * rst_penalty * (1 + path_bonus * 10)
            
            # Choose best strategic option with some randomness
            choices = list(strategic_weights.keys())
            weights = list(strategic_weights.values())
            
            # Add some noise to simulate human decision making
            noisy_weights = [w * random.uniform(0.7, 1.3) for w in weights]
            
            chosen_word = random.choices(choices, weights=noisy_weights)[0]
            
            print(f"\n   ✅ CHOICE: '{chosen_word}' (weight: {neighbors[chosen_word]:.3f})")
            
            # Explain the choice
            is_trap_choice = chosen_word in self.available_rst
            future_path, future_prob = self.find_simple_path(chosen_word, target_word, max_depth=max_turns-turn)
            
            if is_trap_choice:
                print(f"      🎯 ENTERED RST TRAP! Player chose another function word.")
                print(f"      💡 Strategic implication: Function words lead to more function words")
            
            if future_path:
                print(f"      📈 Strategic outlook: {future_prob:.2%} success probability remaining")
                if future_prob < 0.1:
                    print(f"      ⚠️  Player is in TRAP TERRITORY - low escape probability!")
            else:
                print(f"      💀 Strategic outlook: NO CLEAR PATH TO TARGET - Deep in trap!")
            
            current_word = chosen_word
            path_taken.append(current_word)
        
        # Game conclusion
        print(f"\n🏁 GAME CONCLUSION:")
        print("=" * 50)
        
        if current_word == target_word:
            print(f"   🎉 SUCCESS! Reached target '{target_word}' in {turn} turns")
            print(f"   📍 Final path: {' → '.join(path_taken)}")
        else:
            print(f"   💀 TRAPPED! Failed to reach '{target_word}' in {max_turns} turns")
            print(f"   📍 Path taken: {' → '.join(path_taken)}")
            print(f"   🎯 Ended at: '{current_word}'")
            
            # Count RST words encountered
            rst_encountered = [w for w in path_taken if w in self.available_rst]
            print(f"   📊 RST trap words encountered: {len(rst_encountered)} ({', '.join(rst_encountered)})")
            
            if len(rst_encountered) > len(path_taken) // 2:
                print(f"   🚨 TRAP ANALYSIS: Player fell into RST cycle - spent most time in function words!")
            elif len(rst_encountered) > 2:
                print(f"   ⚠️  TRAP ANALYSIS: Multiple RST encounters indicate trap susceptibility")
            else:
                print(f"   ✅ TRAP ANALYSIS: Avoided major RST traps but still failed")

def main():
    # Use the larger merged dataset
    dataset_path = "data/merged/merged_association_graph.csv"
    
    simulator = VerboseTrapSimulator(dataset_path)
    
    print("\n" + "🔥" * 20 + " TRAP WORD ANALYSIS " + "🔥" * 20)
    
    # Analyze a few specific trap words
    for trap_word in ["the", "and", "it"][:2]:  # Limit to avoid too much output
        if trap_word in simulator.available_rst:
            simulator.analyze_trap_potential(trap_word)
    
    # Run a verbose game simulation
    print("\n\n" + "⚡" * 20 + " STARTING VERBOSE SIMULATION " + "⚡" * 20)
    simulator.simulate_verbose_game()
    
    # Run another one with a specific challenging setup
    print("\n\n" + "🔥" * 20 + " CHALLENGING SCENARIO " + "🔥" * 20)
    simulator.simulate_verbose_game(starting_word="the", target_word="elephant")

if __name__ == "__main__":
    main()
