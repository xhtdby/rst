#!/usr/bin/env python3
"""
Quick ML Gaming Engine Demo

This demonstrates how to extend the current RST foundation into an ML-based
gaming engine. Shows the immediate potential for your fork.

Usage:
    python examples/ml_gaming_demo.py
"""

import sys
import random
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Add src directory to path  
sys.path.append(str(Path(__file__).parent.parent / "src"))

from rst_trap_finder.core import WordAssociationGraph

@dataclass
class GameState:
    """Enhanced game state for ML training"""
    current_word: str
    move_history: List[str]
    available_moves: List[str]
    current_player: int
    threat_level: float  # 0.0 = safe, 1.0 = immediate RST danger
    
    def to_feature_vector(self) -> List[float]:
        """Convert game state to ML features"""
        features = []
        
        # Basic state features
        features.append(len(self.move_history))  # Game progress
        features.append(self.threat_level)       # Current danger
        features.append(len(self.available_moves))  # Move options
        
        # Word characteristics
        features.append(len(self.current_word))  # Word length
        features.append(1.0 if self.current_word[0].lower() in 'rst' else 0.0)  # RST word
        
        # History patterns
        rst_count = sum(1 for word in self.move_history if word[0].lower() in 'rst')
        features.append(rst_count / max(1, len(self.move_history)))  # RST frequency
        
        return features

class SimpleNeuralPlayer:
    """
    Simple neural network player using basic feedforward network.
    This demonstrates the ML integration potential - you can replace
    this with sophisticated transformers, GNNs, or RL agents.
    """
    
    def __init__(self, name: str = "SimpleNeural"):
        self.name = name
        self.weights = self._initialize_weights()
        self.learning_rate = 0.01
        self.training_data = []
        
    def _initialize_weights(self) -> List[float]:
        """Initialize simple neural network weights"""
        # 6 input features -> 4 hidden -> 1 output
        return [random.uniform(-1, 1) for _ in range(6 * 4 + 4 * 1 + 4 + 1)]
    
    def _forward_pass(self, features: List[float]) -> float:
        """Simple feedforward neural network"""
        # This is a simplified NN - replace with PyTorch/TensorFlow for real models
        
        # Input to hidden layer (6 -> 4)
        hidden = []
        for h in range(4):
            activation = self.weights[24 + h]  # bias
            for i in range(6):
                activation += features[i] * self.weights[h * 6 + i]
            hidden.append(max(0, activation))  # ReLU
        
        # Hidden to output (4 -> 1)  
        output = self.weights[-1]  # bias
        for h in range(4):
            output += hidden[h] * self.weights[24 + 4 + h]
            
        return 1.0 / (1.0 + math.exp(-output))  # sigmoid
    
    def evaluate_move(self, game_state: GameState, move: str) -> float:
        """Evaluate how good a move is (0.0 = bad, 1.0 = excellent)"""
        # Create hypothetical state after this move
        features = game_state.to_feature_vector()
        
        # Add move-specific features
        move_features = features.copy()
        move_features.append(len(move))  # Move word length
        move_features.append(1.0 if move[0].lower() in 'rst' else 0.0)  # RST move
        
        # Pad to expected input size
        while len(move_features) < 6:
            move_features.append(0.0)
        move_features = move_features[:6]
        
        return self._forward_pass(move_features)
    
    def choose_move(self, game_state: GameState) -> str:
        """Choose best move using neural network evaluation"""
        if not game_state.available_moves:
            return "pass"
            
        move_scores = {}
        for move in game_state.available_moves:
            score = self.evaluate_move(game_state, move)
            move_scores[move] = score
        
        # Choose move with highest score (with some randomness for exploration)
        if random.random() < 0.1:  # 10% exploration
            return random.choice(game_state.available_moves)
        else:
            return max(move_scores.items(), key=lambda x: x[1])[0]
    
    def learn_from_game(self, game_result: Dict):
        """Simple learning from game outcomes"""
        # In a real implementation, this would do backpropagation
        # For now, just record data for batch training
        self.training_data.append(game_result)
        
        # Simple weight adjustment based on win/loss
        adjustment = 0.01 if game_result['won'] else -0.01
        for i in range(len(self.weights)):
            self.weights[i] += adjustment * random.uniform(-1, 1)

class MLGameEngine:
    """Enhanced game engine with ML player support"""
    
    def __init__(self):
        self.graph = None
        self.players = {}
        self.game_history = []
        
    def load_graph(self, graph_path: Optional[str] = None):
        """Load word association graph"""
        if graph_path:
            self.graph = WordAssociationGraph.from_csv(graph_path)
        else:
            # Create simple demo graph for testing
            demo_graph = {
                'cat': {'dog': 0.8, 'pet': 0.6, 'animal': 0.5},
                'dog': {'cat': 0.8, 'pet': 0.7, 'bark': 0.6},
                'pet': {'cat': 0.6, 'dog': 0.7, 'animal': 0.4},
                'animal': {'cat': 0.5, 'dog': 0.6, 'wild': 0.3},
                'bark': {'dog': 0.6, 'tree': 0.4, 'sound': 0.3},
                'tree': {'bark': 0.4, 'leaf': 0.5, 'forest': 0.6},
                'forest': {'tree': 0.6, 'wild': 0.5, 'green': 0.4},
                'wild': {'animal': 0.3, 'forest': 0.5, 'nature': 0.4},
                'nature': {'wild': 0.4, 'green': 0.5, 'earth': 0.3},
                'green': {'tree': 0.3, 'forest': 0.4, 'nature': 0.5}
            }
            self.graph = WordAssociationGraph(demo_graph)
    
    def add_player(self, name: str, player):
        """Add a player to the engine"""
        self.players[name] = player
    
    def get_available_moves(self, current_word: str, max_moves: int = 8) -> List[str]:
        """Get available moves from current word"""
        if not self.graph or current_word not in self.graph.graph:
            # Fallback to demo moves
            return ['dog', 'pet', 'animal', 'house', 'friend'][:max_moves]
        
        # Get connected words, filter out RST words, sort by strength
        moves = []
        for word, strength in self.graph.graph[current_word].items():
            if not word[0].lower() in 'rst':  # Avoid RST words
                moves.append((word, strength))
        
        # Sort by association strength and return top moves
        moves.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in moves[:max_moves]]
    
    def calculate_threat_level(self, current_word: str, available_moves: List[str]) -> float:
        """Calculate how dangerous the current position is"""
        if not available_moves:
            return 1.0  # Maximum danger if no moves
            
        rst_moves = sum(1 for move in available_moves if move[0].lower() in 'rst')
        return rst_moves / len(available_moves)
    
    def play_game(self, player1_name: str, player2_name: str, starting_word: str = "cat") -> Dict:
        """Play a complete game between two players"""
        player1 = self.players[player1_name]
        player2 = self.players[player2_name]
        
        game_state = GameState(
            current_word=starting_word,
            move_history=[starting_word],
            available_moves=self.get_available_moves(starting_word),
            current_player=0,
            threat_level=0.0
        )
        
        print(f"\n🎮 GAME: {player1_name} vs {player2_name}")
        print(f"Starting word: {starting_word}")
        print("=" * 50)
        
        turn = 0
        max_turns = 20  # Prevent infinite games
        
        while turn < max_turns:
            current_player = player1 if game_state.current_player == 0 else player2
            current_name = player1_name if game_state.current_player == 0 else player2_name
            
            # Update threat level
            game_state.threat_level = self.calculate_threat_level(
                game_state.current_word, 
                game_state.available_moves
            )
            
            print(f"\nTurn {turn + 1} - {current_name}'s move")
            print(f"Current word: {game_state.current_word}")
            print(f"Available moves: {', '.join(game_state.available_moves)}")
            print(f"Threat level: {game_state.threat_level:.2f}")
            
            # Player chooses move
            chosen_move = current_player.choose_move(game_state)
            
            # Check if move is valid
            if chosen_move not in game_state.available_moves:
                print(f"❌ Invalid move '{chosen_move}' by {current_name}!")
                winner = player2_name if current_name == player1_name else player1_name
                break
            
            # Check if it's an RST word (player loses)
            if chosen_move[0].lower() in 'rst':
                print(f"💀 {current_name} chose RST word '{chosen_move}' and loses!")
                winner = player2_name if current_name == player1_name else player1_name
                break
            
            print(f"✅ {current_name} chose: {chosen_move}")
            
            # Update game state
            game_state.current_word = chosen_move
            game_state.move_history.append(chosen_move)
            game_state.available_moves = self.get_available_moves(chosen_move)
            game_state.current_player = 1 - game_state.current_player
            
            # Check if no moves available (current player loses)
            if not game_state.available_moves:
                print(f"💀 No moves available! {current_name} loses!")
                winner = player2_name if current_name == player1_name else player1_name
                break
                
            turn += 1
        
        else:
            # Max turns reached - declare draw or choose winner by threat level
            winner = "draw"
            print(f"🤝 Game ended in draw after {max_turns} turns")
        
        # Record game result
        game_result = {
            'player1': player1_name,
            'player2': player2_name,
            'winner': winner,
            'turns': turn + 1,
            'move_history': game_state.move_history.copy(),
            'final_threat': game_state.threat_level
        }
        
        self.game_history.append(game_result)
        
        print(f"\n🏆 Winner: {winner}")
        print(f"Game length: {turn + 1} turns")
        print(f"Move sequence: {' -> '.join(game_state.move_history)}")
        
        return game_result
    
    def run_tournament(self, num_games: int = 5) -> Dict[str, int]:
        """Run tournament between all registered players"""
        print(f"\n🏆 TOURNAMENT: {num_games} games between all players")
        print("=" * 60)
        
        player_names = list(self.players.keys())
        wins = {name: 0 for name in player_names}
        
        game_count = 0
        for i, player1 in enumerate(player_names):
            for j, player2 in enumerate(player_names):
                if i != j:  # Don't play against self
                    for game_num in range(num_games):
                        result = self.play_game(player1, player2)
                        if result['winner'] == player1:
                            wins[player1] += 1
                        elif result['winner'] == player2:
                            wins[player2] += 1
                        game_count += 1
        
        # Display results
        print(f"\n🏆 TOURNAMENT RESULTS ({game_count} total games)")
        print("=" * 40)
        for player, win_count in sorted(wins.items(), key=lambda x: x[1], reverse=True):
            win_rate = win_count / game_count * 100 if game_count > 0 else 0
            print(f"{player:<20} {win_count:>3} wins ({win_rate:>5.1f}%)")
        
        return wins

def main():
    """Demonstrate ML gaming engine capabilities"""
    print("🤖 RST ML Gaming Engine Demo")
    print("=" * 40)
    print("This shows how to extend RST into an ML-based gaming engine!")
    print()
    
    # Initialize engine
    engine = MLGameEngine()
    engine.load_graph()  # Use demo graph
    
    # Create different types of players
    
    # 1. Simple rule-based player (baseline)
    class RuleBasedPlayer:
        def __init__(self, name="RuleBot"):
            self.name = name
            
        def choose_move(self, game_state: GameState) -> str:
            # Simple strategy: avoid RST words, prefer longer words
            safe_moves = [m for m in game_state.available_moves if not m[0].lower() in 'rst']
            if safe_moves:
                return max(safe_moves, key=len)  # Choose longest word
            return game_state.available_moves[0] if game_state.available_moves else "pass"
    
    # 2. Random player (for comparison)
    class RandomPlayer:
        def __init__(self, name="RandomBot"):
            self.name = name
            
        def choose_move(self, game_state: GameState) -> str:
            safe_moves = [m for m in game_state.available_moves if not m[0].lower() in 'rst']
            return random.choice(safe_moves) if safe_moves else random.choice(game_state.available_moves)
    
    # 3. Neural network player
    neural_player = SimpleNeuralPlayer("NeuralBot")
    
    # Add players to engine
    engine.add_player("RuleBot", RuleBasedPlayer())
    engine.add_player("RandomBot", RandomPlayer())
    engine.add_player("NeuralBot", neural_player)
    
    print("🎮 Players registered:")
    for name, player in engine.players.items():
        print(f"  - {name}: {type(player).__name__}")
    
    # Run some demo games
    print("\n🎯 Running demo games...")
    
    # Single game demonstration
    engine.play_game("RuleBot", "NeuralBot", "cat")
    
    # Mini tournament
    engine.run_tournament(num_games=2)  # Small tournament for demo
    
    print("\n🚀 ML GAMING ENGINE POTENTIAL:")
    print("=" * 40)
    print("✅ Game state representation for ML training")
    print("✅ Player interface for different AI architectures") 
    print("✅ Tournament system for model comparison")
    print("✅ Training data generation from game outcomes")
    print("✅ Easy integration with PyTorch/TensorFlow")
    print()
    print("🎯 YOUR NEXT STEPS:")
    print("1. Fork this repository")
    print("2. Replace SimpleNeuralPlayer with real neural networks")
    print("3. Add PyTorch/TensorFlow models (GNNs, Transformers, RL)")
    print("4. Implement training pipelines and model optimization")
    print("5. Create web interface for online gaming")
    print()
    print("💡 This is just the beginning - the foundation is PERFECT for ML!")

if __name__ == "__main__":
    main()