#!/usr/bin/env python3
"""
Interactive RST Word Association Game CLI

Supports:
- Player vs Computer
- Player vs Player  
- Computer vs Computer
- Word analysis and web exploration
- Adversarial mode (avoid-RST)
"""

import sys
import argparse
import random
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from rst_trap_finder.core import WordAssociationGraph
from rst_trap_finder.multistep import MultiStepAnalyzer

class GameState:
    """Track game state and history"""
    
    def __init__(self, graph: WordAssociationGraph, target_word: str = None, max_turns: int = 20):
        self.graph = graph
        self.target_word = target_word
        self.max_turns = max_turns
        self.current_word = None
        self.path = []
        self.turn = 0
        self.game_over = False
        self.winner = None
        self.analyzer = MultiStepAnalyzer(graph)
        
    def start_game(self, starting_word: str):
        """Initialize game with starting word"""
        self.current_word = starting_word
        self.path = [starting_word]
        self.turn = 1
        
    def make_move(self, next_word: str) -> bool:
        """Make a move to next_word. Returns True if valid, False otherwise."""
        if not self.is_valid_move(next_word):
            return False
            
        self.current_word = next_word
        self.path.append(next_word)
        self.turn += 1
        
        # Check win condition
        if self.target_word and next_word == self.target_word:
            self.game_over = True
            self.winner = "current_player"
            
        # Check max turns
        if self.turn > self.max_turns:
            self.game_over = True
            self.winner = "timeout"
            
        return True
        
    def is_valid_move(self, next_word: str) -> bool:
        """Check if move is valid (word is neighbor of current word)"""
        if not self.current_word:
            return False
        neighbors = self.graph.get_neighbors(self.current_word)
        return next_word in neighbors
        
    def get_available_moves(self) -> Dict[str, float]:
        """Get all valid moves from current position"""
        if not self.current_word:
            return {}
        return self.graph.get_neighbors(self.current_word)

class Player:
    """Base player class"""
    
    def __init__(self, name: str, is_human: bool = False):
        self.name = name
        self.is_human = is_human
        
    def get_move(self, game_state: GameState) -> str:
        """Get next move. Override in subclasses."""
        raise NotImplementedError

class HumanPlayer(Player):
    """Human player that gets input from console"""
    
    def __init__(self, name: str = "Human"):
        super().__init__(name, is_human=True)
        
    def get_move(self, game_state: GameState) -> str:
        """Get move from human input"""
        available = game_state.get_available_moves()
        
        if not available:
            print("💀 No available moves! You're trapped!")
            return ""
            
        print(f"\n🎯 Your turn! Current word: '{game_state.current_word}'")
        if game_state.target_word:
            print(f"🏁 Target: '{game_state.target_word}'")
        print(f"⏱️  Turn {game_state.turn}/{game_state.max_turns}")
        
        # Show available moves
        sorted_moves = sorted(available.items(), key=lambda x: x[1], reverse=True)
        print(f"\n📋 Available moves ({len(sorted_moves)}):")
        
        # Show top 10 moves
        for i, (word, weight) in enumerate(sorted_moves[:10]):
            rst_marker = "🎯" if word[0].lower() in game_state.graph.trap_letters else "  "
            print(f"  {i+1:2}. {rst_marker} {word:<20} (weight: {weight:.3f})")
            
        if len(sorted_moves) > 10:
            print(f"     ... and {len(sorted_moves) - 10} more options")
            
        # Get input
        while True:
            try:
                choice = input("\n💭 Enter word (or number 1-10, 'help', 'analyze'): ").strip()
                
                if choice.lower() == 'help':
                    self._show_help(game_state)
                    continue
                elif choice.lower() == 'analyze':
                    self._show_analysis(game_state)
                    continue
                elif choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < min(10, len(sorted_moves)):
                        return sorted_moves[idx][0]
                    else:
                        print(f"❌ Invalid number. Choose 1-{min(10, len(sorted_moves))}")
                        continue
                elif choice in available:
                    return choice
                else:
                    print(f"❌ Invalid move. '{choice}' is not connected to '{game_state.current_word}'")
                    continue
                    
            except (KeyboardInterrupt, EOFError):
                print("\n👋 Game interrupted by user")
                return ""
                
    def _show_help(self, game_state: GameState):
        """Show help information"""
        print("\n" + "="*60)
        print("🆘 HELP")
        print("="*60)
        print("Goal: Navigate from word to word using associations")
        if game_state.target_word:
            print(f"Win condition: Reach the target word '{game_state.target_word}'")
        print("🎯 Words starting with R/S/T are traps (harder to escape)")
        print("💡 Higher weight = stronger association")
        print("Commands:")
        print("  - Type a word to move to it")
        print("  - Type a number (1-10) to choose from top moves")
        print("  - Type 'analyze' to see detailed word analysis")
        print("  - Type 'help' to see this help")
        print("="*60)
        
    def _show_analysis(self, game_state: GameState):
        """Show detailed analysis of current word"""
        word = game_state.current_word
        analyze_word(game_state.graph, word, detailed=True)

class ComputerPlayer(Player):
    """Computer player with different strategies"""
    
    def __init__(self, name: str = "Computer", strategy: str = "smart", difficulty: float = 0.8):
        super().__init__(name, is_human=False)
        self.strategy = strategy
        self.difficulty = difficulty  # 0.0 = random, 1.0 = optimal
        
    def get_move(self, game_state: GameState) -> str:
        """Get computer move based on strategy"""
        available = game_state.get_available_moves()
        
        if not available:
            return ""
            
        if self.strategy == "random":
            return random.choice(list(available.keys()))
        elif self.strategy == "greedy":
            return max(available.items(), key=lambda x: x[1])[0]
        elif self.strategy == "smart":
            return self._smart_move(game_state, available)
        elif self.strategy == "avoid_rst":
            return self._avoid_rst_move(game_state, available)
        else:
            return self._smart_move(game_state, available)
            
    def _smart_move(self, game_state: GameState, available: Dict[str, float]) -> str:
        """Smart strategy considering target and trap avoidance"""
        if not game_state.target_word:
            # No target, avoid RST words and prefer high weights
            scored_moves = []
            for word, weight in available.items():
                score = weight
                # Penalty for RST words
                if word[0].lower() in game_state.graph.trap_letters:
                    score *= 0.3
                scored_moves.append((word, score))
        else:
            # Target exists, try to find path to target
            scored_moves = []
            for word, weight in available.items():
                score = weight
                
                # Check if this word can reach target
                try:
                    # Simple BFS to check reachability
                    if self._can_reach_target(word, game_state.target_word, game_state.graph, max_depth=3):
                        score *= 2.0  # Boost words that can reach target
                except:
                    pass
                
                # Penalty for RST words unless we're close to target
                if word[0].lower() in game_state.graph.trap_letters:
                    score *= 0.5
                    
                scored_moves.append((word, score))
        
        # Sort by score and add some randomness based on difficulty
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        
        # Choose from top moves with some randomness
        if random.random() < self.difficulty:
            # Choose from top 3 moves
            top_moves = scored_moves[:3]
            return random.choice(top_moves)[0]
        else:
            # Choose randomly
            return random.choice(scored_moves)[0]
            
    def _avoid_rst_move(self, game_state: GameState, available: Dict[str, float]) -> str:
        """Adversarial strategy: minimize opponent's chance of reaching RST"""
        scored_moves = []
        
        for word, weight in available.items():
            score = weight
            
            # Check what moves opponent would have from this word
            opponent_moves = game_state.graph.get_neighbors(word)
            rst_risk = 0
            
            for opp_word in opponent_moves:
                if opp_word[0].lower() in game_state.graph.trap_letters:
                    rst_risk += opponent_moves[opp_word]
                    
            # Prefer moves that give opponent high RST risk
            if opponent_moves:
                rst_ratio = rst_risk / sum(opponent_moves.values())
                score *= (1 + rst_ratio)
                
            scored_moves.append((word, score))
            
        # Choose best adversarial move
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        return scored_moves[0][0]
        
    def _can_reach_target(self, start: str, target: str, graph: WordAssociationGraph, max_depth: int = 3) -> bool:
        """Simple BFS to check if target is reachable"""
        if start == target:
            return True
            
        visited = {start}
        queue = [(start, 0)]
        
        while queue:
            current, depth = queue.pop(0)
            if depth >= max_depth:
                continue
                
            neighbors = graph.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor == target:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))
                    
        return False

def analyze_word(graph: WordAssociationGraph, word: str, detailed: bool = False):
    """Analyze a word and show its connections and properties"""
    print(f"\n" + "🔍" * 20 + f" ANALYZING: '{word}' " + "🔍" * 20)
    
    if word not in graph.graph:
        print(f"❌ Word '{word}' not found in graph")
        return
        
    neighbors = graph.get_neighbors(word)
    
    # Basic stats
    print(f"📊 BASIC STATS:")
    print(f"   Connections: {len(neighbors)}")
    print(f"   Is RST word: {'✅ Yes' if word[0].lower() in graph.trap_letters else '❌ No'}")
    
    if neighbors:
        weights = list(neighbors.values())
        print(f"   Weight range: {min(weights):.3f} - {max(weights):.3f}")
        print(f"   Average weight: {sum(weights) / len(weights):.3f}")
    
    # Top connections
    print(f"\n🔗 TOP CONNECTIONS:")
    sorted_neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)
    for i, (neighbor, weight) in enumerate(sorted_neighbors[:10]):
        rst_marker = "🎯" if neighbor[0].lower() in graph.trap_letters else "  "
        print(f"   {i+1:2}. {rst_marker} {neighbor:<20} ({weight:.3f})")
        
    if len(sorted_neighbors) > 10:
        print(f"   ... and {len(sorted_neighbors) - 10} more")
    
    # RST connections
    rst_connections = [(n, w) for n, w in neighbors.items() if n[0].lower() in graph.trap_letters]
    if rst_connections:
        print(f"\n🎯 RST CONNECTIONS ({len(rst_connections)}):")
        for neighbor, weight in sorted(rst_connections, key=lambda x: x[1], reverse=True)[:5]:
            print(f"   🎯 {neighbor:<20} ({weight:.3f})")
    else:
        print(f"\n✅ NO RST CONNECTIONS (good for avoiding traps)")
    
    if detailed:
        # Incoming connections
        print(f"\n⬅️  INCOMING CONNECTIONS:")
        incoming = []
        for source, targets in graph.graph.items():
            if word in targets:
                incoming.append((source, targets[word]))
                
        if incoming:
            sorted_incoming = sorted(incoming, key=lambda x: x[1], reverse=True)[:10]
            for source, weight in sorted_incoming:
                rst_marker = "🎯" if source[0].lower() in graph.trap_letters else "  "
                print(f"   {rst_marker} {source:<20} ({weight:.3f})")
        else:
            print("   No incoming connections found")
            
        # PageRank score
        try:
            pagerank_scores = graph.biased_pagerank()
            if word in pagerank_scores:
                pr_score = pagerank_scores[word]
                print(f"\n📈 PAGERANK SCORE: {pr_score:.6f}")
                
                # Rank among all words
                sorted_words = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
                rank = next(i for i, (w, _) in enumerate(sorted_words) if w == word) + 1
                print(f"   Rank: #{rank} out of {len(sorted_words)} words")
        except:
            print(f"\n📈 PageRank calculation failed")
    
    print("=" * 60)

def start_game_cli():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="RST Word Association Game")
    
    # Dataset argument
    parser.add_argument("dataset", nargs="?", default="data/merged/pruned_association_graph.csv",
                       help="Path to word association CSV file")
    
    # Game mode
    parser.add_argument("--mode", choices=["pvp", "pvc", "cvc"], default="pvc",
                       help="Game mode: pvp (player vs player), pvc (player vs computer), cvc (computer vs computer)")
    
    # Game settings
    parser.add_argument("--target", type=str, help="Target word to reach (optional)")
    parser.add_argument("--start", type=str, help="Starting word (random if not specified)")
    parser.add_argument("--turns", type=int, default=20, help="Maximum number of turns")
    parser.add_argument("--steps", type=int, help="For cvc mode: number of games to simulate")
    
    # Computer player settings
    parser.add_argument("--strategy", choices=["random", "greedy", "smart", "avoid_rst"], 
                       default="smart", help="Computer strategy")
    parser.add_argument("--difficulty", type=float, default=0.8, 
                       help="Computer difficulty (0.0=random, 1.0=optimal)")
    
    # Analysis mode
    parser.add_argument("--analyze", type=str, help="Analyze a specific word")
    parser.add_argument("--detailed", action="store_true", help="Show detailed analysis")
    
    args = parser.parse_args()
    
    # Load graph
    print(f"🔄 Loading word association graph from {args.dataset}")
    try:
        graph = WordAssociationGraph.from_csv(args.dataset)
        words = len(graph.get_all_words())
        edges = sum(len(neighbors) for neighbors in graph.graph.values())
        print(f"📊 Loaded: {words:,} words, {edges:,} edges")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Analysis mode
    if args.analyze:
        analyze_word(graph, args.analyze, args.detailed)
        return
    
    # Choose starting word
    all_words = list(graph.get_all_words())
    if args.start:
        if args.start not in all_words:
            print(f"❌ Starting word '{args.start}' not found in dataset")
            return
        starting_word = args.start
    else:
        starting_word = random.choice(all_words)
    
    # Choose target word
    target_word = None
    if args.target:
        if args.target not in all_words:
            print(f"❌ Target word '{args.target}' not found in dataset")
            return
        target_word = args.target
    
    # Set up game
    game_state = GameState(graph, target_word, args.turns)
    
    if args.mode == "pvp":
        player1 = HumanPlayer("Player 1")
        player2 = HumanPlayer("Player 2")
    elif args.mode == "pvc":
        player1 = HumanPlayer("Player")
        player2 = ComputerPlayer("Computer", args.strategy, args.difficulty)
    else:  # cvc
        player1 = ComputerPlayer("Computer 1", args.strategy, args.difficulty)
        player2 = ComputerPlayer("Computer 2", "smart", 0.7)
    
    # Run game(s)
    if args.mode == "cvc" and args.steps:
        # Simulate multiple games
        run_simulations(graph, player1, player2, args.steps, args.turns)
    else:
        # Run single interactive game
        run_single_game(game_state, player1, player2, starting_word)

def run_single_game(game_state: GameState, player1: Player, player2: Player, starting_word: str):
    """Run a single interactive game"""
    print(f"\n🎮 STARTING GAME: {player1.name} vs {player2.name}")
    print(f"🎯 Starting word: '{starting_word}'")
    if game_state.target_word:
        print(f"🏁 Target word: '{game_state.target_word}'")
    print(f"⏱️  Max turns: {game_state.max_turns}")
    print("=" * 60)
    
    game_state.start_game(starting_word)
    players = [player1, player2]
    current_player_idx = 0
    
    while not game_state.game_over:
        current_player = players[current_player_idx]
        
        print(f"\n🎯 {current_player.name}'s turn (Turn {game_state.turn})")
        print(f"📍 Current word: '{game_state.current_word}'")
        print(f"🛤️  Path: {' → '.join(game_state.path)}")
        
        # Get move
        try:
            if current_player.is_human:
                move = current_player.get_move(game_state)
            else:
                available = game_state.get_available_moves()
                if available:
                    move = current_player.get_move(game_state)
                    print(f"🤖 {current_player.name} chooses: '{move}'")
                    time.sleep(1)  # Pause for dramatic effect
                else:
                    move = ""
                    
            if not move:
                print(f"💀 {current_player.name} is trapped! Game over.")
                game_state.game_over = True
                game_state.winner = players[1 - current_player_idx].name
                break
                
            # Make move
            if game_state.make_move(move):
                if game_state.game_over and game_state.winner == "current_player":
                    game_state.winner = current_player.name
            else:
                print(f"❌ Invalid move: '{move}'")
                continue
                
        except KeyboardInterrupt:
            print("\n👋 Game interrupted")
            return
            
        # Switch players
        current_player_idx = 1 - current_player_idx
    
    # Game over
    print(f"\n🏁 GAME OVER!")
    print(f"🛤️  Final path: {' → '.join(game_state.path)}")
    print(f"📊 Total turns: {game_state.turn - 1}")
    
    if game_state.winner == "timeout":
        print(f"⏰ Game ended due to turn limit")
    elif game_state.target_word and game_state.current_word == game_state.target_word:
        print(f"🎉 {game_state.winner} wins by reaching the target!")
    else:
        print(f"🏆 {game_state.winner} wins!")

def run_simulations(graph: WordAssociationGraph, player1: Player, player2: Player, 
                   num_games: int, max_turns: int):
    """Run multiple computer vs computer simulations"""
    print(f"\n🤖 RUNNING {num_games} SIMULATIONS: {player1.name} vs {player2.name}")
    
    results = {
        player1.name: 0,
        player2.name: 0,
        "timeout": 0,
        "trapped": 0
    }
    
    total_turns = []
    all_words = list(graph.get_all_words())
    
    for game_num in range(num_games):
        if game_num % 10 == 0:
            print(f"🎮 Game {game_num + 1}/{num_games}")
            
        # Random starting word for each game
        starting_word = random.choice(all_words)
        game_state = GameState(graph, None, max_turns)
        game_state.start_game(starting_word)
        
        players = [player1, player2]
        current_player_idx = 0
        
        while not game_state.game_over:
            current_player = players[current_player_idx]
            available = game_state.get_available_moves()
            
            if not available:
                results["trapped"] += 1
                break
                
            move = current_player.get_move(game_state)
            if not move or not game_state.make_move(move):
                results["trapped"] += 1
                break
                
            current_player_idx = 1 - current_player_idx
        
        # Record results
        total_turns.append(game_state.turn - 1)
        
        if game_state.winner == "timeout":
            results["timeout"] += 1
        elif game_state.winner:
            results[game_state.winner] += 1
    
    # Print results
    print(f"\n📊 SIMULATION RESULTS ({num_games} games):")
    print("=" * 40)
    for outcome, count in results.items():
        percentage = (count / num_games) * 100
        print(f"{outcome:15}: {count:4} ({percentage:5.1f}%)")
    
    if total_turns:
        avg_turns = sum(total_turns) / len(total_turns)
        print(f"\n📈 Average game length: {avg_turns:.1f} turns")
        print(f"📈 Turn range: {min(total_turns)} - {max(total_turns)}")

if __name__ == "__main__":
    start_game_cli()