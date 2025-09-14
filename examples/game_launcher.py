#!/usr/bin/env python3
"""
RST Game Launcher - Consolidated Interface

This consolidated launcher provides all game modes and analysis tools in one place:
- Quick text-based gameplay (no dependencies)
- Full interactive game with AI opponents
- Word analysis and strategic tools
- Computer vs computer demonstrations

Usage:
    python examples/game_launcher.py [options]
    
Examples:
    python examples/game_launcher.py                    # Interactive menu
    python examples/game_launcher.py --simple          # Simple text game
    python examples/game_launcher.py --pvp             # Player vs Player
    python examples/game_launcher.py --pvc medium      # Player vs Computer
    python examples/game_launcher.py --demo            # Computer demo
    python examples/game_launcher.py --analyze word    # Analyze a word
"""

import sys
import os
import argparse
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def simple_text_game():
    """Play simple text-based game without complex imports"""
    print("🎮 RST WORD ASSOCIATION - SIMPLE MODE")
    print("=" * 50)
    print("🎯 Goal: Avoid words starting with R, S, or T")
    print("💡 Choose words that associate with the current word")
    print()
    
    trap_letters = {'r', 's', 't'}
    current_word = "cat"
    turn = 1
    max_turns = 20
    
    print(f"🎲 Starting word: '{current_word}'")
    print("🎮 Enter associated words, or 'quit' to exit")
    print("⚠️  Avoid words starting with R, S, or T!")
    print("-" * 50)
    
    while turn <= max_turns:
        try:
            print(f"\nTurn {turn}: Current word is '{current_word}'")
            next_word = input("Your word: ").strip().lower()
            
            if next_word in ['quit', 'exit', 'q']:
                print("👋 Thanks for playing!")
                break
                
            if not next_word or not next_word.replace('-', '').replace("'", "").isalpha():
                print("❌ Please enter a valid word!")
                continue
                
            if next_word[0] in trap_letters:
                print(f"💥 Oh no! '{next_word}' starts with {next_word[0].upper()}!")
                print("🎯 You hit a trap letter! Game over!")
                print(f"🏁 You survived {turn} turns. Good game!")
                break
                
            if len(next_word) < 2:
                print("❌ Word too short! Please enter a longer word.")
                continue
                
            print(f"✅ Good choice! '{next_word}' is safe.")
            current_word = next_word
            turn += 1
            
            # Simple computer response
            if turn <= max_turns:
                computer_words = ["happy", "blue", "ocean", "music", "book", "flower", "smile", "light"]
                safe_words = [w for w in computer_words if w[0] not in trap_letters]
                if safe_words:
                    import random
                    computer_choice = random.choice(safe_words)
                    print(f"🤖 Computer plays: '{computer_choice}'")
                    current_word = computer_choice
                    turn += 1
                
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Thanks for playing!")
            break
    
    if turn > max_turns:
        print(f"\n🎉 Congratulations! You survived all {max_turns} turns!")
    
    print(f"\n📊 Final stats: {turn-1} turns completed")

def launch_full_game(mode=None, **kwargs):
    """Launch the full interactive game"""
    try:
        from game_cli import main as game_main, start_pvp_game, start_pvc_game, start_cvc_game, word_analysis_mode
        
        if mode == "pvp":
            print("🎮 Player vs Player Mode")
            start_pvp_game("cat")
        elif mode == "pvc":
            difficulty = kwargs.get('difficulty', 'medium')
            print(f"🤖 Player vs Computer Mode ({difficulty})")
            start_pvc_game("color", difficulty=difficulty)
        elif mode == "demo":
            print("🎭 Computer vs Computer Demo")
            start_cvc_game("start", num_games=3, difficulty1="medium", difficulty2="hard")
        elif mode == "analyze":
            word = kwargs.get('word')
            if word:
                print(f"🔍 Analyzing word: '{word}'")
                word_analysis_mode(word)
            else:
                print("🔍 Word Analysis Mode")
                word_analysis_mode()
        else:
            # Launch full interactive menu
            game_main()
            
    except ImportError as e:
        print("❌ Full game features require word association data and game components.")
        print(f"   Import error: {e}")
        print("💡 Try --simple for a basic game that works without dependencies.")
        return False
    except Exception as e:
        print(f"❌ Error launching game: {e}")
        return False
    
    return True

def show_interactive_menu():
    """Show interactive menu for mode selection"""
    print("🎮 RST WORD ASSOCIATION GAME")
    print("=" * 50)
    print()
    print("Choose your game mode:")
    print("  [1] Simple Text Game (no setup needed)")
    print("  [2] Player vs Player")
    print("  [3] Player vs Computer")
    print("  [4] Computer vs Computer Demo")
    print("  [5] Word Analysis Tool")
    print("  [6] Full Interactive Interface")
    print("  [7] Exit")
    print()
    
    while True:
        try:
            choice = input("Select option (1-7): ").strip()
            
            if choice == '1':
                simple_text_game()
                break
            elif choice == '2':
                if not launch_full_game("pvp"):
                    simple_text_game()
                break
            elif choice == '3':
                difficulty = input("Choose difficulty (easy/medium/hard/expert) [medium]: ").strip()
                if not difficulty:
                    difficulty = "medium"
                if not launch_full_game("pvc", difficulty=difficulty):
                    simple_text_game()
                break
            elif choice == '4':
                launch_full_game("demo")
                break
            elif choice == '5':
                word = input("Enter word to analyze (or press Enter for interactive): ").strip()
                launch_full_game("analyze", word=word if word else None)
                break
            elif choice == '6':
                launch_full_game()
                break
            elif choice == '7':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-7.")
                
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="RST Word Association Game - Consolidated Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Game Modes:
  --simple              Simple text-based game (no dependencies)
  --pvp                 Player vs Player mode
  --pvc DIFFICULTY      Player vs Computer (easy/medium/hard/expert)
  --demo                Computer vs Computer demonstration
  --analyze [WORD]      Analyze word strategic value
  
Examples:
  python examples/game_launcher.py                    # Interactive menu
  python examples/game_launcher.py --simple          # Simple game
  python examples/game_launcher.py --pvp             # PvP mode
  python examples/game_launcher.py --pvc hard        # PvC hard mode
  python examples/game_launcher.py --analyze color   # Analyze "color"
        """
    )
    
    parser.add_argument('--simple', action='store_true',
                       help='Play simple text-based game')
    parser.add_argument('--pvp', action='store_true',
                       help='Player vs Player mode')
    parser.add_argument('--pvc', metavar='DIFFICULTY',
                       help='Player vs Computer mode with difficulty')
    parser.add_argument('--demo', action='store_true',
                       help='Computer vs Computer demonstration')
    parser.add_argument('--analyze', nargs='?', const=True, metavar='WORD',
                       help='Analyze word strategic value')
    
    args = parser.parse_args()
    
    try:
        # Handle specific modes
        if args.simple:
            simple_text_game()
        elif args.pvp:
            if not launch_full_game("pvp"):
                print("Falling back to simple mode...")
                simple_text_game()
        elif args.pvc:
            difficulty = args.pvc if args.pvc in ['easy', 'medium', 'hard', 'expert'] else 'medium'
            if not launch_full_game("pvc", difficulty=difficulty):
                print("Falling back to simple mode...")
                simple_text_game()
        elif args.demo:
            launch_full_game("demo")
        elif args.analyze is not None:
            word = args.analyze if isinstance(args.analyze, str) else None
            launch_full_game("analyze", word=word)
        else:
            # No specific mode - show interactive menu
            show_interactive_menu()
            
    except KeyboardInterrupt:
        print("\n\n👋 Thanks for playing!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("🔧 Try --simple for a basic game that always works.")

if __name__ == "__main__":
    main()