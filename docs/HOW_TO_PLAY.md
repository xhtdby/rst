# 🎮 How to Play RST Word Association Game

## Quick Start - Choose Your Adventure!

### 🚀 **Instant Play** (Simplest)
```bash
python simple_game.py
```
- No setup required
- Basic text-based gameplay
- Perfect for quick games

### ⚡ **Quick Launchers**
```bash
python play_game.py         # Simple menu launcher
python run_game.py          # Advanced launcher with options
python run_game.py --pvp    # Direct Player vs Player
python run_game.py --pvc    # Direct Player vs Computer
python run_game.py --demo   # Watch computer vs computer
```

### 🎯 **Full Experience** (Most Features)
```bash
python game_cli.py
```
- Complete interactive interface
- All game modes and analysis tools
- Strategic AI opponents
- Word analysis features

## Game Modes Available

### 1. 👥 Player vs Player (PvP)
- Two humans take turns
- Choose words that associate with the current word
- Avoid words starting with R, S, or T
- First to hit a trap letter loses!

### 2. 🤖 Player vs Computer (PvC)  
- Human vs AI opponent
- Multiple difficulty levels:
  - **Easy**: Basic random choices
  - **Medium**: Strategic word selection
  - **Hard**: Advanced multi-step analysis
  - **Expert**: Adversarial minimax strategies

### 3. 🎭 Computer vs Computer (CvC)
- Watch AI players compete
- Study different strategies
- Learn optimal gameplay patterns

### 4. 🔍 Word Analysis Tool
- Analyze strategic value of any word
- See PageRank scores and trap probabilities
- Get strategic recommendations

## 🎯 Game Rules

1. **Goal**: Avoid words starting with R, S, or T
2. **Gameplay**: Choose words that associate with the current word
3. **Winning**: Force your opponent to a trap letter
4. **Losing**: Choose a word starting with R, S, or T

## 📁 File Guide

| File | Purpose | Best For |
|------|---------|----------|
| `simple_game.py` | Basic text game | Quick play, no setup |
| `play_game.py` | Simple launcher | Easy menu navigation |
| `run_game.py` | Advanced launcher | Command-line options |
| `game_cli.py` | Full interface | Complete experience |
| `instant_play.py` | Auto-start game | One-click gaming |

## 🔧 Troubleshooting

### "Could not import game components"
- Make sure you're in the RST project directory
- Try `python simple_game.py` for a version that works without imports

### "No word association data found"
- The full game requires word association data files
- Use `simple_game.py` for a demo version without data dependencies

### "Command not found"
- Make sure Python is installed and in your PATH
- Try `python3` instead of `python` on some systems

## 🎓 Learning More

- Read `ASSUMPTIONS_AND_LIMITATIONS.md` for game theory details
- Check `adversarial_strategy.py` for AI strategy insights
- Run unit tests with `python test_multistep_tiny.py`

## 🎉 Examples

### Quick PvP Game
```bash
$ python run_game.py --pvp
🎮 Quick Player vs Player Game
Starting word: 'cat'
Player 1: dog
Player 2: bone
Player 1: dig
Player 2: ...
```

### Analyze a Word
```bash
$ python run_game.py --analyze
🔍 Enter word to analyze: color
📊 PageRank Score: 0.85
🎯 Trap Probability: 0.23
💡 Strategic Value: High
```

---

**🎮 Ready to play? Choose any file above and start your RST adventure!**