# RST Trap Finder Examples

This directory contains example code, demonstrations, and interactive tools for the RST Trap Finder project.

## 🎮 Interactive Examples

### Game Interface
- **[game_launcher.py](game_launcher.py)** - Consolidated game launcher with all modes (recommended entry point)
- **[game_cli.py](game_cli.py)** - Full-featured interactive game interface with AI opponents

### Analysis Examples  
- **[rst_analysis_clean.py](rst_analysis_clean.py)** - Unified analysis across datasets; top traps, pathways, simulation
- **[multistep_demo.py](multistep_demo.py)** - Minimal showcase of k-step and path analysis
- **[verbose_trap_simulation.py](verbose_trap_simulation.py)** - Step-by-step simulated play with detailed logging
- **[rst_connection_analysis.py](rst_connection_analysis.py)** - Connectivity-focused RST exploration

### Advanced Strategies
- **[adversarial_strategy.py](adversarial_strategy.py)** - Sophisticated adversarial AI with minimax evaluation
- **[test_framework.py](test_framework.py)** - Testing and validation framework for algorithms

## 🚀 Quick Start

### Play the Game
```bash
# Consolidated launcher (recommended)
python examples/game_launcher.py

# Simple text game (no dependencies)
python examples/game_launcher.py --simple

# Player vs Computer with medium AI
python examples/game_launcher.py --pvc medium
```

### Run Analysis Examples
```bash
# Multi-step analysis demonstration
python examples/multistep_demo.py

# Comprehensive RST analysis
python examples/rst_analysis_clean.py

# Verbose gameplay simulation
python examples/verbose_trap_simulation.py
```

### Explore Advanced Features
```bash
# Adversarial strategy demonstration
python examples/adversarial_strategy.py

# Connection analysis
python examples/rst_connection_analysis.py
```

## 📋 Example Categories

### 🎮 **Gameplay & Interactive**
Interactive examples for playing and experiencing the RST game:
- `game_launcher.py` - Main game interface (all modes)
- `game_cli.py` - Advanced interactive features

### 📊 **Analysis & Algorithms**  
Examples demonstrating the analytical capabilities:
- `rst_analysis_clean.py` - Comprehensive analysis framework
- `multistep_demo.py` - Multi-step probability analysis
- `rst_connection_analysis.py` - Graph connectivity analysis

### 🤖 **AI & Strategy**
Advanced AI and strategic gameplay examples:
- `adversarial_strategy.py` - Sophisticated AI opponents
- `verbose_trap_simulation.py` - Detailed gameplay simulation

### 🧪 **Testing & Validation**
Tools for testing and validating the system:
- `test_framework.py` - Algorithm testing framework

## 💡 Usage Notes

- **Run from repository root**: All examples manage `sys.path` for `src/` imports
- **Data requirements**: Some examples require word association data in `data/`
- **Fallback modes**: Game examples include simple text modes for when data is unavailable
- **Dependencies**: Basic examples work with minimal dependencies; advanced features may require additional packages

## 🎯 Recommended Learning Path

1. **Start playing**: `python examples/game_launcher.py --simple`
2. **Try advanced gameplay**: `python examples/game_launcher.py`
3. **Explore analysis**: `python examples/multistep_demo.py`
4. **Study strategies**: `python examples/adversarial_strategy.py`
5. **Run comprehensive analysis**: `python examples/rst_analysis_clean.py`

## 🔍 Finding Specific Features

| Feature | Example File |
|---------|-------------|
| Basic gameplay | `game_launcher.py --simple` |
| AI opponents | `game_launcher.py --pvc` |
| Word analysis | `game_launcher.py --analyze` |
| Multi-step algorithms | `multistep_demo.py` |
| Strategic analysis | `adversarial_strategy.py` |
| Graph connectivity | `rst_connection_analysis.py` |
| Simulation framework | `verbose_trap_simulation.py` |

---

*All examples are self-contained and include usage instructions. For comprehensive documentation, see the `docs/` directory.*
