# RST Trap Finder

A focused toolkit for analyzing word association graphs to identify "trap words" that effectively lead opponents toward words starting with specific letters (R, S, T by default).

## Quick Start

### Installation
```bash
pip install -e .
```

### Setup Real Datasets (Optional)
```bash
# Download and process research datasets (ConceptNet/SWOW/EAT)
python scripts/enhanced_dataset_integration.py

# Optionally create reduced-size datasets for faster iteration
python scripts/intelligent_dataset_reduction.py
```

### Usage
```bash
# Find top trap words
rst-find analyze data/edges.sample.csv --top 10

# Use real datasets (after setup)
rst-find analyze data/merged/merged_association_graph.csv --top 10

# Get strategic recommendations
rst-find recommend "start" data/edges.sample.csv

# Analyze specific word
rst-find word "color" data/edges.sample.csv
```

### Python API
```python
from rst_trap_finder import load_graph

graph = load_graph("data/edges.sample.csv")
top_words = graph.rank_words(top_k=10)
recommendations = graph.recommend_next_word("start")
```

## What It Does

This tool analyzes word association graphs to find words that are strategically valuable in word games. A "trap word" is one that tends to lead opponents toward words starting with trap letters (R, S, T), which can be advantageous in games like 20 Questions or word association games.

## Core Features

- **Load graphs** from CSV files with word associations
- **Score words** using multiple algorithms (one-step probability, escape hardness, biased PageRank)
- **Rank words** by trap effectiveness
- **Get recommendations** for strategic word choices
- **Export results** to CSV

## Data Sources

### Sample Data
The toolkit includes sample data in `data/edges.sample.csv` for immediate testing.

### Real Research Datasets
For production use, you can download and process real word association datasets:

- **Small World of Words (SWOW-EN)**: English word association norms
- **USF Free Association Norms**: University of South Florida database  
- **ConceptNet 5.7**: Large-scale semantic knowledge graph
- **Edinburgh Associative Thesaurus (EAT)**: Word association corpus

Run `python scripts/enhanced_dataset_integration.py` to download/process these datasets.
See `docs/ASSUMPTIONS_AND_LIMITATIONS.md` for modeling notes.

## Data Format

CSV files with word associations:
```csv
src,dst,weight
start,color,1.0
color,red,2.0
color,blue,1.0
```

## CLI Commands

- `rst-find analyze <csv> [--top N]` - Show top trap words
- `rst-find word <word> <csv>` - Analyze specific word  
- `rst-find recommend <word> <csv>` - Get recommendations
- `rst-find export <csv> <output>` - Export scores

## Repository Structure

```
rst-trap-finder/
├── src/rst_trap_finder/     # Core library package
│   ├── core.py              # Graph loading, scoring, ranking
│   ├── multistep.py         # Multi-step analysis algorithms
│   └── cli.py               # Command-line interface
├── data/                    # Word association datasets
│   ├── edges.sample.csv     # Quick test data
│   ├── processed/           # Cleaned datasets
│   ├── merged/              # Combined datasets
│   └── reduced/             # Optimized datasets
├── examples/                # Interactive examples & demos
│   ├── game_launcher.py     # Consolidated game interface (recommended)
│   ├── game_cli.py          # Full-featured game with AI
│   ├── rst_analysis_clean.py # Comprehensive analysis framework
│   ├── multistep_demo.py    # Multi-step algorithm showcase
│   ├── adversarial_strategy.py # Advanced AI strategies
│   └── ...                  # More examples and analysis tools
├── scripts/                 # Data processing pipelines
│   ├── enhanced_dataset_integration.py # Download & process datasets
│   ├── intelligent_dataset_reduction.py # Create optimized datasets
│   └── ...                  # More processing utilities
├── tests/                   # Unit and integration tests
├── docs/                    # Comprehensive documentation
│   ├── ASSUMPTIONS_AND_LIMITATIONS.md # Technical details
│   ├── HOW_TO_PLAY.md       # Game instructions
│   └── ...                  # More documentation
└── README.md               # This file
```

## Quick Start Paths

### 🎮 **Play the Game**
```bash
python examples/game_launcher.py
```

### 📊 **Analyze Words**
```bash
rst-find analyze data/edges.sample.csv --top 10
```

### 🔬 **Set Up Research Data**
```bash
python scripts/enhanced_dataset_integration.py
python scripts/intelligent_dataset_reduction.py
```

### 🧪 **Explore Examples**
```bash
python examples/multistep_demo.py
python examples/rst_analysis_clean.py
```

## Example Output

```
Top 5 trap words:
1. tiger: 0.8023 (100% one-step, 100% hardness)
2. apple: 0.8022 (100% one-step, 100% hardness)  
3. stone: 0.8019 (100% one-step, 100% hardness)

Best recommendation from 'start': color (score: 0.5770)
```

## Dependencies

- Python 3.9+

See `docs/ASSUMPTIONS_AND_LIMITATIONS.md` and `docs/HOW_TO_PLAY.md` for details.

## License

MIT License
