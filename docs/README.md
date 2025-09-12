# RST Trap Finder - Comprehensive Documentation

## Overview

RST Trap Finder is a comprehensive toolkit for analyzing word association graphs to identify "death words" - prompts that tend to funnel an opponent's responses toward words starting with the trap letters R, S, or T. This toolkit has been significantly enhanced from its original form to provide advanced analytics, machine learning capabilities, and extensive visualization options.

## Features

### Core Functionality
- **Multiple Scoring Algorithms**: Including one-step probability, escape hardness, biased PageRank, k-step analysis, minimax scoring, and composite metrics
- **Advanced Scoring Methods**: Centrality measures, flow-based analysis, clustering coefficients, spectral analysis, and game-theoretic scoring
- **Machine Learning Integration**: Trap prediction models, word embeddings, and automated parameter optimization
- **Rich Data Processing**: Support for CSV, JSON, GraphML, pickle formats with validation and preprocessing

### Analysis Capabilities
- **Graph Metrics**: Comprehensive network analysis including community detection and path analysis
- **Visualization**: Static plots with matplotlib and interactive visualizations with Plotly
- **Performance Analytics**: Large-scale graph processing with optimization and benchmarking

### User Interface
- **Enhanced CLI**: Rich formatting, interactive modes, progress bars, and configuration management
- **Jupyter Integration**: Example notebooks demonstrating all features
- **Web Interface**: FastAPI-based REST API and web dashboard (when installed)

## Installation

### Basic Installation
```bash
pip install rst_trap_finder
```

### Full Installation with All Features
```bash
pip install rst_trap_finder[all]
```

### Development Installation
```bash
git clone https://github.com/xhtdby/rst.git
cd rst
pip install -e .[dev]
```

### Optional Dependencies

Install specific feature sets as needed:

```bash
# Visualization features
pip install rst_trap_finder[viz]

# Machine learning features  
pip install rst_trap_finder[ml]

# Web interface
pip install rst_trap_finder[web]

# Development tools
pip install rst_trap_finder[dev]
```

## Quick Start

### Basic Usage

```python
from rst_trap_finder.io import load_csv
from rst_trap_finder.scores import biased_pagerank, composite
from rst_trap_finder import TRAP_LETTERS

# Load word association graph
graph = load_csv("data/edges.sample.csv")

# Compute PageRank with bias toward trap letters
pagerank = biased_pagerank(graph, TRAP_LETTERS)

# Calculate composite trap scores
scores = {}
for word in graph:
    scores[word] = composite(word, graph, TRAP_LETTERS, pagerank)

# Find top trap words
top_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top trap words:", top_words)
```

### Command Line Interface

```bash
# Rank words by trap effectiveness
rst_trap_finder rank --csv data/edges.sample.csv --top 15

# Get recommendations from current word
rst_trap_finder next --word "color" --csv data/edges.sample.csv

# Comprehensive analysis with visualizations
rst_trap_finder analyze --csv data/edges.sample.csv --include-viz --include-ml

# Convert between formats
rst_trap_finder convert --input-file data.csv --output-file data.json --output-format json
```

## Data Format

### CSV Format
The primary input format is CSV with three required columns:

```csv
src,dst,weight
start,color,1.0
start,animal,1.0
color,red,2.0
color,blue,1.0
animal,tiger,3.0
```

### JSON Format
Alternative JSON formats are supported:

```json
{
  "metadata": {
    "name": "Sample Graph",
    "num_nodes": 5,
    "num_edges": 7
  },
  "graph": {
    "start": {"color": 1.0, "animal": 1.0},
    "color": {"red": 2.0, "blue": 1.0},
    "animal": {"tiger": 3.0}
  }
}
```

## Scoring Algorithms

### Basic Scores

1. **One-Step RST Probability (S1)**: Probability that a one-step reply starts with R/S/T
2. **Escape Hardness (H)**: Fraction of strong edges leading to trap letters
3. **Biased PageRank (PR)**: PageRank with edges to R/S/T words boosted
4. **K-Step RST (K2)**: Probability of reaching R/S/T within k steps
5. **Minimax Top-M (MM)**: Worst-case analysis of opponent's top-m choices

### Advanced Scores

6. **Centrality Measures**: Betweenness, closeness, eigenvector centralities
7. **Flow-Based Scores**: Maximum flow analysis to trap nodes
8. **Clustering Metrics**: Local clustering with trap-aware weighting
9. **Spectral Analysis**: Graph Laplacian eigenvalue analysis
10. **Game-Theoretic**: Nash equilibrium-based scoring
11. **Resistance Analysis**: Effective resistance to trap nodes

### Composite Scoring

The composite score combines multiple metrics with configurable weights:

```python
composite_score = λ₁·S1 + λ₂·H + λ₃·PR + λ₄·K2 + λ₅·MM
```

Default weights: λ = (0.35, 0.2, 0.25, 0.1, 0.1)

## Machine Learning Features

### Trap Prediction

Train models to predict trap effectiveness:

```python
from rst_trap_finder.ml_models import TrapPredictor

predictor = TrapPredictor(model_type='random_forest')
metrics = predictor.train(graph)
predictions = predictor.predict(graph)
```

### Feature Engineering

Extract comprehensive features for ML models:

```python
from rst_trap_finder.ml_models import FeatureExtractor

extractor = FeatureExtractor(graph)
features = extractor.extract_node_features("word")
```

Features include:
- Graph structure metrics (degree, weights, clustering)
- Linguistic features (length, vowel ratio, consonant clusters)
- Network position features (centrality, neighborhood analysis)
- Trap-specific metrics (trap neighbor ratio, escape routes)

### Parameter Optimization

Automatically optimize scoring parameters:

```python
from rst_trap_finder.ml_models import ParameterOptimizer

optimizer = ParameterOptimizer(graph)
optimal_weights = optimizer.optimize_composite_weights(n_trials=100)
optimal_alpha = optimizer.optimize_pagerank_alpha(n_trials=50)
```

### Word Embeddings

Analyze semantic relationships:

```python
from rst_trap_finder.ml_models import WordEmbeddingAnalyzer

analyzer = WordEmbeddingAnalyzer()
model = analyzer.train_embeddings(graph)
similar_words = analyzer.get_similar_words("word", top_k=10)
```

## Visualization

### Network Visualization

Create static network plots:

```python
from rst_trap_finder.analysis import GraphAnalyzer

analyzer = GraphAnalyzer(graph)
fig = analyzer.visualize_network(
    layout='spring',
    node_size_metric='pagerank',
    highlight_traps=True
)
```

### Interactive Visualizations

Generate interactive plots with Plotly:

```python
interactive_fig = analyzer.interactive_network()
interactive_fig.show()
```

### Score Analysis

Visualize score distributions:

```python
fig = analyzer.score_distribution_plot(scores, "Composite Scores")
fig.show()
```

## Advanced Analysis

### Community Detection

Identify word communities:

```python
communities = analyzer.community_detection(algorithm='louvain')
```

### Path Analysis

Find paths from words to trap nodes:

```python
paths = analyzer.path_analysis("start", targets=["red", "stone", "tree"])
```

### Graph Statistics

Compute comprehensive graph metrics:

```python
stats = analyzer.basic_stats()
```

## Performance Optimization

### Large Graph Processing

The toolkit is optimized for large graphs:

```python
# Enable progress tracking for large operations
from rst_trap_finder.enhanced_cli import config
config.set('show_progress', True)

# Use parallel processing where available
import multiprocessing
config.set('n_jobs', multiprocessing.cpu_count())
```

### Caching

Enable result caching for repeated analysis:

```python
config.set('enable_cache', True)
config.set('cache_dir', '/path/to/cache')
```

### Memory Optimization

For very large graphs, use memory-efficient algorithms:

```python
# Use sparse matrix operations
pagerank = biased_pagerank(graph, sparse=True)

# Process in batches
batch_size = 1000
for i in range(0, len(nodes), batch_size):
    batch = nodes[i:i+batch_size]
    # Process batch
```

## Configuration

### CLI Configuration

Manage settings with the config command:

```bash
# View current configuration
rst_trap_finder config --list-all

# Set default parameters
rst_trap_finder config --key default_lambdas --value "[0.4, 0.2, 0.2, 0.1, 0.1]"
rst_trap_finder config --key default_alpha --value 2.0

# Reset to defaults
rst_trap_finder config --reset
```

### Programmatic Configuration

```python
from rst_trap_finder.enhanced_cli import config

# Set configuration options
config.set('default_top', 20)
config.set('output_format', 'json')
config.set('auto_save_results', True)
```

## Data Validation

### Graph Validation

Validate graph structure before analysis:

```python
from rst_trap_finder.data_processing import DataProcessor

issues = DataProcessor.validate_graph(
    graph,
    allow_self_loops=False,
    min_out_degree=1,
    max_out_degree=100
)

if issues:
    print("Graph validation issues:", issues)
```

### Data Preprocessing

Clean and preprocess data:

```python
# Load with validation and preprocessing
graph = DataProcessor.load_csv(
    "data.csv",
    validate=True,
    lowercase=True,
    min_weight=0.1,
    max_edges_per_node=50
)

# Get data summary
summary = DataProcessor.get_data_summary(graph)
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rst_trap_finder

# Run performance benchmarks
pytest -m benchmark

# Run property-based tests
pytest tests/test_advanced.py::TestPropertyBasedScoring
```

### Custom Test Data

Generate test graphs for development:

```python
from tests.test_advanced import graph_strategy
from hypothesis import given

@given(graph_strategy())
def test_custom_function(graph):
    # Your test code here
    pass
```

## Examples

See the `examples/` directory for:
- `basic_usage.ipynb`: Getting started tutorial
- `advanced_analysis.ipynb`: Comprehensive analysis walkthrough  
- `machine_learning.ipynb`: ML features demonstration
- `visualization_guide.ipynb`: Visualization examples
- `performance_optimization.ipynb`: Large-scale processing

## API Reference

### Core Modules

- `rst_trap_finder.scores`: Basic scoring algorithms
- `rst_trap_finder.advanced_scores`: Advanced scoring methods
- `rst_trap_finder.analysis`: Graph analysis and visualization
- `rst_trap_finder.ml_models`: Machine learning integration
- `rst_trap_finder.data_processing`: Data I/O and validation

### CLI Commands

- `rank`: Rank words by trap effectiveness
- `next`: Get next word recommendations
- `analyze`: Comprehensive graph analysis
- `convert`: Format conversion utilities
- `config`: Configuration management

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Install development dependencies: `pip install -e .[dev]`
4. Run tests: `pytest`
5. Submit a pull request

### Development Setup

```bash
# Clone repository
git clone https://github.com/xhtdby/rst.git
cd rst

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use this software in academic research, please cite:

```bibtex
@software{rst_trap_finder,
  title={RST Trap Finder: Comprehensive Word Graph Analysis Toolkit},
  author={AutoGen},
  year={2025},
  url={https://github.com/xhtdby/rst}
}
```