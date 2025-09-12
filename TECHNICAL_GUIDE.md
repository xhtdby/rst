# RST Trap Finder - Technical Deep Dive

This document provides a comprehensive technical explanation of how the RST Trap Finder works, including algorithms, implementation details, and mathematical foundations.

## Table of Contents

1. [Core Concept](#core-concept)
2. [Graph Representation](#graph-representation)
3. [Scoring Algorithms](#scoring-algorithms)
4. [Implementation Details](#implementation-details)
5. [Performance Characteristics](#performance-characteristics)
6. [Configuration Options](#configuration-options)
7. [Examples and Edge Cases](#examples-and-edge-cases)

## Core Concept

### The "Trap Word" Strategy

In word association games, a **trap word** is one that tends to lead opponents toward words starting with specific letters (by default R, S, T). This creates strategic advantage because:

1. **Predictable responses**: Certain words have strong associations with trap letters
2. **Limited escape routes**: Good trap words have few non-trap alternatives
3. **Cascading effects**: Reaching one trap word often leads to more trap words

### Mathematical Foundation

The tool models word associations as a **directed, weighted graph** where:
- **Nodes (V)**: Individual words
- **Edges (E)**: Word associations with strength weights
- **Weights (w)**: Association strength between word pairs
- **Trap set (T)**: Words starting with specified letters

## Graph Representation

### Data Structure

```python
Graph = Dict[str, Dict[str, float]]
# Example: {"start": {"color": 1.0, "animal": 1.0}}
```

### Loading from CSV

The system expects CSV format:
```csv
src,dst,weight
start,color,1.0
color,red,2.0
```

**Processing steps:**
1. **Normalization**: Convert to lowercase, strip whitespace
2. **Validation**: Skip empty/invalid entries, require positive weights
3. **Aggregation**: Sum weights for duplicate edges
4. **Indexing**: Build efficient lookup structures

## Scoring Algorithms

### 1. One-Step Probability (S1)

**Definition**: Probability that the immediate next word starts with a trap letter.

**Formula**:
```
S1(u) = Σ{v∈T} w(u,v) / Σ{v} w(u,v)
```

Where:
- `u` = source word
- `T` = set of words starting with trap letters
- `w(u,v)` = weight of edge from u to v

**Implementation**:
```python
def one_step_rst_probability(self, word: str) -> float:
    neighbors = self.get_neighbors(word)
    if not neighbors:
        return 0.0
        
    total_weight = sum(neighbors.values())
    trap_weight = sum(
        weight for neighbor, weight in neighbors.items()
        if neighbor and neighbor[0] in self.trap_letters
    )
    
    return trap_weight / total_weight
```

**Interpretation**:
- `S1 = 1.0`: All outgoing edges lead to trap words (perfect trap)
- `S1 = 0.0`: No outgoing edges lead to trap words
- `S1 = 0.5`: Half the association strength goes to trap words

### 2. Escape Hardness (H)

**Definition**: Fraction of "strong" edges that lead to trap words, where strong edges are those above a threshold.

**Formula**:
```
H(u) = |{v∈T : w(u,v) ≥ θ}| / |{v : w(u,v) ≥ θ}|
```

Where:
- `θ = min_weight_fraction × max{w(u,v)}`
- Default `min_weight_fraction = 0.05`

**Implementation**:
```python
def escape_hardness(self, word: str, min_weight_fraction: float = 0.05) -> float:
    neighbors = self.get_neighbors(word)
    if not neighbors:
        return 0.0
        
    max_weight = max(neighbors.values())
    threshold = min_weight_fraction * max_weight
    
    strong_edges = [
        neighbor for neighbor, weight in neighbors.items()
        if weight >= threshold
    ]
    
    if not strong_edges:
        return 0.0
        
    trap_edges = [
        neighbor for neighbor in strong_edges
        if neighbor and neighbor[0] in self.trap_letters
    ]
    
    return len(trap_edges) / len(strong_edges)
```

**Interpretation**:
- `H = 1.0`: All strong alternatives are traps
- `H = 0.0`: No strong alternatives are traps
- Focuses on the "best" responses rather than all responses

### 3. Biased PageRank (PR)

**Definition**: Standard PageRank algorithm with increased transition probability to trap words.

**Formula**:
```
PR(v) = (1-d)/N + d × Σ{u} PR(u) × P_bias(u,v)
```

Where:
- `d = damping factor` (default 0.85)
- `N = total number of nodes`
- `P_bias(u,v) = biased transition probability`

**Biased Transition Probability**:
```
P_bias(u,v) = (w(u,v) × bias(v)) / Σ{w} (w(u,w) × bias(w))

bias(v) = {
    trap_bias  if v starts with trap letter
    1.0        otherwise
}
```

**Implementation**:
```python
def biased_pagerank(self, trap_bias: float = 1.5, damping: float = 0.85, 
                   max_iterations: int = 100, tolerance: float = 1e-10) -> ScoreDict:
    all_words = list(self.get_all_words())
    n_words = len(all_words)
    
    # Initialize uniformly
    pagerank = {word: 1.0 / n_words for word in all_words}
    
    for iteration in range(max_iterations):
        prev_pagerank = pagerank.copy()
        
        # Handle dangling nodes
        dangling_sum = sum(prev_pagerank[word] for word in dangling_nodes)
        base_value = (1 - damping) / n_words + damping * dangling_sum / n_words
        
        # Reset and compute contributions
        for word in all_words:
            pagerank[word] = base_value
        
        for source_word in all_words:
            neighbors = self.graph.get(source_word, {})
            if not neighbors:
                continue
            
            # Calculate biased weights
            biased_weights = {}
            for neighbor, weight in neighbors.items():
                bias = trap_bias if neighbor and neighbor[0] in self.trap_letters else 1.0
                biased_weights[neighbor] = weight * bias
            
            total_biased_weight = sum(biased_weights.values())
            
            # Distribute PageRank
            source_contribution = damping * prev_pagerank[source_word]
            for neighbor, biased_weight in biased_weights.items():
                probability = biased_weight / total_biased_weight
                pagerank[neighbor] += source_contribution * probability
        
        # Check convergence
        total_diff = sum(abs(pagerank[word] - prev_pagerank[word]) for word in all_words)
        if total_diff < tolerance:
            break
    
    # Normalize
    total = sum(pagerank.values())
    return {word: score / total for word, score in pagerank.items()}
```

**Interpretation**:
- Higher PR scores indicate globally important words with trap bias
- Captures both local associations and global graph structure
- `trap_bias > 1.0` increases probability of transitioning to trap words

### 4. K-Step Probability

**Definition**: Probability of reaching a trap word within exactly k steps.

**Formula** (recursive):
```
K_k(u) = {
    1.0                           if k=0 and u∈T
    0.0                           if k=0 and u∉T  
    1.0                           if u∈T (early termination)
    Σ{v} P(u,v) × K_{k-1}(v)     otherwise
}
```

Where `P(u,v) = w(u,v) / Σ{w} w(u,w)` is the transition probability.

**Implementation**:
```python
def k_step_rst_probability(self, word: str, k: int = 2) -> float:
    if k == 0:
        return 1.0 if word and word[0] in self.trap_letters else 0.0
    
    # Early termination if already at trap
    if word and word[0] in self.trap_letters:
        return 1.0
    
    neighbors = self.get_neighbors(word)
    if not neighbors:
        return 0.0
    
    total_weight = sum(neighbors.values())
    probability = 0.0
    
    for neighbor, weight in neighbors.items():
        transition_prob = weight / total_weight
        future_prob = self.k_step_rst_probability(neighbor, k - 1)
        probability += transition_prob * future_prob
        
    return probability
```

### 5. Composite Score

**Definition**: Weighted combination of multiple metrics.

**Formula**:
```
Composite(u) = λ₁×S1(u) + λ₂×H(u) + λ₃×PR(u)
```

**Default weights**: `λ = (0.5, 0.3, 0.2)`

**Rationale**:
- **S1** (50%): Immediate trap probability is most important
- **H** (30%): Quality of alternatives matters significantly  
- **PR** (20%): Global importance provides tiebreaking

## Implementation Details

### Core Class: WordAssociationGraph

```python
class WordAssociationGraph:
    def __init__(self, graph: Optional[Graph] = None, 
                 trap_letters: Optional[FrozenSet[str]] = None):
        self.graph: Graph = graph or {}
        self.trap_letters = trap_letters or TRAP_LETTERS
        self._out_sums: Optional[Dict[str, float]] = None  # Cached sums
```

### Key Optimizations

1. **Cached Out-Sums**: Precompute and cache outgoing weight sums
2. **Lazy Evaluation**: Only compute PageRank when needed
3. **Early Termination**: K-step stops at trap words
4. **Sparse Representation**: Dictionary-based graph storage

### Memory Complexity

- **Graph Storage**: O(|E|) where E is number of edges
- **PageRank**: O(|V|) where V is number of vertices  
- **Cached Sums**: O(|V|)
- **Total**: O(|V| + |E|)

### Time Complexity

- **One-Step**: O(degree(u))
- **Escape Hardness**: O(degree(u))
- **PageRank**: O(iterations × |E|)
- **K-Step**: O(k × |V|^k) worst case, often much better
- **Composite**: O(PageRank + |V| × degree_avg)

## Performance Characteristics

### Scalability

**Small graphs** (< 1,000 words):
- All algorithms run in milliseconds
- Memory usage negligible
- Interactive performance

**Medium graphs** (1,000 - 10,000 words):
- PageRank: ~100ms - 1s
- Total analysis: ~1-5s
- Memory: ~10-50MB

**Large graphs** (> 10,000 words):
- PageRank: ~1-10s depending on iterations
- Memory scales linearly with edges
- May need batch processing for very large datasets

### Convergence Properties

**PageRank convergence**:
- Typically converges in 10-50 iterations
- Convergence rate depends on graph structure
- Bipartite or nearly-bipartite graphs converge slower
- Dense graphs with high connectivity converge faster

## Configuration Options

### Trap Letters

```python
# Default: R, S, T
TRAP_LETTERS = frozenset({"r", "s", "t"})

# Custom: vowels
custom_traps = frozenset({"a", "e", "i", "o", "u"})
graph = WordAssociationGraph.from_csv("data.csv", trap_letters=custom_traps)
```

### Scoring Weights

```python
# Conservative (favor immediate traps)
conservative_weights = (0.7, 0.2, 0.1)

# Balanced (default)
balanced_weights = (0.5, 0.3, 0.2)

# Global-focused (favor PageRank)
global_weights = (0.3, 0.2, 0.5)
```

### PageRank Parameters

```python
pagerank_scores = graph.biased_pagerank(
    trap_bias=2.0,      # Higher = stronger bias toward traps
    damping=0.85,       # Standard value, higher = more global influence
    max_iterations=100, # More iterations = better convergence
    tolerance=1e-10     # Lower = stricter convergence
)
```

### Escape Hardness Threshold

```python
# Strict (only very strong edges)
hardness = graph.escape_hardness(word, min_weight_fraction=0.1)

# Lenient (include weaker edges)  
hardness = graph.escape_hardness(word, min_weight_fraction=0.01)
```

## Examples and Edge Cases

### Example 1: Perfect Trap Word

```python
# Word that only leads to trap words
graph = {
    "perfect": {"red": 1.0, "stone": 2.0, "table": 1.0}
}

# Results:
# S1 = 1.0 (all neighbors start with R/S/T)
# H = 1.0 (all strong edges are traps)
# Composite ≈ 1.0 (near perfect score)
```

### Example 2: Anti-Trap Word

```python
# Word that avoids trap words
graph = {
    "safe": {"blue": 1.0, "green": 1.0, "yellow": 1.0}
}

# Results:
# S1 = 0.0 (no neighbors start with R/S/T)
# H = 0.0 (no strong edges are traps)
# Composite ≈ 0.0 (very low score)
```

### Example 3: Mixed Strategy

```python
# Word with both trap and non-trap options
graph = {
    "mixed": {"red": 3.0, "blue": 1.0, "stone": 2.0, "green": 1.0}
}

# Results:
# S1 = (3.0 + 2.0) / (3.0 + 1.0 + 2.0 + 1.0) = 5/7 ≈ 0.714
# H depends on threshold, likely high since "red" and "stone" are strong
# Good strategic word
```

### Edge Cases

1. **Terminal words** (no outgoing edges): All scores = 0.0
2. **Self-loops**: Handled normally, contributes to scores
3. **Disconnected components**: PageRank still computed globally
4. **Single-word graphs**: Degenerate case, minimal scores
5. **Very sparse graphs**: PageRank may converge slowly

### Common Patterns

**High-scoring words typically have**:
- Multiple trap neighbors
- High-weight edges to traps
- Few strong non-trap alternatives
- Good global connectivity (for PageRank)

**Low-scoring words typically have**:
- Few or no trap neighbors
- Strong edges to non-trap words
- Many escape routes
- Poor global connectivity

## Validation and Testing

### Test Coverage

The implementation includes comprehensive tests for:
- Basic scoring functions with known inputs/outputs
- PageRank normalization and convergence
- Edge cases (empty graphs, terminal nodes)
- CLI functionality
- CSV loading and validation

### Property-Based Testing

Key properties verified:
- PageRank scores sum to 1.0
- Probabilities are in [0,1] range
- Scores are deterministic for same input
- Monotonicity where expected

### Performance Benchmarks

Regular benchmarks ensure:
- No performance regressions
- Scalability within expected bounds
- Memory usage stays reasonable

This technical foundation enables reliable, efficient analysis of word association graphs for strategic game play and linguistic research.