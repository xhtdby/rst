# RST Trap Finder - Assumptions and Documentation

## Overview

The RST Trap Finder is a strategic analysis tool for word association games, specifically designed to identify "trap words" that tend to lead opponents toward words starting with R, S, or T letters. This document outlines the key assumptions, limitations, and design decisions.

## Core Assumptions

### 1. Trap Letters
- **Default Trap Letters**: R, S, T
- **Assumption**: These letters represent "losing conditions" in word association games
- **Rationale**: Common in many word games where certain letters end gameplay
- **Configurable**: Can be modified via `trap_letters` parameter

### 2. Graph Structure
- **Directed Graph**: Word associations are directional (A→B doesn't imply B→A)
- **Weighted Edges**: Association strengths are represented as positive floating-point numbers
- **Normalization**: Outgoing edge weights are normalized to probabilities for random walk analysis
- **No Self-Loops**: Words cannot directly associate with themselves

### 3. Probability Model
- **Random Walk**: Player behavior modeled as random walk through word association graph
- **Transition Probabilities**: Based on normalized edge weights from association data
- **Independence**: Each step is independent given current word position
- **Memoryless**: Previous path doesn't affect future transitions (Markov property)

### 4. Scoring Methodology

#### PageRank Assumptions
- **Teleportation**: 15% probability of random jump (damping factor = 0.85)
- **Uniform Distribution**: Random jumps are uniformly distributed across all words
- **Convergence**: Algorithm converges to stable probability distribution
- **Interpretation**: Higher PageRank indicates more "central" or "important" words

#### Trap Effectiveness
- **One-Step Probability**: Direct probability of reaching trap letters in one move
- **Composite Scoring (Implemented)**: Current implementation combines four components: one-step probability, neighborhood richness, escape hardness, and trap-biased PageRank. Default weights are `(0.4, 0.2, 0.2, 0.2)`.
- **Note**: Weights are configurable at call sites and may be tuned per dataset.

### 5. Multi-Step Analysis
- **Dynamic Programming**: k-step probabilities calculated using memoized recursion
- **Path Independence**: Multiple paths to same word are treated independently
- **Information Theory**: Entropy and information gain provide strategic insights
- **Bounded Analysis**: Practical limits on analysis depth (typically k ≤ 5-10 steps)

## Known Limitations

### 1. Data Quality Dependent
- **Source Reliability**: Results only as good as underlying word association data
- **Coverage Gaps**: Missing associations can lead to underestimated trap effectiveness
- **Bias Inheritance**: Cultural, linguistic, or demographic biases in source data propagate to results

### 2. Simplistic Player Model
- **Random Walk Assumption**: Real players use strategy, memory, and planning
- **No Learning**: Model doesn't account for players learning opponent patterns
- **Equal Skill**: Assumes all players have similar association knowledge
- **No Deliberate Avoidance**: Players may actively avoid certain patterns

### 3. Static Analysis
- **Fixed Graph**: Word associations assumed constant during game
- **No Context**: Doesn't consider game context, player state, or external factors
- **No Adaptation**: Strategies don't evolve based on opponent behavior

### 4. Computational Constraints
- **Memory Usage**: Large graphs may require significant memory for PageRank and caching
- **Analysis Depth**: Multi-step analysis becomes exponentially expensive
- **Real-Time Limits**: Complex analysis may be too slow for real-time gameplay

### 5. Game Mechanics
- **Word Validity**: Assumes all words in graph are valid game words
- **Turn Structure**: Designed for alternating turn-based play
- **Winning Conditions**: Assumes reaching trap letters constitutes loss/disadvantage

## Threshold and Parameter Justification

### Default Configuration Values

#### PageRank Parameters
- **Damping Factor**: 0.85 (standard value balancing convergence and teleportation)
- **Iterations**: 100 (sufficient for convergence on most graphs)
- **Tolerance**: 1e-6 (high precision for stable results)

#### Scoring Weights
- **PageRank Weight**: 0.7 (emphasizes word centrality and connectivity)
- **Trap Probability Weight**: 0.3 (balances immediate threat with strategic position)
- **Rationale**: Empirically tested to provide balanced strategic recommendations

#### Multi-Step Analysis
- **Default max_k**: 5 steps (balances depth with computational cost)
- **Path Limits**: 10 paths maximum (prevents combinatorial explosion)
- **Cache Limits**: Configurable based on available memory

### Threshold Rationales

#### High-Value Word Identification
- **Top 10%**: Words scoring in top decile considered "high-value"
- **Score > 0.5**: Above-median performance threshold
- **Relative Ranking**: Focus on comparative rather than absolute scores

#### Path Analysis
- **Probability > 0.1**: Meaningful strategic paths (10%+ chance)
- **Information Gain > 0.5**: Significant strategic value threshold
- **Entropy < 2.0**: Reasonably predictable outcomes

## Usage Guidelines

### Recommended Use Cases
1. **Strategic Analysis**: Pre-game planning and word selection
2. **Educational Tools**: Understanding word association patterns
3. **Game Development**: AI opponent strategies and difficulty tuning
4. **Research**: Analyzing linguistic and cognitive association patterns

### Not Recommended For
1. **Real-Time Competitive Play**: Analysis may be too slow
2. **Natural Language Processing**: Designed specifically for game scenarios
3. **General Word Similarity**: Optimized for trap-finding, not semantic similarity

## Validation and Testing

### Unit Testing Strategy
- **Tiny Graphs**: 1-3 node graphs for algorithm validation
- **Edge Cases**: Isolated nodes, cycles, disconnected components
- **Probability Consistency**: Ensures valid probability distributions
- **Performance Bounds**: Validates reasonable execution times

### Integration Testing
- **Real Data**: Testing on actual word association datasets
- **Cross-Validation**: Comparing different data sources and parameters
- **Human Validation**: Expert evaluation of strategic recommendations

## Future Improvements

### Potential Enhancements
1. **Adaptive Learning**: Models that learn from player behavior
2. **Context Awareness**: Incorporating game state and history
3. **Advanced Player Models**: Modeling deliberate strategy and planning
4. **Real-Time Optimization**: Faster algorithms for live gameplay
5. **Multi-Objective Optimization**: Balancing multiple strategic goals

### Research Directions
1. **Cognitive Modeling**: Better player behavior models
2. **Dynamic Graphs**: Time-evolving word associations
3. **Personalization**: Player-specific association models
4. **Game Theory**: Formal analysis of optimal strategies

## Technical Notes

### Performance Characteristics
- **Graph Loading**: O(E) where E is number of edges
- **PageRank**: O(V × iterations) where V is number of vertices
- **Multi-Step Analysis**: O(V^k) in worst case for k steps
- **Memory Usage**: O(V + E) for graph storage, O(V) for PageRank cache

### Numerical Stability
- **Probability Normalization**: Ensures valid probability distributions
- **Floating Point**: Standard IEEE 754 double precision
- **Convergence Checks**: Explicit tolerance testing for iterative algorithms

### Data Format Requirements
```csv
src,dst,weight
word1,word2,1.0
word2,word3,0.5
```
- **Headers**: Must include 'src', 'dst', 'weight' columns
- **Encoding**: UTF-8 recommended for international characters
- **Weights**: Positive floating-point numbers (higher = stronger association)

---

*This document should be updated as the system evolves and new insights are gained through usage and testing.*
