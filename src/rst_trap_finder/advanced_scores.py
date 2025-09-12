"""Advanced scoring functions for comprehensive trap analysis."""
from __future__ import annotations

import math
from typing import Dict, FrozenSet, List, Mapping, Optional, Set, Tuple

import networkx as nx
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from . import TRAP_LETTERS
from .graph import Graph


def centrality_scores(graph: Graph, trap: FrozenSet[str] = TRAP_LETTERS) -> Dict[str, Dict[str, float]]:
    """Compute various centrality measures for all nodes."""
    G = nx.DiGraph()
    for u, neighbors in graph.items():
        for v, weight in neighbors.items():
            G.add_edge(u, v, weight=weight)
    
    # Basic centrality measures
    betweenness = nx.betweenness_centrality(G, weight='weight')
    closeness = nx.closeness_centrality(G, distance='weight')
    eigenvector = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
    katz = nx.katz_centrality(G, weight='weight', max_iter=1000)
    
    # Page rank variants
    pagerank = nx.pagerank(G, weight='weight')
    personalized_pr = {}
    for node in G.nodes():
        if node and node[0] in trap:
            personalization = {n: 1.0 if n == node else 0.0 for n in G.nodes()}
            personalized_pr[node] = nx.pagerank(G, personalization=personalization, weight='weight')
    
    results = {}
    for node in G.nodes():
        results[node] = {
            'betweenness': betweenness.get(node, 0.0),
            'closeness': closeness.get(node, 0.0),
            'eigenvector': eigenvector.get(node, 0.0),
            'katz': katz.get(node, 0.0),
            'pagerank': pagerank.get(node, 0.0),
            'trap_influence': sum(
                personalized_pr.get(trap_node, {}).get(node, 0.0) 
                for trap_node in personalized_pr
            )
        }
    
    return results


def flow_based_scores(graph: Graph, trap: FrozenSet[str] = TRAP_LETTERS) -> Dict[str, float]:
    """Compute flow-based trap scores using max flow analysis."""
    G = nx.DiGraph()
    for u, neighbors in graph.items():
        for v, weight in neighbors.items():
            G.add_edge(u, v, capacity=weight)
    
    # Create virtual trap sink
    trap_nodes = [node for node in G.nodes() if node and node[0] in trap]
    if not trap_nodes:
        return {node: 0.0 for node in G.nodes()}
    
    G.add_node('TRAP_SINK')
    for trap_node in trap_nodes:
        G.add_edge(trap_node, 'TRAP_SINK', capacity=float('inf'))
    
    flow_scores = {}
    for node in G.nodes():
        if node == 'TRAP_SINK' or (node and node[0] in trap):
            flow_scores[node] = 0.0
            continue
        
        try:
            flow_value, _ = nx.maximum_flow(G, node, 'TRAP_SINK')
            total_capacity = sum(
                data.get('capacity', 0) for _, _, data in G.edges(node, data=True)
            )
            flow_scores[node] = flow_value / total_capacity if total_capacity > 0 else 0.0
        except (nx.NetworkXError, nx.NetworkXNoPath):
            flow_scores[node] = 0.0
    
    # Remove virtual sink from results
    flow_scores.pop('TRAP_SINK', None)
    return flow_scores


def clustering_based_scores(graph: Graph, trap: FrozenSet[str] = TRAP_LETTERS) -> Dict[str, float]:
    """Compute clustering-based scores indicating trap neighborhood density."""
    G = nx.Graph()  # Use undirected for clustering
    for u, neighbors in graph.items():
        for v, weight in neighbors.items():
            if G.has_edge(u, v):
                G[u][v]['weight'] += weight
            else:
                G.add_edge(u, v, weight=weight)
    
    clustering = nx.clustering(G, weight='weight')
    
    # Compute trap-weighted clustering
    trap_clustering = {}
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if not neighbors:
            trap_clustering[node] = 0.0
            continue
        
        trap_neighbors = [n for n in neighbors if n and n[0] in trap]
        trap_ratio = len(trap_neighbors) / len(neighbors)
        
        # Weight clustering by trap presence
        trap_clustering[node] = clustering.get(node, 0.0) * (1 + trap_ratio)
    
    return trap_clustering


def spectral_scores(graph: Graph, trap: FrozenSet[str] = TRAP_LETTERS) -> Dict[str, float]:
    """Compute spectral analysis scores using graph Laplacian."""
    G = nx.DiGraph()
    for u, neighbors in graph.items():
        for v, weight in neighbors.items():
            G.add_edge(u, v, weight=weight)
    
    if len(G) < 2:
        return {node: 0.0 for node in G.nodes()}
    
    # Create adjacency matrix
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    
    # Build weighted adjacency matrix
    A = sparse.lil_matrix((n, n))
    for u, neighbors in graph.items():
        if u in node_to_idx:
            u_idx = node_to_idx[u]
            for v, weight in neighbors.items():
                if v in node_to_idx:
                    v_idx = node_to_idx[v]
                    A[u_idx, v_idx] = weight
    
    # Compute Laplacian
    A = A.tocsr()
    degrees = np.array(A.sum(axis=1)).flatten()
    D = sparse.diags(degrees)
    L = D - A
    
    try:
        # Compute second smallest eigenvalue and eigenvector (Fiedler)
        eigenvals, eigenvecs = eigsh(L, k=min(3, n-1), which='SM', sigma=0.0)
        
        if len(eigenvals) > 1:
            fiedler_vector = eigenvecs[:, 1]  # Second smallest eigenvalue
            spectral_scores_dict = {
                nodes[i]: abs(fiedler_vector[i]) for i in range(n)
            }
        else:
            spectral_scores_dict = {node: 0.0 for node in nodes}
    
    except Exception:
        spectral_scores_dict = {node: 0.0 for node in nodes}
    
    return spectral_scores_dict


def game_theoretic_scores(graph: Graph, trap: FrozenSet[str] = TRAP_LETTERS, 
                         iterations: int = 100) -> Dict[str, float]:
    """Compute game-theoretic equilibrium scores."""
    nodes = list(graph.keys())
    if not nodes:
        return {}
    
    # Initialize mixed strategies (uniform)
    strategies = {node: 1.0 / len(graph.get(node, {})) if graph.get(node) else 0.0 
                 for node in nodes}
    
    for _ in range(iterations):
        new_strategies = {}
        
        for node in nodes:
            neighbors = graph.get(node, {})
            if not neighbors:
                new_strategies[node] = 0.0
                continue
            
            # Compute expected payoffs for each action
            payoffs = {}
            for target, weight in neighbors.items():
                # Payoff is higher for trap targets
                base_payoff = 1.0 if target and target[0] in trap else 0.0
                # Weight by edge strength and opponent's strategy
                expected_payoff = base_payoff * weight * strategies.get(target, 0.0)
                payoffs[target] = expected_payoff
            
            # Update strategy (best response)
            if payoffs:
                max_payoff = max(payoffs.values())
                best_actions = [action for action, payoff in payoffs.items() 
                              if payoff == max_payoff]
                new_strategies[node] = max_payoff * len(best_actions) / len(neighbors)
            else:
                new_strategies[node] = 0.0
        
        # Convergence check
        convergence = all(
            abs(new_strategies.get(node, 0) - strategies.get(node, 0)) < 1e-6
            for node in nodes
        )
        
        strategies = new_strategies
        if convergence:
            break
    
    return strategies


def resistance_scores(graph: Graph, trap: FrozenSet[str] = TRAP_LETTERS) -> Dict[str, float]:
    """Compute resistance-based scores using effective resistance."""
    G = nx.Graph()  # Convert to undirected for resistance calculation
    for u, neighbors in graph.items():
        for v, weight in neighbors.items():
            # Use conductance (1/resistance) as edge weight
            conductance = weight
            if G.has_edge(u, v):
                G[u][v]['weight'] += conductance
            else:
                G.add_edge(u, v, weight=conductance)
    
    if len(G) < 2:
        return {node: 0.0 for node in G.nodes()}
    
    # Compute effective resistance to trap nodes
    trap_nodes = [node for node in G.nodes() if node and node[0] in trap]
    if not trap_nodes:
        return {node: 0.0 for node in G.nodes()}
    
    resistance_scores_dict = {}
    
    for node in G.nodes():
        if node in trap_nodes:
            resistance_scores_dict[node] = 1.0  # Already a trap
            continue
        
        # Compute average resistance to all trap nodes
        resistances = []
        for trap_node in trap_nodes:
            try:
                if nx.has_path(G, node, trap_node):
                    resistance = nx.resistance_distance(G, node, trap_node, weight='weight')
                    resistances.append(1.0 / (1.0 + resistance))  # Convert to similarity
                else:
                    resistances.append(0.0)
            except (nx.NetworkXError, ZeroDivisionError):
                resistances.append(0.0)
        
        resistance_scores_dict[node] = np.mean(resistances) if resistances else 0.0
    
    return resistance_scores_dict


def multi_step_analysis(graph: Graph, trap: FrozenSet[str] = TRAP_LETTERS, 
                       max_steps: int = 5) -> Dict[str, Dict[int, float]]:
    """Analyze trap probabilities over multiple steps."""
    results = {}
    
    for node in graph:
        step_probs = {}
        current_dist = {node: 1.0}
        
        for step in range(1, max_steps + 1):
            next_dist = {}
            trap_prob = 0.0
            
            for current_node, prob in current_dist.items():
                if current_node and current_node[0] in trap:
                    trap_prob += prob
                    continue
                
                neighbors = graph.get(current_node, {})
                if not neighbors:
                    continue
                
                total_weight = sum(neighbors.values())
                for next_node, weight in neighbors.items():
                    transition_prob = weight / total_weight
                    next_dist[next_node] = next_dist.get(next_node, 0.0) + prob * transition_prob
            
            step_probs[step] = trap_prob
            current_dist = next_dist
            
            # Early termination if we've reached high probability
            if trap_prob > 0.99:
                break
        
        results[node] = step_probs
    
    return results


def comprehensive_score(node: str, graph: Graph, trap: FrozenSet[str] = TRAP_LETTERS,
                       weights: Optional[Dict[str, float]] = None) -> float:
    """Compute a comprehensive score combining all advanced metrics."""
    if weights is None:
        weights = {
            'centrality': 0.2,
            'flow': 0.25,
            'clustering': 0.15,
            'spectral': 0.1,
            'game_theoretic': 0.2,
            'resistance': 0.1
        }
    
    # Compute all scores (expensive, so cache in real implementation)
    centrality = centrality_scores(graph, trap).get(node, {})
    flow = flow_based_scores(graph, trap).get(node, 0.0)
    clustering = clustering_based_scores(graph, trap).get(node, 0.0)
    spectral = spectral_scores(graph, trap).get(node, 0.0)
    game_theory = game_theoretic_scores(graph, trap).get(node, 0.0)
    resistance = resistance_scores(graph, trap).get(node, 0.0)
    
    # Normalize centrality scores
    centrality_score = (
        centrality.get('betweenness', 0.0) * 0.3 +
        centrality.get('closeness', 0.0) * 0.3 +
        centrality.get('pagerank', 0.0) * 0.4
    )
    
    comprehensive = (
        weights['centrality'] * centrality_score +
        weights['flow'] * flow +
        weights['clustering'] * clustering +
        weights['spectral'] * spectral +
        weights['game_theoretic'] * game_theory +
        weights['resistance'] * resistance
    )
    
    return comprehensive