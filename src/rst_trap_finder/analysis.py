"""Graph analysis and visualization utilities."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from . import TRAP_LETTERS
from .graph import Graph


class GraphAnalyzer:
    """Comprehensive graph analysis and visualization."""
    
    def __init__(self, graph: Graph, trap_letters: set = None):
        self.graph = graph
        self.trap_letters = trap_letters or TRAP_LETTERS
        self._nx_graph = None
        self._stats = None
    
    @property
    def nx_graph(self) -> nx.DiGraph:
        """Convert to NetworkX graph (cached)."""
        if self._nx_graph is None:
            self._nx_graph = nx.DiGraph()
            for u, neighbors in self.graph.items():
                for v, weight in neighbors.items():
                    self._nx_graph.add_edge(u, v, weight=weight)
        return self._nx_graph
    
    def basic_stats(self) -> Dict[str, Union[int, float]]:
        """Compute basic graph statistics."""
        if self._stats is None:
            G = self.nx_graph
            
            # Basic counts
            num_nodes = G.number_of_nodes()
            num_edges = G.number_of_edges()
            
            # Connectivity
            is_strongly_connected = nx.is_strongly_connected(G)
            num_strongly_connected_components = nx.number_strongly_connected_components(G)
            num_weakly_connected_components = nx.number_weakly_connected_components(G)
            
            # Density and clustering
            density = nx.density(G)
            try:
                avg_clustering = nx.average_clustering(G.to_undirected())
            except ZeroDivisionError:
                avg_clustering = 0.0
            
            # Paths
            try:
                avg_shortest_path = nx.average_shortest_path_length(G, weight='weight')
            except nx.NetworkXError:
                avg_shortest_path = float('inf')
            
            try:
                diameter = nx.diameter(G.to_undirected())
            except nx.NetworkXError:
                diameter = float('inf')
            
            # Degree statistics
            in_degrees = [d for n, d in G.in_degree()]
            out_degrees = [d for n, d in G.out_degree()]
            
            # Trap node analysis
            trap_nodes = [n for n in G.nodes() if n and n[0] in self.trap_letters]
            trap_ratio = len(trap_nodes) / num_nodes if num_nodes > 0 else 0.0
            
            self._stats = {
                'num_nodes': num_nodes,
                'num_edges': num_edges,
                'density': density,
                'is_strongly_connected': is_strongly_connected,
                'num_strongly_connected_components': num_strongly_connected_components,
                'num_weakly_connected_components': num_weakly_connected_components,
                'avg_clustering': avg_clustering,
                'avg_shortest_path': avg_shortest_path,
                'diameter': diameter,
                'avg_in_degree': np.mean(in_degrees) if in_degrees else 0.0,
                'avg_out_degree': np.mean(out_degrees) if out_degrees else 0.0,
                'max_in_degree': max(in_degrees) if in_degrees else 0,
                'max_out_degree': max(out_degrees) if out_degrees else 0,
                'trap_nodes_count': len(trap_nodes),
                'trap_ratio': trap_ratio,
            }
        
        return self._stats
    
    def community_detection(self, algorithm: str = 'louvain') -> Dict[str, int]:
        """Detect communities in the graph."""
        G = self.nx_graph.to_undirected()
        
        if algorithm == 'louvain':
            try:
                import community as community_louvain
                communities = community_louvain.best_partition(G, weight='weight')
            except ImportError:
                # Fallback to networkx implementation
                communities = {}
                for i, component in enumerate(nx.connected_components(G)):
                    for node in component:
                        communities[node] = i
        elif algorithm == 'greedy_modularity':
            community_sets = nx.community.greedy_modularity_communities(G, weight='weight')
            communities = {}
            for i, community_set in enumerate(community_sets):
                for node in community_set:
                    communities[node] = i
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return communities
    
    def path_analysis(self, source: str, targets: Optional[List[str]] = None) -> Dict[str, List[List[str]]]:
        """Analyze paths from source to targets."""
        G = self.nx_graph
        
        if targets is None:
            targets = [n for n in G.nodes() if n and n[0] in self.trap_letters]
        
        paths = {}
        for target in targets:
            try:
                if nx.has_path(G, source, target):
                    # Find multiple paths
                    simple_paths = list(nx.all_simple_paths(G, source, target, cutoff=5))
                    # Sort by length and weight
                    weighted_paths = []
                    for path in simple_paths[:10]:  # Limit to top 10
                        weight = sum(
                            G[path[i]][path[i+1]]['weight'] 
                            for i in range(len(path) - 1)
                        )
                        weighted_paths.append((path, weight))
                    
                    weighted_paths.sort(key=lambda x: (len(x[0]), -x[1]))
                    paths[target] = [path for path, _ in weighted_paths[:5]]
                else:
                    paths[target] = []
            except nx.NetworkXNoPath:
                paths[target] = []
        
        return paths
    
    def visualize_network(self, 
                         layout: str = 'spring',
                         node_size_metric: str = 'degree',
                         edge_width_scaling: float = 1.0,
                         highlight_traps: bool = True,
                         save_path: Optional[Path] = None,
                         figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Create a matplotlib visualization of the network."""
        G = self.nx_graph
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=1/math.sqrt(len(G)), iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.random_layout(G)
        
        # Node sizes based on metric
        if node_size_metric == 'degree':
            node_sizes = [G.degree(n) * 100 + 50 for n in G.nodes()]
        elif node_size_metric == 'pagerank':
            pr = nx.pagerank(G, weight='weight')
            node_sizes = [pr.get(n, 0) * 5000 + 50 for n in G.nodes()]
        else:
            node_sizes = [100] * len(G.nodes())
        
        # Node colors
        if highlight_traps:
            node_colors = ['red' if n and n[0] in self.trap_letters else 'lightblue' 
                          for n in G.nodes()]
        else:
            node_colors = ['lightblue'] * len(G.nodes())
        
        # Edge widths
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [w / max_weight * 3 * edge_width_scaling for w in edge_weights]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw network
        nx.draw(G, pos, 
                node_color=node_colors,
                node_size=node_sizes,
                width=edge_widths,
                with_labels=True,
                font_size=8,
                font_weight='bold',
                arrows=True,
                arrowsize=10,
                ax=ax)
        
        ax.set_title("Word Association Network")
        
        # Add legend
        if highlight_traps:
            trap_legend = plt.Line2D([0], [0], marker='o', color='w', 
                                   markerfacecolor='red', markersize=10, label='Trap words (R/S/T)')
            normal_legend = plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor='lightblue', markersize=10, label='Other words')
            ax.legend(handles=[trap_legend, normal_legend])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def interactive_network(self, 
                           node_size_metric: str = 'degree',
                           highlight_traps: bool = True) -> go.Figure:
        """Create an interactive Plotly visualization."""
        G = self.nx_graph
        
        # Layout
        pos = nx.spring_layout(G, k=1/math.sqrt(len(G)), iterations=50)
        
        # Node traces
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        # Node sizes and colors
        if node_size_metric == 'degree':
            node_sizes = [G.degree(n) * 5 + 10 for n in G.nodes()]
        elif node_size_metric == 'pagerank':
            pr = nx.pagerank(G, weight='weight')
            node_sizes = [pr.get(n, 0) * 500 + 10 for n in G.nodes()]
        else:
            node_sizes = [15] * len(G.nodes())
        
        if highlight_traps:
            node_colors = ['red' if n and n[0] in self.trap_letters else 'lightblue' 
                          for n in G.nodes()]
        else:
            node_colors = ['lightblue'] * len(G.nodes())
        
        # Edge traces
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            weight = G[edge[0]][edge[1]]['weight']
            edge_info.append(f"{edge[0]} -> {edge[1]}: {weight:.2f}")
        
        # Create traces
        edge_trace = go.Scatter(x=edge_x, y=edge_y,
                               line=dict(width=0.5, color='#888'),
                               hoverinfo='none',
                               mode='lines')
        
        node_trace = go.Scatter(x=node_x, y=node_y,
                               mode='markers+text',
                               hoverinfo='text',
                               text=list(G.nodes()),
                               textposition="middle center",
                               marker=dict(size=node_sizes,
                                         color=node_colors,
                                         line=dict(width=2, color='black')))
        
        # Add hover info
        node_info = []
        for node in G.nodes():
            degree = G.degree(node)
            is_trap = node and node[0] in self.trap_letters
            info = f"Node: {node}<br>Degree: {degree}<br>Trap: {is_trap}"
            node_info.append(info)
        
        node_trace.hovertext = node_info
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Interactive Word Association Network',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Hover over nodes for details",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002 ) ],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        return fig
    
    def score_distribution_plot(self, scores: Dict[str, float], 
                               score_name: str = "Score") -> go.Figure:
        """Create distribution plots for scoring metrics."""
        df = pd.DataFrame([
            {
                'word': word, 
                'score': score, 
                'is_trap': word and word[0] in self.trap_letters,
                'first_letter': word[0] if word else ''
            }
            for word, score in scores.items()
        ])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Score Distribution', 'Trap vs Non-Trap', 
                           'Top 20 Words', 'Score by First Letter'),
            specs=[[{"type": "histogram"}, {"type": "box"}],
                   [{"type": "bar"}, {"type": "violin"}]]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=df['score'], name='All Words', nbinsx=30),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(y=df[df['is_trap']]['score'], name='Trap Words'),
            row=1, col=2
        )
        fig.add_trace(
            go.Box(y=df[~df['is_trap']]['score'], name='Non-Trap Words'),
            row=1, col=2
        )
        
        # Top words bar chart
        top_20 = df.nlargest(20, 'score')
        fig.add_trace(
            go.Bar(x=top_20['word'], y=top_20['score'], 
                  marker_color=['red' if is_trap else 'blue' for is_trap in top_20['is_trap']]),
            row=2, col=1
        )
        
        # First letter violin plot
        for letter in sorted(df['first_letter'].unique()):
            letter_data = df[df['first_letter'] == letter]['score']
            if len(letter_data) > 0:
                fig.add_trace(
                    go.Violin(y=letter_data, name=letter, 
                             line_color='red' if letter in self.trap_letters else 'blue'),
                    row=2, col=2
                )
        
        fig.update_layout(
            title=f'{score_name} Analysis',
            showlegend=True,
            height=800
        )
        
        return fig
    
    def export_analysis(self, output_dir: Path, prefix: str = "graph_analysis"):
        """Export comprehensive analysis to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Basic statistics
        stats = self.basic_stats()
        stats_df = pd.DataFrame([stats]).T
        stats_df.columns = ['Value']
        stats_df.to_csv(output_dir / f"{prefix}_stats.csv")
        
        # Node information
        G = self.nx_graph
        nodes_data = []
        for node in G.nodes():
            nodes_data.append({
                'node': node,
                'in_degree': G.in_degree(node),
                'out_degree': G.out_degree(node),
                'is_trap': node and node[0] in self.trap_letters,
                'first_letter': node[0] if node else ''
            })
        
        nodes_df = pd.DataFrame(nodes_data)
        nodes_df.to_csv(output_dir / f"{prefix}_nodes.csv", index=False)
        
        # Edge information
        edges_data = []
        for u, v, data in G.edges(data=True):
            edges_data.append({
                'source': u,
                'target': v,
                'weight': data['weight'],
                'source_is_trap': u and u[0] in self.trap_letters,
                'target_is_trap': v and v[0] in self.trap_letters
            })
        
        edges_df = pd.DataFrame(edges_data)
        edges_df.to_csv(output_dir / f"{prefix}_edges.csv", index=False)
        
        # Community detection
        communities = self.community_detection()
        community_df = pd.DataFrame([
            {'node': node, 'community': community}
            for node, community in communities.items()
        ])
        community_df.to_csv(output_dir / f"{prefix}_communities.csv", index=False)
        
        print(f"Analysis exported to {output_dir}")