"""Enhanced data processing and validation utilities."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import numpy as np

try:
    from pydantic import BaseModel, Field, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Fallback implementations
    class BaseModel:
        pass
    def Field(*args, **kwargs):
        return None
    def validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from .graph import Graph


class EdgeData(BaseModel):
    """Pydantic model for edge validation."""
    src: str = Field(..., min_length=1, description="Source node")
    dst: str = Field(..., min_length=1, description="Destination node") 
    weight: float = Field(..., gt=0, description="Edge weight (must be positive)")


class GraphMetadata(BaseModel):
    """Metadata for graph datasets."""
    name: str = Field(..., description="Dataset name")
    description: Optional[str] = Field(None, description="Dataset description")
    source: Optional[str] = Field(None, description="Data source")
    version: str = Field("1.0", description="Dataset version")
    created_date: Optional[str] = Field(None, description="Creation date")
    num_nodes: int = Field(..., ge=0, description="Number of nodes")
    num_edges: int = Field(..., ge=0, description="Number of edges")
    trap_letters: Set[str] = Field(default_factory=lambda: {"r", "s", "t"})
    
    @validator('trap_letters')
    def validate_trap_letters(cls, v):
        if not v:
            raise ValueError("trap_letters cannot be empty")
        return {letter.lower() for letter in v}


class GraphData(BaseModel):
    """Complete graph data with metadata and validation."""
    metadata: GraphMetadata
    edges: List[EdgeData]
    
    @validator('edges')
    def validate_edges_consistency(cls, v, values):
        if 'metadata' in values:
            if len(v) != values['metadata'].num_edges:
                raise ValueError(f"Edge count mismatch: metadata says {values['metadata'].num_edges}, but got {len(v)}")
        return v


class DataProcessor:
    """Enhanced data processing with validation and multiple format support."""
    
    @staticmethod
    def load_csv(path: Union[str, Path], 
                 validate: bool = True,
                 lowercase: bool = True,
                 min_weight: float = 0.0,
                 max_edges_per_node: Optional[int] = None) -> Graph:
        """Load and validate CSV data with enhanced options."""
        df = pd.read_csv(path)
        
        # Validate required columns
        required_cols = {'src', 'dst', 'weight'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")
        
        # Clean data
        df = df.dropna(subset=['src', 'dst', 'weight'])
        
        if lowercase:
            df['src'] = df['src'].astype(str).str.lower().str.strip()
            df['dst'] = df['dst'].astype(str).str.lower().str.strip()
        
        # Filter by weight
        df = df[df['weight'] > min_weight]
        
        # Remove empty strings
        df = df[(df['src'] != '') & (df['dst'] != '')]
        
        # Validate if requested
        if validate:
            edges = [EdgeData(src=row['src'], dst=row['dst'], weight=row['weight']) 
                    for _, row in df.iterrows()]
        
        # Build graph
        graph: Graph = {}
        edge_counts = {}
        
        for _, row in df.iterrows():
            src, dst, weight = row['src'], row['dst'], row['weight']
            
            # Apply max edges per node limit
            if max_edges_per_node:
                edge_counts[src] = edge_counts.get(src, 0) + 1
                if edge_counts[src] > max_edges_per_node:
                    continue
            
            if src not in graph:
                graph[src] = {}
            
            # Accumulate weights for duplicate edges
            graph[src][dst] = graph[src].get(dst, 0.0) + weight
        
        return graph
    
    @staticmethod
    def load_json(path: Union[str, Path]) -> Graph:
        """Load graph from JSON format."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'graph' in data:
            # Format: {"graph": {"node1": {"node2": weight, ...}, ...}}
            return data['graph']
        elif 'edges' in data:
            # Format: {"edges": [{"src": "...", "dst": "...", "weight": ...}, ...]}
            graph: Graph = {}
            for edge in data['edges']:
                src, dst, weight = edge['src'], edge['dst'], edge['weight']
                if src not in graph:
                    graph[src] = {}
                graph[src][dst] = graph[src].get(dst, 0.0) + weight
            return graph
        else:
            # Assume direct adjacency format
            return data
    
    @staticmethod
    def load_adjacency_matrix(path: Union[str, Path], 
                             nodes: Optional[List[str]] = None) -> Graph:
        """Load graph from adjacency matrix (CSV or NPY)."""
        path = Path(path)
        
        if path.suffix == '.npy':
            matrix = np.load(path)
        else:
            matrix = np.loadtxt(path, delimiter=',')
        
        n = matrix.shape[0]
        if matrix.shape[1] != n:
            raise ValueError("Adjacency matrix must be square")
        
        if nodes is None:
            nodes = [f"node_{i}" for i in range(n)]
        elif len(nodes) != n:
            raise ValueError(f"Number of node names ({len(nodes)}) doesn't match matrix size ({n})")
        
        graph: Graph = {}
        for i in range(n):
            for j in range(n):
                weight = matrix[i, j]
                if weight > 0:
                    src, dst = nodes[i], nodes[j]
                    if src not in graph:
                        graph[src] = {}
                    graph[src][dst] = weight
        
        return graph
    
    @staticmethod
    def save_csv(graph: Graph, path: Union[str, Path], include_metadata: bool = True):
        """Save graph to CSV format."""
        edges = []
        for src, targets in graph.items():
            for dst, weight in targets.items():
                edges.append({'src': src, 'dst': dst, 'weight': weight})
        
        df = pd.DataFrame(edges)
        
        if include_metadata:
            # Add metadata as comments
            metadata_lines = [
                f"# Graph exported from rst_trap_finder",
                f"# Nodes: {len(graph)}",
                f"# Edges: {len(edges)}",
                f"# Date: {pd.Timestamp.now().isoformat()}",
            ]
            
            with open(path, 'w', encoding='utf-8') as f:
                for line in metadata_lines:
                    f.write(line + '\n')
                df.to_csv(f, index=False)
        else:
            df.to_csv(path, index=False)
    
    @staticmethod
    def save_json(graph: Graph, path: Union[str, Path], 
                  format_type: str = 'adjacency', include_metadata: bool = True):
        """Save graph to JSON format."""
        data = {}
        
        if include_metadata:
            num_edges = sum(len(targets) for targets in graph.values())
            data['metadata'] = {
                'num_nodes': len(graph),
                'num_edges': num_edges,
                'format': format_type,
                'created_date': pd.Timestamp.now().isoformat(),
            }
        
        if format_type == 'adjacency':
            data['graph'] = graph
        elif format_type == 'edge_list':
            edges = []
            for src, targets in graph.items():
                for dst, weight in targets.items():
                    edges.append({'src': src, 'dst': dst, 'weight': weight})
            data['edges'] = edges
        else:
            raise ValueError(f"Unknown format type: {format_type}")
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def save_graphml(graph: Graph, path: Union[str, Path]):
        """Save graph to GraphML format (requires networkx)."""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("NetworkX required for GraphML export")
        
        G = nx.DiGraph()
        for src, targets in graph.items():
            for dst, weight in targets.items():
                G.add_edge(src, dst, weight=weight)
        
        nx.write_graphml(G, path)
    
    @staticmethod
    def save_pickle(graph: Graph, path: Union[str, Path]):
        """Save graph to pickle format (fast loading)."""
        with open(path, 'wb') as f:
            pickle.dump(graph, f)
    
    @staticmethod
    def load_pickle(path: Union[str, Path]) -> Graph:
        """Load graph from pickle format."""
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def preprocess_text_data(text_data: List[str], 
                           min_length: int = 2,
                           max_length: int = 50,
                           allowed_chars: Optional[Set[str]] = None) -> List[str]:
        """Preprocess text data for graph construction."""
        if allowed_chars is None:
            allowed_chars = set('abcdefghijklmnopqrstuvwxyz')
        
        processed = []
        for text in text_data:
            # Clean text
            text = text.lower().strip()
            
            # Filter by character set
            text = ''.join(c for c in text if c in allowed_chars)
            
            # Filter by length
            if min_length <= len(text) <= max_length:
                processed.append(text)
        
        return processed
    
    @staticmethod
    def build_association_graph(word_pairs: List[Tuple[str, str]], 
                               weights: Optional[List[float]] = None) -> Graph:
        """Build graph from word association pairs."""
        if weights is None:
            weights = [1.0] * len(word_pairs)
        
        if len(word_pairs) != len(weights):
            raise ValueError("Number of pairs and weights must match")
        
        graph: Graph = {}
        for (src, dst), weight in zip(word_pairs, weights):
            if src not in graph:
                graph[src] = {}
            graph[src][dst] = graph[src].get(dst, 0.0) + weight
        
        return graph
    
    @staticmethod
    def validate_graph(graph: Graph, 
                      allow_self_loops: bool = False,
                      min_out_degree: int = 0,
                      max_out_degree: Optional[int] = None) -> List[str]:
        """Validate graph structure and return list of issues."""
        issues = []
        
        for src, targets in graph.items():
            # Check for empty source names
            if not src or not isinstance(src, str):
                issues.append(f"Invalid source node: {repr(src)}")
                continue
            
            # Check out-degree constraints
            out_degree = len(targets)
            if out_degree < min_out_degree:
                issues.append(f"Node {src} has out-degree {out_degree} < minimum {min_out_degree}")
            
            if max_out_degree and out_degree > max_out_degree:
                issues.append(f"Node {src} has out-degree {out_degree} > maximum {max_out_degree}")
            
            for dst, weight in targets.items():
                # Check for empty destination names
                if not dst or not isinstance(dst, str):
                    issues.append(f"Invalid destination node: {repr(dst)} from {src}")
                
                # Check weights
                if not isinstance(weight, (int, float)) or weight <= 0:
                    issues.append(f"Invalid weight {weight} for edge {src} -> {dst}")
                
                # Check self-loops
                if not allow_self_loops and src == dst:
                    issues.append(f"Self-loop detected: {src} -> {dst}")
        
        return issues
    
    @staticmethod
    def get_data_summary(graph: Graph) -> Dict[str, Any]:
        """Get comprehensive summary of graph data."""
        nodes = set(graph.keys())
        all_targets = set()
        total_edges = 0
        total_weight = 0.0
        weights = []
        out_degrees = []
        
        for src, targets in graph.items():
            all_targets.update(targets.keys())
            out_degrees.append(len(targets))
            for weight in targets.values():
                total_edges += 1
                total_weight += weight
                weights.append(weight)
        
        all_nodes = nodes | all_targets
        in_degrees = {node: 0 for node in all_nodes}
        for targets in graph.values():
            for dst in targets:
                in_degrees[dst] += 1
        
        return {
            'num_nodes': len(all_nodes),
            'num_edges': total_edges,
            'total_weight': total_weight,
            'avg_weight': total_weight / total_edges if total_edges > 0 else 0.0,
            'min_weight': min(weights) if weights else 0.0,
            'max_weight': max(weights) if weights else 0.0,
            'avg_out_degree': np.mean(out_degrees) if out_degrees else 0.0,
            'max_out_degree': max(out_degrees) if out_degrees else 0,
            'avg_in_degree': np.mean(list(in_degrees.values())),
            'max_in_degree': max(in_degrees.values()) if in_degrees else 0,
            'density': total_edges / (len(all_nodes) * (len(all_nodes) - 1)) if len(all_nodes) > 1 else 0.0,
            'nodes_with_no_outgoing': len([node for node in all_nodes if node not in graph]),
            'nodes_with_no_incoming': len([node for node, degree in in_degrees.items() if degree == 0]),
        }