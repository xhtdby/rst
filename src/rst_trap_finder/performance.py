"""Performance optimization utilities for large-scale graph processing."""
from __future__ import annotations

import functools
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
import hashlib
import json

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from .graph import Graph

F = TypeVar('F', bound=Callable[..., Any])


class Cache:
    """Simple file-based cache for expensive computations."""
    
    def __init__(self, cache_dir: Union[str, Path] = None, max_size: int = 1000):
        self.cache_dir = Path(cache_dir or Path.home() / '.rst_cache')
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size = max_size
        self._access_times = {}
        
    def _get_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate a cache key from function name and arguments."""
        # Create a deterministic hash of the arguments
        args_str = str(args) + str(sorted(kwargs.items()))
        key_hash = hashlib.md5(args_str.encode()).hexdigest()
        return f"{func_name}_{key_hash}"
    
    def _get_cache_path(self, key: str) -> Path:
        """Get the cache file path for a given key."""
        return self.cache_dir / f"{key}.pkl"
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    result = pickle.load(f)
                self._access_times[key] = time.time()
                return result
            except (pickle.PickleError, OSError):
                # Remove corrupted cache file
                cache_path.unlink(missing_ok=True)
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in cache."""
        # Clean up old entries if cache is full
        if len(self._access_times) >= self.max_size:
            self._cleanup_old_entries()
        
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            self._access_times[key] = time.time()
        except (pickle.PickleError, OSError):
            pass  # Fail silently on cache write errors
    
    def _cleanup_old_entries(self) -> None:
        """Remove old cache entries to stay under max_size."""
        if len(self._access_times) <= self.max_size // 2:
            return
        
        # Sort by access time and remove oldest entries
        sorted_items = sorted(self._access_times.items(), key=lambda x: x[1])
        to_remove = sorted_items[:len(sorted_items) // 2]
        
        for key, _ in to_remove:
            cache_path = self._get_cache_path(key)
            cache_path.unlink(missing_ok=True)
            del self._access_times[key]
    
    def clear(self) -> None:
        """Clear all cache entries."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink(missing_ok=True)
        self._access_times.clear()
    
    def size(self) -> int:
        """Get number of cached items."""
        return len(list(self.cache_dir.glob("*.pkl")))


# Global cache instance
_cache = Cache()


def cached(cache_instance: Optional[Cache] = None) -> Callable[[F], F]:
    """Decorator to cache function results."""
    def decorator(func: F) -> F:
        cache_obj = cache_instance or _cache
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache_obj._get_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            result = cache_obj.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache_obj.set(cache_key, result)
            return result
        
        return wrapper
    return decorator


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"{self.name} took {duration:.3f} seconds")
    
    @property
    def duration(self) -> float:
        """Get the duration in seconds."""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0


class BatchProcessor:
    """Process large graphs in batches to manage memory usage."""
    
    def __init__(self, batch_size: int = 1000, n_jobs: int = 1):
        self.batch_size = batch_size
        self.n_jobs = n_jobs if JOBLIB_AVAILABLE else 1
    
    def process_nodes(self, nodes: List[str], func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Process nodes in batches using the given function."""
        results = {}
        
        if self.n_jobs > 1 and JOBLIB_AVAILABLE:
            # Parallel processing
            def process_batch(batch_nodes):
                return {node: func(node, *args, **kwargs) for node in batch_nodes}
            
            # Split into batches
            batches = [nodes[i:i + self.batch_size] 
                      for i in range(0, len(nodes), self.batch_size)]
            
            # Process batches in parallel
            batch_results = Parallel(n_jobs=self.n_jobs)(
                delayed(process_batch)(batch) for batch in batches
            )
            
            # Combine results
            for batch_result in batch_results:
                results.update(batch_result)
        
        else:
            # Sequential processing
            for i in range(0, len(nodes), self.batch_size):
                batch = nodes[i:i + self.batch_size]
                print(f"Processing batch {i//self.batch_size + 1}/{(len(nodes)-1)//self.batch_size + 1}")
                
                for node in batch:
                    results[node] = func(node, *args, **kwargs)
        
        return results


class MemoryOptimizer:
    """Utilities for memory-efficient graph processing."""
    
    @staticmethod
    def sparse_graph_representation(graph: Graph) -> Dict[str, Any]:
        """Convert graph to sparse representation for memory efficiency."""
        if not NUMPY_AVAILABLE:
            return graph
        
        # Create node index mapping
        all_nodes = set(graph.keys())
        for neighbors in graph.values():
            all_nodes.update(neighbors.keys())
        
        node_to_idx = {node: i for i, node in enumerate(sorted(all_nodes))}
        idx_to_node = {i: node for node, i in node_to_idx.items()}
        
        # Build sparse matrix representation
        rows, cols, data = [], [], []
        
        for src, neighbors in graph.items():
            src_idx = node_to_idx[src]
            for dst, weight in neighbors.items():
                dst_idx = node_to_idx[dst]
                rows.append(src_idx)
                cols.append(dst_idx)
                data.append(weight)
        
        return {
            'shape': (len(all_nodes), len(all_nodes)),
            'rows': np.array(rows),
            'cols': np.array(cols),
            'data': np.array(data),
            'node_to_idx': node_to_idx,
            'idx_to_node': idx_to_node
        }
    
    @staticmethod
    def estimate_memory_usage(graph: Graph) -> Dict[str, float]:
        """Estimate memory usage of graph and operations."""
        num_nodes = len(set(graph.keys()) | 
                         set().union(*[neighbors.keys() for neighbors in graph.values()]))
        num_edges = sum(len(neighbors) for neighbors in graph.values())
        
        # Rough estimates in MB
        estimates = {
            'graph_dict': (num_edges * 64 + num_nodes * 32) / (1024 * 1024),  # Dict overhead
            'pagerank_dense': num_nodes * 8 / (1024 * 1024),  # Float64 array
            'adjacency_matrix': num_nodes * num_nodes * 8 / (1024 * 1024),  # Full matrix
            'sparse_matrix': num_edges * 24 / (1024 * 1024),  # COO format
        }
        
        return estimates
    
    @staticmethod
    def suggest_batch_size(graph: Graph, available_memory_mb: float = 1000) -> int:
        """Suggest optimal batch size based on available memory."""
        estimates = MemoryOptimizer.estimate_memory_usage(graph)
        per_node_memory = estimates['graph_dict'] / len(graph) if graph else 1.0
        
        # Use 80% of available memory
        usable_memory = available_memory_mb * 0.8
        suggested_size = int(usable_memory / per_node_memory)
        
        # Clamp to reasonable range
        return max(100, min(suggested_size, 10000))


class Profiler:
    """Simple profiler for analyzing performance bottlenecks."""
    
    def __init__(self):
        self.timings = {}
        self.call_counts = {}
    
    def time_function(self, func_name: str = None):
        """Decorator to time function calls."""
        def decorator(func):
            name = func_name or func.__name__
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                self.timings[name] = self.timings.get(name, 0) + duration
                self.call_counts[name] = self.call_counts.get(name, 0) + 1
                
                return result
            return wrapper
        return decorator
    
    def get_profile_report(self) -> str:
        """Generate a performance report."""
        if not self.timings:
            return "No timing data collected."
        
        lines = ["Performance Profile:", "=" * 50]
        
        # Sort by total time
        sorted_items = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
        
        for func_name, total_time in sorted_items:
            count = self.call_counts[func_name]
            avg_time = total_time / count
            lines.append(f"{func_name:30} | {total_time:8.3f}s | {count:6d} calls | {avg_time:8.3f}s avg")
        
        return "\n".join(lines)
    
    def reset(self):
        """Reset all timing data."""
        self.timings.clear()
        self.call_counts.clear()


class ProgressTracker:
    """Simple progress tracking for long operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0
    
    def update(self, increment: int = 1):
        """Update progress."""
        self.current += increment
        current_time = time.time()
        
        # Update every second or at completion
        if current_time - self.last_update > 1.0 or self.current >= self.total:
            self._print_progress()
            self.last_update = current_time
    
    def _print_progress(self):
        """Print progress bar."""
        percent = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time
        
        if self.current > 0:
            eta = elapsed * (self.total / self.current) - elapsed
            eta_str = f"ETA: {eta:.1f}s"
        else:
            eta_str = "ETA: --"
        
        # Simple text progress bar
        bar_length = 30
        filled_length = int(bar_length * self.current / self.total)
        bar = "█" * filled_length + "-" * (bar_length - filled_length)
        
        print(f"\r{self.description}: |{bar}| {percent:.1f}% ({self.current}/{self.total}) {eta_str}", 
              end="", flush=True)
        
        if self.current >= self.total:
            print()  # New line when complete


# Optimized scoring functions for large graphs
@cached()
def optimized_pagerank(graph: Graph, trap_letters: set, alpha: float = 1.5, 
                      sparse: bool = True) -> Dict[str, float]:
    """Memory-optimized PageRank computation."""
    if sparse and NUMPY_AVAILABLE:
        # Use sparse matrix representation for large graphs
        return _sparse_pagerank(graph, trap_letters, alpha)
    else:
        # Fall back to standard implementation
        from .scores import biased_pagerank
        return biased_pagerank(graph, trap_letters, alpha)


def _sparse_pagerank(graph: Graph, trap_letters: set, alpha: float) -> Dict[str, float]:
    """Sparse matrix implementation of PageRank."""
    try:
        from scipy import sparse
        from scipy.sparse.linalg import eigs
    except ImportError:
        # Fall back to dense implementation
        from .scores import biased_pagerank
        return biased_pagerank(graph, trap_letters, alpha)
    
    # Convert to sparse representation
    sparse_repr = MemoryOptimizer.sparse_graph_representation(graph)
    
    n = sparse_repr['shape'][0]
    if n == 0:
        return {}
    
    # Build transition matrix
    A = sparse.coo_matrix(
        (sparse_repr['data'], (sparse_repr['rows'], sparse_repr['cols'])),
        shape=sparse_repr['shape']
    ).tocsr()
    
    # Apply bias to trap letters
    node_to_idx = sparse_repr['node_to_idx']
    for node, idx in node_to_idx.items():
        if node and node[0] in trap_letters:
            A[idx, :] *= alpha
    
    # Normalize rows
    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    D_inv = sparse.diags(1 / row_sums)
    P = D_inv @ A
    
    # Power iteration for PageRank
    d = 0.85  # Damping factor
    x = np.ones(n) / n
    
    for _ in range(50):  # Max iterations
        x_new = (1 - d) / n + d * P.T @ x
        if np.linalg.norm(x_new - x) < 1e-6:
            break
        x = x_new
    
    # Convert back to dictionary
    idx_to_node = sparse_repr['idx_to_node']
    return {idx_to_node[i]: x[i] for i in range(n)}


def benchmark_scoring_methods(graph: Graph, num_runs: int = 5) -> Dict[str, float]:
    """Benchmark different scoring methods."""
    from .scores import biased_pagerank, composite, one_step_rst_prob
    
    results = {}
    
    # Test PageRank
    with Timer("PageRank") as timer:
        for _ in range(num_runs):
            biased_pagerank(graph, {frozenset(['r', 's', 't'])})
    results['pagerank'] = timer.duration / num_runs
    
    # Test composite scoring
    pr = biased_pagerank(graph, {frozenset(['r', 's', 't'])})
    test_nodes = list(graph.keys())[:min(100, len(graph))]
    
    with Timer("Composite scoring") as timer:
        for _ in range(num_runs):
            for node in test_nodes:
                composite(node, graph, {frozenset(['r', 's', 't'])}, pr)
    results['composite'] = timer.duration / num_runs
    
    # Test one-step probability
    with Timer("One-step probability") as timer:
        for _ in range(num_runs):
            for node in test_nodes:
                one_step_rst_prob(node, graph, {frozenset(['r', 's', 't'])})
    results['one_step'] = timer.duration / num_runs
    
    return results


def optimize_for_large_graph(graph: Graph, memory_limit_mb: float = 2000) -> Dict[str, Any]:
    """Optimize processing parameters for large graphs."""
    estimates = MemoryOptimizer.estimate_memory_usage(graph)
    
    recommendations = {
        'use_sparse': estimates['adjacency_matrix'] > memory_limit_mb,
        'batch_size': MemoryOptimizer.suggest_batch_size(graph, memory_limit_mb),
        'enable_caching': len(graph) > 1000,
        'use_parallel': len(graph) > 5000,
        'memory_estimates': estimates
    }
    
    if recommendations['use_sparse']:
        recommendations['pagerank_method'] = 'sparse'
    else:
        recommendations['pagerank_method'] = 'dense'
    
    return recommendations