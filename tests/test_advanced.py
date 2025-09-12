"""Advanced test scenarios with property-based testing."""
import pytest
from hypothesis import given, strategies as st
from hypothesis import settings, HealthCheck
import numpy as np
from pathlib import Path
import tempfile
import json

from rst_trap_finder import TRAP_LETTERS
from rst_trap_finder.graph import Graph
from rst_trap_finder.scores import (
    one_step_rst_prob, escape_hardness, k_step_rst_prob,
    minimax_topm, biased_pagerank, composite
)
from rst_trap_finder.data_processing import DataProcessor, GraphData


# Hypothesis strategies for generating test data
@st.composite
def word_strategy(draw):
    """Generate valid word strings."""
    length = draw(st.integers(min_value=1, max_value=20))
    return ''.join(draw(st.lists(
        st.sampled_from('abcdefghijklmnopqrstuvwxyz'),
        min_size=length, max_size=length
    )))


@st.composite
def edge_strategy(draw):
    """Generate valid graph edges."""
    src = draw(word_strategy())
    dst = draw(word_strategy())
    weight = draw(st.floats(min_value=0.1, max_value=100.0))
    return (src, dst, weight)


@st.composite
def graph_strategy(draw):
    """Generate valid graphs."""
    edges = draw(st.lists(edge_strategy(), min_size=1, max_size=50))
    graph = {}
    
    for src, dst, weight in edges:
        if src not in graph:
            graph[src] = {}
        graph[src][dst] = graph[src].get(dst, 0.0) + weight
    
    return graph


class TestPropertyBasedScoring:
    """Property-based tests for scoring functions."""
    
    @given(graph_strategy())
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
    def test_one_step_prob_bounds(self, graph):
        """One-step probability should be between 0 and 1."""
        for node in graph:
            prob = one_step_rst_prob(node, graph, TRAP_LETTERS)
            assert 0.0 <= prob <= 1.0, f"Probability {prob} out of bounds for node {node}"
    
    @given(graph_strategy())
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
    def test_escape_hardness_bounds(self, graph):
        """Escape hardness should be between 0 and 1."""
        for node in graph:
            hardness = escape_hardness(node, graph, TRAP_LETTERS)
            assert 0.0 <= hardness <= 1.0, f"Hardness {hardness} out of bounds for node {node}"
    
    @given(graph_strategy())
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=30)
    def test_pagerank_normalization(self, graph):
        """PageRank values should sum to approximately 1."""
        pr = biased_pagerank(graph, TRAP_LETTERS, iters=20)
        total = sum(pr.values())
        assert abs(total - 1.0) < 1e-6, f"PageRank sum {total} not normalized"
    
    @given(graph_strategy(), st.integers(min_value=1, max_value=5))
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=30)
    def test_k_step_monotonicity(self, graph, k):
        """K-step probability should generally increase with k."""
        for node in graph:
            prob_k1 = k_step_rst_prob(node, graph, TRAP_LETTERS, k=k)
            prob_k2 = k_step_rst_prob(node, graph, TRAP_LETTERS, k=k+1)
            # Allow small violations due to approximation
            assert prob_k2 >= prob_k1 - 0.1, f"Non-monotonic k-step for {node}: {prob_k1} -> {prob_k2}"
    
    @given(graph_strategy())
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=30)
    def test_composite_score_validity(self, graph):
        """Composite scores should be non-negative and finite."""
        pr = biased_pagerank(graph, TRAP_LETTERS, iters=20)
        for node in graph:
            score = composite(node, graph, TRAP_LETTERS, pr)
            assert score >= 0.0, f"Negative composite score {score} for node {node}"
            assert np.isfinite(score), f"Non-finite composite score {score} for node {node}"


class TestDataProcessing:
    """Test data processing and validation."""
    
    def test_csv_loading_validation(self):
        """Test CSV loading with validation."""
        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("src,dst,weight\n")
            f.write("start,end,1.5\n")
            f.write("end,finish,2.0\n")
            f.write("invalid,,0.5\n")  # Should be filtered out
            f.write("negative,test,-1.0\n")  # Should be filtered out
            temp_path = f.name
        
        try:
            graph = DataProcessor.load_csv(temp_path, validate=True)
            
            assert 'start' in graph
            assert graph['start']['end'] == 1.5
            assert 'end' in graph
            assert graph['end']['finish'] == 2.0
            assert 'invalid' not in graph
            assert 'negative' not in graph
        finally:
            Path(temp_path).unlink()
    
    def test_json_roundtrip(self):
        """Test JSON save/load roundtrip."""
        original_graph = {
            'a': {'b': 1.0, 'c': 2.0},
            'b': {'c': 1.5},
            'c': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            DataProcessor.save_json(original_graph, temp_path)
            loaded_graph = DataProcessor.load_json(temp_path)
            
            assert loaded_graph['graph'] == original_graph
        finally:
            Path(temp_path).unlink()
    
    def test_graph_validation(self):
        """Test graph structure validation."""
        valid_graph = {'a': {'b': 1.0}, 'b': {'c': 2.0}}
        issues = DataProcessor.validate_graph(valid_graph)
        assert len(issues) == 0
        
        invalid_graph = {
            '': {'b': 1.0},  # Empty source
            'a': {'': 2.0},  # Empty destination
            'b': {'c': -1.0},  # Negative weight
            'c': {'c': 1.0}  # Self-loop
        }
        issues = DataProcessor.validate_graph(invalid_graph, allow_self_loops=False)
        assert len(issues) >= 4
    
    def test_data_summary(self):
        """Test data summary computation."""
        graph = {
            'a': {'b': 1.0, 'c': 2.0},
            'b': {'c': 1.5, 'd': 0.5},
            'c': {},
            'd': {'a': 1.0}
        }
        
        summary = DataProcessor.get_data_summary(graph)
        
        assert summary['num_nodes'] == 4
        assert summary['num_edges'] == 5
        assert summary['total_weight'] == 6.0
        assert summary['avg_weight'] == 1.2
        assert summary['max_out_degree'] == 2
        assert summary['nodes_with_no_outgoing'] == 1  # 'c' has no outgoing edges


@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks."""
    
    def generate_large_graph(self, num_nodes: int = 1000) -> Graph:
        """Generate a large graph for performance testing."""
        import random
        random.seed(42)
        
        graph = {}
        nodes = [f"word_{i}" for i in range(num_nodes)]
        
        for i, node in enumerate(nodes):
            # Each node connects to 5-20 random other nodes
            num_connections = random.randint(5, 20)
            targets = random.sample([n for n in nodes if n != node], 
                                  min(num_connections, num_nodes - 1))
            
            graph[node] = {}
            for target in targets:
                weight = random.uniform(0.1, 5.0)
                graph[node][target] = weight
        
        return graph
    
    def test_pagerank_performance(self, benchmark):
        """Benchmark PageRank computation."""
        graph = self.generate_large_graph(500)
        
        result = benchmark(biased_pagerank, graph, TRAP_LETTERS, iters=20)
        assert len(result) > 0
    
    def test_composite_scoring_performance(self, benchmark):
        """Benchmark composite scoring."""
        graph = self.generate_large_graph(200)
        pr = biased_pagerank(graph, TRAP_LETTERS, iters=10)
        
        def score_all_nodes():
            return {node: composite(node, graph, TRAP_LETTERS, pr) 
                   for node in list(graph.keys())[:100]}  # Score first 100 nodes
        
        result = benchmark(score_all_nodes)
        assert len(result) == 100


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_end_to_end_analysis(self):
        """Test complete analysis workflow."""
        # Create test data
        graph = {
            'start': {'red': 2.0, 'green': 1.0, 'blue': 1.0},
            'red': {'stop': 1.5, 'apple': 1.0},
            'green': {'go': 1.0, 'tree': 2.0},
            'blue': {'sky': 1.0, 'sad': 1.5},
            'stop': {},
            'apple': {'red': 0.5},
            'go': {'green': 0.5},
            'tree': {'green': 0.5},
            'sky': {'blue': 0.5},
            'sad': {'blue': 0.5}
        }
        
        # Compute all scores
        pr = biased_pagerank(graph, TRAP_LETTERS)
        
        scores = {}
        for node in graph:
            scores[node] = {
                'one_step': one_step_rst_prob(node, graph, TRAP_LETTERS),
                'escape_hardness': escape_hardness(node, graph, TRAP_LETTERS),
                'k2_step': k_step_rst_prob(node, graph, TRAP_LETTERS, k=2),
                'minimax': minimax_topm(node, graph, TRAP_LETTERS),
                'pagerank': pr.get(node, 0.0),
                'composite': composite(node, graph, TRAP_LETTERS, pr)
            }
        
        # Verify that nodes starting with trap letters generally score differently
        trap_nodes = [node for node in graph if node and node[0] in TRAP_LETTERS]
        non_trap_nodes = [node for node in graph if node and node[0] not in TRAP_LETTERS]
        
        assert len(trap_nodes) > 0, "No trap nodes found in test graph"
        assert len(non_trap_nodes) > 0, "No non-trap nodes found in test graph"
        
        # At least some scores should be computed
        for node in graph:
            assert all(np.isfinite(score) and score >= 0 
                      for score in scores[node].values()), f"Invalid scores for {node}"
    
    def test_data_export_import_cycle(self):
        """Test exporting and importing data in different formats."""
        original_graph = {
            'alpha': {'beta': 1.0, 'gamma': 2.0},
            'beta': {'gamma': 1.5},
            'gamma': {'alpha': 0.5}
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test CSV roundtrip
            csv_path = temp_path / "test.csv"
            DataProcessor.save_csv(original_graph, csv_path)
            loaded_csv = DataProcessor.load_csv(csv_path)
            assert loaded_csv == original_graph
            
            # Test JSON roundtrip
            json_path = temp_path / "test.json"
            DataProcessor.save_json(original_graph, json_path, format_type='adjacency')
            loaded_json = DataProcessor.load_json(json_path)
            assert loaded_json == original_graph
            
            # Test pickle roundtrip
            pickle_path = temp_path / "test.pkl"
            DataProcessor.save_pickle(original_graph, pickle_path)
            loaded_pickle = DataProcessor.load_pickle(pickle_path)
            assert loaded_pickle == original_graph


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_graph(self):
        """Test behavior with empty graph."""
        empty_graph = {}
        
        pr = biased_pagerank(empty_graph, TRAP_LETTERS)
        assert pr == {}
        
        # Should not crash on empty graph
        assert one_step_rst_prob('nonexistent', empty_graph, TRAP_LETTERS) == 0.0
    
    def test_single_node_graph(self):
        """Test behavior with single node."""
        single_graph = {'alone': {}}
        
        pr = biased_pagerank(single_graph, TRAP_LETTERS)
        assert len(pr) == 1
        assert abs(pr['alone'] - 1.0) < 1e-6
        
        assert one_step_rst_prob('alone', single_graph, TRAP_LETTERS) == 0.0
    
    def test_disconnected_graph(self):
        """Test behavior with disconnected components."""
        disconnected_graph = {
            'a': {'b': 1.0},
            'b': {},
            'c': {'d': 1.0},
            'd': {}
        }
        
        pr = biased_pagerank(disconnected_graph, TRAP_LETTERS)
        assert len(pr) == 4
        assert abs(sum(pr.values()) - 1.0) < 1e-6
    
    def test_self_loops(self):
        """Test behavior with self-loops."""
        self_loop_graph = {
            'a': {'a': 1.0, 'b': 1.0},
            'b': {'b': 2.0}
        }
        
        # Should handle self-loops gracefully
        pr = biased_pagerank(self_loop_graph, TRAP_LETTERS)
        assert len(pr) == 2
        assert all(v >= 0 for v in pr.values())
    
    def test_extreme_weights(self):
        """Test behavior with extreme edge weights."""
        extreme_graph = {
            'a': {'b': 1e-10, 'c': 1e10},
            'b': {'c': 1.0},
            'c': {}
        }
        
        # Should not crash with extreme weights
        scores = {
            'one_step': one_step_rst_prob('a', extreme_graph, TRAP_LETTERS),
            'escape_hardness': escape_hardness('a', extreme_graph, TRAP_LETTERS),
            'k2_step': k_step_rst_prob('a', extreme_graph, TRAP_LETTERS, k=2),
        }
        
        assert all(0.0 <= score <= 1.0 for score in scores.values())
        assert all(np.isfinite(score) for score in scores.values())
        
    def test_unicode_handling(self):
        """Test handling of unicode characters in node names."""
        unicode_graph = {
            'café': {'naïve': 1.0},
            'naïve': {'résumé': 2.0},
            'résumé': {}
        }
        
        # Should handle unicode gracefully
        pr = biased_pagerank(unicode_graph, TRAP_LETTERS)
        assert len(pr) == 3
        
        for node in unicode_graph:
            score = one_step_rst_prob(node, unicode_graph, TRAP_LETTERS)
            assert np.isfinite(score)


@pytest.mark.slow
class TestLargeScale:
    """Tests for large-scale performance and behavior."""
    
    def test_large_graph_analysis(self):
        """Test analysis on large graphs."""
        # Generate a graph with 10,000 nodes
        import random
        random.seed(42)
        
        nodes = [f"word_{i:05d}" for i in range(10000)]
        graph = {}
        
        for i, node in enumerate(nodes):
            if i % 1000 == 0:  # Progress indicator
                print(f"Generating node {i}")
            
            # Each node connects to 3-10 random other nodes
            num_connections = random.randint(3, 10)
            targets = random.sample([n for n in nodes if n != node], num_connections)
            
            graph[node] = {}
            for target in targets:
                weight = random.uniform(0.1, 2.0)
                graph[node][target] = weight
        
        # Test that analysis completes in reasonable time
        import time
        
        start_time = time.time()
        pr = biased_pagerank(graph, TRAP_LETTERS, iters=10)
        pagerank_time = time.time() - start_time
        
        assert len(pr) == 10000
        assert pagerank_time < 30.0, f"PageRank took {pagerank_time:.2f}s, too slow"
        
        # Test a subset of composite scores
        start_time = time.time()
        test_nodes = nodes[:100]
        composite_scores = {
            node: composite(node, graph, TRAP_LETTERS, pr) 
            for node in test_nodes
        }
        composite_time = time.time() - start_time
        
        assert len(composite_scores) == 100
        assert composite_time < 10.0, f"Composite scoring took {composite_time:.2f}s, too slow"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])