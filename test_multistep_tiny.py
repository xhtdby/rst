#!/usr/bin/env python3
"""
Unit tests for multi-step analysis methods on tiny graphs (1-3 nodes)
Tests k_step_probability_exact/cumulative and path scoring functionality
"""

import unittest
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from rst_trap_finder.core import WordAssociationGraph
from rst_trap_finder.multistep import MultiStepAnalyzer

class TestMultiStepTinyGraphs(unittest.TestCase):
    """Test multi-step analysis on minimal graphs for validation"""
    
    def setUp(self):
        """Set up test graphs"""
        self.trap_letters = frozenset(['r', 's', 't'])
    
    def create_single_node_graph(self):
        """Create a graph with single isolated node"""
        graph = {"word": {}}
        return WordAssociationGraph(graph, self.trap_letters)
    
    def create_two_node_graph(self):
        """Create a simple two-node graph: word1 -> word2"""
        graph = {
            "word1": {"word2": 1.0},
            "word2": {}
        }
        return WordAssociationGraph(graph, self.trap_letters)
    
    def create_two_node_cycle(self):
        """Create a two-node cycle: word1 <-> word2"""
        graph = {
            "word1": {"word2": 1.0},
            "word2": {"word1": 1.0}
        }
        return WordAssociationGraph(graph, self.trap_letters)
    
    def create_three_node_linear(self):
        """Create linear graph: word1 -> word2 -> trap"""
        graph = {
            "word1": {"word2": 1.0},
            "word2": {"trap": 1.0},
            "trap": {}
        }
        return WordAssociationGraph(graph, self.trap_letters)
    
    def create_three_node_star(self):
        """Create star graph: center -> {word1, word2, trap}"""
        graph = {
            "center": {"word1": 0.5, "word2": 0.3, "trap": 0.2},
            "word1": {},
            "word2": {},
            "trap": {}
        }
        return WordAssociationGraph(graph, self.trap_letters)
    
    def test_single_node_probabilities(self):
        """Test k-step probabilities on isolated node"""
        wag = self.create_single_node_graph()
        analyzer = MultiStepAnalyzer(wag)
        
        # Single isolated node should have 0 probability of reaching anywhere
        prob_1 = analyzer.k_step_probability_exact("word", k=1)
        prob_2 = analyzer.k_step_probability_exact("word", k=2)
        cumulative = analyzer.k_step_probability_cumulative("word", k=2)
        
        self.assertEqual(prob_1, 0.0, "Isolated node should have 0 probability at k=1")
        self.assertEqual(prob_2, 0.0, "Isolated node should have 0 probability at k=2")
        self.assertEqual(cumulative, 0.0, "Isolated node should have 0 cumulative probability")
    
    def test_two_node_linear_probabilities(self):
        """Test k-step probabilities on two-node linear graph"""
        wag = self.create_two_node_graph()
        analyzer = MultiStepAnalyzer(wag)
        
        # word1 -> word2 (which has no RST connections)
        prob_1 = analyzer.k_step_probability_exact("word1", k=1)
        prob_2 = analyzer.k_step_probability_exact("word1", k=2)
        cumulative = analyzer.k_step_probability_cumulative("word1", k=2)
        
        self.assertEqual(prob_1, 0.0, "No RST words reachable in 1 step")
        self.assertEqual(prob_2, 0.0, "No RST words reachable in 2 steps")
        self.assertEqual(cumulative, 0.0, "No RST words reachable at all")
        
        # word2 has no outgoing connections
        prob_1_w2 = analyzer.k_step_probability_exact("word2", k=1)
        self.assertEqual(prob_1_w2, 0.0, "word2 has no outgoing connections")
    
    def test_two_node_cycle_probabilities(self):
        """Test k-step probabilities on two-node cycle"""
        wag = self.create_two_node_cycle()
        analyzer = MultiStepAnalyzer(wag)
        
        # In cycle with no RST words, probability should remain 0
        prob_1 = analyzer.k_step_probability_exact("word1", k=1)
        prob_2 = analyzer.k_step_probability_exact("word1", k=2)
        cumulative = analyzer.k_step_probability_cumulative("word1", k=3)
        
        self.assertEqual(prob_1, 0.0, "No RST words in cycle")
        self.assertEqual(prob_2, 0.0, "Still no RST words after 2 steps")
        self.assertEqual(cumulative, 0.0, "Cumulative should be 0 with no RST words")
    
    def test_three_node_linear_with_trap(self):
        """Test k-step probabilities with RST word at end"""
        wag = self.create_three_node_linear()
        analyzer = MultiStepAnalyzer(wag)
        
        # word1 -> word2 -> trap
        prob_1 = analyzer.k_step_probability_exact("word1", k=1)
        prob_2 = analyzer.k_step_probability_exact("word1", k=2)
        cumulative = analyzer.k_step_probability_cumulative("word1", k=2)
        
        self.assertEqual(prob_1, 0.0, "No RST reachable in 1 step")
        self.assertEqual(prob_2, 1.0, "RST definitely reachable in 2 steps")
        self.assertEqual(cumulative, 1.0, "Cumulative should be 1.0")
        
        # word2 should reach trap in 1 step
        prob_1_w2 = analyzer.k_step_probability_exact("word2", k=1)
        self.assertEqual(prob_1_w2, 1.0, "word2 directly connects to trap")
    
    def test_three_node_star_with_trap(self):
        """Test probabilistic RST connections in star graph"""
        wag = self.create_three_node_star()
        analyzer = MultiStepAnalyzer(wag)
        
        # center connects to trap with probability 0.2/(0.5+0.3+0.2) = 0.2
        prob_1 = analyzer.k_step_probability_exact("center", k=1)
        expected_prob = 0.2 / (0.5 + 0.3 + 0.2)  # 0.2 / 1.0 = 0.2
        
        self.assertAlmostEqual(prob_1, expected_prob, places=3, 
                              msg="Probability should match weight ratio")
        
        # k=2 should have same probability (no further connections)
        prob_2 = analyzer.k_step_probability_exact("center", k=2)
        self.assertEqual(prob_2, 0.0, "No additional RST paths at k=2")
        
        # Cumulative should be just the k=1 probability
        cumulative = analyzer.k_step_probability_cumulative("center", k=2)
        self.assertAlmostEqual(cumulative, expected_prob, places=3)
    
    def test_multi_step_analysis_integration(self):
        """Test full multi-step analysis on small graph"""
        wag = self.create_three_node_linear()
        analyzer = MultiStepAnalyzer(wag)
        
        # Analyze word1 which can reach trap in 2 steps
        result = analyzer.analyze_multi_step("word1", max_k=3)
        
        self.assertIsNotNone(result, "Multi-step analysis should return result")
        self.assertEqual(result.word, "word1", "Should analyze correct word")
        self.assertEqual(result.k_steps, 3, "Should have correct max_k")
        
        # Check that cumulative probability correctly shows we can reach trap in 2 steps
        self.assertGreater(result.total_probability, 0.0, "Should have positive probability to reach trap")
        
        # Strategic score should be positive (since trap is reachable)
        self.assertGreater(result.strategic_score, 0, "Strategic score should be positive")
        
        # Should find at least one path to trap
        trap_paths = [p for p in result.optimal_paths if p.rst_endpoint == "trap"]
        self.assertGreater(len(trap_paths), 0, "Should find path to trap")
    
    def test_path_finding_on_tiny_graphs(self):
        """Test optimal path finding on small graphs"""
        wag = self.create_three_node_linear()
        analyzer = MultiStepAnalyzer(wag)
        
        # Find paths from word1 
        paths = analyzer.find_optimal_paths("word1", max_steps=3, max_paths=5)
        
        self.assertGreater(len(paths), 0, "Should find at least one path")
        
        # Check if path to trap is found
        trap_paths = [p for p in paths if p.path[-1] == "trap"]
        self.assertGreater(len(trap_paths), 0, "Should find path to trap")
        
        if trap_paths:
            best_trap_path = trap_paths[0]
            expected_path = ["word1", "word2", "trap"]
            self.assertEqual(best_trap_path.path, expected_path, "Path should be word1->word2->trap")
            self.assertEqual(best_trap_path.probability, 1.0, "Path probability should be 1.0")
    
    def test_edge_cases_and_robustness(self):
        """Test edge cases and error handling"""
        wag = self.create_single_node_graph()
        analyzer = MultiStepAnalyzer(wag)
        
        # Test with non-existent word
        prob = analyzer.k_step_probability_exact("nonexistent", k=1)
        self.assertEqual(prob, 0.0, "Non-existent word should return 0 probability")
        
        # Test with k=0
        prob_k0 = analyzer.k_step_probability_exact("word", k=0)
        self.assertEqual(prob_k0, 0.0, "k=0 should return 0 probability")
        
        # Test multi-step analysis on non-existent word
        result = analyzer.analyze_multi_step("nonexistent", max_k=2)
        self.assertIsNotNone(result, "Should handle non-existent words gracefully")
    
    def test_probability_consistency(self):
        """Test that probabilities are consistent and bounded"""
        wag = self.create_three_node_star()
        analyzer = MultiStepAnalyzer(wag)
        
        # All probabilities should be between 0 and 1
        for k in range(1, 4):
            prob = analyzer.k_step_probability_exact("center", k=k)
            self.assertGreaterEqual(prob, 0.0, f"Probability at k={k} should be >= 0")
            self.assertLessEqual(prob, 1.0, f"Probability at k={k} should be <= 1")
        
        # Cumulative probability should be monotonically increasing
        cumulative_k1 = analyzer.k_step_probability_cumulative("center", k=1)
        cumulative_k2 = analyzer.k_step_probability_cumulative("center", k=2)
        cumulative_k3 = analyzer.k_step_probability_cumulative("center", k=3)
        
        self.assertLessEqual(cumulative_k1, cumulative_k2, "Cumulative should be non-decreasing")
        self.assertLessEqual(cumulative_k2, cumulative_k3, "Cumulative should be non-decreasing")

class TestTinyGraphPerformance(unittest.TestCase):
    """Performance tests on tiny graphs to ensure algorithms scale"""
    
    def setUp(self):
        self.trap_letters = frozenset(['r', 's', 't'])
    
    def test_analysis_speed_tiny_graphs(self):
        """Ensure analysis completes quickly on tiny graphs"""
        import time
        
        # Create slightly larger test graph (10 nodes)
        graph = {}
        for i in range(10):
            neighbors = {}
            for j in range(min(3, 10-i-1)):  # Connect to next few nodes
                if i + j + 1 < 10:
                    neighbors[f"word{i+j+1}"] = 1.0 / (j + 1)
            graph[f"word{i}"] = neighbors
        
        # Add one trap word
        graph["trap"] = {}
        graph["word9"]["trap"] = 0.5
        
        wag = WordAssociationGraph(graph, self.trap_letters)
        analyzer = MultiStepAnalyzer(wag)
        
        # Time the analysis
        start_time = time.time()
        
        for word in ["word0", "word1", "word2"]:
            result = analyzer.analyze_multi_step(word, max_k=3)
            paths = analyzer.find_optimal_paths(word, max_steps=3, max_paths=3)
        
        elapsed_time = time.time() - start_time
        
        # Should complete in reasonable time (less than 1 second for tiny graph)
        self.assertLess(elapsed_time, 1.0, "Analysis should be fast on tiny graphs")

def create_test_suite():
    """Create comprehensive test suite"""
    suite = unittest.TestSuite()
    
    # Add all test methods from both classes
    for test_class in [TestMultiStepTinyGraphs, TestTinyGraphPerformance]:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite

def run_tests():
    """Run all tests with detailed output"""
    print("🧪 RUNNING MULTI-STEP UNIT TESTS ON TINY GRAPHS")
    print("=" * 60)
    
    # Create and run test suite
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 TEST SUMMARY:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback.split('Exception')[-1].strip()}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)