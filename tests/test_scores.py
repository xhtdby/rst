"""Tests for the core RST Trap Finder functionality."""
import pytest
from rst_trap_finder import WordAssociationGraph, TRAP_LETTERS


def make_test_graph():
    """Create a simple test graph."""
    return WordAssociationGraph({
        "a": {"red": 1.0, "blue": 1.0},
        "red": {"stop": 1.0},
        "blue": {"go": 1.0},
    })


def test_basic_scores():
    """Test basic scoring functions."""
    graph = make_test_graph()
    
    # Test one-step probability
    assert graph.one_step_rst_probability("a") == 0.5
    
    # Test escape hardness 
    assert graph.escape_hardness("a", 0.05) == 0.5
    
    # Test k-step probability
    # From 'a': 50% to 'red' (trap), 50% to 'blue' -> 'go' (not trap)
    # So 2-step probability should be 0.5
    assert graph.k_step_rst_probability("a", 2) == 0.5


def test_pagerank_normalized():
    """Test that PageRank scores are properly normalized."""
    graph = make_test_graph()
    pr = graph.biased_pagerank(trap_bias=1.5, max_iterations=10)
    
    assert all(v >= 0 for v in pr.values())
    assert pytest.approx(sum(pr.values()), rel=1e-6) == 1.0


def test_word_analysis():
    """Test detailed word analysis."""
    graph = make_test_graph()
    analysis = graph.get_word_analysis("a")
    
    assert analysis['word'] == "a"
    assert 'one_step_probability' in analysis
    assert 'escape_hardness' in analysis
    assert 'pagerank_score' in analysis
    assert 'composite_score' in analysis
    assert analysis['neighbor_count'] == 2


def test_recommendations():
    """Test word recommendations."""
    graph = make_test_graph()
    recommendations = graph.recommend_next_word("a")
    
    assert len(recommendations) == 2
    assert all('word' in rec for rec in recommendations)
    assert all('score' in rec for rec in recommendations)


def test_ranking():
    """Test word ranking."""
    graph = make_test_graph()
    rankings = graph.rank_words(top_k=2)
    
    assert len(rankings) <= 2
    assert all(isinstance(item, tuple) for item in rankings)
    assert all(len(item) == 2 for item in rankings)
    
    # Check that rankings are sorted by score (descending)
    if len(rankings) > 1:
        assert rankings[0][1] >= rankings[1][1]


def test_has_word():
    """Test word existence checking."""
    graph = make_test_graph()
    
    assert graph.has_word("a")
    assert graph.has_word("red")
    assert not graph.has_word("nonexistent")


def test_neighbors():
    """Test neighbor retrieval."""
    graph = make_test_graph()
    neighbors = graph.get_neighbors("a")
    
    expected = {"red": 1.0, "blue": 1.0}
    assert neighbors == expected


def test_all_words():
    """Test getting all words in graph."""
    graph = make_test_graph()
    all_words = graph.get_all_words()
    
    expected = {"a", "red", "blue", "stop", "go"}
    assert all_words == expected
