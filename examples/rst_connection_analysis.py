#!/usr/bin/env python3
"""
RST Connection Analysis - Find words with most RST connections (original goal)

Note: Example/demo script. Not part of the core package API.
"""

import sys
from pathlib import Path
from collections import Counter, defaultdict

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from rst_trap_finder.core import WordAssociationGraph

class RSTConnectionAnalyzer:
    """Analyze which words have the most connections to RST words"""
    
    def __init__(self, graph_path):
        print(f"🔄 Loading word association graph from {graph_path}")
        self.graph = WordAssociationGraph.from_csv(graph_path)
        
        words = len(self.graph.get_all_words())
        edges = sum(len(neighbors) for neighbors in self.graph.graph.values())
        print(f"📊 Loaded graph: {words:,} words, {edges:,} edges")
        
        # Define RST words (function words that are traps)
        self.rst_words = set(["the", "and", "of", "to", "a", "in", "is", "it", "you", "that", 
                             "he", "was", "for", "on", "are", "as", "with", "his", "they", "i",
                             "be", "have", "not", "will", "from", "they", "we", "she", "or",
                             "an", "do", "been", "this", "but", "by", "at", "can", "would"])
        
        # Filter to only RST words that exist in our graph
        self.available_rst = set(word for word in self.rst_words if word in self.graph.graph)
        print(f"🎯 Available RST words in graph: {len(self.available_rst)}")
        print(f"   RST words: {', '.join(sorted(list(self.available_rst))[:10])}...")
    
    def analyze_rst_connections(self):
        """Find words with most connections TO rst words (trap-prone words)"""
        print(f"\n🔍 ANALYZING RST CONNECTIONS")
        print("=" * 80)
        
        # Count how many RST words each word connects TO
        rst_connection_counts = defaultdict(set)
        total_connections = defaultdict(int)
        
        print(f"📊 Scanning {len(self.graph.get_all_words()):,} words for RST connections...")
        
        for word in self.graph.get_all_words():
            neighbors = self.graph.get_neighbors(word)
            total_connections[word] = len(neighbors)
            
            # Count RST words this word connects to
            for neighbor in neighbors:
                if neighbor in self.available_rst:
                    rst_connection_counts[word].add(neighbor)
        
        # Convert to counts
        rst_counts = {word: len(rst_set) for word, rst_set in rst_connection_counts.items()}
        
        # Sort by RST connection count
        sorted_by_rst = sorted(rst_counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 TOP WORDS BY RST CONNECTIONS (trap-prone words):")
        print(f"{'Word':<20} {'RST Connections':<15} {'Total Connections':<18} {'RST %':<8} {'RST Words Connected To'}")
        print("-" * 100)
        
        for i, (word, rst_count) in enumerate(sorted_by_rst[:25]):
            if rst_count == 0:
                break
                
            total_conn = total_connections[word]
            rst_percent = (rst_count / total_conn * 100) if total_conn > 0 else 0
            rst_words_connected = ', '.join(sorted(list(rst_connection_counts[word])))
            
            print(f"{word:<20} {rst_count:<15} {total_conn:<18} {rst_percent:<7.1f}% {rst_words_connected}")
        
        return sorted_by_rst
    
    def analyze_rst_word_reach(self):
        """Analyze how well-connected each RST word is (trap strength)"""
        print(f"\n🎯 RST WORD REACH ANALYSIS (trap strength)")
        print("=" * 80)
        
        rst_stats = []
        
        for rst_word in sorted(self.available_rst):
            neighbors = self.graph.get_neighbors(rst_word)
            
            # Count how many words connect TO this RST word
            incoming_connections = 0
            for word in self.graph.get_all_words():
                if rst_word in self.graph.get_neighbors(word):
                    incoming_connections += 1
            
            # Count other RST words this connects to
            rst_neighbors = sum(1 for neighbor in neighbors if neighbor in self.available_rst)
            
            rst_stats.append({
                'word': rst_word,
                'outgoing': len(neighbors),
                'incoming': incoming_connections,
                'rst_neighbors': rst_neighbors,
                'trap_score': incoming_connections * (1 + rst_neighbors)  # More incoming + RST cycling = stronger trap
            })
        
        # Sort by trap strength
        rst_stats.sort(key=lambda x: x['trap_score'], reverse=True)
        
        print(f"{'RST Word':<15} {'Outgoing':<10} {'Incoming':<10} {'RST Neighbors':<15} {'Trap Score':<12} {'Analysis'}")
        print("-" * 100)
        
        for stat in rst_stats:
            analysis = ""
            if stat['trap_score'] > 100:
                analysis = "🚨 SUPER TRAP"
            elif stat['trap_score'] > 50:
                analysis = "⚠️ STRONG TRAP"
            elif stat['trap_score'] > 10:
                analysis = "⚖️ MODERATE TRAP"
            elif stat['outgoing'] == 0:
                analysis = "💀 DEAD END"
            else:
                analysis = "✅ WEAK TRAP"
            
            print(f"{stat['word']:<15} {stat['outgoing']:<10} {stat['incoming']:<10} {stat['rst_neighbors']:<15} {stat['trap_score']:<12} {analysis}")
        
        return rst_stats
    
    def find_rst_clustering(self):
        """Find clusters of words that heavily connect to RST words"""
        print(f"\n🕸️  RST CLUSTERING ANALYSIS")
        print("=" * 80)
        
        # Find words that connect to multiple RST words
        multi_rst_words = []
        
        for word in self.graph.get_all_words():
            if word in self.available_rst:
                continue  # Skip RST words themselves
                
            neighbors = self.graph.get_neighbors(word)
            rst_connections = [n for n in neighbors if n in self.available_rst]
            
            if len(rst_connections) >= 2:  # Connects to 2+ RST words
                total_weight = sum(neighbors[rst] for rst in rst_connections)
                multi_rst_words.append({
                    'word': word,
                    'rst_count': len(rst_connections),
                    'rst_words': rst_connections,
                    'total_rst_weight': total_weight,
                    'total_connections': len(neighbors)
                })
        
        # Sort by RST connection strength
        multi_rst_words.sort(key=lambda x: (x['rst_count'], x['total_rst_weight']), reverse=True)
        
        print(f"🎯 Words connecting to multiple RST words (RST cluster centers):")
        print(f"{'Word':<20} {'RST Count':<12} {'Total Weight':<15} {'Total Conn':<12} {'RST Words'}")
        print("-" * 100)
        
        for item in multi_rst_words[:20]:
            rst_list = ', '.join(item['rst_words'])
            print(f"{item['word']:<20} {item['rst_count']:<12} {item['total_rst_weight']:<15.3f} {item['total_connections']:<12} {rst_list}")
        
        return multi_rst_words
    
    def find_escape_routes(self):
        """Find words that can help escape RST traps"""
        print(f"\n🚪 RST ESCAPE ROUTE ANALYSIS")
        print("=" * 80)
        
        escape_words = []
        
        for word in self.graph.get_all_words():
            if word in self.available_rst:
                continue
                
            neighbors = self.graph.get_neighbors(word)
            
            # Count RST neighbors vs non-RST neighbors
            rst_neighbors = sum(1 for n in neighbors if n in self.available_rst)
            non_rst_neighbors = len(neighbors) - rst_neighbors
            
            # Good escape words: connect to RST but mostly to non-RST
            if rst_neighbors > 0 and non_rst_neighbors > rst_neighbors * 3:  # 3:1 ratio
                escape_score = non_rst_neighbors / (rst_neighbors + 1)  # Higher = better escape
                
                escape_words.append({
                    'word': word,
                    'rst_neighbors': rst_neighbors,
                    'non_rst_neighbors': non_rst_neighbors,
                    'escape_score': escape_score,
                    'total_connections': len(neighbors)
                })
        
        escape_words.sort(key=lambda x: x['escape_score'], reverse=True)
        
        print(f"🚪 Best escape route words (connect to RST but mostly to non-RST):")
        print(f"{'Word':<20} {'RST Conn':<10} {'Non-RST Conn':<15} {'Escape Score':<15} {'Total Conn'}")
        print("-" * 80)
        
        for item in escape_words[:15]:
            print(f"{item['word']:<20} {item['rst_neighbors']:<10} {item['non_rst_neighbors']:<15} {item['escape_score']:<15.2f} {item['total_connections']}")
        
        return escape_words

def main():
    # Use the pruned dataset (articles and conjunctions removed)
    dataset_path = "data/merged/pruned_association_graph.csv"
    
    analyzer = RSTConnectionAnalyzer(dataset_path)
    
    print(f"\n{'='*20} RST CONNECTION ANALYSIS {'='*20}")
    print(f"ORIGINAL GOAL: Find words with most RST connections")
    print(f"{'='*60}")
    
    # Core analysis: words with most RST connections
    rst_connected_words = analyzer.analyze_rst_connections()
    
    # RST word strength analysis
    rst_stats = analyzer.analyze_rst_word_reach()
    
    # Find RST clusters
    rst_clusters = analyzer.find_rst_clustering()
    
    # Find escape routes
    escape_routes = analyzer.find_escape_routes()
    
    # Summary insights
    print(f"\n🎯 KEY INSIGHTS:")
    print("=" * 50)
    
    if rst_connected_words:
        top_word, top_count = rst_connected_words[0]
        print(f"🏆 Most RST-connected word: '{top_word}' ({top_count} RST connections)")
    
    if rst_stats:
        strongest_trap = rst_stats[0]
        print(f"🚨 Strongest RST trap: '{strongest_trap['word']}' (trap score: {strongest_trap['trap_score']})")
    
    if rst_clusters:
        cluster_center = rst_clusters[0]
        print(f"🕸️  RST cluster center: '{cluster_center['word']}' (connects to {cluster_center['rst_count']} RST words)")
    
    if escape_routes:
        best_escape = escape_routes[0]
        print(f"🚪 Best escape word: '{best_escape['word']}' (escape score: {best_escape['escape_score']:.2f})")

if __name__ == "__main__":
    main()
