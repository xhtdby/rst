#!/usr/bin/env python3
"""
RST Trap Analysis - Deep dive into word association patterns

This script performs comprehensive analysis using our integrated datasets:
- Strategic trap word identification
- Pathway analysis for multi-step planning
- Word effectiveness comparison across datasets
- Real-world RST game simulation
"""

import pandas as pd
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from rst_trap_finder.core import WordAssociationGraph

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


class RSTAnalyzer:
    """Advanced RST analysis with multiple datasets."""
    
    def __init__(self):
        self.datasets = self._find_available_datasets()
        self.graphs = {}
        self.results = {}
        
    def _find_available_datasets(self) -> Dict[str, Path]:
        """Find all available datasets for analysis."""
        datasets = {}
        
        # Complete dataset (highest priority)
        complete_path = Path("data/merged/complete_rst_dataset.csv")
        if complete_path.exists():
            datasets["complete"] = complete_path
        
        # Reduced datasets
        reduced_dir = Path("data/reduced")
        if reduced_dir.exists():
            for file in reduced_dir.glob("reduced_rst_dataset_*.csv"):
                # Get most recent by timestamp
                timestamp = file.stem.split("_")[-2:]  # Get date and time
                key = f"reduced_{timestamp[0]}_{timestamp[1]}"
                datasets[key] = file
        
        # Legacy datasets
        legacy_paths = [
            ("conceptnet", "data/processed/edges_conceptnet.csv"),
            ("usf", "data/processed/edges_usf.csv"),
            ("swow", "data/processed/edges_swow_en18.csv"),
        ]
        
        for name, path in legacy_paths:
            path_obj = Path(path)
            if path_obj.exists():
                datasets[name] = path_obj
        
        return datasets
    
    def load_datasets(self, limit: int = 3):
        """Load the most relevant datasets for analysis."""
        print("🔍 Loading Datasets for Analysis")
        print("=" * 40)
        
        # Prioritize complete dataset and most recent reduced
        priority_order = ["complete"]
        
        # Add most recent reduced dataset
        reduced_datasets = [k for k in self.datasets.keys() if k.startswith("reduced_")]
        if reduced_datasets:
            priority_order.append(max(reduced_datasets))  # Most recent by name
        
        # Add individual source datasets for comparison
        for name in ["usf", "swow"]:
            if name in self.datasets:
                priority_order.append(name)
        
        # Load up to limit datasets
        loaded_count = 0
        for dataset_name in priority_order:
            if loaded_count >= limit:
                break
                
            if dataset_name in self.datasets:
                print(f"\n📊 Loading {dataset_name}...")
                try:
                    graph = WordAssociationGraph.from_csv(self.datasets[dataset_name])
                    self.graphs[dataset_name] = graph
                    
                    words = len(graph.get_all_words())
                    edges = sum(len(neighbors) for neighbors in graph.graph.values())
                    print(f"   ✅ {words:,} words, {edges:,} edges")
                    loaded_count += 1
                    
                except Exception as e:
                    print(f"   ❌ Failed to load {dataset_name}: {e}")
        
        print(f"\n✅ Loaded {len(self.graphs)} datasets for analysis")
        return len(self.graphs) > 0
    
    def analyze_top_traps(self, top_k: int = 50):
        """Analyze top trap words across datasets."""
        print(f"\n🎯 Top {top_k} Trap Word Analysis")
        print("=" * 50)
        
        all_results = {}
        
        for dataset_name, graph in self.graphs.items():
            print(f"\n📊 Analyzing {dataset_name}...")
            
            # Get top words
            start_time = time.time()
            top_words = graph.rank_words(top_k=top_k)
            analysis_time = time.time() - start_time
            
            all_results[dataset_name] = {
                "top_words": top_words,
                "analysis_time": analysis_time,
                "total_words": len(graph.get_all_words())
            }
            
            print(f"   ⚡ Analysis time: {analysis_time:.2f}s")
            print(f"   🏆 Top 5 trap words:")
            for i, (word, score) in enumerate(top_words[:5]):
                rst_prob = graph.one_step_rst_probability(word)
                neighbors = len(graph.get_neighbors(word))
                print(f"      {i+1}. {word:15s}: {score:.4f} (RST: {rst_prob:.3f}, N: {neighbors:2d})")
        
        # Cross-dataset comparison
        if len(all_results) > 1:
            self._compare_top_words(all_results, top_k=20)
        
        self.results["top_traps"] = all_results
        return all_results
    
    def _compare_top_words(self, results: Dict, top_k: int = 20):
        """Compare top words across datasets."""
        print(f"\n🔄 Cross-Dataset Comparison (Top {top_k})")
        print("=" * 45)
        
        # Get top words from each dataset
        dataset_words = {}
        for name, data in results.items():
            top_words = [word for word, score in data["top_words"][:top_k]]
            dataset_words[name] = set(top_words)
        
        if len(dataset_words) >= 2:
            dataset_names = list(dataset_words.keys())
            
            # Find overlaps
            for i in range(len(dataset_names)):
                for j in range(i + 1, len(dataset_names)):
                    name1, name2 = dataset_names[i], dataset_names[j]
                    overlap = dataset_words[name1] & dataset_words[name2]
                    total_unique = dataset_words[name1] | dataset_words[name2]
                    
                    overlap_pct = (len(overlap) / len(total_unique)) * 100
                    print(f"📈 {name1} ↔ {name2}: {len(overlap)}/{len(total_unique)} overlap ({overlap_pct:.1f}%)")
                    
                    if overlap:
                        sample_overlap = list(overlap)[:5]
                        print(f"   Common words: {', '.join(sample_overlap)}")
    
    def analyze_rst_targets(self):
        """Analyze RST words as game-ending targets."""
        print(f"\n🎯 RST Target Analysis")
        print("=" * 30)
        
        target_results = {}
        
        for dataset_name, graph in self.graphs.items():
            print(f"\n📊 {dataset_name} RST targets:")
            
            # Analyze RST targets
            rst_targets = graph.analyze_rst_targets(top_k=10)
            
            for letter in ['R', 'S', 'T']:
                targets = rst_targets[letter]
                total_targets = len(targets)
                
                if targets:
                    print(f"   🎯 {letter} targets ({total_targets} total):")
                    for i, (word, incoming) in enumerate(targets[:5]):
                        print(f"      {i+1}. {word:12s}: {incoming:3d} ways to reach")
                else:
                    print(f"   🎯 {letter} targets: None found")
            
            target_results[dataset_name] = rst_targets
        
        self.results["rst_targets"] = target_results
        return target_results
    
    def analyze_strategic_pathways(self, test_words: List[str] = None):
        """Analyze strategic pathways for specific starting words."""
        if test_words is None:
            test_words = ["word", "color", "think", "big", "red", "fast", "good"]
        
        print(f"\n🛤️  Strategic Pathway Analysis")
        print("=" * 40)
        
        pathway_results = {}
        
        for dataset_name, graph in self.graphs.items():
            print(f"\n📊 {dataset_name} pathways:")
            dataset_results = {}
            
            for word in test_words:
                if graph.has_word(word):
                    # Analyze pathways
                    pathway_data = graph.analyze_trap_pathways(word, max_steps=3)
                    
                    rst_prob = graph.one_step_rst_probability(word)
                    neighbors = len(graph.get_neighbors(word))
                    
                    print(f"   🎯 {word:8s}: {rst_prob:.3f} RST prob, {neighbors:2d} neighbors")
                    
                    # Show top pathways
                    stats = pathway_data.get('statistics', {})
                    total_paths = stats.get('total_paths_found', 0)
                    if total_paths > 0:
                        print(f"      🔗 {total_paths} pathways found")
                    
                    dataset_results[word] = {
                        "rst_probability": rst_prob,
                        "neighbors": neighbors,
                        "pathways": total_paths,
                        "pathway_data": pathway_data
                    }
            
            pathway_results[dataset_name] = dataset_results
        
        self.results["pathways"] = pathway_results
        return pathway_results
        """Analyze strategic pathways for specific starting words."""
        if test_words is None:
            test_words = ["start", "color", "think", "run", "big", "red", "fast", "good"]
        
        print(f"\n🛤️  Strategic Pathway Analysis")
        print("=" * 40)
        
        pathway_results = {}
        
        for dataset_name, graph in self.graphs.items():
            print(f"\n📊 {dataset_name} pathways:")
            dataset_results = {}
            
            for word in test_words:
                if graph.has_word(word):
                    # Analyze pathways
                    pathway_data = graph.analyze_trap_pathways(word, max_steps=3)
                    
                    rst_prob = graph.one_step_rst_probability(word)
                    neighbors = len(graph.get_neighbors(word))
                    
                    print(f"   🎯 {word:8s}: {rst_prob:.3f} RST prob, {neighbors:2d} neighbors")
                    
                    # Show top pathways
                    stats = pathway_data.get('statistics', {})
                    total_paths = stats.get('total_paths_found', 0)
                    if total_paths > 0:
                        print(f"      🔗 {total_paths} pathways found")
                    
                    dataset_results[word] = {
                        "rst_probability": rst_prob,
                        "neighbors": neighbors,
                        "pathways": total_paths,
                        "pathway_data": pathway_data
                    }
            
            pathway_results[dataset_name] = dataset_results
        
        self.results["pathways"] = pathway_results
        return pathway_results
    
    def simulate_rst_game(self, starting_word: str = "word", max_turns: int = 10, 
                         use_multistep: bool = True, lookahead_depth: int = 3):
        """Simulate an RST game with strategic recommendations."""
        print(f"\n🎮 RST Game Simulation")
        if use_multistep:
            print("   🧠 Using Multi-Step Strategic Analysis")
        print("=" * 30)
        
        # Use the most complete dataset
        primary_dataset = None
        if "complete" in self.graphs:
            primary_dataset = "complete"
        elif self.graphs:
            primary_dataset = list(self.graphs.keys())[0]
        
        if not primary_dataset:
            print("❌ No datasets loaded for simulation")
            return
        
        graph = self.graphs[primary_dataset]
        print(f"📊 Using dataset: {primary_dataset}")
        print(f"🎯 Starting word: {starting_word}")
        print(f"🎮 Max turns: {max_turns}")
        if use_multistep:
            print(f"🔍 Lookahead depth: {lookahead_depth}")
        
        # Initialize multi-step analyzer if requested
        if use_multistep:
            from rst_trap_finder.multistep import MultiStepAnalyzer
            multistep_analyzer = MultiStepAnalyzer(graph)
        
        current_word = starting_word
        turn = 0
        game_log = []
        
        while turn < max_turns:
            turn += 1
            
            if not graph.has_word(current_word):
                print(f"\n❌ Turn {turn}: Word '{current_word}' not in dataset")
                break
            
            # Get recommendations with multi-step analysis
            if use_multistep:
                recommendations = self._get_multistep_recommendations(
                    graph, multistep_analyzer, current_word, lookahead_depth
                )
            else:
                recommendations = graph.recommend_next_word(current_word, top_k=5)
            
            if not recommendations:
                print(f"\n❌ Turn {turn}: No recommendations for '{current_word}'")
                break
            
            # Show current situation
            rst_prob = graph.one_step_rst_probability(current_word)
            print(f"\n🎯 Turn {turn}: '{current_word}' (RST prob: {rst_prob:.3f})")
            
            # Show top recommendations with strategic analysis
            print("   💡 Recommendations:")
            for i, rec in enumerate(recommendations[:3]):
                if use_multistep and isinstance(rec, dict) and 'multistep_score' in rec:
                    word = rec['word']
                    score = rec['multistep_score']
                    k_step_prob = rec.get('k_step_probability', 0)
                    strategic_score = rec.get('strategic_score', 0)
                    print(f"      {i+1}. {word:12s}: MS:{score:.4f} K-step:{k_step_prob:.3f} Strategic:{strategic_score:.3f}")
                else:
                    word = rec['word'] if isinstance(rec, dict) else rec
                    if isinstance(rec, dict):
                        score = rec.get('score', 0)
                        rst_p = graph.one_step_rst_probability(word)
                        print(f"      {i+1}. {word:12s}: {score:.4f} (RST: {rst_p:.3f})")
                    else:
                        print(f"      {i+1}. {word}")
            
            # Simulate opponent choosing highest-scoring recommendation
            if isinstance(recommendations[0], dict):
                next_word = recommendations[0]['word']
            else:
                next_word = recommendations[0]
                
            game_log.append({
                "turn": turn,
                "current_word": current_word,
                "rst_probability": rst_prob,
                "chosen_word": next_word,
                "recommendations": recommendations[:3],
                "multistep_analysis": use_multistep
            })
            
            # Check if we hit an RST word
            if next_word[0].lower() in {'r', 's', 't'}:
                print(f"\n🎊 SUCCESS! Turn {turn}: Opponent chose '{next_word}' (RST word)")
                print(f"🏆 Game won in {turn} turns!")
                break
            
            current_word = next_word
        
        else:
            print(f"\n⏰ Game ended after {max_turns} turns without hitting RST")
        
        # Game analysis with multi-step insights
        print(f"\n📊 Game Analysis:")
        avg_rst_prob = sum(log['rst_probability'] for log in game_log) / len(game_log) if game_log else 0
        print(f"   📈 Average RST probability: {avg_rst_prob:.3f}")
        print(f"   🔄 Total turns played: {len(game_log)}")
        
        if use_multistep and game_log:
            print(f"   🧠 Multi-step strategy employed")
            print(f"   🔍 Lookahead depth: {lookahead_depth}")
        
        success = len(game_log) > 0 and game_log[-1]["chosen_word"][0].lower() in {'r', 's', 't'}
        self.results["simulation"] = {
            "starting_word": starting_word,
            "game_log": game_log,
            "success": success,
            "turns": len(game_log),
            "multistep_analysis": use_multistep,
            "lookahead_depth": lookahead_depth if use_multistep else None
        }
        
        return game_log
    
    def _get_multistep_recommendations(self, graph, multistep_analyzer, current_word, 
                                     lookahead_depth):
        """Get recommendations enhanced with multi-step strategic analysis."""
        # Get basic recommendations
        basic_recs = graph.recommend_next_word(current_word, top_k=10)
        
        if not basic_recs:
            return []
        
        enhanced_recs = []
        
        for rec in basic_recs:
            word = rec['word']
            
            # Calculate multi-step metrics
            k_step_prob = multistep_analyzer.k_step_probability_cumulative(word, lookahead_depth)
            multistep_result = multistep_analyzer.analyze_multi_step(word, max_k=lookahead_depth)
            
            # Enhanced scoring combining original and multi-step analysis
            original_score = rec['score']
            strategic_score = multistep_result.strategic_score
            
            # Weighted combination favoring multi-step insights
            multistep_score = (
                0.4 * original_score +
                0.4 * strategic_score +
                0.2 * k_step_prob
            )
            
            enhanced_rec = {
                'word': word,
                'score': original_score,
                'multistep_score': multistep_score,
                'k_step_probability': k_step_prob,
                'strategic_score': strategic_score,
                'path_count': multistep_result.path_count,
                'information_gain': multistep_result.max_information_gain
            }
            
            enhanced_recs.append(enhanced_rec)
        
        # Sort by multi-step score
        enhanced_recs.sort(key=lambda x: x['multistep_score'], reverse=True)
        
        return enhanced_recs
        """Simulate an RST game with strategic recommendations."""
        print(f"\n🎮 RST Game Simulation")
        print("=" * 30)
        
        # Use the most complete dataset
        primary_dataset = None
        if "complete" in self.graphs:
            primary_dataset = "complete"
        elif self.graphs:
            primary_dataset = list(self.graphs.keys())[0]
        
        if not primary_dataset:
            print("❌ No datasets loaded for simulation")
            return
        
        graph = self.graphs[primary_dataset]
        print(f"📊 Using dataset: {primary_dataset}")
        print(f"🎯 Starting word: {starting_word}")
        print(f"🎮 Max turns: {max_turns}")
        
        current_word = starting_word
        turn = 0
        game_log = []
        
        while turn < max_turns:
            turn += 1
            
            if not graph.has_word(current_word):
                print(f"\n❌ Turn {turn}: Word '{current_word}' not in dataset")
                break
            
            # Get recommendations
            recommendations = graph.recommend_next_word(current_word, top_k=5)
            
            if not recommendations:
                print(f"\n❌ Turn {turn}: No recommendations for '{current_word}'")
                break
            
            # Show current situation
            rst_prob = graph.one_step_rst_probability(current_word)
            print(f"\n🎯 Turn {turn}: '{current_word}' (RST prob: {rst_prob:.3f})")
            
            # Show top recommendations
            print("   💡 Recommendations:")
            for i, rec in enumerate(recommendations[:3]):
                word = rec['word']
                score = rec['score']
                rst_p = graph.one_step_rst_probability(word)
                print(f"      {i+1}. {word:12s}: {score:.4f} (RST: {rst_p:.3f})")
            
            # Simulate opponent choosing highest-scoring recommendation
            next_word = recommendations[0]['word']
            game_log.append({
                "turn": turn,
                "current_word": current_word,
                "rst_probability": rst_prob,
                "chosen_word": next_word,
                "recommendations": recommendations[:3]
            })
            
            # Check if we hit an RST word
            if next_word[0].lower() in {'r', 's', 't'}:
                print(f"\n🎊 SUCCESS! Turn {turn}: Opponent chose '{next_word}' (RST word)")
                print(f"🏆 Game won in {turn} turns!")
                break
            
            current_word = next_word
        
        else:
            print(f"\n⏰ Game ended after {max_turns} turns without hitting RST")
        
        # Game analysis
        print(f"\n📊 Game Analysis:")
        avg_rst_prob = sum(log['rst_probability'] for log in game_log) / len(game_log) if game_log else 0
        print(f"   📈 Average RST probability: {avg_rst_prob:.3f}")
        print(f"   🔄 Total turns played: {len(game_log)}")
        
        self.results["simulation"] = {
            "starting_word": starting_word,
            "game_log": game_log,
            "success": len(game_log) > 0 and game_log[-1]["chosen_word"][0].lower() in {'r', 's', 't'},
            "turns": len(game_log)
        }
        
        return game_log
    
    def performance_comparison(self):
        """Compare performance across different dataset sizes."""
        print(f"\n⚡ Performance Comparison")
        print("=" * 30)
        
        perf_results = {}
        
        for dataset_name, graph in self.graphs.items():
            words = len(graph.get_all_words())
            edges = sum(len(neighbors) for neighbors in graph.graph.values())
            
            print(f"\n📊 {dataset_name}:")
            print(f"   📏 Size: {words:,} words, {edges:,} edges")
            
            # Time a small ranking operation
            start_time = time.time()
            top_10 = graph.rank_words(top_k=10)
            ranking_time = time.time() - start_time
            
            words_per_second = words / ranking_time if ranking_time > 0 else 0
            
            print(f"   ⚡ Ranking time: {ranking_time:.2f}s")
            print(f"   🚀 Speed: {words_per_second:,.0f} words/second")
            
            perf_results[dataset_name] = {
                "words": words,
                "edges": edges,
                "ranking_time": ranking_time,
                "words_per_second": words_per_second
            }
        
        self.results["performance"] = perf_results
        return perf_results
    
    def save_analysis_report(self):
        """Save comprehensive analysis report."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = Path(f"RST_Analysis_Report_{timestamp}.json")
        
        # Add metadata
        report_data = {
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "datasets_analyzed": list(self.graphs.keys()),
            "analysis_results": self.results
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Analysis report saved: {report_path}")
        return report_path


def main():
    """Run comprehensive RST analysis."""
    print("🚀 RST Trap Analysis - Deep Dive")
    print("=" * 50)
    print("   Analyzing word association patterns for strategic gameplay")
    print()
    
    analyzer = RSTAnalyzer()
    
    # Load datasets
    if not analyzer.load_datasets(limit=3):
        print("❌ No datasets available for analysis")
        return
    
    # Run comprehensive analysis
    print("\n" + "="*60)
    print("🔍 COMPREHENSIVE RST ANALYSIS")
    print("="*60)
    
    # 1. Top trap analysis
    analyzer.analyze_top_traps(top_k=30)
    
    # 2. RST target analysis
    analyzer.analyze_rst_targets()
    
    # 3. Strategic pathways
    analyzer.analyze_strategic_pathways([
        "start", "color", "think", "run", "big", "red", "fast", 
        "good", "work", "time", "love", "life"
    ])
    
    # 4. Game simulation with multi-step analysis
    analyzer.simulate_rst_game("word", max_turns=8, use_multistep=True, lookahead_depth=3)
    
    # 5. Performance comparison
    analyzer.performance_comparison()
    
    # 6. Save report
    analyzer.save_analysis_report()
    
    print(f"\n🎯 Analysis Complete!")
    print(f"   Ready for strategic RST gameplay! 🎮")


if __name__ == "__main__":
    main()