#!/usr/bin/env python3
"""
RST Trap Analysis - Clean, unified analysis framework

This script performs comprehensive analysis using integrated datasets:
- Strategic trap word identification  
- Pathway analysis for multi-step planning
- Word effectiveness comparison across datasets
- Real-world RST game simulation with multi-step strategy
"""

import pandas as pd
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Union
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
    """Advanced RST analysis with multiple datasets and unified simulation."""
    
    def __init__(self):
        self.datasets = self._find_available_datasets()
        self.graphs = {}
        self.results = {}
        
    def _find_available_datasets(self) -> Dict[str, Path]:
        """Find all available datasets for analysis."""
        datasets = {}
        
        # Pruned dataset (preferred - articles/conjunctions removed)
        pruned_path = Path("data/merged/pruned_association_graph.csv")
        if pruned_path.exists():
            datasets["pruned"] = pruned_path
        
        # Complete dataset
        complete_path = Path("data/merged/complete_rst_dataset.csv")
        if complete_path.exists():
            datasets["complete"] = complete_path
            
        # Merged dataset (largest)
        merged_path = Path("data/merged/merged_association_graph.csv")
        if merged_path.exists():
            datasets["merged"] = merged_path
        
        # Individual dataset files
        individual_dir = Path("data/processed")
        if individual_dir.exists():
            for file in individual_dir.glob("edges_*.parquet"):
                name = file.stem.replace("edges_", "")
                datasets[name] = file
                
        return datasets
    
    def load_datasets(self, dataset_names: Optional[List[str]] = None):
        """Load specified datasets or all available ones."""
        if dataset_names is None:
            dataset_names = list(self.datasets.keys())
        
        print(f"🔄 Loading {len(dataset_names)} datasets...")
        
        for name in tqdm(dataset_names, desc="Loading datasets"):
            if name not in self.datasets:
                print(f"⚠️  Dataset '{name}' not found")
                continue
                
            try:
                path = self.datasets[name]
                print(f"📂 Loading {name}: {path}")
                
                # Load different file types
                if path.suffix == '.parquet':
                    df = pd.read_parquet(path)
                    # Convert to CSV format temporarily
                    temp_csv = f"temp_{name}.csv"
                    df.to_csv(temp_csv, index=False)
                    graph = WordAssociationGraph.from_csv(temp_csv)
                    Path(temp_csv).unlink()  # Clean up temp file
                else:
                    graph = WordAssociationGraph.from_csv(path)
                
                self.graphs[name] = graph
                
                words = len(graph.get_all_words())
                edges = sum(len(neighbors) for neighbors in graph.graph.values())
                print(f"   ✅ {name}: {words:,} words, {edges:,} edges")
                
            except Exception as e:
                print(f"   ❌ Error loading {name}: {e}")
        
        print(f"✅ Loaded {len(self.graphs)} datasets successfully")
    
    def analyze_trap_effectiveness(self, top_k: int = 20) -> Dict:
        """Analyze trap word effectiveness across all datasets."""
        print(f"\n🎯 ANALYZING TRAP EFFECTIVENESS (Top {top_k})")
        print("=" * 60)
        
        trap_results = {}
        
        for dataset_name, graph in self.graphs.items():
            print(f"\n📊 Dataset: {dataset_name}")
            
            # Get all words and calculate trap metrics
            all_words = list(graph.get_all_words())
            word_scores = []
            
            for word in tqdm(all_words[:1000], desc=f"Analyzing {dataset_name}"):  # Limit for performance
                try:
                    rst_prob = graph.one_step_rst_probability(word)
                    k_step_prob = graph.k_step_rst_probability(word, k=2)
                    
                    # Get neighbors count as connectivity metric
                    neighbors = graph.get_neighbors(word)
                    connectivity = len(neighbors)
                    
                    word_scores.append({
                        'word': word,
                        'rst_prob_1step': rst_prob,
                        'rst_prob_kstep': k_step_prob,
                        'connectivity': connectivity,
                        'is_rst': word[0].lower() in graph.trap_letters,
                        'combined_score': rst_prob * 0.7 + k_step_prob * 0.3
                    })
                except Exception as e:
                    continue
            
            # Sort by effectiveness
            word_scores.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # Display top results
            print(f"🏆 Top {top_k} trap words in {dataset_name}:")
            print(f"{'Word':<15} {'1-step':<8} {'K-step':<8} {'Connect':<8} {'RST':<5} {'Combined'}")
            print("-" * 65)
            
            for i, result in enumerate(word_scores[:top_k]):
                rst_marker = "🎯" if result['is_rst'] else "  "
                print(f"{result['word']:<15} {result['rst_prob_1step']:<8.3f} "
                      f"{result['rst_prob_kstep']:<8.3f} {result['connectivity']:<8} "
                      f"{rst_marker:<5} {result['combined_score']:.3f}")
            
            trap_results[dataset_name] = word_scores[:top_k]
        
        self.results["trap_analysis"] = trap_results
        return trap_results
    
    def analyze_pathways(self, test_words: List[str] = None) -> Dict:
        """Analyze multi-step pathways for strategic planning."""
        if test_words is None:
            test_words = ["word", "computer", "language", "science", "game"]
        
        print(f"\n🛤️  PATHWAY ANALYSIS")
        print("=" * 40)
        
        pathway_results = {}
        
        for dataset_name, graph in self.graphs.items():
            print(f"\n📊 Dataset: {dataset_name}")
            
            # Initialize multi-step analyzer
            try:
                from rst_trap_finder.multistep import MultiStepAnalyzer
                analyzer = MultiStepAnalyzer(graph)
                
                dataset_results = {}
                
                for word in test_words:
                    if word not in graph.graph:
                        continue
                    
                    print(f"🔍 Analyzing pathways from '{word}'")
                    
                    # Multi-step analysis
                    result = analyzer.analyze_multi_step(word, max_k=3)
                    
                    # Find optimal paths
                    paths = analyzer.find_optimal_paths(word, max_steps=3, max_paths=5)
                    
                    dataset_results[word] = {
                        'k_step_probabilities': result.k_step_probabilities,
                        'strategic_score': result.strategic_score,
                        'optimal_paths': [(p.path, p.probability) for p in paths],
                        'multistep_score': result.strategic_score
                    }
                    
                    print(f"   📈 Strategic score: {result.strategic_score:.4f}")
                    print(f"   🎯 Found {len(paths)} optimal paths")
                
                pathway_results[dataset_name] = dataset_results
                
            except ImportError:
                print(f"   ⚠️  Multi-step analysis not available for {dataset_name}")
                continue
            except Exception as e:
                print(f"   ❌ Error in pathway analysis: {e}")
                continue
        
        self.results["pathways"] = pathway_results
        return pathway_results
    
    def simulate_strategic_game(self, starting_word: str = "word", target_word: str = None,
                              max_turns: int = 15, strategy: str = "multistep") -> Dict:
        """Unified game simulation with multiple strategies."""
        print(f"\n🎮 STRATEGIC GAME SIMULATION")
        print("=" * 50)
        
        # Use best available dataset
        primary_dataset = self._get_primary_dataset()
        if not primary_dataset:
            print("❌ No datasets loaded for simulation")
            return {}
        
        graph = self.graphs[primary_dataset]
        print(f"📊 Dataset: {primary_dataset}")
        print(f"🎯 Starting: '{starting_word}' → Target: '{target_word or 'RST word'}'")
        print(f"🎮 Max turns: {max_turns}")
        print(f"🧠 Strategy: {strategy}")
        
        # Initialize strategy-specific components
        if strategy == "multistep":
            try:
                from rst_trap_finder.multistep import MultiStepAnalyzer
                analyzer = MultiStepAnalyzer(graph)
                print("   ✅ Multi-step analyzer loaded")
            except ImportError:
                print("   ⚠️  Falling back to basic strategy")
                strategy = "basic"
                analyzer = None
        else:
            analyzer = None
        
        # Game state
        current_word = starting_word
        path = [starting_word]
        turn = 0
        game_won = False
        
        # Game loop
        while turn < max_turns and not game_won:
            turn += 1
            
            if current_word not in graph.graph:
                print(f"\n❌ Turn {turn}: '{current_word}' not in dataset")
                break
            
            # Get strategic recommendations
            if strategy == "multistep" and analyzer:
                recommendations = self._get_multistep_recommendations(
                    graph, analyzer, current_word, lookahead=3
                )
            else:
                recommendations = self._get_basic_recommendations(graph, current_word)
            
            if not recommendations:
                print(f"\n💀 Turn {turn}: No moves available from '{current_word}'")
                break
            
            # Display current state
            rst_prob = graph.one_step_rst_probability(current_word)
            is_rst = current_word[0].lower() in graph.trap_letters
            rst_indicator = "🎯" if is_rst else "  "
            
            print(f"\n{rst_indicator} Turn {turn}: '{current_word}' (RST prob: {rst_prob:.3f})")
            
            # Show top recommendations
            print("   💡 Strategic options:")
            for i, rec in enumerate(recommendations[:3]):
                word = rec['word']
                score = rec.get('score', 0)
                is_target = word == target_word if target_word else False
                target_marker = "🏁" if is_target else "  "
                rst_marker = "🎯" if word[0].lower() in graph.trap_letters else "  "
                
                extra_info = ""
                if 'multistep_score' in rec:
                    extra_info = f" MS:{rec['multistep_score']:.3f}"
                if 'k_step_prob' in rec:
                    extra_info += f" K:{rec['k_step_prob']:.3f}"
                
                print(f"      {i+1}.{target_marker}{rst_marker} {word:<15}: {score:.4f}{extra_info}")
            
            # Choose move (simulate opponent choosing best option)
            chosen_move = recommendations[0]['word']
            current_word = chosen_move
            path.append(current_word)
            
            # Check win conditions
            if target_word and current_word == target_word:
                print(f"\n🎉 TARGET REACHED! Won in {turn} turns")
                game_won = True
            elif current_word[0].lower() in graph.trap_letters:
                print(f"\n🎯 RST WORD HIT! '{current_word}' - Strategic success!")
                game_won = True
        
        # Game summary
        print(f"\n📊 GAME SUMMARY:")
        print(f"   🛤️  Path: {' → '.join(path)}")
        print(f"   🎮 Turns: {turn}")
        print(f"   🏆 Result: {'Victory' if game_won else 'Draw/Timeout'}")
        print(f"   📈 Success rate: {1.0 if game_won else 0.0}")
        
        # Store results
        simulation_result = {
            "starting_word": starting_word,
            "target_word": target_word,
            "path": path,
            "turns": turn,
            "max_turns": max_turns,
            "strategy": strategy,
            "game_won": game_won,
            "dataset_used": primary_dataset
        }
        
        self.results["simulation"] = simulation_result
        return simulation_result
    
    def _get_primary_dataset(self) -> Optional[str]:
        """Get the best available dataset for analysis."""
        priority_order = ["pruned", "merged", "complete"]
        for dataset in priority_order:
            if dataset in self.graphs:
                return dataset
        return list(self.graphs.keys())[0] if self.graphs else None
    
    def _get_multistep_recommendations(self, graph, analyzer, current_word, lookahead=3):
        """Get recommendations enhanced with multi-step analysis."""
        basic_recs = graph.recommend_next_word(current_word, top_k=10)
        
        if not basic_recs:
            return []
        
        enhanced_recs = []
        
        for rec in basic_recs:
            word = rec['word']
            
            try:
                # Multi-step analysis
                k_step_prob = analyzer.k_step_probability_cumulative(word, lookahead)
                multistep_result = analyzer.analyze_multi_step(word, max_k=lookahead)
                
                # Enhanced scoring
                original_score = rec['score']
                strategic_score = multistep_result.strategic_score
                
                # Combined score with multi-step weighting
                combined_score = (original_score * 0.4 + 
                                strategic_score * 0.4 + 
                                k_step_prob * 0.2)
                
                enhanced_recs.append({
                    'word': word,
                    'score': combined_score,
                    'original_score': original_score,
                    'multistep_score': strategic_score,
                    'k_step_prob': k_step_prob
                })
                
            except Exception:
                # Fallback to basic recommendation
                enhanced_recs.append(rec)
        
        # Sort by combined score
        enhanced_recs.sort(key=lambda x: x.get('score', 0), reverse=True)
        return enhanced_recs
    
    def _get_basic_recommendations(self, graph, current_word):
        """Get basic recommendations without multi-step analysis."""
        return graph.recommend_next_word(current_word, top_k=10)
    
    def export_results(self, output_path: str = "rst_analysis_results.json"):
        """Export all analysis results to JSON."""
        print(f"\n💾 EXPORTING RESULTS to {output_path}")
        
        # Convert Path objects to strings for JSON serialization
        serializable_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                serializable_results[key] = self._make_serializable(value)
            else:
                serializable_results[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"✅ Results exported successfully")
    
    def _make_serializable(self, obj):
        """Make object JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj

def main():
    """Run comprehensive RST analysis."""
    print("🚀 RST TRAP ANALYSIS - Comprehensive Framework")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = RSTAnalyzer()
    
    # Load datasets
    print(f"\n📂 Available datasets: {list(analyzer.datasets.keys())}")
    analyzer.load_datasets(["pruned"])  # Load just the pruned dataset for speed
    
    if not analyzer.graphs:
        print("❌ No datasets loaded. Exiting.")
        return
    
    # Run analyses
    print("\n🎯 STARTING COMPREHENSIVE ANALYSIS")
    print("=" * 50)
    
    # 1. Trap effectiveness analysis
    trap_results = analyzer.analyze_trap_effectiveness(top_k=15)
    
    # 2. Pathway analysis
    pathway_results = analyzer.analyze_pathways(["word", "computer", "science"])
    
    # 3. Strategic game simulation
    simulation_result = analyzer.simulate_strategic_game(
        starting_word="computer",
        target_word="science", 
        max_turns=10,
        strategy="multistep"
    )
    
    # 4. Export results
    analyzer.export_results("comprehensive_rst_analysis.json")
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print("📊 Results summary:")
    print(f"   🎯 Analyzed {len(analyzer.graphs)} datasets")
    print(f"   📈 Found trap patterns across {sum(len(r) for r in trap_results.values())} words")
    print(f"   🛤️  Analyzed pathways for {len(pathway_results.get(list(pathway_results.keys())[0], {}) if pathway_results else {})} test words")
    print(f"   🎮 Simulated strategic game with {simulation_result.get('turns', 0)} turns")

if __name__ == "__main__":
    main()