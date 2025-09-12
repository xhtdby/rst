#!/usr/bin/env python3
"""
RST Trap Finder - Comprehensive Test Framework

This module provides an intuitive test system that can work with both sample
and full datasets, enabling rapid development and validation.

Key Features:
- Sample vs Full dataset testing
- Configurable test parameters  
- Performance benchmarking
- Result validation and comparison
- Progress tracking for long tests
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import sys

# Add src to path for testing
sys.path.append(str(Path(__file__).parent / 'src'))

from rst_trap_finder.core import WordAssociationGraph


class TestDataManager:
    """Manages test datasets and provides standardized access."""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("data/processed")
        self.datasets = self._discover_datasets()
    
    def _discover_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Discover available datasets and their properties."""
        datasets = {}
        
        # Check for ConceptNet (our primary dataset)
        if (self.data_dir / "edges_conceptnet.csv").exists():
            datasets["conceptnet"] = {
                "path": self.data_dir / "edges_conceptnet.csv",
                "name": "ConceptNet 5.7",
                "size": "large",
                "description": "ConceptNet 5.7 English associations (500K edges, 189K words)",
                "type": "primary"
            }
        
        # Check for merged dataset
        merged_path = Path("data/merged/merged_association_graph.csv")
        if merged_path.exists():
            datasets["merged"] = {
                "path": merged_path,
                "name": "Merged Dataset",
                "size": "large", 
                "description": "Combined word association database",
                "type": "primary"
            }
        
        # USF is currently just a small placeholder/test dataset
        if (self.data_dir / "edges_usf.csv").exists():
            datasets["usf_placeholder"] = {
                "path": self.data_dir / "edges_usf.csv",
                "name": "USF (Placeholder)",
                "size": "tiny",
                "description": "Small test dataset (14 words) - NOT for real analysis",
                "type": "test_only"
            }
        
        if (self.data_dir / "USF_AppendixA.csv").exists():
            datasets["usf_raw_placeholder"] = {
                "path": self.data_dir / "USF_AppendixA.csv",
                "name": "USF Raw (Placeholder)",
                "size": "tiny", 
                "description": "Original format placeholder - NOT for real analysis",
                "type": "test_only"
            }
        
        return datasets
    
    def get_dataset(self, name: str) -> Optional[Path]:
        """Get path to a specific dataset."""
        return self.datasets.get(name, {}).get("path")
    
    def list_datasets(self) -> Dict[str, Dict[str, Any]]:
        """List all available datasets."""
        return self.datasets
    
    def get_sample_dataset(self) -> Path:
        """Get the best small dataset for quick testing."""
        # Use USF placeholder for quick tests only
        if "usf_placeholder" in self.datasets:
            return self.datasets["usf_placeholder"]["path"]
        elif "usf_raw_placeholder" in self.datasets:
            return self.datasets["usf_raw_placeholder"]["path"]
        else:
            # If no test dataset, use ConceptNet (but warn it's slow)
            return self.get_full_dataset()
    
    def get_full_dataset(self) -> Path:
        """Get the largest available dataset for real analysis."""
        # Prefer merged dataset, then ConceptNet
        if "merged" in self.datasets:
            return self.datasets["merged"]["path"]
        elif "conceptnet" in self.datasets:
            return self.datasets["conceptnet"]["path"]
        else:
            # Fallback to any available
            primary_datasets = [d for d in self.datasets.values() if d.get("type") == "primary"]
            if primary_datasets:
                return primary_datasets[0]["path"]
            else:
                return list(self.datasets.values())[0]["path"]


class TestRunner:
    """Runs standardized tests with configurable parameters."""
    
    def __init__(self, data_manager: TestDataManager):
        self.data_manager = data_manager
        self.results = {}
    
    def run_basic_functionality_test(self, dataset_name: str = "sample") -> Dict[str, Any]:
        """Test basic RST functionality."""
        print(f"\n🧪 Running Basic Functionality Test ({dataset_name})")
        print("=" * 50)
        
        # Get dataset
        if dataset_name == "sample":
            dataset_path = self.data_manager.get_sample_dataset()
        elif dataset_name == "full":
            dataset_path = self.data_manager.get_full_dataset()
        else:
            dataset_path = self.data_manager.get_dataset(dataset_name)
        
        if not dataset_path:
            return {"error": f"Dataset '{dataset_name}' not found"}
        
        start_time = time.time()
        
        # Load graph
        print(f"📊 Loading dataset: {dataset_path.name}")
        graph = WordAssociationGraph.from_csv(dataset_path)
        load_time = time.time() - start_time
        
        # Basic metrics
        all_words = graph.get_all_words()
        total_words = len(all_words)
        total_edges = sum(len(neighbors) for neighbors in graph.graph.values())
        
        print(f"✅ Loaded: {total_words:,} words, {total_edges:,} edges ({load_time:.2f}s)")
        
        # Test core functions
        print("🔍 Testing core functions...")
        
        # Get some test words
        test_words = list(graph.graph.keys())[:min(5, len(graph.graph))]
        
        results = {
            "dataset": dataset_path.name,
            "load_time": load_time,
            "total_words": total_words,
            "total_edges": total_edges,
            "test_results": {}
        }
        
        for word in test_words:
            # Test basic functions
            neighbors = graph.get_neighbors(word)
            one_step_prob = graph.one_step_rst_probability(word)
            escape_hard = graph.escape_hardness(word)
            
            results["test_results"][word] = {
                "neighbors": len(neighbors),
                "one_step_rst_prob": one_step_prob,
                "escape_hardness": escape_hard
            }
            
            print(f"  {word:12s}: {len(neighbors):2d} neighbors, "
                  f"RST prob: {one_step_prob:.3f}, hardness: {escape_hard:.3f}")
        
        # Test ranking with more comprehensive results
        print("🏆 Testing word ranking...")
        rank_start = time.time()
        
        # Get different ranking sizes based on dataset size
        if total_words < 100:
            top_k = min(10, total_words)
            extended_k = min(25, total_words)
        elif total_words < 10000:
            top_k = 20
            extended_k = 50
        else:
            top_k = 25
            extended_k = 100
        
        top_words = graph.rank_words(top_k=extended_k)
        rank_time = time.time() - rank_start
        
        results["ranking"] = {
            "time": rank_time,
            "top_k": top_k,
            "extended_k": extended_k,
            "top_words": [(word, float(score)) for word, score in top_words]
        }
        
        print(f"   Ranking time: {rank_time:.2f}s")
        print(f"\n   🎯 Top {top_k} Most Effective Trap Words:")
        print("   " + "="*60)
        print("   Rank  Word              Score    RST Prob  Neighbors")
        print("   " + "-"*60)
        
        for i, (word, score) in enumerate(top_words[:top_k], 1):
            # Get additional details for top words
            neighbors = len(graph.get_neighbors(word))
            rst_prob = graph.one_step_rst_probability(word)
            
            print(f"   {i:4d}  {word:15s}  {score:7.4f}  {rst_prob:8.3f}  {neighbors:9d}")
        
        # Show score distribution summary
        all_scores = [score for _, score in top_words]
        if len(all_scores) > 10:
            print(f"\n   📊 Score Distribution (Top {extended_k}):")
            print(f"      Highest: {max(all_scores):.4f}")
            print(f"      Median:  {all_scores[len(all_scores)//2]:.4f}")
            print(f"      Lowest:  {min(all_scores):.4f}")
            print(f"      Range:   {max(all_scores) - min(all_scores):.4f}")
            
            # Show score tiers
            high_tier = [s for s in all_scores if s >= max(all_scores) - 0.001]
            mid_tier = [s for s in all_scores if 0.5 <= s < max(all_scores) - 0.001]
            low_tier = [s for s in all_scores if s < 0.5]
            
            print(f"      Score Tiers:")
            print(f"        High (≥{max(all_scores)-0.001:.3f}): {len(high_tier)} words")
            print(f"        Mid  (0.5-{max(all_scores)-0.001:.3f}): {len(mid_tier)} words") 
            print(f"        Low  (<0.5): {len(low_tier)} words")
        
        # Test pathway analysis
        if test_words:
            print("🛤️  Testing pathway analysis...")
            pathway_start = time.time()
            pathways = graph.analyze_trap_pathways(test_words[0], max_steps=2)
            pathway_time = time.time() - pathway_start
            
            results["pathways"] = {
                "time": pathway_time,
                "test_word": test_words[0],
                "direct_rst_prob": pathways["statistics"]["direct_rst_probability"],
                "total_paths": pathways["statistics"]["total_paths_found"]
            }
            
            print(f"   Pathway analysis time: {pathway_time:.2f}s")
            print(f"   Direct RST probability: {pathways['statistics']['direct_rst_probability']:.3f}")
        
        total_time = time.time() - start_time
        results["total_time"] = total_time
        
        print(f"\n✅ Test completed in {total_time:.2f}s")
        return results
    
    def run_performance_benchmark(self, dataset_sizes: List[str] = None) -> Dict[str, Any]:
        """Run performance benchmarks across different dataset sizes."""
        if dataset_sizes is None:
            dataset_sizes = ["sample", "full"]
        
        print(f"\n⚡ Running Performance Benchmark")
        print("=" * 50)
        
        benchmark_results = {}
        
        for size in dataset_sizes:
            if size == "sample":
                datasets_to_test = ["usf", "usf_raw"] 
            else:
                datasets_to_test = ["conceptnet"]
            
            for dataset_name in datasets_to_test:
                dataset_path = self.data_manager.get_dataset(dataset_name)
                if not dataset_path or not dataset_path.exists():
                    continue
                
                print(f"\n📊 Benchmarking: {dataset_name}")
                result = self.run_basic_functionality_test(dataset_name)
                benchmark_results[dataset_name] = result
        
        # Summary
        print(f"\n📈 Performance Summary:")
        for name, result in benchmark_results.items():
            if "error" not in result:
                print(f"  {name:12s}: {result['total_words']:6,} words, "
                      f"{result['total_time']:6.2f}s total, "
                      f"{result['ranking']['time']:6.2f}s ranking")
        
        return benchmark_results
    
    def run_correctness_validation(self) -> Dict[str, Any]:
        """Validate correctness of RST analysis algorithms."""
        print(f"\n✅ Running Correctness Validation")
        print("=" * 50)
        
        # Use sample data for validation
        dataset_path = self.data_manager.get_sample_dataset()
        graph = WordAssociationGraph.from_csv(dataset_path)
        
        validation_results = {
            "tests_passed": 0,
            "tests_failed": 0,
            "details": []
        }
        
        # Test 1: RST probability bounds
        print("🔍 Test 1: RST probability bounds (0 ≤ p ≤ 1)")
        test_words = list(graph.graph.keys())[:10]
        
        for word in test_words:
            prob = graph.one_step_rst_probability(word)
            if 0 <= prob <= 1:
                validation_results["tests_passed"] += 1
            else:
                validation_results["tests_failed"] += 1
                validation_results["details"].append(
                    f"FAIL: {word} has invalid probability {prob}"
                )
        
        # Test 2: Escape hardness bounds
        print("🔍 Test 2: Escape hardness bounds (0 ≤ h ≤ 1)")
        for word in test_words:
            hardness = graph.escape_hardness(word)
            if 0 <= hardness <= 1:
                validation_results["tests_passed"] += 1
            else:
                validation_results["tests_failed"] += 1
                validation_results["details"].append(
                    f"FAIL: {word} has invalid hardness {hardness}"
                )
        
        # Test 3: Graph consistency
        print("🔍 Test 3: Graph consistency checks")
        
        # Check for self-loops and invalid weights
        for word, neighbors in graph.graph.items():
            for neighbor, weight in neighbors.items():
                # No self-loops
                if word == neighbor:
                    validation_results["tests_failed"] += 1
                    validation_results["details"].append(
                        f"FAIL: Self-loop found: {word} -> {neighbor}"
                    )
                # Positive weights
                elif weight <= 0:
                    validation_results["tests_failed"] += 1  
                    validation_results["details"].append(
                        f"FAIL: Non-positive weight: {word} -> {neighbor} ({weight})"
                    )
                else:
                    validation_results["tests_passed"] += 1
        
        total_tests = validation_results["tests_passed"] + validation_results["tests_failed"]
        success_rate = validation_results["tests_passed"] / total_tests if total_tests > 0 else 0
        
        print(f"\n📊 Validation Results:")
        print(f"   Passed: {validation_results['tests_passed']}")
        print(f"   Failed: {validation_results['tests_failed']}")
        print(f"   Success Rate: {success_rate:.1%}")
        
        if validation_results["details"]:
            print(f"\n❌ Failures:")
            for detail in validation_results["details"][:5]:  # Show first 5
                print(f"   {detail}")
        
        validation_results["success_rate"] = success_rate
        return validation_results


def main():
    """Main test runner with user-friendly interface."""
    print("🧪 RST Trap Finder - Comprehensive Test Suite")
    print("=" * 70)
    
    # Initialize test system
    data_manager = TestDataManager()
    runner = TestRunner(data_manager)
    
    # Show available datasets with clear categorization
    print("📊 Dataset Status Report:")
    print("-" * 70)
    datasets = data_manager.list_datasets()
    if not datasets:
        print("   ❌ No datasets found! Please ensure data files are in data/processed/")
        print("   💡 Run dataset_integration.py to download and process datasets")
        return
    
    # Categorize datasets
    primary_datasets = [d for d in datasets.values() if d.get("type") == "primary"]
    test_datasets = [d for d in datasets.values() if d.get("type") == "test_only"]
    
    if primary_datasets:
        print("   🎯 Primary Datasets (Real Analysis):")
        for info in primary_datasets:
            size_mb = info["path"].stat().st_size / 1024 / 1024 if info["path"].exists() else 0
            print(f"      ✅ {info['name']:25s} - {info['description']} ({size_mb:.1f} MB)")
    else:
        print("   ⚠️  No primary datasets available for real analysis!")
    
    if test_datasets:
        print("   🧪 Test Datasets (Development Only):")
        for info in test_datasets:
            size_kb = info["path"].stat().st_size / 1024 if info["path"].exists() else 0
            print(f"      🔬 {info['name']:25s} - {info['description']} ({size_kb:.1f} KB)")
    
    # Dataset recommendations
    print("\n   💡 Recommendations:")
    if not primary_datasets:
        print("      ❗ Download real datasets using dataset_integration.py")
        print("      ❗ Current test datasets are too small for meaningful analysis")
    elif len(primary_datasets) == 1:
        print("      ✅ ConceptNet provides solid foundation for RST analysis")
        print("      💡 Consider adding SWOW/EAT datasets for more comprehensive coverage")
    else:
        print("      ✅ Multiple datasets available - excellent for robust analysis!")
    
    # Run tests based on what we have
    print(f"\n🚀 Running Test Suite...")
    print("=" * 70)
    
    if primary_datasets:
        # Test with real data
        print("🎯 REAL DATA ANALYSIS")
        full_results = runner.run_basic_functionality_test("full")
        
        # Also test with placeholder if available (for speed comparison)
        if test_datasets:
            print("\n🧪 QUICK TEST (Placeholder Data)")
            sample_results = runner.run_basic_functionality_test("sample")
    else:
        # Only test data available
        print("⚠️  PLACEHOLDER DATA ONLY (Not suitable for real analysis)")
        sample_results = runner.run_basic_functionality_test("sample")
    
    # Performance benchmark
    if primary_datasets:
        benchmark_results = runner.run_performance_benchmark(["full"])
    
    # Correctness validation
    validation_results = runner.run_correctness_validation()
    
    # Final comprehensive summary
    print(f"\n🎯 COMPREHENSIVE TEST SUMMARY")
    print("=" * 70)
    
    # Dataset summary
    total_words = 0
    total_edges = 0
    if primary_datasets:
        # Use the largest primary dataset for summary
        largest_dataset = max(primary_datasets, key=lambda x: x["path"].stat().st_size)
        from rst_trap_finder.core import WordAssociationGraph
        graph = WordAssociationGraph.from_csv(largest_dataset["path"])
        total_words = len(graph.get_all_words())
        total_edges = sum(len(neighbors) for neighbors in graph.graph.values())
    
    print(f"📊 Data Coverage:")
    print(f"   • Available datasets: {len(datasets)} ({len(primary_datasets)} primary, {len(test_datasets)} test)")
    print(f"   • Analysis-ready words: {total_words:,}")
    print(f"   • Word associations: {total_edges:,}")
    print(f"   • Average connections per word: {total_edges/total_words:.1f}" if total_words > 0 else "   • No primary data available")
    
    # Algorithm performance
    print(f"\n⚡ Algorithm Performance:")
    print(f"   • Validation success rate: {validation_results['success_rate']:.1%}")
    if primary_datasets and 'full_results' in locals():
        print(f"   • Large dataset ranking: {full_results['ranking']['time']:.1f}s for {total_words:,} words")
        print(f"   • Processing speed: {total_words/full_results['ranking']['time']:,.0f} words/second")
    
    # Readiness assessment
    print(f"\n🎯 Development Readiness:")
    if validation_results['success_rate'] > 0.95 and primary_datasets:
        print("   ✅ READY FOR DEVELOPMENT")
        print("   ✅ Algorithms validated and working correctly")
        print("   ✅ Real datasets available for meaningful analysis")
        print("   ✅ Performance benchmarks established")
    elif primary_datasets:
        print("   ⚠️  PARTIALLY READY")
        print("   ✅ Real datasets available")
        print("   ❗ Some algorithm validation issues detected")
    else:
        print("   ❌ NOT READY FOR REAL ANALYSIS")
        print("   ❗ No primary datasets available")
        print("   💡 Run dataset_integration.py to download ConceptNet")
    
    print(f"\n🚀 Next Steps:")
    if not primary_datasets:
        print("   1. Download ConceptNet using dataset_integration.py")
        print("   2. Re-run tests with real data")
    elif len(primary_datasets) < 2:
        print("   1. ✅ Ready for intelligent dataset reduction")
        print("   2. ✅ Ready for multi-step analysis framework") 
        print("   3. 💡 Consider adding more datasets (SWOW, EAT)")
    else:
        print("   1. ✅ Ready for all advanced development tasks")
        print("   2. ✅ Excellent foundation for research-quality analysis")


if __name__ == "__main__":
    main()