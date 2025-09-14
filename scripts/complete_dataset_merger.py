#!/usr/bin/env python3
"""
Complete Dataset Merger - Include manually processed SWOW-EN18

Merges all three datasets with proper weighting:
1. SWOW-EN18 (manual) - 3.0x weight (highest quality)
2. USF - 1.5x weight (classic supplement)  
3. ConceptNet-light - 0.3x weight (filler)
"""

import pandas as pd
import json
from pathlib import Path
from typing import List
import time

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


def merge_complete_datasets():
    """Merge all available processed datasets."""
    print("🎯 Complete RST Dataset Merger")
    print("   SWOW-EN18 (3.0x) + USF (1.5x) + ConceptNet (0.3x)")
    print("=" * 60)
    
    processed_dir = Path("data/processed")
    merged_dir = Path("data/merged")
    merged_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset files and weights
    datasets = [
        ("edges_swow_en18.csv", 3.0, "PRIMARY"),
        ("edges_usf.csv", 1.5, "SECONDARY"), 
        ("edges_conceptnet_light.csv", 0.3, "FILLER")
    ]
    
    # Check available files
    available_datasets = []
    for filename, weight, priority in datasets:
        file_path = processed_dir / filename
        if file_path.exists():
            available_datasets.append((file_path, weight, priority))
            print(f"✅ Found: {filename} (weight: {weight}x, {priority})")
        else:
            print(f"❌ Missing: {filename}")
    
    if not available_datasets:
        print("❌ No datasets found to merge")
        return None
    
    print(f"\n🔗 Merging {len(available_datasets)} datasets...")
    
    merged_edges = {}
    source_stats = {}
    
    for file_path, weight, priority in available_datasets:
        dataset_name = file_path.stem.replace('edges_', '')
        
        print(f"\n   📖 Processing {file_path.name}...")
        df = pd.read_csv(file_path)
        
        edges_added = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Merging {dataset_name}"):
            src = str(row['src']).strip().lower()
            dst = str(row['dst']).strip().lower()
            edge_weight = float(row['weight']) * weight
            
            key = (src, dst)
            if key in merged_edges:
                # Weighted average for overlapping edges
                merged_edges[key] = (merged_edges[key] + edge_weight) / 2
            else:
                merged_edges[key] = edge_weight
            
            edges_added += 1
        
        source_stats[dataset_name] = {
            "original_edges": len(df),
            "weight_multiplier": weight,
            "priority": priority,
            "edges_added": edges_added
        }
        
        print(f"      ✅ Added {edges_added:,} weighted edges ({priority})")
    
    # Create final dataset
    print(f"\n🔗 Creating final merged dataset...")
    
    final_edges = []
    for (src, dst), weight in tqdm(merged_edges.items(), desc="Finalizing"):
        final_edges.append({
            'src': src,
            'dst': dst,
            'weight': round(weight, 6)
        })
    
    # Sort by weight (highest first)
    final_edges.sort(key=lambda x: x['weight'], reverse=True)
    
    # Save complete dataset
    output_path = merged_dir / "complete_rst_dataset.csv"
    df = pd.DataFrame(final_edges)
    df.to_csv(output_path, index=False)
    
    # Calculate statistics
    unique_words = set()
    for src, dst in merged_edges.keys():
        unique_words.add(src)
        unique_words.add(dst)
    
    weights = list(merged_edges.values())
    
    print(f"\n✅ Complete RST Dataset Created:")
    print(f"   📁 File: {output_path}")
    print(f"   🔗 Total edges: {len(merged_edges):,}")
    print(f"   📝 Unique words: {len(unique_words):,}")
    print(f"   📊 Average degree: {2 * len(merged_edges) / len(unique_words):.1f}")
    print(f"   ⚖️  Weight range: {min(weights):.3f} - {max(weights):.3f}")
    print(f"   📈 Median weight: {sorted(weights)[len(weights)//2]:.3f}")
    
    # Detailed breakdown
    print(f"\n📊 Dataset Composition:")
    total_original_edges = sum(stats["original_edges"] for stats in source_stats.values())
    
    for dataset, stats in source_stats.items():
        pct = (stats["original_edges"] / total_original_edges) * 100
        print(f"   • {dataset:15s}: {stats['original_edges']:6,} edges ({pct:4.1f}%) - {stats['priority']}")
    
    # Save comprehensive metadata
    metadata = {
        "created_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "description": "Complete RST dataset with all three sources",
        "weighting_strategy": {
            "SWOW-EN18": "3.0x (highest quality human associations)",
            "USF": "1.5x (classic word association norms)",
            "ConceptNet": "0.3x (semantic knowledge filler)"
        },
        "source_datasets": source_stats,
        "final_statistics": {
            "total_edges": len(merged_edges),
            "unique_words": len(unique_words),
            "average_degree": 2 * len(merged_edges) / len(unique_words),
            "weight_statistics": {
                "min": min(weights),
                "max": max(weights),
                "median": sorted(weights)[len(weights)//2],
                "mean": sum(weights) / len(weights)
            }
        },
        "quality_notes": [
            "SWOW-EN18 provides highest quality human free-association data",
            "USF adds classic word association norms for broader coverage", 
            "ConceptNet provides semantic filler for vocabulary completeness",
            "All datasets filtered for English-only, alphabetic words"
        ]
    }
    
    metadata_path = merged_dir / "complete_rst_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return output_path


def test_complete_dataset(dataset_path: Path):
    """Quick test of the complete dataset."""
    print(f"\n🧪 Testing Complete Dataset...")
    
    try:
        import sys
        sys.path.append('.')
        from rst_trap_finder.core import WordAssociationGraph
        
        graph = WordAssociationGraph.from_csv(dataset_path)
        all_words = graph.get_all_words()
        total_edges = sum(len(neighbors) for neighbors in graph.graph.values())
        
        print(f"✅ Dataset validation:")
        print(f"   📊 {len(all_words):,} words, {total_edges:,} edges")
        print(f"   📈 {total_edges / len(all_words):.1f} average edges per word")
        
        # Test sample words
        test_words = ['start', 'color', 'run', 'big', 'think']
        available_test_words = [w for w in test_words if graph.has_word(w)]
        
        if available_test_words:
            print(f"\n   🎯 Sample RST analysis:")
            for word in available_test_words[:3]:
                rst_prob = graph.one_step_rst_probability(word)
                neighbors = len(graph.get_neighbors(word))
                print(f"      {word:8s}: {rst_prob:.3f} RST prob, {neighbors:3d} neighbors")
        
        print(f"\n🚀 Ready for comprehensive RST analysis!")
        print(f"   Use: test_framework.py with complete_rst_dataset.csv")
        
    except Exception as e:
        print(f"⚠️  Dataset created but validation failed: {e}")


if __name__ == "__main__":
    result = merge_complete_datasets()
    if result:
        test_complete_dataset(result)
    else:
        print("❌ Dataset merge failed")