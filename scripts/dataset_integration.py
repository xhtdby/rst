#!/usr/bin/env python3
"""
RST Dataset Integration - Download and merge all major word association datasets

This module downloads and integrates multiple word association datasets:
- SWOW (Small World of Words) 
- EAT (Edinburgh Associative Thesaurus)
- ConceptNet (already have)
- USF (University of South Florida - already have)

Creates a unified, context-free word association database.

Note: Legacy script. Prefer `enhanced_dataset_integration.py` for the up-to-date pipeline.
"""

import requests
import pandas as pd
import json
import gzip
from pathlib import Path
from typing import Dict, List, Tuple, Set
import time
from urllib.parse import urlparse
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


class DatasetDownloader:
    """Downloads word association datasets from their source URLs."""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("data/raw")
        self.processed_dir = Path("data/processed")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset URLs and information
        self.datasets = {
            "swow": {
                "name": "Small World of Words (English)",
                "url": "https://smallworldofwords.org/files/SWOW-EN.R100.csv",
                "format": "csv",
                "description": "Continuous word association data from SWOW project"
            },
            "eat": {
                "name": "Edinburgh Associative Thesaurus", 
                "url": "http://rali.iro.umontreal.ca/rali/sites/default/files/publis/eat.dat.gz",
                "format": "dat.gz",
                "description": "Classic word association norms from Edinburgh"
            }
        }
    
    def download_file(self, url: str, output_path: Path) -> bool:
        """Download a file with progress bar."""
        try:
            print(f"📥 Downloading: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as file:
                if total_size > 0:
                    progress = tqdm(
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        desc=f"Downloading {output_path.name}"
                    )
                else:
                    progress = None
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        if progress:
                            progress.update(len(chunk))
                
                if progress:
                    progress.close()
            
            print(f"✅ Downloaded: {output_path.name} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download {url}: {e}")
            return False
    
    def download_swow(self) -> Path:
        """Download SWOW dataset."""
        output_path = self.data_dir / "SWOW-EN.R100.csv"
        
        if output_path.exists():
            print(f"✅ SWOW already exists: {output_path}")
            return output_path
        
        url = self.datasets["swow"]["url"]
        if self.download_file(url, output_path):
            return output_path
        return None
    
    def download_eat(self) -> Path:
        """Download EAT dataset."""
        output_path = self.data_dir / "eat.dat.gz"
        
        if output_path.exists():
            print(f"✅ EAT already exists: {output_path}")
            return output_path
        
        url = self.datasets["eat"]["url"]
        if self.download_file(url, output_path):
            return output_path
        return None
    
    def download_all(self) -> Dict[str, Path]:
        """Download all available datasets."""
        print("🌐 Downloading Word Association Datasets")
        print("=" * 50)
        
        results = {}
        
        # Download SWOW
        swow_path = self.download_swow()
        if swow_path:
            results["swow"] = swow_path
        
        # Download EAT  
        eat_path = self.download_eat()
        if eat_path:
            results["eat"] = eat_path
        
        print(f"\n📊 Download Summary:")
        print(f"   Successfully downloaded: {len(results)} datasets")
        for name, path in results.items():
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"   • {name.upper():8s}: {path.name} ({size_mb:.1f} MB)")
        
        return results


class DatasetProcessor:
    """Processes raw datasets into standardized format."""
    
    def __init__(self, processed_dir: Path = None):
        self.processed_dir = processed_dir or Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def process_swow(self, raw_path: Path) -> Path:
        """Process SWOW dataset into standard format."""
        print(f"\n🔄 Processing SWOW dataset...")
        
        output_path = self.processed_dir / "edges_swow.csv"
        
        try:
            # Read SWOW data
            print("   📖 Reading SWOW CSV...")
            df = pd.read_csv(raw_path)
            
            print(f"   📊 Raw data: {len(df):,} associations")
            
            # SWOW format: cue, response, strength, etc.
            # Map to our standard format: src, dst, weight
            edges = []
            
            # Use tqdm for progress
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing SWOW"):
                cue = str(row.get('cue', '')).strip().lower()
                response = str(row.get('response', '')).strip().lower()
                strength = float(row.get('R1.Strength', 0))  # Response strength
                
                # Filter valid associations
                if cue and response and cue != response and strength > 0:
                    edges.append({
                        'src': cue,
                        'dst': response, 
                        'weight': strength
                    })
            
            # Create output DataFrame
            edges_df = pd.DataFrame(edges)
            
            # Save to CSV
            edges_df.to_csv(output_path, index=False)
            
            print(f"   ✅ Processed: {len(edges):,} valid edges saved to {output_path.name}")
            
            # Save metadata
            metadata = {
                "dataset": "SWOW",
                "source": "Small World of Words (English)",
                "processed_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "raw_associations": len(df),
                "valid_edges": len(edges),
                "format": "src,dst,weight"
            }
            
            metadata_path = self.processed_dir / "metadata_swow.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return output_path
            
        except Exception as e:
            print(f"   ❌ Error processing SWOW: {e}")
            return None
    
    def process_eat(self, raw_path: Path) -> Path:
        """Process EAT dataset into standard format."""
        print(f"\n🔄 Processing EAT dataset...")
        
        output_path = self.processed_dir / "edges_eat.csv"
        
        try:
            # Read EAT data (it's gzipped)
            print("   📖 Reading EAT data...")
            
            edges = []
            total_lines = 0
            
            with gzip.open(raw_path, 'rt', encoding='utf-8') as f:
                # EAT format is typically: stimulus response frequency
                for line_num, line in enumerate(f):
                    total_lines += 1
                    line = line.strip()
                    
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 3:
                        stimulus = parts[0].lower()
                        response = parts[1].lower()
                        frequency = float(parts[2])
                        
                        if stimulus and response and stimulus != response and frequency > 0:
                            edges.append({
                                'src': stimulus,
                                'dst': response,
                                'weight': frequency
                            })
            
            # Create output DataFrame
            edges_df = pd.DataFrame(edges)
            
            # Save to CSV
            edges_df.to_csv(output_path, index=False)
            
            print(f"   ✅ Processed: {len(edges):,} valid edges from {total_lines:,} lines")
            
            # Save metadata
            metadata = {
                "dataset": "EAT",
                "source": "Edinburgh Associative Thesaurus",
                "processed_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "raw_lines": total_lines,
                "valid_edges": len(edges),
                "format": "src,dst,weight"
            }
            
            metadata_path = self.processed_dir / "metadata_eat.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return output_path
            
        except Exception as e:
            print(f"   ❌ Error processing EAT: {e}")
            return None


class DatasetMerger:
    """Merges multiple datasets into unified word association database."""
    
    def __init__(self, processed_dir: Path = None):
        self.processed_dir = processed_dir or Path("data/processed")
        self.merged_dir = Path("data/merged")
        self.merged_dir.mkdir(parents=True, exist_ok=True)
    
    def find_available_datasets(self) -> List[Path]:
        """Find all available processed datasets."""
        datasets = []
        
        # Look for edge files
        for csv_file in self.processed_dir.glob("edges_*.csv"):
            datasets.append(csv_file)
        
        return sorted(datasets)
    
    def merge_datasets(self, dataset_paths: List[Path] = None) -> Path:
        """Merge multiple datasets with deduplication and weighting."""
        if dataset_paths is None:
            dataset_paths = self.find_available_datasets()
        
        print(f"\n🔗 Merging Word Association Datasets")
        print("=" * 50)
        
        if not dataset_paths:
            print("❌ No datasets found to merge!")
            return None
        
        print(f"📊 Found {len(dataset_paths)} datasets to merge:")
        for path in dataset_paths:
            print(f"   • {path.name}")
        
        # Merge strategy: combine all edges, sum weights for duplicates
        merged_edges = {}
        dataset_stats = {}
        
        for dataset_path in dataset_paths:
            print(f"\n   📖 Reading {dataset_path.name}...")
            
            try:
                df = pd.read_csv(dataset_path)
                dataset_name = dataset_path.stem.replace('edges_', '')
                
                edges_added = 0
                for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Merging {dataset_name}"):
                    src = str(row['src']).strip().lower()
                    dst = str(row['dst']).strip().lower()
                    weight = float(row['weight'])
                    
                    if src and dst and src != dst:
                        edge_key = (src, dst)
                        if edge_key in merged_edges:
                            merged_edges[edge_key] += weight
                        else:
                            merged_edges[edge_key] = weight
                        edges_added += 1
                
                dataset_stats[dataset_name] = {
                    "raw_edges": len(df),
                    "valid_edges": edges_added
                }
                
                print(f"   ✅ Added {edges_added:,} edges from {dataset_name}")
                
            except Exception as e:
                print(f"   ❌ Error reading {dataset_path.name}: {e}")
        
        # Create final merged dataset
        print(f"\n🔗 Creating merged dataset...")
        final_edges = []
        
        for (src, dst), weight in tqdm(merged_edges.items(), desc="Finalizing edges"):
            final_edges.append({
                'src': src,
                'dst': dst,
                'weight': weight
            })
        
        # Save merged dataset
        output_path = self.merged_dir / "merged_association_graph.csv"
        final_df = pd.DataFrame(final_edges)
        final_df.to_csv(output_path, index=False)
        
        # Calculate statistics
        unique_words = set()
        for src, dst in merged_edges.keys():
            unique_words.add(src)
            unique_words.add(dst)
        
        print(f"\n✅ Merged Dataset Created:")
        print(f"   • Output: {output_path}")
        print(f"   • Total edges: {len(merged_edges):,}")
        print(f"   • Unique words: {len(unique_words):,}")
        print(f"   • Average degree: {2 * len(merged_edges) / len(unique_words):.1f}")
        
        # Save merge metadata
        merge_metadata = {
            "created_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_datasets": dataset_stats,
            "merged_stats": {
                "total_edges": len(merged_edges),
                "unique_words": len(unique_words),
                "average_degree": 2 * len(merged_edges) / len(unique_words)
            }
        }
        
        metadata_path = self.merged_dir / "merge_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(merge_metadata, f, indent=2)
        
        return output_path


def main():
    """Main pipeline for dataset integration."""
    print("🌐 RST Dataset Integration Pipeline")
    print("=" * 60)
    
    # Step 1: Download datasets
    downloader = DatasetDownloader()
    downloaded = downloader.download_all()
    
    # Step 2: Process datasets
    processor = DatasetProcessor()
    processed = {}
    
    if "swow" in downloaded:
        swow_processed = processor.process_swow(downloaded["swow"])
        if swow_processed:
            processed["swow"] = swow_processed
    
    if "eat" in downloaded:
        eat_processed = processor.process_eat(downloaded["eat"])
        if eat_processed:
            processed["eat"] = eat_processed
    
    # Step 3: Merge all datasets (including existing ones)
    merger = DatasetMerger()
    merged_path = merger.merge_datasets()
    
    # Step 4: Test the merged dataset
    if merged_path and merged_path.exists():
        print(f"\n🧪 Testing merged dataset with our framework...")
        
        # Try to import test framework from examples directory
        import sys
        from pathlib import Path
        examples_path = Path(__file__).parent.parent / "examples"
        sys.path.insert(0, str(examples_path))
        
        try:
            import test_framework  # type: ignore
            TestDataManager = test_framework.TestDataManager
            TestRunner = test_framework.TestRunner
        except (ImportError, AttributeError) as e:
            print(f"   ⚠️  Test framework not available ({e}), skipping validation tests")
            print(f"   📁 Dataset saved to: {merged_path}")
            return merged_path
        
        # Add merged dataset to test
        data_manager = TestDataManager()
        runner = TestRunner(data_manager)
        
        # Quick test
        from rst_trap_finder.core import WordAssociationGraph
        graph = WordAssociationGraph.from_csv(merged_path)
        
        print(f"✅ Merged dataset test:")
        print(f"   • Words: {len(graph.get_all_words()):,}")
        print(f"   • Edges: {sum(len(neighbors) for neighbors in graph.graph.values()):,}")
    
    print(f"\n🎯 Dataset Integration Complete!")
    print(f"   Ready for intelligent reduction and advanced algorithms!")


if __name__ == "__main__":
    main()
