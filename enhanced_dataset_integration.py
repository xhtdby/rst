#!/usr/bin/env python3
"""
Enhanced RST Dataset Integration - Download and merge large-scale word association datasets

This module downloads and integrates multiple word association datasets with working URLs:
- SWOW-EN18: Small World of Words English dataset (12K cues, 100 responses each)
- ConceptNet 5.7: Large-scale semantic knowledge graph 
- EAT: Edinburgh Associative Thesaurus (classic dataset)
- Additional research datasets

Creates a comprehensive, context-free word association database optimized for RST analysis.
"""

import requests
import pandas as pd
import json
import gzip
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import time
import csv
import re
from urllib.parse import urlparse
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


class EnhancedDatasetDownloader:
    """Downloads large-scale word association datasets from verified research sources."""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("data/raw")
        self.processed_dir = Path("data/processed")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Verified dataset URLs (as of 2024)
        self.datasets = {
            "swow_en18": {
                "name": "Small World of Words English 2018",
                "url": "https://smallworldofwords.org/files/SWOW-EN18.zip",
                "format": "zip",
                "description": "12,292 cues with 100 responses each (1.2M+ associations)",
                "citation": "De Deyne et al. (2019) Behavior Research Methods"
            },
            "conceptnet57": {
                "name": "ConceptNet 5.7",
                "url": "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz",
                "format": "csv.gz", 
                "description": "21M+ assertions from multiple sources",
                "citation": "Speer et al. (2017) AAAI"
            },
            "eat": {
                "name": "Edinburgh Associative Thesaurus",
                "url": "http://www.eat.rl.ac.uk/EAT.TXT",
                "format": "txt",
                "description": "Classic word association norms (8K+ associations)",
                "citation": "Kiss et al. (1973)"
            }
        }
    
    def download_file(self, url: str, output_path: Path, headers: Dict = None) -> bool:
        """Download a file with progress bar and error handling."""
        try:
            print(f"📥 Downloading: {url}")
            
            # Some servers require User-Agent header
            default_headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            if headers:
                default_headers.update(headers)
            
            response = requests.get(url, stream=True, headers=default_headers, timeout=30)
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
            
            size_mb = output_path.stat().st_size / 1024 / 1024
            print(f"✅ Downloaded: {output_path.name} ({size_mb:.1f} MB)")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download {url}: {e}")
            if output_path.exists():
                output_path.unlink()  # Clean up partial file
            return False
    
    def download_swow_en18(self) -> Optional[Path]:
        """Download SWOW-EN18 dataset."""
        output_path = self.data_dir / "SWOW-EN18.zip"
        
        if output_path.exists():
            print(f"✅ SWOW-EN18 already exists: {output_path}")
            return output_path
        
        url = self.datasets["swow_en18"]["url"]
        if self.download_file(url, output_path):
            return output_path
        return None
    
    def download_conceptnet57(self) -> Optional[Path]:
        """Download ConceptNet 5.7 dataset."""
        output_path = self.data_dir / "conceptnet-assertions-5.7.0.csv.gz"
        
        if output_path.exists():
            print(f"✅ ConceptNet 5.7 already exists: {output_path}")
            return output_path
        
        url = self.datasets["conceptnet57"]["url"]
        if self.download_file(url, output_path):
            return output_path
        return None
    
    def download_eat(self) -> Optional[Path]:
        """Download EAT dataset."""
        output_path = self.data_dir / "EAT.TXT"
        
        if output_path.exists():
            print(f"✅ EAT already exists: {output_path}")
            return output_path
        
        url = self.datasets["eat"]["url"]
        if self.download_file(url, output_path):
            return output_path
        return None
    
    def download_all(self) -> Dict[str, Path]:
        """Download all available datasets."""
        print("🌐 Enhanced Dataset Download - Large-Scale Word Associations")
        print("=" * 65)
        
        results = {}
        
        # Download ConceptNet first (most reliable)
        print("\n📊 ConceptNet 5.7 (Large-scale semantic knowledge)")
        conceptnet_path = self.download_conceptnet57()
        if conceptnet_path:
            results["conceptnet57"] = conceptnet_path
        
        # Download SWOW-EN18 (high-quality research dataset)
        print("\n📊 SWOW-EN18 (Comprehensive word associations)")
        swow_path = self.download_swow_en18()
        if swow_path:
            results["swow_en18"] = swow_path
        
        # Download EAT (classic dataset)
        print("\n📊 EAT (Classic word association norms)")
        eat_path = self.download_eat()
        if eat_path:
            results["eat"] = eat_path
        
        print(f"\n📈 Download Summary:")
        print(f"   Successfully downloaded: {len(results)}/{len(self.datasets)} datasets")
        
        total_size = 0
        for name, path in results.items():
            size_mb = path.stat().st_size / 1024 / 1024
            total_size += size_mb
            desc = self.datasets[name]["description"]
            print(f"   • {name.upper():12s}: {desc[:50]:<50} ({size_mb:.1f} MB)")
        
        print(f"   📦 Total size: {total_size:.1f} MB")
        return results


class EnhancedDatasetProcessor:
    """Processes raw datasets into standardized, optimized format for RST analysis."""
    
    def __init__(self, processed_dir: Path = None):
        self.processed_dir = processed_dir or Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def process_conceptnet57(self, raw_path: Path) -> Optional[Path]:
        """Process ConceptNet 5.7 into standard format with English filtering."""
        print(f"\n🔄 Processing ConceptNet 5.7...")
        
        output_path = self.processed_dir / "edges_conceptnet57.csv"
        
        try:
            print("   📖 Reading ConceptNet assertions (this may take a while)...")
            
            edges = []
            total_lines = 0
            valid_relations = {
                '/r/RelatedTo', '/r/IsA', '/r/PartOf', '/r/HasA', '/r/UsedFor',
                '/r/CapableOf', '/r/AtLocation', '/r/Causes', '/r/HasSubevent',
                '/r/HasFirstSubevent', '/r/HasLastSubevent', '/r/HasPrerequisite',
                '/r/HasProperty', '/r/MotivatedByGoal', '/r/ObstructedBy',
                '/r/Desires', '/r/CreatedBy', '/r/Synonym', '/r/Antonym',
                '/r/DerivedFrom', '/r/EtymologicallyDerivedFrom', '/r/EtymologicallyRelatedTo',
                '/r/FormOf', '/r/DefinedAs', '/r/MannerOf', '/r/LocatedNear',
                '/r/HasContext', '/r/SimilarTo', '/r/DistinctFrom'
            }
            
            with gzip.open(raw_path, 'rt', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                
                for line_num, row in enumerate(reader):
                    total_lines += 1
                    
                    if line_num % 100000 == 0 and line_num > 0:
                        print(f"   📊 Processed {line_num:,} lines, found {len(edges):,} valid edges...")
                    
                    if len(row) < 5:
                        continue
                    
                    try:
                        relation = row[1]
                        start_node = row[2]
                        end_node = row[3]
                        info = json.loads(row[4])
                        
                        # Filter for English words only
                        if not (start_node.startswith('/c/en/') and end_node.startswith('/c/en/')):
                            continue
                        
                        # Extract word from URI (e.g., '/c/en/dog/n' -> 'dog')
                        start_word = start_node.split('/')[3].lower()
                        end_word = end_node.split('/')[3].lower()
                        
                        # Skip invalid words
                        if not start_word or not end_word or start_word == end_word:
                            continue
                        
                        # Only keep meaningful words (letters only, reasonable length)
                        if not (re.match(r'^[a-z]+$', start_word) and re.match(r'^[a-z]+$', end_word)):
                            continue
                        
                        if len(start_word) > 20 or len(end_word) > 20:
                            continue
                        
                        # Filter for useful relations
                        if relation not in valid_relations:
                            continue
                        
                        weight = float(info.get('weight', 1.0))
                        if weight <= 0:
                            continue
                        
                        edges.append({
                            'src': start_word,
                            'dst': end_word,
                            'weight': weight,
                            'relation': relation
                        })
                        
                    except (json.JSONDecodeError, ValueError, IndexError):
                        continue
            
            print(f"   📊 Raw processing complete: {len(edges):,} valid English edges from {total_lines:,} lines")
            
            # Aggregate duplicate edges
            print("   🔗 Aggregating duplicate edges...")
            edge_weights = {}
            
            for edge in tqdm(edges, desc="Aggregating edges"):
                key = (edge['src'], edge['dst'])
                if key in edge_weights:
                    edge_weights[key] += edge['weight']
                else:
                    edge_weights[key] = edge['weight']
            
            # Create final edge list
            final_edges = []
            for (src, dst), weight in edge_weights.items():
                final_edges.append({
                    'src': src,
                    'dst': dst,
                    'weight': weight
                })
            
            # Save to CSV
            df = pd.DataFrame(final_edges)
            df.to_csv(output_path, index=False)
            
            print(f"   ✅ Processed: {len(final_edges):,} unique edges saved to {output_path.name}")
            
            # Save metadata
            unique_words = set()
            for src, dst in edge_weights.keys():
                unique_words.add(src)
                unique_words.add(dst)
            
            metadata = {
                "dataset": "ConceptNet 5.7",
                "source": "ConceptNet semantic knowledge graph",
                "processed_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "raw_lines": total_lines,
                "valid_edges": len(final_edges),
                "unique_words": len(unique_words),
                "format": "src,dst,weight",
                "filters_applied": [
                    "English language only",
                    "Valid word format (letters only)",
                    "Meaningful relations only",
                    "Duplicate edge aggregation"
                ]
            }
            
            metadata_path = self.processed_dir / "metadata_conceptnet57.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return output_path
            
        except Exception as e:
            print(f"   ❌ Error processing ConceptNet: {e}")
            return None
    
    def process_swow_en18(self, raw_path: Path) -> Optional[Path]:
        """Process SWOW-EN18 dataset."""
        print(f"\n🔄 Processing SWOW-EN18...")
        
        output_path = self.processed_dir / "edges_swow_en18.csv"
        
        try:
            # Extract ZIP file first
            import zipfile
            extract_dir = self.processed_dir / "swow_extracted"
            extract_dir.mkdir(exist_ok=True)
            
            print("   📦 Extracting ZIP file...")
            with zipfile.ZipFile(raw_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Find the main data file (usually strength.SWOW-EN.R123.csv)
            data_files = list(extract_dir.glob("*.csv"))
            strength_file = None
            
            for file in data_files:
                if "strength" in file.name and "R123" in file.name:
                    strength_file = file
                    break
            
            if not strength_file:
                print(f"   ❌ Could not find strength data file in {raw_path}")
                return None
            
            print(f"   📖 Reading SWOW data from {strength_file.name}...")
            
            # Read with special CSV handling for SWOW format
            edges = []
            
            with open(strength_file, 'r', encoding='utf-8') as f:
                # Use custom CSV reader to handle quotes properly
                reader = csv.DictReader(f, delimiter='\t')
                
                for row_num, row in enumerate(tqdm(reader, desc="Processing SWOW")):
                    if row_num % 10000 == 0 and row_num > 0:
                        print(f"   📊 Processed {row_num:,} rows...")
                    
                    try:
                        cue = row.get('cue', '').strip().lower()
                        response = row.get('response', '').strip().lower()
                        strength = float(row.get('strength.R123', 0))
                        
                        # Filter valid associations
                        if (cue and response and cue != response and 
                            strength > 0 and len(cue) <= 20 and len(response) <= 20 and
                            re.match(r'^[a-z]+$', cue) and re.match(r'^[a-z]+$', response)):
                            
                            edges.append({
                                'src': cue,
                                'dst': response,
                                'weight': strength
                            })
                            
                    except (ValueError, KeyError):
                        continue
            
            # Save processed data
            df = pd.DataFrame(edges)
            df.to_csv(output_path, index=False)
            
            print(f"   ✅ Processed: {len(edges):,} valid associations saved to {output_path.name}")
            
            # Clean up extracted files
            import shutil
            shutil.rmtree(extract_dir)
            
            # Save metadata
            unique_words = set()
            for edge in edges:
                unique_words.add(edge['src'])
                unique_words.add(edge['dst'])
            
            metadata = {
                "dataset": "SWOW-EN18",
                "source": "Small World of Words English 2018",
                "processed_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "valid_edges": len(edges),
                "unique_words": len(unique_words),
                "format": "src,dst,weight"
            }
            
            metadata_path = self.processed_dir / "metadata_swow_en18.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return output_path
            
        except Exception as e:
            print(f"   ❌ Error processing SWOW-EN18: {e}")
            return None
    
    def process_eat(self, raw_path: Path) -> Optional[Path]:
        """Process EAT dataset."""
        print(f"\n🔄 Processing EAT...")
        
        output_path = self.processed_dir / "edges_eat.csv"
        
        try:
            print("   📖 Reading EAT data...")
            
            edges = []
            
            with open(raw_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    
                    if not line or line.startswith('#'):
                        continue
                    
                    # EAT format: stimulus response frequency
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            stimulus = parts[0].lower()
                            response = parts[1].lower()
                            frequency = float(parts[2])
                            
                            if (stimulus and response and stimulus != response and 
                                frequency > 0 and len(stimulus) <= 20 and len(response) <= 20 and
                                re.match(r'^[a-z]+$', stimulus) and re.match(r'^[a-z]+$', response)):
                                
                                edges.append({
                                    'src': stimulus,
                                    'dst': response,
                                    'weight': frequency
                                })
                        except ValueError:
                            continue
            
            # Save processed data
            df = pd.DataFrame(edges)
            df.to_csv(output_path, index=False)
            
            print(f"   ✅ Processed: {len(edges):,} valid associations saved to {output_path.name}")
            
            # Save metadata
            unique_words = set()
            for edge in edges:
                unique_words.add(edge['src'])
                unique_words.add(edge['dst'])
            
            metadata = {
                "dataset": "EAT",
                "source": "Edinburgh Associative Thesaurus",
                "processed_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "valid_edges": len(edges),
                "unique_words": len(unique_words),
                "format": "src,dst,weight"
            }
            
            metadata_path = self.processed_dir / "metadata_eat.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return output_path
            
        except Exception as e:
            print(f"   ❌ Error processing EAT: {e}")
            return None


class IntelligentDatasetMerger:
    """Merges multiple datasets with intelligent deduplication and weighting."""
    
    def __init__(self, processed_dir: Path = None):
        self.processed_dir = processed_dir or Path("data/processed")
        self.merged_dir = Path("data/merged")
        self.merged_dir.mkdir(parents=True, exist_ok=True)
    
    def find_available_datasets(self) -> List[Path]:
        """Find all available processed datasets."""
        datasets = []
        
        # Look for edge files
        for csv_file in self.processed_dir.glob("edges_*.csv"):
            if "usf" not in csv_file.name:  # Skip small test datasets
                datasets.append(csv_file)
        
        return sorted(datasets)
    
    def merge_with_intelligent_weighting(self, dataset_paths: List[Path] = None) -> Optional[Path]:
        """Merge datasets with intelligent weighting based on source reliability."""
        if dataset_paths is None:
            dataset_paths = self.find_available_datasets()
        
        print(f"\n🧠 Intelligent Dataset Merging")
        print("=" * 50)
        
        if not dataset_paths:
            print("❌ No datasets found to merge!")
            return None
        
        print(f"📊 Found {len(dataset_paths)} datasets to merge:")
        
        # Dataset weighting based on reliability and coverage
        dataset_weights = {
            "conceptnet57": 1.0,    # High reliability, broad coverage
            "swow_en18": 2.0,       # Highest quality, controlled collection
            "eat": 1.5,             # Classic, reliable but smaller
            "conceptnet": 1.0       # Legacy ConceptNet
        }
        
        total_edges = 0
        merged_edges = {}
        dataset_stats = {}
        
        for dataset_path in dataset_paths:
            dataset_name = dataset_path.stem.replace('edges_', '')
            weight_multiplier = dataset_weights.get(dataset_name, 1.0)
            
            print(f"\n   📖 Processing {dataset_path.name} (weight: {weight_multiplier}x)...")
            
            try:
                df = pd.read_csv(dataset_path)
                
                edges_added = 0
                for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Merging {dataset_name}"):
                    src = str(row['src']).strip().lower()
                    dst = str(row['dst']).strip().lower()
                    weight = float(row['weight']) * weight_multiplier
                    
                    if src and dst and src != dst:
                        edge_key = (src, dst)
                        if edge_key in merged_edges:
                            # Use weighted average for conflicting edges
                            merged_edges[edge_key] = (merged_edges[edge_key] + weight) / 2
                        else:
                            merged_edges[edge_key] = weight
                        edges_added += 1
                
                dataset_stats[dataset_name] = {
                    "raw_edges": len(df),
                    "weight_multiplier": weight_multiplier,
                    "edges_added": edges_added
                }
                
                total_edges += edges_added
                print(f"   ✅ Added {edges_added:,} weighted edges from {dataset_name}")
                
            except Exception as e:
                print(f"   ❌ Error reading {dataset_path.name}: {e}")
        
        # Create final merged dataset
        print(f"\n🔗 Creating intelligent merged dataset...")
        final_edges = []
        
        # Sort by weight for better compression and analysis
        sorted_edges = sorted(merged_edges.items(), key=lambda x: x[1], reverse=True)
        
        for (src, dst), weight in tqdm(sorted_edges, desc="Finalizing edges"):
            final_edges.append({
                'src': src,
                'dst': dst,
                'weight': round(weight, 6)
            })
        
        # Save merged dataset
        output_path = self.merged_dir / "enhanced_merged_association_graph.csv"
        final_df = pd.DataFrame(final_edges)
        final_df.to_csv(output_path, index=False)
        
        # Calculate statistics
        unique_words = set()
        for src, dst in merged_edges.keys():
            unique_words.add(src)
            unique_words.add(dst)
        
        # Weight statistics
        weights = list(merged_edges.values())
        
        print(f"\n✅ Enhanced Merged Dataset Created:")
        print(f"   • Output: {output_path}")
        print(f"   • Total edges: {len(merged_edges):,}")
        print(f"   • Unique words: {len(unique_words):,}")
        print(f"   • Average degree: {2 * len(merged_edges) / len(unique_words):.1f}")
        print(f"   • Weight range: {min(weights):.3f} - {max(weights):.3f}")
        print(f"   • Median weight: {sorted(weights)[len(weights)//2]:.3f}")
        
        # Save comprehensive metadata
        merge_metadata = {
            "created_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "merge_strategy": "intelligent_weighting",
            "dataset_weights": dataset_weights,
            "source_datasets": dataset_stats,
            "merged_stats": {
                "total_edges": len(merged_edges),
                "unique_words": len(unique_words),
                "average_degree": 2 * len(merged_edges) / len(unique_words),
                "weight_statistics": {
                    "min": min(weights),
                    "max": max(weights),
                    "median": sorted(weights)[len(weights)//2],
                    "mean": sum(weights) / len(weights)
                }
            }
        }
        
        metadata_path = self.merged_dir / "enhanced_merge_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(merge_metadata, f, indent=2)
        
        return output_path


def main():
    """Enhanced dataset integration pipeline."""
    print("🚀 Enhanced RST Dataset Integration Pipeline")
    print("=" * 70)
    print("   Downloading large-scale research datasets for comprehensive RST analysis")
    print()
    
    # Step 1: Download datasets
    downloader = EnhancedDatasetDownloader()
    downloaded = downloader.download_all()
    
    if not downloaded:
        print("\n❌ No datasets downloaded successfully. Check network connection.")
        return
    
    # Step 2: Process datasets
    processor = EnhancedDatasetProcessor()
    processed = {}
    
    print(f"\n🔄 Processing {len(downloaded)} downloaded datasets...")
    
    if "conceptnet57" in downloaded:
        conceptnet_processed = processor.process_conceptnet57(downloaded["conceptnet57"])
        if conceptnet_processed:
            processed["conceptnet57"] = conceptnet_processed
    
    if "swow_en18" in downloaded:
        swow_processed = processor.process_swow_en18(downloaded["swow_en18"])
        if swow_processed:
            processed["swow_en18"] = swow_processed
    
    if "eat" in downloaded:
        eat_processed = processor.process_eat(downloaded["eat"])
        if eat_processed:
            processed["eat"] = eat_processed
    
    # Step 3: Intelligent merging
    merger = IntelligentDatasetMerger()
    merged_path = merger.merge_with_intelligent_weighting()
    
    # Step 4: Integration with test framework
    if merged_path and merged_path.exists():
        print(f"\n🧪 Testing enhanced dataset with RST framework...")
        
        try:
            # Quick validation test
            sys.path.append('.')
            from rst_trap_finder.core import WordAssociationGraph
            
            graph = WordAssociationGraph.from_csv(merged_path)
            all_words = graph.get_all_words()
            edge_count = sum(len(neighbors) for neighbors in graph.graph.values())
            
            print(f"✅ Enhanced dataset validation:")
            print(f"   • Words: {len(all_words):,}")
            print(f"   • Edges: {edge_count:,}")
            print(f"   • Density: {edge_count / len(all_words):.1f} edges/word")
            
            # Quick trap analysis sample
            sample_words = ['color', 'start', 'think', 'run', 'big']
            available_words = [w for w in sample_words if graph.has_word(w)]
            
            if available_words:
                print(f"\n   📊 Sample RST analysis:")
                for word in available_words[:3]:
                    rst_prob = graph.one_step_rst_probability(word)
                    neighbors = len(graph.get_neighbors(word))
                    print(f"      {word:8s}: {rst_prob:.3f} RST prob, {neighbors:3d} neighbors")
            
        except Exception as e:
            print(f"⚠️  Dataset created but validation failed: {e}")
    
    print(f"\n🎯 Enhanced Dataset Integration Complete!")
    print(f"   Ready for intelligent reduction and advanced RST algorithms!")
    print(f"   Use test_framework.py to analyze the enhanced dataset.")


if __name__ == "__main__":
    main()