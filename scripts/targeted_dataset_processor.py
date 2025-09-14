#!/usr/bin/env python3
"""
Targeted RST Dataset Processor - Focus on SWOW-EN18 + USF + ConceptNet

Processes the downloaded datasets with proper prioritization:
1. SWOW-EN18 (R123) - Primary, highest weight (cleanest human associations)
2. USF - Secondary, medium weight (classic supplement)
3. ConceptNet - Low weight filler only (broad coverage)

Optimized for RST trap word analysis.
"""

import pandas as pd
import json
import gzip
import zipfile
import xml.etree.ElementTree as ET
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import sys

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


class TargetedDatasetProcessor:
    """Processes SWOW-EN18, USF, and ConceptNet with proper weights for RST analysis."""
    
    def __init__(self):
        self.raw_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")
        self.merged_dir = Path("data/merged")
        
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.merged_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset weights for merging (following your recommendations)
        self.weights = {
            "swow_en18": 3.0,      # Primary: cleanest human associations
            "usf": 1.5,            # Secondary: classic supplement  
            "conceptnet": 0.3      # Low weight: filler only
        }
    
    def process_swow_en18(self) -> Optional[Path]:
        """Process SWOW-EN18 R123 associative strength file."""
        print("🎯 Processing SWOW-EN18 (Primary Dataset)")
        print("=" * 50)
        
        zip_path = self.raw_dir / "SWOW-EN18.zip"
        if not zip_path.exists():
            print(f"❌ SWOW-EN18.zip not found in {self.raw_dir}")
            return None
        
        output_path = self.processed_dir / "edges_swow_en18.csv"
        
        try:
            # Extract and find the R123 strength file
            extract_dir = self.processed_dir / "swow_temp"
            extract_dir.mkdir(exist_ok=True)
            
            print("   📦 Extracting SWOW-EN18.zip...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Find strength file (usually strength.SWOW-EN.R123.csv)
            strength_files = list(extract_dir.glob("**/strength*R123*.csv"))
            if not strength_files:
                strength_files = list(extract_dir.glob("**/*R123*.csv"))
            
            if not strength_files:
                print(f"   ❌ Could not find R123 strength file in extracted data")
                return None
            
            strength_file = strength_files[0]
            print(f"   📖 Processing {strength_file.name}...")
            
            edges = []
            total_processed = 0
            
            # Read SWOW data with proper CSV handling and increased field size limit
            csv.field_size_limit(1000000)  # Increase field size limit
            
            with open(strength_file, 'r', encoding='utf-8') as f:
                # Check first few lines to understand format
                sample_lines = [f.readline() for _ in range(3)]
                f.seek(0)
                
                # Determine delimiter
                delimiter = '\t' if '\t' in sample_lines[1] else ','
                
                reader = csv.DictReader(f, delimiter=delimiter)
                
                for row in tqdm(reader, desc="Processing SWOW-EN18"):
                    total_processed += 1
                    
                    try:
                        # Extract cue, response, and strength
                        cue = str(row.get('cue', '')).strip().lower()
                        response = str(row.get('response', '')).strip().lower()
                        
                        # Try different column names for strength
                        strength = None
                        for col in ['strength.R123', 'R123.Strength', 'strength', 'R123']:
                            if col in row and row[col]:
                                strength = float(row[col])
                                break
                        
                        if strength is None:
                            continue
                        
                        # Validate and clean
                        if (cue and response and cue != response and 
                            strength > 0 and len(cue) <= 25 and len(response) <= 25):
                            
                            # Keep only alphabetic words
                            if re.match(r'^[a-z]+$', cue) and re.match(r'^[a-z]+$', response):
                                edges.append({
                                    'src': cue,
                                    'dst': response,
                                    'weight': strength
                                })
                                
                    except (ValueError, KeyError) as e:
                        continue
            
            # Save processed edges
            df = pd.DataFrame(edges)
            df.to_csv(output_path, index=False)
            
            # Clean up temp directory
            import shutil
            shutil.rmtree(extract_dir)
            
            # Calculate statistics
            unique_words = set()
            for edge in edges:
                unique_words.add(edge['src'])
                unique_words.add(edge['dst'])
            
            print(f"   ✅ SWOW-EN18 processed:")
            print(f"      📊 {total_processed:,} total rows processed")
            print(f"      🔗 {len(edges):,} valid word associations")
            print(f"      📝 {len(unique_words):,} unique words")
            print(f"      💾 Saved to {output_path.name}")
            
            # Save metadata
            metadata = {
                "dataset": "SWOW-EN18",
                "source": "Small World of Words English 2018 (R123)",
                "description": "Human free-association data, 3 responses per cue",
                "processed_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_rows": total_processed,
                "valid_edges": len(edges),
                "unique_words": len(unique_words),
                "weight_in_merge": self.weights["swow_en18"],
                "priority": "PRIMARY"
            }
            
            with open(self.processed_dir / "metadata_swow_en18.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return output_path
            
        except Exception as e:
            print(f"   ❌ Error processing SWOW-EN18: {e}")
            return None
    
    def process_usf(self) -> Optional[Path]:
        """Process USF Free Association Norms."""
        print("\n📚 Processing USF Free Association Norms (Secondary Dataset)")
        print("=" * 60)
        
        zip_path = self.raw_dir / "USF-cue-target.xml.zip"
        if not zip_path.exists():
            print(f"❌ USF file not found in {self.raw_dir}")
            return None
        
        output_path = self.processed_dir / "edges_usf.csv"
        
        try:
            # Extract USF data
            extract_dir = self.processed_dir / "usf_temp"
            extract_dir.mkdir(exist_ok=True)
            
            print("   📦 Extracting USF data...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Find XML file
            xml_files = list(extract_dir.glob("**/*.xml"))
            if not xml_files:
                # Try text files
                txt_files = list(extract_dir.glob("**/*.txt"))
                if txt_files:
                    return self._process_usf_txt(txt_files[0], output_path, extract_dir)
                else:
                    print(f"   ❌ No XML or TXT files found in USF data")
                    return None
            
            xml_file = xml_files[0]
            print(f"   📖 Processing {xml_file.name}...")
            
            edges = []
            
            # Parse XML
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            for cue_elem in tqdm(root.findall('.//cue'), desc="Processing USF"):
                cue_word = cue_elem.get('word', '').strip().lower()
                
                if not cue_word or not re.match(r'^[a-z]+$', cue_word):
                    continue
                
                for target_elem in cue_elem.findall('.//target'):
                    target_word = target_elem.get('word', '').strip().lower()
                    fsg = target_elem.get('fsg', '0')  # Forward Strength
                    
                    try:
                        strength = float(fsg)
                        
                        if (target_word and target_word != cue_word and 
                            strength > 0 and len(target_word) <= 25 and
                            re.match(r'^[a-z]+$', target_word)):
                            
                            edges.append({
                                'src': cue_word,
                                'dst': target_word,
                                'weight': strength
                            })
                            
                    except ValueError:
                        continue
            
            # Save and cleanup
            df = pd.DataFrame(edges)
            df.to_csv(output_path, index=False)
            
            import shutil
            shutil.rmtree(extract_dir)
            
            unique_words = set()
            for edge in edges:
                unique_words.add(edge['src'])
                unique_words.add(edge['dst'])
            
            print(f"   ✅ USF processed:")
            print(f"      🔗 {len(edges):,} word associations")
            print(f"      📝 {len(unique_words):,} unique words")
            print(f"      💾 Saved to {output_path.name}")
            
            # Save metadata
            metadata = {
                "dataset": "USF",
                "source": "University of South Florida Free Association Norms",
                "description": "Classic word association norms with forward strength",
                "processed_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "valid_edges": len(edges),
                "unique_words": len(unique_words),
                "weight_in_merge": self.weights["usf"],
                "priority": "SECONDARY"
            }
            
            with open(self.processed_dir / "metadata_usf.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return output_path
            
        except Exception as e:
            print(f"   ❌ Error processing USF: {e}")
            return None
    
    def _process_usf_txt(self, txt_file: Path, output_path: Path, extract_dir: Path) -> Optional[Path]:
        """Process USF if it's in text format."""
        print(f"   📖 Processing USF text file: {txt_file.name}")
        
        edges = []
        
        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Try different formats: "cue target strength" or "cue: target1 freq1, target2 freq2"
                if ':' in line:
                    parts = line.split(':')
                    if len(parts) == 2:
                        cue = parts[0].strip().lower()
                        targets_str = parts[1].strip()
                        
                        # Parse targets "target1 freq1, target2 freq2"
                        for target_part in targets_str.split(','):
                            target_data = target_part.strip().split()
                            if len(target_data) >= 2:
                                target = target_data[0].lower()
                                try:
                                    freq = float(target_data[1])
                                    if (cue and target and cue != target and freq > 0 and
                                        re.match(r'^[a-z]+$', cue) and re.match(r'^[a-z]+$', target)):
                                        edges.append({'src': cue, 'dst': target, 'weight': freq})
                                except ValueError:
                                    continue
                else:
                    # Simple format: "cue target strength"
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            cue = parts[0].lower()
                            target = parts[1].lower()
                            strength = float(parts[2])
                            
                            if (cue and target and cue != target and strength > 0 and
                                re.match(r'^[a-z]+$', cue) and re.match(r'^[a-z]+$', target)):
                                edges.append({'src': cue, 'dst': target, 'weight': strength})
                        except ValueError:
                            continue
        
        if edges:
            df = pd.DataFrame(edges)
            df.to_csv(output_path, index=False)
            
            import shutil
            shutil.rmtree(extract_dir)
            
            return output_path
        
        return None
    
    def process_conceptnet_light(self) -> Optional[Path]:
        """Process ConceptNet with light filtering for filler edges only."""
        print("\n🧠 Processing ConceptNet 5.7 (Low-Weight Filler)")
        print("=" * 50)
        
        gz_path = self.raw_dir / "conceptnet-assertions-5.7.0.csv.gz"
        if not gz_path.exists():
            print(f"❌ ConceptNet file not found in {self.raw_dir}")
            return None
        
        output_path = self.processed_dir / "edges_conceptnet_light.csv"
        
        try:
            print("   📖 Processing ConceptNet (light filtering)...")
            
            edges = []
            processed = 0
            
            # Focus on high-quality relations only
            good_relations = {'/r/RelatedTo', '/r/Synonym', '/r/IsA', '/r/UsedFor', '/r/CapableOf'}
            
            with gzip.open(gz_path, 'rt', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                
                for row in reader:
                    processed += 1
                    
                    # Process in smaller chunks with progress
                    if processed % 250000 == 0 and processed > 0:
                        print(f"      📊 Processed {processed:,} lines, found {len(edges):,} edges...")
                        
                        # Early exit if we have enough filler data
                        if len(edges) >= 30000:
                            print(f"      🛑 Early exit: sufficient filler data collected")
                            break
                    
                    if len(row) < 5:
                        continue
                    
                    try:
                        relation = row[1]
                        start_node = row[2]
                        end_node = row[3]
                        
                        # Quick filters
                        if relation not in good_relations:
                            continue
                        
                        if not (start_node.startswith('/c/en/') and end_node.startswith('/c/en/')):
                            continue
                        
                        # Extract words
                        start_word = start_node.split('/')[3].lower()
                        end_word = end_node.split('/')[3].lower()
                        
                        # Strict filtering for quality
                        if (len(start_word) > 15 or len(end_word) > 15 or
                            not re.match(r'^[a-z]+$', start_word) or 
                            not re.match(r'^[a-z]+$', end_word) or
                            start_word == end_word):
                            continue
                        
                        # Parse weight
                        info = json.loads(row[4])
                        weight = float(info.get('weight', 1.0))
                        
                        if weight > 0.5:  # Only high-confidence edges
                            edges.append({
                                'src': start_word,
                                'dst': end_word,
                                'weight': weight
                            })
                        
                            
                    except (json.JSONDecodeError, ValueError, IndexError):
                        continue
            
            # Deduplicate
            print("   🔗 Deduplicating edges...")
            edge_dict = {}
            for edge in edges:
                key = (edge['src'], edge['dst'])
                if key in edge_dict:
                    edge_dict[key] = max(edge_dict[key], edge['weight'])
                else:
                    edge_dict[key] = edge['weight']
            
            final_edges = [{'src': src, 'dst': dst, 'weight': weight} 
                          for (src, dst), weight in edge_dict.items()]
            
            # Save
            df = pd.DataFrame(final_edges)
            df.to_csv(output_path, index=False)
            
            unique_words = set()
            for src, dst in edge_dict.keys():
                unique_words.add(src)
                unique_words.add(dst)
            
            print(f"   ✅ ConceptNet processed (light):")
            print(f"      🔗 {len(final_edges):,} high-quality edges")
            print(f"      📝 {len(unique_words):,} unique words")
            print(f"      💾 Saved to {output_path.name}")
            
            # Save metadata
            metadata = {
                "dataset": "ConceptNet-Light",
                "source": "ConceptNet 5.7 (filtered for quality)",
                "description": "High-confidence semantic edges for vocabulary coverage",
                "processed_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "valid_edges": len(final_edges),
                "unique_words": len(unique_words),
                "weight_in_merge": self.weights["conceptnet"],
                "priority": "FILLER"
            }
            
            with open(self.processed_dir / "metadata_conceptnet_light.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return output_path
            
        except Exception as e:
            print(f"   ❌ Error processing ConceptNet: {e}")
            return None
    
    def merge_targeted_datasets(self, processed_files: List[Path]) -> Optional[Path]:
        """Merge datasets with targeted weighting."""
        print("\n🎯 Creating Targeted RST Dataset")
        print("=" * 40)
        
        if not processed_files:
            print("❌ No processed files to merge")
            return None
        
        print(f"📊 Merging {len(processed_files)} datasets with targeted weights:")
        for dataset, weight in self.weights.items():
            print(f"   • {dataset:15s}: {weight:3.1f}x weight")
        
        merged_edges = {}
        total_stats = {}
        
        for file_path in processed_files:
            dataset_name = file_path.stem.replace('edges_', '')
            weight = self.weights.get(dataset_name.replace('_light', ''), 1.0)
            
            print(f"\n   📖 Merging {file_path.name} (weight: {weight}x)")
            
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
            
            total_stats[dataset_name] = {
                "edges": len(df),
                "weight_multiplier": weight,
                "edges_added": edges_added
            }
            
            print(f"      ✅ Added {edges_added:,} weighted edges")
        
        # Create final dataset
        print(f"\n🔗 Finalizing targeted dataset...")
        
        final_edges = []
        for (src, dst), weight in tqdm(merged_edges.items(), desc="Creating final dataset"):
            final_edges.append({
                'src': src,
                'dst': dst,
                'weight': round(weight, 6)
            })
        
        # Sort by weight for better analysis
        final_edges.sort(key=lambda x: x['weight'], reverse=True)
        
        # Save
        output_path = self.merged_dir / "targeted_rst_dataset.csv"
        df = pd.DataFrame(final_edges)
        df.to_csv(output_path, index=False)
        
        # Statistics
        unique_words = set()
        for src, dst in merged_edges.keys():
            unique_words.add(src)
            unique_words.add(dst)
        
        weights = list(merged_edges.values())
        
        print(f"\n✅ Targeted RST Dataset Created:")
        print(f"   📁 Output: {output_path}")
        print(f"   🔗 Total edges: {len(merged_edges):,}")
        print(f"   📝 Unique words: {len(unique_words):,}")
        print(f"   📊 Avg degree: {2 * len(merged_edges) / len(unique_words):.1f}")
        print(f"   ⚖️  Weight range: {min(weights):.3f} - {max(weights):.3f}")
        
        # Detailed source breakdown
        print(f"\n📈 Source Breakdown:")
        for dataset, stats in total_stats.items():
            priority = "PRIMARY" if "swow" in dataset else "SECONDARY" if "usf" in dataset else "FILLER"
            print(f"   • {dataset:15s}: {stats['edges']:6,} edges ({priority})")
        
        # Save metadata
        metadata = {
            "created_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Targeted RST dataset with SWOW-EN18 primary, USF secondary, ConceptNet filler",
            "weighting_strategy": "SWOW(3.0x) + USF(1.5x) + ConceptNet(0.3x)",
            "source_datasets": total_stats,
            "final_stats": {
                "total_edges": len(merged_edges),
                "unique_words": len(unique_words),
                "average_degree": 2 * len(merged_edges) / len(unique_words),
                "weight_range": [min(weights), max(weights)]
            },
            "optimization": "Optimized for RST trap word analysis"
        }
        
        with open(self.merged_dir / "targeted_rst_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return output_path


def main():
    """Run targeted dataset processing."""
    print("🎯 Targeted RST Dataset Processing")
    print("   SWOW-EN18 (primary) + USF (secondary) + ConceptNet (filler)")
    print("=" * 70)
    
    processor = TargetedDatasetProcessor()
    processed_files = []
    
    # Process SWOW-EN18 (highest priority)
    swow_file = processor.process_swow_en18()
    if swow_file:
        processed_files.append(swow_file)
    
    # Process USF (secondary)
    usf_file = processor.process_usf()
    if usf_file:
        processed_files.append(usf_file)
    
    # Process ConceptNet (filler)
    conceptnet_file = processor.process_conceptnet_light()
    if conceptnet_file:
        processed_files.append(conceptnet_file)
    
    # Merge with targeted weights
    if processed_files:
        merged_file = processor.merge_targeted_datasets(processed_files)
        
        if merged_file:
            print(f"\n🧪 Testing with RST framework...")
            try:
                sys.path.append('.')
                from rst_trap_finder.core import WordAssociationGraph
                
                graph = WordAssociationGraph.from_csv(merged_file)
                words = graph.get_all_words()
                edges = sum(len(neighbors) for neighbors in graph.graph.values())
                
                print(f"✅ Targeted dataset ready:")
                print(f"   📊 {len(words):,} words, {edges:,} edges")
                print(f"   🎯 Optimized for RST trap analysis")
                print(f"   🚀 Ready for test_framework.py")
                
            except Exception as e:
                print(f"⚠️  Dataset created but test failed: {e}")
    else:
        print("❌ No datasets processed successfully")


if __name__ == "__main__":
    main()