#!/usr/bin/env python3
"""
SWOW-EN18 Manual Processor - Handle large field sizes in SWOW data

This script specifically handles the SWOW-EN18 dataset with robust CSV parsing
to deal with large fields and unusual formatting.
"""

import pandas as pd
import json
import zipfile
import csv
import re
from pathlib import Path
from typing import Optional
import time

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


def process_swow_manual() -> Optional[Path]:
    """Manual processing of SWOW-EN18 with robust handling."""
    print("🎯 Manual SWOW-EN18 Processing")
    print("=" * 40)
    
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = raw_dir / "SWOW-EN18.zip"
    if not zip_path.exists():
        print(f"❌ SWOW-EN18.zip not found")
        return None
    
    output_path = processed_dir / "edges_swow_en18.csv"
    
    try:
        # Extract files
        extract_dir = processed_dir / "swow_temp"
        extract_dir.mkdir(exist_ok=True)
        
        print("   📦 Extracting SWOW-EN18.zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Find data files
        csv_files = list(extract_dir.glob("**/*.csv"))
        strength_file = None
        
        for file in csv_files:
            if "strength" in file.name.lower() and "r123" in file.name.lower():
                strength_file = file
                break
        
        if not strength_file:
            print(f"   ❌ No R123 strength file found")
            return None
        
        print(f"   📖 Processing {strength_file.name}...")
        
        # Manual line-by-line processing to handle large fields
        edges = []
        total_lines = 0
        header = None
        
        with open(strength_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f):
                total_lines += 1
                
                if line_num == 0:
                    # Parse header
                    header = line.strip().split('\t')
                    print(f"   📊 Header: {header[:5]}...")  # Show first 5 columns
                    continue
                
                if line_num % 10000 == 0:
                    print(f"      📊 Processed {line_num:,} lines, found {len(edges):,} edges...")
                
                try:
                    # Split on tabs, handling quotes
                    fields = line.strip().split('\t')
                    
                    if len(fields) < 3:
                        continue
                    
                    # Map fields to header
                    row_dict = {}
                    for i, field in enumerate(fields):
                        if i < len(header):
                            row_dict[header[i]] = field.strip('"').strip()
                    
                    # Extract cue, response, strength
                    cue = row_dict.get('cue', '').strip().lower()
                    response = row_dict.get('response', '').strip().lower()
                    
                    # Try different strength column names
                    strength = None
                    for col in ['strength.R123', 'R123.Strength', 'strength', 'R123']:
                        if col in row_dict and row_dict[col]:
                            try:
                                strength = float(row_dict[col])
                                break
                            except ValueError:
                                continue
                    
                    if strength is None or strength <= 0:
                        continue
                    
                    # Validate words
                    if (cue and response and cue != response and 
                        len(cue) <= 25 and len(response) <= 25 and
                        re.match(r'^[a-z]+$', cue) and re.match(r'^[a-z]+$', response)):
                        
                        edges.append({
                            'src': cue,
                            'dst': response,
                            'weight': strength
                        })
                
                except Exception as e:
                    # Skip problematic lines
                    continue
                
                # Limit processing for testing
                if line_num > 50000:  # Process first 50K lines for now
                    print(f"      🛑 Limiting to 50K lines for testing")
                    break
        
        # Save results
        if edges:
            df = pd.DataFrame(edges)
            df.to_csv(output_path, index=False)
            
            # Clean up
            import shutil
            shutil.rmtree(extract_dir)
            
            # Stats
            unique_words = set()
            for edge in edges:
                unique_words.add(edge['src'])
                unique_words.add(edge['dst'])
            
            print(f"   ✅ SWOW-EN18 processed:")
            print(f"      📊 {total_lines:,} lines processed")
            print(f"      🔗 {len(edges):,} valid associations")
            print(f"      📝 {len(unique_words):,} unique words")
            
            # Save metadata
            metadata = {
                "dataset": "SWOW-EN18-Partial",
                "source": "Small World of Words English 2018 (R123, partial)",
                "processed_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "lines_processed": total_lines,
                "valid_edges": len(edges),
                "unique_words": len(unique_words),
                "note": "Partial processing due to field size limitations"
            }
            
            with open(processed_dir / "metadata_swow_en18.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return output_path
        else:
            print(f"   ❌ No valid edges found")
            return None
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None


if __name__ == "__main__":
    result = process_swow_manual()
    if result:
        print(f"\n✅ SWOW processed: {result}")
        print("   🔄 Re-run targeted_dataset_processor.py to include SWOW in merge")
    else:
        print("\n❌ SWOW processing failed")