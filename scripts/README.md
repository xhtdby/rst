# RST Trap Finder Scripts

This directory contains data processing pipelines and utility scripts for preparing word association datasets.

## 📊 Data Pipeline Scripts

### Core Integration Pipeline
- **[enhanced_dataset_integration.py](enhanced_dataset_integration.py)** - Download and process research datasets (SWOW/EAT/ConceptNet) (recommended)
- **[dataset_integration.py](dataset_integration.py)** - Basic dataset integration pipeline
- **[complete_dataset_merger.py](complete_dataset_merger.py)** - Merge processed datasets (legacy/simple pipeline)

### Dataset Processing & Optimization
- **[intelligent_dataset_reduction.py](intelligent_dataset_reduction.py)** - Create reduced datasets preserving useful structure
- **[targeted_dataset_processor.py](targeted_dataset_processor.py)** - Focused merging and filtering
- **[selective_pruner.py](selective_pruner.py)** - Heuristic pruning tooling

### Specialized Processors
- **[swow_manual_processor.py](swow_manual_processor.py)** - Manual SWOW dataset helpers

## 🚀 Quick Start

### Basic Dataset Setup
```bash
# Download and process all major datasets (recommended)
python scripts/enhanced_dataset_integration.py

# Create a reduced dataset for faster analysis
python scripts/intelligent_dataset_reduction.py
```

### Advanced Processing
```bash
# Targeted processing for specific research goals
python scripts/targeted_dataset_processor.py

# Custom pruning based on heuristics
python scripts/selective_pruner.py
```

## 📋 Pipeline Overview

### 1. **Data Acquisition & Integration**
- `enhanced_dataset_integration.py` - Downloads SWOW, EAT, ConceptNet datasets
- `dataset_integration.py` - Basic integration for quick setup

### 2. **Processing & Cleaning**
- Standardizes word formats and association weights
- Filters low-quality associations
- Merges complementary datasets

### 3. **Optimization & Reduction**
- `intelligent_dataset_reduction.py` - Preserves important graph structure
- `selective_pruner.py` - Removes low-value nodes/edges
- `targeted_dataset_processor.py` - Custom filtering strategies

### 4. **Output Organization**
All scripts output to organized data directories:
- `data/raw/` - Original downloaded datasets
- `data/processed/` - Cleaned and standardized data
- `data/merged/` - Combined datasets
- `data/reduced/` - Optimized datasets for analysis

## 💡 Usage Guidelines

### Running Scripts
- **Always run from repository root**: Scripts expect repo structure
- **Check dependencies**: Some scripts require internet access for downloads
- **Monitor outputs**: Scripts create progress logs and summary statistics

### Data Pipeline Flow
```
Raw Data → Processing → Merging → Reduction → Analysis-Ready
     ↓           ↓          ↓         ↓            ↓
  Download   Cleaning   Combining  Optimizing   Ready for RST
```

### Recommended Workflow
1. **Start with enhanced integration**: `python scripts/enhanced_dataset_integration.py`
2. **Create reduced dataset**: `python scripts/intelligent_dataset_reduction.py`
3. **Test with examples**: `python examples/game_launcher.py`
4. **Iterate with targeted processing** as needed

## 🔧 Script Features

### Data Sources Supported
- **SWOW (Small World of Words)** - Crowd-sourced word associations
- **EAT (Edinburgh Associative Thesaurus)** - Classic psychology dataset  
- **ConceptNet** - Large-scale semantic network
- **USF (University of South Florida)** - Free association norms

### Processing Capabilities
- **Format standardization** across different dataset schemas
- **Quality filtering** based on association strength and frequency
- **Graph optimization** preserving connectivity while reducing size
- **Metadata tracking** for reproducibility and debugging

### Output Formats
- **CSV files** for easy inspection and analysis
- **Parquet files** for efficient storage and loading
- **JSON metadata** with processing statistics and parameters

## 📊 Performance Notes

| Script | Typical Runtime | Memory Usage | Output Size |
|--------|-----------------|--------------|-------------|
| `enhanced_dataset_integration.py` | 5-15 minutes | 2-4 GB | 100-500 MB |
| `intelligent_dataset_reduction.py` | 2-5 minutes | 1-2 GB | 10-50 MB |
| `targeted_dataset_processor.py` | 1-3 minutes | 500 MB-1 GB | 5-100 MB |

## 🔍 Troubleshooting

### Common Issues
- **Memory errors**: Use reduction scripts for large datasets
- **Download failures**: Check internet connection and retry
- **Path errors**: Ensure running from repository root
- **Format errors**: Check input data format and encoding

### Debugging Tips
- Enable verbose logging with `--verbose` flag (where available)
- Check `data/processed/metadata_*.json` for processing statistics
- Use smaller datasets for testing with `--sample` options

---

*For more information about the RST analysis algorithms, see the `docs/` directory.*
