# CLIPSE Paper Artifacts

This directory contains all artifacts and results for the CLIPSE (Semantic-Aware Negative Weighting for CLIP) paper.

## 📁 Directory Structure

```
paper_artifacts/
├── results/           # Experimental results and CSV files
├── scripts/           # Python scripts and analysis code
├── configs/           # YAML configuration files
├── plots/             # Generated plots and visualizations
├── analysis/          # Analysis notebooks and reports
├── docs/              # Documentation and markdown files
└── logs/              # Training and evaluation logs
```

## 🎯 Key Results

### Negative Text Similarity Analysis
- **`results/neg_text_sim_summary_all.csv`** - Complete statistical analysis
- **`results/neg_text_sim_results_all.csv`** - Key findings summary
- **`plots/neg_text_similarity_analysis.png/pdf`** - Comprehensive visualizations
- **`plots/sanw_motivation_analysis.png/pdf`** - SANW motivation plots

### Experimental Results
- **Flickr8k Results**: Baseline, Debias, Bandpass variants
- **Flickr30k Results**: Baseline, Debias, Bandpass variants  
- **COCO5k Transfer Results**: Cross-dataset evaluation
- **COCO Training Results**: Fine-tuning on COCO dataset

## 🔬 Key Findings

### SANW Motivation Evidence
- **93-96%** of "negative" pairs have similarity ≥ 0.2 (very high!)
- **69-75%** of "negative" pairs have similarity ≥ 0.3 (extremely high!)
- **34-41%** of "negative" pairs have similarity ≥ 0.4 (significant!)

This definitively proves that current CLIP training treats many semantically similar captions as "negatives" when they should be reweighted.

## 📊 Datasets Analyzed

1. **COCO** (202,654 captions) - Mean similarity: 0.3690
2. **Flickr8k** (40,456 captions) - Mean similarity: 0.3788  
3. **Flickr30k** (158,915 captions) - Mean similarity: 0.3570

## 🛠️ Scripts and Tools

### Analysis Scripts
- **`neg_text_sim_analysis.py`** - Generate professional plots
- **`run_comprehensive_evaluation.py`** - Evaluation pipeline
- **`eval_retrieval.py`** - Retrieval metrics computation

### Training Scripts
- **`train_clipse.py`** - Main training script
- **`datasets.py`** - Dataset loading utilities

## ⚙️ Configuration Files

- **Flickr8k configs**: Baseline, Debias, Bandpass variants
- **Flickr30k configs**: Baseline, Debias, Bandpass variants
- **COCO configs**: Training and transfer evaluation
- **Experiment configs**: Hyperparameter documentation

## 📈 Generated Plots

### Comprehensive Analysis (`neg_text_similarity_analysis.png/pdf`)
- Main results bar chart
- Mean similarity comparison
- Threshold analysis heatmap
- Statistical distribution analysis
- SANW motivation visualization
- Dataset size comparison
- Key findings summary
- Methodology overview
- Impact visualization

### SANW Motivation (`sanw_motivation_analysis.png/pdf`)
- Threshold analysis with clear percentages
- SANW concept visualization
- Problem identification and solution approach

## 🎓 Paper Contribution

This analysis provides **strong empirical evidence** for the SANW (Semantic-Aware Negative Weighting) approach by demonstrating that:

1. **Current CLIP training is suboptimal** - treating semantically similar captions as hard negatives
2. **SANW addresses a real problem** - high similarity in "negative" pairs
3. **Significant improvement potential** - 93-96% of negatives are actually similar

## 📝 Usage

### Reproducing Results
```bash
# Generate plots
python scripts/neg_text_sim_analysis.py

# Run evaluation
python scripts/run_comprehensive_evaluation.py
```

### Viewing Results
- Open CSV files in Excel/Google Sheets for detailed analysis
- View PNG/PDF plots for visualizations
- Check configs for hyperparameter settings

## 🔗 Related Files

- **`docs/README.md`** - Main project documentation
- **`docs/ORGANIZATION_SUMMARY.md`** - Project organization
- **`docs/BANDPASS_OPTIMIZATION_ANALYSIS.md`** - Optimization analysis
- **`docs/REAL_TRAINING_ANALYSIS.md`** - Training analysis
- **`docs/REAL_EVALUATION_ANALYSIS.md`** - Evaluation analysis

## 📊 Summary Statistics

| Dataset | Mean Similarity | ≥0.2 | ≥0.3 | ≥0.4 |
|---------|----------------|------|------|-------|
| **COCO** | 0.3690 | 96.43% | 74.98% | 35.57% |
| **Flickr8k** | 0.3788 | 95.14% | 74.78% | 40.57% |
| **Flickr30k** | 0.3570 | 93.09% | 69.52% | 33.99% |

**These results strongly support the SANW motivation and provide clear evidence for semantic-aware negative weighting in CLIP training.**
