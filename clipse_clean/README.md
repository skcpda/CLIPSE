# CLIPSE: CLIP with SANW (Similarity-Aware Negative Weighting)

A research implementation of CLIP training with SANW-Debias and SANW-Bandpass loss functions for improved image-text retrieval.

## Overview

This project implements two variants of Similarity-Aware Negative Weighting (SANW) for CLIP training:

- **SANW-Debias**: Down-weights semantically close negatives to reduce false negatives
- **SANW-Bandpass**: Emphasizes mid-hard negatives while avoiding very easy and very hard ones

## Project Structure

```
clipse/
├── src/                    # Core source code
│   ├── train_clipse.py    # Main training script
│   ├── eval_retrieval.py  # Retrieval evaluation script
│   ├── datasets.py        # Dataset loading utilities
│   ├── losses/            # Loss function implementations
│   ├── data/              # Data collation utilities
│   ├── utils/             # Logging and utilities
│   └── samplers/          # Topical batch sampling
├── scripts/               # SLURM job scripts
├── configs/               # Configuration files
├── models/                # Pre-trained model storage
├── results/               # Training results and logs
└── docs/                  # Documentation
```

## Quick Start

### Training

```bash
# SANW-Debias training
sbatch scripts/train_sanw_debias.sbatch

# SANW-Bandpass training  
sbatch scripts/train_sanw_bandpass.sbatch
```

### Evaluation

```bash
# Evaluate trained model
sbatch scripts/eval_retrieval.sbatch
```

## Requirements

- Python 3.10+
- PyTorch 2.5+
- OpenCLIP
- CUDA support for GPU training

## Configuration

Training parameters can be configured in the SLURM scripts or via command line arguments:

- `--loss_fn`: Loss function (`clip_ce`, `sanw_debias`, `sanw_bandpass`)
- `--sanw_alpha`: Alpha parameter for SANW losses
- `--sanw_delta`: Delta parameter for SANW-Debias
- `--sanw_lambda`: Lambda parameter for SANW-Debias
- `--sanw_m1`, `--sanw_m2`: Band parameters for SANW-Bandpass
- `--sanw_gamma`: Gamma parameter for SANW-Bandpass

## Results

Training results are logged to CSV files in the `results/` directory with metrics including:
- Loss values and convergence
- Temperature (logit_scale) dynamics
- SANW weight statistics
- Retrieval metrics (R@1, R@5, R@10, MR, mAP@10)
