# CLIPSE Architecture

## Overview

CLIPSE implements CLIP training with SANW (Similarity-Aware Negative Weighting) variants for improved image-text retrieval performance.

## Core Components

### 1. Training Pipeline (`src/train_clipse.py`)

The main training script that orchestrates the entire training process:

- **Model Loading**: Uses OpenCLIP for model initialization with offline checkpoint support
- **Data Loading**: Loads Flickr8k dataset with proper train/val/test splits
- **Loss Computation**: Supports baseline InfoNCE, SANW-Debias, and SANW-Bandpass losses
- **Optimization**: Adam optimizer with cosine annealing and warmup scheduling
- **Logging**: Comprehensive CSV logging of metrics and training dynamics

### 2. Loss Functions (`src/losses/`)

#### Baseline InfoNCE (`clip_ce.py`)
Standard contrastive learning loss used as baseline.

#### SANW-Debias (`sanw_debias.py`)
Down-weights semantically close negatives to reduce false negatives:
- **Parameters**: `alpha` (strength), `delta` (threshold), `lambda` (weighting)
- **Mechanism**: Computes pairwise cosine similarities and applies debiasing weights

#### SANW-Bandpass (`sanw_bandpass.py`)
Emphasizes mid-hard negatives while avoiding very easy and very hard ones:
- **Parameters**: `alpha` (strength), `m1` (lower bound), `m2` (upper bound), `gamma` (weighting)
- **Mechanism**: Creates bandpass filter based on similarity scores

### 3. Data Handling (`src/data/`)

#### Dataset Loading (`datasets.py`)
- **Flickr8kDataset**: Loads and preprocesses Flickr8k image-caption pairs
- **Splits**: 30k train, 5k val, 5k test samples
- **Preprocessing**: OpenCLIP-compatible image transforms

#### Data Collation (`collate.py`)
- **CLIPCollator**: Efficient batch collation for CLIP training
- **Tokenization**: OpenCLIP tokenizer integration
- **Device Handling**: Proper tensor device management

### 4. Evaluation (`src/eval_retrieval.py`)

Comprehensive retrieval evaluation with metrics:
- **R@K**: Recall at K (K=1,5,10) for both Image→Text and Text→Image
- **MR**: Median Rank
- **mAP@10**: Mean Average Precision at 10

### 5. Utilities (`src/utils/`)

#### Logging (`logging.py`)
- **CSVLogger**: Structured logging for training metrics
- **Dynamic Fields**: Supports variable metric logging
- **Epoch/Step Logging**: Comprehensive training progress tracking

### 6. Sampling (`src/samplers/`)

#### Topical Batching (`topical_batch.py`)
- **Adaptive Sampling**: K-means clustering for topical batch construction
- **Semantic Coherence**: Groups semantically similar samples together
- **SANW Integration**: Enhances SANW effectiveness through better negative sampling

## Training Configuration

### Hyperparameters

#### SANW-Debias
- `alpha`: 0.1 (conservative) → 0.3 (aggressive)
- `delta`: 0.2 (threshold) → 0.5 (wider)
- `lambda`: 0.5 (weighting) → 2.0 (stronger)

#### SANW-Bandpass
- `alpha`: 0.1 (conservative) → 0.2 (aggressive)
- `m1`: 0.1 (lower bound) → 0.3 (narrower)
- `m2`: 0.9 (upper bound) → 0.8 (narrower)
- `gamma`: 0.5 (weighting) → 1.0 (stronger)

### Optimization
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: 1e-5 with cosine annealing
- **Warmup**: 1 epoch linear warmup
- **Gradient Clipping**: 1.0 norm
- **Mixed Precision**: Automatic (AMP)

## Cluster Integration

### SLURM Scripts (`scripts/`)
- **Environment**: Conda environment with CUDA PyTorch
- **Offline Mode**: Pre-staged model loading
- **Resource Management**: GPU allocation and memory limits
- **Logging**: Comprehensive job logging and error handling

### Model Management
- **Pre-staging**: Models downloaded and cached locally
- **Offline Loading**: No network dependencies during training
- **Checkpointing**: Regular model state saving

## Results and Analysis

### Metrics Tracked
- **Loss Convergence**: Training and validation loss curves
- **Temperature Dynamics**: Logit scale evolution
- **SANW Statistics**: Weight distributions and coverage
- **Retrieval Performance**: Comprehensive evaluation metrics

### Logging Structure
```
results/
├── training_*.csv          # Training metrics
├── run_registry.csv       # Experiment registry
└── checkpoints/           # Model checkpoints
```

## Future Extensions

1. **Multi-GPU Training**: Distributed training support
2. **Additional Datasets**: Support for other image-text datasets
3. **Advanced SANW Variants**: New weighting strategies
4. **Hyperparameter Optimization**: Automated parameter tuning
5. **Visualization**: Training dynamics and weight analysis
