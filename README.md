# CLIP Similarity-Aware Negative Weighting (SANW)

This project implements and evaluates Similarity-Aware Negative Weighting for CLIP training, addressing the false negative problem in contrastive learning.

## Quick Start

1. **Setup environment:**
   ```bash
   make setup
   ```

2. **Prepare data:**
   - Place Flickr8k images in `data/flickr8k/Images/`
   - Create `data/flickr8k/captions.tsv` with format: `image_path<TAB>caption`
   - Example: `Images/12345.jpg	A close-up photo of a pink flower`

3. **Run experiments:**
   ```bash
   make run-baseline    # Vanilla CLIP (20 epochs, batch 256)
   make run-debias      # SANW debias mode 
   make run-bandpass    # SANW bandpass mode
   make run-all-seeds   # Run all experiments with seeds {13, 17, 23}
   ```

## Method

The key insight is that standard CLIP treats all off-diagonal pairs as equally negative, even when they're semantically related (e.g., different flower images in the same batch). SANW uses text-text and image-image similarities to:

- **Debias mode**: Down-weight near-positives to preserve local structure
- **Bandpass mode**: Emphasize hard negatives while avoiding false negatives

## Key Features

### **Topical Batching**
- **K-means clustering** of captions (k=80 clusters)
- **Probability-based sampling**: p_topical=0.5-0.6 for topical batches
- **Cluster refresh**: Recompute clusters every 2 epochs
- **Cross-cluster spill**: Allow 10% spillover for small clusters

### **Advanced Training**
- **Model**: ViT-B/32 with CLIP text transformer
- **Optimizer**: AdamW (β1=0.9, β2=0.98, eps=1e-6, wd=0.2)
- **LR Schedule**: Cosine decay with 1-epoch warmup (base LR: 5e-4)
- **Precision**: FP16/AMP for faster training
- **Logit Scale**: Initialized at ln(1/0.07) ≈ 2.659, clamped to ln(100)

### **Comprehensive Evaluation**
- **Retrieval**: Both I→T and T→I directions (R@1, R@5, R@10)
- **Zero-shot**: CIFAR-10 classification
- **Metrics**: Mean R@1 as primary model selection criterion

## Expected Results

Under **topical batches** (semantically related images in same batch):
- **SANW should outperform baseline** by 2-5 points in R@5/R@10
- **Zero-shot CIFAR-10 accuracy** should improve by 0.5-2%
- **Training stability** should be better (lower variance)

Under **uniform batches** (random sampling):
- **Gaps may shrink** as false-negative pressure is reduced
- **Baseline and SANW should converge** to similar performance

## Outputs

Check `runs/` directory for experiment outputs:
- **`metrics.csv`**: Per-epoch metrics (R@1/R@5/R@10, zero-shot, temperature, LR)
- **`best_model.pt`**: Best model based on mean R@1
- **`epoch_XX.pt`**: Epoch checkpoints
- **Console logs**: Training progress with loss, LR, temperature

## Configuration

Three experiment configs in `configs/`:
- **`flickr8k_vitb32_baseline.yaml`**: Vanilla CLIP with topical batches
- **`flickr8k_vitb32_debias.yaml`**: SANW debias mode (α=0.5, δ=0.6, λ=4.0)
- **`flickr8k_vitb32_bandpass.yaml`**: SANW bandpass mode (α=0.5, m1=0.3, m2=0.8, γ=0.05)

All configs use identical hyperparameters for fair comparison across seeds {13, 17, 23}.

## Success Criteria

**We call it a win if:**
- At **equal compute**, SANW-Debias and/or Band-Pass gives a **consistent gain in R@1 or R@5** (I→T or T→I) across **all 3 seeds**
- Training slowdown is **≤10%** compared to baseline
- Results are **statistically significant** across multiple runs
