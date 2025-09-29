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

3. **Run experiments:**
   ```bash
   make run-baseline    # Vanilla CLIP
   make run-debias      # SANW debias mode
   make run-bandpass    # SANW bandpass mode
   ```

## Method

The key insight is that standard CLIP treats all off-diagonal pairs as equally negative, even when they're semantically related (e.g., different flower images in the same batch). SANW uses text-text and image-image similarities to:

- **Debias mode**: Down-weight near-positives to preserve local structure
- **Bandpass mode**: Emphasize hard negatives while avoiding false negatives

## Results

Check `runs/` directory for experiment outputs including:
- Training loss curves
- Retrieval R@1/R@5/R@10 metrics
- Zero-shot CIFAR-10 accuracy
- Model checkpoints
