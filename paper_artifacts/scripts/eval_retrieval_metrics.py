#!/usr/bin/env python3
"""
Retrieval evaluation script for CLIP with SANW variants.
Computes R@1/5/10, Median Rank, and mAP@10 for image-text retrieval.
"""

import os
import sys
import math
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import create_flickr8k_datasets
from src.data.collate import CLIPCollator
import open_clip


def compute_retrieval_metrics(logits_per_image, logits_per_text):
    """
    Compute retrieval metrics for image-text and text-image directions.
    
    Args:
        logits_per_image: [B, B] similarity matrix (image -> text)
        logits_per_text: [B, B] similarity matrix (text -> image)
    
    Returns:
        metrics: Dict with R@1/5/10, MR, mAP@10 for both directions
    """
    B = logits_per_image.size(0)
    device = logits_per_image.device
    
    # Image -> Text retrieval
    # For each image, find the rank of its correct text
    ranks_i2t = []
    for i in range(B):
        # Get similarities for image i with all texts
        sims = logits_per_image[i]  # [B]
        # Sort in descending order
        _, indices = torch.sort(sims, descending=True)
        # Find rank of correct text (index i)
        rank = (indices == i).nonzero(as_tuple=True)[0].item() + 1
        ranks_i2t.append(rank)
    
    # Text -> Image retrieval
    # For each text, find the rank of its correct image
    ranks_t2i = []
    for i in range(B):
        # Get similarities for text i with all images
        sims = logits_per_text[i]  # [B]
        # Sort in descending order
        _, indices = torch.sort(sims, descending=True)
        # Find rank of correct image (index i)
        rank = (indices == i).nonzero(as_tuple=True)[0].item() + 1
        ranks_t2i.append(rank)
    
    # Convert to numpy for easier computation
    ranks_i2t = np.array(ranks_i2t)
    ranks_t2i = np.array(ranks_t2i)
    
    # Compute metrics
    def compute_recall_at_k(ranks, k):
        return (ranks <= k).mean()
    
    def compute_median_rank(ranks):
        return np.median(ranks)
    
    def compute_map_at_k(ranks, k):
        """Compute mAP@k for retrieval."""
        # For each query, compute average precision
        aps = []
        for rank in ranks:
            if rank <= k:
                ap = 1.0 / rank  # Perfect precision for rank 1
            else:
                ap = 0.0
            aps.append(ap)
        return np.mean(aps)
    
    metrics = {
        # Image -> Text
        'R1_I2T': compute_recall_at_k(ranks_i2t, 1),
        'R5_I2T': compute_recall_at_k(ranks_i2t, 5),
        'R10_I2T': compute_recall_at_k(ranks_i2t, 10),
        'MR_I2T': compute_median_rank(ranks_i2t),
        'mAP10_I2T': compute_map_at_k(ranks_i2t, 10),
        
        # Text -> Image
        'R1_T2I': compute_recall_at_k(ranks_t2i, 1),
        'R5_T2I': compute_recall_at_k(ranks_t2i, 5),
        'R10_T2I': compute_recall_at_k(ranks_t2i, 10),
        'MR_T2I': compute_median_rank(ranks_t2i),
        'mAP10_T2I': compute_map_at_k(ranks_t2i, 10),
        
        # Average metrics
        'R1_avg': (compute_recall_at_k(ranks_i2t, 1) + compute_recall_at_k(ranks_t2i, 1)) / 2,
        'R5_avg': (compute_recall_at_k(ranks_i2t, 5) + compute_recall_at_k(ranks_t2i, 5)) / 2,
        'R10_avg': (compute_recall_at_k(ranks_i2t, 10) + compute_recall_at_k(ranks_t2i, 10)) / 2,
        'MR_avg': (compute_median_rank(ranks_i2t) + compute_median_rank(ranks_t2i)) / 2,
        'mAP10_avg': (compute_map_at_k(ranks_i2t, 10) + compute_map_at_k(ranks_t2i, 10)) / 2,
    }
    
    return metrics


def evaluate_model(model, dataloader, device, method_name="baseline"):
    """
    Evaluate model on retrieval task.
    
    Args:
        model: CLIP model
        dataloader: DataLoader for evaluation
        device: Device to run on
        method_name: Name of the method for logging
    
    Returns:
        metrics: Dict with retrieval metrics
    """
    model.eval()
    
    all_logits_i2t = []
    all_logits_t2i = []
    
    print(f"Evaluating {method_name} on {len(dataloader)} batches...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {method_name}"):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            
            # Get features
            image_features = model.encode_image(pixel_values)
            text_features = model.encode_text(input_ids)
            
            # Normalize features
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            # Compute logits with temperature
            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            
            all_logits_i2t.append(logits_per_image.cpu())
            all_logits_t2i.append(logits_per_text.cpu())
    
    # Concatenate all logits
    all_logits_i2t = torch.cat(all_logits_i2t, dim=0)
    all_logits_t2i = torch.cat(all_logits_t2i, dim=0)
    
    print(f"Computing retrieval metrics for {all_logits_i2t.size(0)} samples...")
    
    # Compute metrics
    metrics = compute_retrieval_metrics(all_logits_i2t, all_logits_t2i)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP with SANW variants on retrieval")
    parser.add_argument("--model_name", type=str, default="ViT-B-32", help="OpenCLIP model name")
    parser.add_argument("--pretrained_tag", type=str, default="laion2b_s34b_b79k", help="OpenCLIP pretrained tag")
    parser.add_argument("--local_checkpoint_path", type=str, default=None, help="Path to local checkpoint")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data workers")
    parser.add_argument("--output_file", type=str, default="retrieval_metrics.csv", help="Output CSV file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    if args.local_checkpoint_path and os.path.exists(args.local_checkpoint_path):
        model = open_clip.create_model(args.model_name, pretrained=args.local_checkpoint_path, device=device)
        tokenizer = open_clip.get_tokenizer(args.model_name)
        import torchvision.transforms as transforms
        preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                              std=[0.26862954, 0.26130258, 0.27577711])
        ])
    else:
        model, preprocess, tokenizer = open_clip.create_model_and_transforms(
            args.model_name, pretrained=args.pretrained_tag, device=device
        )
    
    # Create dataset
    print("Creating dataset...")
    _, val_loader, test_loader = create_flickr8k_datasets(
        args.dataset_path,
        preprocess_fn=preprocess,
        tokenizer_fn=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device
    )
    
    # Evaluate on validation set
    print("Evaluating on validation set...")
    val_metrics = evaluate_model(model, val_loader, device, "validation")
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device, "test")
    
    # Print results
    print("\n" + "="*60)
    print("RETRIEVAL EVALUATION RESULTS")
    print("="*60)
    
    print("\nValidation Set:")
    for key, value in val_metrics.items():
        if 'avg' in key:
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value:.4f}")
    
    print("\nTest Set:")
    for key, value in test_metrics.items():
        if 'avg' in key:
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value:.4f}")
    
    # Save results to CSV
    results = {
        'method': args.model_name,
        'pretrained': args.pretrained_tag,
        'seed': args.seed,
        **{f'val_{k}': v for k, v in val_metrics.items()},
        **{f'test_{k}': v for k, v in test_metrics.items()}
    }
    
    df = pd.DataFrame([results])
    df.to_csv(args.output_file, index=False)
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
