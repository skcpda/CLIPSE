"""
Main training script for CLIP with SANW variants using OpenCLIP.
"""
import os
import math
import time
import argparse
import yaml
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import open_clip

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.losses import clip_ce_loss, sanw_debias_loss, sanw_bandpass_loss
from src.data.collate import CLIPCollator
from src.utils.logging import CSVLogger

def ce_with_sanw(L, pos_idx, W, beta=0.2, eps=1e-6):
    """
    CE loss with SANW denominator modulation.
    L: [B,B] logits (image->text or text->image)
    pos_idx: [B] positive column indices (usually torch.arange(B))
    W: [B,B] SANW weights for negatives (row-normalized, diag=0, >=eps)
    beta: reweight strength (0 => vanilla CE)
    """
    B = L.size(0)
    expL = torch.exp(L)

    eye = torch.eye(B, device=L.device, dtype=torch.bool)
    neg_scale = 1 + beta * W  # >= 1
    D = expL * torch.where(eye, torch.ones_like(expL), neg_scale)

    num = L[torch.arange(B), pos_idx]
    den = D.sum(dim=1).clamp_min(eps)
    return -(num - torch.log(den)).mean()


def build_model(model_name: str, pretrained_tag: str, device: str, local_checkpoint_path: str = None):
    """Build OpenCLIP model, preprocess function, and tokenizer.

    Supports two paths:
    - Online/pretrained tag
    - Offline local checkpoint saved as a state_dict (with or without 'model_state_dict' key)
    """
    if local_checkpoint_path and os.path.exists(local_checkpoint_path):
        print(f"Loading OpenCLIP model weights from local file: {local_checkpoint_path}")
        # Build architecture & transforms without attempting to download weights
        model, _preprocess_train, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=None, device=device
        )
        tokenizer = open_clip.get_tokenizer(model_name)

        # Load the saved state dict (support either raw dict or dict with 'model_state_dict')
        sd = torch.load(local_checkpoint_path, map_location="cpu")
        if isinstance(sd, dict) and 'model_state_dict' in sd:
            sd = sd['model_state_dict']
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[build_model] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[build_model] Unexpected keys: {len(unexpected)}")
    else:
        print(f"Loading OpenCLIP model: {model_name}, pretrained: {pretrained_tag}")
        model, preprocess, tokenizer = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_tag, device=device
        )
    return model, preprocess, tokenizer


def encode_img_txt(model, pixel_values, input_ids):
    """Extract normalized image and text features from OpenCLIP model."""
    with torch.no_grad():
        image_features = model.encode_image(pixel_values)
        text_features = model.encode_text(input_ids)
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
    
    return image_features, text_features


def train(cfg):
    """Main training function."""
    # Set random seeds
    torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg['seed'])
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Build model
    print("Loading OpenCLIP model...")
    model, preprocess, tokenizer = build_model(cfg['model_name'], cfg['pretrained_tag'], device, cfg.get('local_checkpoint_path'))
    model = model.to(device)
    
    # Initialize logit_scale to a sane starting point
    with torch.no_grad():
        # OpenAI init is ln(1/0.07) ≈ 2.659 -> temp ≈ 14.3
        model.logit_scale.fill_(math.log(1/0.07))
    
    # Optional: freeze logit_scale for warmup epochs
    freeze_logit_until = cfg.get("freeze_logit_until", 2)
    if freeze_logit_until > 0:
        model.logit_scale.requires_grad = False
    
    # Setup data
    print("Setting up data...")
    # Select dataset factory based on dataset_name
    dataset_name = cfg.get('dataset_name', 'flickr8k').lower()
    if dataset_name == 'flickr30k':
        from src.datasets import create_flickr30k_datasets as create_datasets
    elif dataset_name == 'coco':
        from src.datasets import create_coco_datasets as create_datasets
    else:
        from src.datasets import create_flickr8k_datasets as create_datasets

    # Create dataset loaders
    train_loader, val_loader, test_loader = create_datasets(
        cfg['dataset_path'],
        preprocess_fn=preprocess,
        tokenizer_fn=tokenizer,
        batch_size=cfg['batch_size'],
        num_workers=cfg.get('num_workers', 0),
        device=device
    )
    
    # Setup optimizer with separate groups for logit_scale
    main_params, logit_params = [], []
    for n, p in model.named_parameters():
        if n == "logit_scale":
            logit_params.append(p)
        else:
            main_params.append(p)
    
    optimizer = torch.optim.AdamW([
        {"params": main_params, "lr": cfg['lr'], "weight_decay": cfg.get('wd', 0.01)},
        # Very small LR and NO weight decay for temperature
        {"params": logit_params, "lr": cfg.get('lr_logit_scale', 1e-6), "weight_decay": 0.0},
    ])
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and cfg.get('amp', True)))
    
    # Setup logging
    log_file = f"{cfg['log_dir']}/training_{cfg.get('run_id', 'clip')}.csv"
    os.makedirs(cfg['log_dir'], exist_ok=True)
    logger = CSVLogger(log_file)
    
    # Training loop
    print("Starting training...")
    run_start_time_sec = time.time()
    per_epoch_durations_sec = []
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        epoch_start = time.time()

        # SANW beta warmup (linear ramp over specified epochs)
        if cfg.get('sanw_beta_warmup_epochs', 0) > 0:
            warm = cfg['sanw_beta_warmup_epochs']
            # progress from 0 to 1 over warm epochs, then 1 afterwards
            ramp = min(1.0, max(0.0, (epoch - 1) / max(1, warm)))
            cfg['sanw_beta_eff'] = cfg.get('sanw_beta', 0.2) * ramp
        else:
            cfg['sanw_beta_eff'] = cfg.get('sanw_beta', 0.2)
        
        for step, batch in enumerate(train_loader):
            # Move batch to device
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=(device == "cuda" and cfg.get('amp', True))):
                # Get image and text features
                image_features = model.encode_image(pixel_values)
                text_features = model.encode_text(input_ids)
                
                # Normalize features
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
                # Compute logits with temperature clamping
                logit_scale = model.logit_scale.clamp(-2.0, 4.6052).exp()  # clamp to [0.135, 100]
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()
                
                # Compute loss
                loss_name = cfg['loss_fn']
                if loss_name == "baseline":
                    loss = clip_ce_loss(logits_per_image, logits_per_text)
                    stats = {}
                elif loss_name == "sanw_debias":
                    loss, stats = sanw_debias_loss(
                        logits_per_image, logits_per_text,
                        image_features, text_features,
                        alpha=cfg['sanw_alpha'],
                        delta=cfg['sanw_delta'],
                        lam=cfg['sanw_lambda'],
                        beta=cfg.get('sanw_beta_eff', cfg.get('sanw_beta', 0.2))
                    )
                elif loss_name == "sanw_bandpass":
                    loss, stats = sanw_bandpass_loss(
                        logits_per_image, logits_per_text,
                        image_features, text_features,
                        alpha=cfg['sanw_alpha'],
                        m1=cfg['sanw_m1'],
                        m2=cfg['sanw_m2'],
                        gamma=cfg['sanw_gamma'],
                        beta=cfg.get('sanw_beta_eff', cfg.get('sanw_beta', 0.2))
                    )
                else:
                    raise ValueError(f"Unknown loss: {loss_name}")
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if cfg.get('grad_clip', 0.0) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            
            scaler.step(optimizer)
            scaler.update()
            
            # Clamp temperature to prevent saturation (much narrower clamp)
            with torch.no_grad():
                # Keep logit_scale in [ln(1/10), ln(50)] -> temp in [0.1, 50]
                model.logit_scale.clamp_(math.log(1/10), math.log(50))
            
            # Unfreeze logit_scale after warmup epochs
            if freeze_logit_until and (epoch == freeze_logit_until):
                model.logit_scale.requires_grad = True
            
            optimizer.zero_grad(set_to_none=True)
            
            # Logging
            if (step + 1) % cfg.get('log_every', 20) == 0:
                temp = float(model.logit_scale.exp().detach().cpu())
                lr = optimizer.param_groups[0]['lr']
                
                logger.log_step(
                    epoch=epoch,
                    step=step + 1,
                    loss=loss.item(),
                    temp=temp,
                    lr=lr,
                    **stats
                )
                
                print(f"Epoch {epoch}, Step {step+1}: loss={loss.item():.4f}, temp={temp:.3f}")
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        per_epoch_durations_sec.append(float(epoch_time))
        print(f"Epoch {epoch} completed in {epoch_time:.1f}s")

        # Persist epoch timing into the training CSV
        logger.log_epoch(epoch=epoch, epoch_time_sec=float(epoch_time))
        
        # Save checkpoint
        checkpoint_dir = Path(cfg['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': cfg
        }, checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")
    
    # Final training summary (wall time and GPU-hours)
    total_training_time_sec = time.time() - run_start_time_sec
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    # Guard against 0 GPUs when running on CPU; we still report wall-clock hours
    gpu_hours = (total_training_time_sec / 3600.0) * float(max(1, num_gpus))

    # Write a compact run summary CSV alongside training logs
    summary_file = Path(cfg['log_dir']) / "run_summaries.csv"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_row = {
        "run_id": cfg.get("run_id", "unknown"),
        "method": cfg.get("loss_fn", "baseline"),
        "model_name": cfg.get("model_name", ""),
        "pretrained_tag": cfg.get("pretrained_tag", ""),
        "dataset_name": cfg.get("dataset_name", ""),
        "seed": cfg.get("seed", 0),
        "epochs": cfg.get("epochs", 0),
        "batch_size": cfg.get("batch_size", 0),
        "num_gpus": num_gpus,
        "total_time_sec": float(total_training_time_sec),
        "avg_epoch_time_sec": float(total_training_time_sec / max(1, cfg.get("epochs", 1))),
        "gpu_hours": float(gpu_hours),
    }
    header = list(summary_row.keys())
    write_header = not summary_file.exists()
    with open(summary_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(summary_row)

    print("Training completed!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train CLIP with SANW variants")
    parser.add_argument("--model_name", type=str, default="ViT-B-32", help="OpenCLIP model name")
    parser.add_argument("--pretrained_tag", type=str, default="laion2b_s34b_b79k", help="OpenCLIP pretrained tag")
    parser.add_argument("--local_checkpoint_path", type=str, default=None, help="Path to local checkpoint")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--dataset_name", type=str, default="flickr8k", help="Dataset name: flickr8k|flickr30k|coco (currently flickr8k/flickr30k supported)")
    parser.add_argument("--log_dir", type=str, required=True, help="Log directory")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--run_id", type=str, required=True, help="Run ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--loss_fn", type=str, default="baseline", help="Loss function")
    parser.add_argument("--sanw_alpha", type=float, default=0.1, help="SANW alpha parameter")
    parser.add_argument("--sanw_delta", type=float, default=0.2, help="SANW delta parameter")
    parser.add_argument("--sanw_lambda", type=float, default=0.5, help="SANW lambda parameter")
    parser.add_argument("--sanw_m1", type=float, default=0.1, help="SANW m1 parameter")
    parser.add_argument("--sanw_m2", type=float, default=0.9, help="SANW m2 parameter")
    parser.add_argument("--sanw_gamma", type=float, default=0.5, help="SANW gamma parameter")
    parser.add_argument("--sanw_beta", type=float, default=0.2, help="SANW beta parameter (strength of denominator modulation)")
    parser.add_argument("--sanw_beta_warmup_epochs", type=int, default=0, help="Warmup epochs to linearly ramp SANW beta from 0 to target")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Warmup epochs")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--log_every", type=int, default=20, help="Log every N steps")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data workers")
    parser.add_argument("--freeze_logit_scale_epochs", type=int, default=0, help="Number of epochs to freeze logit scale")
    
    args = parser.parse_args()
    
    # Convert args to config dict
    cfg = vars(args)
    
    # Start training
    train(cfg)


if __name__ == "__main__":
    main()