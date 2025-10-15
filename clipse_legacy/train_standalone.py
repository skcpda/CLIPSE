#!/usr/bin/env python3
"""
Standalone training script for SANW experiments.
"""
import os
import sys
import math
import time
import argparse
import yaml
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor

# Add src to path
sys.path.insert(0, 'src')

# Import our modules
from losses import clip_ce_loss, sanw_debias_loss, sanw_bandpass_loss
from data import CLIPCollator
from utils import CSVLogger
from datasets import create_flickr8k_datasets


def build_model(model_id: str):
    """Build CLIP model and processor."""
    # Check if it's a local path or HuggingFace model ID
    if os.path.exists(model_id):
        # Local model directory
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HOME"] = os.path.dirname(model_id)
        model = CLIPModel.from_pretrained(model_id, local_files_only=True)
        processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
    else:
        # HuggingFace model ID - download online with safetensors
        model = CLIPModel.from_pretrained(model_id, use_safetensors=True)
        processor = CLIPProcessor.from_pretrained(model_id)
    
    return model, processor


def select_device():
    """Select training device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def encode_img_txt(model, pixel_values, input_ids, attention_mask):
    """Extract normalized image and text features from CLIP model."""
    with torch.no_grad():
        # Get features from vision and text encoders
        img = model.vision_model(pixel_values=pixel_values).pooler_output
        txt = model.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
    
    # Normalize features as in CLIP
    img = F.normalize(img, dim=-1)
    txt = F.normalize(txt, dim=-1)
    
    return img, txt


def train(cfg):
    """Main training function."""
    # Set random seeds
    torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg['seed'])
    
    # Setup device
    device = select_device()
    print(f"Using device: {device}")
    
    # Build model
    print("Loading CLIP model...")
    model, processor = build_model(cfg['model_dir'])
    model = model.to(device)
    
    # Setup data
    print("Setting up data...")
    dataset_root = cfg.get('dataset', {}).get('root', '/nfs_home/users/poonam/sanw_experiments/data/flickr8k')
    train_loader, val_loader, test_loader = create_flickr8k_datasets(
        dataset_root,
        batch_size=cfg['batch_size'],
        num_workers=cfg.get('num_workers', 0)
    )
    
    # Setup optimizer and scaler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg['lr'], 
        weight_decay=cfg.get('wd', 0.01)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and cfg.get('amp', True)))
    
    # Setup logging
    log_file = f"logs/training_{cfg.get('experiment_name', 'clip')}.csv"
    logger = CSVLogger(log_file)
    
    # Training loop
    print("Starting training...")
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        epoch_start = time.time()
        
        for step, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=(device == "cuda" and cfg.get('amp', True))):
                out = model(**batch)
                
                # Extract features for SANW losses
                img_feats, txt_feats = encode_img_txt(
                    model, 
                    batch["pixel_values"],
                    batch["input_ids"], 
                    batch["attention_mask"]
                )
                
                # Compute loss
                loss_name = cfg['loss']['name']
                if loss_name == "baseline":
                    loss = clip_ce_loss(out.logits_per_image, out.logits_per_text)
                    stats = {}
                elif loss_name == "sanw_debias":
                    loss, stats = sanw_debias_loss(
                        out.logits_per_image, out.logits_per_text,
                        img_feats, txt_feats,
                        alpha=cfg['loss']['alpha'],
                        delta=cfg['loss']['delta'],
                        lam=cfg['loss']['lam']
                    )
                elif loss_name == "sanw_bandpass":
                    loss, stats = sanw_bandpass_loss(
                        out.logits_per_image, out.logits_per_text,
                        img_feats, txt_feats,
                        alpha=cfg['loss']['alpha'],
                        m1=cfg['loss']['m1'],
                        m2=cfg['loss']['m2'],
                        gamma=cfg['loss']['gamma']
                    )
                else:
                    raise ValueError(f"Unknown loss: {loss_name}")
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
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
        print(f"Epoch {epoch} completed in {epoch_time:.1f}s")
        
        # Save checkpoint
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': cfg
        }, checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")
    
    print("Training completed!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train CLIP with SANW variants")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Expand environment variables
    for key, value in cfg.items():
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]
            cfg[key] = os.environ.get(env_var, value)
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Start training
    train(cfg)


if __name__ == "__main__":
    main()
