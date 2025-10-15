"""
Main training script for CLIP with SANW variants using OpenCLIP.
"""
import os
import math
import time
import argparse
import yaml
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


def build_model(model_name: str, pretrained_tag: str, device: str, local_checkpoint_path: str = None):
    """Build OpenCLIP model, preprocess function, and tokenizer."""
    if local_checkpoint_path and os.path.exists(local_checkpoint_path):
        print(f"Loading OpenCLIP model from local checkpoint: {local_checkpoint_path}")
        model = open_clip.create_model(model_name, pretrained=local_checkpoint_path, device=device)
        # When loading from a local checkpoint, we need to get the tokenizer separately
        # to avoid downloading from HuggingFace Hub
        tokenizer = open_clip.get_tokenizer(model_name)
        # For preprocess, we'll use a simple transform since we have the model loaded
        import torchvision.transforms as transforms
        preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                              std=[0.26862954, 0.26130258, 0.27577711])
        ])
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
    
    # Setup data
    print("Setting up data...")
    from src.datasets import create_flickr8k_datasets
    
    # Create dataset loaders
    train_loader, val_loader, test_loader = create_flickr8k_datasets(
        cfg['dataset_path'],
        preprocess_fn=preprocess,
        tokenizer_fn=tokenizer,
        batch_size=cfg['batch_size'],
        num_workers=cfg.get('num_workers', 0),
        device=device
    )
    
    # Setup optimizer and scaler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg['lr'], 
        weight_decay=cfg.get('wd', 0.01)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and cfg.get('amp', True)))
    
    # Setup logging
    log_file = f"{cfg['log_dir']}/training_{cfg.get('run_id', 'clip')}.csv"
    os.makedirs(cfg['log_dir'], exist_ok=True)
    logger = CSVLogger(log_file)
    
    # Training loop
    print("Starting training...")
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        epoch_start = time.time()
        
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
                
                # Compute logits
                logit_scale = model.logit_scale.exp()
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
                        lam=cfg['sanw_lambda']
                    )
                elif loss_name == "sanw_bandpass":
                    loss, stats = sanw_bandpass_loss(
                        logits_per_image, logits_per_text,
                        image_features, text_features,
                        alpha=cfg['sanw_alpha'],
                        m1=cfg['sanw_m1'],
                        m2=cfg['sanw_m2'],
                        gamma=cfg['sanw_gamma']
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
        checkpoint_dir = Path(cfg['checkpoint_dir'])
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
    parser.add_argument("--model_name", type=str, default="ViT-B-32", help="OpenCLIP model name")
    parser.add_argument("--pretrained_tag", type=str, default="laion2b_s34b_b79k", help="OpenCLIP pretrained tag")
    parser.add_argument("--local_checkpoint_path", type=str, default=None, help="Path to local checkpoint")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
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
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Warmup epochs")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--log_every", type=int, default=20, help="Log every N steps")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data workers")
    
    args = parser.parse_args()
    
    # Convert args to config dict
    cfg = vars(args)
    
    # Start training
    train(cfg)


if __name__ == "__main__":
    main()