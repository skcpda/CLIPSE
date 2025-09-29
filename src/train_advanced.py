import os, yaml, math, random, csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import open_clip
from tqdm import tqdm
import time

from .datasets import FlickrTSV, collate_text_tokenize
from .dummy_dataset import DummyFlickrDataset
from .topical_batch_sampler import TopicalBatchSampler
from .losses import clip_ce_loss, sanw_debias_loss, sanw_bandpass_loss
from .eval_utils import evaluate_retrieval, evaluate_zero_shot_cifar10

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """Create cosine LR schedule with warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg.get("seed", 42))
    
    # Handle device selection
    device_config = cfg.get("device", "auto")
    if device_config == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_config
    
    print(f"Using device: {device}")
    if device == "cpu":
        print("Note: Training on CPU will be slower. Consider using CUDA if available.")

    os.makedirs(cfg["output_dir"], exist_ok=True)

    # Model + tokenizer
    model_name = cfg["model"]["name"]
    pretrained = cfg["model"]["pretrained"]
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)

    model = model.to(device)
    model.train()

    # Initialize logit scale
    logit_scale_cfg = cfg.get("logit_scale", {})
    init_scale = logit_scale_cfg.get("init", 2.659)
    clamp_max = logit_scale_cfg.get("clamp_max", 4.605170)
    
    with torch.no_grad():
        model.logit_scale.data.fill_(init_scale)
    model.logit_scale.requires_grad_(True)

    # Data
    use_dummy = cfg["data"].get("use_dummy", False)
    if use_dummy:
        train_ds = DummyFlickrDataset(num_samples=200, split="train", val_ratio=0.1, transform=preprocess)
        val_ds = DummyFlickrDataset(num_samples=200, split="val", val_ratio=0.1, transform=preprocess)
    else:
        ds_cfg = cfg["data"]["flickr8k"]
        train_ds = FlickrTSV(ds_cfg["root"], ds_cfg["tsv"], split="train", val_ratio=0.1, transform=preprocess)
        val_ds = FlickrTSV(ds_cfg["root"], ds_cfg["tsv"], split="val", val_ratio=0.1, transform=preprocess)

    # Topical batching
    topical_cfg = cfg["data"].get("topical_batching", {})
    if topical_cfg.get("enabled", False):
        batch_sampler = TopicalBatchSampler(
            train_ds, 
            batch_size=cfg["data"]["batch_size"],
            k_clusters=topical_cfg.get("k_clusters", 80),
            p_topical=topical_cfg.get("p_topical", 0.5),
            refresh_every_epochs=topical_cfg.get("refresh_every_epochs", 2),
            sampler_temperature=topical_cfg.get("sampler_temperature", 0.7),
            seed=cfg.get("seed", 42)
        )
        train_loader = DataLoader(
            train_ds, 
            batch_sampler=batch_sampler,
            num_workers=cfg["data"]["num_workers"],
            collate_fn=lambda b: collate_text_tokenize(b, tokenizer)
        )
    else:
        train_loader = DataLoader(
            train_ds, 
            batch_size=cfg["data"]["batch_size"], 
            shuffle=True,
            num_workers=cfg["data"]["num_workers"],
            collate_fn=lambda b: collate_text_tokenize(b, tokenizer)
        )

    # Optimizer
    optim_cfg = cfg["optim"]
    params = list(model.parameters())
    opt = torch.optim.AdamW(
        params, 
        lr=optim_cfg["lr"], 
        betas=optim_cfg.get("betas", [0.9, 0.98]),
        eps=optim_cfg.get("eps", 1e-6),
        weight_decay=optim_cfg["weight_decay"]
    )

    # Scheduler
    scheduler_cfg = cfg.get("scheduler", {})
    if scheduler_cfg.get("name") == "cosine":
        warmup_epochs = scheduler_cfg.get("warmup_epochs", 1)
        total_steps = len(train_loader) * cfg["epochs"]
        warmup_steps = len(train_loader) * warmup_epochs
        scheduler = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)
    else:
        scheduler = None

    # Precision
    use_amp = cfg.get("precision") == "amp" and device == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    grad_clip = optim_cfg.get("grad_clip", 1.0)

    # Loss configuration
    loss_cfg = cfg["loss"]
    loss_name = loss_cfg["name"]

    epochs = cfg["epochs"]
    log_interval = cfg["log"]["interval"]

    # Training loop
    global_step = 0
    best_mean_r1 = 0.0
    
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        start_time = time.time()
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{epochs}")
        
        for it, (images, tokens, ids) in pbar:
            images = images.to(device, non_blocking=True)
            tokens = tokens.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                img_feat = model.encode_image(images)
                txt_feat = model.encode_text(tokens)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

                # Compute loss
                if loss_name == "clip_ce":
                    loss = clip_ce_loss(img_feat, txt_feat, model.logit_scale)
                elif loss_name == "sanw_debias":
                    loss = sanw_debias_loss(
                        img_feat, txt_feat, model.logit_scale,
                        alpha=loss_cfg.get("alpha", 0.5),
                        delta=loss_cfg.get("delta", 0.6),
                        lam=loss_cfg.get("lambda", 4.0),
                        stopgrad_weights=loss_cfg.get("stopgrad_weights", True),
                        geometry_align_enabled=loss_cfg.get("geometry_align", {}).get("enabled", False)
                    )
                elif loss_name == "sanw_bandpass":
                    loss = sanw_bandpass_loss(
                        img_feat, txt_feat, model.logit_scale,
                        alpha=loss_cfg.get("alpha", 0.5),
                        m1=loss_cfg.get("m1", 0.3),
                        m2=loss_cfg.get("m2", 0.8),
                        gamma=loss_cfg.get("gamma", 0.05),
                        floor=loss_cfg.get("floor", 1e-3),
                        stopgrad_weights=loss_cfg.get("stopgrad_weights", True)
                    )
                else:
                    raise ValueError(f"Unknown loss: {loss_name}")

            # Backward pass
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()

            # Clamp logit scale
            with torch.no_grad():
                model.logit_scale.data.clamp_(max=clamp_max)

            # Update scheduler
            if scheduler is not None:
                scheduler.step()

            running_loss += loss.item()
            if (it + 1) % log_interval == 0:
                avg_loss = running_loss / log_interval
                lr = opt.param_groups[0]['lr']
                temp = model.logit_scale.exp().item()
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}", temp=f"{temp:.3f}")
                running_loss = 0.0
            global_step += 1

        # Evaluation
        eval_cfg = cfg.get("eval", {})
        if epoch % eval_cfg.get("every_epochs", 1) == 0:
            model.eval()
            with torch.no_grad():
                # Retrieval evaluation
                retrieval_results = evaluate_retrieval(model, tokenizer, val_ds, device)
                
                # Zero-shot CIFAR-10
                zs_cifar10 = evaluate_zero_shot_cifar10(model, device)
                
                # Print results
                print(f"\n[Epoch {epoch}]")
                print(f"I→T: R@1={retrieval_results['i2t']['r1']:.2f} R@5={retrieval_results['i2t']['r5']:.2f} R@10={retrieval_results['i2t']['r10']:.2f}")
                print(f"T→I: R@1={retrieval_results['t2i']['r1']:.2f} R@5={retrieval_results['t2i']['r5']:.2f} R@10={retrieval_results['t2i']['r10']:.2f}")
                print(f"Mean R@1: {retrieval_results['mean_r1']:.2f}")
                print(f"Zero-shot CIFAR-10: {zs_cifar10:.2f}%")
                
                # Save best model
                if retrieval_results['mean_r1'] > best_mean_r1:
                    best_mean_r1 = retrieval_results['mean_r1']
                    torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': opt.state_dict(),
                        'scheduler': scheduler.state_dict() if scheduler else None,
                        'logit_scale': model.logit_scale.data,
                        'best_mean_r1': best_mean_r1
                    }, os.path.join(cfg["output_dir"], "best_model.pt"))
                
                # CSV logging
                log_path = os.path.join(cfg["output_dir"], "metrics.csv")
                header = ["epoch", "r1_i2t", "r5_i2t", "r10_i2t", "r1_t2i", "r5_t2i", "r10_t2i", "mean_r1", "zs_cifar10", "temp", "lr", "loss"]
                row = [
                    epoch,
                    retrieval_results['i2t']['r1'],
                    retrieval_results['i2t']['r5'],
                    retrieval_results['i2t']['r10'],
                    retrieval_results['t2i']['r1'],
                    retrieval_results['t2i']['r5'],
                    retrieval_results['t2i']['r10'],
                    retrieval_results['mean_r1'],
                    zs_cifar10,
                    model.logit_scale.exp().item(),
                    opt.param_groups[0]['lr'],
                    running_loss / len(train_loader)
                ]
                write_header = not os.path.exists(log_path)
                with open(log_path, "a", newline="") as f:
                    w = csv.writer(f)
                    if write_header:
                        w.writerow(header)
                    w.writerow(row)
            
            model.train()

        # Save epoch checkpoint
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': opt.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'logit_scale': model.logit_scale.data,
        }, os.path.join(cfg["output_dir"], f"epoch_{epoch:02d}.pt"))

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch} completed in {epoch_time:.1f}s")

    print(f"Training completed. Best mean R@1: {best_mean_r1:.2f}")

if __name__ == "__main__":
    main()
