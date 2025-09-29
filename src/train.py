import os, yaml, math, random, csv
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
import open_clip
from tqdm import tqdm

from .datasets import FlickrTSV, collate_text_tokenize
from .dummy_dataset import DummyFlickrDataset
from .topical_sampler import TopicalBatchSampler
from .losses import clip_ce_loss, sanw_clip_loss
from .eval_utils import evaluate_retrieval, evaluate_zero_shot_cifar10

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    # model + tokenizer
    model_name = cfg["model"]["name"]
    pretrained = cfg["model"]["pretrained"]
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)

    model = model.to(device)
    model.train()

    # data - use dummy dataset for testing
    use_dummy = cfg["data"].get("use_dummy", True)
    if use_dummy:
        train_ds = DummyFlickrDataset(num_samples=200, split="train", val_ratio=0.1, transform=preprocess)
        val_ds = DummyFlickrDataset(num_samples=200, split="val", val_ratio=0.1, transform=preprocess)
    else:
        ds_cfg = cfg["data"]["flickr8k"]
        train_ds = FlickrTSV(ds_cfg["root"], ds_cfg["tsv"], split="train", val_ratio=0.1, transform=preprocess)
        val_ds   = FlickrTSV(ds_cfg["root"], ds_cfg["tsv"], split="val",   val_ratio=0.1, transform=preprocess)
    
    # caption shuffle stress test
    caption_shuffle_prob = cfg["data"].get("caption_shuffle_prob", 0.0)

    topical = cfg["data"].get("topical_batches", True)
    if topical:
        batch_sampler = TopicalBatchSampler(train_ds, batch_size=cfg["data"]["batch_size"], key=cfg["data"].get("topical_key","noun"))
        train_loader = DataLoader(train_ds, batch_sampler=batch_sampler,
                                  num_workers=cfg["data"]["num_workers"],
                                  collate_fn=lambda b: collate_text_tokenize(b, tokenizer))
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True,
                                  num_workers=cfg["data"]["num_workers"],
                                  collate_fn=lambda b: collate_text_tokenize(b, tokenizer))

    val_subset = cfg["eval"].get("val_subset", None)

    # optim
    optim_cfg = cfg["optim"]
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=optim_cfg["lr"], weight_decay=optim_cfg["weight_decay"])
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
    grad_clip = optim_cfg.get("grad_clip", 1.0)

    # loss mode
    loss_cfg = cfg["loss"]
    loss_name = loss_cfg["name"]
    temperature_learnable = loss_cfg.get("temperature_learnable", True)
    if temperature_learnable:
        # OpenCLIP exposes model.logit_scale
        model.logit_scale.requires_grad_(True)

    epochs = optim_cfg["epochs"]
    log_interval = cfg["log"]["interval"]

    global_step = 0
    for epoch in range(1, epochs+1):
        running = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{epochs}")
        for it, (images, tokens, ids) in pbar:
            images = images.to(device, non_blocking=True)
            tokens = tokens.to(device)
            
            # Caption shuffle stress test
            if caption_shuffle_prob > 0 and random.random() < caption_shuffle_prob:
                # Shuffle captions within batch to inject false negatives
                batch_size = tokens.size(0)
                shuffle_indices = torch.randperm(batch_size, device=tokens.device)
                tokens = tokens[shuffle_indices]

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                img_feat, txt_feat, _ = model.encode_image(images), model.encode_text(tokens), None
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

                if loss_name == "clip_ce":
                    loss = clip_ce_loss(img_feat, txt_feat, model.logit_scale)
                elif loss_name == "sanw":
                    mode = loss_cfg["mode"]
                    loss = sanw_clip_loss(
                        img_feat, txt_feat, model.logit_scale,
                        mode=mode,
                        alpha=loss_cfg.get("alpha", 0.5),
                        delta=loss_cfg.get("delta", 0.6),
                        lam=loss_cfg.get("lam", 4.0),
                        m1=loss_cfg.get("m1", 0.3),
                        m2=loss_cfg.get("m2", 0.8),
                        gamma=loss_cfg.get("gamma", 0.05)
                    )
                else:
                    raise ValueError("Unknown loss")

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

            running += loss.item()
            if (it + 1) % log_interval == 0:
                avg = running / log_interval
                pbar.set_postfix(loss=f"{avg:.4f}", temp=f"{model.logit_scale.exp().item():.3f}")
                running = 0.0
            global_step += 1

        # evaluation
        if (epoch % cfg["eval"].get("retrieval_every_epochs", 1)) == 0:
            model.eval()
            with torch.no_grad():
                r1, r5, r10 = evaluate_retrieval(model, tokenizer, val_ds, device, subset=val_subset)
                print(f"[Val Retrieval] R@1={r1:.2f} R@5={r5:.2f} R@10={r10:.2f}")
                if cfg["eval"].get("zs_cifar10", True):
                    top1 = evaluate_zero_shot_cifar10(model, device)
                    print(f"[Zero-shot CIFAR10] top1={top1:.2f}")
                else:
                    top1 = 0.0
            model.train()
            
            # CSV logging
            log_path = os.path.join(cfg["output_dir"], "metrics.csv")
            header = ["epoch", "r1", "r5", "r10", "zs_cifar10", "temp"]
            row = [epoch, r1, r5, r10, top1, float(model.logit_scale.exp().item())]
            write_header = not os.path.exists(log_path)
            with open(log_path, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(header)
                w.writerow(row)

        # save epoch checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "logit_scale": model.logit_scale.data,
        }
        torch.save(ckpt, os.path.join(cfg["output_dir"], f"epoch_{epoch:02d}.pt"))

if __name__ == "__main__":
    main()
