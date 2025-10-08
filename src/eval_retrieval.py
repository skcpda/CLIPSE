"""
Retrieval evaluation for CLIP models using OpenCLIP and Flickr8k.
"""
import os
import csv
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
import open_clip


def build_openclip(model_name: str, pretrained: str, device: str, local_checkpoint_path: str | None = None):
    """Create OpenCLIP model, preprocess, and tokenizer.

    If local_checkpoint_path is provided, it's passed directly to pretrained.
    """
    if local_checkpoint_path and os.path.exists(local_checkpoint_path):
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=local_checkpoint_path, device=device
        )
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


@torch.no_grad()
def extract_features(model, loader, device):
    image_feats = []
    text_feats = []
    for batch in tqdm(loader, desc="Extract"):
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)

        img = model.encode_image(pixel_values)
        txt = model.encode_text(input_ids)
        img = F.normalize(img, dim=-1)
        txt = F.normalize(txt, dim=-1)
        image_feats.append(img.cpu())
        text_feats.append(txt.cpu())
    return torch.cat(image_feats, dim=0), torch.cat(text_feats, dim=0)


def recall_at_k(similarity: torch.Tensor, k: int, dim: int = 1) -> float:
    if dim == 1:  # rows: images, cols: texts
        _, idx = torch.topk(similarity, k, dim=1)
        targets = torch.arange(similarity.size(0)).unsqueeze(1)
        return (idx == targets).any(dim=1).float().mean().item()
    else:  # rows: texts, cols: images
        _, idx = torch.topk(similarity.t(), k, dim=1)
        targets = torch.arange(similarity.size(1)).unsqueeze(1)
        return (idx == targets).any(dim=1).float().mean().item()


def median_rank(similarity: torch.Tensor, dim: int = 1) -> float:
    if dim == 1:  # image -> text
        ranks = torch.argsort(similarity, dim=1, descending=True)
        targets = torch.arange(similarity.size(0)).unsqueeze(1)
        pos = (ranks == targets).nonzero(as_tuple=False)[:, 1].float() + 1.0
        return pos.median().item()
    else:  # text -> image
        ranks = torch.argsort(similarity.t(), dim=1, descending=True)
        targets = torch.arange(similarity.size(1)).unsqueeze(1)
        pos = (ranks == targets).nonzero(as_tuple=False)[:, 1].float() + 1.0
        return pos.median().item()


def map_at_k(similarity: torch.Tensor, k: int = 10, dim: int = 1) -> float:
    # One positive per row; AP@k reduces to 1/rank if rank<=k else 0
    if dim == 1:
        ranks = torch.argsort(similarity, dim=1, descending=True)
        targets = torch.arange(similarity.size(0)).unsqueeze(1)
        pos = (ranks == targets).nonzero(as_tuple=False)[:, 1] + 1
        ap = torch.where(pos <= k, 1.0 / pos.float(), torch.zeros_like(pos, dtype=torch.float))
        return ap.mean().item()
    else:
        ranks = torch.argsort(similarity.t(), dim=1, descending=True)
        targets = torch.arange(similarity.size(1)).unsqueeze(1)
        pos = (ranks == targets).nonzero(as_tuple=False)[:, 1] + 1
        ap = torch.where(pos <= k, 1.0 / pos.float(), torch.zeros_like(pos, dtype=torch.float))
        return ap.mean().item()


def evaluate_retrieval(model, test_loader, device, k_values=(1, 5, 10)):
    model.eval()
    img, txt = extract_features(model, test_loader, device)
    sim = img @ txt.t()

    results = {}
    for k in k_values:
        results[f"R1_I2T" if k == 1 else f"R{k}_I2T"] = recall_at_k(sim, k, dim=1)
        results[f"R1_T2I" if k == 1 else f"R{k}_T2I"] = recall_at_k(sim, k, dim=0)
    results["MR_I2T"] = median_rank(sim, dim=1)
    results["MR_T2I"] = median_rank(sim, dim=0)
    results["mAP10_I2T"] = map_at_k(sim, k=10, dim=1)
    results["mAP10_T2I"] = map_at_k(sim, k=10, dim=0)
    return results


def append_run_registry(row: dict, registry_path: str = "results/run_registry.csv"):
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    file_exists = os.path.exists(registry_path)
    with open(registry_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval on Flickr8k with OpenCLIP")
    parser.add_argument("--model_name", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained_tag", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--local_checkpoint_path", type=str, default="", help="Absolute path to .bin/.safetensors")
    parser.add_argument("--dataset_path", type=str, default="/nfs_home/users/poonam/clipse/data/flickr8k")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", type=str, default="baseline", help="baseline|sanw_debias|sanw_bandpass")
    parser.add_argument("--epoch", type=int, default=-1)
    parser.add_argument("--log_file", type=str, default="results/run_registry.csv")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, preprocess, tokenizer = build_openclip(
        args.model_name, args.pretrained_tag, device,
        local_checkpoint_path=args.local_checkpoint_path or None,
    )

    # Build Flickr8k loaders using our collator
    from .datasets import create_flickr8k_datasets
    test_loader_only = create_flickr8k_datasets(
        args.dataset_path, preprocess_fn=preprocess, tokenizer_fn=tokenizer,
        batch_size=args.batch_size, num_workers=0, device=device
    )[2]

    scores = evaluate_retrieval(model, test_loader_only, device)

    row = {
        "method": args.method,
        "seed": args.seed,
        "epoch": args.epoch,
        "model": args.model_name,
        "pretrained": args.pretrained_tag,
        **scores,
    }
    append_run_registry(row, args.log_file)

    print("Evaluation:")
    for k, v in row.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
