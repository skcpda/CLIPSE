import os, random
from typing import Optional
import torch
from torch.utils.data import DataLoader
import torchvision
import open_clip
import numpy as np
from .datasets import collate_text_tokenize

def _encode_pairs(model, tokenizer, dataset, device, subset=None):
    idxs = list(range(len(dataset)))
    if subset is not None and subset < len(dataset):
        idxs = idxs[:subset]
    images, texts = [], []
    for i in idxs:
        img, cap, _ = dataset[i]
        images.append(img)
        texts.append(cap)
    images = torch.stack(images, dim=0).to(device)
    tokens = tokenizer(texts)
    tokens = tokens.to(device)

    with torch.no_grad():
        img = model.encode_image(images)
        txt = model.encode_text(tokens)
        img = img / img.norm(dim=-1, keepdim=True)
        txt = txt / txt.norm(dim=-1, keepdim=True)
    return img, txt

def evaluate_retrieval(model, tokenizer, val_ds, device, subset=None):
    img, txt = _encode_pairs(model, tokenizer, val_ds, device, subset=subset)
    sims = (img @ txt.t()).float().cpu().numpy()

    # I->T retrieval
    ranks_i2t = np.argsort(-sims, axis=1)  # for each image, ranking of texts
    def recall_at_i2t(k):
        hits = 0
        for i in range(ranks_i2t.shape[0]):
            if i in ranks_i2t[i, :k]:
                hits += 1
        return 100.0 * hits / ranks_i2t.shape[0]
    r1_i2t, r5_i2t, r10_i2t = recall_at_i2t(1), recall_at_i2t(5), recall_at_i2t(10)

    # T->I retrieval
    ranks_t2i = np.argsort(-sims, axis=0)  # for each text, ranking of images
    def recall_at_t2i(k):
        hits = 0
        for j in range(ranks_t2i.shape[1]):
            if j in ranks_t2i[:k, j]:
                hits += 1
        return 100.0 * hits / ranks_t2i.shape[1]
    r1_t2i, r5_t2i, r10_t2i = recall_at_t2i(1), recall_at_t2i(5), recall_at_t2i(10)

    # Return both directions
    return {
        'i2t': {'r1': r1_i2t, 'r5': r5_i2t, 'r10': r10_i2t},
        't2i': {'r1': r1_t2i, 'r5': r5_t2i, 'r10': r10_t2i},
        'mean_r1': (r1_i2t + r1_t2i) / 2
    }

def evaluate_zero_shot_cifar10(model, device):
    model.eval()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.48145466,0.4578275,0.40821073),
                                         (0.26862954,0.26130258,0.27577711)),
    ])
    ds = torchvision.datasets.CIFAR10(root="data/cifar10", train=False, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2)

    classnames = ds.classes
    prompts = [f"a photo of a {c}" for c in classnames]
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    tokens = tokenizer(prompts)
    tokens = tokens.to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            img_features = model.encode_image(imgs)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * img_features @ text_features.t()  # scaled for sharper softmax
            pred = logits.argmax(dim=1).cpu()
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total
