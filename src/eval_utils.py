import os, random
from typing import Optional
import torch
from torch.utils.data import DataLoader
import torchvision
# Use transformers CLIP as primary implementation
try:
    from transformers import CLIPModel, CLIPProcessor
    TRANSFORMERS_CLIP_AVAILABLE = True
    OPEN_CLIP_AVAILABLE = False
except ImportError:
    TRANSFORMERS_CLIP_AVAILABLE = False
    # Fallback: try open_clip
    try:
        import open_clip
        OPEN_CLIP_AVAILABLE = True
    except ImportError:
        OPEN_CLIP_AVAILABLE = False
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
    # Handle both tensor and BatchFeature inputs (same fix as in datasets.py)
    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images, dim=0).to(device)
    else:
        # Handle BatchFeature objects from transformers CLIP processor
        from transformers import BatchFeature
        if isinstance(images[0], BatchFeature):
            # Extract pixel_values from BatchFeature and convert to tensors
            pixel_values = []
            for img in images:
                if isinstance(img.pixel_values, torch.Tensor):
                    pixel_values.append(img.pixel_values)
                else:
                    # Convert list to tensor if needed
                    pixel_values.append(torch.tensor(img.pixel_values))
            images = torch.stack(pixel_values, dim=0).to(device)
        else:
            # Fallback: try to convert to tensor
            images = torch.stack(images, dim=0).to(device)
    
    # Fix tensor dimensions for transformers CLIP (same fix as in training)
    if TRANSFORMERS_CLIP_AVAILABLE:
        # Remove extra dimension if present (batch, 1, 3, 224, 224) -> (batch, 3, 224, 224)
        if images.dim() == 5 and images.size(1) == 1:
            images = images.squeeze(1)
        # Ensure images are in correct format: (batch, 3, 224, 224)
        if images.dim() != 4:
            print(f"Warning: Unexpected image tensor shape in evaluation: {images.shape}")
    
    tokens = tokenizer(texts)
    tokens = tokens.to(device)

    with torch.no_grad():
        if TRANSFORMERS_CLIP_AVAILABLE:
            # Transformers CLIP primary implementation
            outputs = model(pixel_values=images, input_ids=tokens)
            img = outputs.image_embeds
            txt = outputs.text_embeds
        elif OPEN_CLIP_AVAILABLE:
            # OpenCLIP fallback
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
            # For image i, check if its corresponding text (at position i) is in top-k
            if i in ranks_i2t[i, :k]:
                hits += 1
        return 100.0 * hits / ranks_i2t.shape[0]
    r1_i2t, r5_i2t, r10_i2t = recall_at_i2t(1), recall_at_i2t(5), recall_at_i2t(10)

    # T->I retrieval
    ranks_t2i = np.argsort(-sims, axis=0)  # for each text, ranking of images
    def recall_at_t2i(k):
        hits = 0
        for j in range(ranks_t2i.shape[1]):
            # For text j, check if its corresponding image (at position j) is in top-k
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

def evaluate_zero_shot_cifar10(model, device, tokenizer=None):
    model.eval()
    # Use integer interpolation for compatibility with older torchvision
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256, interpolation=3),  # 3 = BICUBIC
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.48145466,0.4578275,0.40821073),
                                         (0.26862954,0.26130258,0.27577711)),
    ])
    ds = torchvision.datasets.CIFAR10(root="data/cifar10", train=False, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)  # Disable multiprocessing

    classnames = ds.classes
    prompts = [f"a photo of a {c}" for c in classnames]
    
    if TRANSFORMERS_CLIP_AVAILABLE:
        # Use the provided tokenizer or create a simple one
        if tokenizer is not None:
            tokens = tokenizer(prompts, return_tensors="pt", padding=True)
            if isinstance(tokens, dict):
                tokens = tokens["input_ids"]
            else:
                tokens = tokens
        else:
            # Create a simple tokenizer as fallback
            class SimpleTokenizer:
                def __init__(self):
                    self.model_max_length = 77
                
                def __call__(self, texts, return_tensors=None, padding=True, max_length=77):
                    if isinstance(texts, str):
                        texts = [texts]
                    
                    # Simple tokenization
                    token_ids = []
                    for text in texts:
                        words = text.lower().split()[:max_length-2]
                        tokens = [1] + [hash(word) % 1000 + 10 for word in words] + [2]
                        while len(tokens) < max_length:
                            tokens.append(0)
                        token_ids.append(tokens[:max_length])
                    
                    import torch
                    result = torch.tensor(token_ids)
                    if return_tensors == "pt":
                        return result
                    return result
            
            tokenizer = SimpleTokenizer()
            tokens = tokenizer(prompts, return_tensors="pt", padding=True)
    elif OPEN_CLIP_AVAILABLE:
        # OpenCLIP fallback
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        tokens = tokenizer(prompts)
    
    tokens = tokens.to(device)
    with torch.no_grad():
        if TRANSFORMERS_CLIP_AVAILABLE:
            # Transformers CLIP primary implementation - text only
            # We need to get text embeddings without images
            text_features = model.get_text_features(input_ids=tokens)
        elif OPEN_CLIP_AVAILABLE:
            # OpenCLIP fallback
            text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            if TRANSFORMERS_CLIP_AVAILABLE:
                # Transformers CLIP primary implementation - image only
                img_features = model.get_image_features(pixel_values=imgs)
            elif OPEN_CLIP_AVAILABLE:
                # OpenCLIP fallback
                img_features = model.encode_image(imgs)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * img_features @ text_features.t()  # scaled for sharper softmax
            pred = logits.argmax(dim=1).cpu()
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total
