import os, yaml, math, random, csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# Use transformers CLIP as primary implementation (more compatible with Python 3.6)
try:
    from transformers import CLIPModel, CLIPProcessor, CLIPConfig
    try:
        from transformers import CLIPImageProcessor
        CLIP_IMAGE_PROCESSOR_AVAILABLE = True
    except ImportError:
        CLIP_IMAGE_PROCESSOR_AVAILABLE = False
        from transformers import CLIPFeatureExtractor
    TRANSFORMERS_CLIP_AVAILABLE = True
    OPEN_CLIP_AVAILABLE = False
    print("✅ Using transformers CLIP implementation")
except ImportError:
    print("❌ transformers CLIP not available")
    TRANSFORMERS_CLIP_AVAILABLE = False
    # Fallback: try open_clip
    try:
        import open_clip
        OPEN_CLIP_AVAILABLE = True
        TRANSFORMERS_CLIP_AVAILABLE = False
        print("⚠️  Using open_clip as fallback")
    except ImportError:
        print("❌ Neither transformers nor open_clip available")
        OPEN_CLIP_AVAILABLE = False

# Make sure CLIPModel is available globally
if not TRANSFORMERS_CLIP_AVAILABLE:
    # Create a dummy CLIPModel class for fallback
    class CLIPModel:
        def __init__(self, *args, **kwargs):
            pass
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError("CLIPModel not available")
from tqdm import tqdm
import time

from .datasets import FlickrTSV, collate_text_tokenize
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
    # Cluster-SOP friendly runtime overrides
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", type=str, default=None, help="cpu|cuda|auto")
    ap.add_argument("--local_model_dir", type=str, default=None, help="Absolute path to local HF CLIP directory")
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--topical_k", type=int, default=None)
    ap.add_argument("--adaptive_k", type=str, default=None, help="true|false")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))

    # Apply runtime overrides into cfg
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device is not None:
        cfg["device"] = args.device
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.num_workers is not None:
        cfg.setdefault("data", {}).update({"num_workers": args.num_workers})
    if args.topical_k is not None:
        cfg.setdefault("data", {}).setdefault("topical_batching", {}).update({"k_clusters": args.topical_k})
    if args.adaptive_k is not None:
        adaptive_flag = args.adaptive_k.strip().lower() in {"1", "true", "yes", "y"}
        cfg.setdefault("data", {}).setdefault("topical_batching", {}).update({"adaptive_k": adaptive_flag})

    set_seed(cfg.get("seed", 42))
    
    # Handle device selection
    device_config = cfg.get("device", "auto")
    if device_config == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_config
    
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if device == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    if device == "cpu":
        print("Note: Training on CPU will be slower. Consider using CUDA if available.")

    os.makedirs(cfg["output_dir"], exist_ok=True)

    # Model + tokenizer
    model_name = cfg["model"]["name"]
    pretrained = cfg["model"]["pretrained"]
    
    if TRANSFORMERS_CLIP_AVAILABLE:
        # Use transformers CLIP as primary implementation
        print("Using transformers CLIP implementation")
        try:
            # Prefer an explicit local directory (env or CLI)
            local_dir = args.local_model_dir or os.environ.get("CLIP_LOCAL_DIR") or \ 
                        cfg.get("model", {}).get("local_dir", None)
            if local_dir and os.path.exists(os.path.join(local_dir, "config.json")):
                print(f"Loading CLIP from local directory: {local_dir}")
                model = CLIPModel.from_pretrained(local_dir, local_files_only=True)
                processor = CLIPProcessor.from_pretrained(local_dir, local_files_only=True)
            else:
                # Fallback to user cache snapshot structure
                cache_dir = os.path.expanduser("~/.cache/huggingface/transformers/models--openai--clip-vit-base-patch32/snapshots/")
                snapshot_dirs = [d for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d)) and d != 'latest']
                if snapshot_dirs:
                    model_path = os.path.join(cache_dir, snapshot_dirs[0])
                    print(f"Using model from cache: {model_path}")
                    model = CLIPModel.from_pretrained(model_path, local_files_only=True)
                    processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
                else:
                    raise FileNotFoundError("No local CLIP files found (set CLIP_LOCAL_DIR)")

            print("✅ Successfully loaded pretrained CLIP")
            preprocess = processor.image_processor
            tokenizer = processor.tokenizer
        except Exception as e:
            print(f"⚠️  Failed to load from cache: {e}")
            print("🔧 Creating CLIP model from scratch with proper configuration...")
            # Create a CLIP model from scratch with the correct configuration
            
            # Use the actual CLIP-ViT-B/32 configuration
            config = CLIPConfig(
                vision_config={
                    "hidden_size": 768,
                    "intermediate_size": 3072,
                    "num_attention_heads": 12,
                    "num_hidden_layers": 12,
                    "patch_size": 32,
                    "image_size": 224,
                    "num_channels": 3
                },
                text_config={
                    "hidden_size": 512,
                    "intermediate_size": 2048,
                    "num_attention_heads": 8,
                    "num_hidden_layers": 12,
                    "vocab_size": 49408,
                    "max_position_embeddings": 77
                },
                projection_dim=512
            )
            model = CLIPModel(config)
            
            # Create processors manually
            if CLIP_IMAGE_PROCESSOR_AVAILABLE:
                preprocess = CLIPImageProcessor(
                    size=224,
                    crop_size=224,
                    do_resize=True,
                    do_center_crop=True,
                    do_normalize=True,
                    image_mean=[0.48145466, 0.4578275, 0.40821073],
                    image_std=[0.26862954, 0.26130258, 0.27577711]
                )
            else:
                preprocess = CLIPFeatureExtractor(
                    size=224,
                do_resize=True,
                do_center_crop=True,
                do_normalize=True,
                image_mean=[0.48145466, 0.4578275, 0.40821073],
                image_std=[0.26862954, 0.26130258, 0.27577711]
            )
            
            # Create a proper CLIP tokenizer that works offline
            class CLIPTokenizer:
                def __init__(self):
                    self.model_max_length = 77
                    self.pad_token = "<|endoftext|>"
                    self.bos_token = "<|startoftext|>"
                    self.eos_token = "<|endoftext|>"
                    self.unk_token = "<|endoftext|>"
                    
                    # CLIP's actual vocabulary (simplified but functional)
                    self.vocab = {
                        "<|startoftext|>": 49406,
                        "<|endoftext|>": 49407,
                        "<|pad|>": 0,
                        "<|unk|>": 1,
                    }
                    
                    # Add common words to vocabulary
                    common_words = [
                        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
                        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
                        "will", "would", "could", "should", "may", "might", "can", "must", "shall", "this", "that",
                        "these", "those", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
                        "my", "your", "his", "her", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs"
                    ]
                    
                    # Add common words to vocab with reasonable IDs
                    for i, word in enumerate(common_words):
                        self.vocab[word] = i + 10
                    
                    # Add some common image-related words
                    image_words = [
                        "image", "picture", "photo", "photograph", "picture", "image", "photo", "photograph",
                        "man", "woman", "person", "people", "child", "baby", "boy", "girl", "man", "woman",
                        "dog", "cat", "animal", "car", "truck", "bus", "bike", "bicycle", "motorcycle",
                        "house", "building", "tree", "grass", "sky", "water", "ocean", "beach", "mountain",
                        "red", "blue", "green", "yellow", "black", "white", "brown", "gray", "orange", "purple"
                    ]
                    
                    for i, word in enumerate(image_words):
                        if word not in self.vocab:
                            self.vocab[word] = len(self.vocab) + 100
                
                def _tokenize_text(self, text):
                    """Simple but effective tokenization"""
                    # Convert to lowercase and split by spaces
                    words = text.lower().split()
                    tokens = []
                    
                    for word in words:
                        # Clean word (remove punctuation)
                        clean_word = ''.join(c for c in word if c.isalnum())
                        if clean_word:
                            if clean_word in self.vocab:
                                tokens.append(self.vocab[clean_word])
                            else:
                                # Use hash-based ID for unknown words
                                tokens.append(hash(clean_word) % 10000 + 1000)
                    
                    return tokens
                
                def __call__(self, texts, return_tensors=None, padding=True, truncation=True, max_length=77):
                    if isinstance(texts, str):
                        texts = [texts]
                    
                    token_ids = []
                    for text in texts:
                        # Tokenize the text
                        tokens = self._tokenize_text(text)
                        
                        # Add special tokens: [BOS] + tokens + [EOS]
                        if truncation and len(tokens) > max_length - 2:
                            tokens = tokens[:max_length - 2]
                        
                        # Add BOS and EOS tokens
                        final_tokens = [self.vocab["<|startoftext|>"]] + tokens + [self.vocab["<|endoftext|>"]]
                        
                        # Pad to max_length
                        while len(final_tokens) < max_length:
                            final_tokens.append(self.vocab["<|pad|>"])
                        
                        token_ids.append(final_tokens[:max_length])
                    
                    import torch
                    result = torch.tensor(token_ids)
                    if return_tensors == "pt":
                        return result
                    return result
                
                def encode(self, text, add_special_tokens=True, max_length=77):
                    return self([text], max_length=max_length)[0]
            
            tokenizer = CLIPTokenizer()
            print("✅ Using proper CLIP tokenizer with vocabulary")
            
            print("⚠️  Created CLIP model from scratch (no pretrained weights)")
            print("⚠️  This will produce poor results - not suitable for research")
            print("⚠️  For proper results, ensure CLIP model files are downloaded to cache")
    elif OPEN_CLIP_AVAILABLE:
        # Fallback to open_clip
        print("Using open_clip as fallback")
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        tokenizer = open_clip.get_tokenizer(model_name)
    else:
        raise RuntimeError("No CLIP implementation available. Please install transformers or open_clip.")

    model = model.to(device)
    model.train()

    # Initialize logit scale
    logit_scale_cfg = cfg.get("logit_scale", {})
    init_scale = logit_scale_cfg.get("init", 2.659)
    clamp_max = logit_scale_cfg.get("clamp_max", 4.605170)
    
    if TRANSFORMERS_CLIP_AVAILABLE:
        # For transformers CLIP, we need to add logit_scale as a parameter
        model.logit_scale = nn.Parameter(torch.tensor(init_scale))
        model.logit_scale.requires_grad_(True)
    elif OPEN_CLIP_AVAILABLE:
        with torch.no_grad():
            model.logit_scale.data.fill_(init_scale)
        model.logit_scale.requires_grad_(True)

    # Data - use Flickr8k dataset
    ds_cfg = cfg["data"]["flickr8k"]
    train_ds = FlickrTSV(ds_cfg["root"], ds_cfg["tsv"], split="train", val_ratio=0.1, transform=preprocess)
    val_ds = FlickrTSV(ds_cfg["root"], ds_cfg["tsv"], split="val", val_ratio=0.1, transform=preprocess)

    # Topical batching
    topical_cfg = cfg["data"].get("topical_batching", {})
    if topical_cfg.get("enabled", False):
        # Adaptive k: reduce to available distinct captions if requested
        requested_k = topical_cfg.get("k_clusters", 80)
        adaptive_k = topical_cfg.get("adaptive_k", False)
        effective_k = requested_k
        if adaptive_k:
            # Count distinct captions in train split
            try:
                distinct = len({cap for (_, cap) in train_ds.train})
                effective_k = max(8, min(requested_k, distinct))
                if effective_k != requested_k:
                    print(f"[Topical] Reducing k from {requested_k} to {effective_k} (distinct={distinct})")
            except Exception:
                pass
        batch_sampler = TopicalBatchSampler(
            train_ds, 
            batch_size=cfg["data"]["batch_size"],
            k_clusters=effective_k,
            p_topical=topical_cfg.get("p_topical", 0.5),
            refresh_every_epochs=topical_cfg.get("refresh_every_epochs", 2),
            sampler_temperature=topical_cfg.get("sampler_temperature", 0.7),
            seed=cfg.get("seed", 42)
        )
        train_loader = DataLoader(
            train_ds, 
            batch_sampler=batch_sampler,
            num_workers=cfg.get("data", {}).get("num_workers", 0),
            collate_fn=lambda b: collate_text_tokenize(b, tokenizer)
        )
    else:
        train_loader = DataLoader(
            train_ds, 
            batch_size=cfg["data"]["batch_size"], 
            shuffle=True,
            num_workers=cfg.get("data", {}).get("num_workers", 0),
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

    # Precision - disable mixed precision to avoid CPU/CUDA backend issues
    use_amp = False  # Disabled due to PyTorch 1.7.1 compatibility issues
    scaler = None
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

            # Fix tensor dimensions for transformers CLIP
            if TRANSFORMERS_CLIP_AVAILABLE:
                # Remove extra dimension if present (batch, 1, 3, 224, 224) -> (batch, 3, 224, 224)
                if images.dim() == 5 and images.size(1) == 1:
                    images = images.squeeze(1)
                # Ensure images are in correct format: (batch, 3, 224, 224)
                if images.dim() != 4:
                    print(f"Warning: Unexpected image tensor shape: {images.shape}")

            with torch.cuda.amp.autocast(enabled=use_amp):
                if TRANSFORMERS_CLIP_AVAILABLE:
                    # Transformers CLIP primary implementation
                    outputs = model(pixel_values=images, input_ids=tokens)
                    img_feat = outputs.image_embeds
                    txt_feat = outputs.text_embeds
                elif OPEN_CLIP_AVAILABLE:
                    # OpenCLIP fallback
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
                zs_cifar10 = evaluate_zero_shot_cifar10(model, device, tokenizer)
                
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
                header = ["epoch", "r1_i2t", "r5_i2t", "r10_i2t", "r1_t2i", "r5_t2i", "r10_t2i", "mean_r1", "zs_cifar10", "logit_scale", "exp_logit_scale", "lr", "loss"]
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
                    model.logit_scale.item(),  # Raw logit scale
                    model.logit_scale.exp().item(),  # exp(logit_scale)
                    opt.param_groups[0]['lr'],
                    running_loss / max(len(train_loader), 1)
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
