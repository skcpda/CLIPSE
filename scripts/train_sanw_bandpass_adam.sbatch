#!/bin/bash
#SBATCH --job-name=clipse_bandpass_adam
#SBATCH --output=clipse_bandpass_adam_%j.log
#SBATCH --error=clipse_bandpass_adam_%j.err
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "=== LIMITS ==="
ulimit -a || true
cat /proc/self/limits || true
echo

echo "=== CREATE VENV UNDER /tmp ==="
VENV_DIR="/tmp/${SLURM_JOB_ID}/venv"
PY="$(command -v python3.11 || command -v python3 || command -v python)"
mkdir -p "$(dirname "$VENV_DIR")"
"$PY" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m ensurepip --upgrade || true
python -m pip install --upgrade pip

echo "=== INSTALL DEPENDENCIES ==="
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install open_clip_torch scikit-learn pandas pillow tqdm

echo "=== CUDA INFO ==="
python - <<'PY'
import torch
print('Torch:', torch.__version__)
print('CUDA is_available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
PY

echo "=== RUNNING SANW-BANDPASS WITH ADAM OPTIMIZER ==="
python - <<'PY'
import sys
import os
import torch
import open_clip
import time
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
import numpy as np

# Add src to path
sys.path.append('/nfs_home/users/poonam/clipse/src')

# Import our modules
from losses.sanw_bandpass import sanw_bandpass_loss
from utils.logging import CSVLogger

def log_with_timestamp(msg):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()

log_with_timestamp("Starting SANW-Bandpass with Adam Optimizer...")

# Configuration with ADAM OPTIMIZER and LEARNING RATE SCHEDULING
config = {
    'model_path': '/nfs_home/users/poonam/clipse/models/laion2b_s34b_b79k',
    'dataset_path': '/nfs_home/users/poonam/clipse/data/flickr8k',
    'batch_size': 32,
    'epochs': 10,
    'lr': 1e-4,              # Higher initial LR for Adam
    'weight_decay': 0.01,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'log_every': 25,
    'grad_clip': 1.0,
    # SANW-Bandpass parameters - ULTRA CONSERVATIVE
    'alpha': 0.1,            # Much smaller
    'm1': 0.1,              # Tighter lower bound
    'm2': 0.9,              # Tighter upper bound
    'gamma': 0.5,           # Much smaller penalty
    # Adam optimizer settings
    'betas': (0.9, 0.999),
    'eps': 1e-8,
    # Learning rate scheduling
    'lr_scheduler': 'cosine',  # Cosine annealing
    'warmup_epochs': 2
}

log_with_timestamp(f"Config: {config}")

# Load model
log_with_timestamp("Loading OpenCLIP model...")
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", 
    pretrained=None,
    device=config['device']
)

# Load weights manually
model_file = os.path.join(config['model_path'], 'open_clip_pytorch_model.bin')
state_dict = torch.load(model_file, map_location=config['device'])
model.load_state_dict(state_dict)
log_with_timestamp("✅ Model loaded successfully!")

# Get tokenizer
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Setup ADAM optimizer with learning rate scheduling
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=config['lr'],
    weight_decay=config['weight_decay'],
    betas=config['betas'],
    eps=config['eps']
)

# Learning rate scheduler
if config['lr_scheduler'] == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['epochs'],
        eta_min=config['lr'] * 0.01
    )
else:
    scheduler = None

scaler = torch.cuda.amp.GradScaler(enabled=(config['device'] == 'cuda'))

# Setup logging
logger = CSVLogger(f'/nfs_home/users/poonam/clipse/logs/sanw_bandpass_adam_{int(time.time())}.csv')

# Create REAL Flickr8k dataset
class Flickr8kDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split
        
        # Load captions
        captions_file = os.path.join(root_dir, 'captions.tsv')
        self.df = pd.read_csv(captions_file, sep='\t')
        
        # Filter by split
        if split == 'train':
            split_file = os.path.join(root_dir, 'Flickr_8k.trainImages.txt')
        elif split == 'val':
            split_file = os.path.join(root_dir, 'Flickr_8k.devImages.txt')
        else:  # test
            split_file = os.path.join(root_dir, 'Flickr_8k.testImages.txt')
        
        with open(split_file, 'r') as f:
            split_images = set(line.strip() for line in f)
        
        self.df = self.df[self.df['image_id'].isin(split_images)]
        log_with_timestamp(f"Loaded {len(self.df)} samples for {split} split")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.root_dir, 'Images', row['image_id'])
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            log_with_timestamp(f"Error loading image {image_path}: {e}")
            # Create a dummy image if loading fails
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        text = row['caption']
        return {'image': image, 'text': text}

# Create data loaders
log_with_timestamp("Creating datasets...")
train_dataset = Flickr8kDataset(config['dataset_path'], split='train')
val_dataset = Flickr8kDataset(config['dataset_path'], split='val')

def collate_fn(batch):
    images = [b['image'] for b in batch]
    texts = [b['text'] for b in batch]
    
    # Process images
    pixel_values = torch.stack([preprocess(img) for img in images])
    
    # Process texts
    text_tokens = tokenizer(texts)
    
    return {
        'pixel_values': pixel_values,
        'input_ids': text_tokens
    }

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn, num_workers=2)

log_with_timestamp(f"Training samples: {len(train_dataset)}")
log_with_timestamp(f"Validation samples: {len(val_dataset)}")
log_with_timestamp("Starting SANW-Bandpass training with Adam optimizer...")

for epoch in range(config['epochs']):
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    
    # Warmup learning rate for first few epochs
    if epoch < config['warmup_epochs']:
        warmup_lr = config['lr'] * (epoch + 1) / config['warmup_epochs']
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr
    
    for step, batch in enumerate(train_loader):
        # Move batch to device
        batch = {k: v.to(config['device']) for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=(config['device'] == 'cuda')):
            # Get image and text features
            image_features = model.encode_image(batch['pixel_values'])
            text_features = model.encode_text(batch['input_ids'])
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute logits
            logit_scale = model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            
            # Compute SANW-Bandpass loss with ULTRA CONSERVATIVE parameters
            loss, stats = sanw_bandpass_loss(
                logits_per_image, 
                logits_per_text, 
                image_features, 
                text_features,
                alpha=config['alpha'],
                m1=config['m1'],
                m2=config['m2'],
                gamma=config['gamma']
            )
        
        # Backward pass with gradient clipping
        scaler.scale(loss).backward()
        
        # Gradient clipping for stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        num_batches += 1
        
        if (step + 1) % config['log_every'] == 0:
            current_lr = optimizer.param_groups[0]['lr']
            log_with_timestamp(f"Epoch {epoch+1}/{config['epochs']}, Step {step+1}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
            log_with_timestamp(f"SANW-Bandpass stats: {stats}")
            logger.log_step(
                epoch=epoch+1, 
                step=step+1, 
                loss=loss.item(), 
                temp=float(logit_scale.detach().cpu()),
                lr=current_lr,
                **stats
            )
    
    # Update learning rate scheduler
    if scheduler is not None and epoch >= config['warmup_epochs']:
        scheduler.step()
    
    avg_loss = epoch_loss / num_batches
    current_lr = optimizer.param_groups[0]['lr']
    log_with_timestamp(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
    logger.log_epoch(epoch=epoch+1, avg_loss=avg_loss, lr=current_lr)

log_with_timestamp("✅ SANW-Bandpass with Adam optimizer completed successfully!")
log_with_timestamp("🎉 SANW-Bandpass training finished!")
PY

echo "=== DONE ==="
