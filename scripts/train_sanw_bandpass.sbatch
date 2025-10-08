#!/bin/bash
#SBATCH --job-name=clipse_bandpass
#SBATCH --output=clipse_bandpass_%j.log
#SBATCH --error=clipse_bandpass_%j.err
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

echo "=== RUNNING SANW-BANDPASS EXPERIMENT ==="
python - <<'PY'
import sys
import os
import torch
import open_clip
import time
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# Add src to path
sys.path.append('/nfs_home/users/poonam/clipse/src')

# Import our modules
from losses.sanw_bandpass import sanw_bandpass_loss
from data.collate import CLIPCollator
from utils.logging import CSVLogger

def log_with_timestamp(msg):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()

log_with_timestamp("Starting CLIPSE SANW-Bandpass Experiment...")

# Configuration
config = {
    'model_path': '/nfs_home/users/poonam/clipse/models/laion2b_s34b_b79k',
    'dataset_path': '/nfs_home/users/poonam/clipse/data/flickr8k',
    'batch_size': 32,
    'epochs': 5,
    'lr': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'log_every': 10,
    # SANW-Bandpass parameters
    'alpha': 0.5,
    'm1': 0.3,
    'm2': 0.7,
    'gamma': 2.0
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

# Setup optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
scaler = torch.cuda.amp.GradScaler(enabled=(config['device'] == 'cuda'))

# Setup logging
logger = CSVLogger(f'/nfs_home/users/poonam/clipse/logs/sanw_bandpass_{int(time.time())}.csv')

# Create dummy dataset for testing
class DummyDataset:
    def __init__(self, size=100):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Create dummy image and text
        image = Image.new('RGB', (224, 224), color=(idx % 255, (idx * 2) % 255, (idx * 3) % 255))
        text = f"image {idx}"
        return {'image': image, 'text': text}

# Create data loader
dataset = DummyDataset(100)
collate_fn = CLIPCollator(preprocess)
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)

log_with_timestamp("Starting SANW-Bandpass training...")

for epoch in range(config['epochs']):
    model.train()
    epoch_loss = 0.0
    
    for step, batch in enumerate(dataloader):
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
            
            # Compute SANW-Bandpass loss
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
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        
        if (step + 1) % config['log_every'] == 0:
            log_with_timestamp(f"Epoch {epoch+1}/{config['epochs']}, Step {step+1}, Loss: {loss.item():.4f}")
            log_with_timestamp(f"SANW-Bandpass stats: {stats}")
            logger.log_step(
                epoch=epoch+1, 
                step=step+1, 
                loss=loss.item(), 
                temp=float(logit_scale.detach().cpu()),
                lr=config['lr'],
                **stats
            )
    
    avg_loss = epoch_loss / len(dataloader)
    log_with_timestamp(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
    logger.log_epoch(epoch=epoch+1, avg_loss=avg_loss)

log_with_timestamp("✅ SANW-Bandpass experiment completed successfully!")
log_with_timestamp("🎉 SANW-Bandpass training finished!")
PY

echo "=== DONE ==="
