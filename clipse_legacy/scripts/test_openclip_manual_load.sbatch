#!/bin/bash
#SBATCH --job-name=test_openclip_man
#SBATCH --output=test_openclip_man_%j.log
#SBATCH --error=test_openclip_man_%j.err
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:15:00

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

echo "=== INSTALL PYTORCH + OPENCLIP ==="
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install open_clip_torch

echo "=== CUDA INFO ==="
python - <<'PY'
import torch
print('Torch:', torch.__version__)
print('CUDA is_available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
PY

echo "=== CHECK LOCAL MODELS ==="
MODEL_PATH="/nfs_home/users/poonam/clipse/models/laion2b_s34b_b79k"
echo "Looking for models at: $MODEL_PATH"
ls -la "$MODEL_PATH" || echo "Models not found at expected path"
du -sh "$MODEL_PATH" || echo "Cannot get size"

echo "=== TESTING OPENCLIP MANUAL LOADING ==="
python - <<'PY'
import traceback
import torch
import open_clip
import time
import sys
import os
import json

def log_with_timestamp(msg):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()

log_with_timestamp("Starting OpenCLIP manual loading test...")

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_with_timestamp(f"Using device: {device}")
    
    model_path = "/nfs_home/users/poonam/clipse/models/laion2b_s34b_b79k"
    log_with_timestamp(f"Loading model from: {model_path}")
    
    # Method: Load model architecture first, then load weights manually
    start_time = time.time()
    
    # Step 1: Create model architecture
    log_with_timestamp("Creating model architecture...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", 
        pretrained=None,  # Don't load pretrained weights
        device=device
    )
    
    # Step 2: Load weights manually from our files
    log_with_timestamp("Loading weights manually...")
    
    # Find the model file
    model_file = None
    for f in os.listdir(model_path):
        if f.endswith('.bin') and 'open_clip' in f:
            model_file = os.path.join(model_path, f)
            break
    
    if not model_file:
        # Try safetensors
        for f in os.listdir(model_path):
            if f.endswith('.safetensors') and 'open_clip' in f:
                model_file = os.path.join(model_path, f)
                break
    
    if model_file:
        log_with_timestamp(f"Loading weights from: {model_file}")
        
        # Load state dict
        if model_file.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(model_file)
        else:
            state_dict = torch.load(model_file, map_location=device)
        
        # Load weights into model
        model.load_state_dict(state_dict)
        log_with_timestamp("✅ Weights loaded successfully!")
        
        load_time = time.time() - start_time
        log_with_timestamp(f"✅ Model loaded in {load_time:.2f} seconds!")
        
        # Test the model
        log_with_timestamp(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
        log_with_timestamp(f"Logit scale: {float(model.logit_scale.exp()):.3f}")
        
        # Quick inference test
        log_with_timestamp("Testing inference...")
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        text = tokenizer(["a photo of a cat", "a photo of a dog"])
        
        with torch.no_grad():
            text_features = model.encode_text(text)
        
        log_with_timestamp(f"Text features shape: {text_features.shape}")
        log_with_timestamp(f"✅ OpenCLIP manual loading works!")
        log_with_timestamp("🎉 Ready for SANW experiments!")
        
    else:
        log_with_timestamp("❌ No model file found")
    
except Exception as e:
    log_with_timestamp(f"❌ Test failed: {repr(e)}")
    log_with_timestamp("Full traceback:")
    traceback.print_exc()
    log_with_timestamp("❌ Still hitting issues.")
PY

echo "=== DONE ==="
