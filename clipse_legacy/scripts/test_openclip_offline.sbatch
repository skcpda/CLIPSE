#!/bin/bash
#SBATCH --job-name=test_openclip_off
#SBATCH --output=test_openclip_off_%j.log
#SBATCH --error=test_openclip_off_%j.err
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

# Set offline mode
export HF_HUB_OFFLINE=1
export OPENCLIP_CACHE_PATH=$PWD/.openclip_cache
mkdir -p "$OPENCLIP_CACHE_PATH"

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
ls -la models/
du -sh models/*

echo "=== TESTING OPENCLIP OFFLINE LOADING ==="
python - <<'PY'
import traceback
import torch
import open_clip
import time
import sys
import os

def log_with_timestamp(msg):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()

log_with_timestamp("Starting OpenCLIP offline test...")

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_with_timestamp(f"Using device: {device}")
    
    # Test loading from local model files
    model_path = "./models/laion2b_s34b_b79k"
    log_with_timestamp(f"Loading model from local path: {model_path}")
    
    # List files in model directory
    if os.path.exists(model_path):
        files = os.listdir(model_path)
        log_with_timestamp(f"Model files: {files}")
        
        # Find the model file
        model_file = None
        for f in files:
            if f.endswith('.bin') or f.endswith('.safetensors'):
                model_file = os.path.join(model_path, f)
                break
        
        if model_file:
            log_with_timestamp(f"Found model file: {model_file}")
            
            # Try to load with OpenCLIP
            start_time = time.time()
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained=model_file, device=device
            )
            
            load_time = time.time() - start_time
            log_with_timestamp(f"✅ Model loaded successfully in {load_time:.2f} seconds!")
            log_with_timestamp(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
            log_with_timestamp(f"Logit scale: {float(model.logit_scale.exp()):.3f}")
            
            # Quick test
            log_with_timestamp("Testing inference...")
            tokenizer = open_clip.get_tokenizer('ViT-B-32')
            text = tokenizer(["a photo of a cat", "a photo of a dog"])
            
            with torch.no_grad():
                text_features = model.encode_text(text)
            
            log_with_timestamp(f"Text features shape: {text_features.shape}")
            log_with_timestamp(f"✅ Offline test successful!")
            
        else:
            log_with_timestamp("❌ No model file found in directory")
    else:
        log_with_timestamp(f"❌ Model directory not found: {model_path}")
    
    log_with_timestamp("🎉 Offline OpenCLIP loading works! Ready for SANW experiments.")
    
except Exception as e:
    log_with_timestamp(f"❌ Test failed: {repr(e)}")
    log_with_timestamp("Full traceback:")
    traceback.print_exc()
    log_with_timestamp("❌ Still hitting issues.")
PY

echo "=== DONE ==="
