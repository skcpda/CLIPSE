#!/bin/bash
#SBATCH --job-name=test_openclip_hf
#SBATCH --output=test_openclip_hf_%j.log
#SBATCH --error=test_openclip_hf_%j.err
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:30:00

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# HF Hub environment for faster, reliable downloads
export HF_HOME=$PWD/.hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_TELEMETRY=1
export OPENCLIP_CACHE_PATH=$PWD/.openclip_cache
mkdir -p "$HF_HOME" "$OPENCLIP_CACHE_PATH"

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

echo "=== TESTING OPENCLIP WITH HF MIRROR ==="
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

log_with_timestamp("Starting OpenCLIP test with HF mirror...")

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_with_timestamp(f"Using device: {device}")
    
    # Test both models with HF Hub paths
    models_to_test = [
        ("hf-hub:laion/CLIP-ViT-B-32-laion400M-e32", "Primary (laion400m_e32)"),
        ("hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K", "Confirmatory (laion2b_s34b_b79k)")
    ]
    
    for ckpt, name in models_to_test:
        log_with_timestamp(f"Loading {name}: {ckpt}")
        start_time = time.time()
        
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=ckpt, device=device
        )
        
        load_time = time.time() - start_time
        log_with_timestamp(f"✅ {name} loaded successfully in {load_time:.2f} seconds!")
        log_with_timestamp(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
        log_with_timestamp(f"Logit scale: {float(model.logit_scale.exp()):.3f}")
        
        # Quick test
        log_with_timestamp("Testing inference...")
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        text = tokenizer(["a photo of a cat", "a photo of a dog"])
        
        with torch.no_grad():
            text_features = model.encode_text(text)
        
        log_with_timestamp(f"Text features shape: {text_features.shape}")
        log_with_timestamp(f"✅ {name} test successful!")
        
        # Clean up for next model
        del model, preprocess, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    log_with_timestamp("🎉 All models loaded successfully! Ready for SANW experiments.")
    
except Exception as e:
    log_with_timestamp(f"❌ Test failed: {repr(e)}")
    log_with_timestamp("Full traceback:")
    traceback.print_exc()
    log_with_timestamp("❌ Still hitting issues.")
PY

echo "=== DONE ==="
