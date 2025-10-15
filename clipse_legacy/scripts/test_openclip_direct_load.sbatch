#!/bin/bash
#SBATCH --job-name=test_openclip_dir
#SBATCH --output=test_openclip_dir_%j.log
#SBATCH --error=test_openclip_dir_%j.err
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
MODEL_PATH="/nfs_home/users/poonam/clipse/models/laion2b_s34b_b79k"
echo "Looking for models at: $MODEL_PATH"
ls -la "$MODEL_PATH" || echo "Models not found at expected path"
du -sh "$MODEL_PATH" || echo "Cannot get size"

echo "=== TESTING OPENCLIP DIRECT LOADING ==="
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

log_with_timestamp("Starting OpenCLIP direct loading test...")

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_with_timestamp(f"Using device: {device}")
    
    # Method 1: Try loading with local cache directory
    model_path = "/nfs_home/users/poonam/clipse/models/laion2b_s34b_b79k"
    log_with_timestamp(f"Loading model from: {model_path}")
    
    # Set the cache directory to our model path
    import open_clip
    open_clip.list_pretrained()
    
    # Try to load using the model path directly
    start_time = time.time()
    
    # Method 1: Try with local_files_only
    try:
        log_with_timestamp("Attempting direct model loading...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", 
            pretrained=model_path,
            device=device
        )
        load_time = time.time() - start_time
        log_with_timestamp(f"✅ Direct loading successful in {load_time:.2f} seconds!")
        
    except Exception as e1:
        log_with_timestamp(f"Direct loading failed: {e1}")
        
        # Method 2: Try copying model files to expected cache location
        log_with_timestamp("Attempting cache-based loading...")
        cache_dir = "/tmp/${SLURM_JOB_ID}/openclip_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Copy model files to cache
        import shutil
        target_dir = os.path.join(cache_dir, "laion2b_s34b_b79k")
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.copytree(model_path, target_dir)
        
        # Try loading from cache
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", 
            pretrained="laion2b_s34b_b79k",
            device=device
        )
        load_time = time.time() - start_time
        log_with_timestamp(f"✅ Cache-based loading successful in {load_time:.2f} seconds!")
    
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
    log_with_timestamp(f"✅ OpenCLIP offline loading works!")
    log_with_timestamp("🎉 Ready for SANW experiments!")
    
except Exception as e:
    log_with_timestamp(f"❌ Test failed: {repr(e)}")
    log_with_timestamp("Full traceback:")
    traceback.print_exc()
    log_with_timestamp("❌ Still hitting issues.")
PY

echo "=== DONE ==="
