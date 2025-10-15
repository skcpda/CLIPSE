#!/bin/bash
#SBATCH --job-name=test_openclip_log
#SBATCH --output=test_openclip_log_%j.log
#SBATCH --error=test_openclip_log_%j.err
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

echo "=== TESTING OPENCLIP WITH LIVE LOGGING ==="
python - <<'PY'
import traceback
import open_clip
import torch
import time
import sys

def log_with_timestamp(msg):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()

log_with_timestamp("Starting OpenCLIP test...")

try:
    log_with_timestamp("Importing open_clip...")
    import open_clip
    
    log_with_timestamp("Creating model and transforms...")
    log_with_timestamp("Model: ViT-B-32, Pretrained: laion400m_e32")
    
    # Add timeout and progress tracking
    start_time = time.time()
    log_with_timestamp("Loading model (this may take a while)...")
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion400m_e32'
    )
    
    load_time = time.time() - start_time
    log_with_timestamp(f"Model loaded successfully in {load_time:.2f} seconds!")
    
    log_with_timestamp("Getting tokenizer...")
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    log_with_timestamp("✅ PRIMARY model loaded successfully!")
    log_with_timestamp(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    log_with_timestamp(f"Model device: {next(model.parameters()).device}")
    
    log_with_timestamp("Moving model to GPU...")
    model = model.cuda()
    
    log_with_timestamp("Testing text encoding...")
    text = tokenizer(["a photo of a cat", "a photo of a dog"])
    
    log_with_timestamp("Running inference...")
    with torch.no_grad():
        text_features = model.encode_text(text)
    
    log_with_timestamp(f"Text features shape: {text_features.shape}")
    log_with_timestamp("✅ PRIMARY tier test successful!")
    log_with_timestamp("🎉 OpenCLIP works! Ready for SANW experiments.")
    
except Exception as e:
    log_with_timestamp(f"❌ PRIMARY tier failed: {repr(e)}")
    log_with_timestamp("Full traceback:")
    traceback.print_exc()
    log_with_timestamp("❌ Still hitting issues.")
PY

echo "=== DONE ==="
