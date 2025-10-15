#!/bin/bash
#SBATCH --job-name=test_clip_cpu
#SBATCH --output=test_clip_cpu_%j.log
#SBATCH --error=test_clip_cpu_%j.err
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

echo "=== INSTALL CPU-ONLY PYTORCH (FAST) ==="
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers scikit-learn pandas pillow safetensors

echo "=== CUDA INFO ==="
python - <<'PY'
import torch
print('Torch:', torch.__version__)
print('CUDA is_available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
PY

echo "=== TESTING CLIP LOAD (CPU) ==="
python - <<'PY'
import traceback
from transformers import CLIPModel
print('Loading CLIP...')
try:
    m = CLIPModel.from_pretrained('openai/clip-vit-base-patch32', use_safetensors=True)
    print('Loaded. Params:', sum(p.numel() for p in m.parameters()))
    print('Model device:', next(m.parameters()).device)
except Exception as e:
    print('EXC:', repr(e))
    traceback.print_exc()
PY

echo "=== DONE ==="
