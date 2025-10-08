#!/bin/bash
#SBATCH --job-name=test_clip_mp
#SBATCH --output=test_clip_mp_%j.log
#SBATCH --error=test_clip_mp_%j.err
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

echo "=== INSTALL PYTORCH + TRANSFORMERS ==="
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install transformers scikit-learn pandas pillow safetensors

echo "=== CUDA INFO ==="
python - <<'PY'
import torch
print('Torch:', torch.__version__)
print('CUDA is_available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
PY

echo "=== TESTING CLIP LOAD WITH MULTIPROCESS ==="
python - <<'PY'
import traceback
import torch
import torch.multiprocessing as mp
from transformers import CLIPModel

def load_clip_in_process():
    try:
        print('Loading CLIP in separate process...')
        # Try loading without safetensors first
        model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32', use_safetensors=False)
        print('Loaded without safetensors. Params:', sum(p.numel() for p in model.parameters()))
        return True
    except Exception as e:
        print('Failed without safetensors:', repr(e))
        try:
            # Try with safetensors but smaller batch
            model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32', use_safetensors=True, torch_dtype=torch.float16)
            print('Loaded with float16. Params:', sum(p.numel() for p in model.parameters()))
            return True
        except Exception as e2:
            print('Failed with float16:', repr(e2))
            return False

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    success = load_clip_in_process()
    print('Success:', success)
PY

echo "=== DONE ==="
