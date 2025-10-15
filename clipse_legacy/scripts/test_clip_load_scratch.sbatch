#!/bin/bash
#SBATCH --job-name=test_clip_scratch
#SBATCH --output=test_clip_scratch_%j.log
#SBATCH --error=test_clip_scratch_%j.err
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:45:00

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "=== SLURM INFO ==="
echo "Job: $SLURM_JOB_ID on $SLURM_JOB_PARTITION"
ulimit -a || true
cat /proc/self/limits || true
echo

SCRATCH_BASE="/scratch/$USER/clipse_env"
MINIFORGE_DIR="$SCRATCH_BASE/miniforge"
ENV_NAME="torch310_fixed"
PY_BIN="$MINIFORGE_DIR/envs/$ENV_NAME/bin/python"
PIP_BIN="$MINIFORGE_DIR/envs/$ENV_NAME/bin/pip"

mkdir -p "$SCRATCH_BASE"
cd "$SCRATCH_BASE"

echo "=== SETUP MINIFORGE UNDER SCRATCH ==="
if [ ! -x "$MINIFORGE_DIR/bin/conda" ]; then
  echo "Installing Miniforge to $MINIFORGE_DIR"
  curl -L -o Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash Miniforge3.sh -b -p "$MINIFORGE_DIR"
fi

echo "=== CREATE/UPDATE CONDA ENV ($ENV_NAME) ==="
"$MINIFORGE_DIR/bin/conda" create -y -n "$ENV_NAME" python=3.10 || true

echo "=== INSTALL PYTORCH CU118 + TRANSFORMERS ==="
"$MINIFORGE_DIR/bin/conda" run -n "$ENV_NAME" python -m pip install --upgrade pip
"$MINIFORGE_DIR/bin/conda" run -n "$ENV_NAME" python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
"$MINIFORGE_DIR/bin/conda" run -n "$ENV_NAME" python -m pip install transformers scikit-learn pandas pillow safetensors

echo "=== CUDA INFO ==="
"$MINIFORGE_DIR/bin/conda" run -n "$ENV_NAME" python - <<'PY'
import torch
print('Torch:', torch.__version__)
print('CUDA is_available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
PY

echo "=== TESTING CLIP LOAD ==="
"$MINIFORGE_DIR/bin/conda" run -n "$ENV_NAME" python - <<'PY'
import traceback
from transformers import CLIPModel
print('Loading CLIP...')
try:
    m = CLIPModel.from_pretrained('openai/clip-vit-base-patch32', use_safetensors=True)
    print('Loaded. Params:', sum(p.numel() for p in m.parameters()))
except Exception as e:
    print('EXC:', repr(e))
    traceback.print_exc()
PY

echo "=== DONE ==="


