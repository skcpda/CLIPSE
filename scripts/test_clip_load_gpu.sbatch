#!/bin/bash
#SBATCH --job-name=test_clip_gpu
#SBATCH --output=test_clip_gpu_%j.log
#SBATCH --error=test_clip_gpu_%j.err
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "=== SLURM LIMITS ==="
ulimit -a
echo
echo "=== /proc/self/limits ==="
cat /proc/self/limits || true
echo

echo "=== CUDA INFO ==="
/nfs_home/software/miniconda/bin/conda run -n torch310_fixed python - <<'PY'
import torch
print('Torch:', torch.__version__)
print('CUDA is_available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
PY

echo "=== TESTING CLIP LOAD ==="
/nfs_home/software/miniconda/bin/conda run -n torch310_fixed python - <<'PY'
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


