#!/bin/bash
#SBATCH --job-name=clipse_eval
#SBATCH --output=scripts/clipse_eval_%j.log
#SBATCH --error=scripts/clipse_eval_%j.err
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

export HF_HUB_OFFLINE=1
export HF_HOME="/nfs_home/users/poonam/clipse/.hf_cache"
export OPENCLIP_CACHE_PATH="/nfs_home/users/poonam/clipse/.openclip_cache"
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

echo "=== INSTALL DEPENDENCIES ==="
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install open_clip_torch timm ftfy regex tqdm pandas pillow scikit-learn pyyaml

echo "=== CUDA INFO ==="
python - <<'PY'
import torch
print('Torch:', torch.__version__)
print('CUDA is_available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
PY

echo "=== RUN EVALUATION ==="
python -u /nfs_home/users/poonam/clipse/src/eval_retrieval.py \
  --model_name ViT-B-32 \
  --pretrained_tag laion2b_s34b_b79k \
  --local_checkpoint_path /nfs_home/users/poonam/clipse/models/laion2b_s34b_b79k/open_clip_pytorch_model.bin \
  --dataset_path /nfs_home/users/poonam/clipse/data/flickr8k \
  --batch_size 128 \
  --seed 42 \
  --method baseline \
  --epoch -1 \
  --log_file /nfs_home/users/poonam/clipse/results/run_registry.csv

echo "=== DONE ==="


