#!/bin/bash
#SBATCH --job-name=sanw_deb
#SBATCH --output=scripts/sanw_debias_%j.log
#SBATCH --error=scripts/sanw_debias_%j.err
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Use offline mode with local models
export HF_HUB_OFFLINE=1
export HF_HOME="/nfs_home/users/poonam/clipse/.hf_cache"
export OPENCLIP_CACHE_PATH="/nfs_home/users/poonam/clipse/models"
mkdir -p "$HF_HOME" "$OPENCLIP_CACHE_PATH"

echo "=== LIMITS ==="
ulimit -a || true
cat /proc/self/limits || true
echo

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
command -v python3.11 || true
python3.11 -V || true

echo "=== USING CONDA ENVIRONMENT (torch310) ==="
# Use existing conda environment with CUDA PyTorch
source /nfs_home/software/miniconda/etc/profile.d/conda.sh
conda activate torch310
echo "Using conda environment: torch310"

echo "=== INSTALL DEPENDENCIES ==="
# Install additional packages in the conda environment
python -m pip install open_clip_torch timm ftfy regex tqdm pandas pillow scikit-learn pyyaml

echo "=== CUDA INFO ==="
python - <<'PY'
import torch
print('Torch:', torch.__version__)
print('CUDA is_available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
PY

echo "=== RUN SANW-DEBIAS TRAINING ==="
       MODEL_NAME="ViT-B-32"
       PRETRAINED_TAG="laion2b_s34b_b79k"
       LOCAL_CHECKPOINT_DIR="/nfs_home/users/poonam/clipse/models/laion2b_s34b_b79k"
       LOCAL_CHECKPOINT_FILE="${LOCAL_CHECKPOINT_DIR}/open_clip_pytorch_model.bin"
DATASET_PATH="/nfs_home/users/poonam/clipse/data/flickr8k"
LOG_DIR="/nfs_home/users/poonam/clipse/results"
CHECKPOINT_DIR="/nfs_home/users/poonam/clipse/checkpoints"
RUN_ID="sanw_debias_${SLURM_JOB_ID}"
SEED=13

# SANW-Debias parameters (conservative start)
ALPHA=0.1
DELTA=0.2
LAMBDA=0.5

python -u /nfs_home/users/poonam/clipse/src/train_clipse.py \
    --model_name "${MODEL_NAME}" \
    --pretrained_tag "${PRETRAINED_TAG}" \
    --local_checkpoint_path "${LOCAL_CHECKPOINT_FILE}" \
    --dataset_path "${DATASET_PATH}" \
    --log_dir "${LOG_DIR}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --run_id "${RUN_ID}" \
    --seed "${SEED}" \
    --epochs 10 \
    --batch_size 32 \
    --loss_fn "sanw_debias" \
    --sanw_alpha "${ALPHA}" \
    --sanw_delta "${DELTA}" \
    --sanw_lambda "${LAMBDA}" \
    --lr 1e-5 \
    --wd 0.01 \
    --warmup_epochs 1 \
    --grad_clip 1.0

echo "=== DONE ==="