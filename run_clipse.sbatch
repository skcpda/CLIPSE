#!/bin/bash
#SBATCH -J clipse
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

module purge
module load miniconda
source activate clipse

source env.cluster.sh

export JOB_CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/${USER}/clipse_${SLURM_JOB_ID}"
mkdir -p "$JOB_CACHE_ROOT"
export HF_HOME="$JOB_CACHE_ROOT/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$JOB_CACHE_ROOT/torch"
export MPLCONFIGDIR="$JOB_CACHE_ROOT/mpl"

# EDIT THESE PATHS
export PROJECT_ROOT="/path/to/CLIPSE"
export CLIP_LOCAL_DIR="/path/to/hf_clip_vitb32"

mkdir -p "$TRANSFORMERS_CACHE/openai/clip-vit-base-patch32" logs
rsync -a "$CLIP_LOCAL_DIR/" "$TRANSFORMERS_CACHE/openai/clip-vit-base-patch32/"

cd "$PROJECT_ROOT"

python -m src.train_advanced \
  --config configs/flickr8k_vitb32_baseline.yaml \
  --seed 13 \
  --device auto \
  --local_model_dir "$TRANSFORMERS_CACHE/openai/clip-vit-base-patch32" \
  --num_workers 4 \
  --topical_k 80 \
  --adaptive_k true \
  --epochs 5

python -m src.train_advanced \
  --config configs/flickr8k_vitb32_debias.yaml \
  --seed 13 \
  --device auto \
  --local_model_dir "$TRANSFORMERS_CACHE/openai/clip-vit-base-patch32" \
  --num_workers 4 \
  --topical_k 80 \
  --adaptive_k true \
  --epochs 5

python -m src.train_advanced \
  --config configs/flickr8k_vitb32_bandpass.yaml \
  --seed 13 \
  --device auto \
  --local_model_dir "$TRANSFORMERS_CACHE/openai/clip-vit-base-patch32" \
  --num_workers 4 \
  --topical_k 80 \
  --adaptive_k true \
  --epochs 5

echo "Done."

