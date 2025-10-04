#!/bin/bash
#SBATCH -J clipse-full
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 24:00:00
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

# EDIT
export PROJECT_ROOT="/path/to/CLIPSE"
export CLIP_LOCAL_DIR="/path/to/hf_clip_vitb32"

mkdir -p "$TRANSFORMERS_CACHE/openai/clip-vit-base-patch32" logs
rsync -a "$CLIP_LOCAL_DIR/" "$TRANSFORMERS_CACHE/openai/clip-vit-base-patch32/"

cd "$PROJECT_ROOT"

seeds=(13 17 23)
configs=(configs/flickr8k_vitb32_baseline.yaml configs/flickr8k_vitb32_debias.yaml configs/flickr8k_vitb32_bandpass.yaml)

for s in "${seeds[@]}"; do
  for cfg in "${configs[@]}"; do
    python -m src.train_advanced \
      --config "$cfg" \
      --seed "$s" \
      --device auto \
      --local_model_dir "$TRANSFORMERS_CACHE/openai/clip-vit-base-patch32" \
      --num_workers 4 \
      --topical_k 80 \
      --adaptive_k true \
      --epochs 20
  done
done

echo "All runs completed."

