#!/bin/bash
#SBATCH -J clipse-train
#SBATCH -p gpu-short
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err

set -euo pipefail
umask 007

# Paths (keep consistent with earlier setup)
MODEL_DIR="/nfs_home/users/poonam/sanw_experiments/data/clip_models/clip-vit-base-patch32"
ENV_DIR="/nfs_home/users/poonam/sanw_experiments/venv"
HF_HOME="$(dirname "$MODEL_DIR")"
CFG="$1"  # pass a config path, e.g., configs/flickr8k_vitb32_debias.yaml

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1 HF_HOME MODEL_DIR

# Activate conda environment
source ~/.conda/envs/torch310/bin/activate

# Run training
python -u -m src.train_clipse --config "$CFG"
