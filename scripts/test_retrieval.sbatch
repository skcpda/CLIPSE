#!/bin/bash
#SBATCH -J clipse-eval
#SBATCH -p gpu-short
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err

set -euo pipefail
umask 007

# Paths
MODEL_DIR="/nfs_home/users/poonam/sanw_experiments/data/clip_models/clip-vit-base-patch32"
HF_HOME="$(dirname "$MODEL_DIR")"
CHECKPOINT="$1"  # path to checkpoint file

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1 HF_HOME MODEL_DIR

# Activate conda environment
source ~/.conda/envs/torch310/bin/activate

# Run evaluation
python -u -m src.eval_retrieval --checkpoint "$CHECKPOINT"
