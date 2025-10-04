#!/bin/bash
#SBATCH --job-name=sanw_single
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=sanw_single_%j.out
#SBATCH --error=sanw_single_%j.err

echo "🚀 Running single SANW experiment: $1"
echo "Date: $(date)"

cd ~/sanw_experiments
source .venv/bin/activate

echo "🔍 Environment check:"
echo "Python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

echo "🧪 Running experiment: $1"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -m src.train_advanced --config "$1"

echo "✅ Experiment completed at $(date)"
