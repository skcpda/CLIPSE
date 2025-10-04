#!/bin/bash
#SBATCH --job-name=sanw_simple
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=sanw_simple_%j.out
#SBATCH --error=sanw_simple_%j.err

echo "🚀 Starting SANW experiments (simplified)"
echo "Date: $(date)"
echo "Working directory: $(pwd)"

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Check environment
echo "🔍 Environment check:"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Run experiments directly
echo "🧪 Running experiments..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python run_all_experiments.py

echo "✅ Experiments completed at $(date)"
