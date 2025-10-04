#!/bin/bash
#SBATCH --job-name=sanw_simple
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=sanw_simple_%j.out
#SBATCH --error=sanw_simple_%j.err

echo "🚀 Starting SANW Experiments - Simple Approach"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working Directory: $(pwd)"
echo "Date: $(date)"
echo ""

# Load system modules
module load python/3.8
module load cuda/11.0

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

echo "🔧 Environment Setup:"
echo "Python: $(which python3)"
echo "Python version: $(python3 --version)"
echo "CUDA: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo ""

# Try to run with system packages first
echo "🔍 Testing system packages..."
python3 -c "import sys; print('Python path:', sys.path)"
python3 -c "import numpy; print('NumPy available')" 2>/dev/null || echo "NumPy not available"
python3 -c "import torch; print('PyTorch available')" 2>/dev/null || echo "PyTorch not available"

echo ""
echo "🚀 Starting SANW Experiments..."
echo "Running all experiments with seeds [13, 17, 23]"
echo ""

# Run the experiments
python3 run_all_experiments.py

echo ""
echo "✅ All SANW experiments completed!"
echo "📊 Results saved in runs/ directory"
echo "Date: $(date)"
