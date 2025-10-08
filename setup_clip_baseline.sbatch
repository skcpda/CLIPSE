#!/bin/bash
#SBATCH -J clip-setup
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

echo "🚀 Setting up CLIP baseline model on CSIS Cluster"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Create logs directory
mkdir -p logs

# Set up environment
echo "🔧 Setting up Python environment..."

# Try to use existing conda/mamba if available
if command -v conda &> /dev/null; then
    echo "Using conda environment..."
    source ~/.bashrc
    conda activate base
elif command -v mamba &> /dev/null; then
    echo "Using mamba environment..."
    source ~/.bashrc
    mamba activate base
else
    echo "Using system Python..."
fi

# Install required packages if not available
echo "📦 Installing required packages..."
/usr/bin/python3.12 -m pip install --user --upgrade pip
/usr/bin/python3.12 -m pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cu118
/usr/bin/python3.12 -m pip install --user transformers datasets pillow requests numpy

# Run the setup script
echo "🧪 Running CLIP setup test..."
/usr/bin/python3.12 setup_clip_baseline.py

echo "✅ CLIP baseline setup complete!"
