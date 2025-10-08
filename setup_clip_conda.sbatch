#!/bin/bash
#SBATCH -J clip-conda-setup
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

echo "🚀 Setting up CLIP with conda (no-plugins mode)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Create logs directory
mkdir -p logs

# Set up conda environment
echo "🔧 Setting up conda environment..."
export CONDA_NO_PLUGINS=true
source /usr/local/bin/conda

# Create a new environment
echo "📦 Creating conda environment..."
conda create -n clip_env python=3.10 -y
source activate clip_env

# Install packages
echo "📦 Installing ML packages..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install transformers datasets pillow numpy -c conda-forge -y

# Test the setup
echo "🧪 Testing CLIP setup..."
python -c "
import torch
from transformers import CLIPModel, CLIPProcessor
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name(0)}')

# Test CLIP loading
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
print('✅ CLIP model loaded successfully!')
"

echo "✅ CLIP baseline setup complete!"
