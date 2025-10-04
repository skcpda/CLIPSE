#!/bin/bash
#SBATCH --job-name=sanw_system
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=sanw_system_%j.out
#SBATCH --error=sanw_system_%j.err

echo "🚀 Starting SANW Experiments - System Environment"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working Directory: $(pwd)"
echo "Date: $(date)"
echo ""

# Load system modules
module load python/3.8
module load cuda/11.0
module load gcc/9.3.0

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

echo "🔧 Environment Setup:"
echo "Python: $(which python3)"
echo "Python version: $(python3 --version)"
echo "CUDA: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo ""

# Use system packages and install missing ones
echo "📦 Setting up environment..."
python3 -m pip install --user --upgrade pip
python3 -m pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu110
python3 -m pip install --user transformers datasets accelerate
python3 -m pip install --user numpy pandas scikit-learn matplotlib seaborn
python3 -m pip install --user tqdm wandb

echo ""
echo "🔍 Verifying installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"

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
