#!/bin/bash
#SBATCH --job-name=sanw_direct
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=sanw_direct_%j.out
#SBATCH --error=sanw_direct_%j.err

echo "🚀 Starting SANW Experiments - Direct Approach"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working Directory: $(pwd)"
echo "Date: $(date)"
echo ""

# Load necessary modules
module load python/3.8
module load cuda/11.0

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

echo "🔧 Environment Setup:"
echo "Python: $(which python3)"
echo "CUDA: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo ""

# Install requirements if needed
echo "📦 Installing requirements..."
pip3 install --user torch torchvision transformers datasets accelerate
pip3 install --user numpy pandas scikit-learn matplotlib seaborn
pip3 install --user tqdm wandb

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
