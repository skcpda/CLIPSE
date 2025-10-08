#!/bin/bash
#SBATCH -J quick-test
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -o quick-test_%j.out
#SBATCH -e quick-test_%j.err

set -euo pipefail

echo "🚀 Quick SANW Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Activate conda environment
source ~/.conda/envs/torch310/bin/activate

# Run quick test
cd clipse
python quick_test.py

echo "✅ Quick test completed!"
