#!/bin/bash
#SBATCH --job-name=basic_test
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=basic_%j.out
#SBATCH --error=basic_%j.err

echo "🚀 Basic test starting at $(date)"
echo "Working directory: $(pwd)"
echo "User: $(whoami)"
echo "Hostname: $(hostname)"
echo "🚀 Basic test completed at $(date)"
