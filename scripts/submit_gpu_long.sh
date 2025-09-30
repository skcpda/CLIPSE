#!/bin/bash
#SBATCH --job-name=sanw_experiments_long
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=12:00:00
#SBATCH --output=sanw_experiments_long_%j.out
#SBATCH --error=sanw_experiments_long_%j.err

echo "🚀 Starting SANW Experiments on GPU Cluster (Long Version)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working Directory: $(pwd)"
echo "Date: $(date)"
echo ""

# Load modules
module load python/3.9
module load cuda/11.8
module load gcc/9.3.0

# Set up environment
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check CUDA availability
echo "🔍 Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"CPU\"}')"

# Install dependencies if needed
echo "📦 Installing dependencies..."
pip install --user torch torchvision open-clip-torch timm numpy pandas scikit-learn Pillow tqdm PyYAML matplotlib seaborn

# Run all experiments
echo "🧪 Running all SANW experiments..."
python run_all_experiments.py

# Analyze results
echo "📊 Analyzing results..."
python analyze_results.py

echo "✅ All experiments completed!"
echo "Results saved in runs/ directory"
echo "Summary files: results_summary.txt, results_table.tex"
