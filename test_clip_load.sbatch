#!/bin/bash
#SBATCH --job-name=test_clip_load
#SBATCH --output=test_clip_load_%j.log
#SBATCH --error=test_clip_load_%j.err
#SBATCH --partition=cpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00

# Set threading limits
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export CUDA_VISIBLE_DEVICES=''

echo "=== SLURM JOB INFO ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo

echo "=== MEMORY AND LIMITS ==="
free -h
echo
ulimit -a
echo

echo "=== TESTING CLIP MODEL LOADING ==="
/nfs_home/users/poonam/.conda/envs/torch310_fixed/bin/python -c 'from transformers import CLIPModel; print("Loading CLIP model..."); model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True); print("CLIP model loaded successfully!"); print(f"Model device: {next(model.parameters()).device}"); print(f"Model size: {sum(p.numel() for p in model.parameters()):,} parameters")'

echo
echo "=== TEST COMPLETE ==="

