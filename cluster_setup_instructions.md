# GPU Cluster Setup Instructions

## 🚀 **Cluster Details**
- **IP**: 172.24.16.132
- **Username**: poonam
- **Password**: 900n@M
- **Working Directory**: ~/sanw_experiments

## 📋 **Step-by-Step Setup**

### 1. **Connect to Cluster**
```bash
ssh poonam@172.24.16.132
# Password: 900n@M
```

### 2. **Create Project Directory**
```bash
mkdir -p ~/sanw_experiments
cd ~/sanw_experiments
```

### 3. **Transfer Code from Local Machine**
From your local terminal (not on cluster):
```bash
cd /Users/priyankjairaj/Downloads/clipse
scp -r . poonam@172.24.16.132:~/sanw_experiments/
```

### 4. **On the Cluster - Install Dependencies**
```bash
# Load required modules
module load python/3.9
module load cuda/11.8
module load gcc/9.3.0

# Install Python packages
pip install --user torch torchvision open-clip-torch timm numpy pandas scikit-learn Pillow tqdm PyYAML matplotlib seaborn
```

### 5. **Check GPU Availability**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"
```

### 6. **Submit Job**
```bash
sbatch scripts/submit_cluster_job.sh
```

### 7. **Monitor Job**
```bash
# Check job status
squeue -u poonam

# Check job output
tail -f sanw_experiments_*.out

# Check for errors
tail -f sanw_experiments_*.err
```

### 8. **Check Results**
```bash
# List experiment results
ls runs/

# View summary
cat results_summary.txt

# View LaTeX table
cat results_table.tex
```

### 9. **Transfer Results Back**
From your local machine:
```bash
# Transfer results directory
scp -r poonam@172.24.16.132:~/sanw_experiments/runs/ /Users/priyankjairaj/Downloads/clipse/

# Transfer summary files
scp poonam@172.24.16.132:~/sanw_experiments/results_*.txt /Users/priyankjairaj/Downloads/clipse/
scp poonam@172.24.16.132:~/sanw_experiments/results_*.tex /Users/priyankjairaj/Downloads/clipse/
```

## ⚙️ **Job Configuration**
- **Partition**: gpu
- **GPU**: 1 GPU
- **CPUs**: 4 cores
- **Memory**: 16GB
- **Time Limit**: 12 hours
- **Expected Runtime**: ~2-4 hours

## 📊 **Experiments to Run**
1. **Baseline (ViTB32)**: 20 epochs × 3 seeds
2. **SANW-Debias (ViTB32)**: 20 epochs × 3 seeds  
3. **SANW-Bandpass (ViTB32)**: 50 epochs × 3 seeds

**Total**: 270 epochs across all experiments

## 🔧 **Troubleshooting**
- If job fails, check error logs: `cat sanw_experiments_*.err`
- If CUDA not available, check: `nvidia-smi`
- If dependencies missing, reinstall: `pip install --user <package>`
- If job times out, increase time limit in `scripts/submit_cluster_job.sh`

## 📈 **Expected Results**
- **Baseline**: ~5-10% R@1
- **SANW-Debias**: ~15-25% R@1  
- **SANW-Bandpass**: ~20-30% R@1

Results will be saved in `runs/` directory with CSV metrics and model checkpoints.
