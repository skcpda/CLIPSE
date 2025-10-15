# CLIPSE Deployment Guide

## Cluster Setup

### Prerequisites
- SLURM workload manager
- CUDA-enabled GPU nodes
- Conda environment with PyTorch
- Sufficient storage for models and datasets

### Environment Setup

1. **Activate Conda Environment**
```bash
source /path/to/miniconda/etc/profile.d/conda.sh
conda activate torch310
```

2. **Install Dependencies**
```bash
pip install open-clip-torch timm ftfy regex tqdm pandas pillow scikit-learn pyyaml
```

### Model Pre-staging

1. **Download Models Locally**
```bash
python download_models_locally.py
```

2. **Transfer to Cluster**
```bash
scp -r models/ user@cluster:/path/to/clipse/models/
```

### Dataset Preparation

1. **Flickr8k Dataset**
```bash
# Download and extract Flickr8k
wget http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_Dataset.zip
unzip Flickr8k_Dataset.zip
```

2. **Organize Structure**
```
data/flickr8k/
├── Images/
├── captions.tsv
├── Flickr_8k.trainImages.txt
├── Flickr_8k.devImages.txt
└── Flickr_8k.testImages.txt
```

## Job Submission

### Training Jobs

1. **SANW-Debias Training**
```bash
sbatch scripts/train_sanw_debias.sbatch
```

2. **SANW-Bandpass Training**
```bash
sbatch scripts/train_sanw_bandpass.sbatch
```

### Evaluation Jobs

```bash
sbatch scripts/eval_retrieval.sbatch
```

## Monitoring

### Job Status
```bash
squeue -u $USER
```

### Live Logs
```bash
tail -f scripts/sanw_debias_*.log
tail -f scripts/sanw_bandpass_*.log
```

### Resource Usage
```bash
sstat -j <job_id>
```

## Configuration

### SLURM Parameters
- **Partition**: gpu-long (for training), gpu-short (for evaluation)
- **Memory**: 48G (adjust based on model size)
- **Time**: 2:00:00 (training), 0:30:00 (evaluation)
- **GPU**: 1 GPU per job

### Training Parameters
- **Batch Size**: 32 (adjust based on GPU memory)
- **Epochs**: 10 (adjust based on convergence)
- **Learning Rate**: 1e-5 (conservative start)
- **SANW Parameters**: Start conservative, increase gradually

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce batch size
   - Use gradient accumulation
   - Check model size

2. **CUDA Errors**
   - Verify GPU availability
   - Check CUDA version compatibility
   - Ensure proper environment

3. **Import Errors**
   - Verify Python path
   - Check package installation
   - Ensure proper module structure

4. **Data Loading Issues**
   - Verify dataset paths
   - Check file permissions
   - Ensure proper data format

### Debugging

1. **Check Logs**
```bash
cat scripts/*.err
cat scripts/*.log
```

2. **Test Environment**
```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import open_clip; print('OpenCLIP available')"
```

3. **Verify Data**
```bash
python -c "from src.datasets import Flickr8kDataset; print('Dataset OK')"
```

## Performance Optimization

### Memory Optimization
- Use mixed precision training
- Implement gradient checkpointing
- Optimize batch size

### Speed Optimization
- Use multiple workers for data loading
- Enable cuDNN optimizations
- Use efficient data formats

### Storage Optimization
- Compress model checkpoints
- Use efficient logging
- Clean up temporary files

## Results Analysis

### Training Metrics
- Loss convergence curves
- Temperature dynamics
- SANW weight statistics

### Evaluation Metrics
- Retrieval performance (R@K, MR, mAP)
- Zero-shot classification
- Computational efficiency

### Visualization
- Plot training curves
- Analyze weight distributions
- Compare method performance
