#!/bin/bash
#SBATCH --job-name=sanw_experiments
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=4:00:00
#SBATCH --output=sanw_experiments_%j.out
#SBATCH --error=sanw_experiments_%j.err

set -euo pipefail

echo "🚀 Starting SANW Experiments on GPU Cluster"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Working Directory: $(pwd)"
echo "Date: $(date)"
echo ""

# Environment
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# Use system Python with virtual environment (conda is broken)
echo "🔧 Using system Python with virtual environment..."
PY_EXE="$(command -v python3 || command -v python)"
echo "Using system Python: $PY_EXE"

# Create and activate virtual environment
VENV_DIR=".venv"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "🐍 Creating virtual environment..."
  "$PY_EXE" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

echo "Using Python at: ${PY_EXE}"

echo "Using Python: $(python --version)"
echo "Using pip: $(pip --version)"

# Install dependencies with modern Python
echo "📦 Installing dependencies with Python 3.9..."

# Upgrade pip to latest version
echo "Upgrading pip to latest version..."
python -m pip install --upgrade pip
echo "Pip version: $(pip --version)"

# Install dataclasses first (required by torch on Python 3.6)
echo "Installing dataclasses..."
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org dataclasses || {
  echo "⚠️  dataclasses not available, continuing without it..."
}

# Install typing-extensions first (compatible with Python 3.6)
echo "Installing typing-extensions..."
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org typing-extensions || {
  echo "⚠️  typing-extensions not available, continuing without it..."
}

# Install numpy first (required by torch)
echo "Installing numpy..."
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org numpy==1.19.5

# Install Pillow first (required by torchvision)
echo "Installing Pillow..."
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org Pillow==8.3.2

# Install PyTorch with CUDA support for GPU cluster
echo "Installing PyTorch with CUDA support for Python 3.6..."
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org torch==1.7.1+cu110 torchvision==0.8.2+cu110 --index-url https://download.pytorch.org/whl/cu110

# Verify CUDA installation
echo "🔍 Verifying CUDA installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA compiled version: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  Device {i}: {torch.cuda.get_device_name(i)}')
else:
    print('❌ CUDA not available - checking if CPU version was installed instead')
    print('This might be due to network issues or incompatible CUDA version')
"

# Install other dependencies with compatible versions (all at once)
echo "Installing other packages with compatible versions..."
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --timeout 300 --retries 3 timm==0.4.12 pandas==1.1.5 scikit-learn==0.24.2 tqdm==4.64.1 PyYAML==5.4.1 matplotlib==3.3.4 seaborn==0.11.2

# Skip open-clip-torch (too many compatibility issues with Python 3.6)
echo "⚠️  Skipping open-clip-torch (compatibility issues with Python 3.6)..."

# Install transformers as primary CLIP implementation (last version compatible with Python 3.6)
echo "🔧 Installing transformers as CLIP implementation (Python 3.6 compatible)..."
pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --timeout 300 --retries 3 transformers==4.18.0

# Download CLIP model files manually to avoid Python network issues
echo "🔧 Downloading CLIP model files manually..."
mkdir -p ~/.cache/huggingface/transformers/models--openai--clip-vit-base-patch32/snapshots/

# Try to download the model files using wget/curl
echo "Attempting to download CLIP model files..."
python -c "
import os
import requests
import json
import hashlib

# Create the model directory structure with proper hash
model_hash = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'  # This is the actual commit hash
model_dir = os.path.expanduser(f'~/.cache/huggingface/transformers/models--openai--clip-vit-base-patch32/snapshots/{model_hash}/')
os.makedirs(model_dir, exist_ok=True)

# Try to download the model files
try:
    # Download config.json
    config_url = 'https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/config.json'
    response = requests.get(config_url, timeout=30)
    if response.status_code == 200:
        with open(os.path.join(model_dir, 'config.json'), 'w') as f:
            f.write(response.text)
        print('✅ Downloaded config.json')
    else:
        print('❌ Failed to download config.json')
        
    # Download pytorch_model.bin
    model_url = 'https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/pytorch_model.bin'
    response = requests.get(model_url, timeout=60)
    if response.status_code == 200:
        with open(os.path.join(model_dir, 'pytorch_model.bin'), 'wb') as f:
            f.write(response.content)
        print('✅ Downloaded pytorch_model.bin')
    else:
        print('❌ Failed to download pytorch_model.bin')
        
    # Download preprocessor_config.json
    preprocessor_url = 'https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/preprocessor_config.json'
    response = requests.get(preprocessor_url, timeout=30)
    if response.status_code == 200:
        with open(os.path.join(model_dir, 'preprocessor_config.json'), 'w') as f:
            f.write(response.text)
        print('✅ Downloaded preprocessor_config.json')
    else:
        print('❌ Failed to download preprocessor_config.json')
        
    # Download tokenizer files
    tokenizer_url = 'https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/tokenizer.json'
    response = requests.get(tokenizer_url, timeout=30)
    if response.status_code == 200:
        with open(os.path.join(model_dir, 'tokenizer.json'), 'w') as f:
            f.write(response.text)
        print('✅ Downloaded tokenizer.json')
    else:
        print('❌ Failed to download tokenizer.json')
        
    # Download tokenizer_config.json
    tokenizer_config_url = 'https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/tokenizer_config.json'
    response = requests.get(tokenizer_config_url, timeout=30)
    if response.status_code == 200:
        with open(os.path.join(model_dir, 'tokenizer_config.json'), 'w') as f:
            f.write(response.text)
        print('✅ Downloaded tokenizer_config.json')
    else:
        print('❌ Failed to download tokenizer_config.json')
        
    # Create a symlink to the latest snapshot
    latest_dir = os.path.expanduser('~/.cache/huggingface/transformers/models--openai--clip-vit-base-patch32/snapshots/')
    if not os.path.exists(os.path.join(latest_dir, 'latest')):
        os.symlink(model_hash, os.path.join(latest_dir, 'latest'))
        print('✅ Created latest symlink')
        
    print('✅ CLIP model files downloaded successfully')
except Exception as e:
    print(f'⚠️  Failed to download CLIP model files: {e}')
    print('Will proceed with local-only loading')
"

# Check CUDA availability and version
echo "🔍 Checking CUDA availability..."
python - << 'PYCODE'
import torch
import os
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    print(f"Current device: {torch.cuda.current_device()}")
else:
    print("❌ CUDA not available - will use CPU")
print(f"Device used: {'cuda' if torch.cuda.is_available() else 'cpu'}")
PYCODE

# Test CLIP model loading
echo "🔍 Testing CLIP model loading..."
python -c "
try:
    from transformers import CLIPModel, CLIPProcessor
    print('✅ CLIP imports successful')
    
    # Test loading from cache
    import os
    cache_dir = os.path.expanduser('~/.cache/huggingface/transformers/models--openai--clip-vit-base-patch32/snapshots/')
    if os.path.exists(cache_dir):
        snapshot_dirs = [d for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d)) and d != 'latest']
        if snapshot_dirs:
            model_path = os.path.join(cache_dir, snapshot_dirs[0])
            print(f'Found model at: {model_path}')
            model = CLIPModel.from_pretrained(model_path, local_files_only=True)
            print('✅ CLIP model loaded successfully')
        else:
            print('⚠️  No snapshot directories found')
    else:
        print('⚠️  Cache directory not found')
except Exception as e:
    print(f'❌ CLIP model test failed: {e}')
"

# Run experiments
echo "🧪 Running SANW experiments..."
source .venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python run_all_experiments.py

# Analyze results
echo "📊 Analyzing results..."
source .venv/bin/activate
python analyze_results.py

echo "✅ Experiments completed!"
echo "Results saved in runs/ directory"
echo "Summary files: results_summary.txt, results_table.tex"
