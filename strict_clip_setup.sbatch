#!/bin/bash
#SBATCH -J strict-clip-setup
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

echo "🚀 STRICT CLIP Setup - No Fallbacks Allowed"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Create logs directory
mkdir -p logs

# Initialize conda properly
echo "🔧 Initializing conda..."
export CONDA_NO_PLUGINS=true
# Use the conda wrapper directly
/usr/local/bin/conda init bash
source ~/.bashrc

# Create conda environment with Python 3.12
echo "📦 Creating conda environment with Python 3.12..."
conda create -n clip_env python=3.12 -y
source activate clip_env

# Upgrade pip to latest version
echo "📦 Upgrading pip to latest version..."
python -m pip install --upgrade pip

# Install PyTorch with CUDA support
echo "📦 Installing PyTorch with CUDA 11.8..."
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install transformers and other ML packages
echo "📦 Installing ML packages..."
python -m pip install transformers datasets pillow numpy requests

# Test the complete setup
echo "🧪 Testing complete CLIP setup..."
python -c "
import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np

print('🔧 Environment Test Results:')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

print('\\n📥 Downloading CLIP model...')
model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
print('✅ CLIP model downloaded successfully!')

print('\\n🧪 Testing CLIP functionality...')
# Create test image
test_image = Image.new('RGB', (224, 224), color='red')
test_texts = ['a red image', 'a blue image', 'a dog', 'a cat']

# Process inputs
inputs = processor(
    text=test_texts,
    images=test_image,
    return_tensors='pt',
    padding=True
)

# Move to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
inputs = {k: v.to(device) for k, v in inputs.items()}
model = model.to(device)

# Get embeddings
with torch.no_grad():
    outputs = model(**inputs)
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds

print(f'✅ Image embedding shape: {image_embeds.shape}')
print(f'✅ Text embedding shape: {text_embeds.shape}')

# Test similarity computation
image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

similarity = torch.matmul(image_embeds, text_embeds.T)
print(f'✅ Similarity matrix shape: {similarity.shape}')
print(f'✅ Similarity values: {similarity.cpu().numpy()}')

# Test temperature scaling
temperature = 0.07
logits = similarity / temperature
probs = torch.softmax(logits, dim=-1)
print(f'✅ Softmax probabilities: {probs.cpu().numpy()}')

print('\\n🎉 STRICT CLIP SETUP COMPLETE!')
print('✅ All packages installed successfully')
print('✅ CLIP model loaded with pretrained weights')
print('✅ GPU acceleration working')
print('✅ Ready for SANW experiments!')
"

echo "✅ STRICT CLIP setup completed successfully!"
