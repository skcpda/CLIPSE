#!/bin/bash
#SBATCH --job-name=test_openclip_pri
#SBATCH --output=test_openclip_pri_%j.log
#SBATCH --error=test_openclip_pri_%j.err
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:30:00

set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "=== LIMITS ==="
ulimit -a || true
cat /proc/self/limits || true
echo

echo "=== CREATE VENV UNDER /tmp ==="
VENV_DIR="/tmp/${SLURM_JOB_ID}/venv"
PY="$(command -v python3.11 || command -v python3 || command -v python)"
mkdir -p "$(dirname "$VENV_DIR")"
"$PY" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m ensurepip --upgrade || true
python -m pip install --upgrade pip

echo "=== INSTALL PYTORCH + OPENCLIP ==="
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install open_clip_torch

echo "=== CUDA INFO ==="
python - <<'PY'
import torch
print('Torch:', torch.__version__)
print('CUDA is_available:', torch.cuda.is_available())
print('Device count:', torch.cuda.device_count())
PY

echo "=== TESTING OPENCLIP PRIMARY TIER ONLY ==="
python - <<'PY'
import traceback
import open_clip
import torch

print('Loading OpenCLIP PRIMARY tier (laion400m_e32)...')
try:
    # Load only the lighter primary model
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion400m_e32'
    )
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    print('✅ PRIMARY model loaded successfully!')
    print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Model device: {next(model.parameters()).device}')
    
    # Move to GPU and test
    model = model.cuda()
    text = tokenizer(["a photo of a cat", "a photo of a dog"])
    
    with torch.no_grad():
        text_features = model.encode_text(text)
    
    print(f'Text features shape: {text_features.shape}')
    print('✅ PRIMARY tier test successful!')
    print('🎉 OpenCLIP works! Ready for SANW experiments.')
    
except Exception as e:
    print(f'❌ PRIMARY tier failed: {repr(e)}')
    traceback.print_exc()
    print('❌ Still hitting memory limits.')
PY

echo "=== DONE ==="
