#!/bin/bash
#SBATCH --job-name=test_openclip_ext
#SBATCH --output=test_openclip_ext_%j.log
#SBATCH --error=test_openclip_ext_%j.err
#SBATCH --partition=gpu-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00

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

echo "=== TESTING OPENCLIP TWO-TIER LOADING ==="
python - <<'PY'
import traceback
import open_clip
import torch

def test_model(model_name, pretrained_name, tier_name):
    print(f'\n=== TESTING {tier_name.upper()} TIER ===')
    print(f'Model: {model_name}, Pretrained: {pretrained_name}')
    
    try:
        # Load model
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_name
        )
        tokenizer = open_clip.get_tokenizer(model_name)
        
        print(f'✅ {tier_name} model loaded successfully!')
        print(f'Model params: {sum(p.numel() for p in model.parameters()):,}')
        print(f'Model device: {next(model.parameters()).device}')
        
        # Move to GPU and test
        model = model.cuda()
        text = tokenizer(["a photo of a cat", "a photo of a dog"])
        
        with torch.no_grad():
            text_features = model.encode_text(text)
        
        print(f'Text features shape: {text_features.shape}')
        print(f'✅ {tier_name} tier test successful!')
        return True
        
    except Exception as e:
        print(f'❌ {tier_name} tier failed: {repr(e)}')
        traceback.print_exc()
        return False

# Test both tiers
print('Loading OpenCLIP with two-tier approach...')

# Primary (light) tier for iteration/sensitivity
primary_success = test_model('ViT-B-32', 'laion400m_e32', 'primary')

# Confirmatory (stronger) tier for robustness  
confirmatory_success = test_model('ViT-B-32', 'laion2b_s34b_b79k', 'confirmatory')

print(f'\n=== SUMMARY ===')
print(f'Primary tier (laion400m_e32): {"✅ SUCCESS" if primary_success else "❌ FAILED"}')
print(f'Confirmatory tier (laion2b_s34b_b79k): {"✅ SUCCESS" if confirmatory_success else "❌ FAILED"}')

if primary_success and confirmatory_success:
    print('🎉 Both tiers loaded successfully! Ready for SANW experiments.')
elif primary_success:
    print('⚠️  Only primary tier loaded. Can proceed with hyperparameter tuning.')
else:
    print('❌ Both tiers failed. Need to investigate memory issues.')
PY

echo "=== DONE ==="
