#!/bin/bash
#SBATCH --job-name=test_openclip_mon
#SBATCH --output=test_openclip_mon_%j.log
#SBATCH --error=test_openclip_mon_%j.err
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

# HF Hub environment for faster, reliable downloads
export HF_HOME=$PWD/.hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_TELEMETRY=1
export OPENCLIP_CACHE_PATH=$PWD/.openclip_cache
mkdir -p "$HF_HOME" "$OPENCLIP_CACHE_PATH"

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

echo "=== TESTING OPENCLIP WITH DOWNLOAD MONITORING ==="
python - <<'PY'
import traceback
import torch
import open_clip
import time
import sys
import os
import subprocess
import threading

def log_with_timestamp(msg):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()

def monitor_downloads():
    """Monitor cache directory sizes and network activity"""
    while True:
        try:
            # Check cache directory sizes
            hf_size = subprocess.check_output(['du', '-sh', '.hf_cache'], stderr=subprocess.DEVNULL).decode().split()[0] if os.path.exists('.hf_cache') else "0"
            openclip_size = subprocess.check_output(['du', '-sh', '.openclip_cache'], stderr=subprocess.DEVNULL).decode().split()[0] if os.path.exists('.openclip_cache') else "0"
            
            # Check network activity
            try:
                netstat = subprocess.check_output(['netstat', '-i'], stderr=subprocess.DEVNULL).decode()
                # Look for any network interface activity
                lines = netstat.split('\n')
                for line in lines[2:]:  # Skip header lines
                    if line.strip() and not line.startswith('Iface'):
                        parts = line.split()
                        if len(parts) >= 4:
                            rx_bytes = parts[2] if parts[2].isdigit() else "0"
                            tx_bytes = parts[6] if len(parts) > 6 and parts[6].isdigit() else "0"
                            if rx_bytes != "0" or tx_bytes != "0":
                                log_with_timestamp(f"Network activity - RX: {rx_bytes}, TX: {tx_bytes}")
                                break
            except:
                pass
            
            log_with_timestamp(f"Cache sizes - HF: {hf_size}, OpenCLIP: {openclip_size}")
            time.sleep(5)
        except:
            time.sleep(5)

log_with_timestamp("Starting OpenCLIP test with download monitoring...")

# Start monitoring thread
monitor_thread = threading.Thread(target=monitor_downloads, daemon=True)
monitor_thread.start()

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_with_timestamp(f"Using device: {device}")
    
    # Test the lighter model first
    log_with_timestamp("Loading Primary (laion400m_e32): laion400m_e32")
    start_time = time.time()
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion400m_e32", device=device
    )
    
    load_time = time.time() - start_time
    log_with_timestamp(f"✅ Primary model loaded successfully in {load_time:.2f} seconds!")
    log_with_timestamp(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    log_with_timestamp(f"Logit scale: {float(model.logit_scale.exp()):.3f}")
    
    # Quick test
    log_with_timestamp("Testing inference...")
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    text = tokenizer(["a photo of a cat", "a photo of a dog"])
    
    with torch.no_grad():
        text_features = model.encode_text(text)
    
    log_with_timestamp(f"Text features shape: {text_features.shape}")
    log_with_timestamp(f"✅ Primary test successful!")
    
    # Clean up
    del model, preprocess, tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    log_with_timestamp("🎉 Primary model loaded successfully! Ready for SANW experiments.")
    
except Exception as e:
    log_with_timestamp(f"❌ Test failed: {repr(e)}")
    log_with_timestamp("Full traceback:")
    traceback.print_exc()
    log_with_timestamp("❌ Still hitting issues.")
PY

echo "=== FINAL CACHE SIZES ==="
du -sh .hf_cache .openclip_cache 2>/dev/null || echo "No cache directories found"

echo "=== DONE ==="
