#!/usr/bin/env python3
"""
Download OpenCLIP models locally and prepare for transfer to cluster.
This bypasses the network throttling issue on cluster workers.
"""

import os
import sys
import subprocess
from pathlib import Path

def download_model(repo_id, local_dir):
    """Download a model from HuggingFace Hub to local directory."""
    print(f"Downloading {repo_id} to {local_dir}...")
    
    # Create local directory
    os.makedirs(local_dir, exist_ok=True)
    
    # Download using huggingface_hub
    try:
        from huggingface_hub import snapshot_download
        
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False  # Copy files, don't symlink
        )
        print(f"✅ Downloaded {repo_id} to {local_dir}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {repo_id}: {e}")
        return False

def get_directory_size(path):
    """Get directory size in MB."""
    try:
        result = subprocess.run(['du', '-sm', path], capture_output=True, text=True)
        if result.returncode == 0:
            size_mb = int(result.stdout.split()[0])
            return size_mb
    except:
        pass
    return 0

def main():
    print("=== OpenCLIP Model Downloader ===")
    print("This will download models locally for transfer to cluster.")
    print()
    
    # Models to download
    models = [
        {
            "repo_id": "laion/CLIP-ViT-B-32-laion400M-e32",
            "local_dir": "./models/laion400m_e32",
            "name": "Primary (laion400m_e32)"
        },
        {
            "repo_id": "laion/CLIP-ViT-B-32-laion2B-s34B-b79K", 
            "local_dir": "./models/laion2b_s34b_b79k",
            "name": "Confirmatory (laion2b_s34b_b79k)"
        }
    ]
    
    # Check if huggingface_hub is available
    try:
        import huggingface_hub
        print(f"✅ HuggingFace Hub version: {huggingface_hub.__version__}")
    except ImportError:
        print("❌ huggingface_hub not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        import huggingface_hub
        print(f"✅ Installed HuggingFace Hub version: {huggingface_hub.__version__}")
    
    # Download each model
    total_size = 0
    for model in models:
        print(f"\n--- {model['name']} ---")
        success = download_model(model["repo_id"], model["local_dir"])
        
        if success:
            size_mb = get_directory_size(model["local_dir"])
            total_size += size_mb
            print(f"📁 Size: {size_mb} MB")
        else:
            print(f"❌ Failed to download {model['name']}")
    
    print(f"\n=== Summary ===")
    print(f"Total size: {total_size} MB")
    print(f"Models downloaded to: ./models/")
    print()
    print("Next steps:")
    print("1. Transfer models to cluster: scp -r ./models/ user@cluster:/path/to/models/")
    print("2. Update OpenCLIP loading to use local paths")
    print("3. Test offline loading on cluster")

if __name__ == "__main__":
    main()
