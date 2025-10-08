#!/usr/bin/env python3
"""
Simple setup script to install and test CLIP using system Python.
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run a command and return success status."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(f"Command: {cmd}")
        print(f"Return code: {result.returncode}")
        if result.stdout:
            print(f"STDOUT: {result.stdout}")
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def main():
    print("🔧 Simple CLIP setup using system Python")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Try to install packages using different methods
    print("\n📦 Attempting to install packages...")
    
    # Method 1: Try using pip3
    if run_command("pip3 install --user torch torchvision transformers datasets pillow numpy"):
        print("✅ Packages installed with pip3")
    else:
        print("❌ pip3 failed")
    
    # Method 2: Try using python -m pip
    if run_command(f"{sys.executable} -m pip install --user torch torchvision transformers datasets pillow numpy"):
        print("✅ Packages installed with python -m pip")
    else:
        print("❌ python -m pip failed")
    
    # Method 3: Try using easy_install
    if run_command("easy_install --user torch torchvision transformers datasets pillow numpy"):
        print("✅ Packages installed with easy_install")
    else:
        print("❌ easy_install failed")
    
    # Test if packages are available
    print("\n🧪 Testing package availability...")
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch not available")
    
    try:
        from transformers import CLIPModel, CLIPProcessor
        print("✅ Transformers available")
    except ImportError:
        print("❌ Transformers not available")
    
    try:
        from PIL import Image
        print("✅ PIL available")
    except ImportError:
        print("❌ PIL not available")
    
    try:
        import numpy as np
        print("✅ NumPy available")
    except ImportError:
        print("❌ NumPy not available")

if __name__ == "__main__":
    main()
