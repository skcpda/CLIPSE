#!/usr/bin/env python3
"""
Setup script for SANW experiments.
"""
import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def main():
    """Setup the SANW experiment environment."""
    print("🚀 Setting up SANW experiments...")
    
    # Check Python version
    if sys.version_info < (3, 6):
        print("❌ Python 3.6+ required")
        return 1
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        return 1
    
    # Create necessary directories
    os.makedirs("runs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    print("✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Place your Flickr8K data in data/flickr8k/")
    print("2. Run experiments with: python run_all_experiments.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
