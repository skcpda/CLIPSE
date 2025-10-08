#!/usr/bin/env python3
"""
Direct experiment runner for SANW experiments.
"""
import os
import sys
import argparse
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def run_experiment(config_path, experiment_name):
    """Run a single experiment."""
    print(f"🚀 Running {experiment_name} experiment...")
    print(f"Config: {config_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    print(f"Experiment: {cfg['experiment_name']}")
    print(f"Loss: {cfg['loss']['name']}")
    print(f"Batch size: {cfg['batch_size']}")
    print(f"Epochs: {cfg['epochs']}")
    
    # Import and run training
    from train_clipse import train
    
    try:
        train(cfg)
        print(f"✅ {experiment_name} experiment completed successfully!")
        return True
    except Exception as e:
        print(f"❌ {experiment_name} experiment failed: {e}")
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run SANW experiments")
    parser.add_argument("--experiment", type=str, required=True, 
                       choices=["baseline", "debias", "bandpass"],
                       help="Experiment to run")
    args = parser.parse_args()
    
    # Map experiment names to config files
    config_map = {
        "baseline": "configs/cluster_baseline.yaml",
        "debias": "configs/cluster_debias.yaml", 
        "bandpass": "configs/cluster_bandpass.yaml"
    }
    
    config_path = config_map[args.experiment]
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    return run_experiment(config_path, args.experiment)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
