#!/usr/bin/env python3
"""
Run all SANW experiments with multiple seeds for statistical significance.
"""
import os
import subprocess
import sys
from pathlib import Path

def run_experiment(config_path, seed):
    """Run a single experiment with specified seed."""
    print(f"\n{'='*60}")
    print(f"Running experiment: {config_path} with seed {seed}")
    print(f"{'='*60}")
    
    # Update seed in config
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Replace seed line
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('seed:'):
            lines[i] = f'seed: {seed}'
            break
    
    # Write updated config
    updated_content = '\n'.join(lines)
    with open(config_path, 'w') as f:
        f.write(updated_content)
    
    # Run experiment
    cmd = [
        sys.executable, '-m', 'src.train_advanced',
        '--config', config_path
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ Experiment completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment failed with exit code {e.returncode}")
        return False

def main():
    """Run all experiments with all seeds."""
    seeds = [13, 17, 23]
    configs = [
        'configs/flickr8k_vitb32_baseline.yaml',
        'configs/flickr8k_vitb32_debias.yaml', 
        'configs/flickr8k_vitb32_bandpass.yaml'
    ]
    
    print("🚀 Starting SANW Research Experiments")
    print(f"Seeds: {seeds}")
    print(f"Configs: {configs}")
    
    results = {}
    
    for config in configs:
        config_name = Path(config).stem
        results[config_name] = {}
        
        for seed in seeds:
            success = run_experiment(config, seed)
            results[config_name][seed] = success
    
    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    for config_name, seed_results in results.items():
        print(f"\n{config_name}:")
        for seed, success in seed_results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"  Seed {seed}: {status}")
    
    # Check if all experiments succeeded
    all_success = all(
        all(seed_results.values()) 
        for seed_results in results.values()
    )
    
    if all_success:
        print(f"\n🎉 All experiments completed successfully!")
        print("📊 Check the runs/ directory for results and metrics.csv files")
    else:
        print(f"\n⚠️  Some experiments failed. Check the output above for details.")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())
