#!/usr/bin/env python3
"""
Test script to verify SANW experiments work with CPU and single epoch.
"""
import sys
import os
sys.path.append(os.getcwd())

from run_all_experiments import run_experiment

def test_single_experiment():
    """Test one experiment with one epoch."""
    print("🧪 Testing SANW experiment setup...")

    # Test baseline with seed 13, 1 epoch
    try:
        print("\n" + "="*60)
        print("Testing Baseline (seed 13, 1 epoch)")
        print("="*60)

        # Temporarily modify config for 1 epoch
        config_path = 'configs/flickr8k_vitb32_baseline.yaml'
        with open(config_path, 'r') as f:
            content = f.read()

        # Update to 1 epoch for testing
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('epochs:'):
                lines[i] = 'epochs: 1'
                break

        with open(config_path, 'w') as f:
            f.write('\n'.join(lines))

        # Update seed to 13
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('seed:'):
                lines[i] = 'seed: 13'
                break

        with open(config_path, 'w') as f:
            f.write('\n'.join(lines))

        success = run_experiment(config_path, 13)

        if success:
            print("✅ Test passed! SANW experiments are working.")
            return True
        else:
            print("❌ Test failed!")
            return False

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    test_single_experiment()
