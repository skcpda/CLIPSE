#!/usr/bin/env python3
"""
Run quick test experiments locally to verify everything works.
"""
import subprocess
import sys
import time
from pathlib import Path

def run_experiment(config_path, seed):
    """Run a single experiment."""
    print(f"🚀 Running {config_path} with seed {seed}")
    start_time = time.time()
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "src.train_advanced", 
            "--config", config_path
        ], capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        if result.returncode == 0:
            elapsed = time.time() - start_time
            print(f"✅ {config_path} completed in {elapsed:.1f}s")
            return True
        else:
            print(f"❌ {config_path} failed:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {config_path} timed out after 30 minutes")
        return False
    except Exception as e:
        print(f"💥 {config_path} error: {e}")
        return False

def main():
    """Run test experiments."""
    print("🧪 Running Quick Test Experiments (Local CPU)")
    print("=" * 50)
    
    # Test configurations
    test_configs = [
        "configs/flickr8k_vitb32_baseline_test.yaml",
        "configs/flickr8k_vitb32_debias_test.yaml", 
        "configs/flickr8k_vitb32_bandpass_test.yaml"
    ]
    
    results = {}
    total_start = time.time()
    
    for config in test_configs:
        if not Path(config).exists():
            print(f"⚠️  Config file not found: {config}")
            continue
            
        success = run_experiment(config, 13)
        results[config] = success
        
        if success:
            print(f"✅ {config} - SUCCESS")
        else:
            print(f"❌ {config} - FAILED")
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    for config, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{config}: {status}")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\n🎯 Overall: {success_count}/{total_count} experiments successful")
    print(f"⏱️  Total time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
    
    if success_count == total_count:
        print("\n🎉 All tests passed! The code is ready for full experiments.")
        print("\nTo run full experiments:")
        print("  python run_all_experiments.py")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
    
    return success_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
