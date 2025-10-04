#!/usr/bin/env python3
"""
Run optimized SANW experiments with better feedback and monitoring.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

def run_single_experiment(config_path, seed, experiment_name):
    """Run a single experiment with specified seed."""
    print(f"\n{'='*70}")
    print(f"🚀 Starting {experiment_name} (seed {seed})")
    print(f"{'='*70}")

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

    start_time = time.time()

    try:
        # Pass environment variables to subprocess
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{env.get('PYTHONPATH', '')}:{os.getcwd()}"

        # Use subprocess with better error handling
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=env,
            bufsize=1
        )

        # Monitor output
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())

        exit_code = process.poll()

        end_time = time.time()
        duration = end_time - start_time

        if exit_code == 0:
            print(f"\n✅ {experiment_name} completed successfully in {duration:.1f}s")
            return True, duration
        else:
            print(f"\n❌ {experiment_name} failed with exit code {exit_code}")
            return False, duration

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n❌ {experiment_name} failed with exception: {e}")
        return False, duration

def main():
    """Run all optimized experiments."""
    print("🚀 Starting Optimized SANW Research Experiments")
    print("=" * 70)

    configs = [
        ('configs/flickr8k_vitb32_baseline.yaml', 'Baseline CLIP'),
        ('configs/flickr8k_vitb32_debias.yaml', 'SANW Debias'),
        ('configs/flickr8k_vitb32_bandpass.yaml', 'SANW Bandpass')
    ]

    seeds = [13, 17, 23]

    results = {}
    total_start_time = time.time()

    for config_path, experiment_name in configs:
        results[experiment_name] = {}

        for seed in seeds:
            success, duration = run_single_experiment(config_path, seed, f"{experiment_name} (seed {seed})")
            results[experiment_name][seed] = {'success': success, 'duration': duration}

            # Small pause between experiments
            if success:
                print(f"⏱️  Cooling down for 10 seconds...")
                time.sleep(10)

    # Print summary
    total_time = time.time() - total_start_time

    print(f"\n{'='*70}")
    print("🎉 OPTIMIZED EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(f"Total runtime: {total_time:.1f}s ({total_time/3600:.2f}h)")

    for experiment_name, seed_results in results.items():
        print(f"\n{experiment_name}:")
        successful_runs = 0
        total_duration = 0

        for seed, result in seed_results.items():
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            duration = result['duration']
            total_duration += duration
            if result['success']:
                successful_runs += 1

            print(f"  Seed {seed}: {status} ({duration:.1f}s)")

        avg_duration = total_duration / len(seeds) if seed_results else 0
        print(f"  Average duration: {avg_duration:.1f}s")
        print(f"  Success rate: {successful_runs}/{len(seeds)} ({successful_runs/len(seeds)*100:.1f}%)")

    # Check if all experiments succeeded
    all_success = all(
        all(seed_result['success'] for seed_result in seed_results.values())
        for seed_results in results.values()
    )

    if all_success:
        print("\n🎉 All experiments completed successfully!")
        print("📊 Check the runs/ directory for results and metrics.csv files")
        print("🔍 Run 'python scripts/analyze_results.py' for comprehensive analysis")
    else:
        print("\n⚠️  Some experiments failed. Check the output above for details.")

    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())
