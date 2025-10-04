#!/usr/bin/env python3
"""
Comprehensive analysis of SANW experiment results.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_metrics(filepath):
    """Load metrics from CSV file."""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"⚠️  {filepath} not found")
        return None

def analyze_experiment(df, name):
    """Analyze a single experiment."""
    if df is None:
        return None
    
    # Get final results
    final = df.iloc[-1]
    
    results = {
        'experiment': name,
        'final_mean_r1': final['mean_r1'],
        'final_r1_i2t': final['r1_i2t'],
        'final_r1_t2i': final['r1_t2i'],
        'final_r5_i2t': final['r5_i2t'],
        'final_r5_t2i': final['r5_t2i'],
        'final_r10_i2t': final['r10_i2t'],
        'final_r10_t2i': final['r10_t2i'],
        'final_zs_cifar10': final['zs_cifar10'],
        'epochs_completed': len(df),
        'best_mean_r1': df['mean_r1'].max(),
        'best_zs_cifar10': df['zs_cifar10'].max()
    }
    
    return results

def create_comparison_plot(baseline_df, debias_df, bandpass_df):
    """Create comparison plots."""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Mean R@1 over epochs
    plt.subplot(2, 3, 1)
    if baseline_df is not None:
        plt.plot(baseline_df['epoch'], baseline_df['mean_r1'], label='Baseline', marker='o')
    if debias_df is not None:
        plt.plot(debias_df['epoch'], debias_df['mean_r1'], label='SANW Debias', marker='s')
    if bandpass_df is not None:
        plt.plot(bandpass_df['epoch'], bandpass_df['mean_r1'], label='SANW Bandpass', marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('Mean R@1')
    plt.title('Mean R@1 Over Training')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Zero-shot CIFAR-10 accuracy
    plt.subplot(2, 3, 2)
    if baseline_df is not None:
        plt.plot(baseline_df['epoch'], baseline_df['zs_cifar10'], label='Baseline', marker='o')
    if debias_df is not None:
        plt.plot(debias_df['epoch'], debias_df['zs_cifar10'], label='SANW Debias', marker='s')
    if bandpass_df is not None:
        plt.plot(bandpass_df['epoch'], bandpass_df['zs_cifar10'], label='SANW Bandpass', marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('Zero-shot CIFAR-10 (%)')
    plt.title('Zero-shot CIFAR-10 Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: I→T R@1
    plt.subplot(2, 3, 3)
    if baseline_df is not None:
        plt.plot(baseline_df['epoch'], baseline_df['r1_i2t'], label='Baseline', marker='o')
    if debias_df is not None:
        plt.plot(debias_df['epoch'], debias_df['r1_i2t'], label='SANW Debias', marker='s')
    if bandpass_df is not None:
        plt.plot(bandpass_df['epoch'], bandpass_df['r1_i2t'], label='SANW Bandpass', marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('I→T R@1')
    plt.title('Image→Text R@1')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: T→I R@1
    plt.subplot(2, 3, 4)
    if baseline_df is not None:
        plt.plot(baseline_df['epoch'], baseline_df['r1_t2i'], label='Baseline', marker='o')
    if debias_df is not None:
        plt.plot(debias_df['epoch'], debias_df['r1_t2i'], label='SANW Debias', marker='s')
    if bandpass_df is not None:
        plt.plot(bandpass_df['epoch'], bandpass_df['r1_t2i'], label='SANW Bandpass', marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('T→I R@1')
    plt.title('Text→Image R@1')
    plt.legend()
    plt.grid(True)
    
    # Plot 5: Loss over time
    plt.subplot(2, 3, 5)
    if baseline_df is not None:
        plt.plot(baseline_df['epoch'], baseline_df['loss'], label='Baseline', marker='o')
    if debias_df is not None:
        plt.plot(debias_df['epoch'], debias_df['loss'], label='SANW Debias', marker='s')
    if bandpass_df is not None:
        plt.plot(bandpass_df['epoch'], bandpass_df['loss'], label='SANW Bandpass', marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot 6: Learning rate schedule
    plt.subplot(2, 3, 6)
    if baseline_df is not None:
        plt.plot(baseline_df['epoch'], baseline_df['lr'], label='Baseline', marker='o')
    if debias_df is not None:
        plt.plot(debias_df['epoch'], debias_df['lr'], label='SANW Debias', marker='s')
    if bandpass_df is not None:
        plt.plot(bandpass_df['epoch'], bandpass_df['lr'], label='SANW Bandpass', marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('sanw_experiment_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main analysis function."""
    print("🔍 Analyzing SANW Experiment Results")
    print("=" * 50)
    
    # Load all experiment results
    baseline_df = load_metrics('baseline_metrics.csv')
    debias_df = load_metrics('debias_metrics.csv')
    bandpass_df = load_metrics('bandpass_metrics.csv')
    
    # Analyze each experiment
    results = []
    for df, name in [(baseline_df, 'Baseline'), (debias_df, 'SANW Debias'), (bandpass_df, 'SANW Bandpass')]:
        result = analyze_experiment(df, name)
        if result:
            results.append(result)
    
    if not results:
        print("❌ No experiment results found!")
        return
    
    # Create results table
    results_df = pd.DataFrame(results)
    print("\n📊 EXPERIMENT RESULTS SUMMARY")
    print("=" * 50)
    print(results_df.to_string(index=False, float_format='%.2f'))
    
    # Create comparison plots
    print("\n📈 Creating comparison plots...")
    create_comparison_plot(baseline_df, debias_df, bandpass_df)
    
    # Save results
    results_df.to_csv('experiment_summary.csv', index=False)
    print("\n💾 Results saved to:")
    print("  - experiment_summary.csv")
    print("  - sanw_experiment_comparison.png")
    
    # Key findings
    print("\n🎯 KEY FINDINGS")
    print("=" * 50)
    if len(results) >= 2:
        baseline_result = results[0] if results[0]['experiment'] == 'Baseline' else results[1]
        sanw_results = [r for r in results if r['experiment'] != 'Baseline']
        
        for sanw in sanw_results:
            mean_r1_diff = sanw['final_mean_r1'] - baseline_result['final_mean_r1']
            zs_diff = sanw['final_zs_cifar10'] - baseline_result['final_zs_cifar10']
            
            print(f"\n{sanw['experiment']} vs Baseline:")
            print(f"  Mean R@1 difference: {mean_r1_diff:+.2f}")
            print(f"  Zero-shot CIFAR-10 difference: {zs_diff:+.2f}%")
            
            if mean_r1_diff > 0:
                print(f"  ✅ {sanw['experiment']} outperforms baseline!")
            else:
                print(f"  ❌ {sanw['experiment']} underperforms baseline")

if __name__ == "__main__":
    main()
