#!/usr/bin/env python3
"""
Weight visualization for SANW experiments.
Generates histograms and heatmaps of weight matrices.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from pathlib import Path

def save_weight_hist(w, path_prefix):
    """Save weight histogram and heatmap."""
    if w is None or w.size == 0:
        print(f"Warning: No weight data for {path_prefix}")
        return
    
    # Remove diagonal and flatten
    off_diag = w[~np.eye(w.shape[0], dtype=bool)]
    
    # Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(off_diag.ravel(), bins=50, alpha=0.7, edgecolor='black')
    plt.title("Weight Distribution (Off-Diagonal)")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{path_prefix}_hist.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Heatmap (64x64 crop if larger)
    if w.shape[0] > 64:
        idx = np.linspace(0, w.shape[0]-1, 64, dtype=int)
        w_crop = w[np.ix_(idx, idx)]
    else:
        w_crop = w
    
    plt.figure(figsize=(8, 8))
    plt.imshow(w_crop, aspect='auto', cmap='viridis')
    plt.colorbar(label='Weight Value')
    plt.title("Weight Matrix (64x64 crop)")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.savefig(f"{path_prefix}_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved visualizations: {path_prefix}_hist.png, {path_prefix}_heatmap.png")

def load_weight_matrices(pattern):
    """Load weight matrices from CSV files."""
    weight_data = []
    for f in glob.glob(pattern):
        try:
            df = pd.read_csv(f)
            # Get final epoch weight stats
            final_row = df.iloc[-1]
            weight_data.append({
                'file': f,
                'w_mean': final_row.get('w_mean', 0.0),
                'w_low_pct': final_row.get('w_low_pct', 0.0),
                'w_high_pct': final_row.get('w_high_pct', 0.0),
                'w_sim_corr': final_row.get('w_sim_corr', 0.0)
            })
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    return weight_data

def create_weight_summary_plot(weight_data, output_path):
    """Create summary plot of weight statistics."""
    if not weight_data:
        print("No weight data found")
        return
    
    methods = ['Baseline', 'SANW-Debias', 'SANW-Bandpass']
    metrics = ['w_mean', 'w_low_pct', 'w_high_pct', 'w_sim_corr']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = [d[metric] for d in weight_data]
        ax.bar(methods[:len(values)], values, alpha=0.7)
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved weight summary: {output_path}")

def main():
    """Main visualization function."""
    print("📊 Generating Weight Visualizations...")
    
    # Create diagnostics directories
    for method in ['baseline', 'debias', 'bandpass']:
        Path(f"runs/flickr8k_{method}/diagnostics").mkdir(parents=True, exist_ok=True)
    
    # Load weight data
    baseline_data = load_weight_matrices("runs/flickr8k_baseline/metrics.csv")
    debias_data = load_weight_matrices("runs/flickr8k_sanw_debias/metrics.csv")
    bandpass_data = load_weight_matrices("runs/flickr8k_sanw_bandpass/metrics.csv")
    
    print(f"📈 Loaded weight data:")
    print(f"  Baseline: {len(baseline_data)} runs")
    print(f"  SANW-Debias: {len(debias_data)} runs")
    print(f"  SANW-Bandpass: {len(bandpass_data)} runs")
    
    # Create summary plot
    all_data = baseline_data + debias_data + bandpass_data
    create_weight_summary_plot(all_data, "weight_summary.png")
    
    # Generate individual visualizations (if we had actual weight matrices)
    # For now, just create placeholder plots
    print("\n📊 Weight Statistics Summary:")
    for method, data in [("Baseline", baseline_data), ("SANW-Debias", debias_data), ("SANW-Bandpass", bandpass_data)]:
        if data:
            print(f"\n{method}:")
            for metric in ['w_mean', 'w_low_pct', 'w_high_pct', 'w_sim_corr']:
                values = [d[metric] for d in data]
                if values:
                    print(f"  {metric}: {np.mean(values):.3f} ± {np.std(values):.3f}")
    
    print(f"\n🎉 Weight visualization complete!")
    print(f"📁 Check weight_summary.png for overview")

if __name__ == "__main__":
    main()
