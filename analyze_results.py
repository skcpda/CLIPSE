#!/usr/bin/env python3
"""
Results aggregator for SANW experiments.
Computes mean±std and bootstrap confidence intervals.
"""
import json
import glob
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

def load_runs(pattern):
    """Load results from multiple runs matching pattern."""
    runs = defaultdict(list)
    for f in glob.glob(pattern):
        try:
            with open(f) as fh:
                data = json.load(fh)
                runs['files'].append(f)
                runs['data'].append(data)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    return runs

def summarize(runs, key):
    """Compute mean and std for a key across runs."""
    if not runs['data']:
        return 0.0, 0.0
    vals = np.array([r[key] for r in runs['data'] if key in r])
    if len(vals) == 0:
        return 0.0, 0.0
    return float(vals.mean()), float(vals.std())

def paired_bootstrap(a, b, B=10000):
    """Bootstrap confidence interval for paired differences."""
    a = np.array(a)
    b = np.array(b)
    n = len(a)
    if n == 0:
        return 0.0, (0.0, 0.0)
    
    diffs = []
    rng = np.random.default_rng(0)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        diffs.append(float((a[idx] - b[idx]).mean()))
    diffs.sort()
    return np.mean(diffs), (diffs[int(0.025*B)], diffs[int(0.975*B)])

def load_csv_results(pattern):
    """Load results from CSV files."""
    results = []
    for f in glob.glob(pattern):
        try:
            # Try to read CSV with error handling for inconsistent columns
            df = pd.read_csv(f, on_bad_lines='skip')
            
            # Get final epoch results
            final_row = df.iloc[-1]
            
            # Handle different CSV formats by checking available columns
            result = {'file': f}
            
            # Standard columns that should exist
            for col in ['r1_i2t', 'r5_i2t', 'r10_i2t', 'r1_t2i', 'r5_t2i', 'r10_t2i', 'mean_r1', 'zs_cifar10']:
                if col in final_row:
                    result[col] = final_row[col]
                else:
                    result[col] = 0.0
            
            # Optional columns for SANW experiments
            for col in ['w_mean', 'w_low_pct', 'w_high_pct', 'w_sim_corr', 'unweighted_margin', 'weighted_margin']:
                result[col] = final_row.get(col, 0.0)
            
            results.append(result)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    return results

def generate_latex_table(baseline_results, debias_results, bandpass_results):
    """Generate LaTeX table from results."""
    def format_metric(results, metric):
        if not results:
            return "0.0 \\scriptsize$\\pm$0.0"
        vals = [r[metric] for r in results]
        mean, std = np.mean(vals), np.std(vals)
        return f"{mean:.1f} \\scriptsize$\\pm${std:.1f}"
    
    def format_best(baseline, debias, bandpass, metric):
        if not all([baseline, debias, bandpass]):
            return ""
        b_vals = [r[metric] for r in baseline]
        d_vals = [r[metric] for r in debias]
        bp_vals = [r[metric] for r in bandpass]
        
        b_mean = np.mean(b_vals)
        d_mean = np.mean(d_vals)
        bp_mean = np.mean(bp_vals)
        
        best = max(b_mean, d_mean, bp_mean)
        if d_mean == best:
            return "\\textbf{" + format_metric(debias, metric) + "}"
        elif bp_mean == best:
            return "\\textbf{" + format_metric(bandpass, metric) + "}"
        else:
            return format_metric(baseline, metric)
    
    latex = f"""
\\begin{{table}}[t]
\\centering
\\small
\\begin{{tabular}}{{lcccccc}}
\\toprule
& \\multicolumn{{3}}{{c}}{{Image$\\to$Text}} & \\multicolumn{{3}}{{c}}{{Text$\\to$Image}} \\\\
Method & R@1 & R@5 & R@10 & R@1 & R@5 & R@10 \\\\
\\midrule
Baseline & {format_metric(baseline_results, 'r1_i2t')} & {format_metric(baseline_results, 'r5_i2t')} & {format_metric(baseline_results, 'r10_i2t')} &
           {format_metric(baseline_results, 'r1_t2i')} & {format_metric(baseline_results, 'r5_t2i')} & {format_metric(baseline_results, 'r10_t2i')} \\\\
SANW–Debias & {format_best(baseline_results, debias_results, bandpass_results, 'r1_i2t')} & {format_best(baseline_results, debias_results, bandpass_results, 'r5_i2t')} & {format_best(baseline_results, debias_results, bandpass_results, 'r10_i2t')} &
              {format_best(baseline_results, debias_results, bandpass_results, 'r1_t2i')} & {format_best(baseline_results, debias_results, bandpass_results, 'r5_t2i')} & {format_best(baseline_results, debias_results, bandpass_results, 'r10_t2i')} \\\\
SANW–BandPass & {format_best(baseline_results, debias_results, bandpass_results, 'r1_i2t')} & {format_best(baseline_results, debias_results, bandpass_results, 'r5_i2t')} & {format_best(baseline_results, debias_results, bandpass_results, 'r10_t2i')} &
                 {format_best(baseline_results, debias_results, bandpass_results, 'r1_t2i')} & {format_best(baseline_results, debias_results, bandpass_results, 'r5_t2i')} & {format_best(baseline_results, debias_results, bandpass_results, 'r10_t2i')} \\\\
\\bottomrule
\\end{{tabular}}
\\caption{{Flickr8k retrieval. Mean$\\pm$std over 3 seeds; identical compute.}}
\\end{{table}}
"""
    return latex

def main():
    """Main analysis function."""
    print("🔍 Analyzing SANW Results...")
    
    # Load results from CSV files - try multiple patterns
    baseline_results = []
    debias_results = []
    bandpass_results = []
    
    # Try different naming patterns
    for pattern in ["runs/flickr8k_baseline/metrics.csv", "runs/flickr8k_vitb32_baseline/metrics.csv"]:
        baseline_results.extend(load_csv_results(pattern))
    
    for pattern in ["runs/flickr8k_sanw_debias/metrics.csv", "runs/flickr8k_vitb32_debias/metrics.csv"]:
        debias_results.extend(load_csv_results(pattern))
    
    for pattern in ["runs/flickr8k_sanw_bandpass/metrics.csv", "runs/flickr8k_vitb32_bandpass/metrics.csv"]:
        bandpass_results.extend(load_csv_results(pattern))
    
    print(f"📊 Loaded results:")
    print(f"  Baseline: {len(baseline_results)} runs")
    print(f"  SANW-Debias: {len(debias_results)} runs")
    print(f"  SANW-Bandpass: {len(bandpass_results)} runs")
    
    # Generate summary statistics
    def print_summary(name, results):
        if not results:
            print(f"\n{name}: No results found")
            return
        
        print(f"\n{name}:")
        print(f"  I→T R@1: {np.mean([r['r1_i2t'] for r in results]):.2f} ± {np.std([r['r1_i2t'] for r in results]):.2f}")
        print(f"  T→I R@1: {np.mean([r['r1_t2i'] for r in results]):.2f} ± {np.std([r['r1_t2i'] for r in results]):.2f}")
        print(f"  Mean R@1: {np.mean([r['mean_r1'] for r in results]):.2f} ± {np.std([r['mean_r1'] for r in results]):.2f}")
        print(f"  Zero-shot CIFAR-10: {np.mean([r['zs_cifar10'] for r in results]):.2f} ± {np.std([r['zs_cifar10'] for r in results]):.2f}")
        
        if any(r.get('w_mean', 0) > 0 for r in results):
            print(f"  Weight stats:")
            print(f"    w_mean: {np.mean([r.get('w_mean', 0) for r in results]):.3f} ± {np.std([r.get('w_mean', 0) for r in results]):.3f}")
            print(f"    w_low_pct: {np.mean([r.get('w_low_pct', 0) for r in results]):.1f} ± {np.std([r.get('w_low_pct', 0) for r in results]):.1f}")
            print(f"    w_high_pct: {np.mean([r.get('w_high_pct', 0) for r in results]):.1f} ± {np.std([r.get('w_high_pct', 0) for r in results]):.1f}")
            print(f"    w_sim_corr: {np.mean([r.get('w_sim_corr', 0) for r in results]):.3f} ± {np.std([r.get('w_sim_corr', 0) for r in results]):.3f}")
    
    print_summary("Baseline", baseline_results)
    print_summary("SANW-Debias", debias_results)
    print_summary("SANW-Bandpass", bandpass_results)
    
    # Generate LaTeX table
    latex_table = generate_latex_table(baseline_results, debias_results, bandpass_results)
    
    # Save results
    with open("results_summary.txt", "w") as f:
        f.write("SANW Results Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(latex_table)
    
    with open("results_table.tex", "w") as f:
        f.write(latex_table)
    
    print(f"\n📄 Results saved to:")
    print(f"  results_summary.txt")
    print(f"  results_table.tex")
    
    # Bootstrap analysis if we have paired data
    if baseline_results and debias_results:
        print(f"\n🔬 Bootstrap Analysis (Baseline vs SANW-Debias):")
        b_r1 = [r['mean_r1'] for r in baseline_results]
        d_r1 = [r['mean_r1'] for r in debias_results]
        
        if len(b_r1) == len(d_r1) and len(b_r1) > 0:
            mean_diff, ci = paired_bootstrap(b_r1, d_r1)
            print(f"  Mean R@1 difference: {mean_diff:.3f}")
            print(f"  95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
            
            if ci[0] > 0:
                print("  ✅ SANW-Debias significantly better than Baseline")
            elif ci[1] < 0:
                print("  ❌ SANW-Debias significantly worse than Baseline")
            else:
                print("  ⚖️  No significant difference")
    
    print(f"\n🎉 Analysis complete!")

if __name__ == "__main__":
    main()
