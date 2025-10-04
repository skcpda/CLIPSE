#!/bin/bash
# Final analysis when all experiments are complete

echo "🔍 Final SANW Experiment Analysis"
echo "================================="

# Check if all experiments are complete
echo "📊 Checking experiment status..."
sshpass -p "900n@M" ssh -o StrictHostKeyChecking=no poonam@172.24.16.132 "cd ~/sanw_experiments && echo 'Baseline:' && ls -la runs/flickr8k_baseline/ | wc -l && echo 'Debias:' && ls -la runs/flickr8k_sanw_debias/ | wc -l && echo 'Bandpass:' && ls -la runs/flickr8k_sanw_bandpass/ | wc -l"

# Download all results
echo "📥 Downloading all experiment results..."
sshpass -p "900n@M" scp -o StrictHostKeyChecking=no poonam@172.24.16.132:~/sanw_experiments/runs/flickr8k_baseline/metrics.csv ./baseline_metrics.csv
sshpass -p "900n@M" scp -o StrictHostKeyChecking=no poonam@172.24.16.132:~/sanw_experiments/runs/flickr8k_sanw_debias/metrics.csv ./debias_metrics.csv
sshpass -p "900n@M" scp -o StrictHostKeyChecking=no poonam@172.24.16.132:~/sanw_experiments/runs/flickr8k_sanw_bandpass/metrics.csv ./bandpass_metrics.csv

# Download model files
echo "📥 Downloading best models..."
sshpass -p "900n@M" scp -o StrictHostKeyChecking=no poonam@172.24.16.132:~/sanw_experiments/runs/flickr8k_baseline/best_model.pt ./baseline_best_model.pt
sshpass -p "900n@M" scp -o StrictHostKeyChecking=no poonam@172.24.16.132:~/sanw_experiments/runs/flickr8k_sanw_debias/best_model.pt ./debias_best_model.pt
sshpass -p "900n@M" scp -o StrictHostKeyChecking=no poonam@172.24.16.132:~/sanw_experiments/runs/flickr8k_sanw_bandpass/best_model.pt ./bandpass_best_model.pt

# Run analysis
echo "🧪 Running comprehensive analysis..."
python scripts/analyze_results.py

echo "✅ Analysis complete! Check the generated files:"
echo "  - experiment_summary.csv"
echo "  - sanw_experiment_comparison.png"
echo "  - baseline_metrics.csv"
echo "  - debias_metrics.csv" 
echo "  - bandpass_metrics.csv"
