#!/bin/bash
# Monitor and download experiment results

echo "🔍 Monitoring SANW experiments..."

# Check job status
sshpass -p "900n@M" ssh -o StrictHostKeyChecking=no poonam@172.24.16.132 "squeue -u poonam"

echo "📊 Checking experiment progress..."
sshpass -p "900n@M" ssh -o StrictHostKeyChecking=no poonam@172.24.16.132 "cd ~/sanw_experiments && echo '=== BASELINE ===' && ls -la runs/flickr8k_baseline/ | wc -l && echo '=== DEBIAS ===' && ls -la runs/flickr8k_sanw_debias/ | wc -l && echo '=== BANDPASS ===' && ls -la runs/flickr8k_sanw_bandpass/ | wc -l"

echo "📥 Downloading completed results..."
sshpass -p "900n@M" scp -o StrictHostKeyChecking=no poonam@172.24.16.132:~/sanw_experiments/runs/flickr8k_baseline/metrics.csv ./baseline_metrics.csv 2>/dev/null || echo "Baseline not ready"
sshpass -p "900n@M" scp -o StrictHostKeyChecking=no poonam@172.24.16.132:~/sanw_experiments/runs/flickr8k_sanw_debias/metrics.csv ./debias_metrics.csv 2>/dev/null || echo "Debias not ready"
sshpass -p "900n@M" scp -o StrictHostKeyChecking=no poonam@172.24.16.132:~/sanw_experiments/runs/flickr8k_sanw_bandpass/metrics.csv ./bandpass_metrics.csv 2>/dev/null || echo "Bandpass not ready"

echo "✅ Monitoring complete"
