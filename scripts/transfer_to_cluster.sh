#!/bin/bash
# Quick transfer script for GPU cluster

echo "📤 Transferring SANW experiments to GPU cluster..."
echo "Cluster: poonam@172.24.16.132"
echo "Password: 900n@M"
echo ""

# Create tar archive for faster transfer
echo "📦 Creating archive..."
tar -czf sanw_experiments.tar.gz --exclude='runs' --exclude='__pycache__' --exclude='*.pyc' --exclude='sanw_experiments.tar.gz' .

# Transfer to cluster
echo "🚀 Transferring to cluster..."
scp sanw_experiments.tar.gz poonam@172.24.16.132:~/sanw_experiments/

echo "✅ Transfer complete!"
echo ""
echo "Next steps:"
echo "1. ssh poonam@172.24.16.132"
echo "2. cd ~/sanw_experiments"
echo "3. tar -xzf sanw_experiments.tar.gz"
echo "4. rm sanw_experiments.tar.gz"
echo "5. sbatch scripts/submit_cluster_job.sh"
echo ""
echo "To monitor job:"
echo "  squeue -u poonam"
echo "  tail -f sanw_experiments_*.out"
