#!/bin/bash
# Setup script for GPU cluster

CLUSTER_HOST="172.24.16.132"
CLUSTER_USER="poonam"
CLUSTER_DIR="~/sanw_experiments"

echo "🚀 Setting up SANW experiments on GPU cluster..."

# Create cluster directory
echo "📁 Creating cluster directory..."
ssh $CLUSTER_USER@$CLUSTER_HOST "mkdir -p $CLUSTER_DIR"

# Transfer the code
echo "📤 Transferring code to cluster..."
scp -r . $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/

echo "✅ Setup complete!"
echo ""
echo "To run experiments on the cluster:"
echo "  ssh $CLUSTER_USER@$CLUSTER_HOST"
echo "  cd $CLUSTER_DIR"
echo "  sbatch scripts/submit_cluster_job.sh"
echo ""
echo "To check job status:"
echo "  squeue -u $CLUSTER_USER"
echo ""
echo "To check results:"
echo "  ls runs/"
echo "  cat results_summary.txt"
