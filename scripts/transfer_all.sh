#!/bin/bash
# Complete transfer script for GPU cluster - handles everything automatically

CLUSTER_HOST="172.24.16.132"
CLUSTER_USER="poonam"
CLUSTER_DIR="~/sanw_experiments"

echo "🚀 Complete SANW Transfer to GPU Cluster"
echo "========================================"

# Create remote directories
echo "📁 Creating remote directories..."
ssh $CLUSTER_USER@$CLUSTER_HOST "mkdir -p $CLUSTER_DIR/scripts"

# Transfer all essential files and directories
echo "📤 Transferring source code..."
scp -r src/ $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/

echo "📤 Transferring configs..."
scp -r configs/ $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/

echo "📤 Transferring Python scripts..."
scp *.py $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/

echo "📤 Transferring job scripts..."
scp scripts/submit_cluster_job.sh $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/scripts/

echo "📤 Transferring requirements..."
scp requirements.txt $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/

echo "📤 Transferring data directory..."
scp -r data/ $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/

echo "📤 Transferring documentation..."
scp *.md $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/ 2>/dev/null || true

echo "✅ Transfer complete!"
echo ""
echo "🔧 Next steps on the cluster:"
echo "1. ssh $CLUSTER_USER@$CLUSTER_HOST"
echo "2. cd $CLUSTER_DIR"
echo "3. ls -la  # Verify all files are there"
echo "4. sbatch scripts/submit_cluster_job.sh"
echo ""
echo "📊 To monitor job:"
echo "  squeue -u $CLUSTER_USER"
echo "  tail -f sanw_experiments_*.out"
