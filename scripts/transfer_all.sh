#!/bin/bash
# Complete transfer script for GPU cluster - handles everything automatically

# Configure via environment for security
# export CLUSTER_HOST=...
# export CLUSTER_USER=...
# export CLUSTER_DIR=~/sanw_experiments
CLUSTER_HOST="${CLUSTER_HOST:-}"
CLUSTER_USER="${CLUSTER_USER:-}"
CLUSTER_DIR="${CLUSTER_DIR:-~/sanw_experiments}"

if [[ -z "$CLUSTER_HOST" || -z "$CLUSTER_USER" ]]; then
  echo "❌ Please set CLUSTER_HOST and CLUSTER_USER environment variables."
  exit 1
fi

echo "🚀 Complete SANW Transfer to GPU Cluster"
echo "========================================"

echo "🔐 Using key-based SSH (recommended). If you need password auth, run: ssh-copy-id $CLUSTER_USER@$CLUSTER_HOST"

# Create remote directories
echo "📁 Creating remote directories..."
ssh "$CLUSTER_USER@$CLUSTER_HOST" "mkdir -p $CLUSTER_DIR/scripts"

# Transfer all essential files and directories
echo "📤 Transferring source code..."
scp -r src/ "$CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/"

echo "📤 Transferring configs..."
scp -r configs/ "$CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/"

echo "📤 Transferring Python scripts..."
scp *.py "$CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/"

echo "📤 Transferring job scripts..."
scp scripts/submit_cluster_job.sh "$CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/scripts/" 2>/dev/null || true

echo "📤 Transferring requirements..."
scp requirements.txt "$CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/"

echo "📤 Transferring data directory..."
scp -r data/ "$CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/"

echo "📤 Transferring documentation..."
scp *.md "$CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/" 2>/dev/null || true

echo "✅ Transfer complete!"
echo ""
echo "🔧 Next steps on the cluster:"
echo "1. ssh $CLUSTER_USER@$CLUSTER_HOST"
echo "2. cd $CLUSTER_DIR"
echo "3. ls -la  # Verify all files are there"
echo "4. sbatch run_clipse.sbatch"
echo ""
echo "📊 To monitor job:"
echo "  squeue -u $CLUSTER_USER"
echo "  tail -f sanw_experiments_*.out"
