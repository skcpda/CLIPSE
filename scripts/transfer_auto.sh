#!/bin/bash
# Automated transfer with password - no manual input needed

CLUSTER_HOST="172.24.16.132"
CLUSTER_USER="poonam"
CLUSTER_PASS="900n@M"
CLUSTER_DIR="~/sanw_experiments"

echo "🚀 Automated SANW Transfer to GPU Cluster"
echo "========================================="

# Install sshpass if not available
if ! command -v sshpass &> /dev/null; then
    echo "📦 Installing sshpass..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if ! command -v brew &> /dev/null; then
            echo "❌ Homebrew not found. Please install sshpass manually:"
            echo "brew install hudochenkov/sshpass/sshpass"
            exit 1
        fi
        brew install hudochenkov/sshpass/sshpass
    else
        # Linux
        sudo apt-get update && sudo apt-get install -y sshpass
    fi
fi

# Create remote directories
echo "📁 Creating remote directories..."
sshpass -p "$CLUSTER_PASS" ssh -o StrictHostKeyChecking=no $CLUSTER_USER@$CLUSTER_HOST "mkdir -p $CLUSTER_DIR/scripts"

# Transfer all essential files and directories
echo "📤 Transferring source code..."
sshpass -p "$CLUSTER_PASS" scp -r -o StrictHostKeyChecking=no src/ $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/

echo "📤 Transferring configs..."
sshpass -p "$CLUSTER_PASS" scp -r -o StrictHostKeyChecking=no configs/ $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/

echo "📤 Transferring Python scripts..."
sshpass -p "$CLUSTER_PASS" scp -o StrictHostKeyChecking=no *.py $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/

echo "📤 Transferring job scripts..."
sshpass -p "$CLUSTER_PASS" scp -o StrictHostKeyChecking=no scripts/submit_cluster_job.sh $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/scripts/

echo "📤 Transferring requirements..."
sshpass -p "$CLUSTER_PASS" scp -o StrictHostKeyChecking=no requirements.txt $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/

echo "📤 Transferring data directory..."
sshpass -p "$CLUSTER_PASS" scp -r -o StrictHostKeyChecking=no data/ $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/

echo "📤 Transferring run script..."
sshpass -p "$CLUSTER_PASS" scp -o StrictHostKeyChecking=no scripts/run_experiments.sh $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/scripts/

echo "📤 Transferring documentation..."
sshpass -p "$CLUSTER_PASS" scp -o StrictHostKeyChecking=no *.md $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/ 2>/dev/null || true

echo "✅ Transfer complete!"
echo ""
echo "🔧 Next steps on the cluster:"
echo "1. ssh $CLUSTER_USER@$CLUSTER_HOST"
echo "2. cd $CLUSTER_DIR"
echo "3. chmod +x scripts/run_experiments.sh"
echo "4. ./scripts/run_experiments.sh"
echo ""
echo "📊 To monitor job:"
echo "  squeue -u $CLUSTER_USER"
echo "  tail -f sanw_experiments_*.out"
