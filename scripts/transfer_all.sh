#!/bin/bash
# Complete transfer script for GPU cluster - handles everything automatically

CLUSTER_HOST="172.24.16.132"
CLUSTER_USER="poonam"
CLUSTER_PASSWORD="900n@M"
CLUSTER_DIR="~/sanw_experiments"

echo "🚀 Complete SANW Transfer to GPU Cluster"
echo "========================================"

# Check if sshpass is available, if not install it
if ! command -v sshpass &> /dev/null; then
    echo "📦 Installing sshpass for password automation..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install hudochenkov/sshpass/sshpass
        else
            echo "❌ Please install Homebrew first: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            echo "Then run: brew install hudochenkov/sshpass/sshpass"
            exit 1
        fi
    else
        # Linux
        sudo apt-get update && sudo apt-get install -y sshpass
    fi
fi

# Create remote directories
echo "📁 Creating remote directories..."
sshpass -p "$CLUSTER_PASSWORD" ssh $CLUSTER_USER@$CLUSTER_HOST "mkdir -p $CLUSTER_DIR/scripts"

# Transfer all essential files and directories
echo "📤 Transferring source code..."
sshpass -p "$CLUSTER_PASSWORD" scp -r src/ $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/

echo "📤 Transferring configs..."
sshpass -p "$CLUSTER_PASSWORD" scp -r configs/ $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/

echo "📤 Transferring Python scripts..."
sshpass -p "$CLUSTER_PASSWORD" scp *.py $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/

echo "📤 Transferring job scripts..."
sshpass -p "$CLUSTER_PASSWORD" scp scripts/submit_cluster_job.sh $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/scripts/

echo "📤 Transferring requirements..."
sshpass -p "$CLUSTER_PASSWORD" scp requirements.txt $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/

echo "📤 Transferring data directory..."
sshpass -p "$CLUSTER_PASSWORD" scp -r data/ $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/

echo "📤 Transferring documentation..."
sshpass -p "$CLUSTER_PASSWORD" scp *.md $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/ 2>/dev/null || true

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
