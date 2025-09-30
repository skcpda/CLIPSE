#!/bin/bash
# Setup script for remote GPU server

REMOTE_HOST="172.24.16.130"
REMOTE_USER="priyank"
REMOTE_DIR="~/clipse"

echo "🚀 Setting up SANW experiments on remote GPU server..."

# Create remote directory
echo "📁 Creating remote directory..."
ssh $REMOTE_USER@$REMOTE_HOST "mkdir -p $REMOTE_DIR"

# Transfer the code
echo "📤 Transferring code to remote server..."
rsync -avz --exclude='runs/' --exclude='data/cifar10/' --exclude='__pycache__/' --exclude='*.pyc' . $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/

# Install dependencies on remote server
echo "📦 Installing dependencies on remote server..."
ssh $REMOTE_USER@$REMOTE_HOST "cd $REMOTE_DIR && pip install -r requirements.txt"

# Check CUDA availability
echo "🔍 Checking CUDA availability on remote server..."
ssh $REMOTE_USER@$REMOTE_HOST "cd $REMOTE_DIR && python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"CUDA devices: {torch.cuda.device_count()}\"); print(f\"Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}\")'"

echo "✅ Setup complete! You can now run experiments on the remote server."
echo ""
echo "To run experiments:"
echo "  ssh $REMOTE_USER@$REMOTE_HOST"
echo "  cd $REMOTE_DIR"
echo "  python run_all_experiments.py"
