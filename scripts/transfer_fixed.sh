#!/bin/bash
# Fixed transfer script for GPU cluster

echo "📤 Transferring SANW experiments to GPU cluster..."

# Create tar archive
echo "📦 Creating archive..."
tar -czf sanw_experiments.tgz --exclude='runs' --exclude='__pycache__' --exclude='*.pyc' --exclude='sanw_experiments.tgz' .

# Create remote directory
echo "📁 Creating remote directory..."
ssh poonam@172.24.16.132 'mkdir -p ~/sanw_experiments'

# Transfer to cluster home
echo "🚀 Transferring to cluster..."
scp sanw_experiments.tgz poonam@172.24.16.132:~/

echo "✅ Transfer complete!"
echo ""
echo "Next steps on the cluster:"
echo "1. ssh poonam@172.24.16.132"
echo "2. tar -xzf ~/sanw_experiments.tgz -C ~/sanw_experiments"
echo "3. rm ~/sanw_experiments.tgz"
echo "4. cd ~/sanw_experiments"
echo "5. chmod +x check_and_submit.sh"
echo "6. ./check_and_submit.sh"
