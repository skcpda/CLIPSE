#!/bin/bash
# Manual transfer script - run these commands one by one

echo "📤 Manual transfer to GPU cluster"
echo "Run these commands one by one:"
echo ""

echo "1. Transfer source code:"
echo "scp -r src/ poonam@172.24.16.132:~/sanw_experiments/"
echo ""

echo "2. Transfer configs:"
echo "scp -r configs/ poonam@172.24.16.132:~/sanw_experiments/"
echo ""

echo "3. Transfer Python scripts:"
echo "scp *.py poonam@172.24.16.132:~/sanw_experiments/"
echo ""

echo "4. Transfer job script:"
echo "scp scripts/submit_cluster_job.sh poonam@172.24.16.132:~/sanw_experiments/scripts/"
echo ""

echo "5. Transfer requirements:"
echo "scp requirements.txt poonam@172.24.16.132:~/sanw_experiments/"
echo ""

echo "Then on the cluster:"
echo "ssh poonam@172.24.16.132"
echo "cd ~/sanw_experiments"
echo "sbatch scripts/submit_cluster_job.sh"
