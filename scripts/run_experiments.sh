#!/bin/bash
# Simple script to run on the cluster after transfer

echo "🔍 Checking cluster setup..."
echo "Available partitions:"
sinfo

echo ""
echo "📁 Checking files..."
ls -la
echo "Source files:"
ls src/
echo "Config files:"
ls configs/

echo ""
echo "🚀 Submitting job..."
sbatch scripts/submit_cluster_job.sh

echo ""
echo "📊 Job submitted! Monitor with:"
echo "  squeue -u poonam"
echo "  tail -f sanw_experiments_*.out"
