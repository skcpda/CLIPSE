#!/bin/bash
# Fix job submission issues and resubmit

echo "🔧 Fixing job submission issues..."

# Cancel current job if it exists
echo "❌ Canceling current job..."
scancel -u poonam

# Check available partitions
echo "🔍 Checking available partitions..."
sinfo

echo ""
echo "📋 Partition details:"
sinfo -o "%P %l %A %T %N"

echo ""
echo "🚀 Submitting updated job with shorter time limits..."
sbatch scripts/submit_cluster_job.sh

echo ""
echo "📊 Check job status:"
echo "  squeue -u poonam"
echo "  tail -f sanw_experiments_*.out"
