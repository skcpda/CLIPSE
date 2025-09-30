#!/bin/bash
# Check cluster partitions and submit job

echo "🔍 Checking available partitions..."
sinfo

echo ""
echo "📋 Available partitions:"
sinfo -o "%P %A %T %N"

echo ""
echo "🚀 Submitting job without partition specification..."
sbatch scripts/submit_cluster_job.sh

echo ""
echo "📊 Check job status:"
echo "  squeue -u poonam"
echo "  squeue -j <JOB_ID>"
