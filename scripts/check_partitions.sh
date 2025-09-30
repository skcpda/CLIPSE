#!/bin/bash
# Check available partitions and their time limits

echo "🔍 Checking available partitions..."
sinfo

echo ""
echo "📋 Detailed partition info:"
sinfo -o "%P %l %A %T %N"

echo ""
echo "⏰ Time limits for each partition:"
scontrol show partition
