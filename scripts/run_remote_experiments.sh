#!/bin/bash
# Run experiments on remote GPU server

REMOTE_HOST="172.24.16.130"
REMOTE_USER="priyank"
REMOTE_DIR="~/clipse"

echo "🚀 Running SANW experiments on remote GPU server..."

# Connect to remote server and run experiments
ssh $REMOTE_USER@$REMOTE_HOST "cd $REMOTE_DIR && python run_all_experiments.py"

echo "✅ Experiments completed on remote server!"
echo ""
echo "To check results:"
echo "  ssh $REMOTE_USER@$REMOTE_HOST"
echo "  cd $REMOTE_DIR"
echo "  ls runs/"
echo "  python analyze_results.py"
