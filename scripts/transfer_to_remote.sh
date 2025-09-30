#!/bin/bash
# Transfer code to remote GPU server

echo "📤 Transferring code to remote GPU server..."
echo "You'll need to enter the password: priyank@123#"
echo ""

# Transfer the tar file
scp clipse_code.tar.gz priyank@172.24.16.130:~/clipse/

echo "✅ Code transferred successfully!"
echo ""
echo "Now connect to the remote server and run:"
echo "  ssh priyank@172.24.16.130"
echo "  cd ~/clipse"
echo "  tar -xzf clipse_code.tar.gz"
echo "  rm clipse_code.tar.gz"
echo "  python run_all_experiments.py"
