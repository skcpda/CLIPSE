#!/usr/bin/expect -f
# Automated transfer using expect - no manual input needed

set CLUSTER_HOST "172.24.16.132"
set CLUSTER_USER "poonam"
set CLUSTER_PASS "900n@M"
set CLUSTER_DIR "~/sanw_experiments"

puts "🚀 Automated SANW Transfer to GPU Cluster"
puts "========================================="

# Create remote directories
puts "📁 Creating remote directories..."
spawn ssh $CLUSTER_USER@$CLUSTER_HOST "mkdir -p $CLUSTER_DIR/scripts"
expect "password:"
send "$CLUSTER_PASS\r"
expect eof

# Transfer source code
puts "📤 Transferring source code..."
spawn scp -r src/ $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/
expect "password:"
send "$CLUSTER_PASS\r"
expect eof

# Transfer configs
puts "📤 Transferring configs..."
spawn scp -r configs/ $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/
expect "password:"
send "$CLUSTER_PASS\r"
expect eof

# Transfer Python scripts
puts "📤 Transferring Python scripts..."
spawn scp *.py $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/
expect "password:"
send "$CLUSTER_PASS\r"
expect eof

# Transfer job script
puts "📤 Transferring job script..."
spawn scp scripts/submit_cluster_job.sh $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/scripts/
expect "password:"
send "$CLUSTER_PASS\r"
expect eof

# Transfer requirements
puts "📤 Transferring requirements..."
spawn scp requirements.txt $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/
expect "password:"
send "$CLUSTER_PASS\r"
expect eof

# Transfer data
puts "📤 Transferring data..."
spawn scp -r data/ $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/
expect "password:"
send "$CLUSTER_PASS\r"
expect eof

# Transfer run script
puts "📤 Transferring run script..."
spawn scp scripts/run_experiments.sh $CLUSTER_USER@$CLUSTER_HOST:$CLUSTER_DIR/scripts/
expect "password:"
send "$CLUSTER_PASS\r"
expect eof

puts "✅ Transfer complete!"
puts ""
puts "🔧 Next steps on the cluster:"
puts "1. ssh $CLUSTER_USER@$CLUSTER_HOST"
puts "2. cd $CLUSTER_DIR"
puts "3. chmod +x scripts/run_experiments.sh"
puts "4. ./scripts/run_experiments.sh"
