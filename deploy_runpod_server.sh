#!/bin/bash

echo "🚀 Deploying OdinFold Server to RunPod..."
echo "=================================================="

# Your RunPod connection details
RUNPOD_HOST="5ocnemvgivdwzq-64410c7b@ssh.runpod.io"
SSH_KEY="~/.ssh/id_ed25519"

# Copy server to RunPod
echo "📤 Uploading server script..."
scp -i $SSH_KEY -o StrictHostKeyChecking=no runpod_server.py $RUNPOD_HOST:/workspace/

# Copy requirements
echo "📦 Uploading requirements..."
scp -i $SSH_KEY -o StrictHostKeyChecking=no runpod_requirements.txt $RUNPOD_HOST:/workspace/

# Install and start server
echo "🔧 Installing dependencies and starting server..."
ssh -i $SSH_KEY -o StrictHostKeyChecking=no $RUNPOD_HOST << 'EOF'
cd /workspace

# Install requirements
echo "📦 Installing Python packages..."
pip install -r runpod_requirements.txt

# Kill any existing server
echo "🛑 Stopping any existing server..."
pkill -f "python.*runpod_server.py" || true
sleep 2

# Start server in background
echo "🚀 Starting OdinFold server..."
nohup python runpod_server.py > server.log 2>&1 &

# Wait a moment for startup
sleep 5

# Check if server is running
echo "🔍 Checking server status..."
curl -s http://localhost:8000/health || echo "❌ Server not responding"

# Show server log
echo "📋 Server log (last 20 lines):"
tail -20 server.log

echo "✅ Deployment complete!"
echo "🌐 External URL: https://5ocnemvgivdwzq-8000.proxy.runpod.net"
EOF

echo "🎉 RunPod server deployment finished!"
echo "🌐 Your RunPod URL: https://5ocnemvgivdwzq-8000.proxy.runpod.net"
echo "💡 Test with: python test_runpod_connection.py"
