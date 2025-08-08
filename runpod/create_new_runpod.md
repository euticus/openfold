# 🚀 Create New RunPod Instance for OdinFold

## Step 1: Create RunPod Instance
1. Go to [RunPod.io](https://runpod.io)
2. Click "Deploy" → "GPU Pods"
3. Select **A100 80GB** (or RTX A6000 if A100 unavailable)
4. Choose **PyTorch 2.1** template
5. Set **Container Disk**: 50GB minimum
6. Click "Deploy"

## Step 2: Get Connection Details
After deployment, note:
- **Pod ID**: (e.g., `abc123def456`)
- **SSH Command**: `ssh root@ssh.runpod.io -p XXXXX`
- **Jupyter URL**: `https://[POD_ID]-8888.proxy.runpod.net`

## Step 3: Quick Setup
SSH into your pod and run:

```bash
# Install dependencies
pip install fastapi uvicorn torch transformers accelerate

# Create simple test server
cat > test_server.py << 'EOF'
from fastapi import FastAPI
import uvicorn
import torch
import os

app = FastAPI()

@app.get("/")
def root():
    return {"message": "OdinFold Server Running!", "gpu": torch.cuda.is_available()}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    }

if __name__ == "__main__":
    print(f"🚀 Starting server on port 8000")
    print(f"🌐 External URL: https://{os.environ.get('RUNPOD_POD_ID', 'unknown')}-8000.proxy.runpod.net")
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

# Start server
python test_server.py
```

## Step 4: Test Connection
Update the URL in `test_runpod_connection.py`:
```python
RUNPOD_URL = "https://[NEW_POD_ID]-8000.proxy.runpod.net"
```

Then test: `python test_runpod_connection.py`

## 🎯 Alternative: Local Testing
If RunPod issues persist, we can test locally:

```bash
# Install locally
pip install fastapi uvicorn

# Run server locally
python runpod_server.py

# Test locally
curl http://localhost:8000/health
```
