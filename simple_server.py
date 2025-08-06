#!/usr/bin/env python3
"""
Simple OdinFold Server - No complications
"""

from fastapi import FastAPI
import uvicorn
import os

app = FastAPI()

@app.get("/")
def root():
    return {
        "message": "OdinFold Server Running!",
        "status": "working",
        "url": "https://5ocnemvgivdwzq-8000.proxy.runpod.net"
    }

@app.get("/health")
def health():
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else "None"
    except:
        gpu_available = False
        gpu_name = "Error"
    
    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "message": "Server is working!"
    }

if __name__ == "__main__":
    print("🚀 Starting Simple OdinFold Server...")
    print("🌐 URL: https://5ocnemvgivdwzq-8000.proxy.runpod.net")
    uvicorn.run(app, host="0.0.0.0", port=8000)
