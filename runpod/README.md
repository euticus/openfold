# RunPod Scripts and Configuration

This folder contains all RunPod-related scripts, configuration files, and documentation for deploying and running benchmarks on RunPod infrastructure.

## 📁 Files Overview

### Documentation
- `RUNPOD_INSTRUCTIONS.md` - General RunPod setup and usage instructions
- `RUNPOD_SETUP_GUIDE.md` - Detailed setup guide for RunPod deployment
- `create_new_runpod.md` - Instructions for creating new RunPod instances

### Setup Scripts
- `quick_runpod_setup.sh` - Quick setup script for RunPod environment
- `runpod_startup.sh` - Startup script that runs when RunPod instance boots
- `runpod_setup.py` - Python setup script for RunPod configuration
- `setup_runpod_benchmark.sh` - Setup script specifically for benchmark environment

### Deployment Scripts
- `deploy_runpod_server.sh` - Deploy server to RunPod
- `runpod_deploy.sh` - Main deployment script
- `runpod_deployment_package.zip` - Pre-packaged deployment bundle

### Server and Client Scripts
- `runpod_server.py` - Main RunPod server implementation
- `runpod_benchmark_client.py` - Client for connecting to RunPod benchmark server
- `test_runpod_connection.py` - Test script for verifying RunPod connectivity

### Benchmark Scripts
- `execute_runpod_benchmark.py` - Execute benchmarks on RunPod
- `runpod_execute.sh` - Shell script for running benchmarks
- `runpod_production_benchmark.py` - Production benchmark runner for RunPod

### Configuration and Results
- `runpod_requirements.txt` - Python dependencies for RunPod environment
- `comprehensive_runpod_results.json` - Sample/previous benchmark results

## 🚀 Quick Start

1. **Setup RunPod Instance:**
   ```bash
   ./quick_runpod_setup.sh
   ```

2. **Deploy to RunPod:**
   ```bash
   ./runpod_deploy.sh
   ```

3. **Run Benchmarks:**
   ```bash
   python execute_runpod_benchmark.py
   ```

## 📖 Documentation

For detailed instructions, see:
- `RUNPOD_INSTRUCTIONS.md` for general usage
- `RUNPOD_SETUP_GUIDE.md` for detailed setup
- `create_new_runpod.md` for creating new instances

## 🔧 Requirements

- RunPod account and API key
- CUDA-compatible GPU instance
- Python 3.8+ with dependencies from `runpod_requirements.txt`
