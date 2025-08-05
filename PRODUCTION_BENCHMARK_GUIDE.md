# 🚀 OdinFold Production Benchmark Guide

**Deploy to GPU servers and call from your local machine**

## 🎯 Quick Start (3 Steps)

### 1️⃣ **Deploy to GPU Servers**

On each GPU server, run:

```bash
# Make deployment script executable
chmod +x deploy.sh

# Deploy with dependency installation and GPU check
./deploy.sh --install-deps --gpu-check --port 8000
```

**Expected output:**
```
🚀 OdinFold Production Benchmark Deployment
==========================================
📋 Configuration:
   Host: 0.0.0.0
   Port: 8000
   Conda env: openfold
🔧 Activating conda environment: openfold
🔍 Checking GPU availability...
NVIDIA A100-SXM4-80GB, 81920, 1024
PyTorch CUDA available: True
GPU: NVIDIA A100-SXM4-80GB
📦 Installing dependencies...
✅ Dependencies installed
✅ All required files found
🌐 Starting OdinFold Benchmark Server...
   URL: http://0.0.0.0:8000
   Health check: http://0.0.0.0:8000/health
   API docs: http://0.0.0.0:8000/docs
🔥 Server starting... (Ctrl+C to stop)
```

### 2️⃣ **Test Server Health**

```bash
# Test from local machine
curl http://YOUR_GPU_SERVER:8000/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "timestamp": 1704067200.0,
  "gpu_available": true,
  "gpu_name": "NVIDIA A100-SXM4-80GB",
  "gpu_memory_total": 85899345920,
  "gpu_memory_allocated": 0
}
```

### 3️⃣ **Run Benchmark from Local Machine**

```bash
# Run distributed benchmark across multiple GPU servers
python benchmark_client.py \
  --servers http://gpu1:8000 http://gpu2:8000 http://gpu3:8000 \
  --sequences "MKWVTFISLLFLFSSAYS" "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG" \
  --models odinfold \
  --output production_results.json
```

## 📊 **Expected Benchmark Output**

```
🚀 Starting distributed benchmark on 3 servers
✅ Server http://gpu1:8000: NVIDIA A100-SXM4-80GB
✅ Server http://gpu2:8000: NVIDIA A100-SXM4-80GB  
✅ Server http://gpu3:8000: NVIDIA RTX 4090
📋 Starting job distributed_job_1704067200_0 on http://gpu1:8000 with 1 sequences
📋 Starting job distributed_job_1704067200_1 on http://gpu2:8000 with 1 sequences
⏳ Waiting for job distributed_job_1704067200_0 on http://gpu1:8000
📊 Job distributed_job_1704067200_0: running (50.0%)
📊 Job distributed_job_1704067200_0: completed (100.0%)
✅ Job distributed_job_1704067200_0 completed successfully!
🎯 Distributed benchmark completed!

🎯 BENCHMARK RESULTS SUMMARY
========================================
📊 Total sequences: 2
🖥️  Servers used: 3
✅ Success rate: 100.0%
⏱️  Average runtime: 1.23s
💾 Average GPU memory: 2847.5MB
🚀 Throughput: 1.63 seq/s
💾 Results saved to: production_results.json
```

## 🔧 **Advanced Configuration**

### Custom Benchmark Config

Create `benchmark_config.json`:

```json
{
  "models": {
    "odinfold": {
      "enabled": true,
      "weights_path": "openfold/resources/openfold_params/openfold_model_1_ptm.pt",
      "config_preset": "model_1_ptm"
    }
  },
  "datasets": {
    "casp14": {
      "fasta_dir": "casp14_data/fasta",
      "pdb_dir": "casp14_data/pdb",
      "enabled": true
    }
  },
  "hardware": {
    "device": "cuda",
    "mixed_precision": true,
    "max_memory_gb": 40
  }
}
```

### Multiple Models

```bash
python benchmark_client.py \
  --servers http://gpu1:8000 http://gpu2:8000 \
  --models odinfold openfold esmfold \
  --sequences "MKWVTFISLLFLFSSAYS" \
  --output multi_model_results.json
```

### CASP14 Dataset Benchmark

```bash
# Benchmark full CASP14 dataset
python benchmark_client.py \
  --servers http://gpu1:8000 http://gpu2:8000 http://gpu3:8000 \
  --sequences $(cat casp14_sequences.txt) \
  --models odinfold \
  --output casp14_results.json
```

## 🐛 **Troubleshooting**

### Server Won't Start

```bash
# Check conda environment
conda env list

# Check GPU availability
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Check port availability
netstat -tulpn | grep :8000
```

### Client Connection Issues

```bash
# Test server connectivity
curl http://YOUR_GPU_SERVER:8000/health

# Check firewall
sudo ufw status
```

### Memory Issues

```bash
# Monitor GPU memory during benchmark
watch -n 1 nvidia-smi

# Reduce batch size in config
# Edit benchmark_config.json -> hardware.max_memory_gb
```

## 📈 **Performance Optimization**

### For A100 (80GB)
- Use `mixed_precision: true`
- Set `max_memory_gb: 70`
- Enable FlashAttention if available

### For RTX 4090 (24GB)
- Use `mixed_precision: true`
- Set `max_memory_gb: 20`
- Reduce sequence length if needed

### For Multiple GPUs
- Deploy one server per GPU
- Use different ports: 8000, 8001, 8002, etc.
- Load balance sequences across servers

## 🎯 **Production Checklist**

- [ ] GPU servers accessible from local machine
- [ ] Conda environment `openfold` exists on all servers
- [ ] OpenFold model weights downloaded
- [ ] Firewall allows access to benchmark ports
- [ ] Sufficient GPU memory for target sequences
- [ ] Network bandwidth adequate for result transfer

## 📞 **API Reference**

### Health Check
```bash
GET /health
```

### Start Benchmark
```bash
POST /benchmark
{
  "sequences": ["MKWVTFISLLFLFSSAYS"],
  "models": ["odinfold"],
  "job_id": "my_job_123"
}
```

### Get Job Status
```bash
GET /benchmark/{job_id}
```

### Get Results
```bash
GET /benchmark/{job_id}/results
```

---

**🚀 You're now ready for production OdinFold benchmarking!**
