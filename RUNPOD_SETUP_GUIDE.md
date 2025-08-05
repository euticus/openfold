# 🚀 RunPod OdinFold Comprehensive Benchmark Guide

**Deploy ALL model weights and CASP datasets to RunPod GPUs**

## 🎯 Quick Start (Copy-Paste Commands)

### 1️⃣ **Deploy to RunPod Instances**

**In each RunPod terminal, run:**

```bash
# Navigate to workspace
cd /workspace

# Clone your repository
git clone https://github.com/euticus/openfold.git
cd openfold

# Copy the deployment files (you'll need to upload these)
# Or create them directly in RunPod

# Make deployment script executable and run
chmod +x runpod_deploy.sh
./runpod_deploy.sh --port 8000
```

### 2️⃣ **Get RunPod URLs**

From your RunPod dashboard, get the **proxy URLs** for each pod:
```
https://abc123-8000.proxy.runpod.net
https://def456-8000.proxy.runpod.net  
https://ghi789-8000.proxy.runpod.net
```

### 3️⃣ **Run Comprehensive Benchmark from Local Machine**

```bash
# Test ALL model weights on CASP datasets
python runpod_benchmark_client.py \
  --runpods https://abc123-8000.proxy.runpod.net https://def456-8000.proxy.runpod.net https://ghi789-8000.proxy.runpod.net \
  --include-casp \
  --output comprehensive_runpod_results.json
```

## 🧬 **What Gets Benchmarked**

### **ALL Available Model Weights:**
- ✅ `openfold_model_1_ptm.pt` - Main PTM model
- ✅ `finetuning_ptm_1.pt` - Finetuned PTM v1  
- ✅ `finetuning_ptm_2.pt` - Finetuned PTM v2
- ✅ `finetuning_no_templ_ptm_1.pt` - No template PTM
- ✅ `finetuning_2.pt` - Finetuning v2
- ✅ `finetuning_3.pt` - Finetuning v3
- ✅ `finetuning_4.pt` - Finetuning v4
- ✅ `finetuning_5.pt` - Finetuning v5
- ✅ `initial_training.pt` - Initial training
- ✅ `openfold_trained_weights.pt` - Trained weights

### **Real CASP Datasets:**
- ✅ **CASP14 targets**: T1024, T1027, T1030, T1040, T1050, etc.
- ✅ **Demo dataset**: Real CASP sequences from `demo_dataset/fasta/`
- ✅ **Reference structures**: PDB files for validation

## 📊 **Expected Comprehensive Results**

```
🎯 COMPREHENSIVE RUNPOD BENCHMARK RESULTS
==================================================
🧬 Total models tested: 10
🧪 Total sequences: 4
🖥️  RunPod instances: 3
🔬 Total experiments: 120
✅ Success rate: 95.8%
⏱️  Total runtime: 847.3s
🚀 Throughput: 0.14 seq/s
💾 Results saved to: comprehensive_runpod_results.json

📊 MODEL PERFORMANCE SUMMARY:
   openfold_model_1_ptm: 2.13s avg, 100.0% success
   finetuning_ptm_1: 2.05s avg, 100.0% success
   finetuning_ptm_2: 2.18s avg, 95.0% success
   finetuning_no_templ_ptm_1: 1.89s avg, 100.0% success
   finetuning_2: 2.34s avg, 90.0% success
   finetuning_3: 2.41s avg, 85.0% success
   finetuning_4: 2.28s avg, 95.0% success
   finetuning_5: 2.52s avg, 80.0% success
   initial_training: 3.15s avg, 75.0% success
   openfold_trained_weights: 2.67s avg, 90.0% success
```

## 🔧 **RunPod-Specific Setup**

### **Recommended RunPod Configuration:**
- **GPU**: RTX 4090 (24GB) or A100 (40GB/80GB)
- **Template**: PyTorch 2.0+ with CUDA 11.8+
- **Storage**: 50GB+ for model weights
- **Network**: Expose port 8000

### **Upload Required Files to RunPod:**

Create these files in your RunPod `/workspace/openfold/` directory:

1. **`runpod_deploy.sh`** - Deployment script
2. **`production_benchmark_setup.py`** - Benchmark runner  
3. **`deploy_benchmark_server.py`** - API server
4. **`runpod_benchmark_client.py`** - Local client (for your machine)

### **Alternative: Direct File Creation**

If you can't upload files, create them directly in RunPod:

```bash
# In RunPod terminal
cd /workspace
git clone https://github.com/euticus/openfold.git
cd openfold

# Create the deployment script
cat > runpod_deploy.sh << 'EOF'
[Copy the entire runpod_deploy.sh content here]
EOF

chmod +x runpod_deploy.sh
./runpod_deploy.sh
```

## 🐛 **RunPod Troubleshooting**

### **Common Issues:**

**1. Port not accessible:**
```bash
# Check if port is open in RunPod
netstat -tulpn | grep :8000

# Make sure you're using the correct proxy URL from RunPod dashboard
```

**2. Model weights missing:**
```bash
# Check weights directory
ls -la openfold/resources/openfold_params/

# Download manually if needed
wget -O openfold/resources/openfold_params/openfold_model_1_ptm.pt \
  https://files.ipd.uw.edu/pub/openfold/openfold_model_1_ptm.pt
```

**3. GPU memory issues:**
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Reduce batch size in config if needed
```

**4. Connection timeout:**
```bash
# Increase timeout in client
python runpod_benchmark_client.py --runpods ... --timeout 3600
```

## 🚀 **Performance Optimization for RunPod**

### **For RTX 4090 (24GB):**
```json
{
  "hardware": {
    "device": "cuda",
    "mixed_precision": true,
    "max_memory_gb": 20
  }
}
```

### **For A100 (80GB):**
```json
{
  "hardware": {
    "device": "cuda", 
    "mixed_precision": true,
    "max_memory_gb": 70
  }
}
```

## 📈 **Scaling Across Multiple RunPods**

### **Load Balancing Strategy:**
- **Pod 1**: Models 1-3 (openfold_model_1_ptm, finetuning_ptm_1, finetuning_ptm_2)
- **Pod 2**: Models 4-6 (finetuning_no_templ_ptm_1, finetuning_2, finetuning_3)  
- **Pod 3**: Models 7-10 (finetuning_4, finetuning_5, initial_training, openfold_trained_weights)

### **Parallel Execution:**
Each RunPod processes different models simultaneously, maximizing GPU utilization.

## 🎯 **Production Checklist**

- [ ] RunPod instances launched with GPU access
- [ ] Port 8000 exposed and accessible via proxy URL
- [ ] All benchmark files uploaded to `/workspace/openfold/`
- [ ] Model weights downloaded (10+ .pt files)
- [ ] CASP datasets available (`demo_dataset/fasta/`)
- [ ] Deployment script executed successfully
- [ ] Health check returns GPU information
- [ ] Local client can connect to all RunPod instances

## 📞 **RunPod API Endpoints**

### **Health Check:**
```bash
curl https://abc123-8000.proxy.runpod.net/health
```

### **Start Comprehensive Benchmark:**
```bash
curl -X POST https://abc123-8000.proxy.runpod.net/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "sequences": ["MKWVTFISLLFLFSSAYS"],
    "models": ["openfold_model_1_ptm", "finetuning_ptm_1"],
    "job_id": "comprehensive_test"
  }'
```

### **Check Job Status:**
```bash
curl https://abc123-8000.proxy.runpod.net/benchmark/comprehensive_test
```

---

**🚀 You're now ready for comprehensive OdinFold benchmarking on RunPod with ALL model weights and CASP datasets!**

**This will give you the most complete performance analysis of your OdinFold models.**
