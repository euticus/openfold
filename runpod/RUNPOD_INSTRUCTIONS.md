# 🚀 RunPod A100 Deployment Instructions

## 📁 Files to Upload to RunPod:

1. **runpod_setup.py** - Setup script (run this first)
2. **benchmark_casp14_foldforever_vs_baselines.py** - Main benchmark
3. **runpod_requirements.txt** - Python dependencies

## 🎯 Step-by-Step Instructions:

### 1. Wait for Pod to Start
- Your pod should be starting now
- You'll get a Jupyter notebook link
- Click "Connect" when ready

### 2. Upload Files
- In Jupyter, click "Upload" 
- Upload all 3 files above
- They should appear in your file browser

### 3. Run Setup
```python
# In a new Jupyter cell, run:
exec(open('runpod_setup.py').read())
```

### 4. Run Benchmark
```python
# In another cell:
!python benchmark_casp14_foldforever_vs_baselines.py --mode full --gpu --sequences 30 --timeout 1800 --verbose
```

### 5. Monitor Progress
- The benchmark will run for ~2-3 hours
- You'll see real-time progress updates
- GPU utilization should be high

### 6. Download Results
- Results will be in `/workspace/foldforever_benchmark/results/`
- Download the entire results folder
- Contains CSV data, plots, and analysis

## 💰 Expected Cost:
- **Runtime:** 2-3 hours
- **Cost:** $3.28 - $4.92 total
- **Value:** Comprehensive A100 benchmark results!

## 🔍 Monitoring:
- Watch GPU usage: `!nvidia-smi`
- Check progress in real-time
- Stop anytime if needed

## 🎉 Expected Results:
- **FoldForever vs ESMFold vs OpenFold vs AlphaFold2**
- **Real CASP14 + CAMEO datasets**
- **Performance metrics, timing, accuracy**
- **Beautiful visualizations and analysis**

Ready to get the best protein folding benchmark results possible! 🧬
