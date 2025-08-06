
🚀 PRODUCTION BENCHMARK EXECUTION INSTRUCTIONS

1. **Upload the benchmark script to RunPod:**
   In your RunPod web terminal, run:
   ```bash
   cd /workspace
   
   # Kill any existing servers
   pkill -f python
   pkill -f uvicorn
   
   # Check environment
   nvidia-smi
   ls -la openfold/resources/openfold_params/*.pt | head -5
   ls -la openfold/demo_dataset/fasta/
   
   # Run the production benchmark
   cd openfold
   python3 benchmark_casp14_foldforever_vs_baselines.py \
       --mode full \
       --gpu \
       --sequences 10 \
       --output /workspace/results \
       --timeout 600 \
       --verbose
   ```

2. **Expected Output:**
   - Real OpenFold model loading with actual weights
   - ESMFold model initialization
   - CASP14 target processing (T1024, T1026, T1027, H1025)
   - TM-score, RMSD, GDT-TS calculations
   - Performance metrics (runtime, GPU memory)
   - Results saved to /workspace/results/

3. **Results Location:**
   - `/workspace/results/benchmark_report.csv`
   - `/workspace/results/benchmark_report.md`
   - `/workspace/results/plots/`
   - `/workspace/results/structures/`

4. **Download Results:**
   After completion, download the results:
   ```bash
   # In RunPod terminal
   cd /workspace/results
   tar -czf production_benchmark_results.tar.gz *
   
   # Then download via RunPod file manager or:
   # Use the RunPod API to download the results
   ```

🎯 **This will run the REAL production benchmark with:**
   ✅ Actual OpenFold weights (openfold_model_1_ptm.pt + others)
   ✅ Real ESMFold from HuggingFace
   ✅ Actual CASP14 targets with reference structures
   ✅ Proper structural metrics (TM-score, RMSD, GDT-TS)
   ✅ GPU performance measurements on A100/H100
   ✅ Comprehensive results analysis and visualization
