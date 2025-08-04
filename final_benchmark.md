Here's a **complete, detailed benchmark plan** in Cursor-style format that you can give to Codex or a dev agent. It’s structured for **maximum reproducibility** and ensures scientific credibility when validating FoldForever's TM-score and performance.

---

# 🧪 Benchmark Plan: FoldForever vs AlphaFold2 / OpenFold / ESMFold

### 📂 Task Name: `benchmark_casp14_foldforever_vs_baselines.py`

---

## 🎯 **Objective**

Evaluate the **folding accuracy**, **runtime**, and **resource consumption** of FoldForever compared to AlphaFold2, OpenFold, and ESMFold using the **CASP14 and CAMEO datasets**.

The benchmark will measure:

* TM-score, RMSD, GDT-TS, lDDT
* Inference runtime per sequence length
* Peak GPU memory usage
* Accuracy vs MSA/template-dependent baselines

---

## 📁 **Benchmark Inputs**

### Dataset A: `CASP14_targets/`

* Format: `.fasta`
* Number of sequences: ≥30 diverse targets
* Length range: 60–900 amino acids
* Ground-truth structures in `.pdb` format

### Dataset B: `CAMEO_novel/`

* Format: `.fasta`
* Low-homology sequences (<30% identity)
* Optional: GPCR-focused proteins or membrane-bound targets

---

## 🧠 **Models to Compare**

| Model       | Source                   | MSA-Free | Template-Free |
| ----------- | ------------------------ | -------- | ------------- |
| FoldForever | Your `FoldEngine`        | ✅        | ✅             |
| OpenFold    | `openfold` repo          | ❌        | ⚠️ optional   |
| AlphaFold2  | DeepMind weights         | ❌        | ❌             |
| ESMFold     | Meta’s HuggingFace model | ✅        | ✅             |

---

## 🔍 **Metrics to Record**

### Structural Accuracy

| Metric   | Tool                   | Description                          |
| -------- | ---------------------- | ------------------------------------ |
| TM-score | `tm-score` binary      | Measures structural similarity (0–1) |
| RMSD     | PyMOL or BioPython     | Backbone alignment distance (Å)      |
| GDT-TS   | `GDT_TS` from TM-align | Global structure score               |
| lDDT     | AlphaFold script       | Local Distance Difference Test       |

### Performance

| Metric         | How to Measure                          |
| -------------- | --------------------------------------- |
| Inference time | `time.perf_counter()` around model call |
| GPU memory     | `torch.cuda.max_memory_allocated()`     |
| Total memory   | `nvidia-smi` logging or PyTorch tracker |

---

## 💻 **Benchmark Hardware**

| Parameter             | Value                          |
| --------------------- | ------------------------------ |
| GPU                   | NVIDIA A100 (80GB) or RTX 4090 |
| CPU                   | 16-core (minimum 3.0 GHz)      |
| RAM                   | 64GB+                          |
| OS                    | Ubuntu 22.04                   |
| CUDA Version          | 11.8+                          |
| PyTorch Version       | ≥2.1                           |
| FlashAttention Kernel | Enabled (if applicable)        |

**Repeat each test on:**

* A100
* RTX 4090
* M1/M2 Mac (if running WASM or Metal version)

---

## 🧪 **Benchmark Procedure**

```python
for sequence in dataset:
    load_sequence(sequence.fasta)
    
    for model in [FoldForever, OpenFold, AlphaFold2, ESMFold]:
        start_timer()
        prediction = model.fold(sequence)
        end_timer()
        
        save_structure(prediction, model_name)
        compute_metrics(prediction.pdb, ground_truth.pdb)
        
        log: {
            "model": model_name,
            "TM": tm_score,
            "RMSD": rmsd,
            "GDT": gdt_ts,
            "lDDT": lddt,
            "runtime_s": end - start,
            "gpu_memory_MB": torch.cuda.max_memory_allocated()
        }
```

---

## 🧾 **Output Format**

Save all results to: `results/benchmark_report.csv`

| Model       | Sequence ID | Length | TM-score | RMSD | GDT-TS | lDDT | Runtime (s) | GPU Mem (MB) |
| ----------- | ----------- | ------ | -------- | ---- | ------ | ---- | ----------- | ------------ |
| FoldForever | T1040       | 230    | 0.89     | 2.1  | 79.2   | 86.4 | 0.21        | 1856         |

Also export:

* `plots/tm_vs_length.png`
* `plots/runtime_vs_length.png`
* `plots/tm_distribution_violinplot.png`
* `plots/gpu_memory_comparison_bar.png`

---

## ✅ **Success Criteria**

* **Accuracy parity or improvement** over ESMFold (TM ≥ 0.78)
* **Runtime <1s** for 100–300AA on A100/4090
* **Memory footprint <4GB** on all test runs
* FoldForever runs without any MSA or template files

---

## 🧠 Bonus (Optional)

* Add mutation scan benchmark:

  * Pick 5 sequences
  * Mutate 10 residues → fold each
  * Record ΔΔG prediction time + accuracy if ground truth exists

---

Let me know if you want a prebuilt `benchmark_runner.py` or full test harness + report generator in Python.
