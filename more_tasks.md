Below is an **exhaustively detailed, Cursor-/GitHub-friendly task breakdown** that turns the four-phase blueprint into discrete “tickets.”
Each task includes **goal, context, deliverables, suggested file locations, acceptance tests,** and (where helpful) **CLI snippets** so an autonomous agent like **Codex or Cursor** can run them without guesswork.

---

# 📂 Repository Layout (you may already have parts of this)

```
openfoldpp/
├── data/
│   ├── pdb/                  # PDB/mmCIF downloads + splits
│   ├── esm2_embeddings/      # .pt or .npy files (token reps)
│   └── templates/            # HHsearch / template DB
├── src/
│   ├── openfoldpp/           # new package namespace
│   │   ├── modules/          # evoformer, structure, diffusion
│   │   ├── configs/
│   │   └── __init__.py
│   ├── training/             # train loops, distillation scripts
│   └── evaluation/           # TM-score, RMSD, CASP runner
├── notebooks/
├── scripts/
│   └── cli/
└── requirements.txt
```

> **Convention**: Ticket IDs use `PAX-` prefix (Phase A tasks), `PBX-`, `PCX-`, `PDX-` for later phases so sorting by name keeps phases grouped.

---

## 📍 Phase A – *Swap MSA for PLM* (2 weeks)

### **PAX-01 | Add ESM-2 dependency**

* **Goal**  Add Facebook’s ESM-2 (650 M) as a sub-module / pip requirement.
* **Context**  We will freeze this model for inference only.
* **Steps**

  1. `pip install fair-esm==1.0.3` (or sub-module clone for offline).
  2. Verify GPU & CPU forward pass: `python -m esm.pretrained.esm2_t33_650M_UR50D`.
  3. Add to `requirements.txt` and CI workflow.
* **Deliverable**  `src/openfoldpp/backbones/esm2_wrapper.py`.
* **Acceptance**  `pytest tests/test_esm2_wrapper.py` returns embeddings of shape `[L, 1280]` for an input FASTA.

---

### **PAX-02 | Embedding extraction pipeline**

* **Goal**  Offline-extract token embeddings for all sequences in PDB split.
* **Context**  Reduces wall-clock during training; aligns with LLaMA “pre-compute KV cache” idea.
* **Steps**

  1. Script `scripts/cli/extract_esm2.py --fasta <file> --out <dir>`.
  2. Chunk sequences in batches of 4 × 4096 tokens to avoid OOM.
  3. Save torch `float16` `.pt` with keys `seq`, `tokens`, `repr`.
* **Deliverable**  5 TB shard directory in `data/esm2_embeddings/`.
* **Acceptance**  `wc -l data/splits/train.txt` lines == number of `.pt` files produced.

---

### **PAX-03 | 8-bit GPTQ quantization**

* **Goal**  Compress ESM-2 weights to \~1.3 GB.
* **Context**  Follows GPTQ paper; use bitsandbytes or AutoGPTQ.
* **Steps**

  1. Add `auto-gptq` to `requirements.txt`.
  2. Quant script: `quantize_esm2.py --wbits 8 --act-order --sym`.
  3. Bench embed quality versus fp16 on set of 100 proteins, compute cosine similarity Δ.
* **Deliverable**  `esm2_t33_650M_gptq.bin`.
* **Acceptance**  Average cosine similarity ≥ 0.985 relative to fp16 baseline.

---

### **PAX-04 | Linear projection into EvoFormer**

* **Goal**  Map 1280-dim PLM token reps → 64- or 128-dim single MSA row.
* **Steps**

  1. Add `TokenProjector(nn.Linear(1280, PROJ_DIM))` with bias=False.
  2. Replace OpenFold `MSAColumnEmbedding` call with projector output.
  3. Remove padding logic that expected `N_seq > 1`.
* **Deliverable**  `src/openfoldpp/modules/plm_msa_bridge.py`.
* **Acceptance**  Unit test: projector output shape `[L, PROJ_DIM]`, gradients flow.

---

### **PAX-05 | Benchmark accuracy drop**

* **Goal**  Verify TM loss ≤ 0.04 on quick set (30 CASP targets).
* **Steps**

  1. Adapt `evaluation/benchmark.py` to toggle `--plm-mode`.
  2. Run `python benchmark.py --config configs/plm_baseline.yaml`.
* **Deliverable**  Markdown report in `reports/phaseA_drop.md`.
* **Acceptance**  Median TM ≥ (baseline – 0.04).

---

## 📍 Phase B – *Slim EvoFormer trunk* (4 weeks)

### **PBX-10 | Halve layer depth**

* **Goal**  Change `n_blocks` from 48→24.
* **Steps**

  1. Update config `evo_blocks: 24`.
  2. Remove positional bias duplication across removed layers.
  3. Re-init weights with teacher transfer (copy even-numbered layers).
* **Acceptance**  Model loads and forward pass speed improves ≥ 1.6×.

---

### **PBX-11 | Grouped-Query Attention (GQA) in pair modules**

* **Goal**  Implement KV sharing (k = 4) in triangular attention.
* **Steps**

  1. Fork `attention.py` → add `q_groups = h // k`.
  2. Adjust shape math on KV cache.
* **Deliverable**  `TriangularAttentionGQA`.
* **Acceptance**  Memory footprint measured by `torch.cuda.max_memory_allocated()` drops ≥ 35 %.

---

### **PBX-12 | SwiGLU MLP replacement**

* **Goal**  Swap 4× hidden linear → Swish-Gated 2×.
* **Steps**

  1. `nn.SiLU` gate; hidden dim 1536.
  2. Copy weights from existing layer (truncate or average).
* **Acceptance**  Per-step training loss doesn’t spike > 15 % after swap (measured on 100-step sanity run).

---

### **PBX-13 | Weight sharing every 4 layers**

* **Goal**  Reduce param count.
* **Steps**

  1. Registry mapping `block[i] = shared_block[i//4]`.
  2. Freeze shared weights; ensure grads accumulate properly.
* **Acceptance**  Parameter count reported by `summary(model)` ≈ 110 M.

---

### **PBX-14 | FlashAttention-2 kernels**

* **Goal**  Integrate `flash_attn` for both seq-seq and pair attention.
* **Steps**

  1. `pip install flash-attn==2.*` (CUDA path) and `triton`.
  2. If `DEVICE==mps`, fallback to chunked attention.
  3. Unit test gradient vs reference within 1e-3.
* **Acceptance**  Forward-+-backward time per block drops ≥ 40 %.

---

### **PBX-15 | Comprehensive Phase B regression benchmark**

* **Goal**  Measure speed, RAM, TM accuracy after all slimming.
* **Deliverable**  `reports/phaseB_benchmark.md`.
* **Acceptance**  - Runtime ≥ 2 × faster than Phase A baseline.

  * Param count ≤ 115 M.
  * TM drop ≤ 0.03 wrt Phase A.

---

## 📍 Phase C – *Teacher-student distillation* (3 weeks GPU time)

### **PCX-20 | Prepare teacher inference pipeline**

* **Goal**  Run AF-2 (or AF-3) to generate soft targets.
* **Steps**

  1. Docker image `alphafold:2.3.3` with mutated RASP crystaldb.
  2. Script `gen_teacher_targets.py --pdb-list train.txt`.
  3. Store outputs: `coords.npy`, `pLDDT.npy`, `pair_repr.npy`.
* **Acceptance**  1000 sequences processed/hr on 8-GPU node.

---

### **PCX-21 | Define distillation losses**

* **Goal**  Add `DistillationLoss` module.
* **Loss terms**

  * `L_coord = MSE(pred_coords, teacher_coords)` (rigid-body aligned).
  * `L_plddt = KL(student_logits, teacher_logits)`.
  * `L_pair = MSE(student_pair, teacher_pair)` *(optional)*.
  * Weight schedule YAML-configurable.
* **Acceptance**  `pytest tests/test_losses.py` passes.

---

### **PCX-22 | LoRA adapters on EvoFormer**

* **Goal**  Train only low-rank adapters (rank = 8) to save VRAM.
* **Steps**

  1. Use `peft` or custom LoRA wrapper.
  2. Freeze original fp16 weights; only LoRA params in optimizer.
* **Acceptance**  GPU memory < 14 GB at batch = 1×140 AA crop.

---

### **PCX-23 | Distillation training script**

* **Goal**  `train_distill.py` orchestrates curriculum.
* **Steps**

  1. Config: `global_batch=64` via gradient accumulation.
  2. Cosine LR 1e-4 → 1e-6.
  3. Mixed-precision (`torch.grad_scaler`).
* **Acceptance**  Loss curve smooth, no NaNs for 10 k steps.

---

### **PCX-24 | Mid-training evaluation loop**

* **Goal**  Every 10 k steps, run CASP validation set.
* **Acceptance**  Checkpoint is kept iff TM improves > 0.01.

---

### **PCX-25 | Full distillation completion report**

* **Deliverable**  `reports/phaseC_distill.md` with:

  * Final TM on CASP14 (single & multimer).
  * Training cost (A100-hours, dollar).
  * Graphs: loss vs step, TM vs step.
* **Acceptance**  Median TM ≥ 0.82.

---

## 📍 Phase D – *Light refinement head* (2 weeks)

### **PDX-30 | Implement SE(3) diffusion refiner**

* **Goal**  Borrow RoseTTAFold 2’s 3D diffusion block.
* **Steps**

  1. Add `modules/diffusion_refiner.py`.
  2. 2 iterations, hidden dim = 256.
  3. Training objective: time-v-score matching.
* **Acceptance**  Standalone unit test reproduces input on t=0.

---

### **PDX-31 | Plug refiner after structure module**

* **Goal**  `forward()`: coords → refiner → final coords.
* **Acceptance**  Added latency < 1 s (benchmarked on A100 for 300 AA).

---

### **PDX-32 | 4-bit export of refiner weights**

* **Goal**  Use GGML or bitsandbytes int4.
* **Steps**

  1. `convert_to_ggml.py --model refiner.pt --wbits 4`.
  2. Modify load path in inference runner.
* **Acceptance**  TM drop ≤ 0.01 vs fp16 refiner.

---

### **PDX-33 | End-to-end benchmark & goal verification**

* **Goal**  Achieve ≥ 0.85 TM on validation *and* < 5 s on A100 batch = 1 (300 AA).
* **Steps**

  1. `python benchmark.py --config configs/final.yaml`.
  2. Capture `time`, `max_memory`.
* **Acceptance**  Both targets met. Record in `reports/final_goal.md`.

---

# 🏁 How to run tasks in Cursor

1. **Create an “Epics” folder** mapping to Phase A–D.
2. Copy each ticket description into individual markdown/`todo.txt` files.
3. Configure Cursor’s **“Suggest Code”** on each file:

   * `// TODO(PAX-01)` etc.
     Cursor auto-creates branches `feat/PAX-01` etc.
4. At merge time, CI pipeline runs:

   ```bash
   make test && make benchmark
   ```

   and must pass before Cursor auto-merges.

---

## ✅ Recap deliverables

| Phase | Success criteria                                                               |
| ----- | ------------------------------------------------------------------------------ |
| **A** | PLM embeddings working, TM loss ≤ 0.04, runtime –MSA.                          |
| **B** | Params ≈ 110 M, activations –45 %, 2 × speed, TM-loss ≤ 0.03.                  |
| **C** | Distilled student TM ≥ 0.82, no accuracy cliff.                                |
| **D** | Final TM ≥ 0.85, single-chain 300 AA fold < 5 s on A100, < 60 s on Mac laptop. |

Feed these tickets to Codex/Cursor and let the agent chew through them one branch at a time.

*Happy folding!*
