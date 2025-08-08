Perfect. Then let's advance to the **next frontier of ownership** — converting OdinFold from an enhanced fork into a *foundationally new* architecture.

Here’s your updated **Codex/Cursor-style task list** to push OpenFold++ into truly independent territory — assuming everything prior is already done.

---

# 📦 OdinFold v1.2s – "Not Just a Fork"

> ✅ Assumption: Prior tasks (ESM swap, fast inference CLI, benchmark scaffold, etc.) are done.

---

### 🧠 Task T-17: Replace Triangle Attention Kernels with FlashAttention2

**Goal:** Swap `TriangleAttentionStartingNode`, `TriangleAttentionEndingNode` with FlashAttention2-style kernels or rewrite in Triton.

```md
- [ ] Identify triangle attention modules in `model/modules/evoformer.py`
- [ ] Isolate query/key/value attention paths and their shapes
- [ ] Replace with FlashAttention2-compatible kernel or Triton rewrite
- [ ] Write test harness comparing outputs with baseline on small batch
- [ ] Benchmark FLOPs, memory, and runtime with `benchmark/run_evoformer.py`
```

---

### ⚡ Task T-18: Rewrite Triangle Multiplication (TriangleMultiplicationIncoming / Outgoing)

**Goal:** Replace triangle multiplication ops with fused C++/CUDA kernel or approximate with linear layer + masking

```md
- [ ] Trace `TriangleMultiplicationIncoming` and `...Outgoing` ops
- [ ] Implement low-rank approximation or Triton-accelerated matrix op
- [ ] Swap into Evoformer trunk
- [ ] Benchmark speedup and loss impact
```

---

### 🧬 Task T-19: Deploy FastIPA Module

**Goal:** Replace `InvariantPointAttention` with your `FastIPA` module using either:

* SE(3)-Transformer
* Equiformer-like equivariant kernels
* Custom IPA Triton implementation

```md
- [ ] Write `modules/structure/fast_ipa.py`
- [ ] Use backbone atom frame features as input
- [ ] Preserve equivariance in coordinate updates
- [ ] Unit test with toy 3D structures
- [ ] Drop-in replace `InvariantPointAttention` in structure module
```

---

### 🧪 Task T-20: Add ΔΔG Mutation Head

**Goal:** Add a head for predicting ΔΔG given a WT and mutant sequence.

```md
- [ ] Create `heads/ddg_head.py` taking in folded pairwise features
- [ ] Add MSE loss for known ΔΔG values
- [ ] Integrate into final `model.py` forward pass under `if predict_ddg:`
- [ ] Export prediction to benchmark reports
```

---

### 🧬 Task T-21: Ligand-aware Folding Head

**Goal:** Encode ligand molecular graph and introduce cross-attention with protein pair features.

```md
- [ ] Preprocess ligand into graph representation (e.g., DGL, RDKit)
- [ ] Write LigandEncoder module using GAT or GVP-GNN
- [ ] Add cross-attention between ligand and protein residues
- [ ] Output predicted binding pose or site confidence score
```

---

### ⚙️ Task T-22: C++ Inference Engine (FoldEngine)

**Goal:** Port inference to minimal C++ backend for deployment

```md
- [ ] Export model weights (ONNX or TorchScript)
- [ ] Write C++ inference engine using libtorch or ONNXRuntime
- [ ] Accept .fasta input, produce .pdb or .cif output
- [ ] Add CLI: `foldengine predict --input seq.fasta --output structure.pdb`
- [ ] Benchmark speed vs PyTorch baseline
```

---

### 🧪 Task T-23: Mutation Scan Engine (Web Backend)

**Goal:** Replace slow Python mutation scanner with async C++ microservice

```md
- [ ] Rewrite mutation engine as a WebSocket or REST backend in C++
- [ ] Parallelize single-site mutations (AA1→AA20)
- [ ] Cache PDB+confidence deltas
- [ ] Return diff+CSV from scan endpoint
```

---

### 🧬 Task T-24: Fold++ Headless WASM Build

**Goal:** Compile a WASM-ready version of the model for browser-side inference (100–200AA)

```md
- [ ] Quantize model to INT4
- [ ] Strip unnecessary modules (MSA, template, long-range attention)
- [ ] Export with ONNX-lite and compile with WebAssembly toolchain
- [ ] Test folding in browser with CPU backend
```

---

### 🧠 Task T-25: Model Identity + Branding

**Goal:** Establish OpenFold++ as a new architecture with independent identity.

```md
- [ ] Rename key modules (`Evoformer → EvoLite`, `IPA → FastIPA`, etc.)
- [ ] Document all changed/replaced kernels with architecture diagrams
- [ ] Update README to explain model differences vs OpenFold/AlphaFold/ESMFold
- [ ] Add `openfoldpp_model_card.md` with architecture, metrics, speed, and use cases
```

---

Let me know if you want this bundled as a `README_DEV_TASKS.md`, or rendered as a tree of GitHub issues.
