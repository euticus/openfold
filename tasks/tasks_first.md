OPENFOLD TASKSAbsolutely — here's the overly detailed, Codex/Cursor-friendly OpenFold++ task list with expanded explanations, step-by-step intent, and clear objectives. It’s verbose on purpose to help clarify scope and give you (or an AI agent) everything needed to execute each task in isolation.

# ✅ OpenFold++ – Expanded Task List (Verbose + Codex/Cursor Friendly)

This markdown document outlines each development task required to build OpenFold++, your high-performance protein folding engine. Each task includes a full breakdown of what to build, how to do it, and what success looks like.

---

## 🧱 Core Model & Optimization

---

### 1. Fork and Baseline Setup

**Goal**: Establish a reproducible baseline using the original OpenFold.

**Instructions**:
- Clone the [OpenFold GitHub repo](https://github.com/aqlaboratory/openfold).
- Set up a new conda environment or use `venv` and install all dependencies from `requirements.txt`.
- Make sure CUDA and PyTorch are compatible with your GPU.
- Run the default `run_pretrained_openfold.py` script with a test FASTA input.
- Confirm you can generate a `.pdb` output successfully.

**Deliverable**: Working local inference with OpenFold on a known protein sequence (e.g., lysozyme or spike protein).

---

### 2. Add Multimer Input Support

**Goal**: Modify the model and input pipeline to accept multiple protein chains.

**Instructions**:
- Extend the featurizer to support chain identifiers per residue.
- Add a chain-wise positional encoding or modify existing positional embeddings to reflect chain separation.
- Modify Evoformer attention masks to allow inter-chain attention.
- Ensure chains are aligned and padded properly in the input tensors.

**Deliverable**: `fold_multimer(sequence_a, sequence_b)` returns a structure containing multiple chains.

---

### 3. Implement Multimer Attention and Contact Loss

**Goal**: Improve folding accuracy on multimer complexes.

**Instructions**:
- Add masking logic in the Evoformer to control inter-chain attention vs intra-chain.
- Compute predicted contact maps between chains.
- Implement a loss function that compares predicted interfacial contacts to ground truth interface residues from `.pdb`.

**Deliverable**: Model training or inference includes interfacial accuracy metrics (e.g., dockQ, RMSD at interface).

---

### 4. Parse and Encode Ligand Input

**Goal**: Allow OpenFold++ to process small molecule ligands as inputs.

**Instructions**:
- Accept SMILES, MOL2, or SDF files as optional inputs.
- Use RDKit or PyTorch Geometric to convert ligands into graph or 3D tensors.
- Embed ligand structure using a GCN or MPNN into a vector representation that can be injected into Evoformer.

**Deliverable**: Ligand embedding tensor is appended to model input features.

---

### 5. Ligand-Aware Folding Integration

**Goal**: Condition structure prediction on presence of a ligand.

**Instructions**:
- Inject ligand embeddings into pair representation or directly into Evoformer.
- Modify structure module to optionally use ligand context during atom placement.
- Train or fine-tune model using PDBbind or CrossDocked datasets that contain ligand complexes.

**Deliverable**: Ligand-aware folding produces structures with plausible binding pockets.

---

### 6. Replace Attention with FlashAttention or Performer

**Goal**: Accelerate Evoformer attention layers.

**Instructions**:
- Install [FlashAttention](https://github.com/Dao-AILab/flash-attention) or [Performer](https://github.com/lucidrains/performer-pytorch).
- Replace standard `nn.MultiheadAttention` layers with FlashAttention modules.
- Ensure rotary position embeddings or relative encodings are preserved.
- Run a timing benchmark before and after to verify speedup.

**Deliverable**: FlashAttention-enabled Evoformer with confirmed performance improvement.

---

### 7. Replace MSA with LM Embeddings

**Goal**: Eliminate dependency on MSA by using protein language model embeddings.

**Instructions**:
- Load a pretrained model like ESM2 or ProtT5.
- For each input sequence, compute residue-level embeddings.
- Replace or supplement the MSA tensor in OpenFold with these embeddings.
- Update Evoformer to operate in single-sequence mode.

**Deliverable**: Fold pipeline with no MSA required that uses LM embeddings for single sequences.

---

### 8. Quantize the Model and Add Checkpointing

**Goal**: Reduce inference memory and support long-sequence prediction.

**Instructions**:
- Apply `bitsandbytes` or PyTorch native quantization to compress model to 8-bit or 4-bit.
- Enable gradient checkpointing for Evoformer blocks (e.g., via `torch.utils.checkpoint`).
- Ensure outputs remain numerically stable.

**Deliverable**: Model runs on 12GB GPUs, supports 3K+ token sequences, and maintains folding accuracy.

---

### 9. Add MD-Based Refinement Post-Fold

**Goal**: Refine predicted structure using physical simulation.

**Instructions**:
- After prediction, export `.pdb` and feed into TorchMD or OpenMM.
- Apply energy minimization or constrained dynamics on sidechains/backbone.
- Return refined `.pdb` or structure tensor.

**Deliverable**: Optional refinement flag in CLI that improves stereochemistry and bond geometry.

---

## 🔁 Real-Time Mutation Folding

---

### 10. Train a Delta Prediction Model (GNN)

**Goal**: Predict small structural changes from local mutations.

**Instructions**:
- Use FoldX or a custom dataset of known mutations and structure pairs.
- Train a GNN or SE(3) model to output Δx, Δy, Δz for residues in a window.
- Evaluate with RMSD or contact change precision.

**Deliverable**: A model callable via `predict_delta(pdb, mutation_site, new_residue)`.

---

### 11. Integrate Delta Predictor into WebSocket Server

**Goal**: Provide real-time mutation patching.

**Instructions**:
- When a mutation request comes in, feed to delta model.
- Receive modified coordinates or side-chain placements.
- Inject into base structure and return new `.pdb`.

**Deliverable**: Client receives a visual update <1s after mutation submission.

---

### 12. Build WebSocket Mutation Server

**Goal**: Enable persistent session-based structure editing.

**Instructions**:
- Use FastAPI or Starlette to create `/ws/mutate`.
- Accept `init` (full structure) and `mutate` (residue update) messages.
- Store cached state server-side (Redis or in-memory).

**Deliverable**: Bi-directional WebSocket for mutation edits with live response.

---

### 13. Add TorchMD or OpenMM Refinement After Mutation

**Goal**: Clean up bond lengths or clashes after mutation.

**Instructions**:
- For large mutations or insertions, call TorchMD as a fallback.
- Apply 20–50 steps of constrained minimization on updated region.

**Deliverable**: More chemically plausible structure in real-time folding.

---

## 🧬 MMseqs2-GPU Integration

---

### 14. Use MMseqs2-GPU for Fast Preprocessing

**Goal**: Accelerate homology search or create LM context embeddings.

**Instructions**:
- Install [MMseqs2-GPU](https://github.com/soedinglab/MMseqs2).
- Use `easy-search` to get top homologs.
- Optionally embed top hits with ESM2 and feed into folding pipeline.
- Replace standard MSA construction.

**Deliverable**: Preprocessor script that runs MMseqs2-GPU and returns formatted tensor for model input.

---

## ⚙️ Triangle Kernel Acceleration in CUDA

---

### 15. Rebuild Triangle Attention and Multiplication in CUDA

**Goal**: Replace slow triangle ops with fast GPU-native implementations.

**Instructions**:
- Write `triangle_attention.cu` and `triangle_multiply.cu` kernels.
- Use shared memory and block tiling for performance.
- Implement custom fused attention if needed.

**Deliverable**: `libtriangle_kernels.so` with `triangle_attention_forward()` callable in Python.

---

### 16. Bind C++ Kernels to Python using pybind11

**Goal**: Make custom kernels usable from PyTorch.

**Instructions**:
- Write `bindings.cpp` that exposes `triangle_attention_forward()` and `triangle_multiply_forward()`.
- Compile using `torch.utils.cpp_extension` or a `setup.py` with CMake backend.
- Validate on dummy tensors.

**Deliverable**: Python call `triangle_kernels.triangle_attention_forward()` works and matches PyTorch output.

---

### 17. Benchmark CUDA Kernels Against PyTorch Equivalents

**Goal**: Confirm speed and correctness gains from custom CUDA kernels.

**Instructions**:
- Compare runtime on same input: `[B, N, N, C]` for triangle attention.
- Use `torch.utils.benchmark.Timer`.
- Compare VRAM, FLOPs, and wall time.

**Deliverable**: Report showing speedup vs OpenFold baseline and accuracy difference (if any).

---

## 🖥️ CLI, API, and Deployment

---

### 18. Build CLI + Python SDK

**Goal**: Provide a user-friendly interface for running folds and mutations.

**Instructions**:
- Create CLI commands: `fold`, `fold_multimer`, `mutate`, `refine`.
- Build corresponding Python SDK with consistent I/O.
- Accept sequence file, optional ligand, output `.pdb`.

**Deliverable**: `python -m openfoldpp fold input.fasta --output output.pdb` works as expected.

---

### 19. Add API with Job Queue

**Goal**: Expose folding and mutation as asynchronous API endpoints.

**Instructions**:
- Use FastAPI with Celery + Redis backend.
- Add `/fold`, `/mutate`, `/status/<job_id>` endpoints.
- Implement automatic GPU job queue.

**Deliverable**: API returns job ID, status, and download links for results.

---

### 20. Export Final Model to TorchScript + ONNX

**Goal**: Make model portable for deployment and inference on non-Python stacks.

**Instructions**:
- Use `torch.jit.trace()` or `torch.jit.script()` to export TorchScript model.
- Use `torch.onnx.export()` for ONNX compatibility.
- Document input/output schema.

**Deliverable**: `openfoldpp.pt` and `openfoldpp.onnx` available for external inference.


OPENFOLD TASKSAbsolutely — here's the overly detailed, Codex/Cursor-friendly OpenFold++ task list with expanded explanations, step-by-step intent, and clear objectives. It’s verbose on purpose to help clarify scope and give you (or an AI agent) everything needed to execute each task in isolation.

# ✅ OpenFold++ – Expanded Task List (Verbose + Codex/Cursor Friendly)

This markdown document outlines each development task required to build OpenFold++, your high-performance protein folding engine. Each task includes a full breakdown of what to build, how to do it, and what success looks like.

---

## 🧱 Core Model & Optimization

---

### 1. Fork and Baseline Setup

**Goal**: Establish a reproducible baseline using the original OpenFold.

**Instructions**:
- Clone the [OpenFold GitHub repo](https://github.com/aqlaboratory/openfold).
- Set up a new conda environment or use `venv` and install all dependencies from `requirements.txt`.
- Make sure CUDA and PyTorch are compatible with your GPU.
- Run the default `run_pretrained_openfold.py` script with a test FASTA input.
- Confirm you can generate a `.pdb` output successfully.

**Deliverable**: Working local inference with OpenFold on a known protein sequence (e.g., lysozyme or spike protein).

---

### 2. Add Multimer Input Support

**Goal**: Modify the model and input pipeline to accept multiple protein chains.

**Instructions**:
- Extend the featurizer to support chain identifiers per residue.
- Add a chain-wise positional encoding or modify existing positional embeddings to reflect chain separation.
- Modify Evoformer attention masks to allow inter-chain attention.
- Ensure chains are aligned and padded properly in the input tensors.

**Deliverable**: `fold_multimer(sequence_a, sequence_b)` returns a structure containing multiple chains.

---

### 3. Implement Multimer Attention and Contact Loss

**Goal**: Improve folding accuracy on multimer complexes.

**Instructions**:
- Add masking logic in the Evoformer to control inter-chain attention vs intra-chain.
- Compute predicted contact maps between chains.
- Implement a loss function that compares predicted interfacial contacts to ground truth interface residues from `.pdb`.

**Deliverable**: Model training or inference includes interfacial accuracy metrics (e.g., dockQ, RMSD at interface).

---

### 4. Parse and Encode Ligand Input

**Goal**: Allow OpenFold++ to process small molecule ligands as inputs.

**Instructions**:
- Accept SMILES, MOL2, or SDF files as optional inputs.
- Use RDKit or PyTorch Geometric to convert ligands into graph or 3D tensors.
- Embed ligand structure using a GCN or MPNN into a vector representation that can be injected into Evoformer.

**Deliverable**: Ligand embedding tensor is appended to model input features.

---

### 5. Ligand-Aware Folding Integration

**Goal**: Condition structure prediction on presence of a ligand.

**Instructions**:
- Inject ligand embeddings into pair representation or directly into Evoformer.
- Modify structure module to optionally use ligand context during atom placement.
- Train or fine-tune model using PDBbind or CrossDocked datasets that contain ligand complexes.

**Deliverable**: Ligand-aware folding produces structures with plausible binding pockets.

---

### 6. Replace Attention with FlashAttention or Performer

**Goal**: Accelerate Evoformer attention layers.

**Instructions**:
- Install [FlashAttention](https://github.com/Dao-AILab/flash-attention) or [Performer](https://github.com/lucidrains/performer-pytorch).
- Replace standard `nn.MultiheadAttention` layers with FlashAttention modules.
- Ensure rotary position embeddings or relative encodings are preserved.
- Run a timing benchmark before and after to verify speedup.

**Deliverable**: FlashAttention-enabled Evoformer with confirmed performance improvement.

---

### 7. Replace MSA with LM Embeddings

**Goal**: Eliminate dependency on MSA by using protein language model embeddings.

**Instructions**:
- Load a pretrained model like ESM2 or ProtT5.
- For each input sequence, compute residue-level embeddings.
- Replace or supplement the MSA tensor in OpenFold with these embeddings.
- Update Evoformer to operate in single-sequence mode.

**Deliverable**: Fold pipeline with no MSA required that uses LM embeddings for single sequences.

---

### 8. Quantize the Model and Add Checkpointing

**Goal**: Reduce inference memory and support long-sequence prediction.

**Instructions**:
- Apply `bitsandbytes` or PyTorch native quantization to compress model to 8-bit or 4-bit.
- Enable gradient checkpointing for Evoformer blocks (e.g., via `torch.utils.checkpoint`).
- Ensure outputs remain numerically stable.

**Deliverable**: Model runs on 12GB GPUs, supports 3K+ token sequences, and maintains folding accuracy.

---

### 9. Add MD-Based Refinement Post-Fold

**Goal**: Refine predicted structure using physical simulation.

**Instructions**:
- After prediction, export `.pdb` and feed into TorchMD or OpenMM.
- Apply energy minimization or constrained dynamics on sidechains/backbone.
- Return refined `.pdb` or structure tensor.

**Deliverable**: Optional refinement flag in CLI that improves stereochemistry and bond geometry.

---

## 🔁 Real-Time Mutation Folding

---

### 10. Train a Delta Prediction Model (GNN)

**Goal**: Predict small structural changes from local mutations.

**Instructions**:
- Use FoldX or a custom dataset of known mutations and structure pairs.
- Train a GNN or SE(3) model to output Δx, Δy, Δz for residues in a window.
- Evaluate with RMSD or contact change precision.

**Deliverable**: A model callable via `predict_delta(pdb, mutation_site, new_residue)`.

---

### 11. Integrate Delta Predictor into WebSocket Server

**Goal**: Provide real-time mutation patching.

**Instructions**:
- When a mutation request comes in, feed to delta model.
- Receive modified coordinates or side-chain placements.
- Inject into base structure and return new `.pdb`.

**Deliverable**: Client receives a visual update <1s after mutation submission.

---

### 12. Build WebSocket Mutation Server

**Goal**: Enable persistent session-based structure editing.

**Instructions**:
- Use FastAPI or Starlette to create `/ws/mutate`.
- Accept `init` (full structure) and `mutate` (residue update) messages.
- Store cached state server-side (Redis or in-memory).

**Deliverable**: Bi-directional WebSocket for mutation edits with live response.

---

### 13. Add TorchMD or OpenMM Refinement After Mutation

**Goal**: Clean up bond lengths or clashes after mutation.

**Instructions**:
- For large mutations or insertions, call TorchMD as a fallback.
- Apply 20–50 steps of constrained minimization on updated region.

**Deliverable**: More chemically plausible structure in real-time folding.

---

## 🧬 MMseqs2-GPU Integration

---

### 14. Use MMseqs2-GPU for Fast Preprocessing

**Goal**: Accelerate homology search or create LM context embeddings.

**Instructions**:
- Install [MMseqs2-GPU](https://github.com/soedinglab/MMseqs2).
- Use `easy-search` to get top homologs.
- Optionally embed top hits with ESM2 and feed into folding pipeline.
- Replace standard MSA construction.

**Deliverable**: Preprocessor script that runs MMseqs2-GPU and returns formatted tensor for model input.

---

## ⚙️ Triangle Kernel Acceleration in CUDA

---

### 15. Rebuild Triangle Attention and Multiplication in CUDA

**Goal**: Replace slow triangle ops with fast GPU-native implementations.

**Instructions**:
- Write `triangle_attention.cu` and `triangle_multiply.cu` kernels.
- Use shared memory and block tiling for performance.
- Implement custom fused attention if needed.

**Deliverable**: `libtriangle_kernels.so` with `triangle_attention_forward()` callable in Python.

---

### 16. Bind C++ Kernels to Python using pybind11

**Goal**: Make custom kernels usable from PyTorch.

**Instructions**:
- Write `bindings.cpp` that exposes `triangle_attention_forward()` and `triangle_multiply_forward()`.
- Compile using `torch.utils.cpp_extension` or a `setup.py` with CMake backend.
- Validate on dummy tensors.

**Deliverable**: Python call `triangle_kernels.triangle_attention_forward()` works and matches PyTorch output.

---

### 17. Benchmark CUDA Kernels Against PyTorch Equivalents

**Goal**: Confirm speed and correctness gains from custom CUDA kernels.

**Instructions**:
- Compare runtime on same input: `[B, N, N, C]` for triangle attention.
- Use `torch.utils.benchmark.Timer`.
- Compare VRAM, FLOPs, and wall time.

**Deliverable**: Report showing speedup vs OpenFold baseline and accuracy difference (if any).

---

## 🖥️ CLI, API, and Deployment

---

### 18. Build CLI + Python SDK

**Goal**: Provide a user-friendly interface for running folds and mutations.

**Instructions**:
- Create CLI commands: `fold`, `fold_multimer`, `mutate`, `refine`.
- Build corresponding Python SDK with consistent I/O.
- Accept sequence file, optional ligand, output `.pdb`.

**Deliverable**: `python -m openfoldpp fold input.fasta --output output.pdb` works as expected.

---

### 19. Add API with Job Queue

**Goal**: Expose folding and mutation as asynchronous API endpoints.

**Instructions**:
- Use FastAPI with Celery + Redis backend.
- Add `/fold`, `/mutate`, `/status/<job_id>` endpoints.
- Implement automatic GPU job queue.

**Deliverable**: API returns job ID, status, and download links for results.

---

### 20. Export Final Model to TorchScript + ONNX

**Goal**: Make model portable for deployment and inference on non-Python stacks.

**Instructions**:
- Use `torch.jit.trace()` or `torch.jit.script()` to export TorchScript model.
- Use `torch.onnx.export()` for ONNX compatibility.
- Document input/output schema.

**Deliverable**: `openfoldpp.pt` and `openfoldpp.onnx` available for external inference.


Absolutely — here's a **comprehensive and up-to-date task list for OdinFold** (formerly OpenFold++) based on everything you've requested so far. This includes model optimization, inference speedups, benchmarking, WASM support, mutation scanning, and more.

---

## 🧬 **OdinFold Task List (T-1 → T-24)**

*(Grouped by focus area and written to be Codex/agent-friendly)*

---

### ### 🧠 MODEL OPTIMIZATION

#### **T-1: Replace MSA with PLM (ESM-2)**

* [ ] Freeze ESM-2-650M.
* [ ] Export per-token embeddings for input sequences.
* [ ] Pipe token embeddings `[seq_len, 1280]` → linear layer → Evoformer input.
* [ ] Compress ESM-2 model with GPTQ (INT4) to \~1.3 GB.

#### **T-2: Slim Evoformer Trunk**

* [ ] Reduce Evoformer depth from 48 → 24 blocks.
* [ ] Use GQA: Attention heads 16 → 8, KV heads 4.
* [ ] Change MLP to SwiGLU; reduce hidden dim from 2048 → 1536.
* [ ] Apply layer-wise weight sharing every 4 layers.
* [ ] Replace attention with FlashAttention-2 or MetalAttention.

#### **T-3: C++ Kernel Rewrite**

* [ ] Rewrite Triangle Attention/Multiplication using C++ for performance.
* [ ] Optimize with CUDA/Triton kernels (NVIDIA) and Metal (Mac).
* [ ] Benchmark speed vs. PyTorch baseline.

#### **T-4: Ligand Binding Support**

* [ ] Add ligand-aware attention heads in Evoformer.
* [ ] Accept ligand PDBQT/SDF files as optional input.
* [ ] Highlight binding region using structural embeddings.

#### **T-5: Multimer Support**

* [ ] Accept FASTA files with multiple chains.
* [ ] Update positional embeddings and chain break logic.
* [ ] Output multimeric PDB file with chain IDs.

---

### ⚙️ INFERENCE & PIPELINE

#### **T-6: TM-Score Confidence Head**

* [ ] Add separate head to predict expected TM-score.
* [ ] Use pretrained regression on CASP dataset.
* [ ] Display as confidence bar per residue and global TM.

#### **T-7: Mutation Scan Job Queue**

* [ ] Accept mutations in JSON: `{position: 123, from: 'A', to: 'V'}`
* [ ] Run background folds for each variant.
* [ ] Generate CSV with TM-score diff, pLDDT diff, RMSD, etc.
* [ ] Overlay structure differences in viewer.

#### **T-8: Ligand Binding Site Predictor**

* [ ] Add LLM prompt or CNN to predict likely ligand binding pockets.
* [ ] Output list of residues with binding probability score.
* [ ] Visualize hotspots in the viewer.

#### **T-9: Real-time Inference API**

* [ ] Serve OdinFold model via FastAPI endpoint: `/predict`
* [ ] Accept JSON: `{ sequence: "..." }`
* [ ] Return coordinates, TM, pLDDT, and download URL.

#### **T-10: Redis-Based Inference Caching**

* [ ] Hash input sequences.
* [ ] If hash in Redis, return cached fold result.
* [ ] If not, run fold, then store to Redis.

---

### 🧪 BENCHMARKING & VALIDATION

#### **T-11: CASP Benchmark Script**

* [ ] Run OdinFold on CASP13/14/15 targets.
* [ ] Measure TM-score, RMSD, GDT-TS vs. ground truth.
* [ ] Generate CSV and boxplots for paper.

#### **T-12: Runtime Benchmarking**

* [ ] Benchmark runtime on:

  * CPU (MacBook)
  * Azure V100 / A100 / H100
  * RunPod.io / LambdaLabs
* [ ] Track fold time vs. AA length.

#### **T-13: Accuracy Comparison**

* [ ] Compare OdinFold predictions to:

  * AlphaFold2
  * ESMFold
  * RoseTTAFold
* [ ] Match TM-score per target and report delta.

---

### 🧩 BUILD & DEPLOYMENT

#### **T-14: Docker Inference Container**

* [ ] Create production-ready `Dockerfile`.
* [ ] Install all OdinFold dependencies and model weights.
* [ ] Launch API server on container start.

#### **T-15: GPU-Aware Scheduler**

* [ ] Auto-detect available CUDA devices.
* [ ] Queue folding jobs by available memory/compute.
* [ ] Fall back to CPU if no GPU found.

#### **T-16: ONNX Export**

* [ ] Convert OdinFold model to ONNX-lite.
* [ ] Strip unused branches (template, long-range).
* [ ] Validate export with `onnxruntime`.

#### **T-17: CLI Wrapper for OdinFold**

* [ ] Create `odinfold.py` with options:

  * `--sequence`
  * `--input_file`
  * `--mutations`
  * `--out_dir`
* [ ] Pipe to API or local inference engine.

---

### 🌐 BROWSER & LOCAL CLIENTS

#### **T-18: WASM-Ready OdinFold Lite**

* [ ] Quantize model to INT4.
* [ ] Remove templates, MSA, multimer logic.
* [ ] Export as ONNX-lite and compile with WebAssembly.
* [ ] Run CPU-only fold in browser.

#### **T-19: Local OdinFold Desktop Client**

* [ ] Bundle OdinFold engine with PyInstaller.
* [ ] Build .app and .exe installers.
* [ ] Add UI with PDB viewer and sequence input.

---

### 🤖 LLM INTEGRATION

#### **T-20: Prompt-to-Mutation Assistant**

* [ ] Add LLM endpoint to accept natural language (e.g. "increase binding affinity").
* [ ] Suggest mutations via BioLLM + ESM embeddings.
* [ ] Return JSON of top ranked variants.

#### **T-21: Explain Fold Output**

* [ ] Let user ask questions about structure.
* [ ] Parse fold output + LLM to explain helices, binding pockets, pLDDT uncertainty.

#### **T-22: Codex Linter for OdinFold**

* [ ] Add script for agents to check PRs for:

  * Inference regression
  * CUDA kernel coverage
  * Benchmark baseline

---

### 🧯 SECURITY & COMPLIANCE

#### **T-23: Secure OdinFold API**

* [ ] Add OAuth2/JWT authentication.
* [ ] Enforce per-user job limits.
* [ ] Log access for auditability.

#### **T-24: HIPAA / GDPR Compliance**

* [ ] Encrypt PII + fold data in transit + at rest.
* [ ] Add data deletion request logic.
* [ ] Add audit logging for fold requests.

---

Would you like this turned into an editable Notion, GitHub Project board, or Trello format for team management or Codex/Factory agents?

