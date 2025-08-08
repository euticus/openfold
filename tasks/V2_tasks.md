Alright — I’ll give you a **full Augment Code–ready task plan** for building these OdinFold upgrades, and I’ll be *overly verbose* so there’s zero ambiguity.

Since you already have an OdinFold engine (essentially an optimized OpenFold++ baseline), I’m going to structure this so you **incrementally refactor and extend** rather than rewrite from scratch. This will let you:

* Keep your current **29.5× speed advantage** intact.
* Gradually replace modules with new C++ kernels and novel architecture pieces.
* Preserve working inference pipeline while testing new ideas in isolation.

---

# **OdinFold v2 Upgrade Plan – Augment Code Task List**

---

## **Phase 0 – Preparation & Baseline**

*(Goal: Lock down current OdinFold as a reproducible baseline so new changes don’t regress speed/accuracy)*

**T0.1 – Freeze Current OdinFold Baseline**

* Create a new `baseline` branch in Git for the current OdinFold engine.
* Run full CASP14 benchmark on current engine, saving:

  * **Per-target TM-scores**
  * **Runtimes**
  * **pLDDT distributions**
* Export all results to `/benchmarks/baseline_results.json`.

**T0.2 – Add Benchmark Harness**

* Create `/benchmarks/run_benchmark.py` that:

  * Accepts a list of targets.
  * Runs fold, records TM-score, RMSD, runtime, and memory usage.
  * Outputs both CSV and JSON.
* This script must run **before and after** every major code change.

---

## **Phase 1 – Structure Prior Predictor**

*(Goal: Inject coarse topology information before Evoformer processing)*

**T1.1 – Create Auxiliary Topology Model**

* Implement `/modules/prior_predictor.py`:

  * Accepts raw AA sequence embeddings.
  * Uses **3–5 layer GNN** to predict:

    * Fold class (categorical).
    * Coarse contact map (binary at 8–12 Å).
  * Outputs:

    * `prior_tokens` → `[seq_len, prior_dim]` tensor.
    * `attention_bias` → `[seq_len, seq_len]` tensor.

**T1.2 – Integrate Priors into Evoformer Input**

* Modify `OdinFoldModel.forward()`:

  * After initial embedding layer, run `prior_predictor()`.
  * Add `prior_tokens` to sequence embeddings (via residual add).
  * Add `attention_bias` to Evoformer attention bias.

**T1.3 – Add Unit Tests**

* Write `/tests/test_prior_predictor.py`:

  * Test that priors change Evoformer outputs.
  * Test shape correctness of tokens and bias.

---

## **Phase 2 – Cross-Attention Feedback Layers**

*(Goal: Add bi-directional information flow between sequence and structure token streams)*

**T2.1 – Create Feedback Layer Module**

* Implement `/modules/feedback_layer.py`:

  * Takes:

    * Sequence token stream: `[seq_len, d_model]`.
    * Structure token stream: `[n_nodes, d_model]`.
  * Runs **multi-head cross-attention** both ways:

    * Sequence → Structure (query seq, key/value struct).
    * Structure → Sequence (query struct, key/value seq).
  * Returns updated streams.

**T2.2 – Insert Feedback Layers into Evoformer**

* After **every 4 Evoformer blocks**:

  * Extract intermediate structural representation from pair activations.
  * Convert to structure tokens via pooling over edges.
  * Pass both streams to `feedback_layer()`.
  * Feed updated streams back into Evoformer.

**T2.3 – Benchmark Impact**

* Run CASP14 benchmark with feedback layers enabled.
* Compare TM-score gain vs baseline.

---

## **Phase 3 – Adaptive Sparse Triangular Updates**

*(Goal: Focus compute on high-confidence residue pairs)*

**T3.1 – Implement Confidence Scoring**

* Modify triangular update module:

  * Before attention, compute per-pair confidence score `p_contact` from pair embeddings.
  * Threshold: keep top `K%` per row/col.

**T3.2 – Implement Sparse Attention in C++**

* Create `/kernels/sparse_triangle.cu`:

  * Accepts mask of allowed edges.
  * Runs attention only on those edges.
  * Supports dynamic sparsity per batch.

**T3.3 – Integrate Into Evoformer**

* Replace existing Python triangular update calls with C++ sparse version.
* Add fallback to dense mode for debugging.

---

## **Phase 4 – LLaMA-Style Geometry Tokenization**

*(Goal: Pretrain on graph language modeling tasks before structure regression)*

**T4.1 – Tokenizer Implementation**

* Create `/modules/geometry_tokenizer.py`:

  * Converts structure graphs (distances, torsion angles) into discrete tokens.
  * Uses learnable codebook (e.g., k-means on continuous values).

**T4.2 – Pretraining Script**

* `/train/pretrain_graph_lm.py`:

  * Train OdinFold backbone to predict next geometry token.
  * Use UniRef50 subset with known structures.

**T4.3 – Integrate Pretrained Weights**

* After pretraining, load weights into main OdinFold backbone before fine-tuning.

---

## **Phase 5 – Active Hard Example Mining**

*(Goal: Close gaps in generalization by retraining on failure cases)*

**T5.1 – Miner Script**

* `/tools/hard_example_miner.py`:

  * Runs inference on validation set.
  * Selects sequences where:

    * pLDDT > 0.7 AND TM-score < 0.6.
    * OR pLDDT < 0.4 AND TM-score > 0.8.
  * Saves to `hard_cases.json`.

**T5.2 – Fine-Tuning Loop**

* `/train/fine_tune_hard_cases.py`:

  * Loads both standard training set + `hard_cases.json`.
  * Oversamples hard cases 3× in training batches.

---

## **Phase 6 – Final Coordinate Precision Upgrade**

*(Goal: Preserve precision without sacrificing speed)*

**T6.1 – Mixed Precision Policy**

* Use bfloat16 for:

  * Sequence embedding updates.
  * Early Evoformer blocks.
* Use FP32 for:

  * Final 2 Evoformer blocks.
  * IPA coordinate projection.

**T6.2 – C++ Fused Evoformer Kernel**

* `/kernels/fused_evoformer.cu`:

  * Combine attention + feedforward + pair update in single GPU pass.
  * Avoid intermediate writes to global memory.

---

## **Phase 7 – WASM Build for ≤ 200 AA**

*(Goal: Client-side folding for small proteins)*

**T7.1 – Model Export**

* Export small OdinFold variant (≤ 200 AA, INT4 quantized) to ONNX-lite.

**T7.2 – WebAssembly Compile**

* Compile ONNX model to WASM with WebGPU backend.
* Host demo in `/web/demo.html`.

---

## **Recommendation:**

Start **by building on top of your current OdinFold** (not rewriting) for 3 reasons:

1. You already have a **stable, benchmarked baseline** with big speed gains.
2. Incremental module swaps let you **measure each idea’s impact** without breaking the whole engine.
3. If a change regresses accuracy or speed, you can roll it back without losing months of work.

---

If you want, I can also turn this into a **GitHub Projects Kanban board export** so Augment Code can automatically manage and check off tasks as it completes them.
Do you want me to prepare that next?
