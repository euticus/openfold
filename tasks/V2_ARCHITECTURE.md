Alright — here’s a full **`ARCHITECTURE.md`** for OdinFold.
This will give you the blueprint for agents (Cursor, Augment, MCP) to implement without ambiguity, while also documenting why each component exists and how it differs from AlphaFold/OpenFold.

---

# **OdinFold Architecture Specification**

*Version 1.0 – August 2025*

---

## **1. High-Level Overview**

OdinFold is a **novel, high-speed protein folding engine** designed to outperform AlphaFold2 in accuracy while maintaining **\~30× faster inference**.
It merges:

* **Sequence pre-processing acceleration** inspired by MMseqs2-GPU.
* **Attention scaling principles** from LLaMA (grouped query attention, token efficiency).
* **Novel cross-modal feedback loops** between 1D sequence and 3D structure token spaces.
* **C++/CUDA fused kernels** for minimal memory passes.

Unlike OpenFold, OdinFold integrates **geometric priors**, adaptive sparse updates, and hard-example mining into its core.

---

## **2. Data Flow Diagram**

```
┌───────────────────┐
│ Input Sequence (AA)│
└─────────┬─────────┘
          │
          ▼
┌───────────────────────────┐
│ Sequence Embedding Layer  │
│ (ESM-lite, 8-bit quantized)│
└─────────┬─────────────────┘
          │
          │   ┌──────────────────────────────────────┐
          └──▶│ Structure Prior Predictor (Aux Model) │
              │ - Fold class/topology classification │
              │ - Outputs prior tokens + bias vectors│
              └──────────────────────────────────────┘
                        │
          ┌─────────────┘
          ▼
┌───────────────────────────────────────────────────┐
│ Cross-Attention Feedback Layers                    │
│ - Sequence tokens ↔ Structure graph tokens         │
│ - Geometry bias injected from prior tokens         │
└───────────────────────────────────────────────────┘
          │
          ▼
┌───────────────────────────────────────────────────┐
│ Evoformer Trunk (Adaptive Sparse Triangular Updates)│
│ - Prunes low-confidence residue pairs dynamically  │
│ - Redistributes attention budget to high-likelihood│
│   contacts                                         │
└───────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────┐
│ Geometry Projection Heads    │
│ - Distance & angle regression│
│ - Mixed precision (bfloat16→FP32)│
└──────────────────────────────┘
          │
          ▼
┌──────────────────────────────┐
│ Final Coordinate Refinement  │
│ - Invariant Point Attention  │
│ - FP32 for final steps       │
└──────────────────────────────┘
          │
          ▼
┌──────────────────────────────┐
│ Output PDB + Confidence Scores│
└──────────────────────────────┘
```

---

## **3. Core Modules**

### **3.1 Sequence Embedding Layer**

* Based on **ESM-lite** transformer block (pretrained on UniRef50, quantized to 8-bit for speed).
* Outputs `[seq_len, d_model]` embeddings.
* **C++ Optimization:** Batch embedding + token compression fused kernel.

---

### **3.2 Structure Prior Predictor**

* Small 3–5 layer GNN that predicts:

  * **Fold Class** (e.g., all-α, α/β, all-β, small proteins).
  * Coarse topology graph (contact map at 8–12 Å threshold).
* Output is injected into **attention bias matrices** before Evoformer.

**Novelty:** Gives OdinFold a **geometric prior** so it begins folding “knowing” the likely topology.

---

### **3.3 Cross-Attention Feedback Layers**

* Sequence tokens ↔ Structure tokens exchange information **before** final structure projection.
* **Difference from OpenFold:** OpenFold does not feed structure back into sequence embeddings until late-stage IPA.
* Implemented with multi-head cross-attention, bias conditioned on priors.

---

### **3.4 Adaptive Sparse Triangular Updates**

* Triangular multiplicative & self-attention updates **modified**:

  * Confidence scoring per residue pair (p\_contact).
  * Threshold prune low-confidence edges, upsample compute to high-confidence ones.
* **Inspired by LLaMA’s GQA** — reduces noise, increases resolution where needed.

---

### **3.5 Geometry Projection Heads**

* Two parallel projection streams:

  1. **Distance & Angle Regression Head** (predicts pairwise distances, torsion angles).
  2. **Confidence Head** (pLDDT, pTM-score).
* **Mixed Precision:** Early heads in bfloat16, final coordinate output in FP32.

---

### **3.6 Final Coordinate Refinement**

* Invariant Point Attention with FP32 precision.
* Performs last structural clean-up and resolves any chain breaks.

---

## **4. Training & Fine-Tuning Strategy**

### **4.1 Pretraining**

* Stage 1: Pretrain sequence + structure tokenization on **graph language modeling** tasks.
* Stage 2: Integrate priors, train on topology + coarse distance prediction.
* Stage 3: Full structure regression.

### **4.2 Hard Example Mining**

* After initial convergence, run inference on validation set.
* Identify:

  * High-confidence, low-TM-score cases.
  * Low-confidence, high-TM-score cases.
* Retrain on these edge cases until convergence plateaus.

---

## **5. Deployment & Optimization**

* **C++/CUDA Kernels:**

  * Fused embedding + Evoformer updates in one pass.
  * GPU shared-memory utilization for pair updates.
* **WASM Build:**

  * Lightweight INT4 model for ≤ 200 AA in-browser folding.
* **Cost Target:** <\$0.001 per 100 AA fold on A100-class GPUs.

---

## **6. Benchmark Targets**

| Metric         | AlphaFold2 | OdinFold Target |
| -------------- | ---------- | --------------- |
| TM-score avg   | 0.788      | ≥ 0.800         |
| RMSD (Å)       | Baseline   | ≤ Baseline -10% |
| Speedup        | 1×         | ≥ 25×           |
| MSA Dependency | Required   | None            |

---

## **7. Next Steps**

* Implement **Structure Prior Predictor** module.
* Add **feedback layers** into Evoformer trunk.
* Write **adaptive sparse attention** C++ kernel.
* Integrate **hard example miner** into training loop.

---

I can now also make you a **visual PDF with architecture diagrams, module breakdown, and implementation notes** so it’s easier to hand to developers or investors.
Do you want me to prepare that as well?
