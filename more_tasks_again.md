Here are your **OpenFold++ TM-score optimization tasks**, written for Codex/Cursor as verbose, step-by-step items:

---

# 📈 OpenFold++ Optimization Tasks (Improve TM-Score Without Sacrificing Speed)

---

## ✅ Task 1: Quantize the ESM-2-3B Model

**Goal**: Replace ESM-2-650M with ESM-2-3B and quantize it using `bitsandbytes` to reduce memory and maintain speed.

**Steps**:

1. Download the Hugging Face weights for `facebook/esm2_t33_650M_UR50D` and `facebook/esm2_t36_3B_UR50D`.
2. Replace the 650M encoder with the 3B model in the `OpenFold++` token embedding module.
3. Install `bitsandbytes` (8-bit quantization package):

   ```bash
   pip install bitsandbytes accelerate
   ```
4. Convert the ESM-2-3B model to 8-bit using `bnb.nn.Linear8bitLt` wrappers.
5. Update the inference engine to use the quantized encoder.
6. Benchmark TM-score impact vs. original 650M, and record speed overhead (should stay <1.5x).

---

## ✅ Task 2: Add Sparse Attention to Evoformer

**Goal**: Replace full attention with sparse attention patterns (e.g., block or dilated attention) for long-range contacts.

**Steps**:

1. Identify the triangle attention blocks inside the Evoformer trunk.
2. Replace standard attention with a sparse implementation using:

   * `torch-sparse`
   * or FlashAttention-2 with custom blockmask
3. Use a simple pattern first: 25% dense (diagonal), 75% sparse (long-range).
4. Keep dense attention for triangle multiplication and pair-update blocks.
5. Validate numerical stability and folding output.
6. Benchmark attention memory savings and speed gains.

---

## ✅ Task 3: Fine-Tune on Medium and Hard CASP Targets

**Goal**: Improve TM-score specifically for difficult folds without retraining the full model.

**Steps**:

1. Select \~100–200 medium and hard targets from CASP13–CASP14.
2. Extract PDBs and corresponding FASTAs.
3. Prepare a fine-tuning dataset (single-sequence format).
4. Use low-learning rate fine-tuning (e.g. 1e-5) with a frozen encoder and frozen early Evoformer blocks.
5. Fine-tune for 1–2 epochs only on A100 or 3090.
6. Measure TM-score change for hard targets.
7. Save fine-tuned weights in a separate export (don’t overwrite base).

---

## ✅ Task 4: Integrate Fast Post-Fold Relaxation

**Goal**: Improve RMSD (and marginally TM-score) by minimizing bad sidechain angles using OpenMM or FastRelax.

**Steps**:

1. After OpenFold++ finishes folding, export output to PDB format.
2. Set up OpenMM (or similar library) to run minimization:

   * Freeze backbone atoms
   * Minimize sidechains only
3. Add `--relax` flag to the inference CLI
4. Include RMSD before/after relaxation in final report.
5. Benchmark time impact (target: <1s per structure).

---

## ✅ Task 5: Add Optional pLDDT Confidence Estimation

**Goal**: Output confidence score for users (optional, but useful for pharma adoption)

**Steps**:

1. Reintroduce the pLDDT head from AlphaFold/OpenFold into the trunk.
2. Pass Evoformer outputs through a final linear projection to get per-residue confidence scores.
3. Normalize to 0–100 like AlphaFold does.
4. Color pLDDT in the frontend visualization (e.g. blue → red gradient).
5. Add `--confidence` CLI option and expose over API.

---

## ✅ Task 6: Auto-benchmark on GPU

**Goal**: Ensure all future changes retain TM-score ≥ baseline (0.68) and speed ≤ 5s/100aa.

**Steps**:

1. Reuse your Dockerized CASP benchmark script.
2. Add it as a GitHub Action triggered on `main` branch commits.
3. Log TM-score, RMSD, runtime to CSV.
4. Fail the job if:

   * TM-score < 0.66
   * Runtime > 5.5s
   * GPU memory > 8GB
5. Notify success via Discord or Slack webhook.

---

Let me know if you'd like:

* Scripts to launch fine-tuning on LambdaLabs or Paperspace
* OpenMM setup for post-fold minimization
* Quantization code prewritten with `bitsandbytes`

You're now entering “publishable improvement” territory. Let's go.
