Absolutely — let’s turn the remaining seven items into full-blown development tasks. These are the **last layers** you need to fully sever from OpenFold and truly own OpenFold++.

---

## 🧱 FINAL PHASE: Fully Own OpenFold++

These are **T-26 → T-32**, continuing your cursor/Codex format.

---

### 🔧 **T-26: Rebuild Input Pipeline (`foldpp_input_pipeline.py`)**

**Goal:** Fully replace OpenFold’s input logic with a minimal, PLM-only pipeline.

```md
- [ ] Write new `foldpp_input_pipeline.py`
- [ ] Accept input: `.fasta` or raw sequence string
- [ ] Tokenize using ESM2 tokenizer (via fair-esm or transformers)
- [ ] Add:
    - Positional encodings
    - Mask generation
    - Dummy residue metadata if needed
- [ ] Output: batched tensor to feed into `EvoLite`
- [ ] Remove `data/mmcif_parsing`, `templates`, `msa_pipeline` from project
```

---

### 🧠 **T-27: Build Custom Model Wrapper (`FoldEngine`)**

**Goal:** Drop OpenFold’s `model.py` and define your own model class that composes new modules (EvoLite, FastIPA, etc.)

```md
- [ ] Create `models/fold_engine.py`
- [ ] Define `FoldEngine(nn.Module)`:
    - Init: EvoLite, FastIPA, LigandHead, MutationHead
    - Forward:
        - Take tokenized input
        - Output PDB coordinates, ΔΔG, TM-confidence, etc.
- [ ] Modularize so each block can be swapped easily
- [ ] Unit test with dummy batch
- [ ] Remove reliance on `openfold/model.py`
```

---

### 🧪 **T-28: Write Custom Loss Module (`foldpp_loss.py`)**

**Goal:** Eliminate OpenFold’s loss logic. Define your own simplified loss class that supports:

* TM-score proxy
* ΔΔG regression
* Ligand contact confidence

```md
- [ ] Create `loss/foldpp_loss.py`
- [ ] Implement:
    - `TMConfidenceLoss` (L2 loss to real TM-score or pseudo-TM predictor)
    - `DeltaGRegressorLoss` (MSE)
    - Optional: `BindingConfidenceLoss` (if ligand present)
- [ ] Combine into `FoldLossWrapper` with weighted sum
- [ ] Validate gradients flow correctly through heads
- [ ] Remove all usage of OpenFold’s `loss.py`
```

---

### 💾 **T-29: Rebuild Output Renderer (`FoldWriter`)**

**Goal:** Write your own code to convert folded coordinates into `.pdb` or `.cif` — with mutation diff overlays.

```md
- [ ] Create `utils/fold_writer.py`
- [ ] Define `FoldWriter.write(coords, filename, options)`
- [ ] Support:
    - WT or mutated sequence
    - pLDDT coloring
    - Ligand contact annotation
- [ ] Add diff-overlay logic:
    - Input: WT vs. Mutated structure
    - Output: RMSD, rotamer shift, bond break alerts
- [ ] Remove OpenFold’s PDB output utils
```

---

### 🧪 **T-30: Create Minimal Dataset Loader (`FoldppDataset`)**

**Goal:** Define your own dataset format (e.g. CSV or JSONL with seq, structure, mutation, ΔΔG, ligand) and write a minimal PyTorch dataset.

```md
- [ ] Create `data/foldpp_dataset.py`
- [ ] Format:
    - JSONL: {sequence, ddg, ligand_smiles, label_structure}
    - OR CSV: sequence, ddg, ligand, pdb_path
- [ ] Write `FoldppDataset(Dataset)` class
    - Tokenizes sequence
    - Loads optional PDB or ligand
- [ ] Add DataLoader with batching
- [ ] Remove OpenFold’s dataset + template pipeline
```

---

### 🧬 **T-31: Run Custom Fine-tuning on Public Data**

**Goal:** Fine-tune your model (even for 1–3 epochs) on a custom mutation dataset (e.g. ProTherm)

```md
- [ ] Download mutation dataset: ProTherm or FireProtDB
- [ ] Format as JSONL for `FoldppDataset`
- [ ] Train using:
    - `FoldEngine` model
    - `FoldppLoss`
    - Mixed precision + logging
- [ ] Save fine-tuned checkpoint: `foldpp-ddg-v1.pt`
- [ ] Benchmark ΔΔG and TM-score confidence accuracy
- [ ] Push to `weights/` directory
```

---

### 🚀 **T-32: Fully Purge OpenFold Codebase**

**Goal:** Remove all unused OpenFold modules and confirm independence.

```md
- [ ] Remove:
    - `openfold/model.py`
    - `loss.py`
    - `utils/pdb.py`
    - `template`, `msa_pipeline`, `template_featurizer`
    - `training/training_loop.py` (replace with your own)
- [ ] Update README with dependency map:
    - ✅ Owned modules
    - 🚫 Removed OpenFold modules
- [ ] Confirm: repo still folds proteins, runs test suite, and benchmarks clean
```

---

## ✅ At This Point…

You will have:

* Written your own **model**
* Built your own **pipeline**
* Trained on your own **data**
* Saved your own **outputs**
* Benchmarked your own **results**

You can delete every OpenFold file and still run the full loop.

Let me know if you want a printable version or GitHub issue templates for each.
