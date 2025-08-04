Here’s a **verbose Codex/Cursor task breakdown** for the 4 new OpenFold++ enhancements:

---

# 🧬 OpenFold++ Roadmap Additions (Multimer, Ligand, Docking, TM Confidence)

---

## ✅ Task 1: Add **Multimer Support** to OpenFold++

### 🧠 Goal:

Enable the model to fold **multi-chain protein complexes**, like AlphaFold-Multimer, including correct chain separation and inter-chain attention.

### 🔧 Steps:

1. **Add chain break logic** in the preprocessing pipeline:

   * Concatenate amino acid sequences of all chains.
   * Insert a special `CHAIN_BREAK_TOKEN` and update positional encoding.

2. **Modify positional encodings**:

   * Add a `chain_id` vector of shape `[seq_len]` (e.g., 0 for chain A, 1 for chain B).
   * Offset position encodings by 1000 per chain or use relative encoding.

3. **Update attention bias** in the Evoformer:

   * Apply **masking logic**: prevent attention across chains where appropriate.
   * Add **inter-chain attention heads** to Evoformer if separate heads are desired.

4. **Update the fold runner** to accept a list of sequences or multi-FASTA files.

5. **Validate** on known dimers (e.g., CASP multimer targets or PDB complexes like 1A0I).

6. Benchmark with **DockQ** or TM-score multimer (available in BioPython extensions).

---

## ✅ Task 2: Add **Ligand-Aware Attention Heads**

### 🧠 Goal:

Allow the Evoformer trunk to condition on ligand atoms, enabling better folding around ligand-binding regions.

### 🔧 Steps:

1. **Define ligand input format**:

   * Accept `.mol2`, `.sdf`, or SMILES.
   * Use RDKit or OpenBabel to convert to atom coordinates `[N_atoms, 3]`.

2. **Embed ligand atoms**:

   * Use learned atom-type embeddings + 3D position encoding.

3. **Modify Evoformer attention layers**:

   * Add a **cross-attention block** from residue tokens to ligand tokens.
   * Can also add a self-attention layer over ligand atoms.

4. **Optional**: Add geometric biases (e.g., distance to ligand, pocket depth).

5. **Update dataloader** to support optional ligand inputs.

6. **Visualize** ligand binding pocket alignment in frontend via PyMOL/NGL coloring.

---

## ✅ Task 3: Add **Post-Fold Molecular Docking**

### 🧠 Goal:

Dock a ligand to the predicted folded structure after inference using a real docking engine.

### 🔧 Steps:

1. **Export folded PDB** after inference.

2. **Accept ligand input** (e.g., `.mol2` or SMILES).

3. **Add docking backend**:

   * Use AutoDock Vina, GNINA, or DiffDock.
   * You can Dockerize this as a separate service or CLI call:

     ```bash
     vina --receptor output.pdbqt --ligand ligand.pdbqt --out docked.pdbqt
     ```

4. **Use PyMOL or NGL to display** docked complex in frontend (ligand + receptor).

5. **Cache** docking result with a compound hash in Redis/Postgres.

6. **Benchmark** with docking scores and visual overlays.

---

## ✅ Task 4: Add **TM-Score Confidence Prediction (Pre-Fold)**

### 🧠 Goal:

Predict the likely TM-score of a given sequence *before* folding, useful for ranking batches or early exits.

### 🔧 Steps:

1. **Add a regression head** on top of ESM token encoder:

   * Pass pooled embedding through a small MLP (e.g., 128 → 32 → 1)

2. **Train the head** on sequences with known folds + their TM-scores.

3. **Add CLI flag `--predict-tm`** to run this head standalone.

4. **Expose confidence in JSON API return** or model UI (e.g., “Estimated TM = 0.72”).

5. **Benchmark** with `R²` and MAE (mean absolute error) against ground truth.

6. Optionally expose top-K filter logic in backend (e.g., only fold if est. TM > 0.6).

---

## 🧰 Suggested File Layout

```bash
openfoldpp/
├── multimer/
│   └── attention_utils.py
├── ligand/
│   ├── ligand_encoder.py
│   └── ligand_cross_attention.py
├── docking/
│   └── vina_runner.py
├── confidence/
│   └── tm_predictor.py
└── main.py  # Add flags: --multimer, --ligand, --dock, --predict-tm
```

---

Would you like me to generate the **starter code templates** (scaffolds) for any of these?
We could also dockerize the docking backend separately or run it inside your existing app container.
