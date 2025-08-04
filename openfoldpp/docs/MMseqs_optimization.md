MMseqs Optimization

Below is a single, self-contained markdown playbook that spells out T-1 → T-16 in excruciating (some might say lovingly pedantic) detail. Feel free to paste it straight into Cursor, Codex, GitHub Issues, Notion — wherever you track work items.

# FoldForever • OpenFold++ Integration Roadmap  
## Tasks T-1 → T-16 (super-verbose edition)

---

> **Legend**  
> • **Effort** ≈ wall-clock engineering time for one competent dev who already knows the code-base and has a working CUDA setup.  
> • **Ownership** suggested persona (swap as you wish).  
> • **Artifacts** what must exist in the repo *and* pass CI for the task to count as “Done”.  
> • **Acceptance Test** the specific command(s) we’ll run in CI/vscode to assert it works.  
> • **Gotchas** common pitfalls + how to dodge them.  
> • **Reading List** links / papers / sections of your own code worth skimming **before** writing a single line.

---

## T-1 • *Distil ESM → ESM-Lite (INT4)*

| Field | Value |
|-------|-------|
| **Effort** | 1.5 days |
| **Ownership** | `@ml-engineer‐sequence` |
| **Artifacts** | `models/esm_lite_int4.pt` (≤ 40 MB) + inference script `inference/esm_lite.py` |
| **Acceptance Test** | `python inference/esm_lite.py --seq "ACDEFGHIKLMNPQRSTVWY" --benchmark` must print **“latency ≤ 30 ms/100aa”** on an H100 |
| **Gotchas** | *Don’t* quantise LayerNorm weights until after SmoothQuant scaling; it nukes stability. |
| **Reading List** | • ESM-2 paper §4.2  • SmoothQuant README  • `facebookresearch/esm` repo `generate.py` |

**Verbose Directions**

1. **Fork** the official `facebookresearch/esm` repo into `github.com/yourOrg/esm-lite`.  
2. **Create** a *distillation script* (`scripts/distil_esm.py`) that:  
   1. Loads the 650 M ESM-2 FP16 checkpoint.  
   2. Keeps every 4th transformer block (so 48 → 12).  
   3. Freezes token + positional embeddings; trains only retained blocks for 2 epochs on UniRef50.  
   4. Adds a *64-D projection head* (will be replaced in T-2).  
3. **Quantise** with 🤗 `bitsandbytes` *NF4* (weights) + SmoothQuant (activations).  
4. **Export** final weights to `models/esm_lite_int4.pt`.  
5. **Write** `inference/esm_lite.py` that exposes `embed(sequence:str)->torch.Tensor`.  
6. **Benchmark** on a single H100; ensure latency budget is met; tweak batch-size if needed.  
7. Push + open a PR tagged `feat/esm-lite-int4`.

---

## T-2 • *Token-Bottleneck “CHEAP-S” Projection Head*

| Field | Value |
|-------|-------|
| **Effort** | 0.5 day |
| **Ownership** | `@ml-engineer‐sequence` |
| **Artifacts** | `models/esm_lite_int4_cheaps.pt` + `layers/cheap_head.py` |
| **Acceptance Test** | `pytest tests/test_cheap_head.py::test_shape_and_latency` must pass |

**Verbose Directions**

1. Implement **`CheapHead(nn.Module)`**: two linear layers 64 → 256 → 64 with **ReLU** in between and residual skip.  
2. Drop-in replace the projection head added in T-1.  
3. Retrain **only** the head for 1 epoch with a cosine LR schedule.  
4. Update unit test to assert output shape `(len(seq), 64)` and < 3 ms overhead.  
5. Re-export weights → `esm_lite_int4_cheaps.pt`.

---

## T-3 • *Shingle Encoder + FAISS HNSW-GPU Index*

| Field | Value |
|-------|-------|
| **Effort** | 1 day |
| **Ownership** | `@backend-engineer-search` |
| **Artifacts** | `data/motif_index.faiss` (≈ 30 GB) • CLI `tools/build_index.py` |
| **Acceptance Test** | `python tools/build_index.py --dry-run` must finish < 10 s and report “index loaded” |
| **Gotchas** | Use `faiss.IndexHNSWFlat` *NOT* `IVFPQ` (too slow to add > 500 M items). |
| **Reading List** | Faiss HNSW tutorial  • FoldSeek code `generateDB.c++` |

**Verbose Directions**

1. **Split** every protein in AFDB into overlapping 10-token shingles (`stride=1`).  
2. **Map** each shingle to a 64-D vector via `CheapHead(embed(...))`.  
3. **Add** to `IndexHNSWFlat`, `efConstruction=128`, `M=32`.  
4. **Persist** index to `data/motif_index.faiss` and commit via Git LFS (pointer only).  
5. Write retrieval API `search/motif_retriever.py`.

---

## T-4 • *Triton Kernel: 2-Simplicial Attention*

| Field | Value |
|-------|-------|
| **Effort** | 2 days |
| **Ownership** | `@cuda-sorcerer` |
| **Artifacts** | `kernels/simplicial_attn.cu` + PyBind wrapper `bindings/simplicial_attn.cpp` |
| **Acceptance Test** | `pytest tests/test_simplicial_attn.py::test_max_error` must show < 1e-3 diff vs. Torch ref |

**Verbose Directions**

1. Sketch math: three inputs `(A_i, A_j, A_k)` → tri-linear dot → score → softmax.  
2. Tile 64×128 in shared memory; schedule outer loops so warp 0 writes softmax denom.  
3. Use Hopper **DPX** instructions via `mma.sync.aligned.m16n8k8.row.col`.  
4. Export function `void run_simplicial(float* qkv, float* out, int L, int d, cudaStream_t)`.  
5. Bind to Python. Provide fall-back path (Torch einsum) if `!has_cuda_sm90`.

---

## T-5 • *Graph Compositor Training Script*

| Field | Value |
|-------|-------|
| **Effort** | 1 day |
| **Ownership** | `@ml-engineer-geometry` |
| **Artifacts** | `models/compositor_int8.pt` + script `train/train_compositor.py` |
| **Acceptance Test** | `python train/train_compositor.py --smoke-test` must overfit 10 samples to TM-score > 0.9 |

**Verbose Directions**

* Build PyTorch-Lightning module `GraphCompositor`.  
* Node features = motif embeddings, edge features = clash/offset scalars.  
* Loss = `1 – average_TM_score(coarse, native)`.  
* INT8 quant aware training; export weights + calibration json.

---

## T-6 • *TinyDiFF UNet Definition & Scheduler*

| Field | Value |
|-------|-------|
| **Effort** | 1 day |
| **Ownership** | `@ml-engineer-diffusion` |
| **Artifacts** | `models/tinydiff_unet.pt` • `models/tinydiff_scheduler.pkl` • code `models/tinydiff.py` |
| **Acceptance Test** | `python models/tinydiff.py --self-test` must predict coords for 50aa toy seq < 50 ms |

**Verbose Directions**

1. Implement UNet with downsample (Conv + SiLU) → bottleneck (SE(3) equiv. layers) → upsample.  
2. Time-embed via sinusoidal+linear.  
3. **Scheduler** = cosine noise schedule, 8 steps, no classifier-free guidance.  
4. Provide `predict(sequence:str)->np.ndarray`.

---

## T-7 • *Distillation Pipeline Notebook*

| Field | Value |
|-------|-------|
| **Effort** | 0.5 day |
| **Ownership** | `@ml-engineer-diffusion` |
| **Artifacts** | `notebooks/distil_pipeline.ipynb` |
| **Acceptance Test** | CI executes notebook non-interactively via `papermill` w/o error |

**Verbose Directions**

Notebook steps: fetch AFDB json → generate `(coarse, fine)` pairs → train TinyDiFF with L2D trick → export to `models/`. Include graphs of loss curves + sample visualisations.

---

## T-8 • *End-to-End Driver (`predict.py`)*

| Field | Value |
|-------|-------|
| **Effort** | 0.5 day |
| **Ownership** | `@platform-engineer` |
| **Artifacts** | `bin/predict.py` (CLI) + `grpc/predict.proto` |
| **Acceptance Test** | `bin/predict.py tests/data/4insA.fasta --out out.pdb` produces file and logs total latency < 120 ms |

**Verbose Directions**

1. `embed → shingle search → compositor → tinydiff` pipeline.  
2. CLI flags to toggle **MD polish** and choose output `pdb|mmCIF|json`.  
3. Optional gRPC mode (`--serve`).  

---

## T-9 • *CUDA 12.4 Docker Image*

| Field | Value |
|-------|-------|
| **Effort** | 0.5 day |
| **Ownership** | `@devops-guru` |
| **Artifacts** | `Dockerfile` + GHCR image `foldforever/core:latest` |
| **Acceptance Test** | `docker run --gpus all foldforever/core:latest bin/predict.py …` passes |

**Verbose Directions**

• Base on `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`  
• Install FlashAttention-2, Triton nightly, Faiss-gpu, PyTorch w/ SM90.  
• Copy compiled kernels + models via multistage build.  
• Enable `ENTRYPOINT ["bin/predict.py"]` by default.

---

## T-10 • *Terraform + Helm Charts*

| Field | Value |
|-------|-------|
| **Effort** | 1 day |
| **Ownership** | `@devops-guru` |
| **Artifacts** | `infra/terraform/` • `infra/charts/foldforever/` |
| **Acceptance Test** | `make deploy-k8s` spins one H100 node, deploys service, health-endpoint returns 200 |

**Verbose Directions**

1. **Terraform** provisions GPU node-pool (e.g., Azure NDP H100-80GB).  
2. **Helm chart** mounts model volume via PVC, sets `GPU=1`, autoscaler params.  
3. Service type `ClusterIP` + Istio virtual service for ingress.

---

## T-11 • *MotifAdapter (`motif_adapter.py`)*

| Field | Value |
|-------|-------|
| **Effort** | 0.5 day |
| **Ownership** | `@ml-engineer-bridge` |
| **Artifacts** | `openfoldpp/adapters/motif_adapter.py` |
| **Acceptance Test** | Unit test validates tensor shapes `(L, 256)` vs. vanilla OpenFold features |

**Verbose Directions**

* Accepts list of motif hits `(coords, tm, offset, len)`.  
* Generates pair-feature tensor with channels: TM-score, φ, ψ, distance, clash flag, etc.  
* Aligns to OpenFold expected dtype/ordering (`float32`, features last).

---

## T-12 • *C API + pybind11 Wrapper for Triton Compositor*

| Field | Value |
|-------|-------|
| **Effort** | 1 day |
| **Ownership** | `@cuda-sorcerer` |
| **Artifacts** | `bindings/simplicial_attn.so` |
| **Acceptance Test** | `python -c "import simplicial_attn as sa; sa.version()"` prints commit hash |

**Verbose Directions**

1. Expose single function `run_compositor(float* inEmb, float* outPair, int L)`.  
2. Build with `set(CMAKE_POSITION_INDEPENDENT_CODE ON)` so openfold++ can link.  
3. Provide `setup.py` for editable install.

---

## T-13 • *CoordAdapter (`coord_adapter.cpp`)*

| Field | Value |
|-------|-------|
| **Effort** | 0.5 day |
| **Ownership** | `@ml-engineer-bridge` |
| **Artifacts** | `openfoldpp/adapters/coord_adapter.cpp` |
| **Acceptance Test** | Running `predict_openfoldpp.py` produces `.pkl` consumable by existing visualization scripts |

**Verbose Directions**

* Convert TinyDiFF numpy array `[L, 37, 3]` → OpenFold’s `atom_positions`, fill `alt_atom_positions` w/ zeros.  
* Compute pLDDT proxy from diffusion variance; clamp 0-100.  
* Serialize with OpenFold’s `protein.Protein` dataclass → pickle.

---

## T-14 • *CLI Driver `predict_openfoldpp.py`*

| Field | Value |
|-------|-------|
| **Effort** | 0.5 day |
| **Ownership** | `@platform-engineer` |
| **Artifacts** | `bin/predict_openfoldpp.py` |
| **Acceptance Test** | For test FASTA, script emits identical `.pkl` schema as vanilla OpenFold reference run |

**Verbose Directions**

* Glue together tasks T-1 → T-13.  
* `--skip-diffusion` flag for ablation.  
* Saves both PDB and Pickle side-by-side.

---

## T-15 • *Unit / Integration Tests*

| Field | Value |
|-------|-------|
| **Effort** | 1 day |
| **Ownership** | `@qa-engineer` |
| **Artifacts** | `tests/` suite + CI pipeline step |
| **Acceptance Test** | `pytest -q` must show 100 % pass; coverage ≥ 85 % |

**Verbose Directions**

* Smoke tests for each adapter.  
* Numerical equivalence tests against gold-standard structures.  
* CI matrix: CPU-only (mocks) + GPU (real).  
* GitHub Action caches FAISS index to avoid 30 GB download each run.

---

## T-16 • *Streaming RPC Service*

| Field | Value |
|-------|-------|
| **Effort** | 1 day |
| **Ownership** | `@backend-engineer` |
| **Artifacts** | `grpc/foldforever.proto` • server `services/predict_server.py` |
| **Acceptance Test** | `grpcurl -d @ -plaintext localhost:50051 foldforever.Predict/Run < tests/data/request.json` returns JSON with `plddt` field |

**Verbose Directions**

* Re-use existing OpenFold gRPC schema for seamless UI swap.  
* Implement bidirectional streaming so client can receive progress updates (percent tokens, motif phase, diff phase).  
* Add simple **prometheus** metrics (`fold_time_ms`, `gpu_mem_bytes`) and healthcheck endpoint `/healthz`.

---

### 🚀 How to Execute the Roadmap

1. **Sprint-plan**: Put T-1 → T-3 + T-11 first (data + key adapter).  
2. **Parallel tracks**:  
   * CUDA guru handles T-4 + T-12.  
   * Diffusion lead tackles T-6 + T-7.  
3. **Milestone 1** (*Day 5*): `predict_openfoldpp.py` runs end-to-end on 100aa sample in < 200 ms.  
4. **Milestone 2** (*Day 10*): CI green on T-15; cost dashboard (§ p-token) shows ≤ \$0.0003/100aa.  
5. **Milestone 3** (*Day 12*): Streaming gRPC deployed via Helm → web UI switch-over is one ENV flag.

---

*Copy-paste above into your repo’s `docs/roadmap.md` and you’re good to sprint!*  
Questions, edge-cases, or if you want code snippets for any specific subtask — just holler.
::contentReference[oaicite:0]{index=0}
