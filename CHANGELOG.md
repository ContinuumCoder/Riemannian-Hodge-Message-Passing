# Changelog

## 2.0.0 (2026-09-28): first release of the `rhmp` library

Version 2 re-implements the model of *Learning Discrete Riemannian Metrics for Physical Fields with Cochain-Frame
Equivariance* (arXiv:2608.14556; v1 code: https://github.com/ContinuumCoder/Riemannian-Hodge-Message-Passing) as a
library.  The core idea is unchanged (fixed topology with `d_{k+1} d_k = 0`, geometry and material only through SPD
cochain metrics, O(C) cochain-frame equivariance, E(n) invariance, Abelian gauge invariance); the implementation, the
algorithms and the task suite are new.  The results are in [REPORT.md](REPORT.md) and
[results/RESULTS.md](results/RESULTS.md).

### Fixes of v1 defects

| # | v1 | v2 |
|---|---|---|
| F1 | metric = per-cell free parameters (`n_k x 8` bases) predicted from a global mean: tied to one mesh, depends on the cell numbering | metric = DEC reference star x bounded learned correction from local invariants; no per-cell parameters; transfers across meshes and resolutions |
| F2 | metric built from the batch mean: predictions depend on the batch (4e-4 at 1K cells, 7e-3 at 100K) | per-sample metrics; outputs independent of the batch composition (tested to 1e-6) |
| F3 | unbounded metric, un-normalised operators, LayerNorm (breaks O(C)) | bounded log-metric, per-sample Gershgorin normalisation (`||L|| <= 1`), scalar-gain RMS and radial gate |
| F4 | one metric variant uses absolute coordinates (breaks E(n), 2-D only) | intrinsic, scale-free geometry; 2-D, surfaces and 3-D |
| F5 | 1-cochain lifting `sym + asym` is not odd (not invariant to vertex relabelling) | strictly odd lifting for degrees >= 1 |
| F6 | inputs only at vertices; edge fields hand-encoded as `(avg, avg dx, avg dy)` | inputs and outputs on any degree; exact gauge-invariant connection mode |
| F7 | `d_1 d_0 = 0` checked with a dense `n_2 x n_0` matrix (20 GB, 6.5 s at 100K faces) | sparse check; a 100K-face complex builds in about 8-10 ms on a GPU |
| F8 | COO products with run-time transposes, recomputed geometry, permute copies, per-sample loops | CSR with cached transposes, `(n, B, C)` layout, cached geometry, fused kernels, block-diagonal variable-mesh batches, AMP, activation checkpointing |
| F9 | triangle meshes, degrees 0-2 | triangles, surfaces, quads and polygons (CW), tetrahedra (degrees 0-3), grids |

### Library

* `rhmp.complex.CochainComplex`: builders (`from_triangles`, `from_polygons`, `from_tetrahedra`, `from_grid`),
  validation of degenerate and duplicate cells, block-diagonal batches (`CochainComplex.batch` / `from_list`,
  `K.graph_ids(k)`), pinned non-blocking device moves.
* `rhmp.geometry`: E(n)-invariant, scale-free descriptors; cotan / barycentric / unit reference stars.
* `rhmp.ops`: CSR `spmm` with a precomputed-transpose backward, Gershgorin bounds, segment operations.
* `rhmp.model.RHMP` / `RHMPConfig`: normalised metric Hodge blocks, half-operator cross terms, polynomial filters,
  radial gate, native inputs on any degree (`[connection | odd | even]` columns), readouts `node_scalar`,
  `node_vector` (damped least squares), `cochain:k`, `even:k` and the exact constraint readouts `grad`, `curl`, `div`,
  `mdiv`; diagnostics, `metric_fields`, checkpoints (`to_checkpoint` / `from_checkpoint`, also the trainer's
  `best.pt` / `last.pt`).
* `rhmp.train`: the trainer CLI with the v1 protocol, resume, JSON histories, block-diagonal batching with a prefetch
  thread, `--eval-ckpt` (transfer evaluation with the training normalisation) and `--eval-v1` (v1 checkpoints with the
  same split and metrics).
* `rhmp.metrics`: R2 / MSE / MAE / NRMSE / SSIM / Pearson matching the v1 evaluation code to 1e-6; uncentred R2 for
  orientation-odd cochain targets.
* `rhmp.tasks`: the paper tasks in native (cochain) and legacy (v1) form, T1q (quad CW complex), T6f / T7f (face
  targets), heterogeneous and anisotropic Poisson (`HP*`, with zero-shot 4x-resolution test sets), tetrahedral
  Poisson (`TET*`).
* Parameter-free orientation output maps for the pseudo-scalar paper targets (v1 targets reproduced to 7e-8).

### Algorithms

* `rhmp.dec`: metric Hodge Laplacians, Hodge decomposition, harmonic bases, batched O(C)-equivariant CG with unrolled
  or implicit gradients, Whitney / Galerkin metrics, Whitney `sharp` and de Rham `flat`.
* Whitney tensor metric (`metric_type='tensor'`): per-cell SPD material tensors through the Galerkin star (P1 stiffness
  at initialisation, Nedelec curl-curl on tetrahedra).  The default parameterisation `tensor_param='full'`,
  `sigma_f = b_f expm(sum_j s_j t_j t_j^T)` with signed bounded `s`, covers all SPD tensors; the cone parameterisation
  (`'cone'`, `a >= 0`) generates only M-matrices and is kept for comparison.
* Resolvent layers (`layers=[..., 'resolvent', ...]`), `metric_reference` (known material in physical log units),
  `learn_metric=False` (pure DEC prior), `latent_dims` (learned per-cell input columns for shared meshes),
  `abs_scale` (a global length-scale column for scale-dependent PDEs).
* Material identification: `material_dims` (material columns reach only the metric heads), `solve` layers in physical
  units with Dirichlet / Neumann handling and a two-level preconditioner (`solve_precond='twolevel'`),
  `RHMPConfig.solver_preset` / trainer `--solver-mode` (linear lifting, one solve layer, least-squares-initialised
  linear readout), `model.operator_residual` and the operator-identification loss `--aux-pde`.
* Extension suite: SURF (variable closed surfaces, geometry and topology transfer), DYN / DYNfix (advection-diffusion
  rollouts, `rollout_eval`, post-hoc mass projection, exactly conservative variants), HP_qual (mesh-quality shift);
  anisotropy suite AHP / ASURF / ACURL / ADARCY with exact FEEC targets and oracle representability analysis.
* Baselines behind one registry and the same trainer: MeshGraphNet, the v1 baselines (GCN, GAT, SchNet, EGNN,
  GaugeEquivCNN, GEM-CNN, MPSN, SCCNN, CW Net, Clifford-SMPN, FNO, DeepONet) re-executed by vectorised cores on the
  vendored v1 modules (`rhmp.baselines.v1`), the v1 model `ours_v1`, and the v2 controls `dec_fixed`, `unit_star`,
  `unit_fixed`; parameter matching with the rule of v1.
* Evaluation scripts: robustness and symmetry tables, mesh / resolution transfer, metric recovery, rollouts.

### Findings that shape the protocol

* **Pseudo-scalar targets.**  The v1 targets of T1, T6/T7 and T3 flip sign under the face-orientation convention or a
  reflection, so an exactly invariant model cannot output them from native inputs; v2 predicts the true cochain and
  applies a fixed orientation map.  The v1 tables of MPSN / SCCNN / Clifford-SMPN are edge-space R2 (the *v1 edge
  protocol*, the default for those baselines on the paper tasks).
* **Uncentred R2 for odd cochain targets.**  The mean of an orientation-odd quantity depends on the orientation
  convention; centring would make R2 depend on the vertex labels on variable meshes.
* **T5.**  The target is the v1 edge-to-node average of the raw potential differences of an unclamped cotan
  Laplacian with negative hull weights; `T5g` (grad readout + the fixed average) is exactly the generator, and only a
  tensor metric represents it.
* **Length scales.**  v2 is exactly invariant to the length unit; PDEs with an absolute length scale need a fixed
  (PDE length) / (mesh spacing) ratio (SURF is generated that way) or the `abs_scale` column.
* **Conservative rollouts.**  A loss on the mass change `M du` makes rollouts diverge on meshes whose lumped masses
  differ 900:1; the conservative targets use `du`, with the division by the mass inside the model (`mdiv:1`).
* **SCCNN.**  Under the v1 protocol, SCCNN keeps the raw Laplacians of v1 and the v1 initialisation; normalised
  Laplacians need identity-initialised filters, without which the network collapses to a constant.
* **Memory.**  Whitney blocks roughly double the memory of a triangle complex and dominate that of a tetrahedral complex
  (about 45 MB for 2.5K vertices); loaders drop them for diagonal metrics, and data sets above the GPU budget stay on
  the CPU with packed pinned transfers.

### Release

* Repository layout: `rhmp/`, `tests/`, `examples/`, `docs/`, `scripts/`, `datasets/`, `bench/`, `results/`, `tools/`,
  `shims/`; the v1 code needed by the baselines is vendored in `rhmp/baselines/v1/`, with credit to the original
  repository.
* `datasets/download_v1.py` (the paper datasets from OSF), generators in `datasets/generators/`.
* `results/`: the result files of every experiment of the report (`results/<group>/<run>/`, described in
  `results/README.md`), `RESULTS.md` (generated by `scripts/collect_results.py`) and a model zoo of fifteen small
  checkpoints.
* `REPORT.md` (English technical report), `docs/REPORT_zh.md` (Chinese report), figures in `docs/figures/`.
* Qualitative field figures (`scripts/make_field_figures.py`, `docs/figures/field*.png`, REPORT.md section 6.8): true
  field, predictions, a baseline and error maps on the mesh for six tasks (first test sample; the median case for
  AHP_r100), with the three comparison runs (general-stack HP_k100, MeshGraphNet on HP_k100 and AHP_r100) in the model
  zoo.
