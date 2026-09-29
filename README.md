# RHMP v2: Riemannian Hodge message passing with learned DEC/Whitney metrics

`rhmp` is a PyTorch library for learning physical fields on meshes with the exact structure of discrete exterior
calculus (DEC) and finite-element exterior calculus (FEEC).  A mesh becomes a cochain complex with exact coboundaries
`d_k`; inputs and outputs live on any degree (vertex scalars, edge 1-forms, face fluxes, cell densities); and every
learned propagation step is a Hodge operator built from `d_k` and an SPD **cochain metric**, which starts at the DEC
Hodge star of the mesh and is corrected by a small network that sees only local, invariant quantities.

This is version 2 (v2) of the code of *Learning Discrete Riemannian Metrics for Physical Fields with Cochain-Frame
Equivariance* (Zheng & Allen-Blanchette, arXiv:2608.14556).  Compared with the paper's original code (v1), the
engineering, the algorithms and the task suite are new.  The technical report with every number is
[REPORT.md](REPORT.md), and the per-run tables are in [results/RESULTS.md](results/RESULTS.md).

## Four results

**1. A tensor metric represents anisotropic media; a diagonal metric cannot.**  Per-cell material tensors
`sigma_f = b_f expm(sum_j s_j t_j t_j^T)` enter through the Whitney/Galerkin Hodge star, so `d_0^T H_1(sigma) d_0` *is*
the anisotropic P1 stiffness matrix, and `d_1^T H_2 d_1` is the Nedelec curl-curl operator on tetrahedra.  A diagonal
metric produces only M-matrix operators, and strong or misaligned anisotropy yields operators outside this class
([docs/THEORY.md](docs/THEORY.md)).  In solver mode (a linear FEEC solver whose only nonlinearity is the learned metric,
described below) the tensor metric is more accurate on every anisotropic task:

| task (solver mode; test R2 / zero-shot R2 on the 4x-resolution test set) | diagonal metric | tensor metric |
|---|---:|---:|
| AHP_r100: 2-D Poisson, misaligned anisotropy ratio 100 | 0.832 / 0.699 | **0.964 / 0.958** |
| AHP_r10: ratio 10 | 0.965 / 0.914 | **0.992 / 0.928** |
| ASURF_r100: fibre diffusion on curved closed surfaces | 0.823 / 0.822 | **0.961 / 0.956** |
| ADARCYp_r100: 3-D Darcy pressure on tetrahedra (1500 samples) | 0.919 / 0.866 | **0.977 / 0.970** |
| T5 (paper task; generator with negative-weight edges) | 0.464 (test100) | **1.000** (test100) |

MeshGraphNet reaches 0.665 / 0.400 on AHP_r100 with 92K parameters; the tensor-metric solver has 2.5K.

![diagonal vs tensor metric](docs/figures/fig3_aniso_diag_vs_tensor.png)

**2. Models transfer across meshes and resolutions.**  The model has no per-cell parameters: metrics are functions of
local invariants, operators are normalised per sample, and descriptors are scale-free.  One T6f model (edge U(1)
connection to face flux), trained on a single mesh, keeps R2 0.9961 and 0.9999 on two unseen random meshes, 0.9999 on a
4x finer and 0.9997 on a 4x coarser mesh; the paper's v1 model cannot be evaluated on another mesh at all.  On
heterogeneous Poisson (HP_k100), the solver-mode model with a learned tensor metric keeps **R2 1.0000 on the
4x-resolution test set**, while MeshGraphNet drops from 0.949 to 0.258 (v2 parameter budget) or from 0.927 to 0.4345 (v1
budget).

![HP_k100 resolution transfer](docs/figures/fig1_hp_k100_transfer.png)

**3. The structure is exact, and tested.**  `d_{k+1} d_k = 0` holds exactly on every complex and every batch.
Channel-frame transformations O(C), E(n) transformations, vertex relabelling, face-orientation changes, Abelian gauge
transformations and changes of the batch composition are exact symmetries: under them, the R2 of the trained T6f and T3
models and of the HP_k100 solver-mode model changes by 1e-11 to 4e-9.  Constraint readouts give curl-free (`grad`),
closed (`curl`), co-closed (`div`) and exactly mass-conserving (`mdiv`) outputs, and `||L|| <= 1` holds for every
sample.  The test suite (1771 tests, CPU and CUDA) checks each of these guarantees.

**4. Physical inversion: when the model solves with its metric, the learned metric is the material.**  In solver mode
the model is a linear lifting, one solve layer and a linear readout, that is, a linear FEEC solver whose only
nonlinearity is the metric map; the material columns reach only the metric heads.  The learned per-face tensors then
match the true conductivity **in physical units**: on HP_k100, face `log det(sigma_f)/2` vs `log sigma_f` gives
r = 0.996, slope 1.04 and intercept 0.005, and the edge action gives r = 0.993, with 2.4K parameters and test R2 0.9999.
Freezing the metric at the true material reproduces the FEM solution without any training (R2 0.99999999, verified by
the test suite).  In the general stack (the nonlinear message-passing network), where the material also enters the
features, the metric does not become the material (|r| <= 0.3 for diagonal metrics, with mixed signs): an accurate model
is not automatically an interpretable one.

![metric recovery](docs/figures/fig2_metric_recovery.png)

The report also documents negative results ([REPORT.md](REPORT.md) sections 8 and 9): v2 trails the v1 checkpoint and
GEM-CNN on the T8 airfoil task (0.51 vs 0.62 / 0.70 test R2); SURF does not discriminate between methods (MeshGraphNet
0.9997 vs v2 0.9967, both transfer without loss); tensor metrics on the 3000-sample tetrahedral sets need about 137 GB
of host memory; and the exactly conservative DYN parameterisation (`DYNfix_cons`) is unstable in rollouts.

## What it looks like

![one heterogeneous Poisson test problem at the training resolution and on a 4x finer mesh](docs/figures/field1_hetero_poisson_compact.png)

The figure shows one heterogeneous-Poisson test problem (conductivity contrast 100; the first test sample, not a
selected one) at the training resolution and on a 4x finer mesh that no model saw in training.  The solver-mode model
(2.4K parameters) cannot be told apart from the FEM solution on either mesh; the general stack and MeshGraphNet lose the
solution on the finer mesh.  [REPORT.md section 6.8](REPORT.md#68-qualitative-comparisons) shows the full figure with
inputs and error maps, and the same kind of comparison for anisotropic media, a gauge field on new meshes, a genus-2
surface, rollouts and a vector field on an ellipsoid (drawn by `scripts/make_field_figures.py`).

## Install

```bash
pip install -e .                 # torch >= 2.4, numpy, scipy (validated: Python 3.12, torch 2.10, CPU and CUDA)
pip install -e ".[viz,dev]"      # + matplotlib (figures, example 06) and pytest
python -m pytest tests -q        # CPU; CUDA tests run when a GPU is visible
```

The datasets are not part of the repository: `python3 datasets/download_v1.py` fetches the paper's datasets from OSF,
and `bash scripts/gen_datasets.sh` generates the v2 sets ([datasets/README.md](datasets/README.md)).  The library, the
examples and most tests need no data.

## Quickstart

```python
import numpy as np
import torch
from scipy.spatial import Delaunay
from rhmp import CochainComplex, RHMP, RHMPConfig

pts = np.random.default_rng(0).random((200, 2))                      # any triangle mesh (numpy or torch)
K = CochainComplex.from_triangles(pts, Delaunay(pts).simplices)      # exact d_k, reference stars, geometry
print(K, "max |d1 d0| =", K.check_d2())

cfg = RHMPConfig(in_dims={0: 1, 1: 1}, even_dims={1: 1},             # a vertex field + an even edge coefficient
                 C=32, n_layers=3, readout="node_scalar", out_dim=1)
model = RHMP(cfg, K.geo_dims)

B = 4                                                                # samples sharing the mesh
inputs = {0: torch.randn(K.n[0], B, 1), 1: torch.randn(K.n[1], B, 1)}   # {degree: (n_k, B, F_k)}
y = model(inputs, K)                                                 # (n_0, B, 1)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss = torch.nn.functional.mse_loss(y, torch.randn_like(y))          # one training step on a dummy target
opt.zero_grad(); loss.backward(); opt.step()
print(tuple(y.shape), f"loss {loss.item():.3f}", f"{model.num_parameters()} parameters")
```

A solver-mode model (the model class behind results 1 and 4) is configured in one line:
`RHMPConfig.solver_preset(in_dims={0: 1, 1: 1, 2: 1}, even_dims={1: 1, 2: 1}, material_dims={1: 1, 2: 1},
metric_type="tensor", solve_precond="twolevel")`.  Next steps: [docs/TUTORIAL.md](docs/TUTORIAL.md) and the runnable
[examples/](examples) (each runs on a CPU in under two minutes):

| example | shows |
|---|---|
| `01_build_complex.py` | all mesh builders, validation, stars, descriptors, `d_k` on `(n_k, B, C)` cochains, batches |
| `02_dec_toolkit.py` | Hodge Laplacians, Betti numbers, Hodge decomposition, batched CG, Whitney/Galerkin metrics, sharp/flat |
| `03_train_poisson.py` | heterogeneous Poisson on random meshes (scipy FEM data), block-diagonal training loop, checkpoints |
| `04_custom_task.py` | your data as a `TaskData`, the library trainer, a gauge connection and a face target, exact gauge check |
| `05_variable_meshes.py` | `CochainComplex.batch`, batch independence, per-graph CG, a variable-mesh `TaskData` |
| `06_metric_inspection.py` | `model.diagnostics`, `model.metric_fields`, the tensor metric drawn as ellipses |

## The model in five lines

```
d_{k+1} d_k = 0                                                     exact topology (integer incidence)
H_m = star_m * exp(ref_m) * exp( a * tanh( MLP_m(psi_m) ) )         metric = DEC star x known material x bounded correction
sigma_f = b_f expm( sum_j s_j t_j t_j^T ),  H_1 = sum_f Whitney(sigma_f)   tensor metric (Galerkin star; P1 stiffness at init)
L_up = S^-1/2 d_k^T H_{k+1} d_k S^-1/2 / beta ,  beta = Gershgorin bound  =>  ||L|| <= 1 for every sample
x_k <- x_k + gamma sigmoid(MLP(log(1 + r))) m_k / r ,   m_k = poly(L) x_k + T x_{k+-1}   |  resolvent / solve layers
```

`psi_m` contains only O(C)- and E(n)-invariant quantities.  Besides the polynomial layer, a **resolvent layer**
`(I + tau L)^{-1}` and a **solve layer** `(Delta_H + lambda)^{-1}` in physical units turn the metric into a learned
discrete Green's function; both are batched, O(C)-equivariant conjugate-gradient solves with implicit gradients, and the
solve layer has a two-level preconditioner.  [docs/MATH.md](docs/MATH.md) gives the mathematics as implemented, with the
tests of every guarantee, and [docs/THEORY.md](docs/THEORY.md) covers representability and identifiability.

## Command line

```bash
python -m rhmp.train --task HP_k100 --check-data                          # load a task and print its summary
python -m rhmp.train --task T6f --native --epochs 100 --out runs/t6f      # a paper task with native cochain inputs
python -m rhmp.train --task HP_k100 --native --solver-mode --solve-precond twolevel --solve-iters 128 \
    --material 1:1,2:1 --metric-type tensor --log-range 3 --epochs 30 --out runs/hp_solver   # solver mode, learned tensor
python -m rhmp.train --task AHP_r100 --native --solver-mode --solve-precond twolevel --solve-iters 128 \
    --metric-type tensor --log-range 5 --lr 3e-4 --epochs 60 --out runs/ahp_tensor      # anisotropic media
python -m rhmp.train --task T6f --model mgn --epochs 100                  # a parameter-matched baseline, same protocol
python -m rhmp.train --task HP_k1000 --eval-ckpt runs/hp_solver --out runs/hp_on_k1000  # transfer evaluation
python3 scripts/eval_robustness.py runs/t6f --all                         # symmetry / robustness table of a run
python3 scripts/metric_recovery.py runs/hp_solver                         # learned metric vs true material
```

Every option of `RHMPConfig` has a flag (`--layers poly,resolvent,poly`, `--metric-ref 1:0`, `--latent 1:8`,
`--aux-pde 1.0`, ...).  The full `--help`, the task registry (paper tasks T1-T8, HP / TET Poisson, SURF, DYN, QUAL,
AHP / ASURF / ACURL / ADARCY) and the model registry (the v2 controls `dec_fixed`, `unit_star`, `unit_fixed`, the v1
model `ours_v1`, MeshGraphNet, GCN, GAT, SchNet, EGNN, GaugeEquivCNN, GEM-CNN, MPSN, SCCNN, CW Net, Clifford-SMPN, FNO,
DeepONet) are documented in [docs/API.md](docs/API.md).  The scripts that produce the reported results are in
[scripts/](scripts) (`run_paper_tasks.sh`, `run_new_tasks.sh`, `run_metric_variants.sh`,
`run_material_identification.sh`, `run_aniso.sh`, `run_baselines.sh`, ...); REPORT.md section 11 lists the command of
each table.

## Model zoo

[results/checkpoints/](results/checkpoints) holds fifteen small trained models (4.0 MB in total), each with its training
configuration and test result.  Among them are the 2.4K-parameter HP_k100 solver with a learned tensor metric, the
AHP_r100 and ASURF_r100 tensor-metric solvers, the T5g solver, the T6f, T6, T3, SURF and DYNfix models, and the
general-stack and MeshGraphNet comparison runs of the field figures:

```python
from rhmp import RHMP
model = RHMP.from_checkpoint("results/checkpoints/HP_k100_solver_tensor_learn/best.pt", map_location="cpu")
```

`python -m rhmp.train --task HP_k100 --eval-ckpt results/checkpoints/HP_k100_solver_tensor_learn --out runs/zoo`
re-evaluates a checkpoint on its task; the checkpoint directory carries the data normalisation of its run.

## What changed compared with the paper code (v1)

The core idea is unchanged: fixed topology with `d_{k+1} d_k = 0`, geometry and material learned only through SPD
cochain metrics, O(C) cochain-frame equivariance, E(n) invariance and Abelian gauge invariance.  The implementation is
new ([CHANGELOG.md](CHANGELOG.md) lists every change):

| | v1 (paper code) | v2 (`rhmp`) |
|---|---|---|
| metric | per-cell free parameters (`n_k x 8` bases) predicted from a global mean: tied to one mesh | DEC reference star x bounded correction from local invariants; full-SPD Whitney tensor metric; no per-cell parameters |
| batches | metric built from the batch mean: predictions depend on the other samples | per-sample metrics; outputs independent of the batch |
| operators | unbounded metric, un-normalised operators, LayerNorm (breaks O(C)) | bounded log-metric, per-sample Gershgorin normalisation (`||L|| <= 1`), radial gate |
| layers | polynomial Hodge message passing | + resolvent and solve layers (learned Green's functions), solver mode, material-only routing |
| geometry | one metric variant uses absolute coordinates | intrinsic geometry only; 2-D, surfaces and 3-D |
| inputs / outputs | vertices only; edge fields hand-encoded to vertices | any degree; exact gauge connection mode; exact constraint readouts |
| complexes | triangles; dense `d_1 d_0` check (20 GB and 4.4 s at 100K faces) | triangles, surfaces, polygons, tetrahedra, grids, block-diagonal batches; a 100K-face complex builds and checks in 8 ms |
| speed | COO products, per-sample loops for variable meshes | CSR with cached transposes, fused kernels: 1.65-1.74x faster training, 2.8-2.9x faster inference, -40 % memory; 6-14x for variable meshes |

**Pseudo-scalar targets.**  The v1 targets of T1, T6/T7 and T3 are pseudo-scalars or pseudo-vectors: they flip sign
under a change of the face-orientation convention or under a reflection.  An exactly invariant model cannot output them
from native cochain inputs; v1 fits them because its per-cell parameters memorise the mesh frame.  v2 predicts the field
on its true cochain (edge connection to face flux) and applies a fixed, parameter-free orientation map, which reproduces
the v1 targets exactly (to 7e-8).  With this map, v2 outperforms the re-evaluated v1 checkpoints on T6 (1.000 vs 0.955
test100 R2), T7 (0.879 vs 0.653), T3 (0.996 vs 0.971) and T5 (1.000 in solver mode vs 0.676).

The v1 model and baselines are vendored in `rhmp.baselines.v1` (from
[ContinuumCoder/Riemannian-Hodge-Message-Passing](https://github.com/ContinuumCoder/Riemannian-Hodge-Message-Passing)),
so that `ours_v1`, these baselines and `--eval-v1` work without a copy of the original repository.

## Repository layout

| path | contents |
|---|---|
| `rhmp/` | the package: complexes, DEC toolkit, metrics, layers, model, readouts, trainer, tasks, baselines (`rhmp/baselines/v1`: vendored v1 code) |
| `tests/` | the test suite (`python -m pytest tests -q`) |
| `examples/` | six runnable examples (CPU) |
| `docs/` | [TUTORIAL](docs/TUTORIAL.md), [MATH](docs/MATH.md), [THEORY](docs/THEORY.md), [API](docs/API.md) (generated by `docs/gen_api.py`), [DESIGN](docs/DESIGN.md), [TASK_SUITE](docs/TASK_SUITE.md), [TASK_SUITE_DETAILS](docs/TASK_SUITE_DETAILS.md), [ANISO_TASKS](docs/ANISO_TASKS.md), [BASELINES](docs/BASELINES.md), [DATASETS](docs/DATASETS.md), [REPORT_zh](docs/REPORT_zh.md) (Chinese report), `figures/` and `figures_zh/` |
| `scripts/` | experiment drivers, dataset generation (`gen_datasets.sh`), evaluation (robustness, transfer, metric recovery, rollouts), `collect_results.py`, `make_figures.py`, `make_field_figures.py` |
| `datasets/` | [README](datasets/README.md), the v1 downloader and the v2 generators (`datasets/generators/`) |
| `bench/` | operator / training-step / batching benchmarks and their results |
| `results/` | per-run result files (`results/<group>/<run>/`; see [results/README.md](results/README.md)), `RESULTS.md` and the model zoo (`results/checkpoints/`) |
| `tools/` | helpers for running experiments on a remote GPU host (sync, run, fetch; `HOST=... tools/remote.sh ...`) |
| `shims/` | a pyvista unpickling shim for the T3 pickle on machines without pyvista |

## Citation and licence

Please cite the paper (metadata in [CITATION.cff](CITATION.cff)):

```
Zheng and Allen-Blanchette, "Learning Discrete Riemannian Metrics for Physical Fields with Cochain-Frame
Equivariance", arXiv:2608.14556, 2026.
```

Licence: to be decided by the authors; see [LICENSE_NOTE.md](LICENSE_NOTE.md).
