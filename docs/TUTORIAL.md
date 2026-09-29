# RHMP v2 tutorial

This tutorial goes from a raw mesh to a trained, checked model: meshes → complexes → geometry and stars → placing
data on degrees → the model → readouts → training on your own data (shared and variable meshes) → advanced options →
diagnostics → checkpoints → symmetry and robustness evaluation.  The mathematics behind every step is in
[`docs/MATH.md`](MATH.md); every public function is listed in [`docs/API.md`](API.md); runnable scripts are in
[`examples/`](../examples).

The `python` code blocks below form one script: run them in order from the repository root (they use tiny sizes and
run on a CPU in well under a minute; `tests/test_examples.py::test_tutorial_code_runs` executes them).  Blocks marked
`bash` or `py` are not executed by that test (they need datasets or run long jobs).

Contents: [0 Install](#0-install-and-conventions) · [1 Complexes](#1-from-a-mesh-to-a-cochain-complex) ·
[2 Geometry](#2-geometry-and-reference-stars) · [3 Data on degrees](#3-putting-your-data-on-the-right-degree) ·
[4 Model](#4-building-and-calling-the-model) · [5 Readouts](#5-choosing-a-readout) · [6 Training](#6-training) ·
[7 Length scales](#7-length-scales) · [8 Advanced options](#8-advanced-model-options) ·
[9 Diagnostics](#9-reading-modeldiagnostics) · [10 Checkpoints](#10-saving-and-loading) ·
[11 Robustness](#11-evaluating-symmetry-and-robustness) · [12 Baselines](#12-baselines) ·
[13 Troubleshooting](#13-troubleshooting)

## 0. Install and conventions

```bash
pip install -e ".[viz,dev]"          # from the repository root: torch>=2.4, numpy, scipy (+ matplotlib, pytest)
python -m pytest tests/test_examples.py -q
```

Conventions used everywhere:

* A **complex** `K` has top degree `K.dim` (2 for surfaces, 3 for tetrahedral volumes) and `K.n[k]` cells of degree k.
* A **cochain feature** of degree k is a tensor of shape `(n_k, B, C)`: cells, samples that share the complex,
  channels.  Model inputs are a dict `{k: (n_k, B, F_k)}`.  `rhmp.ops.to_nbc` / `to_bnc` convert from / to the usual
  `(B, n, C)` layout.
* A **block-diagonal batch** of different meshes is one complex with `B = 1`; `K.batch[k]` holds the graph id of every
  k-cell and `K.meta['ptr'][k]` the offsets of each graph.
* **Orientation:** edges are stored `src < dst`; faces of surfaces keep the orientation you give; values of degree
  k >= 1 are cochains relative to these orientations.

## 1. From a mesh to a cochain complex

Any `(positions, cells)` pair works: numpy or torch arrays, 2-D or 3-D positions, any integer dtype.

```python
import os
import sys
import tempfile

import numpy as np
import torch
from scipy.spatial import Delaunay

from rhmp import CochainComplex

rng = np.random.default_rng(0)
torch.manual_seed(0)
pts = np.concatenate([rng.random((196, 2)), [[0, 0], [1, 0], [0, 1], [1, 1]]])
faces = Delaunay(pts).simplices                   # (n2, 3); the orientation you give is kept
K = CochainComplex.from_triangles(pts, faces)     # star='cotan', validate=True
print(K)                                          # CochainComplex(dim=2, n=[200, ..., ...], ...)
assert K.check_d2() == 0.0                        # d_{k+1} d_k = 0 exactly
```

The builders:

| builder | cells | defaults and notes |
|---|---|---|
| `CochainComplex.from_triangles(pos, faces)` | triangles; planar or surfaces in 3-D; boundaries and non-manifold edges allowed | `star='cotan'`; `validate=True` drops degenerate / duplicate faces with a warning (`meta['validation']`, `meta['face_index']` maps kept faces to input rows) |
| `CochainComplex.from_polygons(pos, faces)` | quads, mixed polygons (CW complexes); `faces` as a list of lists or a `-1`-padded array | `star='barycentric'` (`'cotan'` allowed) |
| `CochainComplex.from_tetrahedra(pos, tets)` | tetrahedra, degrees 0..3 | `star='barycentric'`; faces stored with sorted vertices, orientation in `d_2` |
| `CochainComplex.from_grid((nx, ny), h, cell='quad'\|'tri')` | regular grids, vertex id `i*ny + j` | `from_grid((32, 32), 1/31, cell='tri')` is the T1 mesh of the paper |
| `CochainComplex.batch([K1, K2, ...])` | block-diagonal union | class-level call; parts need the same `dim`, position dimension and device |

```python
Kq = CochainComplex.from_grid((8, 8), 1 / 7, cell="quad")                       # quad grid
Kp = CochainComplex.from_polygons([[0, 0], [1, 0], [1, 1], [0, 1], [2, 0], [2, 1]],
                                  [[0, 1, 2, 3], [1, 4, 5, 2]])                   # two quads, CCW
pts3 = np.concatenate([rng.random((52, 3)), [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)]])
Ktet = CochainComplex.from_tetrahedra(pts3, Delaunay(pts3).simplices)           # degrees 0..3
parts = [CochainComplex.from_triangles(p, Delaunay(p).simplices) for p in (rng.random((50, 2)), rng.random((80, 2)))]
Kb = CochainComplex.batch(parts)                                                 # K.batch is the id list instead
print(Kq.n, Kp.n, Ktet.n, Kb.num_graphs, Kb.meta["ptr"][0].tolist())
K64 = K.to(dtype=torch.float64)                    # float tensors (operators, stars, geometry) in float64
```

`K.to("cuda")` moves everything, including the CSR operators.  Cochains are applied with `K.apply_d(k, x)` /
`K.apply_dT(k, x)` (`(n_k, ...) -> (n_{k+1}, ...)` and back); they use the cached transposes in the backward pass.

## 2. Geometry and reference stars

Each complex caches, per degree, a positive reference Hodge star `K.star[k]`, E(n)-invariant descriptors `K.geo[k]`
and boundary flags `K.boundary[k]`.

* `star='cotan'` (triangles): dual areas on vertices, cotan weights on edges (clamped below at `1e-2 x median`, so
  obtuse pairs stay positive), inverse areas on faces.  The default for triangle meshes.
* `star='barycentric'`: ratios of barycentric dual measures to primal measures; always positive; the default for
  polygons and tetrahedra.
* `star='unit'`: all ones (combinatorial operators; an ablation).

The descriptor columns are listed in `K.meta['geo_features']`; scale quantities enter as `log(x / median_x)` over the
mesh, so they are comparable across meshes and resolutions.  The model is built with `K.geo_dims`.

```python
print(K.meta["geo_features"][1])         # ('log_length', 'log_star', 'cotan_weight', 'boundary', ...)
print(K.geo_dims)                        # {0: 5, 1: 6, 2: 5}: the constructor argument of RHMP
print(bool((K.star[1] > 0).all()), int(K.boundary[1].sum()), K.meta["geometry_stats"]["n_star1_clamped"])
```

Other useful metadata: `meta['face_edges']` / `meta['face_edge_signs']` (edges of each face in cyclic order and
their orientation relative to the face), `meta['sizes']`, `meta['ptr']`, and `K.whitney` (precomputed Galerkin
blocks for the tensor metric; triangles and tetrahedra only).

## 3. Putting your data on the right degree

Place every field where its physics lives:

| degree | cells | odd columns (cochains: change sign with the cell orientation) | even columns (orientation-free) |
|---|---|---|---|
| 0 | vertices | (no parity on vertices: all columns go to the vertex MLP) | even columns also feed the vertex metric head |
| 1 | edges | line integrals `∫_e v·dl` of a vector field, edge fluxes (2-D), gauge connections `θ_e` | edge coefficients: `log σ_e`, `log t_e^T Σ t_e` |
| 2 | faces | fluxes through faces (3-D), circulation/vorticity x area (2-D), plaquette variables | face coefficients: `log σ_f`, tensor invariants |
| 3 | tetrahedra | densities x volume | per-cell coefficients |

The columns of `inputs[k]` are ordered `[connection | odd | even]`: `cfg.connection_dims[k]` connection columns first,
`cfg.even_dims[k]` even columns last.  A vector field at the vertices becomes an edge 1-form with
`rhmp.dec.flat` (trapezoidal line integrals); data that come with their own edge list are aligned with
`rhmp.data.edge_alignment` (`cell_alignment` for simplices of higher degree):

```python
from rhmp import dec
from rhmp.data import edge_alignment

x, y = K.pos[:, 0], K.pos[:, 1]
f = torch.sin(3 * x) * torch.cos(2 * y)                  # scalar field on the vertices, (n0,)
v = torch.stack([-y, x], dim=1)                          # vector field on the vertices, (n0, 2)
v_e = dec.flat(K, v)                                     # (n1,) odd: line integrals along the stored edges
e = K.cells[1]                                           # (n1, 2), src < dst
log_sigma_e = torch.sin(4 * 0.5 * (K.pos[e[:, 0], 0] + K.pos[e[:, 1], 0]))    # (n1,) even edge coefficient

# a solver gave the same 1-form on its own edge list (other order, reversed orientation):
order = torch.randperm(K.n[1], generator=torch.Generator().manual_seed(0))
their_edges, their_values = e[order].flip(1), -v_e[order]
perm, sign = edge_alignment(e, their_edges, K.n[0])
assert torch.allclose(their_values[perm] * sign, v_e)   # odd data: reorder and fix the sign (even data: no sign)

B = 4                                                    # samples sharing this mesh
inputs = {0: f.view(-1, 1, 1).expand(-1, B, 1).contiguous(),                                  # (n0, B, 1)
          1: torch.stack([v_e, log_sigma_e], -1).unsqueeze(1).expand(-1, B, 2).contiguous()}   # (n1, B, 2)
```

Normalisation rules (what `rhmp.data` does for the built-in tasks): even columns and vertex fields get the usual
mean/std (`feature_stats`); odd columns and odd targets are only scaled (`scale_stats`, mean 0), because subtracting a
mean from a cochain would break the orientation parity and the gauge symmetry; vector targets get one isotropic scale
(`scale_stats(..., isotropic=True)`) so that normalisation commutes with rotations.

**Gauge connections.**  A U(1) connection `θ` on edges is declared with `connection_dims={1: 1}`; it then enters the
network only through `d_1 θ`, so the model is exactly invariant under `θ -> θ + d_0 λ` (example 04 checks this on a
trained model).  For non-Abelian connections (e.g. SU(2) components) also set `connection_odd=True`, which lets them
enter the ordinary odd path (exact invariance is then given up, as in the paper).

## 4. Building and calling the model

```python
from rhmp import RHMP, RHMPConfig

cfg = RHMPConfig(in_dims={0: 1, 1: 2}, even_dims={1: 1},     # vertex field; edge [odd 1-form | even coefficient]
                 C=32, n_layers=3, readout="node_scalar", out_dim=1)
model = RHMP(cfg, K.geo_dims)
out = model(inputs, K)                        # (n0, B, 1)
hid = model.hidden(inputs, K)                 # {k: (n_k, B, C)}: features before the readout
print(tuple(out.shape), {k: tuple(t.shape) for k, t in hid.items()}, model.num_parameters(), "parameters")
```

The essential fields of `RHMPConfig` (all fields with defaults and descriptions: [API](API.md#class-rhmpconfig)):

| field | meaning |
|---|---|
| `in_dims={k: F_k}` | input width per degree; every degree with `F_k > 0` must be present in `inputs` |
| `even_dims`, `connection_dims` | number of even (last) and connection (first) columns per degree |
| `C`, `n_layers`, `poly_order` | channels, depth, order of the polynomial Hodge filters (default 128, 4, 2) |
| `readout`, `out_dim` | output (section 5) and its width (`node_vector`: number of vector fields) |
| `log_range` | bound `a` of the learned metric: `H / ⋆ ∈ [e^-a, e^a]` (default 2) |
| `scaling`, `tie_metrics`, `cross`, `gate`, `identity_metric` | operator scaling (`'dec'`), tied star/inverse-star metrics, cross-degree terms, radial gate; ablation switches |
| `metric_type`, `layers`, `metric_reference` | tensor metric, resolvent layers, known material offsets (section 8) |
| `amp`, `checkpoint_layers`, `fused` | bf16 autocast on CUDA, activation checkpointing, fused kernels |

Rules the model enforces with explicit errors: the same `B` on all input degrees; the complex on the model's device
and dtype (`K.to(device, dtype=...)`); a complex of the top degree the model was built for.  No parameter depends on
the number of cells, so one model runs on any mesh with the same descriptor widths (all triangle meshes, all
tetrahedral meshes, ...).  `model.lift(inputs, K)` and `model.propagate(x, K, inputs)` expose the two stages.

## 5. Choosing a readout

| readout | output | use it for | exact property |
|---|---|---|---|
| `node_scalar` | `(n_0, B, out)` | scalar fields at vertices (potentials, pressure, temperature) | E(n)-invariant |
| `node_vector` | `(n_0, B, out·D)` | vector fields at vertices (`vector_mode='ls'` least squares, or `'direct'`) | E(n)-equivariant, orientation-convention invariant |
| `cochain:k` | `(n_k, B, out)` | fluxes, circulations, curvatures: oriented quantities on k-cells | odd (flips with the cell orientation) |
| `even:k` | `(n_k, B, out)` | orientation-free cell quantities (magnitudes, energies) | orientation-invariant |
| `grad` | `(n_1, B, out)` | curl-free edge fields `E = d_0 φ` | `d_1 E = 0` exactly |
| `curl` | `(n_2, B, out)` | closed face fields `F = d_1 a` (magnetic flux) | `d_2 F = 0`; gauge invariant in connection mode |
| `div`, `div:k` | `(n_{k-1}, B, out)` | divergences `d_{k-1}^T b`; `div:1` vertex divergence | co-closed; `div:1` sums to 0 per mesh |

All readouts are equivariant under vertex relabelling and independent of the other samples in the batch.

**Pseudo-scalars and pseudo-vectors.**  Vorticity at vertices, a plaquette flux averaged to vertices, or `n × ∇ψ` change
sign under a reflection or a change of the face-orientation convention.  The model is exactly invariant to both, so
it cannot output such fields directly.  Predict an oriented face cochain and apply a fixed map that supplies the
physical orientation (`rhmp.data.OutputMap`; the built-in tasks do this for T1/T3/T6/T7):

```python
from rhmp.tasks.paper import oriented_face_to_node

omap = oriented_face_to_node(K, out_dim=1)        # y_i = mean over faces f ∋ i of sigma_f x_f, sigma_f = +1 for CCW faces
face_model = RHMP(RHMPConfig(in_dims={0: 1, 1: 2}, even_dims={1: 1}, C=16, n_layers=2,
                             readout=omap.model_readout), K.geo_dims)          # 'cochain:2'
vorticity_like = omap(face_model(inputs, K))      # (n0, B, 1) pseudo-scalar field at the vertices
```

## 6. Training

### 6.1 A plain PyTorch loop

The model is an ordinary `nn.Module`; any loop works.

```python
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
target = torch.randn(K.n[0], B, 1)
model.record_diagnostics = False                  # skip statistics bookkeeping in the hot loop
for step in range(3):
    loss = torch.nn.functional.mse_loss(model(inputs, K), target)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
```

For **variable meshes**, build one complex per sample and batch them block-diagonally per step:
`Kb = CochainComplex.batch([K_i ...])`, inputs `torch.cat([x_i ...])[:, None]` (so `B = 1`), and split outputs with
`Kb.meta['ptr'][model.output_degree]`.  `examples/03_train_poisson.py` is a complete loop of this kind.

### 6.2 Your data as a `TaskData` + the library trainer

`rhmp.data.TaskData` is the container the trainer (`rhmp.train.run`), the evaluation scripts and the baselines use.
For a **shared mesh**, `inputs[k]` is `(N, n_k, F_k)` and `target` is `(N, n_t, out_dim)`, both already normalised:

```python
from rhmp.data import Stats, TaskData, feature_stats, scale_stats, sequential_split

N = 40
F0 = torch.randn(N, K.n[0], 1)                          # your vertex inputs      (N, n0, 1)
E1 = torch.randn(N, K.n[1], 2)                          # your edge inputs        (N, n1, 2) = [odd | even]
Y = torch.randn(N, K.n[0], 1)                           # your vertex targets     (N, n0, 1)
split = sequential_split(N)                             # (train, val, test) int64 indices: first 70 %, 15 %, 15 %
tr = split[0]
odd, even = scale_stats(E1[tr][..., :1]), feature_stats(E1[tr][..., 1:])
x_stats = {0: feature_stats(F0[tr]),
           1: Stats(torch.cat([odd.mean, even.mean]), torch.cat([odd.std, even.std]), "odd|even")}
y_stats = feature_stats(Y[tr])
td = TaskData(name="my_task", K=K,
              inputs={0: x_stats[0].normalize(F0), 1: x_stats[1].normalize(E1)},
              target=y_stats.normalize(Y), target_degree=0, target_kind="node_scalar",
              in_dims={0: 1, 1: 2}, even_dims={1: 1}, connection_dims={},
              split=split, x_stats=x_stats, y_stats=y_stats, spatial_dim=2, out_dim=1,
              meta={"batch_size": 8})
print(td.summary()["split"], td.readout)                # [28, 6, 6] node_scalar
```

`target_kind` is `'node_scalar'`, `'node_vector'`, `'cochain'` or `'even'`; with `target_degree` it defines the readout
(`td.readout`, e.g. `'cochain:2'`).  `meta['model_readout']` overrides it (e.g. `'grad'` for a curl-free edge target);
`output_map` adds a fixed orientation map (section 5).  The trainer reads `meta['batch_size']` (samples, or meshes for
variable meshes), `meta['layers']` and `meta['v1_C']` (defaults for `--layers` / `--C`), `meta['r2_std']` (an
additional v1-normalised R2) and `meta['vector_mode']`.  Train with the CLI parser for the defaults:

```python
from rhmp.train import parse_args, run

run_dir = tempfile.mkdtemp(prefix="rhmp_tutorial_")
args = parse_args(["--task", td.name, "--device", "cpu", "--epochs", "2", "--C", "16", "--layers", "2",
                   "--out", run_dir])
result = run(args, task=td)             # Adam + cosine schedule + clipping; model selection on the val split
print(sorted(os.listdir(run_dir)), round(result["test"]["R2"], 4))   # random targets here: R2 is meaningless
```

The run directory holds `config.json`, `history.json` (per epoch: losses, validation metrics, time, memory, metric
condition numbers), `best.pt`, `last.pt` (re-running the same command resumes) and `result.json` (test metrics:
R2, MSE, MAE in normalised units; NRMSE, SSIM, Pearson in physical units; odd cochain targets use an uncentred R2).

For **variable meshes**, `K` is a list of complexes, `inputs` a list of `{k: (n_k_i, F_k)}` and `target` a list of
`(n_t_i, out_dim)`; the trainer batches `meta['batch_size']` meshes block-diagonally (`--bs-meshes` overrides it):

```python
from rhmp.data import mesh_minibatch

Ks = [CochainComplex.from_triangles(p, Delaunay(p).simplices) for p in (rng.random((int(n), 2))
                                                                        for n in rng.integers(40, 90, 20))]
xs = [torch.randn(Ki.n[0], 1) for Ki in Ks]
ys = [torch.randn(Ki.n[0], 1) for Ki in Ks]
split_v = sequential_split(len(Ks))
xst, yst = feature_stats([xs[i] for i in split_v[0].tolist()]), feature_stats([ys[i] for i in split_v[0].tolist()])
td_var = TaskData(name="my_meshes", K=Ks, inputs=[{0: xst.normalize(x_i)} for x_i in xs],
                  target=[yst.normalize(y_i) for y_i in ys], target_degree=0, target_kind="node_scalar",
                  in_dims={0: 1}, even_dims={}, connection_dims={}, split=split_v, x_stats={0: xst},
                  y_stats=yst, spatial_dim=2, out_dim=1, meta={"batch_size": 4})
Kmb, xmb, ymb = mesh_minibatch(td_var.K, td_var.inputs, td_var.target, [0, 1, 2])   # what one step sees
print(Kmb.num_graphs, tuple(xmb[0].shape), tuple(ymb.shape))
```

`examples/04_custom_task.py` (shared mesh, gauge connection, face target) and `examples/05_variable_meshes.py`
(variable meshes) train such tasks end to end.

### 6.3 Registered tasks and the command line

The built-in tasks load datasets from `datasets/` (paper pickles: `python3 datasets/download_v1.py`; new tasks:
`bash scripts/gen_datasets.sh`; see [datasets/README.md](../datasets/README.md)):

```bash
python -m rhmp.train --task HP_k100 --check-data                          # load and print the data summary
python -m rhmp.train --task HP_k100 --epochs 100 --out runs/hp_k100       # variable meshes, block-diagonal batches
python -m rhmp.train --task T6f --native --epochs 100 --out runs/t6f      # paper task, native cochain inputs
python -m rhmp.train --task HP_k1000 --eval-ckpt runs/hp_k100 --out runs/hp_k100_on_k1000   # transfer evaluation
```

```py
from rhmp.tasks import load_task
td = load_task("HP_k100", root="/path/to/repo", native=True, device="cuda")   # root: repository or datasets dir
```

Task names, defaults and the model registry are listed in [API.md](API.md#registries-generated-from-the-code).

## 7. Length scales

RHMP v2 is exactly invariant to the length unit: the stars are normalised per mesh, the descriptors are
`log(x / median)` and the operators are Gershgorin-normalised (`test_numerics.py::test_uniform_scaling_invariance`).
This is what makes models transfer across meshes and resolutions, but it has a consequence: a PDE with an absolute
length scale (a screening length, a diffusion time, a domain size that changes relative to the mesh spacing) is
determined only up to that scale.  Such problems are learnable when (PDE length) / (mesh spacing) is the same for all
samples, e.g. meshes generated at a fixed physical resolution, or when the scale is given to the model as an input
column (for instance the log median edge length of each mesh as a constant vertex column).  Scale-free problems
(harmonic extension, example 05) and problems whose mesh resolution is fixed need nothing special.

## 8. Advanced model options

* **Known material coefficients** (`metric_reference={m: column}`): an even input column holding a log-scale
  coefficient becomes a fixed offset of the metric, `H_m = ⋆_m exp(ref_m) exp(a tanh(MLP))`, so the learned bounded
  correction is relative to the known material (at initialisation the degree-0 up block is the DEC operator of the
  true coefficient).
* **Tensor metric** (`metric_type='tensor'`): per-triangle (per-tetrahedron) SPD material tensors enter through the
  Whitney/Galerkin star.  The default parameterisation `tensor_param='full'` is `σ_f = b_f expm(sum_j s_fj t_j t_j^T)`
  with signed, bounded `s` (every SPD tensor of bounded condition number, including anisotropy misaligned with the
  edges); `tensor_param='cone'` keeps the older `σ_f = b_f I + sum_j a_fj t_j t_j^T`, `a >= 0`, which only produces
  M-matrix stiffness (the same class as a diagonal metric).  At initialisation `d_0^T H_1 d_0` is exactly the P1 FEM
  stiffness matrix.  Triangle and tetrahedral complexes only; `scaling` `'dec'` or `'none'`.
* **Resolvent layers** (`layers=[..., 'resolvent', ...]`): the self term becomes
  `(I + τ_up L_up + τ_dn L_dn)^{-1} x` solved by batched CG (`resolvent_iters`, `resolvent_grad='implicit'`), which
  couples the whole mesh in one layer.
* **Solver mode** (`RHMPConfig.solver_preset(...)`, trainer `--solver-mode`): linear lifting, one `'solve'` layer
  `y = (Δ_H + λ/L^2)^{-1} x` with the un-normalised metric Hodge Laplacian in physical units (Dirichlet / Neumann
  handling, `solve_precond='twolevel'` for large meshes) and a linear readout.  With `material_dims={k: m}` the
  material columns reach only the metric heads, so the metric is the only path from the material to the output; for
  P1 data `-div(σ grad u) = f` this model class contains the exact discrete solution operator, and the learned metric
  is the material (README, selling point 4).
* **Ablations**: `tie_metrics=False`, `scaling='jacobi'|'none'`, `cross=False`, `gate='relu'`,
  `identity_metric=True`, `poly_order=1`, and `star='unit'` when building the complex.
* **Speed and memory**: `amp=True` (bf16 autocast for the dense parts on CUDA; sparse products stay fp32),
  `checkpoint_layers=True` (about half the activation memory for about 25-30 % more time),
  `model.record_diagnostics = False` in timed loops.  `torch.compile(model, backend='aot_eager')` works; Inductor
  needs Python headers on the machine.

```python
cfg_adv = RHMPConfig(in_dims={0: 1, 1: 2}, even_dims={1: 1}, metric_reference={1: 1},   # edge column 1 = log sigma
                     metric_type="tensor", layers=["poly", "resolvent", "poly"], resolvent_iters=16, C=16)
adv = RHMP(cfg_adv, K.geo_dims)
print(tuple(adv(inputs, K).shape), cfg_adv.layer_types)
```

## 9. Reading `model.diagnostics`

After a forward pass with `model.record_diagnostics = True` (the default), `model.diagnostics` is a flat
`{name: float}` dict (reading it synchronises the device):

| key | meaning | what to look for |
|---|---|---|
| `layer{l}.H{m}.mean/std/min/max` | statistics of the learned part `φ = log(H / (⋆ exp(ref)))` of the degree-m metric | spread shows how far the metric moved from the DEC star |
| `layer{l}.H{m}.cond_learned` | max over samples of `max exp(φ) / min exp(φ)` (the learned part only; 1 for a fresh model) | how anisotropic/heterogeneous the learned correction is |
| `layer{l}.H{m}.cond_total` | max over samples of `max H / min H` (includes the reference star and `metric_reference`) | very large values: badly graded meshes or extreme materials |
| `layer{l}.H{m}.clamp_fraction`, `.sat` | fraction of cells with `|tanh| > 0.99` / `|φ| > 0.95 a` | large values: the bound is active; consider a larger `log_range` or a `metric_reference` |
| `layer{l}.H{m}.tensor_logb_mean, _aniso_mean, _aniso_max, _clamp_b, _s_absmax, _clamp_s` | tensor metric: isotropic part, eigenvalue ratio of `σ_f`, saturation (`_a_mean, _a_max, _clamp_a` for `tensor_param='cone'`) | anisotropy the model uses |
| `layer{l}.beta_up{k}`, `beta_down{k}` | mean Gershgorin normaliser of each block | scale of the un-normalised operators |
| `layer{l}.tau_up{k}`, `tau_down{k}`, `cg_res{k}` | resolvent time steps and final relative CG residual | `cg_res` above ~1e-3: raise `resolvent_iters` |
| `layer{l}.solve_res{k}`, `solve_it{k}`, `lam{k}` | solve layers: final relative CG residual, iterations run, shift | `solve_res` above ~1e-4: raise `solve_iters` or use `solve_precond='twolevel'` |

Untied metrics report `layer{l}.H{m}.up_<stat>` / `layer{l}.H{m}.down_<stat>`.  For per-cell fields use `model.metric_fields(inputs, K)`: one dict per layer
with `log_ratio[m]` (`log(H_m / ⋆_m)`, up to a per-sample constant, `(n_m, B)`), `phi[m]` (its learned part),
`tensor[m] = (b, a)` (coordinates of the tensor metric) and `sigma[m]`, the per-cell SPD material tensors
`(n_top, B, D, D)`; read tensors through `sigma` (their action), since `(b, a)` are not identifiable.

```python
model.record_diagnostics = True
with torch.no_grad():
    model(inputs, K)
diag = model.diagnostics
print({k: round(v, 3) for k, v in diag.items() if k.startswith("layer0.H1")})
fields = model.metric_fields(inputs, K)
print(len(fields), tuple(fields[0]["log_ratio"][1].shape))     # layers, (n1, B)
```

`examples/06_metric_inspection.py` prints these per layer and draws the tensor metric as ellipses;
`scripts/metric_recovery.py` correlates learned metrics with the true material of HP/TET tasks.

## 10. Saving and loading

```python
path = os.path.join(run_dir, "model.pt")
torch.save(model.to_checkpoint(), path)                 # {'cfg': cfg.to_dict(), 'geo_dims': ..., 'state_dict': ...}
model2 = RHMP.from_checkpoint(path, map_location="cpu")
best = RHMP.from_checkpoint(os.path.join(run_dir, "best.pt"))              # a trainer run
last = RHMP.from_checkpoint(torch.load(os.path.join(run_dir, "last.pt"), weights_only=False)["model"])
with torch.no_grad():
    assert torch.equal(model2(inputs, K), model(inputs, K))
```

`RHMPConfig.to_dict()` / `from_dict()` round-trip through JSON (string degree keys are accepted); configs written by
older versions load unchanged (new fields have defaults).  `last.pt` also stores the optimiser, scheduler and RNG
states, which is why it is loaded with `weights_only=False` and its `'model'` entry is passed on.

## 11. Evaluating symmetry and robustness

For a run of the trainer on a registered task, `scripts/eval_robustness.py` produces the robustness table
(test split, data re-normalised with the run's statistics):

```bash
python3 scripts/eval_robustness.py runs/t6f --all          # relabel, flip-orient, rotate (+ reflect), gauge, noise
python3 scripts/eval_robustness.py runs/hp_k100 --task HP_k1000 --batch-sizes 1,8 --relabel --rotate
python3 scripts/eval_on.py --run runs/hp_k100 --task HP_qual_sliver      # transfer / mesh-quality sets
```

Rows: `batch=<b>` (identical metrics, max prediction change at round-off), `relabel` (random vertex permutation,
complex rebuilt), `flip-orient` (half of the faces reversed, odd data sign-flipped), `rotate` / `reflect` (E(n)),
`gauge=<s>` (`θ -> θ + d_0 λ`), `noise=<s>` (input noise).  Exactly symmetric transformations must leave R2 unchanged
up to round-off; `noise` and transfer rows measure robustness.  Outputs: `robustness_<task>.json` and `.md`.  The
same transformations work on your own `TaskData`:

```python
from rhmp.robustness import noise_task, random_rotation, transformed_task
from rhmp.train import evaluate

best.eval()
te = td.split[2]
rows = {"base": evaluate(best, td, te, 8, "cpu")["R2"]}
for name, view in {"relabel + flip half the faces": transformed_task(td, te, perm_seed=0, flip_frac=0.5),
                   "rotate + translate": transformed_task(td, te, Q=random_rotation(2, 0), shift=np.full(2, 0.3)),
                   "input noise 0.1": noise_task(td, te, 0.1)}.items():
    rows[name] = evaluate(best, view, view.split[2], 8, "cpu")["R2"]
print({k: round(v, 6) for k, v in rows.items()})         # exact symmetries: identical R2
```

## 12. Baselines

`python -m rhmp.train --task T6f --model mgn` trains any registered model with the same data, loss, metrics and
trainer; parameter-matched models get the v2 model's parameter count unless `--param-budget` is given.
`rhmp.baselines.registry.applicable(name, task)` tells whether a model can run on a task.  The list of models, the
fairness rules and the applicability table are in [`docs/BASELINES.md`](BASELINES.md); the baselines that wrap the
v1 code use its vendored copy in `rhmp.baselines.v1`.

```python
from rhmp.baselines.registry import applicable

print(applicable("mgn", td), applicable("fno", td))     # (True, '') and (False, reason)
```

## 13. Troubleshooting

| symptom | cause and fix |
|---|---|
| `TypeError: 'NoneType' object is not subscriptable` on `K.batch[k]` | `K.batch` is `None` for a single complex; pass `None if K.batch is None else K.batch[k]` to per-graph functions such as `rhmp.dec.cg_solve` |
| `CochainComplex.batch` vs `K.batch` | build batches with the class-level `CochainComplex.batch([...])` (or `rhmp.complex.batch_complexes`); the instance attribute `K.batch` is the per-degree graph-id list |
| `complex is on cpu but the model is on cuda` / dtype mismatch | `K = K.to(device, dtype=...)`; the model checks both |
| `inputs[k] must have shape (n_k, B, F_k)` | layout is `(cells, samples, features)`; use `rhmp.ops.to_nbc` on `(B, n, F)` data; every degree with `in_dims[k] > 0` must be present |
| `metric_type='tensor' needs Whitney blocks` | the tensor metric needs triangle or tetrahedral complexes (no polygons) |
| a model cannot fit a vorticity / flux-at-vertices target | pseudo-scalar target: predict a face cochain and use an `OutputMap` (section 5) |
| accuracy depends on mesh resolution although the physics does not | see section 7 (length scales) |
| `load_task` cannot find data after `pip install` | pass `root=` (the repository or the datasets directory); the default root is the checkout the package lives in |
| `torch.compile` fails with Inductor | missing Python headers / Triton toolchain; use `backend='aot_eager'` or run eagerly |
| building many complexes (HP/TET loading) is very slow on a busy CPU | many small tensor ops oversubscribe the threads; use `torch.set_num_threads(1)` or `OMP_NUM_THREADS=1` for the build (`from_triangles` itself is vectorised) |
