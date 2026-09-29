# RHMP v2 — engineering redesign of Riemannian Hodge Message Passing

Status: the design specification of the `rhmp/` package (v2), written before and during the implementation.  The
mathematics as finally implemented, with the tests of every guarantee, is [MATH.md](MATH.md); the results are in
[../REPORT.md](../REPORT.md).  The original implementation (v1) is vendored in `rhmp/baselines/v1/` for comparison.

Paper: "Learning Discrete Riemannian Metrics for Physical Fields with Cochain-Frame Equivariance"
(Zheng & Allen-Blanchette, arXiv:2608.14556). v1 code: https://github.com/ContinuumCoder/Riemannian-Hodge-Message-Passing
(vendored: `rhmp/baselines/v1/gauge_hodge_mp/`).

## 0. What we keep (the insight) and what we fix

Keep, exactly:
1. **Topology is fixed and exact.** Coboundaries `d_k` come from oriented incidence; `d_{k+1} d_k = 0` holds
   to machine precision for every complex we build (triangles, quads/polygons, tetrahedra, block-diagonal batches).
2. **Geometry is learned only through SPD cochain metrics** `H_k` (diagonal, one value per k-cell): every
   learnable propagation operator is a metric-weighted Hodge block `d_k^T H_{k+1} d_k` or
   `d_{k-1} H_{k-1} d_{k-1}^T`, plus metric-weighted cross-degree transport `d_k^T H_{k+1} x_{k+1}`, `d_{k-1} H_{k-1} x_{k-1}`.
3. **Cochain-frame equivariance O(C)**: metrics are predicted only from O(C)-invariant statistics; the only
   nonlinearity inside the MP stack is the norm gate `g(||m||) m`; normalisation is RMSNorm with a *scalar* gain.
4. **E(n)**: all geometric inputs are E(n)-invariant; scalar readouts are invariant, vector readouts equivariant.
5. **Abelian gauge invariance** `F = d_1 A` from `d_1 d_0 = 0`.

Fix (v1 defects found by inspection + measurement, see `bench/v1_timing.py`):

| # | v1 defect | v2 fix |
|---|-----------|--------|
| F1 | Metric `softplus(basis @ mlp(global_mean))`: per-cell free parameters (`n_k x 8`), tied to one mesh, depends on cell ordering, no geometry | Metric = **reference DEC Hodge star x bounded learned correction** from *local* invariants; no per-cell parameters; transfers across meshes/resolutions |
| F2 | `forward_batch` builds the metric from the **batch-mean** features: prediction of sample i depends on the other samples in the batch (rel. diff 4e-4 at 1K cells, 7e-3 at 100K) | Per-sample metrics `H: (n_k, B)`; outputs are bitwise independent of batch composition |
| F3 | Unbounded metric range and un-normalised operators; `nn.LayerNorm` (mean-subtraction + per-channel affine) breaks O(C) | Bounded log-metric (`H/star in [e^-a, e^a]`), **per-sample Gershgorin normalisation** (`||L_hat|| <= 1` guaranteed), scalar-gain RMSNorm |
| F4 | `local_rich` metric uses absolute coordinates `pos[:, :2]` (breaks E(n), 2-D only) | only invariant local geometry; 2-D and 3-D |
| F5 | 1-cochain lifting is `sym + asym` -> not orientation-odd -> the network is **not** invariant to vertex relabelling | strictly **odd** lifting for k>=1 (odd linear part x even gate); permutation/orientation tests |
| F6 | Inputs only at nodes; edge fields (connections, fluxes) hand-encoded to nodes `(avg, avg*dx, avg*dy)` | native inputs/outputs on **any degree**; optional exact gauge-invariant "connection" mode |
| F7 | `from_triangulation` checks `d1 @ d0` by densifying an `n2 x n0` matrix (20 GB at 100K cells, 6.5 s) | sparse check, O(nnz) |
| F8 | COO spmm with `.t()`/`.coalesce()` per call, geometry recomputed per forward, `(B,n,C)` permute copies, Python loop in vector readout, per-sample loop for variable meshes, no AMP/TF32 | CSR `d`, `d^T`, `|d|` precomputed; `(n, B, C)` layout (zero-copy spmm view); cached geometry; vectorised readouts; block-diagonal mesh batching; AMP for dense parts, fp32 spmm; optional `torch.compile`, activation checkpointing |
| F9 | Only triangle meshes, degrees 0..2 | general cochain complexes: triangles, quads/polygons (CW), tetrahedra (degrees 0..3), grids, batches |

v1 reference numbers (RTX PRO 6000 Blackwell, `bench/v1_timing.py`):
T6 (n0=1024, C=128, L=4, B=64): train step 104.6 ms, infer 51.0 ms, peak 7.17 GB, 0.43M params.
T7 (C=160): 141.4 ms, 8.89 GB. T6_100K (C=16, L=3, B=2): 25.9 ms/step, complex build 6.5 s.

## 1. Package layout

```
rhmp/
  __init__.py        exports RHMP, RHMPConfig, CochainComplex (lazy), __version__
  complex.py         CochainComplex: builders, validation, block-diagonal batching, sparse operators
  geometry.py        E(n)-invariant per-cell descriptors, reference Hodge stars, mesh quality
  ops.py             spmm on (n,B,C), Gershgorin bounds, segment ops, power iteration
  dec.py             DEC toolkit: metric Hodge Laplacians, Hodge decomposition, batched CG, Whitney metrics (§9.3)
  metric.py          MetricHead (bounded log-metric on top of the reference star)
  layers.py          RHMPLayer (normalised metric Hodge operators, cross terms, poly filter, norm gate), tensor metric,
                     resolvent and solve layers
  lifting.py         CochainLifting (inputs on any degree, odd/even, connection mode)
  readout.py         node_scalar, node_vector (LS), cochain:k (odd), even:k, grad, curl, div, mdiv
  model.py           RHMPConfig dataclass + RHMP module
  metrics.py         R2 / SSIM / Pearson / NRMSE in torch (bit-compatible with the v1 evaluation code)
  data.py            GPU-resident dataset containers, splits, normalisation, variable-mesh batching
  train.py           generic trainer CLI (AMP, resume, JSON history, timing, peak memory)
  robustness.py      symmetry / robustness transforms of a TaskData
  tasks/             adapters for T1,T2,T3,T5,T6,T7,T8 (+ native-cochain variants), HP/TET, the extension and
                     anisotropy suites
  baselines/         registry, MeshGraphNet, v2 controls, vectorised cores around the vendored v1 code (baselines/v1)
tests/               pytest suite
bench/               ops_bench.py, step_bench.py, batch_bench.py, v1_timing.py
datasets/generators/ generators for new tasks (outputs go to datasets/v2/*.pt, git-ignored)
tools/sync.sh, tools/remote.sh, tools/run_bg.sh   (remote workflow, see §8)
```

## 2. Conventions

* **Layout:** every cochain feature tensor is `(n_k, B, C)`, contiguous, float32 (bf16 only inside autocast
  regions for dense MLPs). `B` = number of samples sharing the same complex. For a block-diagonal batch of
  different meshes, `B = 1` and cells of all meshes are concatenated (`K.batch[k]` gives the sample id of each cell).
  Helper: `rhmp.ops.to_nbc(x_bnc)` / `to_bnc(x_nbc)` for user-facing `(B, n, C)` tensors.
* **Orientation:** edges are stored canonically `src < dst` (as in v1). Faces keep the orientation given by the
  input (physical orientation for surfaces); tets keep input order. Features on degree `k >= 1` are
  *orientation-odd* (true cochains); everything even (lengths, areas, stars, even raw inputs) enters only through
  metrics, gates and readouts.
* **Sparse:** `d[k]` is CSR `(n_{k+1}, n_k)` float32 on the complex's device; `dT[k]` is a separately built CSR
  of the transpose; `d_abs[k]`, `dT_abs[k]` are the CSR of `|d_k|`. Never call `.t()`/`.coalesce()` in a forward.
* **Determinism:** the output for sample `i` must not depend on other samples in the batch (test enforced, 1e-6).
* **Devices:** tests run on CPU (and on CUDA when available); every module must work on both.
* **No absolute coordinates** anywhere inside the model, except in `NodeVectorReadout` through edge vectors.
* Python 3.12, torch 2.10 (server). Type hints, docstrings, no PyG/torch_scatter dependency.

## 3. Mathematics of v2

### 3.1 Complex, reference metric, geometry
For a complex with top degree `K` (2 for surfaces, 3 for volumes):
* `d_k : C^k -> C^{k+1}`, `d_{k+1} d_k = 0`.
* Reference diagonal Hodge star `star[k] > 0` per degree (`geometry.py`):
  * triangles, `star='cotan'` (default): `star0` = barycentric dual area (1/3 of incident triangle areas, always > 0);
    `star1` = `(cot a + cot b)/2` (one term at boundary edges) clamped below at `kappa * median(star1)` (`kappa=1e-2`)
    so that it stays > 0 on obtuse pairs; `star2 = 1/area`.
  * `star='barycentric'`: `star1 = |dual edge|/|edge|` with barycentric dual (always > 0); use for tets
    (`star0` dual volume, `star1 = |dual face|/|edge|`, `star2 = |dual edge|/|face|`, `star3 = 1/vol`, barycentric duals).
  * `star='unit'`: all ones (combinatorial; reproduces v1's implicit prior).
  * quads/polygons: barycentric duals with polygon areas (fan triangulation for area/centroid).
* `geo[k]`: `(n_k, G_k)` E(n)-invariant descriptors, standardised (log for positive scale quantities,
  then per-complex z-scoring is **not** allowed — use fixed transforms so that values are comparable across meshes:
  `log(x / median_x_of_this_complex)` is allowed because it is a per-mesh scale normalisation that is itself
  E(n)-invariant and mesh-transferable). Minimum set:
  * k=0: log dual area, log mean incident edge length, valence, boundary flag, angle defect (surfaces; 0 for 2-D/vol)
  * k=1: log length, log star1, clamped raw cotan weight, boundary flag, number of cofaces, cos dihedral angle (surfaces, 1 for flat)
  * k=2: log area, log star2, min interior angle, log aspect ratio (circumradius/inradius), boundary flag
  * k=3: log volume, quality (inradius/circumradius proxy), boundary flag
  `K.geo_dim(k)` exposes `G_k`.
* `boundary[k]`: bool per cell (edges with exactly one coface, their vertices; faces with a boundary edge; ...).

### 3.2 Bounded log-metric (metric.py)
For degree `m`, per sample and per cell:
```
H_m = star[m] * exp( a * tanh( MLP_m(psi_m) ) ),   a = log_range (default 2.0  ->  H/star in [e^-2, e^2])
```
* `psi_m` (O(C)-invariant and E(n)-invariant), concatenated along the last dim, shape `(n_m, B, P_m)`:
  `geo[m]` (broadcast over B), even raw inputs on degree m (if any, e.g. conductivity), and feature invariants
  `log1p(||x_m||^2 / C)`, `log1p(||d_{m-1} x_{m-1}||^2 / C)` (if m>=1), `<x_m, d_{m-1} x_{m-1}> / C` (if m>=1),
  `log1p(||d_m^T x_{m+1}||^2 / C)` (if m<K), and the per-sample mean of `log1p(||x_m||^2/C)` (global context, *per sample*).
* MLP: `P_m -> hidden(32) -> SiLU -> 1`, last layer zero-initialised  =>  `H = star` at init (pure DEC).
* `tie_metrics=True` (default): one metric per degree per layer; the up-block of degree `m-1` uses `H_m`,
  the down-block of degree `m+1` uses `1/H_m` (Riemannian consistency: star and inverse star).
  `tie_metrics=False`: separate heads for up/down (paper style).
* `identity_metric=True` (ablation): `H = 1`.
* Diagnostics: the layer records `log(H/star)` statistics and the condition number `max H / min H` per degree.

### 3.3 Normalised metric Hodge operators (layers.py)
(Revised during the implementation: down-block scaling, cross terms and gate corrected; MATH.md §4-5 is authoritative.)
Work in the symmetrised DEC frame `x_k = star_k^{1/2} u_k` (u = "physical" cochain). Scaling on degree k:
`scaling='dec'` (default): `S_up = star[k]`, `S_down = 1/star[k]`; `'jacobi'`: `S = diag(operator)` per sample;
`'none'`: `S = 1`. Stars are normalised per sample by their geometric mean inside the model (numerical hygiene;
all operators below are invariant to that rescaling).
Up-block of degree k (k < K), `A = d_k`, `H = H_{k+1}` `(n_{k+1}, B)`:
```
L_up  x  = S_up^{-1/2} A^T ( H ⊙ ( A ( S_up^{-1/2} x ) ) ) / beta_up
beta_up  = Gershgorin bound of  S_up^{-1/2} A^T diag(H) A S_up^{-1/2}   (per sample, >= its largest eigenvalue)
         = max_j  S_j^{-1/2} * [ |A|^T ( H ⊙ r ) ]_j ,   r = |A| S^{-1/2}      (two spmm on (n, B) vectors)
T_up     = S_up^{-1/2} A^T diag( sqrt(H / beta_up) )      # half-operator: T_up T_up^T = L_up, ||T_up|| <= 1
cross_up x_{k+1} = T_up x_{k+1} = S_up^{-1/2} A^T ( sqrt(H/beta_up) ⊙ x_{k+1} )
```
At init (H = star) `L_up = star_k^{-1/2} d_k^T star_{k+1} d_k star_k^{-1/2} / beta` (symmetrised DEC up-Laplacian) and
`T_up` is the adjoint of the symmetrised coboundary `star_{k+1}^{1/2} d_k star_k^{-1/2}` (up to `1/sqrt(beta)`).
Down-block of degree k (k > 0), `A = d_{k-1}^T`, `H = H^down_{k-1}` (`= 1/H_{k-1}` if tied, i.e. the inverse star):
```
L_down x = star_k^{+1/2} d_{k-1} ( H ⊙ ( d_{k-1}^T ( star_k^{+1/2} x ) ) ) / beta_down     ('dec': S_down^{-1/2} = star_k^{+1/2})
T_down   = S_down^{-1/2} d_{k-1} diag( sqrt(H / beta_down) ),   cross_down x_{k-1} = T_down x_{k-1}
```
Guarantee: `||L_up||, ||L_down||, ||T_up||, ||T_down|| <= 1` for every sample and every H (tests compare with dense
eigvalsh on small meshes with and without boundary). The cross terms are scale-free in H and S, so they do not change
with mesh spacing (resolution transfer).
Polynomial filter with scalar coefficients (O(C)-equivariant), `poly_order=P` (default 2):
```
m_self = sum_{p=1..P} c^up_p L_up^p x + sum_{p=1..P} c^down_p L_down^p x      (c_1 init 1, c_{p>1} init 0)
m      = m_self + w_cu * cross_up(x_{k+1}) + w_cd * cross_down(x_{k-1})       (w init 0.5)
r      = sqrt( mean_c m^2 + eps )                                             # per-cell rms (O(C)-invariant)
x_k   <- x_k + gamma * sigmoid( MLP( log1p r ) ) * m / r                       # radial norm gate; gamma scalar, MLP 1->16->1
```
(The earlier "gate then per-cell RMSNorm" form cancels the gate exactly and is not used.)
All degrees are updated synchronously from the layer's input features (as in v1). Ablations must be switchable:
`gate='relu'`, `cross=False`, `identity_metric=True`, `scaling='none'`, `poly_order=1`, `star='unit'`.
Activation checkpointing per layer when `cfg.checkpoint_layers`.

### 3.4 Lifting (lifting.py)
Inputs: `inputs: dict[int, Tensor]`, `inputs[k]: (n_k, B, F_k)`. `cfg.in_dims[k]` = F_k, `cfg.even_dims[k]` = how many
of the F_k columns are even (listed **last**: `inputs[k][..., F_k-E_k:]`), the rest are odd (cochain-valued).
`cfg.connection_dims[k]` = number of the odd columns (listed first) that are gauge connections (see 3.6).
```
x_0 = MLP([f_0, geo_0])                                                          (n_0, B, C)
for k = 1..K:
   o_k = Lin_a( d_{k-1} x_{k-1} ) + Lin_b( f_k^odd )              (no bias; odd)
   e_k = MLP([ f_k^even, geo_k, log1p(||f_k^odd||^2), log1p(||d_{k-1} x_{k-1}||^2 / C),
               mean over boundary (k-1)-cells of log1p(||x_{k-1}||^2/C) ])      (C-dim, even)
   x_k = o_k ⊙ (1 + e_k)                                            (odd × even = odd)
```
The lifting defines the hidden frame; it must be orientation-consistent (odd) and permutation-equivariant, but O(C) is
a property of the MP stack, so C-dimensional even gates are allowed here.

### 3.5 Readouts (readout.py)
* `node_scalar`: `MLP(x_0) -> (n_0, B, out_dim)` (E(n)-invariant).
* `cochain:k`: `Lin_nobias(x_k) -> (n_k, B, out_dim)` (odd: correct for fluxes/curvatures/connections).
* `even:k`: `MLP([ ||x_k||^2/C, geo_k, per-channel invariants ... ])`; simplest valid form `MLP([log1p(||x_k||^2/C), geo_k])`.
* `node_vector`: odd per-edge scalar `w_e = Lin_nobias(x_1)_e`, optional even weight `omega_e = softplus(MLP(even))`,
  then least-squares reconstruction per vertex:
  `v_i = (sum_{e∋i} omega_e l_e^2 t_e t_e^T + lam I)^{-1} sum_{e∋i} omega_e l_e w_e t_e`, `t_e` unit edge vector src->dst,
  `l_e` length, `lam = 1e-6 * trace` (surfaces embedded in 3-D: rank-2 systems). Output `(n_0, B, D)`.
  E(n)-equivariant (rotations+reflections+translations), invariant to edge-orientation convention.
  Also provide the v1-style direct variant `v_i = sum_e w_e t_e / deg_i` (`vector_mode='direct'`).

### 3.6 Gauge connection mode
If `connection_dims[1] = c > 0`, the first `c` odd edge columns are U(1)/Lie-algebra connections `A`. They are used
**only** through `d_1 A` (added to `o_2` by a bias-free linear map) and never in any gate/invariant, so the whole network
is exactly invariant under `A -> A + d_0 lambda` (test: 1e-6 in fp32). Non-Abelian components (SU(2), 3 columns) use the
same path plus the usual odd path (not exactly invariant, matching the paper's scope).

## 4. APIs (as specified; the generated reference is API.md)

```python
# rhmp/complex.py
@dataclass
class CochainComplex:
    dim: int                      # top degree K
    n: list[int]                  # [n_0 .. n_K]
    cells: list[Tensor]           # cells[0] = arange(n0)[:,None]; cells[1] = edges (n1,2) src<dst; cells[2] = faces (n2, m) (-1 padded for polygons); cells[3] = tets
    d: list[Tensor]               # CSR, d[k]: (n_{k+1}, n_k), k = 0..K-1
    dT: list[Tensor]; d_abs: list[Tensor]; dT_abs: list[Tensor]
    pos: Tensor                   # (n0, D)
    geo: list[Tensor]             # geo[k]: (n_k, G_k) float32
    star: list[Tensor]            # star[k]: (n_k,) > 0
    boundary: list[Tensor]        # bool (n_k,)
    batch: list[Tensor] | None    # batch[k]: (n_k,) int64 sample ids, or None for a single complex
    num_graphs: int
    meta: dict                    # anything else (e.g. 'star_type', 'cell_type', per-graph sizes)
    def to(self, device) -> "CochainComplex"
    def geo_dim(self, k) -> int
    def apply_d(self, k, x)  -> Tensor    # (n_k,B,C) -> (n_{k+1},B,C)
    def apply_dT(self, k, x) -> Tensor    # (n_{k+1},B,C) -> (n_k,B,C)
    def edge_vectors(self) -> Tensor      # (n1, D) pos[dst]-pos[src]
    def check_d2(self) -> float           # max |d_{k+1} d_k| over k, sparse
    @staticmethod
    def from_triangles(pos, faces, *, star="cotan", validate=True, device=None) -> "CochainComplex"
    @staticmethod
    def from_tetrahedra(pos, tets, *, star="barycentric", validate=True, device=None) -> "CochainComplex"
    @staticmethod
    def from_polygons(pos, faces: list[list[int]] | Tensor, *, star="barycentric", validate=True, device=None) -> "CochainComplex"
    @staticmethod
    def from_grid(shape: tuple[int, int], spacing=1.0, *, cell="quad" | "tri", star="cotan"|"barycentric", device=None) -> "CochainComplex"
    @staticmethod
    def batch(complexes: list["CochainComplex"]) -> "CochainComplex"    # block-diagonal; sets batch[k], num_graphs, meta['sizes']
def validate_triangles(pos, faces) -> tuple[Tensor, dict]   # drops degenerate/duplicate faces, reports stats
```
`from_triangles` must handle: 2-D or 3-D `pos`, surfaces with boundary, non-manifold edges (allowed; they just get
more cofaces), duplicate/degenerate faces (dropped with a warning when `validate=True`), int32/int64 faces, numpy inputs.

```python
# rhmp/ops.py
def spmm(A_csr, x)                          # (m,n) x (n,B,C) -> (m,B,C); x must be contiguous; fp32 (cast bf16 -> fp32 inside)
def gershgorin_bound(A_abs, AT_abs, h, s_inv_sqrt, batch=None, num_graphs=1)  # -> (B,) or (num_graphs,) per-sample bounds
def segment_max(x, index, num_segments); def segment_mean(x, index, num_segments); def segment_sum(...)
def power_iteration_norm(matvec, n, B, iters=30, device=...)   # diagnostics/tests only
def to_nbc(x) / to_bnc(x)
```
```python
# rhmp/model.py
@dataclass
class RHMPConfig:
    in_dims: dict[int, int]; even_dims: dict[int, int] = field(default_factory=dict); connection_dims: dict[int, int] = field(default_factory=dict)
    C: int = 128; n_layers: int = 4; poly_order: int = 2; metric_hidden: int = 32; log_range: float = 2.0
    tie_metrics: bool = True; scaling: str = "dec"; cross: bool = True; gate: str = "norm"; identity_metric: bool = False
    readout: str = "node_scalar"; out_dim: int = 1; vector_mode: str = "ls"; checkpoint_layers: bool = False
    def to_dict(self) / from_dict(d)
class RHMP(nn.Module):
    def __init__(self, cfg: RHMPConfig, geo_dims: dict[int, int])
    def forward(self, inputs: dict[int, Tensor], K: CochainComplex) -> Tensor          # (n_out, B, out_dim)
    def hidden(self, inputs, K) -> dict[int, Tensor]                                      # pre-readout features (tests)
    @property def diagnostics(self) -> dict                                               # last metrics stats, beta, cond numbers
```
Save/load: `torch.save({'cfg': cfg.to_dict(), 'geo_dims': ..., 'state_dict': ...})`.

```python
# rhmp/tasks/__init__.py
def load_task(name: str, root: str, *, native: bool = True, device="cuda") -> TaskData
@dataclass
class TaskData:            # everything on `device`
    name: str
    K: CochainComplex | list[CochainComplex]         # shared complex, or one per sample (variable meshes)
    inputs: dict[int, Tensor]                        # inputs[k]: (N, n_k, F_k) (shared mesh) — or list[dict] for variable meshes
    target: Tensor                                   # (N, n_t, out_dim) or list
    target_degree: int; target_kind: str             # 'node_scalar' | 'node_vector' | 'cochain' | 'even'
    in_dims, even_dims, connection_dims: dict[int,int]
    split: tuple[Tensor, Tensor, Tensor]             # train/val/test indices: sequential 70/15/15 exactly as v1
    x_stats, y_stats                                 # normalisation (mean/std per feature, computed on train, as v1)
    spatial_dim: int
```
`native=True` uses cochain inputs where the physics lives (T5: node_vector readout; T6/T7: edge inputs when the
generator can provide them, else node-encoded legacy); `native=False` reproduces v1's exact inputs/outputs for fair comparison.

## 5. Tests (pytest; `python3 -m pytest tests -q`)
Complex and operators: `d^2 = 0` (all builders, 2-D/3-D, boundary, tets, quads, batches) exactly; validation drops duplicates/degenerates;
`batch()` == individual complexes (features and operators); stars > 0; boundary flags; geometry E(n)-invariance
(rotate/reflect/translate `pos` -> identical `geo`, `star`); `gershgorin_bound >= lambda_max` (dense check) and <= 3x.
Model: O(C) equivariance of `hidden()` (random orthogonal Q, 1e-5 fp32); E(n) invariance/equivariance of readouts;
vertex relabelling (random permutation of vertices, re-derived faces) -> permuted outputs (1e-5); face orientation
flips of odd inputs -> sign-consistent outputs; batch independence (B=1 vs B=64 vs block-diag, 1e-6);
gauge invariance in connection mode (1e-5); operator norm <= 1 (dense eigvalsh, random H); no NaN/inf on sliver meshes
(aspect ratio 1e4) and extreme even inputs (1e-6..1e6); fp64 vs fp32 agreement; config/checkpoint roundtrip;
ablation flags run; `torch.compile` smoke (CUDA only, skipped on CPU).
Trainer and metrics: trainer smoke on a tiny synthetic task (2 epochs, CPU) + metric functions vs the v1 evaluation
code (`rhmp.baselines.v1.metrics_v1`, 1e-6 on random data); block-diagonal batch training == per-sample loop loss
(same params, 1e-5).

## 6. Benchmarks & experiments (one GPU per job)
* `bench/ops_bench.py`: CSR spmm vs COO vs gather/scatter for `d0`, `d1`, `d0^T`, `d1^T` at n0 in {1K, 10K, 100K},
  `B*C` in {128, 8192}; report best per size; `spmm` may dispatch on size if a clear winner exists.
* `bench/step_bench.py`: v1 vs v2 train-step time / inference time / peak memory on T6 (C=128,L=4,B=64), T7, T6_100K
  (C=16,L=3,B=2), and a synthetic 1M-cell Delaunay mesh forward (C=16); with/without AMP, compile, checkpointing.
* `bench/batch_bench.py`: T8-like variable meshes, per-sample loop vs block-diagonal batches of 4/8/16.
* Accuracy: v2 on T1,T2,T3,T5,T6,T7,T8 with the v1 protocol (100 epochs, seed 42, Adam 1e-3 cosine
  to 1e-5, wd 1e-5, batch 64, clip 1.0, sequential 70/15/15 split, same metrics) in both `native` and `legacy` input
  modes; compare to the v1 paper tables. New tasks: hetero/anisotropic-conductivity Poisson on random variable Delaunay meshes
  (train 1-2K nodes, test also at 4x resolution = zero-shot resolution transfer), 3-D tetrahedral Poisson, quad-grid T1.
  Metric interpretability: correlation of learned `log(H_1/star_1)` with true `log sigma` on hetero-Poisson.
* Robustness sweeps: vertex relabelling, edge-orientation convention, E(n) transforms, batch composition, sliver
  meshes, input noise, heterogeneity range 1e1..1e4.

## 7. Coding rules
* Every public function has a docstring with shapes. Keep modules small and pure; no global state.
* No silent fallbacks: raise informative errors (e.g. shape mismatches with the complex).
* `torch.no_grad()` for geometry; everything cached on the complex. The model must not allocate `n_k`-sized parameters.
* Log-domain for all scale features. Clamp and `eps` values are named constants with comments.

## 8. Remote workflow
Code can be edited on a laptop and executed on a GPU machine (`HOST`, checkout `~/$REMOTE_DIR`, see `tools/ssh_opts.sh`):
```
HOST=gpu tools/sync.sh                                   # push the local tree (additive rsync; data and runs excluded)
HOST=gpu tools/remote.sh 'python3 -m pytest tests -q'    # run a command there (GPU=1 selects another GPU)
HOST=gpu NAME=t6_v2 tools/run_bg.sh python3 -u -m rhmp.train --task T6 ...   # background job, log in runs/logs/
HOST=gpu tools/fetch.sh                                  # pull result files (json / md / png) back
```
Datasets: `datasets/*.pkl` and `datasets/v2/*.pt` (formats in `docs/DATASETS.md`, sources in `datasets/README.md`),
v1 checkpoints: `checkpoints_v1/` (`python3 datasets/download_v1.py --ckpt`).

## 9. Stage 2 — algorithm upgrades (pushing the metric insight further)

Goal: make the *metric* the physically meaningful object (a material tensor field), discretised the way finite
element exterior calculus does it, and give the network the global (elliptic) coupling that a metric-defined operator
implies. Everything stays: fixed d_k, SPD metrics as the only learned propagation, O(C)/E(n) exactness, bounded operators.

### 9.1 Whitney-consistent tensor metric (`metric_type='tensor'`; default stays `'diag'` until validated)
For every top cell f (triangle in 2-D/surfaces, tetrahedron in 3-D) the model predicts a SPD material tensor in the
cell's own edge frame (E(n)-equivariant, O(C)-invariant inputs):
```
sigma_f = b_f I + sum_{k in edges(f)} a_{f,k} t_k t_k^T,    b_f = exp(a tanh(.)) > 0,  a_{f,k} = e^{a} sigmoid(.) - ... >= 0
```
(any bounded parameterisation with b_f in [e^-a, e^a], a_{f,k} in [0, e^a]; zero-init => b=1, a=0).
`t_k` = unit vector of edge k of f. The 1-cochain metric is the Galerkin (Whitney) Hodge star induced by sigma:
```
(H_1)_{e e'} = sum_{f ∋ e,e'}  ∫_f w_e · sigma_f w_{e'} dA
             = sum_f [ b_f G_f^(0) + sum_k a_{f,k} G_f^(k) ]_{e e'}
G_f^(0) = (|f|/3) sum_q W_q^T W_q ,   G_f^(k) = (|f|/3) sum_q (W_q^T t_k)(W_q^T t_k)^T      (triangles)
```
`W_q ∈ R^{D x 3}` = Whitney 1-form basis values at the three edge midpoints q (quadratic integrands => exact),
columns signed with `K.meta['face_edge_signs']` so that edge orientations agree with `d_1`. Tets: 6x6 blocks with the
4-point degree-2 rule; Whitney 2-forms (face fluxes) give the degree-2 metric in 3-D the same way. All `G` blocks are
**precomputed geometry** on the complex (`K.whitney[k]`), the learned quantities are 1 + (#edges of f) nonnegative
scalars per top cell per sample. Properties: SPD (positive combination of PSD blocks, b > 0), E(n)-equivariant,
mesh-transferable, interpretable (draw sigma_f as ellipses). At init `d_0^T H_1 d_0` is exactly the P1 FEM stiffness
(= cotan Laplacian); with learned sigma it is the anisotropic FEM operator.
Use: up-blocks and cross-up terms (`d_k^T H_{k+1} (.)`) use the tensor metric; down-blocks keep diagonal (lumped)
inverse metrics (the inverse of a Galerkin star is dense), i.e. "consistent up, lumped down". Application cost:
gather the m edge features of each top cell, one (m x m) block product, scatter back — O(n2 · m^2 · B · C).
Normalisation: `beta = rowsum_max(|H_1|) · beta_unit(A, S)` where `beta_unit` is the Gershgorin bound of
`S^{-1/2} A^T A S^{-1/2}` precomputed once per complex (valid: lambda_max(A^T H A) <= lambda_max(H) lambda_max(A^T A)).
Degree-0 and top-degree metrics stay diagonal (lumped).


**Revision (stage 3, after the representability analysis of [ANISO_TASKS.md](ANISO_TASKS.md)):** the cone `b I + sum_k a_k t_k t_k^T, a_k >= 0`
only produces M-matrix (positive-edge-weight) stiffness matrices on triangles — the same class as a diagonal metric —
so it cannot represent misaligned anisotropy. Default is now `tensor_param='full'`:
`sigma_f = b_f · expm( sum_k s_{f,k} t_k t_k^T )` with signed bounded `s_{f,k} = a·tanh(.)` (zero-init = b I; the edge
dyads span Sym(2) on triangles and Sym(3) on tets, so every SPD tensor with bounded condition number is reachable),
assembled into explicit Galerkin element matrices `M_f = ∫_f w_i · sigma_f w_j` from float64 barycentric gradients
(`rhmp.layers.galerkin_geometry` / `galerkin_blocks` / `TensorMetric`; same slots and orientation as `K.whitney`; H
stays SPD because each element block is built from an SPD `sigma_f`, and the construction is well conditioned on
slivers, unlike a per-cell dyad-Gram inversion).  `tensor_param='cone'` keeps the old behaviour.  See THEORY.md §1.

### 9.2 Resolvent (implicit) Hodge layer (`layer_type='resolvent'`)
`y = (I + tau L_hat)^{-1} x` computed by batched conjugate gradients on the `(n, B, C)` block system with per-sample
Frobenius inner products over (cells, channels) — hence exactly O(C)-equivariant and per-sample; `tau = exp(log_tau)`
learnable per layer/degree/block (cap tau <= 100); since `||L_hat|| <= 1` the condition number is <= 1 + tau, so
`m = 16..32` iterations reach 1e-3..1e-4 residuals. Gradients: unrolled iterations by default (checkpointed);
optional implicit (adjoint-solve) autograd. Blocks: `resolvent_up`, `resolvent_down`, both, and a combined
`(I + tau_u L_up + tau_d L_down)^{-1}` (the metric Hodge-Laplacian resolvent = discrete Green's function of the learned
metric). Layer types are configured per layer, e.g. `layers=['poly','poly','resolvent','poly']`.

### 9.3 DEC toolkit (`rhmp/dec.py`)
Reference stars (diag + Whitney/Galerkin), Laplacians `Δ_k` for any metric, Hodge decomposition of k-cochains
(exact / coexact / harmonic) via the same CG solver, harmonic basis (inverse iteration), Whitney interpolation
(cochain → vector field at vertices/faces, "sharp") and its adjoint ("flat", vector field → cochain), all batched,
metric-aware and differentiable. These are library features for the community, used by tests and by the readouts.
