# Anisotropy task suite: AHP, ASURF, ACURL, ADARCY

Goal: tasks whose operator is the Whitney/Galerkin (FEEC) Hodge star of a **misaligned** SPD material tensor, for
0-forms (scalar diffusion), 1-forms (curl–curl) and 2-forms (Darcy fluxes). They are built so that:

* the targets are exact FEEC solutions from the model's own operator family (`d_k^T H_{k+1}(sigma) d_k` with the
  Whitney Galerkin star of a per-cell tensor);
* a diagonal cochain metric provably cannot represent the operator, and neither can the old tensor cone
  (`tensor_param='cone'`), while the full parameterisation (`tensor_param='full'`) can;
* the learned metric can be compared with the true material tensor (metric recovery), up to what is identifiable.

Files: `datasets/generators/gen_aniso.py` (generator), `datasets/generators/aniso_fields.py` (fields, FEEC element matrices,
representability diagnostics), `rhmp/tasks/aniso.py` (loaders `ANISO_TASKS`, `structure_metrics`),
`tests/test_aniso_tasks.py` (30 tests), `scripts/run_aniso.sh` (comparison protocol), the anisotropy path of
`scripts/metric_recovery.py`, and the registry hook at the end of `rhmp/tasks/suite.py`. The theory behind the
representability statements is in `docs/THEORY.md` §1–2.

## 1. Tasks

| name | mesh (per sample) | PDE and discretisation | inputs (degree: columns; e = even) | target (readout) |
|---|---|---|---|---|
| `AHP_r<R>` | Delaunay mesh of [0,1]², n0 ~ U{1000..2000} (HP meshes) | −div(Σ∇u) = f, u = 0 on ∂Ω; P1: `d0ᵀ M1(Σ) d0 u = M0 f` | 0: f · 1: log tᵀΣt (e) · 2: log det Σ, log ratio (e) | u, nodes (`node_scalar`) |
| `ASURF_r<R>` | closed surface of `gen_surf` (ellipsoid / SH sphere / torus), median edge 0.07, n0 1.5K–3.5K | screened Poisson (M + K(Σ)) u = M f, tangent-plane fibre tensor | as AHP | u (`node_scalar`) |
| `ASURF_heat_r<R>` | as ASURF | heat flow, 10 implicit steps of (M + K(Σ)/10) v' = M v | as AHP | u_T (`node_scalar`) |
| `ACURL_r<R>` | Delaunay tets of [0,1]³, n0 ~ U{2000..3000} (TET meshes) | curl(ν curl A) + εA = J, tangential A = 0 (PEC); Nédélec: `(d1ᵀ M2(ν) d1 + ε M1) A = M1 j` | 1: j (odd), log tᵀνt (e) · 2: log nᵀνn (e) · 3: log det ν, log ratio (e) | A, edges (`cochain:1`) |
| `ACURLb_r<R>` | as ACURL | B = d1 A (magnetic flux through faces) | as ACURL | B, faces (`curl`: B = d1 a, so d2 B = 0) |
| `ADARCY_r<R>` | as ACURL | Darcy −div(K∇p) = f, p = 0 on ∂Ω; mixed RT0–P0 (Whitney 2-/3-forms): `M2(K⁻¹) J − d2ᵀ P = 0`, `d2 J = F` | 0: f · 1: log tᵀKt (e) · 2: log nᵀK⁻¹n (e) · 3: F_T = ∫_T f (odd), log det K, log ratio (e) | J, face fluxes (`cochain:2`) |
| `ADARCYp_r<R>` | as ACURL | the same Darcy problem, primal P1: `d0ᵀ M1(K) d0 p = M0 f` | as ADARCY | p, nodes (`node_scalar`) |

`R ∈ {10, 100}` is the maximal eigenvalue ratio. The two ratios share meshes, fields and sources (sample i of
`X_r10.pt` and `X_r100.pt` differ only in the ratio). `_n1500` variants of the tet tasks (for example
`ACURLb_r100_n1500`) use the first 1500 samples with the standard split; see §7 for why.

## 2. Material fields

* **Principal direction** (misaligned with the mesh): a smooth random vector field `V(x)` (64 random Fourier features
  with shared frequencies, length scale U[0.2, 0.5] in 2-D, U[0.3, 0.6] in 3-D, U[0.4, 0.8] on surfaces). The
  direction is `τ = V/|V|`. On surfaces it is the normalised tangential projection `P_T V / |P_T V|` in the plane of
  each face.
* **Isotropic cores**: the eigenvalue ratio is `r_eff = 1 + (r − 1) w` with `w = |V|²/(|V|² + δ²)`, δ = 0.3. The
  tensor field is therefore continuous even at the zeros of V. Such zeros are unavoidable for line fields on spheres
  (Poincaré–Hopf). `r_eff ≈ r` almost everywhere: the median ratio is 9.4 (R = 10) and 93 (R = 100), and 96 % of
  the cells (99 % of the tets) have `r_eff > R/2` (§6).
* **Tensors** (unit determinant times a magnitude):
  * planar: `A = (I + (r_eff − 1) ττᵀ)/√r_eff`;
  * surfaces: `(P + (r_eff − 1) ττᵀ)/√r_eff`, zero normal part;
  * volumes: prolate `(I + (r_eff − 1) ττᵀ)/r_eff^{1/3}`.
  They are evaluated at cell centroids and are constant per cell (P0), as in the FEM.
* **Magnitude**: `log s = ½ ln(10) tanh(g/σ_g)` with a smooth GRF `g`. This gives contrast 10. The tanh-squashing
  with the exact standard deviation of the drawn feature sum is mesh independent, so the 4x-finer meshes see the same
  field. Σ = s A (AHP), Σ = 0.05 s A (ASURF, the SURF screening length), ν = s A, K = s A.
* **Sources**:
  * AHP and ADARCY: 2–5 Gaussians, the HP recipe (widths U[0.05, 0.2] in 2-D, U[0.1, 0.25] in 3-D);
  * ASURF: the SURF source (embedding-space GRF plus bumps whose centres are vertices of the coarse mesh, shared with
    the fine mesh);
  * ACURL: `J = curl W`, divergence free, with W a random vector field, normalised by its exact RMS. The edge input
    starts from exact line integrals `∫_e J·dl` (closed form for random Fourier features), and the load is
    `b = M1 j` (Whitney interpolant of J). The Whitney interpolant of a divergence-free field is not weakly
    divergence-free, because its normal component jumps across faces. The gradient residue is only 7–10 % of j in
    the M1 norm, but ε damps gradients far less than the curl-curl term damps everything else, so it made up about
    40 % of A (measured before the fix). The generator therefore removes it with a discrete Helmholtz projection,
    `j ← j − d0χ` with `(d0ᵀM1d0)χ = d0ᵀM1j` on the interior nodes. Then `d0ᵀ M1 j = 0` exactly, and the solution is
    weakly divergence-free (`d0ᵀ M1 A = 0`, the `gauge` structure metric): pure curl-curl physics. `j` is the
    projected cochain, exactly consistent with `b = M1 j`.
  * ADARCY: `F_T = ∫_T f` (4-point degree-2 rule) is the tet source of the mixed problem; the P1 problem uses the
    lumped `M0 f`.

The edge inputs `log(t_eᵀ T t_e)` are means over the cells incident to the edge. Their values on the edges of one
cell determine that cell's tensor (3 edges span Sym(2), 6 edges span Sym(3)), so the inputs carry the full material
while staying E(n)-invariant. Constants: ASURF ε = 0.05 (as SURF: the screening length is 3.2 edges), ACURL ε = 10
(screening length ≈ 0.32 ≈ 4 edges for the geometric-mean ν).

## 3. Discretisations: the targets are in the model's operator family

* **P1 = Whitney 1-form Galerkin** (AHP, ASURF, ADARCYp). The P1 stiffness `Σ_T |T| ∇φᵀ Σ_T ∇φ` equals
  `d0ᵀ M1(Σ) d0`. Here `M1(Σ)_{ee'} = Σ_T ∫_T w_e·Σ_T w_{e'}` is the Galerkin Hodge star of Whitney 1-forms,
  i.e. `H_1` of the tensor metric.
* **Nédélec** (ACURL). The operator `d1ᵀ M2(ν) d1 + ε M1` uses the Whitney 2-form Galerkin star
  `M2(ν)_{ff'} = Σ_T ∫_T w_f·ν w_{f'}` (the model's `H_2`) and the Whitney 1-form mass. The PEC condition removes
  the boundary edges.
* **Mixed RT0–P0** (ADARCY). The Whitney 2-form mass of the resistivity K⁻¹ appears in the saddle-point system
  `[[M2(K⁻¹), −d2ᵀ], [−d2, 0]] [J; P] = [0; −F]`, with natural condition p = 0. The fluxes satisfy `d2 J = F` per tet
  and `d1ᵀ M2(K⁻¹) J = 0` (K⁻¹j is a gradient): J is the unique `M2(K⁻¹)`-co-exact 2-cochain with divergence F.
* Bases and orientations are exactly those of `rhmp.geometry`:
  * Whitney 1-forms: `w_ab = λ_a∇λ_b − λ_b∇λ_a` with canonical `src < dst`;
  * Whitney 2-forms: `2(λ_a ∇λ_b × ∇λ_c + cyclic)` with sorted face vertices;
  * quadrature: degree-2 exact rules.
* **Verification** (`tests/test_aniso_tasks.py`):
  * the generator's scipy assemblies (P1(Σ), M1(Σ), `d1ᵀM2(ν)d1 + εM1`, M2(K⁻¹), P1(K)) equal the model's
    float64 Galerkin blocks (`rhmp.layers.galerkin_blocks`, which the full tensor metric uses) to ≤ 1e-9 relative
    Frobenius error on dataset samples, and `rhmp.dec.assemble_whitney_metric` (fp32 blocks) to ≤ 1e-6;
  * manufactured solutions converge at the expected rates: P1 nodal O(h²), RT0 fluxes O(h), Nédélec edge values O(h)
    (`gen_aniso.py --selftest`, §8);
  * the Whitney 2-form basis has unit flux through its own face; `d²=0` holds exactly; the generator's d0/d1/d2 equal
    the complex's.
* Solves are sparse direct: SuperLU, or for the mixed problem a pressure Schur complement solved by PCG with a
  factorised two-point preconditioner, with LU fallback. All relative residuals are ≤ 1e-11 (§6), and conservation
  `max|d2J − F|/max|F|` is ≤ 1e-11. Positions are rounded to float32 **before** the solve, so the stored meshes
  reproduce the operators exactly.

## 4. Why a diagonal metric (and the old tensor cone) cannot represent these operators

**Scalar problems, 2-D and surfaces.** `d0ᵀ diag(h) d0` is the graph Laplacian with edge weights `h_e > 0`, an
M-matrix. The anisotropic P1 stiffness has the edge weights `w_e = −Σ_T |T| ∇λ_iᵀ Σ_T ∇λ_j`, and these are
negative wherever the mesh is not Σ-Delaunay. Explicit counter-example (`test_diagonal_counterexample`): take the
equilateral triangle (0,0), (1,0), (½, √3/2) with fibres perpendicular to the edge (0,0)–(1,0), i.e.
`Σ = diag(1, r)/√r`. The per-face weight of that edge is

    w = (√3/4)(1 − r/3)/√r      (= ½ cot 60° = 0.289 for r = 1;  −0.320 for r = 10;  −1.40 for r = 100),

which is negative as soon as σ_∥/σ_⊥ > 3. No positive diagonal edge star produces it. The off-diagonal *pattern* of
`d0ᵀ diag(h) d0` is fixed by the mesh and its sign by `h > 0`. Anisotropy along a non-edge direction needs the
Whitney coupling of the edges of each cell.

The old cone `σ = b I + Σ_j a_j t_j t_jᵀ` (a ≥ 0) is no better. `a_j t_j t_jᵀ` only adds the positive weight
`a_j |T|/|e_j|²` to edge j, and `b I` adds `b · cotan`, so it also produces only M-matrices: essentially the diagonal
class. A cell tensor lies in the cone iff it is acute in the σ⁻¹ metric. That holds for 20 % of the cells at R = 10
and 2 % at R = 100 (§6). The full family `σ = b expm(Σ_j s_j t_j t_jᵀ)` reaches every SPD tensor because the edge
dyads span Sym(d). Its required range `max_j |s_j|` is listed in §6.

A diagonal metric is a two-point flux approximation. Its best version, `h_e = star_e · t_eᵀΣt_e` (TPFA with the
projected conductivity, i.e. the diag + `--metric-ref` prior), is inconsistent for anisotropic tensors on
non-Σ-orthogonal meshes: the error does not vanish with h. The oracle numbers below measure this gap per task.

**1-forms and 2-forms (ACURL, ADARCY).** Here the metric acts on non-exact cochains, so even the sign argument is not
needed. The Whitney 2-form mass couples the four faces of every tet (`∫ w_f·ν w_{f'} ≠ 0`), and
`d1ᵀ diag(h) d1` has no such coupling, whatever the values of h. For the Darcy fluxes the co-exactness condition
`d1ᵀ M2(K⁻¹) J = 0` depends on the full element blocks. In 3-D the cone fails almost completely: about 1 % of the
tets at R = 10 and 0 % at R = 100 (§6).

## 5. Loaders (`rhmp/tasks/aniso.py`)

* `load_task(name)` resolves every name through the suite hook; `SUITE_TASKS` is extended by `ANISO_TASKS` in one
  failure-safe try-block. Direct use: `ANISO_TASKS[name](root, device=...)` or `load_aniso(name, root, ...)`.
* Data layout:
  * variable meshes, one complex per sample, block-diagonal batches; sequential 70/15/15 split;
  * `TaskData.from_arrays` statistics from the training part: odd columns and odd targets scale-only, vertex and
    even columns mean/std;
  * `spatial_dim` is 2 (AHP) or 3;
  * triangle complexes use `star` (default `cotan`), tet complexes `barycentric`;
  * defaults: C = 128, 4 layers, 8 meshes per batch.
* Extra tests: `fine` holds the first test samples re-solved on meshes with 4x the nodes (same physical instances;
  default: all 500/400 for AHP/ASURF, the first 50 for tet tasks, `fine_max=`). ASURF also has `test_<family>`.
* Metadata:
  * `meta['metric_ref']` is the suggested `--metric-ref`: `1:0` (edge `log tᵀTt`) for the scalar problems; `2:0`
    for ACURL (`log nᵀνn`) and ADARCY (`log nᵀK⁻¹n`: the 2-form metric of the flux problem is the resistivity);
  * `meta['recovery']` names the stored truth for `metric_recovery.py`;
  * `meta['representability']` holds the dataset-wide diagnostics;
  * `meta['pde']` (AHP and ADARCYp: `K u = M f` with M = `star0`) enables `--aux-pde` and `--solver-mode`.
* Options:
  * `fine=False`, `max_samples=` (smoke), `keep_on=`, `n_samples=` (subset, as `_n<N>`), `fine_max=`;
  * `abs_scale=True` via `load_task`: absolute mesh scale. It is relevant for ACURL, whose screening length is fixed
    while n0 varies ±20 %, so h varies ±7 %; ASURF has a fixed physical resolution;
  * legacy mode is not defined, because the tasks are cochain tasks.

## 6. Representability and oracle errors (the theoretical backbone of the comparison)

Dataset-wide per-cell diagnostics are means over all samples of the main set, from the `.json` summaries and
`sample_stats`:

* cone: fraction of cells whose tensor lies in the cone family (`tensor_param='cone'`);
* neg w: fraction of edges with a negative P1 weight (scalar problems; edges between two Dirichlet nodes excluded);
* s_max: the smallest `max_j |s_j|` of the full family per cell (median, fraction ≤ 3, fraction ≤ 5). The full
  family with `--log-range a` represents a cell exactly iff `s_max < a`.

Oracle errors: relative L2 error (mass-weighted for node fields) of the exact solve with the true per-cell tensors
**replaced** by:

* cone: the Frobenius-nearest cone member per cell (NNLS);
* diag ref: the diagonal star `star_k · (projected material input)`, i.e. the model's reference star times the
  material reference, the TPFA prior of `diag + --metric-ref` (in physical scaling);
* diag w⁺: positive-part P1 weights `max(w_e, 0)`, the operator-closest M-matrix (scalar problems);
* iso: the isotropic part `det(T)^{1/d} I`;
* edge recon: the tensor rebuilt per cell from the model's own edge inputs. The edge-averaged projections
  `p_e = t_eᵀTt_e` give `T_f = Σ_j α_j t_j t_jᵀ` with `Σ_j α_j (t_i·t_j)² = p_i` over the cell's edges. This is the
  input-information bound for a tensor metric that decodes the material from its inputs cell by cell;
* full |s| ≤ L: the full family with `s` clipped to `[−L, L]`.

The exact Galerkin tensor gives 0 by definition. These are single-layer operator errors, i.e. the error of the
physics prior a model starts from or can reach with one metric per cell. A trained network can compensate partially
through depth and nonlinearity. `gen_aniso.py --analyse 12` evaluates 12 test samples per task and writes
`datasets/v2/ANISO_representability.json`.

**Per-cell representability** (dataset means over all samples of the main sets):

| set | cone | neg w | s_max median | s_max p90 | s_max ≤ 3 | s_max ≤ 5 | ratio median | ratio > R/2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AHP r10 | 0.204 | 0.225 | 1.37 | 2.23 | 0.964 | 0.993 | 9.4 | 0.964 |
| AHP r100 | 0.026 | 0.269 | 2.80 | 4.55 | 0.582 | 0.929 | 93.1 | 0.957 |
| ASURF r10 | 0.198 | 0.234 | 1.35 | 1.81 | 1.000 | 1.000 | 9.4 | 0.965 |
| ASURF r100 | 0.017 | 0.286 | 2.76 | 3.67 | 0.693 | 0.997 | 93.7 | 0.957 |
| ACURL r10 | 0.010 | – | 1.69 | 3.04 | 0.896 | 0.972 | 9.6 | 0.995 |
| ACURL r100 | 0.000 | – | 3.41 | 6.16 | 0.345 | 0.821 | 96.1 | 0.993 |
| ADARCY: K (P1) r10 | 0.010 | 0.367 | 1.69 | 3.05 | 0.896 | 0.972 | 9.6 | 0.995 |
| ADARCY: K⁻¹ (mixed) r10 | 0.017 | – | 1.69 | 3.05 | 0.896 | 0.972 | 9.6 | 0.995 |
| ADARCY: K (P1) r100 | 0.000 | 0.383 | 3.41 | 6.16 | 0.345 | 0.821 | 96.1 | 0.993 |
| ADARCY: K⁻¹ (mixed) r100 | 0.000 | – | 3.41 | 6.16 | 0.345 | 0.821 | 96.1 | 0.993 |

**Oracle solution errors** (first 12 test samples; mean relative L2 error, mass-weighted for node fields):

| task | field | diag ref | diag w⁺ | cone | iso | edge recon | full \|s\|≤2 | full \|s\|≤3 | full \|s\|≤5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AHP_r10 | u | 0.236 | 0.352 | 0.201 | 0.489 | 0.016 | 0.026 | 0.015 | 0.007 |
| AHP_r100 | u | 0.444 | 0.580 | 0.449 | 1.849 | 0.108 | 0.387 | 0.134 | 0.059 |
| ASURF_r10 | u | 0.122 | 0.106 | 0.079 | 0.189 | 0.006 | 0.003 | 0.000 | 0.000 |
| ASURF_r100 | u | 0.285 | 0.297 | 0.255 | 0.389 | 0.041 | 0.106 | 0.029 | 0.000 |
| ACURL_r10 | A | 0.455 | – | 0.293 | 0.456 | 0.092 | 0.035 | 0.013 | 0.006 |
| ACURL_r10 | B = d1A | 0.626 | – | 0.363 | 0.555 | 0.208 | 0.097 | 0.047 | 0.023 |
| ACURL_r100 | A | 0.853 | – | 0.815 | 0.744 | 0.524 | 0.378 | 0.199 | 0.058 |
| ACURL_r100 | B = d1A | 0.911 | – | 0.852 | 0.808 | 0.621 | 0.536 | 0.330 | 0.140 |
| ADARCY_r10 | p (P1, ADARCYp) | 0.328 | 0.589 | 0.133 | 0.677 | 0.044 | 0.030 | 0.016 | 0.007 |
| ADARCY_r10 | J (mixed, ADARCY) | 0.609 | – | 0.423 | 0.705 | 0.310 | 0.087 | 0.039 | 0.016 |
| ADARCY_r100 | p (P1, ADARCYp) | 0.430 | 0.679 | 0.291 | 4.432 | 0.140 | 0.380 | 0.134 | 0.048 |
| ADARCY_r100 | J (mixed, ADARCY) | 0.758 | – | 0.639 | 0.855 | 0.364 | 0.423 | 0.237 | 0.093 |

Reading the tables:

* **The old cone is the diagonal class.** In 2-D the cone projection and the diagonal TPFA prior have the same error
  (AHP_r10 0.20 vs 0.24, AHP_r100 0.45 vs 0.44, ASURF_r100 0.26 vs 0.29). The full family with |s| ≤ 3–5 is within
  0–6 %.
* **2-form problems.**
  * ACURL_r100: the diagonal DEC curl-curl star is at 85 % (A) and 91 % (B), and the cone does not improve on the
    isotropic part.
  * ADARCY_r100: the diagonal (TPFA) 2-form star gives 76 % flux error.
  * In both cases the full family reaches 6–14 % with |s| ≤ 5, and exactly 0 without clipping.
* **Log-range.** The full family's clipped errors set the `--log-range` needed. It is 3 (R = 10) and 5 (2-D R = 100);
  3-D R = 100 still loses 6–14 % at 5, so `run_aniso.sh` uses 6 there. The skinny cells of the HP Delaunay meshes
  need larger |s| than the smoothed SURF meshes.
* **Input information.**
  * 2-D: the edge inputs determine the cell tensors well (edge recon 0.6–4 % on ASURF, 1.6 % / 11 % on AHP).
  * 3-D: the loss is structural. A tet mesh has only ~1.3 edges per tet, but a per-tet tensor has 6 entries. Each
    edge projection is averaged over ~5 tets, and the 6×6 Gram inversion is ill-conditioned on sliver tets. Edge
    recon gives ACURL A 9 % at R = 10 and 52 % at R = 100 (single samples 44–61 %). ADARCYp gives 4 % / 14 %. The
    ADARCY flux, whose metric is the inverse tensor, gives 31 % / 36 %.
  * Evaluating the smooth field at the edge midpoints (0.52 → 0.50) or fitting a constant tensor over each tet's
    face-neighbour patch (0.52 → 0.43) helps little: across-fibre screening lengths are ~2 mesh spacings at R = 100,
    so the discrete solution is sensitive to cell-level tensor detail. Exact per-tet projections reconstruct to
    1e-12.
  * Hence, for the 3-D R = 100 sets a model that sees only edge/face/tet scalars is information-limited, whatever its
    metric family. The per-cell truth is stored (`nu_tet`, `K_tet`; `rhmp.tasks.aniso.cell_tensors`).
  * Model-side recommendation: an E(n)-equivariant per-cell tensor input (a world-frame tensor per top cell,
    projected by the model onto the cell's own edges, `t_jᵀ T t_j`, which is invariant). A tensorial
    `metric_reference` built from it would make the fixed full-tensor prior exact, while the diagonal prior stays
    TPFA. The R = 10 sets and all 2-D sets are not affected.

## 7. Data inventory (generated 2026-09-27, 24 CPU workers)

| file | samples | n0 (mean, min–max) | size | gen time | max residual | notes |
|---|---:|---|---:|---:|---:|---|
| `AHP_r10.pt` | 5000 | 1502, 1000–2000 | 0.74 GB | 10 s (24 workers) | 4.1e-12 |  |
| `AHP_r10_fine.pt` | 500 | 6040, 4008–7988 | 0.30 GB | 5 s (24 workers) | 2.7e-12 |  |
| `AHP_r100.pt` | 5000 | 1502, 1000–2000 | 0.74 GB | 10 s (24 workers) | 4.6e-12 |  |
| `AHP_r100_fine.pt` | 500 | 6040, 4008–7988 | 0.30 GB | 5 s (24 workers) | 9.3e-12 |  |
| `ASURF_r10.pt` | 4000 | 2500, 1512–3458 | 1.34 GB | 39 s (24 workers) | 2.9e-14 | heat mass drift ≤ 3e-15 |
| `ASURF_r10_fine.pt` | 400 | 9959, 6146–14408 | 0.53 GB | 22 s (24 workers) | 1.2e-13 | heat mass drift ≤ 5e-15 |
| `ASURF_r100.pt` | 4000 | 2500, 1512–3458 | 1.34 GB | 39 s (24 workers) | 8.5e-14 | heat mass drift ≤ 6e-15 |
| `ASURF_r100_fine.pt` | 400 | 9959, 6146–14408 | 0.53 GB | 22 s (24 workers) | 3.2e-13 | heat mass drift ≤ 1e-14 |
| `ACURL_r10.pt` | 3000 | 2497, 2000–3000 | 3.16 GB | 831 s (24 workers) | 1.6e-12 | source gradient removed 7.6 % |
| `ACURL_r10_fine.pt` | 200 | 9906, 8016–11968 | 0.91 GB | 2715 s (8 workers) | 1.5e-12 | source gradient removed 4.9 % |
| `ACURL_r100.pt` | 3000 | 2497, 2000–3000 | 3.16 GB | 831 s (24 workers) | 2.1e-11 | source gradient removed 7.6 % |
| `ACURL_r100_fine.pt` | 200 | 9906, 8016–11968 | 0.91 GB | 2715 s (8 workers) | 1.4e-11 | source gradient removed 4.9 % |
| `ADARCY_r10.pt` | 3000 | 2495, 2000–3000 | 3.61 GB | 2429 s (24 workers) | 8.4e-13 | conservation ≤ 4e-13 |
| `ADARCY_r10_fine.pt` | 200 | 10084, 8068–11980 | 1.07 GB | 3710 s (8 workers) | 8.6e-13 | conservation ≤ 4e-13 |
| `ADARCY_r100.pt` | 3000 | 2495, 2000–3000 | 3.61 GB | 2429 s (24 workers) | 4.4e-12 | conservation ≤ 3e-13 |
| `ADARCY_r100_fine.pt` | 200 | 10084, 8068–11980 | 1.07 GB | 3710 s (8 workers) | 2.9e-12 | conservation ≤ 3e-13 |

Generation times are wall-clock times with 24 worker processes (fine 3-D sets: 8, for memory) on a shared CPU (load 25–60 from other jobs), excluding packing. Each file has a `.json` summary. The oracle report is `datasets/v2/ANISO_representability.json` (`--analyse 12`; a copy is `results/cab75/aniso_analysis/all.json`).

Memory of the loaded complexes (host RAM; `--data-on cpu` batches are moved per step):

| set | MB per complex, with Whitney blocks (tensor metrics) | without (diagonal metrics) |
|---|---|---|
| AHP | 2.7 | 1.6 |
| ASURF | 4.6 | 2.8 |
| ACURL / ADARCY | 46–48 | 14.3 |
| tet `fine` (4x) | ~170 | ~55 |

The 3000-sample tet sets therefore need about 140 GB for tensor-metric runs. `run_aniso.sh` uses the `_n1500`
variants by default (about 70 GB with the blocks). Diagonal-metric runs load without the blocks:
`load_task(..., whitney=False)`, which the trainer requests for diagonal metrics, drops them per complex while
building. Measured: `ACURLb_r100_n1500` loads in 122 s with 21.4 GB of host RAM; `AHP_r100` with the blocks loads in
49 s (13.5 GB). `N3D=""` selects all 3000 tet samples. The full-tensor path does not read the fp32 `G0`/`Gk` blocks
(22 MB of the 46 MB); the cone and fixed-star paths do.

## 8. Structure metrics (`rhmp.tasks.aniso.structure_metrics`)

Physics residuals of the (denormalised) predictions. Operators come from the stored tensors through the model's
float64 Galerkin blocks. The same residuals of the stored targets are reported as `target_*`; float32 storage sets
that floor, at roughly condition number × 6e-8.

| task | metrics |
|---|---|
| AHP, ADARCYp | `pde_res` = ‖K u − M f‖_I / ‖M f‖_I (interior nodes, M = `star0`); `bc` = ‖u_∂‖ / ‖u‖ |
| ASURF | `pde_res` = ‖(M + K)u − M f‖ / ‖M f‖; `integral` = \|Σ M u − Σ M f\| / Σ M\|f\| (exactly conserved) |
| ASURF_heat | `integral` (heat flow conserves Σ M u exactly) |
| ACURL | `pde_res` = ‖(d1ᵀM2(ν)d1 + εM1)A − M1 j‖_I / ‖M1 j‖_I (interior edges); `bc` = ‖A_∂‖/‖A‖ (PEC); `gauge` = ‖d0ᵀ M1 A‖_I / ‖M1 A‖ (weak Coulomb gauge, exactly 0 for the true solution) |
| ACURLb | `div` = ‖d2 B‖/‖B‖ (0 by construction for the `curl` readout, not for baselines); `bc` = ‖B_∂‖/‖B‖ (normal B = 0) |
| ADARCY | `cons` = ‖d2 J − F‖/‖F‖ (per-tet mass balance); `irrot` = ‖d1ᵀ M2(K⁻¹) J‖ / ‖M2(K⁻¹) J‖ (K⁻¹ j is a gradient) |

Self-test (manufactured solutions, constant misaligned ratio-10 tensors), errors at n0 ≈ 500/2000/8000 (2-D) and
≈ 600/2400/9600 (3-D):

* P1 max nodal error: 1.6e-2, 4.3e-3, 1.0e-3 (O(h²));
* RT0 relative flux error: 9.7e-2, 5.6e-2, 3.3e-2 (O(h));
* Nédélec relative edge error: 5.2e-2, 3.5e-2, 2.1e-2 (O(h)).

## 9. Metric recovery (`scripts/metric_recovery.py RUN`)

The script runs the model on the first `--n` test samples, reads every layer's metric with `RHMP.metric_fields` and
compares it with the true material taken from the model's own inputs (de-normalised with the run's statistics; raw
material / reference columns stay raw).  Per layer it reports Pearson and Spearman correlations, the least-squares
slope and intercept of `learned = slope * true + intercept`, per-sample correlations and the spread of the
per-sample intercepts (`<run>/metric_recovery_<task>.json`, a hexbin PNG unless `--no-plots`, and a markdown table
over many runs with `--scan DIR ... --table FILE`):

* **diagonal metrics**: `log(H_1/star_1)` and its learned part `phi` vs the edge input (`log t_e^T Sigma t_e`, the
  projected material), plus the clamp saturation.  With `--metric-ref` the `log_ratio` correlation is trivially about
  1, because the reference is that input; `phi` is the informative number;
* **tensor metrics** (per-cell `sigma_f = metric_fields(...)[l]['sigma']`): the edge action
  `log mean_{f ni e} t_e^T sigma_f t_e` vs the edge input (the identifiable quantity), `log det(sigma_f)/2` and the log
  mean eigenvalue vs the face input, and statistics of the eigenvalue ratio `lambda_max / lambda_min`; on the HP
  tensor sets (`HP_k100_aniso<R>`) also the log-ratio correlation and the principal-direction error against the
  stored true tensors.  For the AHP / ASURF tasks the face comparison uses the stored face column `log det Sigma`
  itself, so a perfect recovery there has slope 0.5 (correlations are unaffected).
* Principal directions of learned vs true tensors on one mesh are drawn by `scripts/make_figures.py` (figure 4,
  `docs/figures/fig4_tensor_ellipses.png`).

**Identifiability caveat** (THEORY.md §2): only the action of a metric is identifiable. `d0ᵀH_1d0` has one weight
per edge but a triangle tensor has three entries (the per-face tensors have a ~3 n0-dimensional kernel), and the
(b, s) or (b, a) head outputs overparameterise each tensor. In addition, the scale of H is a per-sample gauge of
the normalised operators, and layers need not reproduce the physical metric at all (normalised polynomial
operators approximate inverses). The edge action is therefore the recovery measure. Per-cell angles,
ratios and log det are interpretability diagnostics, most meaningful in solver mode or with `--material`, where the
metric is the only route from the material to the output.

## 10. Comparison protocol (`scripts/run_aniso.sh`)

Sequential runs, 50 epochs, seed 42. `--log-range` is 3 (R = 10), 5 (2-D R = 100) or 6 (tet R = 100), which
covers the s-range of §6.
`--metric-ref` is the task's material reference (§5). After each run the script writes `metric_recovery_<task>.json`
(+ PNG) and `structure_metrics.json` (test + fine). `SUMMARY.md` collects test R², fine R², s/epoch, peak
GB, parameters, structure residuals and last-layer recovery.

| variant | flags | question |
|---|---|---|
| `diag` / `diag+ref` | `--metric-type diag [--metric-ref K:0]` | diagonal metric (the M-matrix / TPFA family) |
| `tensor` / `tensor+ref` | `--metric-type tensor --tensor-param full [...]` | full Galerkin tensor metric |
| `cone+ref` | `--metric-type tensor --tensor-param cone --metric-ref ...` | the old cone family (≈ diagonal class) |
| `diagfixed+ref` | `--metric-type diag --no-learn-metric --metric-ref ...` | fixed DEC/TPFA prior (`learn_metric=False`) |
| `tensorfixed+ref` | `--metric-type tensor --no-learn-metric --metric-ref ...` | fixed isotropic Galerkin prior (`learn_metric=False`) |
| `diag-solver+ref` / `tensor-solver+ref` | `--solver-mode` (AHP, ADARCYp) | one physical solve layer: the model *is* the FEM solution map of its metric, so accuracy measures representability directly |
| baselines | `--model mgn` / `egnn` / `dec_fixed` | param-matched non-Hodge baselines and the frozen DEC metric |

Presets:

* `PRESET=core` (default): AHP_r100, ASURF_r100, ACURLb_r100_n1500, ADARCY_r100_n1500, ADARCYp_r100_n1500, AHP_r10
  × {diag+ref, tensor+ref, cone+ref, tensorfixed+ref, diag-solver+ref, tensor-solver+ref} + mgn, dec_fixed.
* `PRESET=full`: all 14 tasks × all 9 variants + mgn, egnn, dec_fixed.

Exact commands:

```
GPU=1 NAME=aniso tools/run_bg.sh bash scripts/run_aniso.sh                     # core preset, runs/aniso/
PRESET=full GPU=1 NAME=aniso_full tools/run_bg.sh bash scripts/run_aniso.sh     # everything
TASKS="AHP_r100 AHP_r10" VARIANTS="diag-solver+ref tensor-solver+ref" MODELS="" bash scripts/run_aniso.sh
EXTRA="--abs-scale" TASKS="ACURLb_r100_n1500" bash scripts/run_aniso.sh      # absolute mesh scale (ACURL)
python3 scripts/metric_recovery.py runs/aniso/AHP_r100_tensor+ref_s42 --n 32
```

The script was validated end to end on the CPU: all 9 variants and 3 baselines on AHP_r100 with tiny sizes
(`EPOCHS=1 EXTRA="--max-samples 60 --max-train-batches 3 --device cpu --C 16"`). All exited 0, with recovery,
structure metrics and SUMMARY.md written.

Expected cost on a shared GPU, extrapolated from HP/TET runs: AHP about 1–1.5 h per 50-epoch run, ASURF about 2 h,
tet `_n1500` tasks about 2–3 h. The core preset is about 40 runs, i.e. several GPU-days sequentially; split it across
GPUs with `TASKS=`.

What the numbers should show if the theory holds:

* scalar problems, `solver` variants: the diagonal metric plateaus at the diag oracle error (it can reach at best the
  M-matrix closest to the truth), while the full tensor can reach the exact operator;
* 2-form problems (ACURLb, ADARCY): the diagonal and cone metrics have a large irreducible operator error;
* recovery: the operator action error of the full tensor metric falls far below that of diag/cone, and its principal
  directions align with the fibres (small angle errors, also for out-of-cone cells);
* structure: exactly zero `div` for ACURLb (curl readout) for every rhmp variant; conservation and irrotationality
  residuals of the flux predictions.

## 11. Tests (`tests/test_aniso_tasks.py`, 30 tests, 15–30 s on a CPU)

* convergence self-test (small sizes);
* tet topology equals the complex (`d0`, `d1`, `d2`, boundary flags) and `d²=0`;
* unit flux of the Whitney 2-forms;
* operator equivalence on dataset samples: triangles and surfaces (P1 vs `d0ᵀH_1d0`, Whitney 1-form blocks) and tets
  (curl–curl, M1, M2(K⁻¹), P1(K)), ≤ 1e-9 vs the model's float64 blocks and ≤ 1e-6 vs `rhmp.dec`;
* solver residuals, PEC, `d2 B = 0`, `d2 J = F`, `d1ᵀM2J = 0`;
* representability: cone membership, full-parameterisation round trip, cone projection, the counter-example;
* tiny generated sets for all tasks and both ratios: files, loaders for all 7 kinds × 2 ratios (shapes, degrees,
  readouts, splits, `fine`, family subsets, normalisation, B = d1A exactly); the `load_task` hook, `task_defaults`,
  `list_tasks`, `whitney=False`, `abs_scale`, subsets;
* structure metrics (oracle predictions at the float32 floor; random model finite; `div = 0` with the curl readout);
* one trainer epoch with the full tensor metric and the metric reference on AHP, ASURF, ACURLb and ADARCY.

## 12. Smoke runs (20 CPU steps, full tensor metric, C = 32, 4 layers, 1 mesh per step, 21 samples per split part)

| task | load (63 samples) | 20 steps | val R² | test R² | fine R² | params | MB / complex |
|---|---:|---:|---:|---:|---:|---:|---:|
| AHP_r10 | 2.4 s | 5.0 s | 0.056 | 0.113 | −0.041 | 20 125 | 2.7 |
| AHP_r100 | 2.8 s | 7.6 s | 0.023 | 0.096 | −0.164 | 20 125 | 2.7 |
| ASURF_r100 | 3.5 s | 8.8 s | 0.363 | 0.434 | 0.401 | 20 125 | 4.6 |
| ASURF_heat_r100 | 1.3 s | 6.7 s | 0.292 | 0.349 | 0.253 | 20 125 | 4.6 |
| ACURL_r100 | 9.3 s | 25.9 s | 0.195 | 0.161 (u) | −0.021 (u) | 28 168 | 48.3 |
| ACURLb_r10 | 19.0 s | 34.0 s | 0.019 | 0.021 (u) | −0.037 (u) | 28 168 | 48.3 |
| ACURLb_r100 | 13.9 s | 15.7 s | 0.010 | 0.010 (u) | −0.036 (u) | 28 168 | 48.3 |
| ADARCY_r10 | 6.8 s | 14.2 s | 0.020 | 0.020 (u) | −0.047 (u) | 28 200 | 46.2 |
| ADARCY_r100 | 7.2 s | 13.9 s | 0.007 | 0.006 (u) | −0.048 (u) | 28 200 | 46.2 |
| ADARCYp_r10 | 6.7 s | 9.2 s | 0.110 | 0.146 | 0.131 | 29 257 | 46.2 |
| ADARCYp_r100 | 7.0 s | 15.8 s | 0.098 | 0.144 | 0.115 | 29 257 | 46.2 |

(u) marks uncentred R² of orientation-odd cochain targets. The fine split holds 2 of the 4x samples. These runs only
validate the pipeline: every task loads through `load_task` (the hook) on the real data, trains with the full tensor
metric and the task's metric reference, and is evaluated on the test and 4x-finer splits.

Two further CPU checks:

* The comparison script ran end to end on AHP_r100 (§10).
* Solver mode on AHP_r100 with 100 training samples and 10 epochs (130 steps): `diag-solver+ref` test R² 0.423,
  `tensor-solver+ref` 0.421. This is inconclusive at that size. The tensor metric had learned
  anisotropic tensors (median ratio 8.9 vs the true 95) whose directions were not yet aligned (median angle error
  56°); the real comparison needs the full runs.
