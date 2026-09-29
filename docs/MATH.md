# The mathematics of RHMP v2, as implemented

This document states the operators that the `rhmp` package computes, the guarantees they satisfy, and the tests in
`tests/` that check each guarantee.  It follows the design specification (`docs/DESIGN.md` §3 and §9) and records
where the implementation refines it (symmetrised DEC frame for both block types, half-operator cross terms, radial
gate after the RMS normalisation, full-SPD tensor parameterisation, solve layers).  Code references are given as
`module.function`.  Test names are `file::test` and run with `python -m pytest tests/<file>`.

Notation.  `K` is the top degree (2 for surfaces, 3 for tetrahedral volumes); `n_k` the number of k-cells; features
of degree k have the layout `(n_k, B, C)` (cells, samples sharing the complex, channels); `⊙` is the elementwise
product broadcast over the trailing dimensions; `|A|` is the entrywise absolute value of a matrix; `⋆_k` is a
diagonal Hodge star stored as a vector `(n_k,)`.

Contents: [1 Complex](#1-cochain-complex) · [2 Stars and geometry](#2-reference-hodge-stars-and-invariant-geometry) ·
[3 Metric](#3-bounded-log-metric) · [4 Blocks](#4-normalised-metric-hodge-blocks) ·
[5 Layer](#5-the-message-passing-layer) · [6 Lifting](#6-lifting-inputs-on-any-degree) ·
[7 Readouts](#7-readouts) · [8 Tensor metric](#8-whitney-galerkin-tensor-metric) ·
[9 Resolvent](#9-resolvent-layer) · [9b Solve layers](#9b-solve-layers-and-solver-mode) ·
[10 DEC toolkit](#10-dec-toolkit) · [11 Symmetries](#11-summary-of-exact-properties)

---

## 1. Cochain complex

`rhmp.complex.CochainComplex` stores the oriented incidence matrices (coboundaries)

```
d_k : C^k -> C^{k+1},   d_k in {-1, 0, +1}^{n_{k+1} x n_k},   k = 0 .. K-1
(d_0)_{e,i}  = -1 if i = src(e), +1 if i = dst(e)            edges stored with src < dst
(d_1)_{f,e}  = +1 / -1 if e is traversed along / against its orientation by the cyclic vertex order of face f
(d_2)_{t,f}  = (-1)^i * parity(face i of t relative to its stored sorted order)     (face i omits local vertex i)
```

Faces of triangle and polygon meshes keep the input orientation (the physical orientation of a surface); faces of
tetrahedral meshes are stored with sorted vertices and carry the orientation in `d_2`.  The matrices are built from
integer incidence, so

```
d_{k+1} d_k = 0      exactly (no floating-point error; the sparse product has only zero values)
```

for every builder (`from_triangles`, `from_polygons`, `from_tetrahedra`, `from_grid`) and for block-diagonal batches
(`CochainComplex.batch`), which concatenate the parts' matrices block-diagonally.  `K.check_d2()` computes
`max_k max |d_{k+1} d_k|` with a sparse product (v1 densified an `n_2 x n_0` matrix).  The CSR matrices `d[k]`, their
separately built transposes `dT[k]` and `|d_k|`, `|d_k|^T` are cached; the backward pass of every sparse product uses
the cached transpose (`rhmp.ops.spmm`).

| guarantee | tests |
|---|---|
| `d_{k+1} d_k = 0` exactly, all mesh types, grids, batches | `test_complex.py::test_d_squared_is_exactly_zero`, `::test_d_squared_grids`, `::test_d_squared_batches` |
| Euler characteristic and orientation signs of `d_1`, `d_2` | `test_complex.py::test_euler_characteristic`, `::test_face_orientation_and_d1_signs`, `::test_tet_orientation_signs` |
| degenerate / duplicate cells dropped with a warning | `test_complex.py::test_validation_drops_degenerate_and_duplicates`, `::test_polygon_validation` |
| a batch is the disjoint union of its parts (features and operators) | `test_complex.py::test_batch_equals_individual_complexes` |
| sparse products: dense agreement, fp32 under autocast, gradients through the cached transpose | `test_ops.py::test_spmm_matches_dense`, `::test_spmm_under_autocast_stays_fp32`, `::test_spmm_backward_uses_given_transpose`, `::test_spmm_gradcheck_fp64` |

## 2. Reference Hodge stars and invariant geometry

`rhmp.geometry` computes, once per complex in float64 and from intrinsic quantities only (lengths, areas, volumes,
angles), a positive diagonal reference star per degree:

```
star='cotan' (default for triangles; surfaces only)
    ⋆_0 = barycentric dual area (1/3 of the incident triangle areas; polygon corner areas)
    ⋆_1 = max( sum_{f ∋ e} cot(θ_{f,e}) / 2 ,  κ median(⋆_1) ),   κ = 1e-2
          (θ_{f,e} = angle of f opposite e; polygons use the barycentric half dual edge |c_f - m_e| / |e|)
    ⋆_2 = 1 / area
star='barycentric' (default for polygons and tetrahedra; always positive)
    surfaces: ⋆_1 = |dual edge| / |e|, dual edge = polyline edge midpoint -> centroids of the cofaces
    tetrahedra: ⋆_0 = dual volume, ⋆_1 = |dual face| / |e|, ⋆_2 = |dual edge| / |f|, ⋆_3 = 1 / volume
star='unit' : all ones (combinatorial operators; ablation)
```

The clamp keeps `⋆_1 > 0` on meshes with obtuse angle pairs; its lower bound `κ median(⋆_1)` is scale-covariant.
Per-cell descriptors `K.geo[k]` (`rhmp.geometry.GEO_FEATURE_NAMES`, identical for all cell types, widths
`G = (5, 6, 5, 3)`) are E(n)-invariant; scale quantities enter as `log(x / median_x)` over the complex, so the
descriptors are also invariant to a global rescaling of the mesh and comparable across meshes and resolutions.

| guarantee | tests |
|---|---|
| stars positive and finite for every mesh and star type | `test_complex.py::test_stars_positive_and_finite` |
| descriptors and stars invariant under rotations, reflections, translations | `test_complex.py::test_geometry_is_E_n_invariant` |
| descriptors scale-invariant; stars scale with the expected powers of the length unit | `test_complex.py::test_geo_is_scale_invariant_and_stars_scale` |
| exact values on reference meshes; cotan weights; clamp; Gauss-Bonnet / dihedral angles | `test_complex.py::test_square_grid_star_values`, `::test_cotan_weights_match_reference`, `::test_right_triangle_grid_cotan_clamp`, `::test_regular_tet_values`, `::test_equilateral_hexagon_values`, `::test_gauss_bonnet_and_dihedral`, `::test_boundary_flags` |

Inside the model the stars are normalised per graph by their geometric mean,
`⋆̂_k = ⋆_k / exp(mean_graph log ⋆_k)` (`rhmp.layers.make_context`).  Every operator below is invariant under a
per-sample rescaling of the stars and metrics, so this is numerical hygiene, and together with the scale-invariant
descriptors it makes the network exactly invariant to the length unit (section 11).

## 3. Bounded log-metric

The only learned geometric quantities are SPD cochain metrics.  The default (`metric_type='diag'`) metric of degree
`m` is diagonal, one value per m-cell and per sample (`rhmp.metric.MetricHead`, `RHMPLayer.metrics`):

```
H_m = ⋆̂_m ⊙ exp(ref_m) ⊙ exp(φ_m),      φ_m = a tanh( MLP_m(ψ_m) ),      a = log_range (default 2)
MLP_m : P_m -> metric_hidden (32) -> SiLU -> 1,   last layer zero-initialised   =>   H_m = ⋆̂_m exp(ref_m) at init
|log H_m| <= 60 (clamp LOG_H_MAX; inactive for sensible log_range)
```

* `ref_m` is 0 unless `cfg.metric_reference = {m: column}` names an even input column holding a known log-scale
  material coefficient (e.g. `log σ_e` on edges); the learned correction is then relative to the known coefficient.
* `ψ_m` concatenates the descriptors `geo[m]` (broadcast over samples), the even raw inputs of degree `m`, and
  feature invariants computed from the layer input (`rhmp.metric.metric_feature_dim`):

```
log1p(|x_m|^2 / C)
log1p(|q_m^-|^2 / C),  asinh(<x_m, q_m^-> / C)        if m >= 1,   q_m^- = d_{m-1} (s^up_{m-1} ⊙ x_{m-1})
log1p(|q_m^+|^2 / C)                                  if m < K,    q_m^+ = d_m^T (s^dn_{m+1} ⊙ x_{m+1})
mean over the m-cells of the sample of log1p(|x_m|^2 / C)
```

  (`s^up_{m-1}` is the scaling of the up block of degree `m-1` and `s^dn_{m+1}` that of the down block of degree
  `m+1`, section 4; for `scaling='jacobi'` the raw coboundaries are used.  The `q` are the first sparse products of
  those blocks and are shared with them).  Every entry is invariant under the cochain-frame group O(C) (norms and inner products over channels),
  under E(n), under relabelling and orientation changes (products of two odd quantities), and depends on one sample
  only.
* `tie_metrics=True` (default): one metric per degree; the up block of degree `m-1` uses `H_m` and the down block of
  degree `m+1` uses `H_m^{-1}` (a star and its inverse).  `tie_metrics=False`: separate heads for the two uses.
* `identity_metric=True` (ablation): `log H = 0`, i.e. `H = 1`, while the scaling can keep using the star
  (for a frozen `H = ⋆` use the `dec_fixed` baseline).
* `H_m / (⋆̂_m exp(ref_m)) ∈ [e^-a, e^a]`: the learned part is bounded.  `model.diagnostics` reports the fraction of
  saturated cells (`.clamp_fraction`: `|tanh| > 0.99`; `.sat`: `|φ| > 0.95 a`).

| guarantee | tests |
|---|---|
| `H = ⋆ exp(ref)` at initialisation, learned part exactly 0; tensor metrics scale `b` | `test_model.py::test_metric_reference_offset` |
| saturation is reported | `test_model.py::test_clamp_fraction_detects_saturation` |
| finite outputs and gradients with `H/⋆` at `e^{±30}` | `test_numerics.py::test_extreme_log_range_finite` |
| frozen heads keep `H = ⋆` (fixed-geometry control) | `test_baselines.py::test_dec_fixed_keeps_H_equal_star` |

## 4. Normalised metric Hodge blocks

Every learned propagation operator is a metric-weighted Hodge block.  Both block types share one form.  For degree
`k`, a block has an incidence operator `A`, a diagonal metric `H` on the neighbouring cells, and a diagonal scaling
`s = S^{-1/2}` on the k-cells (`rhmp.layers.block_operands`, `apply_L`, `apply_T`):

```
up block   (k < K):  A = d_k       (n_{k+1} x n_k),   H = H_{k+1}
down block (k > 0):  A = d_{k-1}^T (n_{k-1} x n_k),   H = H_{k-1}^{-1}  (tied)

L z = s ⊙ A^T ( h ⊙ A (s ⊙ z) ),          h = H / β                 normalised block operator, n_k x n_k
T y = s ⊙ A^T ( sqrt(h) ⊙ y ),                                      half operator (cross-degree transport)
β   = max_j  s_j [ |A|^T ( H ⊙ (|A| s) ) ]_j     per sample (per graph in a block-diagonal batch)
```

`β` is the Gershgorin bound (maximal absolute row sum) of the SPD matrix `s A^T diag(H) A s`, hence

```
0 <= L,   λ_max(L) <= 1,   L = T T^T,   ||T||_2 <= 1        for every sample and every H > 0.
```

Scalings (`cfg.scaling`):

* `'dec'` (default): the symmetrised DEC frame `x = ⋆^{1/2} u` (`u` the physical cochain) for both block types,
  `S_up = ⋆̂_k` and `S_down = 1 / ⋆̂_k`, i.e. `s_up = ⋆̂_k^{-1/2}`, `s_down = ⋆̂_k^{1/2}`.  At initialisation
  (`H = ⋆`) the blocks are proportional to the DEC Hodge-Laplacian pieces

  ```
  L_up   ∝ ⋆_k^{-1/2} d_k^T ⋆_{k+1} d_k ⋆_k^{-1/2}          L_down ∝ ⋆_k^{1/2} d_{k-1} ⋆_{k-1}^{-1} d_{k-1}^T ⋆_k^{1/2}
  T_up   ∝ ⋆_k^{-1/2} d_k^T ⋆_{k+1}^{1/2}                   T_down ∝ ⋆_k^{1/2} d_{k-1} ⋆_{k-1}^{-1/2}
  ```

  (`T_up` is the adjoint of the symmetrised coboundary `⋆_{k+1}^{1/2} d_k ⋆_k^{-1/2}`).  For `k = 0` with the
  cotan star, `⋆̂_0^{-1/2} (β L_up) ⋆̂_0^{1/2}` equals the DEC Laplacian `⋆_0^{-1} d_0^T ⋆_1 d_0` (the cotan Laplacian,
  a consistent discretisation of `-Δ`) up to the constant per-graph normalisation factor.
* `'jacobi'`: `S = diag(A^T H A)` per sample.  `'none'`: `S = 1`.

The cross terms are the half operators, which are scale-free in `H` and `S`; `L` and `T` are invariant under a
per-sample rescaling of `H` and of `S`.  For `'dec'`/`'none'` the constant scaling is folded into cached copies of the
CSR operators (`A diag(s)`), so no extra elementwise pass runs.

| guarantee | tests |
|---|---|
| `λ(L) ⊂ [0, 1]`, `L = T T^T`, `||T|| <= 1` (dense `eigvalsh`, random `H` spanning `e^{±6}`, all mesh types incl. slivers and non-manifold edges, all scalings); `β` equals `ops.gershgorin_bound` | `test_numerics.py::test_operator_norm_bound` |
| Gershgorin bound `>= λ_max` and `<= 3 λ_max` on test meshes; per-graph bounds in batches; differentiable | `test_ops.py::test_gershgorin_bounds_lambda_max`, `::test_gershgorin_block_diagonal_matches_parts`, `::test_gershgorin_is_differentiable` |
| blocks at initialisation are the symmetrised DEC pieces above | `test_numerics.py::test_dec_blocks_at_init` |
| `L_up` of degree 0 is resolution consistent (cotan Laplacian converging to `-Δ` under refinement) | `test_numerics.py::test_resolution_consistency_L_up` |
| outputs independent of the length unit (scalars invariant, vectors scale as `1/c`) | `test_numerics.py::test_uniform_scaling_invariance` |

## 5. The message-passing layer

A polynomial layer (`rhmp.layers.RHMPLayer`, `kind='poly'`) updates all degrees synchronously from its input:

```
m_k = sum_{p=1..P} c^up_{k,p} L_up^p x_k + sum_{p=1..P} c^dn_{k,p} L_dn^p x_k
      + w^up_k T_up x_{k+1} + w^dn_k T_dn x_{k-1}                          (scalar coefficients; P = poly_order)
r_k = sqrt( |m_k|^2 / C + ε ),   ε = 1e-6                                    per cell and sample (O(C)-invariant)
x_k <- x_k + γ_k sigmoid( MLP(log1p r_k) ) m_k / r_k                         radial gate, MLP: 1 -> 16 -> SiLU -> 1
```

Initialisation: `c_{·,1} = 1`, `c_{·,p>1} = 0`, `w = 0.5`, `γ = 1`.  The update is the message times a positive scalar
computed from its O(C)-invariant norm, so it is O(C)-equivariant; the gate is applied after the normalisation by
`r_k` (the form "gate, then per-cell RMSNorm" would cancel the gate).  The polynomial runs in a fused autograd kernel
(Horner-like recursion; `cfg.fused=False` selects the plain-autograd reference with the same maths).  In `forward` the
last layer only updates the degree read by the readout.  Ablations: `gate='relu'` (breaks O(C) and orientation
parity), `cross=False`, `poly_order=1`, `scaling='none'`, `identity_metric=True`, `star='unit'`.

| guarantee | tests |
|---|---|
| fused kernel equals the reference; gradients | `test_model.py::test_fused_matches_reference`, `test_numerics.py::test_fused_block_gradcheck` |
| O(C)-equivariance of the stack (`propagate(x Q) = propagate(x) Q`, random orthogonal `Q` incl. reflections; 1e-5 fp32, 1e-10 fp64) for every variant | `test_equivariance.py::test_OC_equivariance`, `::test_OC_equivariance_variants`, `::test_hidden_is_lift_then_propagate` |
| activation checkpointing gives identical gradients; bf16 AMP close to fp32 | `test_model.py::test_checkpoint_layers_same_gradients`, `::test_amp_matches_fp32` |

## 6. Lifting: inputs on any degree

`inputs[k]` has shape `(n_k, B, F_k)` with the column layout (`rhmp.lifting.input_layout`)

```
[ connection columns (connection_dims[k]) | ordinary odd columns | even columns (even_dims[k], listed last) ]
```

Odd columns are cochains (they change sign with the orientation of their cell); even columns are
orientation-free coefficients (conductivities, lengths, flags).  The lifting (`rhmp.lifting.CochainLifting`) is

```
x_0 = MLP([f_0, geo_0])                                                (connection columns of degree 0 excluded)
for k = 1..K:
    o_k = Lin_a(d_{k-1} x_{k-1}) + Lin_b(f_k^odd) + Lin_c(d_{k-1} A_{k-1})                 bias-free: odd
    e_k = MLP([f_k^even, log1p(|f_k^odd|^2), log1p(|d_{k-1} x_{k-1}|^2 / C),
               mean over the boundary (k-1)-cells of log1p(|x_{k-1}|^2 / C)], geo_k)          even, C-dimensional
    x_k = o_k ⊙ (1 + e_k)                                                                   odd x even = odd
```

with the last layer of each `e_k` zero-initialised.  `A_{k-1}` are the connection columns of degree `k-1`: they enter
only through `d_{k-1} A_{k-1}` and never through a gate or an invariant, so the network is exactly invariant under
`A -> A + d_{k-2} λ` (`d d = 0`).  With `connection_odd=True` (non-Abelian connections such as SU(2), where only the
Abelian part is a coboundary) the connection columns also enter the ordinary odd path and exact invariance is given
up.  The lifting fixes the hidden frame, so C-dimensional even gates are allowed there; O(C) is a property of the
message-passing stack.

| guarantee | tests |
|---|---|
| vertex relabelling (complex rebuilt, canonical edge orientations change): outputs permuted, odd outputs pick up the orientation signs (1e-5) | `test_equivariance.py::test_vertex_relabelling` |
| face orientation flips change the sign of odd face data and nothing else | `test_equivariance.py::test_face_orientation_flip` |
| gauge invariance `A -> A + d_0 λ` for every readout (1e-5), lost with `connection_odd`; face connections on tetrahedra | `test_equivariance.py::test_gauge_invariance_edge_connection`, `::test_gauge_invariance_face_connection_tets` |
| input validation (shapes, degrees, devices, dtypes) | `test_model.py::test_input_validation` |

## 7. Readouts

`rhmp.readout` reads the hidden cochain of one degree (`model.feature_degree`) and writes cells of degree
`model.output_degree`:

```
node_scalar   y = MLP(x_0)                                  (n_0, B, out)       vertex scalars
cochain:k     y = Lin(x_k), bias-free                       (n_k, B, out)       odd for k >= 1
even:k        y = MLP([log1p(x_k^2), log1p(|x_k|^2/C)], geo_k)   (x_0 itself for k = 0)     orientation-invariant
grad          y = d_0 φ,  φ = MLP(x_0)                      (n_1, B, out)       d_1 y = 0, invariant to φ + const
curl          y = d_1 a,  a = Lin(x_1) bias-free            (n_2, B, out)       d_2 y = 0 on volumes
div, div:k    y = d_{k-1}^T b,  b = Lin(x_k) bias-free      (n_{k-1}, B, out)   d_{k-2}^T y = 0; div:1 sums to 0 per mesh
node_vector   vertex vectors from odd edge scalars w_e = Lin(x_1)_e   (n_0, B, out·D)
```

`node_vector` with `vector_mode='ls'` (default) solves a damped least-squares problem per vertex,

```
M_i = sum_{e ∋ i} ω_e l_e^2 t_e t_e^T,    b_i = sum_{e ∋ i} ω_e l_e w_e t_e,    v_i = (M_i^2 + μ_i^2 I)^{-1} M_i b_i
μ_i = 1e-4 tr(M_i),   ω_e = softplus(MLP([log1p(x_1^2), log1p(|x_1|^2/C)], geo_1)) > 0   (1 at initialisation)
```

(`t_e` unit edge vector, `l_e` length), in float64 with autocast disabled.  It is exact for constant fields when
`w_e = l_e t_e · v`, has zero response along null directions of `M_i` (collinear vertex stars), and on surfaces in
3-D is solved in the vertex tangent plane (basis from the area-weighted normal tensor, which does not depend on the
face orientation).  `vector_mode='direct'`: `v_i = sum_e w_e t_e / deg_i` (v1 style).  Both are E(n)-equivariant and
invariant to the edge-orientation convention (`w_e` and `t_e` flip together).

| guarantee | tests |
|---|---|
| scalar and cochain readouts E(n)-invariant, vectors equivariant (rotations, reflections, translations): complex rebuilt from transformed positions (1e-5; vectors 1e-4) and exact in fp64 (1e-10) | `test_equivariance.py::test_En_invariance_and_equivariance`, `::test_En_equivariance_exact_fp64` |
| constraint readouts: `d_1 y = 0` (grad), `d_2 y = 0` (curl), `d^T y = 0` and zero vertex sums (div) to 1e-6, also on batches | `test_model.py::test_constraint_readouts` |
| LS reconstruction exact for constant fields, tangent on surfaces, rank-deficient stars, gradcheck | `test_numerics.py::test_ls_readout_exact_for_constant_fields`, `::test_ls_readout_tangent_on_surfaces`, `::test_ls_solve_rank_deficient_and_well_posed`, `::test_vector_readout_gradcheck` |

## 8. Whitney-Galerkin tensor metric

`cfg.metric_type='tensor'` (DESIGN §9.1; the default metric is `'diag'`) replaces the diagonal metric of the intermediate
degrees in the up blocks and cross-up terms by the Galerkin (Whitney) Hodge star of a per-top-cell SPD material
tensor (`rhmp.metric.TensorMetricHead`, `rhmp.dec.apply_whitney_metric`):

```
tensor_param='full' (default):  σ_f = b_f expm( sum_j s_{f,j} t_j t_j^T ),   s_{f,j} = a tanh(z_{f,j})  (signed, bounded)
tensor_param='cone':            σ_f = b_f I + sum_j a_{f,j} t_j t_j^T,      a_{f,j} = e^a tanh(max(z_{f,j}, 0)) >= 0
t_j = unit vector of direction edge j of the top cell f;  b_f = exp(a tanh z_b) ∈ [e^-a, e^a];  zero-init: σ_f = I
(H_k)_{e e'} = sum_{f ∋ e, e'} ∫_f w_e · σ_f w_{e'}                     (Galerkin star of the Whitney k-forms)
full:  element matrices M_f = ∫_f w_i · σ_f w_j assembled from float64 barycentric gradients (rhmp.layers.galerkin_blocks)
cone:  M_f = b_f G0_f + sum_j a_{f,j} Gj_f with G0_f = sum_q ω_q W_q^T W_q, Gj_f = sum_q ω_q (W_q^T t_j)(W_q^T t_j)^T
```

The edge dyads `t_j t_j^T` span Sym(2) on triangles and Sym(3) on tetrahedra, so the full parameterisation reaches
every SPD tensor with condition number up to about `e^{2a}` (including anisotropy misaligned with the edges); the cone
only generates M-matrix stiffness matrices, the class of the diagonal metric (THEORY.md §1).

`W_q` are the Whitney basis values (1-forms on edges; for tetrahedra also 2-forms on faces) at quadrature points
exact for the quadratic integrands (triangles: the three edge midpoints with weights `|f|/3`; tetrahedra: the
symmetric 4-point rule with weights `|t|/4`), with signs matching `d`.  The model predicts `1 + m` nonnegative scalars
per top cell and sample from `[geo, invariants, even inputs]` of the top cell and of its direction edges; the metric is
applied matrix-free (gather the `m_k` cochain values of each top cell, one `m_k x m_k` block product, scatter).  Used
degrees: `H_1` for the degree-0 up block on triangle meshes; `H_1` and `H_2` on tetrahedra.  Down blocks keep the
diagonal (lumped) inverse metrics ("consistent up, lumped down"), because the inverse of a Galerkin star is dense.
Normalisation:

```
ρ = max_e rowsum|H|_e >= λ_max(H)            (blockwise bound, rhmp.dec.whitney_rowsum_abs)
β_unit = Gershgorin bound of S^{-1/2} A^T A S^{-1/2} >= λ_max(A^T A)   (per graph, cached)
L = A^T H A / (ρ β_unit)   =>  ||L|| <= 1          cross-up  T = A^T H / (sqrt(β_unit) ρ)   =>  ||T|| <= 1
```

(`A = d_k diag(s)` includes the block scaling of section 4.  Unlike the diagonal case, the tensor cross-up term is
not a square root of `L`; it is normalised separately with `||A^T H|| <= ||A|| ||H||`.)

Properties: `H_k` is SPD (each element block is built from an SPD `σ_f`; for the cone a positive combination of PSD
blocks), E(n)-equivariant, mesh-transferable and per sample.  At initialisation (`b = 1`, `a = 0`), `d_0^T H_1 d_0`
is exactly the P1 finite-element stiffness matrix (the unclamped cotan Laplacian) and on tetrahedra `d_1^T H_2 d_1` is
the Nédélec curl-curl stiffness; with learned `σ` it is the anisotropic FEM operator.  With `metric_reference`, `b_f`
is scaled by `exp(mean of ref over the m-cells of f)` and, for the edge metric, `a_{f,j}` by
`exp(ref of direction edge j)`.
Caveat: the tensor coordinates are not identifiable from `d_0^T H_1 d_0` (which sees `H_1` only on the image of
`d_0`); interpret learned tensors (`model.metric_fields(...)[l]['sigma']`) through their action and anisotropy
statistics.  The tensor metric requires triangle or tetrahedral complexes (`K.whitney` does not exist for polygons)
and `scaling` `'dec'` or `'none'`.

| guarantee | tests |
|---|---|
| P1 stiffness identity (triangles), curl-curl identity (tets) | `test_complex.py::test_whitney_p1_stiffness_identity`, `::test_whitney_curl_curl_identity`, `test_numerics.py::test_tensor_metric_at_init_is_p1_stiffness` |
| SPD for random `b > 0`, `a >= 0`; blocks E(n)-invariant; survive `batch()` and `to()` | `test_complex.py::test_whitney_metric_is_spd`, `::test_whitney_blocks_are_E_n_invariant`, `::test_whitney_and_beta_unit_survive_batch_and_to` |
| matrix-free product equals the assembled matrix; gradients; per sample; block-diagonal batches | `test_dec.py::test_apply_whitney_metric_matches_assembled`, `::test_whitney_metric_gradcheck`, `::test_whitney_metric_is_per_sample`, `::test_whitney_metric_on_block_diagonal_batch` |
| row-sum and `β_unit` bounds; `||L|| <= 1`, `||T|| <= 1` for random tensors | `test_dec.py::test_whitney_rowsum_and_beta_bounds`, `test_numerics.py::test_tensor_metric_spd_and_operator_bounds` |
| symmetries with the tensor metric (O(C), E(n), relabelling, orientation, batch independence) | `test_equivariance.py::test_OC_equivariance_tensor_tets`, `::test_En_equivariance_exact_fp64`, `::test_vertex_relabelling`, `::test_face_orientation_flip`, `::test_batch_independence_tensor` |
| the action of an anisotropic FEM operator is reproducible by `(b, a)` alone (~1e-7) | `test_numerics.py::test_anisotropy_recovery` |
| full parameterisation: misaligned ratio-100 tensors representable (1e-10 vs an independent P1 assembly); SPD, bounds and zero init; misaligned anisotropy recovered from the isotropic start | `test_numerics.py::test_full_tensor_representability_misaligned`, `::test_full_tensor_spd_bounds_and_zero_init`, `::test_full_tensor_anisotropy_recovery_misaligned` |

## 9. Resolvent layer

A resolvent layer (`cfg.layers=[..., 'resolvent', ...]`, DESIGN §9.2) replaces the polynomial self term by the
implicit operator

```
y_k = (I + τ_up L_up + τ_dn L_dn)^{-1} x_k,       τ = exp(log τ) <= 100 (TAU_MAX), learnable per degree and block, init 1
m_k = y_k + w^up_k T_up x_{k+1} + w^dn_k T_dn x_{k-1},        same radial gate as section 5
```

(the combined metric Hodge-Laplacian resolvent, a discrete Green's function of the learned metric).  Since
`||L|| <= 1`, `cond(I + τ_up L_up + τ_dn L_dn) <= 1 + τ_up + τ_dn`.  The solve is batched conjugate gradients on the
`(n, B, C)` block (`rhmp.dec.cg_solve`) with per-sample (per-graph) Frobenius inner products over cells and channels,
so the iterates rotate with any orthogonal channel mixing (O(C)-equivariant) and samples never interact.  A fixed
number of iterations (`resolvent_iters`, default 20) runs without host synchronisation; a sample freezes at relative
residual `1e-12`.  Gradients: `resolvent_grad='implicit'` (default) solves the adjoint system with the same CG,

```
M λ = ∂L/∂y,    ∂L/∂x = λ,    ∂L/∂θ = -λ^T (∂M/∂θ) y         (memory independent of the iterations)
```

or `'unrolled'` differentiates through the checkpointed iterations.  With `resolvent_warm_start=True` a layer starts
from the previous resolvent layer's solution `x0` of the same degree (detached) and solves only for the correction,
`y = x0 + M^{-1}(x - M x0)`; the implicit gradients stay exact.  Resolvent layers use diagonal metrics.

| guarantee | tests |
|---|---|
| CG residual <= 1e-3 at T6 size (n_0 = 1024) for `τ ∈ {1, 10, 100}`, checked against an independent residual | `test_numerics.py::test_resolvent_cg_residual_t6_size` |
| equals a dense solve (1e-9); gradients (implicit and unrolled) pass `gradcheck` | `test_numerics.py::test_resolvent_matches_dense_solve`, `::test_resolvent_gradcheck` |
| model with resolvent layers: finite gradients for `τ`, diagnostics, config round trip | `test_numerics.py::test_resolvent_layer_model` |
| warm start: same result and gradients | `test_model.py::test_resolvent_warm_start` |
| batch independence and O(C)-equivariance with resolvent layers | `test_equivariance.py::test_batch_independence_resolvent`, `::test_OC_equivariance_variants` |
| CG: SPD solves, O(C)-equivariance and per-sample independence, block-diagonal batches, implicit = unrolled = dense gradients, finite fp32 backward after convergence | `test_dec.py::test_cg_solves_spd_systems`, `::test_cg_is_O_C_equivariant_and_per_sample`, `::test_cg_block_diagonal_batch`, `::test_cg_gradients_unrolled_and_implicit_match_dense`, `::test_cg_fp32_unrolled_backward_stays_finite_after_convergence` |

## 9b. Solve layers and solver mode

A solve layer (`cfg.layers=[..., 'solve', ...]`) applies the discrete Green's operator of the learned metric in
physical units (the metric is not normalised, so its magnitude matters):

```
y_k = (Δ_H + λ / L^2)^{-1} x_k      on the free cells;  Δ_H = un-normalised metric Hodge Laplacian (k = 0: FEM
                                     stiffness d_0^T H_1 d_0 over the lumped mass ⋆_0);  λ = softplus(log λ), init 1e-3;
                                     L^D = domain measure per graph
solve_bc='dirichlet': K.meta['dirichlet'][k] or K.boundary[k] fixed;  'none';  'neumann': zero-mean solution of the
                     pure Neumann problem (compatible source by removing its per-graph mean)
```

The solve is preconditioned CG (`rhmp.layers.pcg_solve[_implicit]`) with per-sample, per-graph Frobenius inner
products (O(C)-equivariant, batch-independent), exact implicit gradients and warm starts; `solve_precond='twolevel'`
adds a coarse correction `M^{-1} = I + Z (Z^T A Z)^{-1} Z^T` (Jacobi-scaled frame) on per-graph spatial aggregates of
about 32 vertices (degree 0).  `RHMPConfig.solver_preset(...)` (trainer `--solver-mode`) builds a model that is linear
in the field inputs: linear lifting, one solve layer, no gate, no cross terms, a linear readout fitted by least squares
on the first batch (`RHMP.fit_linear_readout_`).  With `material_dims = {k: m}` the material columns reach only the
metric heads.  For data from `-div(σ grad u) = f` with P1 elements and a lumped-mass right-hand side this model class
contains the exact discrete solution operator (tensor metric with `b_f = σ_f`; the diagonal metric too when the P1
stiffness is an M-matrix).  `model.operator_residual(inputs, K, u, f)` returns `||L^H u - M f||^2 / ||M f||^2`, linear
in a diagonal metric (the convex operator-identification loss `--aux-pde`).

| guarantee | tests |
|---|---|
| solver mode reproduces FEM exactly: diagonal metric (9e-15 abs.), P1 with element-wise σ through the tensor metric (2.5e-9 abs.) | `test_numerics.py::test_solver_mode_reproduces_fem_diag`, `::test_solver_mode_reproduces_p1_fem_tensor` |
| the trainer path with a frozen tensor metric reproduces real HP_k100 data without training (val R2 > 0.99999) | `test_numerics.py::test_trainer_solver_frozen_tensor_faceref_reproduces_real_hp` |
| `grad` readout and Neumann solves reproduce potential differences (T5-like) | `test_numerics.py::test_solver_mode_grad_readout_reproduces_potential_differences`, `::test_solver_mode_grad_with_output_map_neumann_poisson` |
| two-level preconditioner: residual after 64 iterations reduced at least 5x, same solution and gradients, per sample and graph | `test_numerics.py::test_twolevel_preconditioner_cuts_solve_residual`, `::test_twolevel_preconditioner_is_per_sample_and_per_graph` |
| material columns reach only the metric; the operator residual identifies the metric; `--material` / `--aux-pde` in the trainer | `test_numerics.py::test_material_columns_only_reach_the_metric`, `::test_operator_residual_identifies_the_metric`, `::test_trainer_solver_mode_material_aux_pde` |
| `mdiv:1` conserves `sum ⋆_0 y` per graph in block-diagonal batches | `test_numerics.py::test_mdiv_readout_conserves_mass_in_block_diagonal_batches` |

## 10. DEC toolkit

`rhmp.dec` exposes the same operators for direct use.  Metrics `H` are given per degree (list or dict): a diagonal
`(n_j,)` / `(n_j, B)` tensor, a callable `x -> H_j x` (e.g. a Whitney metric), or `None` for the reference star.

```
hodge_laplacian(K, k, H, weak=True):   L_k = d_k^T H_{k+1} d_k + H_k d_{k-1} H_{k-1}^{-1} d_{k-1}^T H_k   (symmetric PSD)
                        weak=False:    Δ_k = H_k^{-1} L_k                                                (H_k-self-adjoint)
hodge_decompose(K, k, x, H):   x = d_{k-1} α + H_k^{-1} d_k^T γ + h,   H_k-orthogonal
        d_{k-1}^T H_k d_{k-1} α = d_{k-1}^T H_k x,    d_k H_k^{-1} d_k^T γ = d_k x      (CG per sample / per graph)
        h is closed (d_k h = 0) and co-closed (d_{k-1}^T H_k h = 0)
harmonic_basis(K, k, H):   H_k-orthonormal basis of the harmonic k-cochains (dimension = k-th Betti number),
        by projecting random probes with hodge_decompose and orthonormalising with rank detection
flat(K, v):     x_e = (v_src + v_dst)/2 · (p_dst - p_src)                 (trapezoidal de Rham map)
sharp(K, x1):   Whitney interpolation w = sum_e x_e (λ_a ∇λ_b - λ_b ∇λ_a) at vertices (area-weighted) or barycentres
cg_solve / cg_solve_implicit:   batched CG described in section 9
whitney_blocks / apply_whitney_metric / whitney_rowsum_abs / assemble_whitney_metric:   section 8
```

`flat` is exact for fields linear along edges; `sharp(flat(v)) = v` for constant and rigid-rotation fields.

| guarantee | tests |
|---|---|
| Laplacians symmetric PSD with the right Betti numbers (torus 1/2/1, sphere 1/0/1, disk and cube contractible); Whitney up metrics | `test_dec.py::test_hodge_laplacian_symmetric_psd_and_betti`, `::test_hodge_laplacian_with_whitney_up_metric` |
| Hodge decomposition orthogonal and complete on a torus, disk, sphere and a batch | `test_dec.py::test_hodge_decompose_torus`, `::test_hodge_decompose_disk_sphere_and_batch` |
| harmonic bases | `test_dec.py::test_harmonic_basis` |
| sharp/flat exactness, shapes, simplicial complexes required | `test_dec.py::test_sharp_flat_exact_on_constant_and_rotation_fields`, `::test_flat_is_exact_line_integral_and_shapes`, `::test_sharp_requires_simplicial_complex` |

## 11. Summary of exact properties

| property | mechanism | tests (tolerance) |
|---|---|---|
| `d_{k+1} d_k = 0` | integer oriented incidence | section 1 (exact) |
| cochain-frame equivariance O(C) | metric heads see only O(C)-invariant statistics; scalar polynomial coefficients; radial gate; CG with Frobenius inner products | `test_OC_equivariance*` (1e-5 fp32, 1e-10 fp64) |
| E(n) invariance (equivariance for `node_vector`) | intrinsic geometry only; positions enter only through edge vectors in the vector readout | `test_En_*` (1e-5, fp64 1e-10), `test_geometry_is_E_n_invariant` |
| vertex relabelling and orientation conventions | odd lifting (odd linear part times even gate), bias-free odd maps, orientation-invariant invariants | `test_vertex_relabelling`, `test_face_orientation_flip` (1e-5) |
| gauge invariance (connection mode) | connections enter only through `d_k A` | `test_gauge_invariance_*` (1e-5) |
| batch independence (sample i unaffected by the rest of the batch) | per-sample metrics, per-sample `β` (segment max per graph), per-graph CG and means; no batch statistics anywhere | `test_batch_independence_*` (1e-6); block-diagonal vs single differ at ~7e-7 on CUDA (row partitioning of cuSPARSE), not bitwise |
| invariance to the length unit | per-graph star normalisation, `log(x/median)` descriptors, Gershgorin-normalised, scale-free operators | `test_uniform_scaling_invariance` |
| bounded operators | Gershgorin normalisation (`||L||, ||T|| <= 1`); bounded log-metric | `test_operator_norm_bound`, `test_tensor_metric_spd_and_operator_bounds` |
| numerical robustness | log-domain features, clamps with named constants, fp32 sparse products, fp64 LS solves | `test_sliver_mesh_finite` (aspect 1e4), `test_extreme_even_inputs_finite` (1e-6..1e6), `test_extreme_log_range_finite`, `test_fp64_vs_fp32` (1e-4) |
| no per-cell parameters | all weights act per cell and are shared | `test_parameter_count_independent_of_mesh_size` |

Consequences worth knowing:

* **Pseudo-scalars.** Because the network is exactly equivariant under orientation relabelling and invariant under
  reflections, it cannot output pseudo-scalars or pseudo-vectors (vorticity at vertices, plaquette flux averaged to
  vertices, `n × ∇ψ`) from native cochain inputs.  Such targets are predicted as a face cochain (`cochain:2`) or as an
  orientation-free quantity followed by a fixed, parameter-free map that supplies the physical orientation
  (`rhmp.data.OutputMap`: oriented face-to-vertex mean, `n × g`).
* **Length scales.** A PDE with an absolute length scale (screening length, diffusion time, a domain size that varies
  relative to the mesh spacing) is only determined up to that scale for a unit-invariant model; keep
  (PDE length) / (mesh spacing) fixed across samples or pass a global scale as an input column (e.g. the log median
  edge length).  See `docs/TUTORIAL.md`, "Length scales".
