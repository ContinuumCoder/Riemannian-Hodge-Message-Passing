# Task suite, extension (SURF, DYN, QUAL): generators, discretisations, sizes, protocols

Scope: the priority-2 tasks of `docs/TASK_SUITE.md`. Generators are in `datasets/generators/`: `gen_surf.py`, `gen_dyn.py`,
`gen_qual.py`, plus the meshing helpers in `surfaces.py`. They use numpy/scipy only and are seeded. Loaders, rollout
evaluation and structure metrics are in `rhmp/tasks/suite.py` (`SUITE_TASKS`). Tests are in `tests/test_suite_tasks.py`,
and `scripts/eval_on.py` evaluates a trained run on another task or rolls it out. File format and loader
conventions follow `datasets/generators/gen_HP.py` and `rhmp/tasks/synthetic.py`: variable meshes are packed as flat
concatenations with offsets `ptr0/ptr1/ptr2`, with float32 fields and int32 cells. Loaders use a sequential 70/15/15
split, statistics from the training part, the `node_scalar` readout, and keep complexes on the GPU only if they fit
`GPU_BUDGET_GB`.

Regenerate everything (≈1 min wall with 24 workers; or `SUITE=1 SKIP_CORE=1 bash scripts/gen_datasets.sh`):

```
python3 -u datasets/generators/gen_surf.py --workers 24   # SURF.pt, SURF_geo.pt, SURF_topo.pt   (40 s)
python3 -u datasets/generators/gen_dyn.py  --workers 24   # DYN.pt, DYNfix.pt                     (7 s)
python3 -u datasets/generators/gen_qual.py --workers 24   # HP_qual_graded.pt, HP_qual_sliver.pt  (6 s)
```

## Inventory (generated 2026-09-27)

| file | samples | nodes per mesh (min–max, mean) | size | gen time¹ | used by |
|---|---|---|---|---|---|
| `SURF.pt` | 5700 surfaces (1900 ellipsoids, 1900 SH-perturbed spheres, 1900 tori) | 1352–3458, 2485 | 683 MB | 33 s | `SURF`, `SURF_heat` |
| `SURF_geo.pt` | 500 superquadrics | 1538–3752, 2483 | 60 MB | 2.8 s | extra test `geo`; `SURF[_heat]_geo` |
| `SURF_topo.pt` | 500 double tori (genus 2) | 3050–6256, 4647 | 112 MB | 2.0 s | extra test `topo`; `SURF[_heat]_topo` |
| `DYN.pt` | 600 trajectories × 101 states, one random mesh each | 1300–1700, 1501 | 423 MB | 2.5 s | `DYN`, `DYN_delta` |
| `DYNfix.pt` | 500 trajectories × 101 states, one fixed mesh | 1500 | 327 MB | 2.3 s | `DYNfix`, `DYNfix_delta` |
| `HP_qual_graded.pt` | 100 HP_k100 test instances × 7 mesh levels = 700 | 1014–1972, 1523 | 92 MB | 1.9 s | `HP_qual_graded[_ref]` |
| `HP_qual_sliver.pt` | 100 HP_k100 test instances × 6 mesh levels = 600 | 1014–1972, 1523 | 79 MB | 1.8 s | `HP_qual_sliver[_ref]` |

¹ Sample generation with 24 worker processes, excluding packing/saving. Every file has a `.json` summary next to it.

Loading (GPU 1 at 100 % load from other jobs):
* `SURF`: 46 s. Complexes are built in a 4-thread pool, 36 s. Stage 2 added the Whitney blocks, so a complex is
  ~4.7 MB and the 5700 SURF complexes take ~27 GB. That is above the 24 GB budget, so they stay on the CPU and the
  trainer moves each block-diagonal batch. `--data-on device` (or `RHMP_GPU_BUDGET_GB=40`) keeps them on the GPU.
* `DYN`: 8 s. `DYNfix`: 0.7 s. `HP_qual_*`: 11 s.
* Host-to-device copies use pinned memory with one synchronisation. Synchronous pageable copies took ~5 ms per small
  tensor on the busy GPU (236 ms per `K.to('cuda')`), which had made the loads take 150–290 s.

## Registry (`rhmp.tasks.suite.SUITE_TASKS`)

| name | inputs (degree: columns) | target | split / extra tests | defaults (C, L, batch) |
|---|---|---|---|---|
| `SURF` | 0: f | u, screened Poisson (node) | 3989/855/856; extra: `geo`, `topo`, `test_ellipsoid`, `test_sphere_pert`, `test_torus` | 128, 4, 8 meshes |
| `SURF_heat` | 0: f | heat flow u_T (node) | as SURF | 128, 4, 8 |
| `SURF_geo`, `SURF_topo`, `SURF_heat_geo`, `SURF_heat_topo` | 0: f | u / u_T | test only: (0, 0, 500) | – |
| `DYN` | 0: u_t; 1: theta (odd) | u_{t+1} (node) | 10500/2250/2250 windows (420/90/90 trajectories × 25) + `task.rollout` (90 × 100 steps) | 128, 4, 16 meshes |
| `DYN_delta` | as DYN | u_{t+1} − u_t | as DYN | 128, 4, 16 |
| `DYNfix`, `DYNfix_delta` | as DYN, one shared mesh | as DYN | 8750/1875/1875 (350/75/75 × 25) + rollout (75 × 100) | 128, 4, 64 windows |
| `DYNfix_cons` | as DYNfix | du = u_{t+1} − u_t (scale-only), predicted through a fixed exactly-conservative map (default: flux density, readout `cochain:1`) | as DYNfix | 128, 4, 64 |
| `DYN_cons` | as DYN | as DYNfix_cons, with the model readout `mdiv:1` (division by the lumped mass inside the model) | as DYN | 128, 4, 16 |
| `DYN_cons_mass`, `DYNfix_cons_mass` | as DYN | first convention, M⊙du with readout `div:1`; ill-conditioned, kept to evaluate old runs | as DYN / DYNfix | as DYN / DYNfix |
| `HP_qual_graded`, `HP_qual_sliver` | as HP_k100 (0: f; 1: log sigma; 2: log sigma) | u, the P1 solution on the shifted mesh | test only: (0, 0, N); one extra test per quality level | – |
| `HP_qual_graded_ref`, `HP_qual_sliver_ref` | as above | u_ref, the 4x-finer reference interpolated to the nodes | as above | – |

Legacy mode (`native=False`): SURF and QUAL behave as their native versions; SURF has node inputs only, and QUAL uses
the HP legacy node encoding. For DYN, legacy feeds `[u_t, v_x, v_y]` at the nodes, i.e. frame-dependent velocity
components as a v1-style model would take them.

Registry: `rhmp.tasks.load_task`, `task_defaults` and `list_tasks` resolve these names (the suite module is
imported lazily), so `python3 -m rhmp.train --task SURF ...` works like any other task; `scripts/eval_on.py`
evaluates a trained run on another task or rolls it out.

---

## SURF: scalar PDEs on variable closed surfaces

### Surfaces (`surfaces.py`)
Every quad of every UV grid is split along a random diagonal, so connectivity changes from sample to sample and
valences range over 3–9. All meshes are closed and consistently oriented counter-clockwise w.r.t. the outward normal,
with no duplicate vertices. The generator asserts closedness, manifoldness, orientation, connectedness, genus, positive
volume, and the absence of degenerate or duplicate faces. The loader checks that `CochainComplex` drops no face and
that `check_d2() == 0`.

* **Genus 0.** An equal-angle cube-sphere: six (n+1)² UV grids on the cube faces, projected to the sphere, with seam
  vertices merged by a KD-tree, giving V = 6n²+2. It is randomly rotated, then mapped radially onto a star-shaped
  surface x = r(ω) ω. Then 30 iterations of tangential umbrella smoothing with radial re-projection equalise the edge
  lengths for any shape map. Radial functions:
  * (a) ellipsoid: r = (Σ ω_i²/a_i²)^(-1/2), with a ~ U[0.5, 1.5]³.
  * (b) superquadric: F(x) = (|x/a|^{2/e2} + |y/b|^{2/e2})^{e2/e1} + |z/c|^{2/e1} is homogeneous of degree 2/e1, so
    r = F(ω)^{-e1/2}. Here a ~ U[0.5, 1.5]³ and e1, e2 ~ U[0.3, 1.4], which gives boxy shapes for e < 1 and pinched
    ones for e > 1.
  * (c) perturbed sphere: r = 1 + Σ_{l=2..4} Σ_m c_lm Y_lm(ω), with real orthonormal SH computed via
    `scipy.special.lpmv`, c_lm ~ N(0, l⁻²), rescaled so that max|r − 1| = amp ~ U[0.1, 0.35].
* **Genus 1.** (d) A periodic UV-grid torus with nu = round(2πR/h) cells around the ring and nv = round(2πr/h) around
  the tube; r/R ~ U[0.25, 0.5]. The tube radius is r(1 + δ(u,v)), where δ is a sum of 4 random Fourier modes with
  |m|, |n| ≤ 2 and max|δ| ~ U[0, 0.15].
* **Genus 2.** (e) Two mirror-image torus grids (r/R ~ U[0.3, 0.5]) centred at ∓D along x, with D = R + r + gap/2 and
  gap ~ U[0.6, 1.6]·k·h. Each torus gets a 2k×2k-cell hole around the point facing the other (k ∈ {2, 3, 4}). The
  8k-vertex hole loops are joined by a neck of rings. Each loop vertex p_A is joined to its mirror image p_B by the
  cubic Hermite curve with end tangents τ w_A and −τ w_B. Here w is the unit tangent (in the torus tangent plane)
  pointing from the loop vertex to its hole centre, and τ = τ_f |p_A − c_A| with τ_f ~ U[0.8, 1.4]. The neck therefore
  leaves both tori tangentially (a G1 join) with a waist of ≈ 1 − τ_f/4 of the hole size. Rings are placed at equal
  arc length (spacing ≈ h), and the neck plus 2 grid rings around each hole are relaxed by 12 umbrella iterations.
  This is a pure grid construction: meshes keep the style of the genus-0/1 families (split quads, valence ~6), so
  the genus-2 test isolates topology and geometry from meshing style (no `skimage` needed).

**Resolution and size.** Each shape is rescaled to a surface area A ~ U[7.5, 15]; for the double torus each lobe is
drawn like a family-(d) torus, so the total area is ≈ 15–30. Shapes are meshed with a median edge length h ≈ 0.07,
calibrated once for genus 0. Measured median edge lengths: 0.067–0.078 over all 6700 surfaces. This *fixed physical
resolution* is deliberate. The v2 model is exactly invariant to the length unit: stars are normalised per sample,
geometry descriptors are log(x/median), and operators are Gershgorin-normalised. A PDE with an absolute length scale
(ε, T below) is therefore learnable only if (PDE length)/(mesh spacing) is the same in every sample. Here
√ε/h ≈ 3.2 edge lengths. Each surface also gets a random rotation and a translation U[−0.5, 0.5]³. Coordinates never
enter v2; coordinate-based baselines must learn the invariance.

Mesh quality (min angle / max angle / 99th-percentile aspect R/(2r) / share of negative cotan weights):

| family | min angle | max angle | p99 aspect | neg. cotan | ms/sample |
|---|---|---|---|---|---|
| ellipsoid | 15.5° | 146.7° | 1.41 | 0.5 % | 97 |
| sphere_pert | 17.6° | 143.5° | 1.39 | 0.4 % | 284 |
| torus | 21.4° | 94.7° | 1.37 | 16.6 %² | 31 |
| superquadric (geo) | 15.7° | 147.2° | 1.43 | 0.6 % | 101 |
| double torus (topo) | 19.8° | 126.3° | 1.36 | 15.4 %² | 64 |

² Tiny negative weights on the diagonals of the nearly rectangular torus cells, where the opposite angles are ≈ 90°.
rhmp's cotan star clamps them at 1e-2 · median; the generator uses the unclamped operator.

### Physics and discretisation (float64)
Positions are rounded to float32 *before* the solve, so the stored mesh reproduces the operators exactly.
* L = d0ᵀ diag(w) d0 with w_e = (cot α + cot β)/2, unclamped. This is the P1-FEM Laplace–Beltrami stiffness:
  PSD, with L1 = 0 and 1ᵀL = 0. M = diag(barycentric dual areas), the lumped mass, identical to rhmp's `star0`
  (checked in the tests, as is w = rhmp `star1` wherever unclamped).
* Source f is an embedding-space Gaussian random field (64 random Fourier features, length scale U[0.25, 0.6], unit
  variance) plus 1–4 Gaussian bumps (centres at random vertices, widths U[0.1, 0.3], amplitudes ±U[1, 2.5]),
  evaluated at the vertices. std(f) = 1.05.
* **SURF**: the screened Poisson equation u − ε Δ_LB u = f, discretised as (M + εL) u = M f with ε = 0.05 (SuperLU).
  std(u) = 0.77.
* **SURF_heat**: u_T = exp(TΔ_LB) f (heat flow ∂_t u = Δ_LB u for time T; Δ_LB ≤ 0), from 10 implicit Euler steps (M + dt L) v^{k+1} = M v^k with
  dt = T/10 and T = 0.05 (= ε, so both operators agree to first order in the spectrum). std(u_T) = 0.73.
* Measured: relative residuals ≤ 9.1e-15 (u) and ≤ 1.5e-15 (heat, max over steps). Heat mass drift ≤ 1e-15. Every
  surface was valid on the first attempt.

### Protocol
Train/val/test are families (a)+(c)+(d), interleaved (sample i → family i mod 3) with a sequential 70/15/15 split,
so every part is balanced. Transfer tests: `geo` (superquadrics: flat faces, high-curvature edges, pinched tips) and
`topo` (genus 2). Both are normalised with the SURF training statistics, and the trainer reports them automatically
as `geo_*` / `topo_*` next to `test_*` and the per-family in-distribution subsets.

### Structure metrics to report
* Accuracy: R², NRMSE, SSIM, Pearson on `test`, `test_<family>`, `geo`, `topo`. Report the drop from `test_torus` to
  `topo`: this is the topology transfer, at the same lobe geometry, resolution and meshing style.
* **Integral constraint** (`suite.surf_integral_error`): both targets conserve Σ M u = Σ M f exactly, because 1ᵀL = 0
  in every solve. Report mean and max of |Σ M u_pred − Σ M f| / Σ M |f|. The stored targets give ~1e-7 (float32).
* Symmetries: E(n) (random rotation/reflection of `pos`; exact for v2, informative for coordinate baselines), vertex
  relabelling, and face-orientation convention (flip a random subset of faces).
* Physics residual of a prediction: ‖(M + εL) u_pred − M f‖ / ‖M f‖, computable from `surfaces.cotan_operators`.

---

## DYN: advection–diffusion rollouts with a velocity 1-form

### Generator (`gen_dyn.py`)
* **Mesh.** DYN uses a new random Delaunay mesh of [0,1]² per trajectory (`gen_HP.draw_mesh_2d`: jittered boundary
  nodes, CCW), with n0 ~ U{1300..1700}. DYNfix uses one fixed 1500-node mesh (seed 2041).
* **Velocity.** The stream function is ψ = Σ_{m,n=1..4} a_mn sin(mπx) sin(nπy), with a_mn ~ N(0,1)(m²+n²)⁻¹ and
  ψ = 0 on ∂Ω. The velocity is v = rot ψ = (∂_yψ, −∂_xψ): div v = 0 and v·n = 0 on ∂Ω. It is rescaled so that
  max|v| = U ~ U[0.6, 1.0]. With ν = 0.01 this gives Pe = UL/ν ∈ [60, 100] and a mesh Péclet number ≤ 3.05.
* **Model inputs.** The edge 1-form θ_e = (v_i + v_j)/2 · (x_j − x_i) on the canonical edges i < j (odd, degree 1;
  the trapezoidal line integral) and the current state u_t (nodes). The solver's dual fluxes are *not* given to the
  model. Mapping θ to fluxes is a Hodge star, which is what the learned metric H₁ is for.
* **Initial state.** 2–5 Gaussian blobs: centres U[0.15, 0.85]², widths U[0.06, 0.15], amplitudes ±U[0.5, 1.5].

### Discretisation: finite volumes on the barycentric dual = DEC with the lumped mass (float64)
* **Dual-edge fluxes from the stream function.** Φ_{i→j} = ψ(b_left) − ψ(b_right), where b are the barycentres of
  the triangles left and right of the edge i→j, and ψ := 0 at boundary-edge midpoints. The flux around every dual
  cell telescopes to zero, so Φ is **exactly** discretely divergence-free: measured max|d0ᵀΦ| / max|Φ| ≤ 4.9e-16.
  This is what makes u = const a discrete steady state.
* **Advection** in central flux form: (A u)_i = Σ_j Φ_{i→j}(u_i + u_j)/2. We have 1ᵀA = 0 exactly (mass
  conservation for any u), A1 = div Φ = 0, and A is skew off the diagonal. An optional upwind blend adds α D, with
  D the graph Laplacian with weights |Φ|/2 (conservative, PSD); α = 0 is used.
* **Diffusion.** ν L with L the cotan/P1 stiffness (zero row and column sums, i.e. a no-flux boundary). Mass M is the
  barycentric lumped mass.
* **Time stepping.** Crank–Nicolson, (M + dt/2 (A + νL)) u^{n+1} = (M − dt/2 (A + νL)) u^n, with dt = 0.005 and
  8 sub-steps per stored step. That gives Δt = 0.04, a stored-step CFL UΔt/h ≈ 1.1, 100 stored steps and T = 4.
  The scheme is unconditionally stable: A + νL has a PSD symmetric part in the M inner product. It is mass-conserving
  to round-off; one sparse LU per trajectory is reused for all steps.
* **Measured.** Per-step mass residual |1ᵀM(u^{n+1} − u^n)| / 1ᵀM|u^n| ≤ 1.4e-16; total drift over 800 sub-steps
  ≤ 2.6e-14 (stored per trajectory). Max-principle overshoot (central scheme) is 1e-4 of the initial range on
  average, max 0.96 %. The field keeps 17 % of its initial std at T = 4.
* The upwind blend makes the overshoot 0 but adds numerical diffusion ~αUh/2 ≈ 1.4αν. It is not used.
* Persistence baselines on the state target: one step R² = 0.994; 10 steps R² = 0.57. The `_delta` variants therefore
  train on increments, which rollouts add back.

### Protocol
* Split by trajectory (70/15/15). One-step windows (u_t, θ) → u_{t+1} every 4 steps (t = 0, 4, …, 96; `windows=`
  selects another count, `'all'` = 100).
* `task.rollout` holds the full test trajectories: physical states, θ, velocities, lumped mass, complexes.
* `rollout_eval(model, task, steps)` feeds the denormalised prediction back for up to 100 steps. It returns R² (v1
  formula in the normalised space), NRMSE, the prediction's mass drift |Σ M u_pred(t) − Σ M u_0| / Σ M|u_0| (mean and
  max over trajectories), the true drift, the persistence R² per horizon, `summary` at horizons 1/5/10/25/50/100,
  and the first non-finite step.
* `scripts/eval_on.py --run RUN --task DYN --rollout` writes `rollout_DYN.json`.
* **Exactly conservative variants.** The conserved quantity is Σ M u (M = lumped mass). Any update that conserves
  it exactly for arbitrary network weights has the form du = M⁻¹ · (a zero-sum vector). The zero-sum vector comes
  from d₀ᵀ: the `div:1` readout, or d₀ᵀ inside a fixed output map.
  * **First convention (`DYN_cons_mass`, `DYNfix_cons_mass`): wrong for rollouts.** The loss was a node-uniform MSE
    on the mass change y = M du (readout `div:1`), and rollouts used u ← u + y/M.
    * The lumped mass varies about 900:1 on the DYN meshes (Delaunay of uniform random points; M_min/M_median =
      0.008). Errors in y come out roughly uniform in absolute size, so they become ~1/M_i errors in du.
    * The trained `DYN_cons_s42` run (`results/cab75/suite/DYN_cons_s42`; 100 epochs, one-step R² 0.954 on y) has one-step errors of up to 6.8 in
      du at the smallest cells, against a typical |du| of 0.01. Its rollouts explode: R2@5 = −7e24, NaN from step 7.
      The largest errors sit exactly at the M/median = 0.008–0.03 nodes and grow about 700× per step.
    * Conservation itself held exactly: predicted y summed to 4e-10 per mesh.
    * The 2-epoch smoke model looked stable only because it predicted y ≈ 0.
  * **Corrected convention (`DYNfix_cons`).** The loss is on du itself (u-space, scale-only normalisation), and the
    conservative structure lives in a fixed output map between network and loss:
    * default `cons_param='flux'`: du = (M̄/M) d₀ᵀ(ℓ*/ℓ̄* ⊙ b), with b an odd edge cochain (readout `cochain:1`).
      This is a finite-volume update from a learned flux density, integrated over the barycentric dual edges (ℓ*)
      and divided by the dual area.
    * `cons_param='div'`: du = (M̄/M) d₀ᵀ b (readout `div:1`).

    Both give Σ M du = 0 exactly for any weights; unit tests check this with random models. The flux form has a
    smaller node-to-node dynamic range (~1/r_i instead of 1/M_i) and trains a little faster. **`DYN_cons`** (variable
    meshes) needs the division by M inside the model, because the trainer's output maps do not see the batch complex:
    the readout `mdiv:1` (`rhmp.readout.MassDivReadout`).
  * **Post-hoc conservation for any model.** `rollout_eval(..., project_mass=True)`, or `eval_on.py --project-mass`,
    shifts each trajectory by a constant after every step so that Σ M u is restored.

Rollouts after the fix (full test sets, 100 steps; R² at horizons 1 / 5 / 10 / 25 / 50 / 100):

| run | rollout R² | mass drift (mean) @10 / @25 / @100 |
|---|---|---|
| DYN_s42 (state; unchanged by the fix) | 0.999 / 0.966 / 0.889 / 0.219 / −3.76 / −20.35 | 0.12 / 0.98 / 5.13 |
| DYN_s42 + `--project-mass` | 0.999 / 0.966 / 0.896 / **0.655 / 0.252 / −0.33** | 0 / 0 / 0 |
| persistence (DYN test set) | 0.989 / 0.791 / 0.398 / −0.62 / −1.54 / −2.51 | – |
| DYN_cons_s42 (first convention) | 0.990 / −7e24 / NaN / NaN / NaN / NaN | NaN (float overflow) |
| DYNfix_cons, flux form, **8-epoch validation only** | 0.990 / 0.823 / 0.492 / −0.38 / −0.93 / −0.99 | **0 / 0 / 0** |
| DYNfix_cons, div form, 8-epoch validation only | 0.990 / 0.816 / 0.468 / −0.47 / −1.14 / −1.32 | 0 / 0 / 0 |
| persistence (DYNfix test set) | 0.989 / 0.792 / 0.373 / −0.89 / −2.13 / −3.24 | – |

Restoring the mass is what keeps the state model's long rollouts meaningful: its dominant error mode is mass drift.

The corrected conservative convention is stable and beats persistence at every horizon, but it learns slowly. After 8
epochs the one-step val R² on du is 0.076 (flux) or 0.060 (div), whereas DYNfix_delta reaches 0.44 after 2 epochs. A
likely reason: the advective flux Φ_e ū_e is bilinear in the odd edge input and the signed node state, and the odd
lifting (odd linear map times an even gate built from norms) can only form that product indirectly.

### Structure metrics to report
R² and NRMSE vs horizon (1–100) and the step at which R² < 0 or the rollout diverges; mass drift of the predictions vs
horizon (the reference dynamics keep it below 3e-14); max-principle violation of the predicted states; the same
curves on DYNfix (fixed mesh, the long-rollout stability study).

---

## QUAL: mesh-quality shift for HP models (test only)

### Generator (`gen_qual.py`)
* **Physical instances.** The first 100 *test* samples of `HP_k100` (ids 4250–4349 of N = 5000). They use the same
  seed (2026), the same conductivity/source fields (`gen_HP.draw_fields`, κ = 100) and the same node counts. Every
  instance is re-meshed at every level, so curves compare identical physical problems.
* **Solver.** `solve_on_mesh` is identical to `gen_HP.solve_sample` (isotropic): P1 FEM for −div(σ∇u) = f with u = 0
  on ∂Ω, centroid σ, lumped right-hand side. It is bitwise equal on the HP mesh (unit test), and the stored `base`
  level equals the stored HP_k100 samples exactly (max|Δu| = 0).
* **Stored keys.** All HP keys (pos, faces, f, u, bnd, sigma_edge, logsigma_face, flux, flux_fem), so the HP loader
  code (`synthetic._build_set`) is reused unchanged. Added: `u_ref`, the P1 solution on the 4x-finer HP reference mesh
  (the `HP_k100_fine` mesh) linearly interpolated to the nodes; `level` codes; per-sample quality statistics.

### Levels (means over 100 instances; aspect = circumradius / (2 · inradius))

| set / level | construction | p50 aspect | p99 aspect | share aspect > 10 | max angle | edge ratio | P1 error vs ref |
|---|---|---|---|---|---|---|---|
| base | original HP mesh (Delaunay of uniform random points) | 1.39 | 9.6 | 0.9 % | 171° | 343 | 0.68 % |
| graded corner_r4 / r16 / r64 | nodes mapped by x → (e^{bx} − 1)/(e^b − 1) per coordinate, b = ln ratio; re-Delaunay | 1.41 / 1.44 / 1.47 | 9.8 / 13.8 / 43.8 | 1.0 / 1.6 / 3.3 % | 172–179° | 564 / 1185 / 2814 | 0.9 / 1.6 / 2.9 % |
| graded circle_a10 / a30 / a100 | interior nodes resampled with density 1 + a·exp(−(\|x − c\| − 0.3)²/(2·0.04²)) | 1.43 / 1.46 / 1.48 | 10.1 / 10.4 / 10.8 | 1.0 / 1.1 / 1.2 % | 172–175° | 562 / 833 / 1289 | 1.2 / 2.4 / 4.8 % |
| sliver a2 / a3 / a4 / a6 / a8 | same nodes; Delaunay of the cloud stretched by `a` along a random direction | 1.75 / 2.52 / 3.58 / 6.25 / 9.86 | 17 / 34 / 59 / 126 / 221 | 2.8 / 8.3 / 17 / 35 / 50 % | 176–179° | 446 – 848 | 1.0 / 2.2 / 4.2 / 9.6 / 16.2 % |

The HP training meshes are Delaunay triangulations of uniform random points. They already contain single triangles
with angles of ~0.04° and aspect ratios up to ~900. The per-mesh **maximum** aspect is dominated by such outliers
(base: 877; a8: 1.2e4), so the p99 aspect and the share of triangles with aspect > 10 describe the levels better.
With stretches a ≥ 16 the P1 solution itself becomes meaningless: the error vs the reference is 47 % at a = 16 and
99 % at a = 300, because the maximum-angle condition is violated. So the sliver levels stop at a = 8, where the
per-mesh maximum aspect is ~1.4e3 (median over instances).

### Protocol and metrics
* Evaluate an HP_k100 run with `python3 scripts/eval_on.py --run RUN --task HP_qual_sliver` (or
  `python3 -m rhmp.train --task HP_qual_sliver --eval-ckpt RUN`). Inputs are re-expressed with the run's
  statistics, and the result has one entry per quality level plus `quality`, the mean mesh statistics per level.
* Plot the error vs level (the accuracy-vs-quality curve per model) for both targets. `u` is the discrete solver on
  the shifted mesh, i.e. the HP definition. `u_ref` is the physical solution; use it for the sliver levels, where the
  P1 solution on the shifted mesh carries up to 16 % discretisation error. Report the solver's own error column above
  as the floor.
* **Use the mass-weighted relative L2 error** `relL2M` = ‖u_pred − u‖_M / ‖u‖_M (M = lumped mass = `K.star[0]`;
  `suite.mass_weighted_errors`, reported by `eval_on.py` for node-scalar tasks) to compare levels. Node-pooled
  R² / NRMSE weight every node equally, so on graded meshes they depend on where the nodes are. Example: a 2-epoch
  HP_k100 smoke model on `HP_qual_graded` gives, for base → corner_r64, pooled R²
  0.705 → 0.381, NRMSE 0.035 → 0.030 (it *improves*: the nodes cluster at the Dirichlet corner where u ≈ 0), and
  relL2M 0.534 → 0.739 (the actual degradation).
* The same model on the other sets, per level (pooled R²):
  * `HP_qual_sliver`: base 0.705, a2 0.694, a3 0.676, a4 0.645, a6 0.580, a8 0.490.
  * `HP_qual_sliver_ref`: a8 0.541.
  * `HP_qual_graded` circle levels: 0.72–0.73 (relL2M 0.54 → 0.61).

  Results are in `results/cab75/e_qual_eval/eval_*.json`. Evaluation takes about 1 min per set on a shared GPU.

---

## Tests (`tests/test_suite_tasks.py`, 35 tests, ~5 s, CPU plus a CUDA-only transfer test)
Mesh validity and genus for every family, `d²=0`, no dropped faces, Euler characteristic. Operators vs the complex
(star0 = lumped mass, star1 = cotan weights where unclamped, L = d0ᵀdiag(w)d0). Solver residuals < 1e-8, constants
preserved, heat mass conservation, stored float32 mesh reproducing the solve. DYN: fluxes exactly divergence-free,
advection conservative and skew, upwind PSD, mass residual < 1e-12, constants steady. QUAL solver bitwise equal to
`gen_HP.solve_sample`; QUAL levels valid (CCW, cover the square, no degenerate faces). End to end: tiny data sets are
generated into a temporary directory and every loader is checked (split sizes, shapes, degrees, extra sets, test-only
views, legacy mode). `rollout_eval` is checked with an oracle model (R² = 1 at every horizon, drift at round-off) and
with a small RHMP. With the `div:1` readout (`DYN_cons`, `DYNfix_cons`), a randomly initialised RHMP conserves Σ M u in rollouts to < 1e-5 while the state changes. `surf_integral_error` and `mass_weighted_errors` are checked. One trainer epoch on tiny SURF and DYN through `rhmp.train.run(args, task=...)`.
The pinned-memory complex transfer is checked to equal `K.to('cuda')` field by field.

---

## Smoke runs (v2 defaults C=128, L=4, 2 epochs, seed 42; one GPU shared with ~10 other processes)
Pipeline validation during development; times are inflated by GPU sharing.  The 100-epoch results are in
[../REPORT.md](../REPORT.md) §6.2 and §6.7.

| task | val R² ep1 / ep2 | test R² | extra tests (R²) | s/epoch | peak GB |
|---|---|---|---|---|---|
| SURF | 0.929 / 0.953 | 0.952 | test_ellipsoid 0.954, test_sphere_pert 0.957, test_torus 0.943, **geo 0.953**, **topo 0.943** | 269 (complexes on CPU) | 1.86 |
| DYN | 0.9955 / 0.9970 | 0.9971 | – | 334 | 10.8³ |
| DYNfix | 0.977 / 0.990 | 0.989 | – | 44 | 6.4 |
| DYNfix_delta | 0.349 / 0.438 (increments) | 0.438 | – | 45 | 6.4 |
| DYNfix_cons_mass (first convention) | 0.015 / 0.038 (mass increments, `div:1`) | 0.035 | – | 45 | 6.4 |

³ Includes the trainer's cache of the batched validation complexes (DYN complexes stay on the GPU).

Rollouts over the 100-step test trajectories (`rollout_eval`), R² at horizons 1 / 5 / 10 / 25 / 50 / 100, with the
persistence baseline u(t) = u_0 in brackets; drift = mean |Σ M u_pred(t) − Σ M u_0| / Σ M |u_0| at 10 / 100:

| task | rollout R² | drift @10 / @100 | max drift, all trajectories and horizons |
|---|---|---|---|
| DYN (90 traj.) | 0.990 / 0.829 / 0.543 / −0.10 / −0.83 / −2.64 (0.989 / 0.791 / 0.397 / −0.62 / −1.54 / −2.51) | 0.16 / 1.49 | 7.7 |
| DYNfix (75 traj.) | 0.971 / −1.96 / −7.88 / −17.5 / −27.9 / −38.1 (0.989 / 0.792 / 0.373 / −0.89 / −2.13 / −3.24) | 1.12 / 1.04 | 4.2 |
| DYNfix_delta | 0.995 / 0.898 / 0.699 / 0.210 / −0.096 / −0.422 (as DYNfix) | 0.12 / 0.61 | 1.9 |
| DYNfix_cons_mass (first convention, 2 epochs, y ≈ 0) | 0.988 / 0.787 / 0.369 / −0.85 / −1.74 / −2.17 (as DYNfix) | 0 / 0 | **4.0e-8** (data: 9.0e-9) |

Reading (2 epochs, i.e. pipeline validation, not final accuracy):
* Geometry and topology transfer on SURF are essentially lossless for the local operator: topo 0.943 vs 0.943 on
  in-distribution tori.
* On the state target, one-step R² is uninformative (persistence already scores 0.989). The increment target
  (`_delta`) is the better training signal and beats persistence at every horizon.
* The first conservative convention (then called DYNfix_cons, now `DYNfix_cons_mass`) kept the mass exactly, as
  the readout (d₀ᵀ) guarantees. It diverges once trained (see "Exactly conservative variants" above).
* Timing: SURF 269 s/epoch, DYN 334 s/epoch, DYNfix 44 s/epoch on the shared GPU. The ~0.7 s per SURF batch of
  20K nodes is ~10× slower than with an exclusive GPU.
