# RHMP v2 task suite — designed around what a metric-learning Hodge model should be good at

This is the design document of the suite; the implemented details are in
[TASK_SUITE_DETAILS.md](TASK_SUITE_DETAILS.md) and [ANISO_TASKS.md](ANISO_TASKS.md), and the results are in
[../REPORT.md](../REPORT.md).

Principle: every field lives on the cochain degree where its physics lives (scalars on nodes, 1-forms on edges,
2-forms on faces, densities on top cells); targets come from trusted discrete solvers (FEM/DEC sparse solves);
protocols expose *structure* (variable meshes, resolution/geometry/topology transfer, material contrast, exact
constraints, scale) instead of one fixed mesh. Each task lists its inputs and outputs by degree, what it shows, its
protocol, its metrics and the applicable baselines. The generators are in `datasets/generators/` (seeded) and the
adapters in `rhmp/tasks/`.

The common metrics are R2 and NRMSE (with SSIM and Pearson correlation where the target is a per-node field); cost is
measured as ms/step, peak GB and parameter count. The structure metrics, reported next to accuracy, are constraint
residuals defined per task, symmetry errors (E(n), vertex relabelling, orientation convention, batch composition) and
gauge-noise robustness.

## Core tasks

### HP — heterogeneous / anisotropic conductivity Poisson on variable 2-D meshes   (tasks `HP*`)
* Each sample has a new random Delaunay mesh of the unit square (1–2K nodes, with boundary) and a log-normal
  conductivity field sigma(x) with contrast kappa in {10, 100, 1000, 10000} (`HP_k<kappa>`), optionally with an
  anisotropy tensor (`_aniso`). The PDE `-div(sigma grad u) = f` with u = 0 on the boundary is solved with P1 FEM / DEC.
* Inputs: f (node, degree 0); sigma at edge midpoints (EVEN edge input, degree 1); log sigma / tensor invariants at faces (EVEN, degree 2); the boundary flag comes from the geometry.
* Targets: `HP`: u (node). `HPflux`: j = -sigma d0 u (edge 1-cochain, odd). `HPgrad`: E = -d0 u through the exact
  readout `grad` (predict a node potential, apply d0) — curl-free by construction.
* Shows: whether the learned metric matches the material response (corr(log H_1/star_1, log sigma)); conditioning at high contrast
  (the paper's stated limitation); variable meshes with block-diagonal batching; zero-shot **4x resolution transfer**
  (test split generated at 4x finer meshes, same sigma/f generators); graded/sliver test meshes (`QUAL` shift).
* Structure metric: conservation residual at interior nodes `|| d0^T (star1 ⊙ j_pred) - star0 ⊙ f ||` for flux targets;
  `|| d1 E ||` for vector/edge E predictions (exactly 0 for `grad`).
* Baselines: mgn, egnn, gat, gcn, schnet (node encodings of edge/face inputs), cw_net, sccnn, mpsn (triangle complexes), dec_fixed, unit_star; `ours_v1` is not applicable (per-cell parameters).

### TET — 3-D Poisson on random tetrahedral meshes   (tasks `TET*`)
* Random points in the unit cube are triangulated into Delaunay tets (2–4K nodes), with a conductivity per tet (EVEN
  degree-3 input) and a source f.
* Targets: u (node); `TETflux`: flux through faces (2-cochain in 3-D).
* Shows: degree-3 complexes, face fluxes in 3-D, variable meshes. Baselines: node-based models only (plus
  dec_fixed/unit_star).

### T6 / T6f / T7f native gauge tasks + `T6g` gauge-noise robustness   (T6g = an evaluation-time transform)
* Edge connection theta (odd, `connection_dims={1:1}`) → face flux (2-cochain) [T6f], or node target via the oriented
  face→node map [T6]. `T6g` adds random gauge transformations `theta + d0 lambda` at test time (lambda ~ N(0, s^2),
  s in {0.1, 1, 10}); the v2 model in connection mode is exactly invariant, and the degradation of each model is
  reported.
* Mesh transfer (`T6m`): a trained T6f model is evaluated on a *different* random Delaunay mesh from the same generator
  (500 test samples regenerated on a new mesh), which is impossible for v1 by construction.

### SCALE — T6_100K and a 1M-cell forward/backward budget
* Time and memory of v2 (C=16, L=3) vs v1 (`bench/RESULTS_step.md`; the 100K accuracy study of the v1 paper used 30
  epochs and 200 training samples).

## Extension tasks (`rhmp.tasks.suite`)

### SURF — scalar PDE on variable closed surfaces
* Each sample is a different surface (random ellipsoids / superquadrics / smoothly perturbed spheres and tori),
  triangulated from a parametric UV grid (~2K nodes); the screened Poisson equation `u - eps * Lap_LB u = f` is solved
  with the cotan Laplacian (DEC), or heat diffusion runs for a fixed time.
* Inputs: f (node). Target: u (node). Shows: E(n) invariance, intrinsic geometry through the DEC star (curvature),
  geometry transfer (train ellipsoids → test superquadrics) and **topology transfer** (train genus 0 and 1 → test genus 2
  double torus): d^2 = 0 and local metrics make this well-defined, whereas baselines with absolute coordinates are
  expected to fail.
* Baselines: node-based, cw_net/sccnn/mpsn, dec_fixed.

### DYN — advection–diffusion rollouts with a 1-form velocity
* Random planar meshes (or a fixed mesh for the rollout-stability study) carry a divergence-free velocity from a random
  stream function, given to the model as the edge 1-form `v·(x_j - x_i)` (odd degree-1 input); implicit DEC time
  stepping generates the trajectories; the model predicts the next state and is evaluated in 50–100-step
  autoregressive rollouts.
* Shows: cochain velocity input, stability of bounded operators (||L|| <= 1) over long rollouts vs baselines.

### QUAL — mesh-quality shift
* HP models are evaluated on graded meshes (refined near a corner) and sliver-heavy meshes (aspect up to 1e3) generated
  with the same physics, which gives an accuracy-vs-mesh-quality curve per model.

## Paper tasks (reference, native formulation)
The paper tasks are T1/T1q (edge velocity 1-form → face vorticity → node map), T2, T3 (node_vector), T5 (node_vector),
T6/T6f, T7/T7f and T8 (variable meshes, block-diagonal batches). Legacy modes reproduce the v1 inputs and outputs
verbatim for transparency. The legacy targets of T1/T5/T6/T7 are frame-dependent pseudo-scalars that an exactly
invariant model cannot represent; v1 represents them through its per-cell parameters.
