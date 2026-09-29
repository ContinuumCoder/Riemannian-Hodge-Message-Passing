# Theory notes: when is the learned cochain metric the material?

These notes state precisely what the RHMP v2 operator family can represent, what is identifiable from data, and why
the stage-1/2 models learned accurate solvers whose metrics were *not* the material (measured in
[../REPORT.md](../REPORT.md) §1.4 and §6.3). They motivate the stage-3
changes (`material_dims`, `layer_type='solve'`, the operator-identification loss).

Notation: complex `K` with coboundaries `d_k`, reference diagonal Hodge stars `star_k > 0`, learned metrics `H_k`
(diagonal `h_k > 0`, or the Whitney tensor metric of DESIGN §9.1). `M_k = diag(star_k)` is the lumped mass.
`A_k(H) := d_k^T H_{k+1} d_k` is the (un-normalised) up-operator on degree k; the model uses
`L_hat = S^{-1/2} A S^{-1/2} / beta` with the per-sample Gershgorin bound `beta`.

## 1. Representability (what the metric family can express)

**P1 finite elements with a tensor coefficient are exactly a metric Hodge operator.** For a triangle (tet) mesh and a
symmetric positive definite tensor `sigma_f` per top cell, the P1 stiffness matrix of `-div(sigma grad u)` is
`K = sum_f |f| (grad phi_i)^T sigma_f (grad phi_j) = d_0^T H_1^W(sigma) d_0`, where `H_1^W(sigma)` is the Whitney
1-form Galerkin star with coefficient `sigma` (`rhmp.dec.assemble_whitney_metric`; tested in `tests/test_dec.py`:
`d0^T H1(b=1,a=0) d0` equals the cotan stiffness). So with `metric_type='tensor'` the model's operator family contains
every P1 anisotropic diffusion operator, and with a `solve` layer it contains the exact discrete solution operator.

**Diagonal metrics are the M-matrix subclass on triangles.** Each edge `e = (i, j)` of a triangle mesh corresponds to
exactly one off-diagonal stiffness entry, so any P1 stiffness matrix can be written `K = d_0^T diag(w) d_0` with
`w_e = -K_ij` (the *edge conductance*: the cotan-weighted, face-averaged projected conductivity). A *positive*
diagonal metric (`h_1 > 0`) represents `K` iff all `w_e > 0`, i.e. iff `K` is an M-matrix. Strong anisotropy or
obtuse angles produce `w_e <= 0`, which no SPD diagonal `H_1` can represent; the tensor metric can (positive
combination of PSD element blocks). This is the first place where the tensor metric is *necessary*.

**Metrics on 1-forms and 2-forms are where diagonality really fails.** The Whitney mass matrix of 1-forms couples the
edges of every face (tets: of every tet), and the Nédélec curl–curl operator `d_1^T M_2(nu) d_1` couples faces
within each tet. A diagonal `H_1` (`H_2`) has no such coupling, whatever its values, so Hodge-Laplacian, curl–curl
and Darcy face-flux problems with tensor coefficients are outside the diagonal family (not just outside its
positivity cone). Tasks `ACURL*`, `ADARCY*`, `ASURF*` (docs/ANISO_TASKS.md) are built to expose this.

## 2. Identifiability (what data can pin down)

**Edge conductances are identifiable; face tensors only through their edge action.** From the operator `K` the
map `w -> d_0^T diag(w) d_0` is injective (edges <-> off-diagonal entries), so `w` is identifiable. The map
`{sigma_f} -> {w_e}` is linear with `3 n_2` unknowns (2-D symmetric tensors) and `n_1` equations; on a triangulation
`n_1 ~ 3 n_0`, `n_2 ~ 2 n_0`, so its kernel has dimension `~3 n_0`: per-face tensors are **not** identifiable from a
scalar Poisson operator alone. What resolves this in v2 is the parameterisation itself: `sigma_f = g_theta(local
invariants, material inputs)` with a *shared* MLP is a strong regulariser — if the true tensor is a function of the
given material inputs, the map `g_theta` is identifiable while individual `(b_f, a_{f,k})` are not (only the action
of `(b, a)` is identifiable; `tests/test_numerics.py::test_anisotropy_recovery` reproduces an anisotropic FEM action
from `(b, a)` to about 1e-7 without recovering the coefficients). Metric-recovery metrics therefore compare
*actions* (edge conductances `t_e^T sigma t_e`, principal directions/ratios of the assembled block) rather than raw
coefficients.

**From data pairs `(f, u)`.** For the diagonal metric the discrete PDE `d_0^T diag(w) d_0 u = M f` is *linear* in
`w`: `n_0` equations per sample, `n_1` unknowns, so three or more generic samples determine `w` (least squares over
the training set). This is the operator-identification loss of stage 3 (`--aux-pde`): a convex problem in `w`
whose solution is the material, obtained without any label on `sigma`.

**Scale gauge.** The normalised operator `L_hat` is invariant under `H -> c H` (per sample), so the global magnitude of
the material is *not* identifiable from `L_hat`; only its spatial pattern is. A solver layer must therefore use the
un-normalised operator (with `H = star exp(ref) exp(phi)`), or a separate scalar channel must carry the scale.

## 3. Why stage-1/2 metrics were not the material

1. **Route redundancy.** Material inputs (e.g. `log sigma_e`) entered the metric heads *and* the lifting gates and
   feature norms. With a scalar target and MSE, nothing prefers the metric route; the optimiser takes the easier
   feature route, and `H` is free to serve another purpose.
2. **Polynomial layers prefer preconditioners.** A layer `sum_p c_p L_hat^p x` approximates the *inverse* operator.
   For a degree-`P` polynomial the best diagonal metric is closer to a Jacobi-type preconditioner (`h ~ 1/sigma`) than
   to `sigma`, which is the sign observed (e.g. face `r = -0.73` in the last layer of a full-tensor polynomial run,
   `results/RESULTS.md` §4).
3. **No layer solves.** Even the resolvent layer `(I + tau L_hat)^{-1}` sits between nonlinear gates and a
   128-channel lifting that can absorb `sigma` elsewhere.

With `material_dims` the only path from `sigma` to the output is `H`; in solver mode the model *is* the FEEC solution
map, so MSE-optimal `H` equals the identifiable part of the true operator (Section 2), and the operator-identification
loss makes this explicit and convex in the diagonal case.

## 4. Resolvent and solve layers as discrete Green's functions

With `H_1 = star_1 sigma` and `S_0 = star_0`, `(I + tau L_hat_up)^{-1}` is the mass-lumped screened-Poisson solver
`(M + tau' K)^{-1} M` in the symmetrised frame (`tau' = tau / beta`), and `tau -> infinity` gives the Poisson solver on
the orthogonal complement of the constants. The `solve` layer removes the identity term (up to a small regulariser)
and applies the Dirichlet mask from `K.boundary`, i.e. it is the discrete Green's operator of the learned metric.
Because `||L_hat|| <= 1`, the CG condition number of the resolvent is `<= 1 + tau`; the solve layer's is that of the
discrete Laplacian (`~h^{-2}`), which is why it uses more iterations and benefits from warm starts and from the
tensor metric's exact element blocks.

## 5. What to measure

* accuracy (R2, NRMSE) and zero-shot resolution/mesh transfer;
* metric recovery: `corr(log h_e/star_e, log(t_e^T sigma t_e))` for diagonal metrics; principal-direction angle,
  eigen-ratio and log-det errors of the assembled per-cell tensor for the tensor metric;
* the operator residual `||A(H) u - M f|| / ||M f||` on held-out data (should be ~0 when the metric is the material);
* the diagonal/tensor gap on tasks of Section 1 (M-matrix violations, 1-form/2-form coupling), where the diagonal
  family is provably insufficient.
