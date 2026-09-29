# Dataset file formats

Where the files come from and how to regenerate them: [datasets/README.md](../datasets/README.md).  Loaders:
`rhmp.tasks.load_task(name, root, ...)` (paper tasks in `rhmp/tasks/paper.py`, HP / TET in `rhmp/tasks/synthetic.py`,
SURF / DYN / QUAL in `rhmp/tasks/suite.py`, the anisotropy suite in `rhmp/tasks/aniso.py`).

## v1 paper pickles (`datasets/*.pkl`)

Shared-mesh sets are dicts with `points (n0, D)`, `faces (n2, 3)`, `X_data (N, n0, F)` (or `(N, n0)`) and
`Y_data (N, n0, O)`.  v1 protocol: sequential 70/15/15 split, per-feature mean / std of the training part
(`X[:nt].mean((0, 1))`, `.std((0, 1)).clamp(1e-6)`).

| file | points | faces | X_data | Y_data | notes |
|---|---|---|---|---|---|
| T1_cns_vorticity.pkl | (1024, 2) f64 | (1922, 3) | (10000, 1024, 4) | (10000, 1024, 1) | regular 32x32 grid triangulated; (rho, Vx, Vy, p)_t -> vorticity_{t+1}; PDEBench CNS |
| T2_torus_advection_diffusion.pkl | (1711, 3) | (3422, 3) | (3000, 1711) | (3000, 1711) | scalar transport on a torus, no velocity input |
| T3_ellipsoid_coexact.pkl | (1157, 3) | (2310, 3) | (10000, 1157, 1) | (10000, 1157, 3) | v = n x grad(psi): tangent vector field from a stream function (node_vector readout) |
| T5_maxwell_poisson.pkl | (1024, 3) f64 | (2027, 3) | (5000, 1024, 1) | (5000, 1024, 2) | charge density -> E (2-D vector at nodes); v1 treated the 2 components as scalars |
| T6_wilson_loop.pkl | (1024, 3) f64 | (2027, 3) | (10000, 1024, 3) | (10000, 1024, 1) | U(1): X = node encoding (avg, avg dx, avg dy) of the edge angle theta; Y = plaquette flux averaged to nodes |
| T6_100K_wilson_loop.pkl | (50010, 3) | (99982, 3) | (500, 50010, 3) | (500, 50010, 1) | same physics, 100K faces |
| T7_yang_mills_su2.pkl | (1024, 3) f64 | (2027, 3) | (10000, 1024, 9) | (10000, 1024, 3) | SU(2): 3 channels x (avg, avg dx, avg dy); F = dA + [A, A] (3 components at nodes) |
| T7_small_yang_mills.pkl | (256, 3) | (494, 3) | (5000, 256, 9) | (5000, 256, 3) | small variant used for the metric-family ablation of the paper |
| T8_airfoil_pressure_persample.pkl | list of 1000 dicts | | | | keys `pts_3d (n, 3)`, `faces (m, 3)`, `x_pres (n, F)`, `y_pres (n, 1)`; about 2000 nodes each; AirfRANS; split 70/30 (first 700 train), no validation split in v1 |

The third coordinate of the planar meshes (T5/T6/T7/T8) is 0; T1 points are 2-D.  The raw edge cochains of T6/T7 are
not stored in the pickles; `datasets/generators/gen_T6_native.py` / `gen_T7_native.py` regenerate them with the v1
generators' random streams (the node encodings are reproduced to 1e-7) into `datasets/v2/T6_native.pt` /
`T7_native.pt`.  The v1 checkpoints (seeds 42, 1, 2; `python3 datasets/download_v1.py --ckpt`) live under
`checkpoints_v1/{task}/{model}/best_model.pt` for `python -m rhmp.train --eval-v1`.

## v2 packed sets (`datasets/v2/*.pt`)

`torch.save`d dicts.  Variable-mesh sets store all samples as flat concatenations with int64 offsets `ptr0 ... ptr3`
(`N + 1` entries each; sample `i` owns rows `ptr_k[i]:ptr_k[i+1]` of every degree-k array), float32 fields, int32
cells, plus `meta` (generator arguments, seeds, residuals) and, for most sets, a `.json` summary next to the file.
Edges are the unique `src < dst` vertex pairs of the cells in lexicographic order, and edge quantities refer to the
orientation `src -> dst`; loaders align every stored cell order with the complex they build
(`rhmp.data.cell_alignment`, `edge_alignment`).  `*_fine.pt` files re-solve the first test-split instances on meshes
with 4x as many vertices (same physical fields): the zero-shot resolution-transfer sets.

| family | per-sample arrays (degree) | notes |
|---|---|---|
| HP (`gen_HP.py`) | `pos` (0), `faces` (2), `f`, `u`, `bnd` (0), `sigma_edge`, `sigma_proj_edge` / `sigma_dir_edge` (1), `logsigma_face`, `logdet_face`, `logratio_face` (2), `flux`, `flux_fem` (1); evaluation only: `theta_face`, `sigma_tensor_face` | `-div(sigma grad u) = f`, u = 0 on the boundary, P1 FEM with element-wise sigma, lumped mass |
| TET (`gen_TET.py`) | `pos`, `tets`, `f`, `u`, conductivity per tet, face fluxes | 3-D P1 FEM on random tetrahedral meshes |
| SURF (`gen_surf.py`) | `pos`, `faces`, `f`, `u` (screened Poisson), `heat` (heat flow), `family` | closed surfaces at a fixed physical resolution (median edge 0.07) |
| DYN (`gen_dyn.py`) | `pos`, `faces`, edge velocity 1-form, 101 states per trajectory | implicit DEC advection-diffusion with a divergence-free velocity |
| HP_qual (`gen_qual.py`) | HP_k100 test instances on graded and sliver meshes, `u` and the 4x-finer reference `u_ref`, per-level quality statistics | test only |
| AHP / ASURF / ACURL / ADARCY (`gen_aniso.py`) | `pos`, `faces` or `tets`, sources, targets (`u`, `heat`, `A`, `J`, `p`), edge projections `logproj_edge`, cell invariants (`logdet_*`, `logratio_*`), true tensors (`sigma_face`, `nu_tet`, `K_tet`) | exact P1 / Nedelec / RT0 solves; `ANISO_representability.json` holds the oracle analysis |

Full field lists, equations and sizes: the module docstrings of the generators, [TASK_SUITE_DETAILS.md](TASK_SUITE_DETAILS.md)
and [ANISO_TASKS.md](ANISO_TASKS.md).  `T8_complexes_<star>.pt` is a cache of the T8 complexes written by the loader.
