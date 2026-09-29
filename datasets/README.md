# Datasets

Datasets are **not** part of the repository.  The loaders (`rhmp.tasks.load_task`, `python -m rhmp.train --task NAME`)
look for them under `datasets/` of the checkout, or under `$RHMP_DATA_ROOT` (the repository root or the datasets
directory itself), or under the `--root` / `root=` you pass.

```
datasets/
  README.md            this file
  download_v1.py       downloads the v1 paper pickles (and optionally the v1 checkpoints) from OSF
  generators/          seeded generators of the v2 sets (numpy / scipy; T6/T7 native also need torch + a GPU)
  *.pkl                v1 paper datasets (downloaded; git-ignored)
  v2/*.pt, v2/*.json   v2 datasets (generated; git-ignored), each .pt with a .json summary
```

## 1. v1 paper datasets (`datasets/*.pkl`)

These are the datasets of the paper "Learning Discrete Riemannian Metrics for Physical Fields with Cochain-Frame
Equivariance" (arXiv:2608.14556), released with the v1 code
(https://github.com/ContinuumCoder/Riemannian-Hodge-Message-Passing).  They are hosted on
[OSF](https://osf.io/35gcm/?view_only=0df302ed83154a8ba6bb1f8d8a1983ec) (1.1 GB archive):

```bash
python3 datasets/download_v1.py            # datasets/*.pkl
python3 datasets/download_v1.py --ckpt     # + v1 checkpoints into checkpoints_v1/ (only for rhmp.train --eval-v1)
```

T1 is derived from [PDEBench](https://github.com/pdebench/PDEBench) (Takamoto et al., NeurIPS 2022) and T8 from
AirfRANS.  All shared-mesh sets are dicts with `points (n0, D)`, `faces (n2, 3)`, `X_data (N, n0, F)` (or `(N, n0)`)
and `Y_data (N, n0, O)`; the v1 protocol is a sequential 70/15/15 split with per-feature statistics of the training
part.

| file | points | faces | X_data | Y_data | content |
|---|---|---|---|---|---|
| `T1_cns_vorticity.pkl` | (1024, 2) | (1922, 3) | (10000, 1024, 4) | (10000, 1024, 1) | regular 32x32 grid; (rho, Vx, Vy, p)_t -> vorticity_{t+1}; PDEBench CNS |
| `T2_torus_advection_diffusion.pkl` | (1711, 3) | (3422, 3) | (3000, 1711) | (3000, 1711) | scalar transport on a torus (the velocity field is fixed and not given) |
| `T3_ellipsoid_coexact.pkl` | (1157, 3) | (2310, 3) | (10000, 1157, 1) | (10000, 1157, 3) | `v = n x grad(psi)`: tangent field from a stream function |
| `T5_maxwell_poisson.pkl` | (1024, 3) | (2027, 3) | (5000, 1024, 1) | (5000, 1024, 2) | charge density -> electric field at vertices |
| `T6_wilson_loop.pkl` | (1024, 3) | (2027, 3) | (10000, 1024, 3) | (10000, 1024, 1) | U(1): node encoding (avg, avg dx, avg dy) of the edge angle; plaquette flux averaged to vertices |
| `T6_100K_wilson_loop.pkl` | (50010, 3) | (99982, 3) | (500, 50010, 3) | (500, 50010, 1) | same physics, 100K faces |
| `T7_yang_mills_su2.pkl` | (1024, 3) | (2027, 3) | (10000, 1024, 9) | (10000, 1024, 3) | SU(2): `F = dA + [A, A]`, 3 components at vertices |
| `T7_small_yang_mills.pkl` | (256, 3) | (494, 3) | (5000, 256, 9) | (5000, 256, 3) | small SU(2) variant (metric-family ablation of the paper) |
| `T8_airfoil_pressure_persample.pkl` | list of 1000 dicts | | | | `pts_3d (n, 3)`, `faces (m, 3)`, `x_pres (n, F)`, `y_pres (n, 1)`; about 2000 nodes each; AirfRANS |

The OSF archive contains the seven main sets (T1, T2, T3, T5, T6, T7, T8).  `T6_100K_wilson_loop.pkl` (the 100K-face
scaling study) and `T7_small_yang_mills.pkl` are produced by the v1 generators `gen_T6_100K_wilson_loop.py` and
`gen_T7_small_yang_mills.py` of the original repository.  The third coordinate of the planar meshes (T5-T8) is 0.  Some pickles (T3) contain pyvista objects next to the numpy
arrays; on a machine without pyvista, put the unpickling shim on the path: `PYTHONPATH=<repo>/shims`.

The native cochain versions of T6/T7 (edge connections, face fluxes) are regenerated from the v1 generators' physics
into `datasets/v2/T6_native.pt` / `T7_native.pt` (section 2); the T8 loader caches its complexes in
`datasets/v2/T8_complexes_<star>.pt` (rebuilt automatically, about 8 s, whenever `rhmp/complex.py` or
`rhmp/geometry.py` change).

## 2. v2 datasets (`datasets/v2/*.pt`)

Generated with the seeded scripts in `datasets/generators/` (outputs go to `datasets/v2/` by default, `--out-dir` /
`--out` to change it).  `bash scripts/gen_datasets.sh` runs the core sets; `SUITE=1` and `ANISO=1` add the extension
suite and the anisotropy suite:

```bash
bash scripts/gen_datasets.sh                                  # T6/T7 native (GPU), HP_*, TET_k100 (CPU, WORKERS=32)
SUITE=1 ANISO=1 bash scripts/gen_datasets.sh                  # + SURF, DYN, HP_qual_*, AHP/ASURF/ACURL/ADARCY
python3 -u datasets/generators/gen_aniso.py --selftest        # FEEC convergence checks of the anisotropy generator
python3 -u datasets/generators/gen_aniso.py --analyse 12      # oracle representability report (ANISO_representability.json)
```

| file(s) | generator | samples | size | tasks (`rhmp.tasks`) |
|---|---|---|---|---|
| `T6_native.pt`, `T7_native.pt` | `gen_T6_native.py`, `gen_T7_native.py` (reproduce the v1 samples with the CUDA RNG, so run on a GPU; need the v1 pickles) | 10000 each | 0.20 / 0.61 GB | `T6`, `T6f`, `T7`, `T7f` (native) |
| `HP_k{10,100,1000}.pt` (+ `_fine`) | `gen_HP.py --kappa K --n 5000 --n-fine 500` | 5000 (+ 500 at 4x resolution) | 0.62 GB (+ 0.25 GB) | `HP[flux\|fluxd\|fluxfem\|grad]_k<K>` |
| `HP_k10000.pt` | `gen_HP.py --kappa 10000 --n 1000 --n-fine 0` | 1000 | 0.12 GB | `HP_k10000` |
| `HP_k100_aniso.pt`, `HP_k100_aniso{10,100}.pt` (+ `_fine`) | `gen_HP.py --kappa 100 --aniso` / `--aniso-max R` | 5000 (+ 500) | 0.70-1.05 GB | `HP_k100_aniso[<R>]` |
| `TET_k100.pt` (+ `_fine`) | `gen_TET.py --kappa 100 --n 3000 --n-fine 200` | 3000 (+ 200) | 1.87 GB (+ 0.54 GB) | `TET[flux\|fluxfem\|grad]_k100` |
| `SURF.pt`, `SURF_geo.pt`, `SURF_topo.pt` | `gen_surf.py` | 5700 / 500 / 500 closed surfaces | 683 / 60 / 112 MB | `SURF`, `SURF_heat` (+ geometry / topology transfer) |
| `DYN.pt`, `DYNfix.pt` | `gen_dyn.py` | 600 / 500 trajectories x 101 states | 423 / 327 MB | `DYN*`, `DYNfix*` |
| `HP_qual_graded.pt`, `HP_qual_sliver.pt` | `gen_qual.py` (needs `HP_k100.pt`) | 100 test instances x 7 / 6 mesh levels | 92 / 79 MB | `HP_qual_*` |
| `AHP_r{10,100}.pt` (+ `_fine`) | `gen_aniso.py --tasks AHP` | 5000 (+ 500) | 0.74 GB (+ 0.30 GB) | `AHP_r<R>` |
| `ASURF_r{10,100}.pt` (+ `_fine`) | `gen_aniso.py --tasks ASURF` | 4000 (+ 400) | 1.34 GB (+ 0.53 GB) | `ASURF[_heat]_r<R>` |
| `ACURL_r{10,100}.pt` (+ `_fine`) | `gen_aniso.py --tasks ACURL` | 3000 (+ 200) | 3.16 GB (+ 0.91 GB) | `ACURL[b]_r<R>[_n1500]` |
| `ADARCY_r{10,100}.pt` (+ `_fine`) | `gen_aniso.py --tasks ADARCY` | 3000 (+ 200) | 3.61 GB (+ 1.07 GB) | `ADARCY[p]_r<R>[_n1500]` |

Equations, discretisations, splits and protocols: [docs/TASK_SUITE.md](../docs/TASK_SUITE.md),
[docs/TASK_SUITE_DETAILS.md](../docs/TASK_SUITE_DETAILS.md) (SURF, DYN, QUAL), [docs/ANISO_TASKS.md](../docs/ANISO_TASKS.md)
(AHP, ASURF, ACURL, ADARCY) and [docs/DATASETS.md](../docs/DATASETS.md) (file formats).  Variable-mesh sets are
packed as flat concatenations with offsets `ptr0..ptr3` (float32 fields, int32 cells); every target is an exact
sparse FEM / DEC / FEEC solve (residuals at round-off, recorded in the `.json` summaries).

Host memory: HP/TET-size sets are built on the CPU (`OMP_NUM_THREADS=1` speeds up the many small operations on a busy
machine).  Tensor-metric runs keep the Whitney blocks of every complex (a 2.5K-node tetrahedral complex is about
45 MB with them), so the 3000-sample tetrahedral sets need about 140 GB of host RAM; the registered `_n1500` subsets
need about 70 GB (see [REPORT.md](../REPORT.md), limitations).
