# Riemannian-Hodge Message Passing

Official experiment code and checkpoints for the Riemannian-Hodge Message Passing paper.

## Overview

This repository contains:
- **Our model**: Riemannian-Hodge Message Passing network (`src/gauge_hodge_mp/`)
- **12 baselines**: GCN, GAT, SchNet, EGNN, MPSN, SCCNN, GaugeEquivCNN, GEM-CNN, CW Net, Clifford-SMPN, FNO, DeepONet
- **7 benchmark tasks** spanning fluid dynamics, gauge theory, and CFD
- **78 pretrained checkpoints** with full training histories, plus **2 additional seeds** (seed=1, seed=2) for variance estimation
- **Evaluation pipeline** computing R², SSIM, Pearson correlation, and NRMSE
- **Multi-seed aggregation** producing mean ± std across 3 training seeds

All results are reported in [RESULTS.md](RESULTS.md).

## Requirements

- Python >= 3.10
- PyTorch >= 2.0 (with CUDA)
- NumPy, SciPy

```bash
pip install torch numpy scipy
```

**Data generation only** (not needed for evaluation):
- `pyvista` (T3 ellipsoid meshes)
- `airfrans` (T8 airfoil data from AirfRANS dataset)

T1 vorticity data is derived from [PDEBench](https://github.com/pdebench/PDEBench) (Takamoto et al., NeurIPS 2022).

## Directory Structure

```
src/
  gauge_hodge_mp/         Our model (Riemannian-Hodge MP)
    network.py              Three-stage architecture (lift -> Hodge MP -> reconstruct)
    lifting.py              E(n)-invariant cochain lifting
    hodge_mp.py             Hodge-decomposed message passing with learnable Riemannian metric
    reconstruction.py       Equivariant signal reconstruction
    cell_complex.py         CW complex with Hodge Laplacians
    utils.py                Scatter, sparse ops, geometric invariants
  baselines_graph.py      GCN, GAT, SchNet, EGNN
  baselines_topo.py       MPSN, SCCNN
  baselines_advanced.py   GaugeEquivCNN, GEM-CNN, CW Net, Clifford-SMPN
  baselines_operator.py   FNO, DeepONet
  pdebench/               PDEBench data loading utilities (for T1)

experiments/
  formal_benchmark.py     Unified training script (all tasks, all models; --multi-seed-dir for seed_<N>/ layout)
  compute_all_metrics.py  Evaluation: R², SSIM, Pearson, NRMSE (--ckpt-root supports multi-seed trees)
  bench_T6_100K.py        100K-grid scaling experiment on T6
  bench_metric_ablation.py  Metric-family ablation (diagonal vs low-rank vs full SPD)
  bootstrap_metrics.py    Test-set bootstrap CIs
  gen_appendix_bootstrap.py LaTeX tables from bootstrap CIs

scripts/                  Multi-seed training pipeline
  build_jobs.py             Build jobs.json from (tasks × models × seeds), LPT-sorted by cost
  scheduler.py              Concurrent N-GPU scheduler with crash/resume
  monitor.py                Live status dashboard (--watch)
  launch.sh / keepalive.sh  Background launcher + watchdog
  aggregate_seeds.py        Collapse per-seed metrics into mean ± std
  gen_appendix_seed.py      LaTeX tables from seed_metrics.json

ablation/                 Symmetry / conservation diagnostic ablations (run_all.sh, ABLATION.md)

download.py               One-command download from OSF (data + ckpts + multi-seed ckpts)
all_metrics.json          Pre-computed R², SSIM, Pearson, NRMSE (all models, seed=42)
datasets/                 .pkl data files + generation scripts (via download.py)
checkpoints/              seed=42 baseline:  {task}/{model}/ -> best_model.pt, history.json
  seed_1/, seed_2/          additional seeds: seed_<N>/{task}/{model}/...
results/                  Per-task summary JSON
status/                   Multi-seed scheduler state + aggregated seed_metrics.json
logs/                     Training logs
```

## Quick Start

### 1. Download data and checkpoints

Datasets (1.1 GB), seed=42 checkpoints (834 MB), and multi-seed checkpoints (315 MB, seed=1 + seed=2) are hosted on [OSF](https://osf.io/35gcm/?view_only=0df302ed83154a8ba6bb1f8d8a1983ec).
The script automatically creates `datasets/`, `checkpoints/`, and `checkpoints/seed_<N>/` directories.

```bash
python3 download.py              # download everything (~2.3 GB)
python3 download.py --data       # datasets only
python3 download.py --ckpt       # seed=42 checkpoints only
python3 download.py --seed-ckpt  # seed=1, seed=2 multi-seed checkpoints only
```

Pre-computed metrics are included in `all_metrics.json` (no download needed to view results).

### 2. Evaluate from checkpoints

```bash
# Single-seed (seed=42 baseline)
python3 experiments/compute_all_metrics.py --n-eval 100

# A specific multi-seed run
python3 experiments/compute_all_metrics.py --ckpt-root checkpoints/seed_1 --n-eval 100

# Aggregate across all 3 seeds (42, 1, 2) -> status/seed_metrics.json with mean ± std
python3 scripts/aggregate_seeds.py --seeds 42 1 2
```

### Train a single model

```bash
# Train our model on T1 (CNS Vorticity)
CUDA_VISIBLE_DEVICES=0 python3 experiments/formal_benchmark.py T1 --model ours

# Train a baseline
CUDA_VISIBLE_DEVICES=0 python3 experiments/formal_benchmark.py T6 --model cw_net

# Smoke test (2 epochs)
python3 experiments/formal_benchmark.py T1 --model ours --smoke

# Multi-seed run: writes to checkpoints/seed_<N>/{task}/{model}/
CUDA_VISIBLE_DEVICES=0 python3 experiments/formal_benchmark.py T1 --model ours --seed 1 --multi-seed-dir
```

Available tasks: `T1`, `T2`, `T3`, `T5`, `T6`, `T7`, `T8`

Available models: `ours`, `gcn`, `gat`, `schnet`, `egnn`, `mpsn`, `sccnn`, `gauge_cnn`, `gem_cnn`, `cw_net`, `clifford`, `fno`, `deeponet`

### Reproduce the multi-seed sweep

```bash
python3 scripts/build_jobs.py --seeds 1 2          # writes status/jobs.json (LPT-sorted)
bash    scripts/launch.sh    --gpus 0,1 --per-gpu 2 # nohup scheduler in background
python3 scripts/monitor.py   --watch                # live dashboard
python3 scripts/aggregate_seeds.py --seeds 42 1 2  # mean ± std across seeds
```

### Load a pretrained model

```python
import torch, pickle, numpy as np
from src.gauge_hodge_mp.cell_complex import CellComplex
from src.gauge_hodge_mp.network import GaugeHodgeNetwork  # = Riemannian-Hodge MP

# Load data
d = pickle.load(open('datasets/T6_wilson_loop.pkl', 'rb'))
pts = torch.tensor(np.array(d['points'], dtype=np.float32))
faces = torch.tensor(np.array(d['faces'], dtype=np.int64))
K = CellComplex.from_triangulation(pts, faces).to('cuda')

# Load model (seed=42 baseline; for other seeds use checkpoints/seed_<N>/...)
state = torch.load('checkpoints/T6_wilson_loop/ours/best_model.pt', map_location='cuda')
model = GaugeHodgeNetwork(
    f_in=3, C=128, n_layers=4,
    n0=K.n0, n1=K.n1, n2=K.n2,
    task='scalar', spatial_dim=2, out_dim=1,
    mp_hidden=16, metric_type='diagonal', metric_rank=8
).to('cuda')
model.load_state_dict(state)
model.eval()

# Inference
X = torch.tensor(np.array(d['X_data'][:1], dtype=np.float32)).to('cuda')
with torch.no_grad():
    pred = model.forward_batch(X, K)
```

## Training Protocol

- 100 epochs, Adam optimizer
- Default seed 42 (baseline); additional seeds 1, 2 ship in `checkpoints/seed_<N>/`
- Cosine annealing: lr 1e-3 -> 1e-5
- Parameter budget: each baseline matched to ours within 20%
- Checkpoints saved at epochs 1, 25, 50, 75, 100 + best model
- Automatic resume from latest checkpoint

## Tasks

| ID | Name | Physics | Mesh | Nodes | Samples |
|---|---|---|---|---|---|
| T1 | CNS Vorticity | Compressible Navier-Stokes (PDEBench) | Regular grid 32x32 | 1024 | 10000 |
| T2 | Torus Advection-Diffusion | Scalar transport on genus-1 surface | Irregular torus | 1711 | 3000 |
| T3 | Ellipsoid Surface Flow | Tangential velocity from stream function | Irregular ellipsoid | 1157 | 10000 |
| T5 | Maxwell Poisson | Electrostatic field from charge density | Irregular Delaunay | 1024 | 5000 |
| T6 | Wilson Loop | U(1) gauge curvature from connection | Irregular Delaunay | 1024 | 10000 |
| T7 | Yang-Mills SU(2) | Non-abelian field strength F=dA+[A,A] | Irregular Delaunay | 1024 | 10000 |
| T8 | Airfoil Pressure | CFD pressure prediction (AirfRANS) | Per-sample mesh | ~2000 | 1000 |

