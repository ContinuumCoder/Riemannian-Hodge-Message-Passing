# Gauge-Structured Hodge Message Passing: Experiment Results

## R² Results (100 epochs, seed=42)

| Model | T1 CNS Vort. | T2 Torus Adv. | T3 Ellip. Flow | T5 Maxwell | T6 Wilson | T7 Yang-Mills | T8 Airfoil |
|---|---|---|---|---|---|---|---|
| **ours** | **0.918** | **0.913** | **0.972** | **0.675** | **0.955** | **0.653** | **0.602** |
| gcn | -0.080 | 0.629 | -0.008 | -0.003 | 0.059 | 0.030 | 0.266 |
| gat | -0.082 | 0.616 | -0.009 | -0.006 | 0.703 | 0.075 | 0.048 |
| schnet | -0.082 | 0.630 | -0.009 | 0.133 | 0.474 | 0.193 | 0.380 |
| egnn | -0.080 | 0.616 | -0.008 | -0.008 | 0.752 | -0.302 | 0.347 |
| mpsn | -0.081 | 0.647 | -0.014 | -0.004 | 0.183 | 0.136 | NA |
| sccnn | -0.081 | 0.670 | -0.014 | 0.256 | 0.527 | 0.125 | NA |
| gauge_cnn | 0.846 | 0.762 | 0.959 | 0.508 | 0.783 | 0.393 | NA |
| gem_cnn | 0.724 | 0.776 | 0.825 | 0.346 | 0.517 | 0.178 | NA |
| cw_net | 0.895 | 0.861 | 0.405 | 0.574 | 0.875 | 0.226 | NA |
| clifford_smpn | 0.710 | 0.798 | -0.005 | 0.360 | 0.765 | 0.148 | NA |
| fno | 0.896 | NA | NA | NA | NA | NA | NA |
| deeponet | -0.074 | -0.012 | -0.007 | 0.035 | 0.158 | -0.009 | NA |

`NA` = incompatible architecture. EGNN receives only scalar-averaged node features on T6/T7 (no directional encoding).
All values computed from `compute_all_metrics.py` on 100 held-out test samples.

## SSIM (100 epochs, seed=42)

Per-sample structural similarity. Global data_range from target, predictions clipped to target range +/- 10%.

| Model | T1 CNS Vort. | T2 Torus Adv. | T3 Ellip. Flow | T5 Maxwell | T6 Wilson | T7 Yang-Mills | T8 Airfoil |
|---|---|---|---|---|---|---|---|
| **ours** | **0.984** | **0.946** | **0.988** | **0.886** | **0.993** | **0.852** | **0.809** |
| gcn | 0.726 | 0.708 | 0.210 | 0.607 | 0.805 | 0.439 | 0.543 |
| gat | 0.716 | 0.719 | 0.210 | 0.606 | 0.943 | 0.487 | 0.503 |
| schnet | 0.719 | 0.766 | 0.209 | 0.637 | 0.897 | 0.569 | 0.574 |
| egnn | 0.725 | 0.539 | 0.210 | 0.603 | 0.956 | 0.355 | 0.695 |
| mpsn | 0.721 | 0.777 | 0.152 | 0.385 | 0.789 | 0.523 | NA |
| sccnn | 0.721 | 0.777 | 0.152 | 0.539 | 0.896 | 0.511 | NA |
| gauge_cnn | 0.973 | 0.846 | **0.981** | 0.821 | 0.967 | 0.712 | NA |
| gem_cnn | 0.936 | 0.865 | 0.882 | 0.755 | 0.917 | 0.563 | NA |
| cw_net | 0.981 | 0.912 | 0.478 | 0.834 | 0.977 | 0.600 | NA |
| clifford_smpn | 0.941 | 0.870 | 0.170 | 0.594 | 0.950 | 0.540 | NA |
| fno | 0.976 | NA | NA | NA | NA | NA | NA |
| deeponet | 0.730 | 0.346 | 0.210 | 0.621 | 0.845 | 0.405 | NA |

## Spatial Pearson Correlation (100 epochs, seed=42)

$$\rho = \frac{\text{Cov}(y, \hat{y})}{\sigma_y \, \sigma_{\hat{y}}}$$

| Model | T1 CNS Vort. | T2 Torus Adv. | T3 Ellip. Flow | T5 Maxwell | T6 Wilson | T7 Yang-Mills | T8 Airfoil |
|---|---|---|---|---|---|---|---|
| **ours** | **0.970** | **0.949** | **0.988** | **0.822** | **0.950** | **0.810** | **0.796** |
| gcn | 0.081 | 0.774 | 0.024 | 0.056 | 0.171 | 0.184 | 0.428 |
| gat | 0.024 | 0.767 | 0.022 | 0.167 | 0.716 | 0.283 | 0.445 |
| schnet | 0.026 | 0.775 | 0.033 | 0.392 | 0.542 | 0.444 | 0.591 |
| egnn | 0.039 | 0.774 | 0.026 | 0.173 | 0.760 | 0.028 | 0.573 |
| mpsn | 0.009 | 0.784 | 0.054 | 0.048 | 0.272 | 0.341 | NA |
| sccnn | 0.016 | 0.794 | 0.048 | 0.473 | 0.587 | 0.325 | NA |
| gauge_cnn | 0.947 | 0.859 | **0.981** | 0.703 | 0.841 | 0.630 | NA |
| gem_cnn | 0.887 | 0.867 | 0.899 | 0.553 | 0.649 | 0.433 | NA |
| cw_net | 0.950 | 0.918 | 0.497 | 0.760 | 0.850 | 0.480 | NA |
| clifford_smpn | 0.822 | 0.883 | 0.114 | 0.598 | 0.766 | 0.389 | NA |
| fno | 0.966 | NA | NA | NA | NA | NA | NA |
| deeponet | 0.244 | 0.125 | 0.072 | 0.322 | 0.296 | 0.028 | NA |

## NRMSE = RMSE / range(y) (lower is better)

| Model | T1 | T2 | T3 | T5 | T6 | T7 | T8 |
|---|---|---|---|---|---|---|---|
| **ours** | **0.0095** | **0.0152** | **0.0095** | **0.0107** | **0.0032** | **0.0212** | **0.0280** |
| gcn | 0.0344 | 0.0314 | 0.0613 | 0.0189 | 0.0144 | 0.0354 | 0.0380 |
| gat | 0.0344 | 0.0319 | 0.0614 | 0.0189 | 0.0081 | 0.0346 | 0.0433 |
| schnet | 0.0344 | 0.0314 | 0.0614 | 0.0175 | 0.0108 | 0.0323 | 0.0350 |
| egnn | 0.0344 | 0.0320 | 0.0614 | 0.0189 | 0.0074 | 0.0411 | 0.0359 |
| mpsn | 0.0344 | 0.0306 | 0.0698 | 0.0287 | 0.0152 | 0.0330 | NA |
| sccnn | 0.0344 | 0.0295 | 0.0699 | 0.0247 | 0.0116 | 0.0332 | NA |
| gauge_cnn | 0.0130 | 0.0251 | 0.0120 | 0.0132 | 0.0069 | 0.0280 | NA |
| gem_cnn | 0.0174 | 0.0244 | 0.0272 | 0.0153 | 0.0103 | 0.0326 | NA |
| cw_net | 0.0107 | 0.0192 | 0.0534 | 0.0123 | 0.0053 | 0.0317 | NA |
| clifford_smpn | 0.0178 | 0.0231 | 0.0695 | 0.0229 | 0.0082 | 0.0328 | NA |
| fno | 0.0107 | NA | NA | NA | NA | NA | NA |
| deeponet | 0.0343 | 0.0519 | 0.0614 | 0.0185 | 0.0136 | 0.0362 | NA |

## Ours vs Best Baseline

| Task | Ours | Best Baseline | Gap |
|---|---|---|---|
| T1 CNS Vorticity | **0.918** | FNO 0.896 | +0.022 |
| T2 Torus AdvDiff | **0.913** | CW Net 0.861 | +0.052 |
| T3 Ellipsoid Flow | **0.972** | GaugeEquivCNN 0.959 | +0.013 |
| T5 Maxwell Poisson | **0.675** | CW Net 0.574 | +0.101 |
| T6 Wilson Loop | **0.955** | CW Net 0.875 | +0.080 |
| T7 Yang-Mills SU(2) | **0.653** | GaugeEquivCNN 0.393 | +0.260 |
| T8 Airfoil Pressure | **0.602** | SchNet 0.380 | +0.222 |

## Baselines (12 + ours = 13 models)

### Graph baselines
| # | Name | Reference | Key idea |
|---|---|---|---|
| 1 | GCN | Kipf & Welling. ICLR 2017. | Isotropic spectral convolution |
| 2 | GAT | Velickovic et al. ICLR 2018. | Attention-weighted message passing |
| 3 | SchNet | Schutt et al. NeurIPS 2017. | Continuous radial filters on distances |
| 4 | EGNN | Satorras, Hoogeboom, Welling. PMLR 139:9323-9332, 2021. | E(n)-equivariant MP (distances + relative positions) |

### Topological baselines
| # | Name | Reference | Key idea |
|---|---|---|---|
| 5 | MPSN | Bodnar et al. ICML 2021. | MP on simplicial adjacencies |
| 6 | SCCNN | Yang, Isufi, Leus. ICASSP 2022 / TMLR 2025. | Polynomial Hodge Laplacian filters |

### Geometric baselines
| # | Name | Reference | Key idea |
|---|---|---|---|
| 7 | GaugeEquivCNN | Cohen et al. ICML 2019. | Parallel transport along edges |
| 8 | GEM-CNN | de Haan et al. ICLR 2021. | Gauge equivariant anisotropic convolution |
| 9 | CW Net | Bodnar et al. NeurIPS 2021. | Cell complex MP via boundary/coboundary |
| 10 | Clifford-SMPN | Liu et al. ICLR 2024. | Clifford algebra on simplicial complexes |

### Operator baselines
| # | Name | Reference | Key idea |
|---|---|---|---|
| 11 | FNO | Li et al. ICLR 2021. | FFT-based spectral convolution (grid only) |
| 12 | DeepONet | Lu et al. Nature MI 2021. | Branch-trunk operator learning |

## Task Descriptions

| ID | Name | Physics | Input / Output | Mesh | Nodes | Samples |
|---|---|---|---|---|---|---|
| T1 | CNS Vorticity | Compressible NS (PDEBench) | (rho,Vx,Vy,p) / omega | Regular grid 32x32 | 1024 | 10000 |
| T2 | Torus Advection-Diffusion | Scalar transport | f(scalar) / u(scalar) | Irregular torus | 1711 | 3000 |
| T3 | Ellipsoid Surface Flow | Surface fluid | psi / v(tangent) | Irregular ellipsoid | 1157 | 10000 |
| T5 | Maxwell Poisson | Electrostatics | rho(charge) / E(field) | Irregular Delaunay | 1024 | 5000 |
| T6 | Wilson Loop | U(1) gauge curvature | theta(connection) / F(curvature) | Irregular Delaunay | 1024 | 10000 |
| T7 | Yang-Mills SU(2) | Non-abelian field strength | A(connection) / F=dA+[A,A] | Irregular Delaunay | 1024 | 10000 |
| T8 | Airfoil Pressure | CFD pressure (AirfRANS) | (sdf,V,alpha) / p | Per-sample mesh | ~2000 | 1000 |

## Protocol

- 100 epochs, seed=42, Adam with cosine annealing (lr 1e-3 to 1e-5)
- Parameter budget: each baseline matched to ours within 20%
- Metrics: R², MSE, MAE (training); R², SSIM, Pearson, NRMSE (evaluation)
- Checkpoints at epochs 1, 25, 50, 75, 100 + best model
- Automatic resume from latest checkpoint

## Directory Structure
```
datasets/           T1-T8 .pkl data + generation scripts
experiments/        formal_benchmark.py, compute_all_metrics.py
src/
  gauge_hodge_mp/   Our model (network, lifting, hodge_mp, reconstruction)
  baselines_*.py    All 12 baseline implementations
checkpoints/        {task}/{model}/ -> best_model.pt, history.json, result.json
logs/               Training logs
results/            Per-task summary JSON
```

## Reproducing Results

```bash
# Train a single model on a single task
CUDA_VISIBLE_DEVICES=0 python3 experiments/formal_benchmark.py T1 --model ours

# Compute all metrics from saved checkpoints
python3 experiments/compute_all_metrics.py --n-eval 100

# Results saved to checkpoints/all_metrics.json
```
