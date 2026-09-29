Test R2: (c) centred, (u) uncentred 1 - SS_res / sum t^2 (orientation-odd cochain targets).  structure: mean physics residuals of the predictions on the test split (rhmp.tasks.aniso.structure_metrics).  recovery (last layer): relative action error of the learned physics operator vs the true one (km = metric degree), and the median principal-direction error of tensor metrics (scripts/metric_recovery.py).

| run | test R2 | fine (4x) R2 | s/epoch | peak GB | params M | structure (test) | recovery (last layer) |
|---|---:|---:|---:|---:|---:|---|---|
| AHP_r100_diag-solver_s42 | 0.8318 (c) | 0.6989 | 31.7 | 0.7 | 0.001 | pde_res 1.32e+01, bc 8.08e-03 |  |
| AHP_r100_tensor-solver_s42 | 0.9190 (c) | 0.7412 | 43.9 | 2.7 | 0.002 | pde_res 1.60e+01, bc 1.22e-02 |  |
| AHP_r10_diag-solver_s42 | 0.9645 (c) | 0.9139 | 36.8 | 1.0 | 0.001 | pde_res 5.28e+00, bc 1.18e-03 |  |
| AHP_r10_tensor-solver_s42 | 0.9922 (c) | 0.9275 | 53.8 | 1.8 | 0.002 | pde_res 6.91e+00, bc 1.26e-03 |  |
