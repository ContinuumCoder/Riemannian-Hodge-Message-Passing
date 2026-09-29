Test R2: (c) centred, (u) uncentred 1 - SS_res / sum t^2 (orientation-odd cochain targets).  structure: mean physics residuals of the predictions on the test split (rhmp.tasks.aniso.structure_metrics).  recovery (last layer): relative action error of the learned physics operator vs the true one (km = metric degree), and the median principal-direction error of tensor metrics (scripts/metric_recovery.py).

| run | test R2 | fine (4x) R2 | s/epoch | peak GB | params M | structure (test) | recovery (last layer) |
|---|---:|---:|---:|---:|---:|---|---|
| ACURLb_r10_n1500_diag+ref_s42 | 0.5477 (u) | 0.3697 | 112.8 | 7.8 | 0.100 | div 1.16e-07, bc 9.77e-02 |  |
| ACURLb_r10_n1500_tensor+ref_s42 | 0.5847 (u) | 0.3934 | 202.8 | 31.6 | 0.109 | div 1.13e-07, bc 9.99e-02 |  |
| ADARCYp_r100_n1500_diag-solver_s42 | 0.9192 (c) | 0.8661 | 24.8 | 7.9 | 0.002 | pde_res 4.04e+00, bc 7.72e-03 |  |
| ADARCYp_r100_n1500_tensor-solver_s42 | 0.9768 (c) | 0.9700 | 63.4 | 23.0 | 0.004 | pde_res 3.24e+00, bc 7.47e-03 |  |
| AHP_r100_diag+ref_s42 | 0.5933 (c) | 0.4487 | 47.3 | 1.2 | 0.090 | pde_res 1.62e+02, bc 2.00e-02 |  |
| AHP_r100_mgn_s42 | 0.6649 (c) | 0.3998 | 29.0 | 1.0 | 0.092 | pde_res 1.01e+02, bc 1.66e-02 |  |
| AHP_r100_tensor+ref_s42 | 0.5304 (c) | 0.4320 | 65.2 | 2.2 | 0.095 | pde_res 2.60e+02, bc 6.73e-03 |  |
| ASURF_r100_dec_fixed_s42 | 0.9182 (c) | 0.8073 | 52.1 | 2.0 | 0.085 | pde_res 1.92e+01, integral 6.06e-02 |  |
| ASURF_r100_diag+ref_s42 | 0.8967 (c) | 0.7833 | 79.4 | 2.0 | 0.090 | pde_res 2.92e+01, integral 7.64e-02 |  |
| ASURF_r100_tensor+ref_s42 | 0.9061 (c) | 0.7927 | 139.0 | 3.0 | 0.095 | pde_res 1.90e+01, integral 6.73e-02 |  |
