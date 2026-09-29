Test R2: (c) centred (v1 definition), (u) uncentred 1 - SS_res / sum t^2 for orientation-odd cochain targets.
Metric recovery r: per layer, signed Pearson correlation of the learned log(H_1/star_1) (diag) or of the learned log det sigma_f (tensor) with the true log conductivity; sat: per-layer fraction of edges whose bounded correction is clamped (|phi| > 0.95 log_range); sym: max |dR2| over relabel / flip-orient / rotate / reflect / gauge noise (exact symmetries -> round-off).

| run | log_range | test R2 | fine (4x) R2 | s/epoch | peak GB | params M | metric recovery r (per layer) | sat (per layer) | sym |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| HP_k100_aniso100_diag-poly_s42 | 3.0 | 0.7590 (c) | 0.6272 | 42.1 | 1.2 | 0.090 | - / - / - / - | 0.00 / 0.16 / 0.18 / 0.20 | 1.2e-08 |
| HP_k100_aniso100_diag-resolvent_s42 | 3.0 | 0.8492 (c) | 0.6423 | 75.6 | 1.4 | 0.090 | - / - / - / - | 0.00 / 0.19 / 0.31 / 0.01 | 1.0e-05 |
| HP_k100_aniso100_tensor-poly_s42 | 3.0 | 0.7394 (c) | 0.6411 | 59.9 | 2.9 | 0.095 | - / - / - / - | - / - / - / - | 2.0e-07 |
| HP_k100_aniso100_tensor-resolvent_s42 | 3.0 | 0.8381 (c) | 0.5528 | 107.2 | 1.6 | 0.094 | - / - / - / - | - / - / 0.30 / - | 4.9e-06 |
| HP_k100_diag-poly_s42 | 3.0 | 0.8022 (c) | 0.6924 | 22.6 | 16.2 | 0.090 | - / - / - / - | 0.00 / 0.00 / 0.08 / 0.04 | 1.7e-08 |
| HP_k100_diag-resolvent_s42 | 3.0 | 0.9082 (c) | 0.6812 | 64.2 | 1.2 | 0.090 | - / - / - / - | 0.04 / 0.09 / 0.34 / 0.01 | 1.1e-05 |
| HP_k100_tensor-poly_s42 | 3.0 | 0.8191 (c) | 0.6985 | 34.3 | 1.4 | 0.095 | - / - / - / - | - / - / - / - | 2.2e-08 |
| HP_k100_tensor-resolvent_s42 | 3.0 | 0.9142 (c) | 0.6540 | 69.3 | 1.8 | 0.093 | - / - / - / - | - / - / 0.38 / - | 4.9e-06 |
| T6f_diag-poly_s42 | 3.0 | 1.0000 (u) |  | 9.7 | 4.6 | 0.073 |  |  | 1.8e-09 |
| T6f_diag-resolvent_s42 | 3.0 | 1.0000 (u) |  | 33.9 | 4.7 | 0.073 |  |  | 7.7e-10 |
| T6f_tensor-poly_s42 | 3.0 | 1.0000 (u) |  | 13.5 | 5.3 | 0.077 |  |  | 1.4e-09 |
| T6f_tensor-resolvent_s42 | 3.0 | 1.0000 (u) |  | 36.3 | 5.2 | 0.076 |  |  | 1.2e-09 |
| TET_k100_diag-poly_s42 | 3.0 | 0.9412 (c) | 0.7134 | 73.7 | 7.7 | 0.117 | - / - / - / - | 0.23 / 0.01 / 0.34 / 0.00 | 1.0e-08 |
| TET_k100_diag-resolvent_s42 | 3.0 | 0.9829 (c) | 0.4818 | 152.0 | 9.6 | 0.117 | - / - / - / - | 0.23 / 0.01 / 0.07 / 0.92 | 7.3e-07 |
