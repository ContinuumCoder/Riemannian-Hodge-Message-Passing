Test R2: (c) centred (v1 definition), (u) uncentred 1 - SS_res / sum t^2 for orientation-odd cochain targets.
Metric recovery r: per layer, signed Pearson correlation of the learned log(H_1/star_1) (diag) or of the learned log det sigma_f (tensor) with the true log conductivity; sat: per-layer fraction of edges whose bounded correction is clamped (|phi| > 0.95 log_range); sym: max |dR2| over relabel / flip-orient / rotate / reflect / gauge noise (exact symmetries -> round-off).

| run | log_range | test R2 | fine (4x) R2 | s/epoch | peak GB | params M | metric recovery r (per layer) | sat (per layer) | sym |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| HP_k100_aniso100_tensor-poly_s42 | 3.0 | 0.7445 (c) | 0.6416 | 95.7 | 1.7 | 0.095 | - / - / - / - | - / - / - / - | 1.6e-07 |
| HP_k100_aniso100_tensor-resolvent_s42 | 3.0 | 0.8392 (c) | 0.5429 | 83.8 | 1.6 | 0.094 | - / - / - / - | - / - / 0.31 / - | 9.7e-06 |
| HP_k100_tensor-poly_s42 | 3.0 | 0.8220 (c) | 0.7101 | 69.2 | 2.3 | 0.095 | - / - / - / - | - / - / - / - | 2.6e-08 |
| HP_k100_tensor-resolvent_s42 | 3.0 | 0.9175 (c) | 0.6009 | 109.1 | 1.9 | 0.093 | - / - / - / - | - / - / 0.31 / - | 1.1e-05 |
