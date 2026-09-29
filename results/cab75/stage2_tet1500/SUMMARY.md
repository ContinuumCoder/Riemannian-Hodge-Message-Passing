Test R2: (c) centred (v1 definition), (u) uncentred 1 - SS_res / sum t^2 for orientation-odd cochain targets.
Metric recovery r: per layer, signed Pearson correlation of the learned log(H_1/star_1) (diag) or of the learned log det sigma_f (tensor) with the true log conductivity; sat: per-layer fraction of edges whose bounded correction is clamped (|phi| > 0.95 log_range); sym: max |dR2| over relabel / flip-orient / rotate / reflect / gauge noise (exact symmetries -> round-off).

| run | log_range | test R2 | fine (4x) R2 | s/epoch | peak GB | params M | metric recovery r (per layer) | sat (per layer) | sym |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|
| TET_k100_diag-poly_s42 | 3.0 | 0.8563 (c) | 0.7960 | 20.7 | 7.9 | 0.117 |  |  |  |
| TET_k100_diag-resolvent_s42 | 3.0 | 0.8868 (c) | 0.7685 | 47.9 | 8.4 | 0.117 |  |  |  |
| TET_k100_tensor-resolvent_s42 | 3.0 | 0.8894 (c) | 0.7416 | 38.9 | 30.3 | 0.123 |  |  |  |
