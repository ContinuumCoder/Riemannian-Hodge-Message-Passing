# Ops benchmark

Written by `bench/ops_bench.py` on an NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPU (torch 2.10.0+cu128, python 3.12.3, 2026-09-27). Times are CUDA-event medians in ms; `fwd` = forward only (no grad), `f+b` = forward + backward w.r.t. the dense input. Meshes: random 2-D Delaunay of the unit square. Layout `(n, B, C)` with `B = 8`; the sparse product sees the zero-copy `(n, B*C)` view (`v1` gets the same data as `(B, n, C)`). `err` = max abs deviation (output and gradient) from the `csr` reference. Each entry is the median over 3 interleaved rounds of 10 timings; small problems are timed over back-to-back calls (per-call time incl. launch overhead).

Methods: `csr` = `ops.spmm(A, x, AT)` (CSR, custom autograd with the precomputed transpose; the default path of `K.apply_d/apply_dT`); `csr_auto` = CSR with torch's built-in backward (no `AT`); `csr_i32` = CSR with int32 indices; `coo` = coalesced COO + precomputed COO transpose; `gather` = `index_select` for `d` (backward via `index_add`), `index_add` for `d^T` (backward via `index_select`); `v1` = v1's `batch_spmm` (coalesced COO, `d.t()` per call, `(B,n,C)` permute copies in and out, torch autograd).

## Operator application

| op | n0 | B*C | pass | csr | csr_auto | csr_i32 | coo | gather | v1 | best | best vs csr | v1 / csr | max err |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| d0 | 1000 | 128 | fwd | 0.014 | 0.014 | 0.014 | 0.046 | 0.033 | 0.039 | csr_auto | 1.03x | 2.7x | 1.4e-06 |
| d0 | 1000 | 128 | f+b | 0.084 | 0.235 | 0.084 | 0.156 | 0.112 | 0.297 | csr | 1.00x | 3.5x | 1.4e-06 |
| d0T | 1000 | 128 | fwd | 0.015 | 0.015 | 0.014 | 0.046 | 0.038 | 0.169 | csr_i32 | 1.07x | 11.2x | 1.4e-06 |
| d0T | 1000 | 128 | f+b | 0.096 | 0.249 | 0.112 | 0.168 | 0.155 | 0.456 | csr | 1.00x | 4.7x | 1.4e-06 |
| d1 | 1000 | 128 | fwd | 0.015 | 0.014 | 0.014 | 0.047 | 0.056 | 0.042 | csr_auto | 1.07x | 2.8x | 4.8e-07 |
| d1 | 1000 | 128 | f+b | 0.116 | 0.272 | 0.116 | 0.187 | 0.187 | 0.330 | csr | 1.00x | 2.8x | 4.8e-07 |
| d1T | 1000 | 128 | fwd | 0.017 | 0.015 | 0.015 | 0.056 | 0.064 | 0.185 | csr_auto | 1.13x | 10.8x | 9.5e-07 |
| d1T | 1000 | 128 | f+b | 0.117 | 0.301 | 0.122 | 0.190 | 0.200 | 0.532 | csr | 1.00x | 4.6x | 9.5e-07 |
| d0 | 1000 | 8192 | fwd | 0.099 | 0.100 | 0.098 | 0.667 | 0.602 | 0.751 | csr_i32 | 1.01x | 7.6x | 1.9e-06 |
| d0 | 1000 | 8192 | f+b | 0.256 | 0.475 | 0.257 | 0.959 | 1.195 | 1.282 | csr | 1.00x | 5.0x | 1.9e-06 |
| d0T | 1000 | 8192 | fwd | 0.053 | 0.052 | 0.053 | 0.313 | 0.622 | 0.572 | csr_auto | 1.02x | 10.7x | 2.9e-06 |
| d0T | 1000 | 8192 | f+b | 0.258 | 0.484 | 0.267 | 0.957 | 1.226 | 1.541 | csr | 1.00x | 6.0x | 2.9e-06 |
| d1 | 1000 | 8192 | fwd | 0.104 | 0.104 | 0.106 | 0.450 | 0.554 | 0.597 | csr | 1.00x | 5.7x | 9.5e-07 |
| d1 | 1000 | 8192 | f+b | 0.309 | 0.530 | 0.312 | 1.151 | 1.617 | 1.633 | csr | 1.00x | 5.3x | 9.5e-07 |
| d1T | 1000 | 8192 | fwd | 0.142 | 0.143 | 0.140 | 0.694 | 0.988 | 0.962 | csr_i32 | 1.01x | 6.7x | 9.5e-07 |
| d1T | 1000 | 8192 | f+b | 0.306 | 0.495 | 0.304 | 1.144 | 1.530 | 1.699 | csr_i32 | 1.01x | 5.6x | 9.5e-07 |
| d0 | 10000 | 128 | fwd | 0.018 | 0.018 | 0.018 | 0.103 | 0.039 | 0.109 | csr_auto | 1.00x | 6.1x | 1.9e-06 |
| d0 | 10000 | 128 | f+b | 0.124 | 0.281 | 0.097 | 0.181 | 0.163 | 0.376 | csr_i32 | 1.27x | 3.0x | 1.9e-06 |
| d0T | 10000 | 128 | fwd | 0.016 | 0.016 | 0.016 | 0.060 | 0.059 | 0.198 | csr_i32 | 1.00x | 12.6x | 2.9e-06 |
| d0T | 10000 | 128 | f+b | 0.121 | 0.281 | 0.122 | 0.200 | 0.158 | 0.512 | csr | 1.00x | 4.2x | 2.9e-06 |
| d1 | 10000 | 128 | fwd | 0.017 | 0.017 | 0.017 | 0.084 | 0.058 | 0.091 | csr_i32 | 1.00x | 5.2x | 9.5e-07 |
| d1 | 10000 | 128 | f+b | 0.136 | 0.310 | 0.135 | 0.219 | 0.211 | 0.380 | csr_i32 | 1.01x | 2.8x | 9.5e-07 |
| d1T | 10000 | 128 | fwd | 0.018 | 0.018 | 0.018 | 0.105 | 0.075 | 0.242 | csr_auto | 1.01x | 13.5x | 9.5e-07 |
| d1T | 10000 | 128 | f+b | 0.099 | 0.309 | 0.098 | 0.216 | 0.186 | 0.540 | csr_i32 | 1.01x | 5.4x | 9.5e-07 |
| d0 | 10000 | 8192 | fwd | 1.824 | 1.822 | 1.828 | 8.671 | 6.792 | 9.740 | csr_auto | 1.00x | 5.3x | 3.8e-06 |
| d0 | 10000 | 8192 | f+b | 2.974 | 3.336 | 2.975 | 12.938 | 13.558 | 16.130 | csr | 1.00x | 5.4x | 3.8e-06 |
| d0T | 10000 | 8192 | fwd | 1.127 | 1.129 | 1.127 | 4.313 | 6.808 | 5.936 | csr_i32 | 1.00x | 5.3x | 2.9e-06 |
| d0T | 10000 | 8192 | f+b | 3.054 | 3.752 | 2.988 | 12.807 | 13.581 | 17.305 | csr_i32 | 1.02x | 5.7x | 2.9e-06 |
| d1 | 10000 | 8192 | fwd | 1.621 | 1.621 | 1.623 | 7.367 | 7.765 | 9.180 | csr_auto | 1.00x | 5.7x | 9.5e-07 |
| d1 | 10000 | 8192 | f+b | 3.582 | 4.355 | 3.591 | 16.661 | 19.551 | 21.371 | csr | 1.00x | 6.0x | 9.5e-07 |
| d1T | 10000 | 8192 | fwd | 1.956 | 1.958 | 1.959 | 9.167 | 10.697 | 10.732 | csr | 1.00x | 5.5x | 9.5e-07 |
| d1T | 10000 | 8192 | f+b | 3.584 | 4.156 | 3.581 | 16.601 | 18.492 | 21.176 | csr_i32 | 1.00x | 5.9x | 9.5e-07 |
| d0 | 100000 | 128 | fwd | 0.286 | 0.285 | 0.282 | 1.175 | 1.037 | 1.294 | csr_i32 | 1.01x | 4.5x | 1.9e-06 |
| d0 | 100000 | 128 | f+b | 0.520 | 0.742 | 0.518 | 1.656 | 2.007 | 2.258 | csr_i32 | 1.01x | 4.3x | 1.9e-06 |
| d0T | 100000 | 128 | fwd | 0.192 | 0.192 | 0.186 | 0.493 | 1.011 | 0.983 | csr_i32 | 1.03x | 5.1x | 2.9e-06 |
| d0T | 100000 | 128 | f+b | 0.516 | 0.824 | 0.512 | 1.647 | 2.049 | 2.727 | csr_i32 | 1.01x | 5.3x | 2.9e-06 |
| d1 | 100000 | 128 | fwd | 0.264 | 0.267 | 0.265 | 0.942 | 1.182 | 1.182 | csr | 1.00x | 4.5x | 9.5e-07 |
| d1 | 100000 | 128 | f+b | 0.631 | 0.940 | 0.627 | 2.163 | 2.976 | 3.079 | csr_i32 | 1.01x | 4.9x | 9.5e-07 |
| d1T | 100000 | 128 | fwd | 0.327 | 0.330 | 0.326 | 1.255 | 1.671 | 1.719 | csr_i32 | 1.00x | 5.2x | 9.5e-07 |
| d1T | 100000 | 128 | f+b | 0.628 | 0.893 | 0.631 | 2.189 | 2.832 | 3.276 | csr | 1.00x | 5.2x | 9.5e-07 |
| d0 | 100000 | 8192 | fwd | 19.050 | 19.075 | 19.014 | 86.620 | 68.791 | 97.007 | csr_i32 | 1.00x | 5.1x | 2.9e-06 |
| d0 | 100000 | 8192 | f+b | 33.742 | 36.090 | 33.579 | 135.169 | 144.777 | 165.855 | csr_i32 | 1.00x | 4.9x | 2.9e-06 |
| d0T | 100000 | 8192 | fwd | 16.010 | 16.032 | 15.738 | 52.966 | 80.139 | 71.233 | csr_i32 | 1.02x | 4.4x | 3.8e-06 |
| d0T | 100000 | 8192 | f+b | 36.953 | 43.722 | 36.750 | 147.849 | 155.528 | 193.228 | csr_i32 | 1.01x | 5.2x | 3.8e-06 |
| d1 | 100000 | 8192 | fwd | 18.458 | 18.454 | 18.426 | 79.389 | 85.405 | 98.562 | csr_i32 | 1.00x | 5.3x | 9.5e-07 |
| d1 | 100000 | 8192 | f+b | 41.281 | 47.962 | 41.169 | 181.342 | 217.263 | 232.109 | csr_i32 | 1.00x | 5.6x | 9.5e-07 |
| d1T | 100000 | 8192 | fwd | 20.809 | 20.794 | 20.772 | 92.903 | 119.789 | 118.783 | csr_i32 | 1.00x | 5.7x | 9.5e-07 |
| d1T | 100000 | 8192 | f+b | 37.702 | 42.109 | 37.629 | 181.334 | 205.230 | 228.131 | csr_i32 | 1.00x | 6.1x | 9.5e-07 |

## Summary

* Dispatch rule (an alternative beating `csr` by > 1.3x in some regime): **none**; `spmm` / `apply_d` keep the CSR path, no dispatch.
* Best non-CSR method (COO or gather/scatter) on problems with n0 >= 10K or B*C = 8192 is 1.3x-6.1x slower than `csr` (extra memory passes; no fusion in eager mode).
* `csr` vs torch's built-in CSR backward (`csr_auto`, f+b): 1.07x-3.12x faster; the gain is launch/setup overhead, largest on small problems. Always pass `AT` (or use `K.apply_d`).
* `csr` vs v1's `batch_spmm`: forward 2.7x-13.5x, forward+backward 2.8x-6.1x faster.
* int32 CSR indices bring no measurable gain; indices stay int64.
* Caveat: at n0 = 100K, B*C = 8192 the working set of the COO / gather / v1 paths (several 6-10 GB temporaries) approaches the 96 GB device memory, so a few of their entries there are erratic (allocator pressure, e.g. `v1` f+b below its fwd on d1T); the `csr` numbers are stable across runs.

## Complex construction

| mesh | n (cells per degree) | device | first call (s) | warm (s) | check_d2 (s) |
|---|---|---|---:|---:|---:|
| random Delaunay, ~100K faces | [50000, 149970, 99971] | cuda | 0.012 | 0.010 | 0.0059 |
| random Delaunay, ~100K faces | [50000, 149970, 99971] | cpu | 0.120 | 0.090 | 0.0023 |
| random Delaunay, ~1M faces | [500000, 1499961, 999962] | cuda | 0.039 | 0.023 | 0.0016 |
| random 3-D Delaunay tets | [20000, 153795, 267461, 133665] | cuda | 0.027 | 0.025 | 0.0072 |
| random 3-D Delaunay tets | [20000, 153795, 267461, 133665] | cpu | 0.274 | 0.253 | 0.0083 |
| T6_100K mesh (v1: 6.5 s + 20 GB dense check) | [50010, 149991, 99982] | cuda | 0.023 | 0.010 | 0.0013 |
| T6_100K mesh (v1: 6.5 s + 20 GB dense check) | [50010, 149991, 99982] | cpu | 0.093 | 0.090 | 0.0022 |

<!-- whitney:begin -->

## Whitney tensor metric (DESIGN §9.1)

`dec.apply_whitney_metric(K, k, b, a, x)` (gather, per-(cell, sample) m_k x m_k block product in a fixed order, scatter; bitwise per-sample) and `dec.whitney_rowsum_abs`; `up-block` = `d_{k-1}^T H_k d_{k-1} x` on (k-1)-cochains with the tensor metric vs the diagonal (lumped) metric; `resolvent` = `cg_solve((I + L_up) y = x, iters=16, early_exit=False)` with the tensor-metric up-block. CUDA-event medians (ms) over 3 rounds x 20 calls; f+b = forward + backward w.r.t. x, b, a.

| case | n_k | n_top | B | C | apply fwd | apply f+b | rowsum fwd | up-block tensor f+b | up-block diag f+b | resolvent (16 it) fwd |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T6 mesh | 3050 | 2027 | 64 | 128 | 1.289 | 3.593 | 0.122 | 3.540 | 0.725 | 26.864 |
| T6_100K mesh | 149991 | 99982 | 2 | 16 | 0.281 | 0.798 | 0.143 | 0.942 | 0.270 | 7.079 |
| random tets (k=1, 6x6) | 153795 | 133665 | 2 | 16 | 1.132 | 2.878 | 0.490 | 3.020 | 0.286 | 20.137 |
| random tets (k=2, 4x4) | 267461 | 133665 | 2 | 16 | 0.527 | 1.506 | 0.225 | 1.784 | 0.405 | 12.791 |

Measured on GPU 1 while no other compute process used it (the script waits until the GPU is free).

<!-- whitney:end -->
