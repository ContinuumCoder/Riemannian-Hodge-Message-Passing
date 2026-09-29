"""Example 03: train RHMP on a tiny heterogeneous-conductivity Poisson problem, one random mesh per sample.

Data (generated in this script, about 2 s): every sample is a new random Delaunay mesh of the unit square (100-170
vertices), a smooth log-normal conductivity field ``sigma(x)`` with contrast ``--kappa`` and a smooth source ``f(x)``;
the target is the P1 finite-element solution of

    -div(sigma grad u) = f   in the square,      u = 0 on the boundary

assembled with numpy and solved with ``scipy.sparse.linalg.spsolve``.  The model sees ``f`` on the vertices (degree 0)
and ``log sigma`` at the edge midpoints as an *even* edge input (degree 1): a material coefficient is a property of
the edge, not an oriented quantity.  Minibatches of 8 meshes are block-diagonal batches (``CochainComplex.batch``).

Run (CPU, about 15-30 s):  python examples/03_train_poisson.py [--steps 200] [--resolvent] [--metric-reference]
Prints the test R2 on held-out meshes (typically about 0.8 after 200 steps with the defaults).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn.functional as F
from scipy.spatial import Delaunay

try:
    import rhmp  # noqa: F401
except ImportError:  # running from a source checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rhmp import RHMP, CochainComplex, RHMPConfig
from rhmp.metrics import r2_score


# ---------------------------------------------------------------------------------------------------------------
# data: random meshes, smooth random fields, P1 finite elements
# ---------------------------------------------------------------------------------------------------------------
def smooth_field(rng: np.random.Generator, n_modes: int = 6, freq: float = 1.5):
    """Random smooth function ``x (n, 2) -> (n,)`` (sum of random Fourier modes, unit variance on average)."""
    w = rng.normal(0.0, freq, (n_modes, 2))
    phase = rng.uniform(0.0, 2 * np.pi, n_modes)
    amp = rng.normal(0.0, 1.0, n_modes) / np.sqrt(n_modes / 2)
    return lambda x: (amp * np.cos(2 * np.pi * x @ w.T + phase)).sum(-1)


def square_mesh(rng: np.random.Generator, m: int):
    """Delaunay mesh of the unit square: ``m`` boundary points per side, jittered interior grid.

    Returns ``pos (n0, 2)``, ``faces (n2, 3)`` and the number of boundary vertices (listed first).
    """
    s = np.linspace(0.0, 1.0, m + 1)[:-1]
    boundary = np.concatenate([np.c_[s, 0 * s], np.c_[1 + 0 * s, s], np.c_[1 - s, 1 + 0 * s], np.c_[0 * s, 1 - s]])
    g = np.arange(1, m) / m
    ii, jj = np.meshgrid(g, g, indexing="ij")
    interior = np.c_[ii.ravel(), jj.ravel()] + rng.uniform(-0.35, 0.35, (g.size ** 2, 2)) / m
    pos = np.concatenate([boundary, interior])
    return pos, Delaunay(pos).simplices.astype(np.int64), len(boundary)


def solve_poisson(pos: np.ndarray, faces: np.ndarray, sigma_face: np.ndarray, f: np.ndarray, n_bnd: int) -> np.ndarray:
    """P1 FEM solution of ``-div(sigma grad u) = f``, ``u = 0`` on the first ``n_bnd`` vertices (lumped mass)."""
    P = pos[faces]                                                        # (n2, 3, 2)
    J = np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]], 1)               # rows: two edge vectors
    grads12 = np.transpose(np.linalg.inv(J), (0, 2, 1))                   # gradients of lambda_1, lambda_2
    G = np.concatenate([-grads12.sum(1, keepdims=True), grads12], 1)      # (n2, 3, 2)
    area = 0.5 * np.abs(np.linalg.det(J))
    local = (sigma_face * area)[:, None, None] * (G @ G.transpose(0, 2, 1))
    n = len(pos)
    A = sp.csr_matrix((local.ravel(), (np.repeat(faces, 3, 1).ravel(), np.tile(faces, 3).ravel())), shape=(n, n))
    mass = np.zeros(n)
    np.add.at(mass, faces.ravel(), np.repeat(area / 3.0, 3))
    inner = np.arange(n_bnd, n)
    u = np.zeros(n)
    u[inner] = spla.spsolve(A[inner][:, inner].tocsc(), (mass * f)[inner])
    return u


def make_sample(rng: np.random.Generator, kappa: float):
    """One sample: ``(K, {0: f (n0, 1), 1: log sigma_e (n1, 1)}, u (n0, 1))``."""
    pos, faces, n_bnd = square_mesh(rng, int(rng.integers(9, 13)))
    log_sigma_field, source_field = smooth_field(rng), smooth_field(rng, freq=1.0)

    def log_sigma(x):                                                     # sigma in [kappa^-1/2, kappa^1/2]
        return 0.5 * math.log(kappa) * np.tanh(log_sigma_field(x))

    f = 1.0 + 0.5 * source_field(pos)
    u = solve_poisson(pos, faces, np.exp(log_sigma(pos[faces].mean(1))), f, n_bnd)
    K = CochainComplex.from_triangles(pos, faces)
    edges = K.cells[1].numpy()                                            # the complex's edge order (src < dst)
    mid = 0.5 * (pos[edges[:, 0]] + pos[edges[:, 1]])
    t = lambda a: torch.tensor(a, dtype=torch.float32)[:, None]           # noqa: E731
    return K, {0: t(f), 1: t(log_sigma(mid))}, t(u)


# ---------------------------------------------------------------------------------------------------------------
# batching and training
# ---------------------------------------------------------------------------------------------------------------
class Normalizer:
    """Mean/std of the node source and of the target from the training meshes (log sigma is already O(1))."""

    def __init__(self, train):
        f = torch.cat([x[0] for _, x, _ in train])
        u = torch.cat([y for _, _, y in train])
        self.f = (f.mean(), f.std())
        self.u = (u.mean(), u.std())


def make_batch(samples, norm: Normalizer):
    """Block-diagonal batch: one complex with all cells concatenated, inputs ``(sum n_k, 1, F_k)`` (B = 1)."""
    K = CochainComplex.batch([K for K, _, _ in samples])
    f = torch.cat([x[0] for _, x, _ in samples])
    inputs = {0: ((f - norm.f[0]) / norm.f[1])[:, None], 1: torch.cat([x[1] for _, x, _ in samples])[:, None]}
    target = ((torch.cat([y for _, _, y in samples]) - norm.u[0]) / norm.u[1])[:, None]
    return K, inputs, target


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=80)
    ap.add_argument("--meshes-per-batch", type=int, default=8)
    ap.add_argument("--kappa", type=float, default=10.0, help="conductivity contrast max(sigma) / min(sigma)")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--resolvent", action="store_true",
                    help="use layers poly, resolvent, poly, poly (implicit CG layer; experimental)")
    ap.add_argument("--metric-reference", action="store_true",
                    help="use log sigma_e as a fixed offset of the edge metric (H_1 = star_1 sigma_e exp(a tanh(.)))")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="directory for the checkpoint (default: a temporary directory)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    data = [make_sample(rng, args.kappa) for _ in range(args.n_train + args.n_test)]
    train, test = data[:args.n_train], data[args.n_train:]
    sizes = [K.n[0] for K, _, _ in data]
    print(f"{len(data)} samples generated in {time.time() - t0:.1f} s; vertices per mesh {min(sizes)}..{max(sizes)}")

    norm = Normalizer(train)
    K_test, x_test, y_test = make_batch(test, norm)
    K_test, x_test, y_test = K_test.to(device), {k: v.to(device) for k, v in x_test.items()}, y_test.to(device)

    cfg = RHMPConfig(
        in_dims={0: 1, 1: 1},                   # f on vertices; log sigma on edges
        even_dims={1: 1},                       # the edge column is even (orientation-free material coefficient)
        metric_reference={1: 0} if args.metric_reference else {},
        C=32, n_layers=4, readout="node_scalar", out_dim=1,
        layers=["poly", "resolvent", "poly", "poly"] if args.resolvent else None,
    )
    model = RHMP(cfg, K_test.geo_dims).to(device)
    model.record_diagnostics = False            # skip the statistics bookkeeping while training
    print(f"model: {model.num_parameters()} parameters, layers {cfg.layer_types}, readout {cfg.readout}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=1e-4)
    gen = torch.Generator().manual_seed(args.seed)

    def test_r2() -> float:
        model.eval()
        with torch.no_grad():
            pred = model(x_test, K_test)        # (sum n0, 1, 1): all test meshes in one block-diagonal batch
        model.train()
        return r2_score(pred[:, 0].cpu(), y_test[:, 0].cpu())

    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randperm(len(train), generator=gen)[:args.meshes_per_batch].tolist()
        K, x, y = make_batch([train[i] for i in idx], norm)
        K, x, y = K.to(device), {k: v.to(device) for k, v in x.items()}, y.to(device)
        loss = F.mse_loss(model(x, K), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0 or step == args.steps:
            print(f"  step {step:4d}  train loss {loss.item():.4f}  test R2 {test_r2():.4f}  ({time.time() - t0:.1f} s)")

    r2 = test_r2()
    model.record_diagnostics = True             # one recorded forward pass for the metric statistics
    with torch.no_grad():
        model(x_test, K_test)
    d = model.diagnostics
    print(f"learned edge metric of layer 0: log(H1/star1) in [{d['layer0.H1.min']:+.3f}, {d['layer0.H1.max']:+.3f}], "
          f"condition number of H1: learned part {d['layer0.H1.cond_learned']:.2f}, "
          f"total (with the DEC star) {d['layer0.H1.cond_total']:.1f}")

    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="rhmp_ex03_"))
    out.mkdir(parents=True, exist_ok=True)
    path = out / "poisson_model.pt"
    torch.save(model.to_checkpoint(), path)     # {'cfg', 'geo_dims', 'state_dict'}
    reloaded = RHMP.from_checkpoint(str(path), map_location=device)
    with torch.no_grad():
        same = torch.equal(reloaded(x_test, K_test), model(x_test, K_test))
    print(f"checkpoint {path} ({os.path.getsize(path) / 1e3:.0f} kB) reloads to identical predictions: {same}")
    assert same and math.isfinite(r2)
    print(f"EXAMPLE 03 OK  test_R2={r2:.4f}")


if __name__ == "__main__":
    main()
