"""Example 06: inspect the learned metrics (``model.diagnostics``, ``model.metric_fields``) of a tensor-metric model.

``metric_type='tensor'`` (DESIGN §9.1) replaces the diagonal edge metric of the degree-0 up block by the
Whitney/Galerkin metric of a per-triangle SPD material tensor ``sigma_f = b_f expm(sum_j s_fj t_j t_j^T)``
(``t_j`` = unit vectors of the three edges of the triangle, signed bounded ``s``; the default
``tensor_param='full'`` covers every SPD tensor of bounded condition number).  At initialisation ``sigma_f = I`` and
``d_0^T H_1 d_0`` is exactly the P1 finite-element stiffness matrix.

Data: anisotropic Poisson ``-div(Sigma(x) grad u) = f`` (u = 0 on the boundary) on one shared random mesh, with a
smooth random SPD tensor field ``Sigma(x)`` per sample (random principal direction, eigenvalue ratio up to
``--ratio``, det 1).  The model sees ``f`` on vertices and ``log(t_e^T Sigma t_e)`` on edges (even): an E(n)-invariant
model can learn anisotropy only from inputs that carry the directions, and the projected conductivity along each edge
does.  After a short training run the script prints, per layer, the metric statistics of ``model.diagnostics`` and
compares the learned ``sigma_f`` with the true ``Sigma_f``; with matplotlib installed it saves an ellipse plot
(true tensors next to the learned ones).  Short runs learn little anisotropy; the full study is
``scripts/metric_recovery.py``.  The per-cell tensors are read from ``model.metric_fields(...)[l]['sigma']``; a
tensor is identifiable only through its action on the edges (``d_0^T H_1 d_0``), so compare actions, principal
directions and anisotropy ratios rather than raw coefficients.

Run (CPU, about 25-45 s):  python examples/06_metric_inspection.py [--steps 300] [--out DIR]
"""
from __future__ import annotations

import argparse
import math
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


def square_mesh(rng: np.random.Generator, m: int = 12):
    """Delaunay mesh of the unit square (``m`` boundary points per side, listed first; jittered interior grid)."""
    s = np.linspace(0.0, 1.0, m + 1)[:-1]
    boundary = np.concatenate([np.c_[s, 0 * s], np.c_[1 + 0 * s, s], np.c_[1 - s, 1 + 0 * s], np.c_[0 * s, 1 - s]])
    g = np.arange(1, m) / m
    ii, jj = np.meshgrid(g, g, indexing="ij")
    interior = np.c_[ii.ravel(), jj.ravel()] + rng.uniform(-0.35, 0.35, (g.size ** 2, 2)) / m
    pos = np.concatenate([boundary, interior])
    return pos, Delaunay(pos).simplices.astype(np.int64), len(boundary)


def smooth_field(rng: np.random.Generator, n_modes: int = 5, freq: float = 1.0):
    w = rng.normal(0.0, freq, (n_modes, 2))
    phase = rng.uniform(0.0, 2 * np.pi, n_modes)
    amp = rng.normal(0.0, 1.0, n_modes) / np.sqrt(n_modes / 2)
    return lambda x: (amp * np.cos(2 * np.pi * x @ w.T + phase)).sum(-1)


def tensor_field(rng: np.random.Generator, max_ratio: float):
    """``x (n, 2) -> Sigma (n, 2, 2)``: SPD, det 1, smooth principal direction, eigenvalue ratio in [1, max_ratio]."""
    ang, rat = smooth_field(rng), smooth_field(rng)

    def Sigma(x):
        th = math.pi * ang(x)
        r = np.exp(math.log(max_ratio) / (1.0 + np.exp(-2.0 * rat(x))))         # ratio in (1, max_ratio)
        c, s = np.cos(th), np.sin(th)
        R = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
        D = np.stack([np.sqrt(r), 1.0 / np.sqrt(r)], -1)
        return R @ (D[..., None] * np.swapaxes(R, -1, -2))
    return Sigma


def solve_aniso(pos, faces, Sigma_f, f, n_bnd):
    """P1 FEM for ``-div(Sigma grad u) = f``, ``u = 0`` on the first ``n_bnd`` vertices (lumped mass)."""
    P = pos[faces]
    J = np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]], 1)
    g12 = np.transpose(np.linalg.inv(J), (0, 2, 1))
    G = np.concatenate([-g12.sum(1, keepdims=True), g12], 1)                  # (n2, 3, 2) barycentric gradients
    area = 0.5 * np.abs(np.linalg.det(J))
    local = area[:, None, None] * (G @ Sigma_f @ G.transpose(0, 2, 1))
    n = len(pos)
    A = sp.csr_matrix((local.ravel(), (np.repeat(faces, 3, 1).ravel(), np.tile(faces, 3).ravel())), shape=(n, n))
    mass = np.zeros(n)
    np.add.at(mass, faces.ravel(), np.repeat(area / 3.0, 3))
    inner = np.arange(n_bnd, n)
    u = np.zeros(n)
    u[inner] = spla.spsolve(A[inner][:, inner].tocsc(), (mass * f)[inner])
    return u


def principal(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalue ratio and principal direction angle (radians, mod pi) of ``(n, 2, 2)`` SPD matrices."""
    ev, V = np.linalg.eigh(S)
    return ev[:, 1] / ev[:, 0], np.arctan2(V[:, 1, 1], V[:, 0, 1]) % np.pi


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=int, default=320)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ratio", type=float, default=10.0, help="maximal eigenvalue ratio of Sigma")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="directory for the ellipse plot (default: a temporary directory)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    out = Path(a.out) if a.out else Path(tempfile.mkdtemp(prefix="rhmp_ex06_"))
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data on one shared mesh
    t0 = time.time()
    pos, faces, n_bnd = square_mesh(rng)
    K = CochainComplex.from_triangles(pos, faces)
    edges = K.cells[1].numpy()
    tvec = pos[edges[:, 1]] - pos[edges[:, 0]]
    tvec /= np.linalg.norm(tvec, axis=1, keepdims=True)
    mid, cen = 0.5 * (pos[edges[:, 0]] + pos[edges[:, 1]]), pos[faces].mean(1)
    fs, ls, us, Sig = [], [], [], []
    for _ in range(a.samples):
        Sigma = tensor_field(rng, a.ratio)
        f = 1.0 + 0.5 * smooth_field(rng)(pos)
        Sig.append(Sigma(cen))
        us.append(solve_aniso(pos, faces, Sig[-1], f, n_bnd))
        fs.append(f)
        ls.append(np.log(np.einsum("ed,edf,ef->e", tvec, Sigma(mid), tvec)))       # log t^T Sigma t (even)
    f_all, l_all, u_all = (torch.tensor(np.stack(v), dtype=torch.float32) for v in (fs, ls, us))   # (N, n_k)
    n_tr = int(0.8 * a.samples)
    f_m, f_s, u_m, u_s = f_all[:n_tr].mean(), f_all[:n_tr].std(), u_all[:n_tr].mean(), u_all[:n_tr].std()
    X0 = ((f_all - f_m) / f_s).T[..., None].contiguous()                     # (n0, N, 1): layout (n_k, B, F)
    X1 = l_all.T[..., None].contiguous()                                       # (n1, N, 1), already O(1)
    Y = ((u_all - u_m) / u_s).T[..., None].contiguous()                       # (n0, N, 1)
    print(f"{a.samples} anisotropic Poisson samples on {K} in {time.time() - t0:.1f} s")

    # ------------------------------------------------------------------ model with the Whitney tensor metric
    cfg = RHMPConfig(in_dims={0: 1, 1: 1}, even_dims={1: 1}, metric_type="tensor", C=32, n_layers=3,
                     readout="node_scalar")
    model = RHMP(cfg, K.geo_dims)
    model.record_diagnostics = False
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=1e-4)
    gen = torch.Generator().manual_seed(a.seed)
    te = torch.arange(n_tr, a.samples)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        idx = torch.randperm(n_tr, generator=gen)[:a.batch]
        x = {0: X0[:, idx].contiguous(), 1: X1[:, idx].contiguous()}
        loss = F.mse_loss(model(x, K), Y[:, idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    x_te = {0: X0[:, te].contiguous(), 1: X1[:, te].contiguous()}
    model.eval()
    model.record_diagnostics = True
    with torch.no_grad():
        pred = model(x_te, K)                                                  # recorded forward pass
    r2 = r2_score(pred.transpose(0, 1), Y[:, te].transpose(0, 1))
    print(f"trained {a.steps} steps in {time.time() - t0:.1f} s; test R2 {r2:.4f}")

    # ------------------------------------------------------------------ 1. model.diagnostics, per layer
    d = model.diagnostics
    print("\nmodel.diagnostics (last forward pass):")
    for layer in range(cfg.n_layers):
        keys = sorted({k.split(".")[1] for k in d if k.startswith(f"layer{layer}.H") and k.endswith(".mean")})
        for name in keys:                                                      # diagonal metrics H{m}
            p = f"layer{layer}.{name}"
            print(f"  layer {layer} {name}: log(H/star) mean {d[p + '.mean']:+.3f} std {d[p + '.std']:.3f} "
                  f"range [{d[p + '.min']:+.3f}, {d[p + '.max']:+.3f}]  cond learned {d[p + '.cond_learned']:.3g} "
                  f"total {d[p + '.cond_total']:.3g}  saturated {d[p + '.sat']:.1%}")
        tens = {k.split(".tensor_", 1)[1]: v for k, v in d.items() if k.startswith(f"layer{layer}.H1.tensor_")}
        if tens:
            print(f"  layer {layer} tensor H1: " + ", ".join(f"{k}={v:.3g}" for k, v in sorted(tens.items())))
        betas = {k.split(".", 1)[1]: v for k, v in d.items() if k.startswith(f"layer{layer}.beta")}
        print(f"  layer {layer} Gershgorin normalisers: " + ", ".join(f"{k}={v:.3g}" for k, v in sorted(betas.items())))

    # ------------------------------------------------------------------ 2. per-cell fields: learned vs true tensors
    sample = 0                                                                 # first test sample
    fields = model.metric_fields({k: v[:, sample:sample + 1].contiguous() for k, v in x_te.items()}, K)
    true_ratio, true_dir = principal(Sig[n_tr + sample])
    print(f"\nmetric_fields: one dict per layer with keys {sorted(fields[0])}; tensor degrees {sorted(fields[0]['tensor'])}")
    learned = []
    for layer, fl in enumerate(fields):
        S = fl["sigma"][1][:, 0].double().numpy()                              # (n2, 2, 2) per-face SPD tensors
        ratio, direc = principal(S)
        sel = true_ratio > 2.0
        err = np.degrees(np.abs((direc - true_dir + np.pi / 2) % np.pi - np.pi / 2))[sel]
        corr = np.corrcoef(np.log(ratio), np.log(true_ratio))[0, 1] if ratio.std() > 0 else float("nan")
        print(f"  layer {layer}: learned anisotropy ratio mean {ratio.mean():.3f} (max {ratio.max():.3f}); "
              f"principal-direction error on faces with true ratio > 2: {err.mean():.1f} deg "
              f"(random guess: 45 deg); corr(log ratio) {corr:+.2f}")
        learned.append(S)

    # ------------------------------------------------------------------ 3. ellipse plot (optional)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import EllipseCollection
    except ImportError:
        print("\nmatplotlib is not installed (pip install 'rhmp[viz]'): skipping the ellipse plot")
    else:
        size = 0.55 * np.sqrt(np.abs(np.linalg.det(np.stack([pos[faces[:, 1]] - pos[faces[:, 0]],
                                                            pos[faces[:, 2]] - pos[faces[:, 0]]], 1))))
        panels = [("true Sigma_f", Sig[n_tr + sample]), ("learned sigma_f, layer 0", learned[0]),
                  (f"learned sigma_f, layer {len(learned) - 1}", learned[-1])]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
        for ax, (title, S) in zip(axes, panels):
            ev, V = np.linalg.eigh(S)
            ev = ev / np.sqrt(ev.prod(1, keepdims=True))                       # unit determinant: shape only
            ang = np.degrees(np.arctan2(V[:, 1, 1], V[:, 0, 1]))
            ec = EllipseCollection(size * np.sqrt(ev[:, 1]), size * np.sqrt(ev[:, 0]), ang, units="xy",
                                   offsets=cen, offset_transform=ax.transData, cmap="viridis")
            ec.set_array(np.log(ev[:, 1] / ev[:, 0]))
            ax.add_collection(ec)
            ax.triplot(pos[:, 0], pos[:, 1], faces, lw=0.2, color="0.7")
            ax.set(title=title, xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), aspect="equal")
            fig.colorbar(ec, ax=ax, shrink=0.8, label="log eigenvalue ratio")
        fig.tight_layout()
        path = out / "tensor_metric_ellipses.png"
        fig.savefig(path, dpi=110)
        print(f"\nsaved the ellipse plot to {path}")

    assert math.isfinite(r2) and all(math.isfinite(v) for v in d.values())
    print(f"EXAMPLE 06 OK  test_R2={r2:.4f}")


if __name__ == "__main__":
    main()
