"""Example 02: the discrete exterior calculus toolkit (``rhmp.dec``), no learning involved.

Covers metric Hodge Laplacians, harmonic forms (Betti numbers), the Hodge decomposition of a 1-cochain, batched
conjugate gradients (``cg_solve``), a material-weighted Laplacian, Whitney/Galerkin tensor metrics (the P1 FEM
stiffness identity and SPD checks), and the Whitney maps ``flat`` (vector field -> 1-cochain) and ``sharp``
(1-cochain -> vector field).  Everything runs in float64 so that the identities are visible to ~1e-10.

Run (CPU, a few seconds):  python examples/02_dec_toolkit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import Delaunay

try:
    import rhmp  # noqa: F401
except ImportError:  # running from a source checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rhmp import CochainComplex, dec

F64 = torch.float64


def torus_mesh(n_major: int = 16, n_minor: int = 8, R: float = 1.0, r: float = 0.4):
    """Closed triangulated torus in 3-D: ``pos (n0, 3)``, ``faces (n2, 3)``."""
    u = 2 * np.pi * np.arange(n_major) / n_major
    w = 2 * np.pi * np.arange(n_minor) / n_minor
    uu, ww = np.meshgrid(u, w, indexing="ij")
    pos = np.stack([(R + r * np.cos(ww)) * np.cos(uu), (R + r * np.cos(ww)) * np.sin(uu), r * np.sin(ww)], -1)
    i, j = np.meshgrid(np.arange(n_major), np.arange(n_minor), indexing="ij")
    i, j = i.ravel(), j.ravel()
    ip, jp = (i + 1) % n_major, (j + 1) % n_minor
    v00, v10, v11, v01 = i * n_minor + j, ip * n_minor + j, ip * n_minor + jp, i * n_minor + jp
    return pos.reshape(-1, 3), np.concatenate([np.stack([v00, v10, v11], 1), np.stack([v00, v11, v01], 1)])


def dense(A: torch.Tensor) -> torch.Tensor:
    """CSR -> dense float64 (small meshes only)."""
    return A.to_dense().to(F64)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


def main() -> None:
    g = torch.Generator().manual_seed(0)

    # ------------------------------------------------------------------ 1. Hodge Laplacians on a torus
    K = CochainComplex.from_triangles(*torus_mesh()).to(dtype=F64)
    print(f"torus: {K}")
    for k in range(K.dim + 1):
        L = dec.hodge_laplacian(K, k)          # weak form d^T H d + H d H^-1 d^T H with the reference stars
        x = torch.randn(K.n[k], 1, generator=g, dtype=F64)
        y = torch.randn(K.n[k], 1, generator=g, dtype=F64)
        asym = float((x * L(y)).sum() - (L(x) * y).sum())
        print(f"  L_{k}: <x, L y> - <L x, y> = {asym:+.1e},  <x, L x> = {float((x * L(x)).sum()):.3f} >= 0")

    # ------------------------------------------------------------------ 2. harmonic forms = Betti numbers
    betti = [dec.harmonic_basis(K, k).shape[1] for k in range(K.dim + 1)]
    print(f"  dimensions of the harmonic spaces (Betti numbers b0, b1, b2) = {betti}   (torus: [1, 2, 1])")

    # ------------------------------------------------------------------ 3. Hodge decomposition of a 1-cochain
    x = torch.randn(K.n[1], 2, generator=g, dtype=F64)                 # (n1, B): two random edge cochains
    exact, coexact, harm = dec.hodge_decompose(K, 1, x)                # x = d0 a + star1^-1 d1^T c + h
    s1 = K.star[1][:, None]

    def ip(a, b):                                                      # star1 inner product, per sample
        return (s1 * a * b).sum(0)

    print("  Hodge decomposition x = exact + coexact + harmonic (star_1 inner product):")
    print(f"    |d1 exact|_max = {float(K.apply_d(1, exact).abs().max()):.1e}   (exact forms are closed)")
    print(f"    |d0^T star1 coexact|_max = {float(K.apply_dT(0, s1 * coexact).abs().max()):.1e}   (co-closed)")
    print(f"    harmonic: |d1 h|_max = {float(K.apply_d(1, harm).abs().max()):.1e}, "
          f"|d0^T star1 h|_max = {float(K.apply_dT(0, s1 * harm).abs().max()):.1e}")
    cos = max(float((ip(a, b).abs() / (ip(a, a) * ip(b, b)).sqrt()).max())
              for a, b in ((exact, coexact), (exact, harm), (coexact, harm)))
    frac = [float(ip(p, p)[0] / ip(x, x)[0]) for p in (exact, coexact, harm)]
    print(f"    max |cosine| between the parts = {cos:.1e}; energy fractions of sample 0 = "
          + ", ".join(f"{f:.3f}" for f in frac))

    # ------------------------------------------------------------------ 4. screened Poisson with batched CG
    rng = np.random.default_rng(0)
    pts = np.concatenate([rng.random((196, 2)), [[0, 0], [1, 0], [0, 1], [1, 1]]])
    Kp = CochainComplex.from_triangles(pts, Delaunay(pts).simplices).to(dtype=F64)
    L0 = dec.hodge_laplacian(Kp, 0)                                    # d0^T star1 d0 = cotan Laplacian
    M = Kp.star[0][:, None]                                            # lumped mass (dual areas), (n0, 1)
    f = torch.randn(Kp.n[0], 3, generator=g, dtype=F64)               # three right-hand sides at once
    u, res = dec.cg_solve(lambda v: M * v + L0(v), M * f, iters=500, tol=1e-12)
    D0 = dense(Kp.d[0])
    A = torch.diag(Kp.star[0]) + D0.T @ torch.diag(Kp.star[1]) @ D0
    u_ref = torch.linalg.solve(A, M * f)
    print(f"\nplanar mesh: {Kp}")
    print(f"  (M + L) u = M f by cg_solve: {res.shape[0] - 1} iterations, final relative residual "
          f"{float(res[-1].max()):.1e}, error vs dense solve {rel(u, u_ref):.1e}")

    # a metric is 'star times material': H_1 = star_1 * sigma_e gives the material-weighted operator
    sigma = torch.exp(torch.randn(Kp.n[1], generator=g, dtype=F64))
    L_sigma = dec.hodge_laplacian(Kp, 0, H={1: Kp.star[1] * sigma})
    v = torch.randn(Kp.n[0], 1, generator=g, dtype=F64)
    print(f"  hodge_laplacian(K, 0, H={{1: star1*sigma}}) == d0^T diag(star1 sigma) d0: "
          f"rel. error {rel(L_sigma(v), D0.T @ ((Kp.star[1] * sigma)[:, None] * (D0 @ v))):.1e}")

    # ------------------------------------------------------------------ 5. Whitney / Galerkin tensor metrics
    Kg = CochainComplex.from_grid((9, 9), 1 / 8, cell="tri", diagonal="main").to(dtype=F64)
    H1 = dense(dec.assemble_whitney_metric(Kg, 1))                    # b = 1, a = 0: the Whitney star
    Dg = dense(Kg.d[0])
    w = Kg.geo[1][:, 2]                                                # raw cotan weight (geo column)
    print(f"\nright-triangle grid: {Kg}")
    print(f"  d0^T H1 d0 (Whitney star) vs cotan Laplacian (= P1 FEM stiffness): "
          f"rel. error {rel(Dg.T @ H1 @ Dg, Dg.T @ torch.diag(w) @ Dg):.1e}")
    n2, m = Kg.n[2], Kg.whitney[1]["t"].shape[1]
    b = torch.exp(0.5 * torch.randn(n2, generator=g, dtype=F64))      # sigma_f = b I + sum_j a_j t_j t_j^T
    a = 2.0 * torch.rand(n2, m, generator=g, dtype=F64)
    Ht = dense(dec.assemble_whitney_metric(Kg, 1, b, a))
    ev = torch.linalg.eigvalsh(Ht)
    bound = float(dec.whitney_rowsum_abs(Kg, 1, b[:, None], a[:, None, :]).max())
    xe = torch.randn(Kg.n[1], 1, 4, generator=g, dtype=F64)
    y = dec.apply_whitney_metric(Kg, 1, b[:, None], a[:, None, :], xe)   # matrix-free, batched (n1, B, C)
    t = Kg.whitney[1]["t"].to(F64)                                     # (n2, m, D) unit edge directions
    sig = b[:, None, None] * torch.eye(2, dtype=F64) + torch.einsum("fj,fjd,fje->fde", a, t, t)
    ratio = torch.linalg.eigvalsh(sig)
    print(f"  random material tensors: anisotropy ratios in [{float((ratio[:, 1] / ratio[:, 0]).min()):.2f}, "
          f"{float((ratio[:, 1] / ratio[:, 0]).max()):.2f}]")
    print(f"  H1(b, a) is SPD: eigenvalues in [{float(ev.min()):.2e}, {float(ev.max()):.2e}], "
          f"row-sum bound {bound:.2e} >= lambda_max")
    print(f"  apply_whitney_metric == assembled matrix: rel. error {rel(y[:, 0], Ht @ xe[:, 0]):.1e}")

    # ------------------------------------------------------------------ 6. flat and sharp (Whitney maps)
    xy = Kp.pos.to(F64)
    for name, field in (("constant (1, -2)", torch.tensor([1.0, -2.0], dtype=F64).expand(Kp.n[0], 2)),
                        ("rotation (-y, x)", torch.stack([-xy[:, 1], xy[:, 0]], 1))):
        c = dec.flat(Kp, field)                                        # (n1,) line integrals along edges
        back = dec.sharp(Kp, c)                                        # (n0, D) Whitney interpolation
        print(f"  sharp(flat(v)) for a {name} field: max error {float((back - field).abs().max()):.1e}")

    assert betti == [1, 2, 1] and cos < 1e-6 and rel(u, u_ref) < 1e-8 and float(ev.min()) > 0
    print("\nEXAMPLE 02 OK")


if __name__ == "__main__":
    main()
