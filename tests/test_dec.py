"""Tests for rhmp.dec (DESIGN §9): Whitney metric application, batched CG, metric Hodge Laplacians,
Hodge decomposition, harmonic forms and Whitney sharp/flat."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import make_batch, make_complex, mesh_plane_grid, random_orthogonal

from rhmp import dec
from rhmp.complex import CochainComplex
from rhmp.ops import beta_unit

F64 = torch.float64


def dense(A: torch.Tensor) -> torch.Tensor:
    return A.to_dense().double().cpu()


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def rand_ba(K, k, B, seed, dtype=torch.float32, device="cpu"):
    W = K.whitney[k]
    n_top, m = W["G0"].shape[0], W["t"].shape[1]
    g = gen(seed)
    b = torch.exp(0.5 * torch.randn(n_top, B, generator=g, dtype=F64)).to(device, dtype)
    a = (2.0 * torch.rand(n_top, B, m, generator=g, dtype=F64)).to(device, dtype)
    return b, a


# ----------------------------------------------------------------------------------------------
# Whitney tensor metric
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,k", [("delaunay", 1), ("sphere", 1), ("tets", 1), ("tets", 2)])
def test_apply_whitney_metric_matches_assembled(name, k, device):
    K = make_complex(name, device)
    b, a = rand_ba(K, k, 3, 0, device=device)
    x = torch.randn(K.n[k], 3, 5, generator=gen(1)).to(device)
    y = dec.apply_whitney_metric(K, k, b, a, x)
    assert y.shape == x.shape and y.dtype == torch.float32
    for s in range(3):
        H = dense(dec.assemble_whitney_metric(K, k, b[:, s].double(), a[:, s].double()))
        ref = H @ x[:, s].double().cpu()
        torch.testing.assert_close(y[:, s].double().cpu(), ref, rtol=1e-5, atol=1e-5 * float(ref.abs().max()))
    # isotropic (a = None) and shared-over-samples b
    y0 = dec.apply_whitney_metric(K, k, b[:, :1], None, x)
    H0 = dense(dec.assemble_whitney_metric(K, k, b[:, 0].double()))
    ref0 = torch.einsum("ij,jbc->ibc", H0, x.double().cpu())
    torch.testing.assert_close(y0.double().cpu(), ref0, rtol=1e-5, atol=1e-5 * float(ref0.abs().max()))


def test_whitney_metric_shape_errors():
    K = make_complex("delaunay")
    b, a = rand_ba(K, 1, 2, 0)
    with pytest.raises(ValueError, match="n_top"):
        dec.apply_whitney_metric(K, 1, b[:-1], a, torch.randn(K.n[1], 2, 3))
    with pytest.raises(ValueError, match="x must be"):
        dec.apply_whitney_metric(K, 1, b, a, torch.randn(K.n[0], 2, 3))
    with pytest.raises(ValueError, match="no Whitney blocks"):
        dec.apply_whitney_metric(make_complex("quads"), 1, b, a, torch.randn(K.n[1], 2, 3))
    with pytest.raises(ValueError, match="m="):
        dec.whitney_blocks(K, 1, b, a[..., :2])


def test_whitney_metric_gradcheck(device):
    K = make_complex("grid", device, nx=4, ny=4).to(dtype=F64)
    b, a = rand_ba(K, 1, 2, 3, F64, device)
    x = torch.randn(K.n[1], 2, 3, dtype=F64, generator=gen(4)).to(device)
    b.requires_grad_(True)
    a.requires_grad_(True)
    x.requires_grad_(True)
    assert torch.autograd.gradcheck(lambda b, a, x: dec.apply_whitney_metric(K, 1, b, a, x), (b, a, x))
    assert torch.autograd.gradcheck(lambda b, a: dec.whitney_rowsum_abs(K, 1, b, a), (b, a))


@pytest.mark.parametrize("name,k", [("delaunay", 1), ("sliver", 1), ("tets", 1), ("tets", 2)])
def test_whitney_rowsum_and_beta_bounds(name, k):
    K = make_complex(name)
    b, a = rand_ba(K, k, 2, 5)
    r = dec.whitney_rowsum_abs(K, k, b, a).double()
    A = dense(K.d[k - 1])
    s = K.star[k - 1].double().rsqrt()
    bu = float(beta_unit(K.d_abs[k - 1], K.dT_abs[k - 1], K.star[k - 1].rsqrt()))
    for j in range(2):
        H = dense(dec.assemble_whitney_metric(K, k, b[:, j].double(), a[:, j].double()))
        assert bool((r[:, j] >= H.abs().sum(1) * (1 - 1e-5)).all())
        assert float(torch.linalg.eigvalsh(H)[-1]) <= float(r[:, j].max()) * (1 + 1e-5)
        L = s[:, None] * (A.T @ H @ A) * s[None, :]  # normalised up-block of degree k-1 with the tensor metric
        assert float(torch.linalg.eigvalsh(L)[-1]) <= float(r[:, j].max()) * bu * (1 + 1e-5)


def test_whitney_metric_is_per_sample(device):
    K = make_complex("delaunay", device)
    b, a = rand_ba(K, 1, 4, 7, device=device)
    x = torch.randn(K.n[1], 4, 8, generator=gen(8)).to(device)
    y = dec.apply_whitney_metric(K, 1, b, a, x)
    for s in range(4):
        y1 = dec.apply_whitney_metric(K, 1, b[:, s:s + 1], a[:, s:s + 1], x[:, s:s + 1].contiguous())
        torch.testing.assert_close(y[:, s:s + 1], y1, rtol=1e-6, atol=1e-6)


def test_whitney_metric_on_block_diagonal_batch(device):
    Kb, parts = make_batch(device)
    b, a = rand_ba(Kb, 1, 1, 9, device=device)
    x = torch.randn(Kb.n[1], 1, 4, generator=gen(10)).to(device)
    y = dec.apply_whitney_metric(Kb, 1, b, a, x)
    ot = oe = 0
    for P in parts:
        yp = dec.apply_whitney_metric(P, 1, b[ot:ot + P.n[2]], a[ot:ot + P.n[2]], x[oe:oe + P.n[1]].contiguous())
        torch.testing.assert_close(y[oe:oe + P.n[1]], yp, rtol=1e-5, atol=1e-5)
        ot, oe = ot + P.n[2], oe + P.n[1]


# ----------------------------------------------------------------------------------------------
# conjugate gradients
# ----------------------------------------------------------------------------------------------
def _shifted_laplacian(K, h, tau=2.0):
    """A = I + tau d0^T diag(h) d0 on (n0, B, ...); h (n1,) or (n1, B)."""
    def mv(x):
        hh = h.view(tuple(h.shape) + (1,) * (x.dim() - h.dim()))
        return x + tau * K.apply_dT(0, (hh * K.apply_d(0, x)).contiguous())
    return mv


def _dense_shifted(K, h, tau=2.0):
    D0 = dense(K.d[0]).to(h.dtype)
    return torch.eye(K.n[0], dtype=h.dtype) + tau * D0.T @ (h[:, None] * D0)


def test_cg_solves_spd_systems(device):
    K = make_complex("delaunay", device).to(dtype=F64)
    h = torch.exp(torch.randn(K.n[1], generator=gen(0), dtype=F64)).to(device)
    rhs = torch.randn(K.n[0], 3, 4, generator=gen(1), dtype=F64).to(device)
    x, res = dec.cg_solve(_shifted_laplacian(K, h), rhs, iters=300, tol=1e-12)
    ref = torch.linalg.solve(_dense_shifted(K, h.cpu()), rhs.cpu().reshape(K.n[0], -1)).reshape(rhs.shape)
    torch.testing.assert_close(x.cpu(), ref, rtol=1e-9, atol=1e-9)
    assert res.shape[1:] == (1, 3) and bool((res[-1] <= 1e-12).all())
    # early_exit=False runs exactly `iters` iterations (no host sync); frozen samples stop changing
    xf, resf = dec.cg_solve(_shifted_laplacian(K, h), rhs, iters=400, tol=1e-12, early_exit=False)
    assert resf.shape[0] == 401
    torch.testing.assert_close(xf, x, rtol=0, atol=0)
    _, res0 = dec.cg_solve(_shifted_laplacian(K, h), rhs, iters=7, tol=0.0, early_exit=False)
    assert res0.shape[0] == 8
    # (n, B) right-hand sides and a warm start
    x2, _ = dec.cg_solve(_shifted_laplacian(K, h), rhs[..., 0].contiguous(), iters=300, tol=1e-12, x0=x[..., 0] * 0.5)
    torch.testing.assert_close(x2.cpu(), ref[..., 0], rtol=1e-9, atol=1e-9)


def test_cg_is_O_C_equivariant_and_per_sample(device):
    K = make_complex("sphere", device).to(dtype=F64)
    h = torch.exp(torch.randn(K.n[1], 3, generator=gen(2), dtype=F64)).to(device)  # per-sample metric
    rhs = torch.randn(K.n[0], 3, 6, generator=gen(3), dtype=F64).to(device)
    A = _shifted_laplacian(K, h)
    for reflect in (False, True):
        R = torch.as_tensor(random_orthogonal(6, seed=4, reflect=reflect), dtype=F64, device=device)
        xa, _ = dec.cg_solve(A, rhs @ R, iters=15, tol=0.0, early_exit=False)  # unconverged on purpose
        xb, _ = dec.cg_solve(A, rhs, iters=15, tol=0.0, early_exit=False)
        torch.testing.assert_close(xa, xb @ R, rtol=1e-10, atol=1e-10)
    x, _ = dec.cg_solve(A, rhs, iters=15, tol=0.0, early_exit=False)
    for s in range(3):
        xs, _ = dec.cg_solve(_shifted_laplacian(K, h[:, s:s + 1]), rhs[:, s:s + 1].contiguous(), iters=15, tol=0.0,
                             early_exit=False)
        torch.testing.assert_close(x[:, s:s + 1], xs, rtol=1e-12, atol=1e-12)


def test_cg_block_diagonal_batch(device):
    Kb, parts = make_batch(device)
    Kb = Kb.to(dtype=F64)
    h = torch.exp(torch.randn(Kb.n[1], generator=gen(5), dtype=F64)).to(device)
    rhs = torch.randn(Kb.n[0], 1, 3, generator=gen(6), dtype=F64).to(device)
    x, res = dec.cg_solve(_shifted_laplacian(Kb, h), rhs, iters=12, tol=0.0, batch=Kb.batch[0],
                          num_graphs=Kb.num_graphs, early_exit=False)
    assert res.shape == (13, 3, 1)
    o0 = o1 = 0
    for P in parts:
        P = P.to(dtype=F64)
        xp, _ = dec.cg_solve(_shifted_laplacian(P, h[o1:o1 + P.n[1]]), rhs[o0:o0 + P.n[0]].contiguous(), iters=12,
                             tol=0.0, early_exit=False)
        torch.testing.assert_close(x[o0:o0 + P.n[0]], xp, rtol=1e-10, atol=1e-10)
        o0, o1 = o0 + P.n[0], o1 + P.n[1]


def test_cg_gradients_unrolled_and_implicit_match_dense():
    K = make_complex("grid", nx=5, ny=4).to(dtype=F64)
    h0 = torch.exp(torch.randn(K.n[1], generator=gen(7), dtype=F64))
    rhs0 = torch.randn(K.n[0], 2, 3, generator=gen(8), dtype=F64)
    w = torch.randn(K.n[0], 2, 3, generator=gen(9), dtype=F64)

    def grads(solver):
        h = h0.clone().requires_grad_(True)
        rhs = rhs0.clone().requires_grad_(True)
        x = solver(h, rhs)
        (x * w).sum().backward()
        return h.grad, rhs.grad

    def dense_solver(h, rhs):
        return torch.linalg.solve(_dense_shifted(K, h), rhs.reshape(K.n[0], -1)).reshape(rhs.shape)

    def unrolled(h, rhs):
        return dec.cg_solve(_shifted_laplacian(K, h), rhs, iters=200, tol=1e-14)[0]

    def implicit(h, rhs):
        return dec.cg_solve_implicit(_shifted_laplacian(K, h), rhs, params=(h,), iters=200, tol=1e-14)[0]

    gh, gr = grads(dense_solver)
    for solver in (unrolled, implicit):
        gh2, gr2 = grads(solver)
        torch.testing.assert_close(gr2, gr, rtol=1e-8, atol=1e-10)
        torch.testing.assert_close(gh2, gh, rtol=1e-8, atol=1e-10)


def test_cg_fp32_unrolled_backward_stays_finite_after_convergence(device):
    """Many more iterations than needed (tol ~ 0): the floor-frozen, rhs-normalised CG keeps gradients finite."""
    K = make_complex("delaunay", device)
    h = torch.exp(torch.randn(K.n[1], generator=gen(20))).to(device).requires_grad_(True)
    rhs = torch.randn(K.n[0], 2, 4, generator=gen(21)).to(device).mul(1e-3).requires_grad_(True)
    x, res = dec.cg_solve(_shifted_laplacian(K, h), rhs, iters=300, tol=0.0, early_exit=False)
    x.square().sum().backward()
    assert bool(torch.isfinite(h.grad).all()) and bool(torch.isfinite(rhs.grad).all())
    assert bool(torch.isfinite(res).all())


# ----------------------------------------------------------------------------------------------
# Hodge Laplacians, decomposition, harmonic forms
# ----------------------------------------------------------------------------------------------
def _dense_operator(f, n, dtype=F64):
    E = torch.eye(n, dtype=dtype)[:, None, :]  # (n, 1, n): columns as channels
    return f(E.contiguous())[:, 0, :].cpu()


@pytest.mark.parametrize("name,k,betti", [("torus", 1, 2), ("torus", 0, 1), ("torus", 2, 1), ("sphere", 1, 0),
                                          ("sphere", 2, 1), ("grid", 1, 0), ("grid", 2, 0), ("tets", 1, 0),
                                          ("tets", 2, 0), ("tets", 3, 0), ("tets", 0, 1)])
def test_hodge_laplacian_symmetric_psd_and_betti(name, k, betti):
    K = make_complex(name).to(dtype=F64)
    L = _dense_operator(dec.hodge_laplacian(K, k), K.n[k])
    torch.testing.assert_close(L, L.T, rtol=1e-10, atol=1e-10 * float(L.abs().max()))
    ev = torch.linalg.eigvalsh(L)
    assert float(ev[0]) > -1e-8 * float(ev[-1])
    assert int((ev < 1e-8 * float(ev[-1])).sum()) == betti
    # strong form = H^-1 weak form
    Ls = _dense_operator(dec.hodge_laplacian(K, k, weak=False), K.n[k])
    torch.testing.assert_close(Ls, L / K.star[k].double()[:, None], rtol=1e-10, atol=1e-10 * float(Ls.abs().max()))


def test_hodge_laplacian_with_whitney_up_metric():
    """L_0 with the Galerkin H_1(b=1, a=0) is the P1 stiffness; symmetric PSD with the tensor metric."""
    K = make_complex("delaunay").to(dtype=F64)
    b, a = rand_ba(K, 1, 1, 11, F64)
    L = _dense_operator(dec.hodge_laplacian(K, 0, {1: lambda x: dec.apply_whitney_metric(K, 1, b, a, x)}), K.n[0])
    torch.testing.assert_close(L, L.T, rtol=1e-10, atol=1e-10 * float(L.abs().max()))
    ev = torch.linalg.eigvalsh(L)
    assert int((ev < 1e-9 * float(ev[-1])).sum()) == 1  # constants only
    with pytest.raises(ValueError, match="inverse"):
        dec.hodge_laplacian(K, 1, {0: lambda x: x})(torch.randn(K.n[1], 1, 1, dtype=F64))


def _Hdot(u, v, h):
    hh = h.view(tuple(h.shape) + (1,) * (u.dim() - h.dim()))
    return (u * hh * v).sum(0)


def test_hodge_decompose_torus():
    K = make_complex("torus").to(dtype=F64)
    B = 2
    h1 = torch.exp(0.5 * torch.randn(K.n[1], B, generator=gen(12), dtype=F64))  # per-sample metric
    x = torch.randn(K.n[1], B, 3, generator=gen(13), dtype=F64)
    ex, co, ha = dec.hodge_decompose(K, 1, x, {1: h1}, iters=2000, tol=1e-13)
    torch.testing.assert_close(ex + co + ha, x)
    scale = float(x.abs().max())
    for u, v in ((ex, co), (ex, ha), (co, ha)):
        assert float(_Hdot(u, v, h1).abs().max()) < 1e-8 * scale ** 2 * K.n[1]
    assert float(K.apply_d(1, ex).abs().max()) < 1e-10 * scale  # exact parts are closed
    assert float(K.apply_dT(0, (h1[..., None] * co).contiguous()).abs().max()) < 1e-7 * scale  # co-closed
    assert float(K.apply_d(1, ha).abs().max()) < 1e-7 * scale
    assert float(K.apply_dT(0, (h1[..., None] * ha).contiguous()).abs().max()) < 1e-7 * scale
    assert float(ha.norm()) > 1e-2 * float(x.norm())  # genus 1: a genuine harmonic part


def test_hodge_decompose_disk_sphere_and_batch():
    Kd = make_complex("grid").to(dtype=F64)  # disk: H^1 = 0
    x = torch.randn(Kd.n[1], 2, 2, generator=gen(14), dtype=F64)
    _, _, ha = dec.hodge_decompose(Kd, 1, x, iters=2000, tol=1e-13)
    assert float(ha.norm()) < 1e-7 * float(x.norm())
    Ks = make_complex("sphere").to(dtype=F64)  # degree 0: harmonic part = H0-weighted mean (constants)
    f = torch.randn(Ks.n[0], 1, 1, generator=gen(15), dtype=F64)
    ex, co, ha = dec.hodge_decompose(Ks, 0, f, iters=2000, tol=1e-13)
    assert float(ex.abs().max()) == 0.0
    s0 = Ks.star[0].double()
    mean = (s0[:, None, None] * f).sum() / s0.sum()
    torch.testing.assert_close(ha, torch.full_like(f, float(mean)), rtol=1e-8, atol=1e-8)
    # block-diagonal batch == individual decompositions (per-graph CG)
    parts = [make_complex("torus"), make_complex("torus", n_major=10, n_minor=6)]
    Kb = CochainComplex.batch(parts).to(dtype=F64)
    xb = torch.randn(Kb.n[1], 1, 2, generator=gen(16), dtype=F64)
    outs = dec.hodge_decompose(Kb, 1, xb, iters=2000, tol=1e-13)
    o = 0
    for P in parts:
        P = P.to(dtype=F64)
        outp = dec.hodge_decompose(P, 1, xb[o:o + P.n[1]].contiguous(), iters=2000, tol=1e-13)
        for u, v in zip(outs, outp):
            torch.testing.assert_close(u[o:o + P.n[1]], v, rtol=1e-7, atol=1e-8)
        o += P.n[1]


@pytest.mark.parametrize("name,k,betti", [("torus", 1, 2), ("sphere", 1, 0), ("sphere", 2, 1), ("torus", 0, 1),
                                          ("grid", 1, 0), ("tets", 1, 0), ("tets", 2, 0)])
def test_harmonic_basis(name, k, betti):
    K = make_complex(name).to(dtype=F64)
    Hb = dec.harmonic_basis(K, k)
    assert Hb.shape == (K.n[k], betti)
    if betti:
        hk = K.star[k].double()
        torch.testing.assert_close(Hb.T @ (hk[:, None] * Hb), torch.eye(betti, dtype=F64), rtol=1e-8, atol=1e-8)
        L = dec.hodge_laplacian(K, k)
        assert float(L(Hb[:, None, :].contiguous()).abs().max()) < 1e-6 * float(Hb.abs().max())


# ----------------------------------------------------------------------------------------------
# sharp / flat
# ----------------------------------------------------------------------------------------------
def _J(p):  # 2-D rotation field J p
    return torch.stack([-p[..., 1], p[..., 0]], -1)


@pytest.mark.parametrize("name", ["delaunay", "grid", "tets"])
def test_sharp_flat_exact_on_constant_and_rotation_fields(name, device):
    K = make_complex(name, device).to(dtype=F64)
    P = K.pos.double()
    D = P.shape[1]
    v0 = torch.tensor([0.3, -1.2, 0.7][:D], dtype=F64, device=device)
    for field in ("const", "rot"):
        if field == "const":
            v = v0.expand(K.n[0], D).clone()
        else:
            v = _J(P) if D == 2 else torch.linalg.cross(v0.expand(K.n[0], 3), P, dim=-1)
        x = dec.flat(K, v)
        vs = dec.sharp(K, x)
        torch.testing.assert_close(vs, v, rtol=1e-5, atol=1e-5 * float(v.abs().max()))
        vc = dec.sharp(K, x, at="cell")
        cen = P[K.cells[K.dim]].mean(1)
        vref = v0.expand_as(cen) if field == "const" else (_J(cen) if D == 2 else torch.linalg.cross(
            v0.expand_as(cen), cen, dim=-1))
        torch.testing.assert_close(vc, vref, rtol=1e-5, atol=1e-5 * float(vref.abs().max()))
        torch.testing.assert_close(dec.flat(K, vs), x, rtol=1e-5, atol=1e-5 * float(x.abs().max()))


def test_flat_is_exact_line_integral_and_shapes():
    K = make_complex("delaunay").to(dtype=F64)
    P = K.pos.double()
    A = torch.tensor([[0.5, -2.0], [1.5, 0.25]], dtype=F64)
    c = torch.tensor([0.1, -0.3], dtype=F64)
    v = P @ A.T + c
    E = K.cells[1]
    ps, pd = P[E[:, 0]], P[E[:, 1]]
    exact = (((ps + pd) / 2) @ A.T + c).mul(pd - ps).sum(-1)  # linear field: midpoint rule is exact
    torch.testing.assert_close(dec.flat(K, v), exact)
    xb = torch.randn(K.n[1], 2, 3, generator=gen(17), dtype=F64)
    assert dec.sharp(K, xb).shape == (K.n[0], 2, 3, 2) and dec.sharp(K, xb, at="cell").shape == (K.n[2], 2, 3, 2)
    assert dec.flat(K, dec.sharp(K, xb)).shape == (K.n[1], 2, 3)
    with pytest.raises(ValueError):
        dec.flat(K, torch.randn(K.n[0], 3, dtype=F64))
    pos, faces = mesh_plane_grid(5, 5, dim=3)
    Q = random_orthogonal(3, seed=2)
    K3 = CochainComplex.from_triangles(pos @ Q.T, faces).to(dtype=F64)  # planar mesh embedded in 3-D
    v3 = torch.tensor([1.0, 2.0, 0.0], dtype=F64) @ torch.tensor(Q.T, dtype=F64)  # in-plane constant field
    vv = v3.expand(K3.n[0], 3).clone()
    torch.testing.assert_close(dec.sharp(K3, dec.flat(K3, vv)), vv, rtol=1e-5, atol=1e-5)


def test_sharp_requires_simplicial_complex():
    with pytest.raises(ValueError, match="Whitney"):
        dec.sharp(make_complex("quads"), torch.randn(make_complex("quads").n[1]))
