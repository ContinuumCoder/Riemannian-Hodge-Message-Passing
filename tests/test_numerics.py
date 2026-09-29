"""Numerical tests: operator norm bounds (dense eigvalsh), DEC form of the blocks at initialisation, fused-kernel
gradients, resolution consistency, robustness (slivers, extreme inputs, extreme log_range), fp64 vs fp32, and the
least-squares vector readout (exactness, tangency, gradcheck).
"""
from __future__ import annotations

import copy
import dataclasses
import math

import pytest
import torch
from torch.autograd import gradcheck

from conftest import make_complex, random_orthogonal
from rhmp import CochainComplex, ops
from rhmp.layers import apply_L, apply_T, block_operands, make_context, poly_block, poly_block_reference, spmm_ad
from rhmp.readout import NodeVectorReadout, vector_geometry
from test_model import F64, all_finite, build_model, loss_of, make_inputs, randomize_, rel, small_cfg


def _dense(apply, n_in: int, B: int, like: torch.Tensor) -> torch.Tensor:
    """Dense matrices of a batched linear map: ``M[b, i, j] = apply(e_j)[i, b]``, shape ``(B, n_out, n_in)``."""
    eye = torch.eye(n_in, dtype=like.dtype, device=like.device)
    z = eye.unsqueeze(1).expand(n_in, B, n_in).contiguous()                  # channel j carries e_j
    return apply(z).permute(1, 0, 2)


def _blocks(K):
    """All (degree, is_up) blocks of a complex."""
    return [(k, up) for k in range(K.dim + 1) for up in (True, False) if (up and k < K.dim) or (not up and k > 0)]


def _nb(k: int, up: bool) -> int:
    return k + 1 if up else k - 1


# ----------------------------------------------------------------------------------------------------------------
# operator bounds and DEC structure
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["grid", "sphere", "torus", "mixed", "tets", "nonmanifold", "sliver"])
@pytest.mark.parametrize("scaling", ["dec", "jacobi", "none"])
def test_operator_norm_bound(name, scaling, device):
    """||L_up||, ||L_down|| <= 1 and ||T|| <= 1 for random metrics (dense eigvalsh), L = T T^T, and the
    Gershgorin normaliser agrees with ``ops.gershgorin_bound``."""
    K = make_complex(name, device).to(dtype=F64)
    B = 3
    ctx = make_context(K, scaling=scaling, B=B, dtype=F64)
    g = torch.Generator().manual_seed(0)
    for k, up in _blocks(K):
        nb = _nb(k, up)
        logH = (torch.randn(K.n[nb], B, generator=g, dtype=F64) * 2.0).to(device)     # H spans ~e^{+-6}
        op = block_operands(ctx, k, up, logH)
        L = _dense(lambda z: apply_L(op, z), K.n[k], B, logH)
        assert (L - L.transpose(1, 2)).abs().max() <= 1e-12 * L.abs().max()
        ev = torch.linalg.eigvalsh(L)
        assert float(ev.max()) <= 1.0 + 1e-10, (k, up, float(ev.max()))
        assert float(ev.min()) >= -1e-10
        assert float(ev.max(dim=1).values.min()) >= 1e-2          # the bound is not grossly loose
        T = _dense(lambda y: apply_T(op, y), K.n[nb], B, logH)    # (B, n_k, m)
        assert rel(T @ T.transpose(1, 2), L) < 1e-12
        assert float(torch.linalg.matrix_norm(T, ord=2).max()) <= 1.0 + 1e-10
        # Gershgorin normaliser == core implementation
        A_abs, AT_abs = (K.d_abs[k], K.dT_abs[k]) if up else (K.dT_abs[k - 1], K.d_abs[k - 1])
        s = op.scale.squeeze(-1) if op.scale is not None else torch.ones(K.n[k], dtype=F64, device=device)
        ref = ops.gershgorin_bound(A_abs, AT_abs, logH.exp(), s)
        assert rel(op.beta.reshape(-1), ref.reshape(-1)) < 1e-12


@pytest.mark.parametrize("name", ["grid", "sphere", "tets"])
def test_dec_blocks_at_init(name, device):
    """With H = star (tied: 1/star for down blocks) and scaling='dec', the blocks are proportional to the
    symmetrised DEC pieces:  up  star^-1/2 d^T star d star^-1/2,  down  star^+1/2 d star^-1 d^T star^+1/2,
    and the transports are  T_up ~ star_k^-1/2 d^T star_{k+1}^+1/2,  T_down ~ star_k^+1/2 d star_{k-1}^-1/2."""
    K = make_complex(name, device).to(dtype=F64)
    ctx = make_context(K, scaling="dec", B=1, dtype=F64)
    star = [s.to(F64) for s in K.star]
    for k, up in _blocks(K):
        nb = _nb(k, up)
        logH = (torch.log(star[nb]) * (1.0 if up else -1.0)).unsqueeze(1)
        op = block_operands(ctx, k, up, logH)
        L = _dense(lambda z: apply_L(op, z), K.n[k], 1, logH)[0]
        T = _dense(lambda y: apply_T(op, y), K.n[nb], 1, logH)[0]
        if up:
            D = K.d[k].to_dense()                                            # (n_{k+1}, n_k)
            Tref = star[k].rsqrt()[:, None] * D.T * star[nb].sqrt()[None, :]
        else:
            D = K.d[k - 1].to_dense()                                        # (n_k, n_{k-1})
            Tref = star[k].sqrt()[:, None] * D * star[nb].rsqrt()[None, :]
        Lref = Tref @ Tref.T
        assert rel(L / L.norm(), Lref / Lref.norm()) < 1e-12, (k, up)
        assert rel(T / T.norm(), Tref / Tref.norm()) < 1e-12, (k, up)


def test_resolution_consistency_L_up(device):
    """At initialisation (H = star) the DEC-symmetrised L_up of degree 0, un-normalised and mapped back to function
    values, is the cotan Laplacian: applied to u = sin(pi x) sin(pi y) on a coarse and a 2x-refined grid, the two
    results agree at the common vertices (and converge to -Delta u = 2 pi^2 u)."""
    res = {}
    for n in (9, 17, 33):
        K = CochainComplex.from_grid((n, n), 1.0 / (n - 1), cell="tri", diagonal="main").to(device, dtype=F64)
        ctx = make_context(K, scaling="dec", B=1, dtype=F64)
        op = block_operands(ctx, 0, True, torch.log(K.star[1]).unsqueeze(1))  # physical star on edges
        x, y = K.pos[:, 0], K.pos[:, 1]
        u = (torch.sin(math.pi * x) * torch.sin(math.pi * y)).view(-1, 1, 1)
        gm0 = torch.exp(torch.log(K.star[0]).mean())
        # frame x = s^-1 u (s = normalised star0^-1/2); beta * L = s d^T star1 d s; map back with s / gm0
        sc = op.scale
        lap = (sc * (op.beta.reshape(()) * apply_L(op, u / sc)) / gm0).view(n, n)
        exact = (2 * math.pi ** 2 * u).view(n, n)
        res[n] = (lap, exact)
    scale = float(res[9][1].abs().max())

    def interior(n_coarse: int, a: torch.Tensor) -> torch.Tensor:
        m = (n_coarse - 1) // 4                                             # stay >= 1/4 away from the boundary
        return a[m:n_coarse - m, m:n_coarse - m]

    errs = []
    for nc, nf in ((9, 17), (17, 33)):
        lc, ec = res[nc]
        lf = res[nf][0][::2, ::2]                                           # fine values at coarse vertices
        diff = float((interior(nc, lc) - interior(nc, lf)).abs().max()) / scale
        errs.append(diff)
        assert diff < 2.0 / (nc - 1), (nc, diff)                            # O(h) agreement (observed O(h^2))
        err_c = float((interior(nc, lc) - interior(nc, ec)).abs().max()) / scale
        assert err_c < 0.05, (nc, err_c)                                    # consistent with -Delta (1% clamp bias)
    assert errs[1] < 0.5 * errs[0]                                          # the discrepancy shrinks with h


def test_uniform_scaling_invariance(device):
    """Scalar outputs do not depend on the length unit (per-sample star normalisation + scale-free operators and
    cross terms); vector outputs scale like 1/c (they reconstruct v from line integrals w = l t.v)."""
    from conftest import mesh_delaunay_2d
    pos, faces = mesh_delaunay_2d(60, seed=3)
    Ks = {c: CochainComplex.from_triangles(pos * c, faces, device=device) for c in (1.0, 1e-3, 1e3)}
    for readout in ("node_scalar", "cochain:1", "even:2", "node_vector"):
        cfg = small_cfg(readout=readout)
        m = build_model(cfg, Ks[1.0])
        inp = make_inputs(cfg, Ks[1.0])
        y1 = m(inp, Ks[1.0])
        for c in (1e-3, 1e3):
            yc = m(inp, Ks[c])
            if readout == "node_vector":
                yc = yc * c
            assert rel(yc, y1) < 1e-5, (readout, c)


# ----------------------------------------------------------------------------------------------------------------
# fused polynomial kernel
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("scaling", ["dec", "none"])
@pytest.mark.parametrize("P", [1, 2, 3])
def test_fused_block_gradcheck(scaling, P, device):
    K = make_complex("grid", device, nx=4, ny=4).to(dtype=F64)
    B, C = 2, 3
    ctx = make_context(K, scaling=scaling, B=B, dtype=F64)
    g = torch.Generator().manual_seed(0)

    def rnd(*shape, pos=False):
        t = torch.randn(*shape, generator=g, dtype=F64)
        return (t.abs() + 0.1 if pos else t).to(device).requires_grad_()

    for k, up in _blocks(K):
        nb = _nb(k, up)
        op = block_operands(ctx, k, up, torch.randn(K.n[nb], B, generator=g, dtype=F64).to(device))
        x, X = rnd(K.n[k], B, C), rnd(K.n[nb], B, C)
        u, c = rnd(K.n[nb], B, 1), rnd(P)
        # an independent metric leaf close to the normalised one (keeps ||L|| ~ 1, so values stay O(1))
        h = (op.log_h.unsqueeze(-1) + 0.3 * torch.randn(K.n[nb], B, 1, generator=g, dtype=F64).to(device)).exp()
        h = h.detach().requires_grad_()

        def fused(x, h, c, u, X):
            q1 = spmm_ad(op.A, op.AT, x if op.s is None else x * op.s)
            return poly_block(q1, h, c, u, X, op)

        def ref(x, h, c, u, X):
            return poly_block_reference(x, h, c, u, X, op)

        assert rel(fused(x, h, c, u, X), ref(x, h, c, u, X)) < 1e-13
        assert gradcheck(fused, (x, h, c, u, X), eps=1e-6, atol=1e-6, rtol=1e-5)
        assert gradcheck(lambda x, h, c: fused(x, h, c, None, None), (x, h, c), eps=1e-6, atol=1e-6, rtol=1e-5)


def test_fused_block_rejects_learned_scaling():
    K = make_complex("grid", nx=4, ny=4).to(dtype=F64)
    ctx = make_context(K, scaling="jacobi", B=1, dtype=F64)
    logH = torch.zeros(K.n[1], 1, dtype=F64, requires_grad=True)
    op = block_operands(ctx, 0, True, logH)
    assert op.s is not None and op.s.requires_grad
    x = torch.randn(K.n[0], 1, 2, dtype=F64)
    with pytest.raises(ValueError):
        poly_block(spmm_ad(op.A, op.AT, x * op.s), op.log_h.exp().unsqueeze(-1), torch.ones(1, dtype=F64), None,
                   None, op)


# ----------------------------------------------------------------------------------------------------------------
# robustness
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("scaling", ["dec", "jacobi", "none"])
@pytest.mark.parametrize("readout", ["node_scalar", "node_vector", "cochain:1"])
def test_sliver_mesh_finite(scaling, readout, device):
    K = make_complex("sliver", device, aspect=1e4)
    assert K.meta["geometry_stats"]["max_log_aspect"] > math.log(1e3)
    cfg = small_cfg(scaling=scaling, readout=readout)
    m = build_model(cfg, K)
    y = m(make_inputs(cfg, K), K)
    loss_of(y).backward()
    assert all_finite(m, y)
    assert all(math.isfinite(v) for v in m.diagnostics.values())


@pytest.mark.parametrize("readout", ["node_scalar", "node_vector", "even:1"])
def test_extreme_even_inputs_finite(readout, device):
    K = make_complex("delaunay", device)
    cfg = small_cfg(in_dims={0: 3, 1: 3, 2: 2}, even_dims={0: 2, 1: 1, 2: 1}, readout=readout)
    m = build_model(cfg, K)
    inp = make_inputs(cfg, K, even_range=(-6.0, 6.0))
    y = m(inp, K)
    loss_of(y).backward()
    assert all_finite(m, y)


def test_extreme_log_range_finite(device):
    K = make_complex("sphere", device)
    cfg = small_cfg(log_range=30.0, readout="node_vector")
    m = build_model(cfg, K)
    randomize_(m, scale=5.0, seed=3)
    with torch.no_grad():                    # saturate the metric heads: H / star at e^{+-30}
        for layer in m.layers:
            for head in layer.heads.values():
                head.fc2.weight.mul_(100.0)
    inp = make_inputs(cfg, K)
    y = m(inp, K)
    loss_of(y).backward()
    assert all_finite(m, y)
    d = m.diagnostics
    assert max(v for k, v in d.items() if k.endswith(".max")) > 10.0          # the metric really is extreme
    # the normalised operators stay bounded for such metrics
    ctx = make_context(K.to(dtype=F64), scaling="dec", B=1, dtype=F64)
    logH = (torch.rand(K.n[1], 1, dtype=F64) * 60.0 - 30.0).to(device)
    op = block_operands(ctx, 0, True, logH)
    L = _dense(lambda z: apply_L(op, z), K.n[0], 1, logH)
    assert float(torch.linalg.eigvalsh(L).max()) <= 1.0 + 1e-9


def test_fp64_vs_fp32(device):
    K32 = make_complex("sphere", device)
    K64 = K32.to(dtype=F64)
    for readout in ("node_scalar", "node_vector", "cochain:1"):
        cfg = small_cfg(readout=readout, n_layers=3)
        m32 = build_model(cfg, K32)
        m64 = copy.deepcopy(m32).to(F64)
        inp = make_inputs(cfg, K32, dtype=F64)
        y64 = m64(inp, K64)
        y32 = m32({k: v.float() for k, v in inp.items()}, K32)
        assert y32.dtype == torch.float32 and y64.dtype == F64
        assert rel(y32, y64) < 1e-4, readout


# ----------------------------------------------------------------------------------------------------------------
# least-squares vector readout
# ----------------------------------------------------------------------------------------------------------------
def _identity_readout(K, mode: str = "ls", C: int = 4) -> NodeVectorReadout:
    ro = NodeVectorReadout(C, 1, K.geo_dim(1), mode).to(device=K.pos.device, dtype=K.star[0].dtype)
    with torch.no_grad():
        ro.w.weight.zero_()
        ro.w.weight[0, 0] = 1.0
    return ro


@pytest.mark.parametrize("name", ["grid", "delaunay", "tets", "quads"])
def test_ls_readout_exact_for_constant_fields(name, device):
    """w_e = l_e t_e . v (exact line integrals of a constant field) -> v at every vertex (up to the damping bias)."""
    K = make_complex(name, device).to(dtype=F64)
    ctx = make_context(K, scaling="dec", B=2, dtype=F64)
    ro = _identity_readout(K)
    D = K.pos.shape[1]
    v = torch.randn(2, D, dtype=F64, generator=torch.Generator().manual_seed(0)).to(device)   # one field per sample
    x1 = torch.zeros(K.n[1], 2, 4, dtype=F64, device=device)
    x1[..., 0] = K.edge_vectors().to(F64) @ v.T
    out = ro([None, x1] + [None] * (K.dim - 1), ctx)                                # (n0, 2, D)
    assert rel(out, v.unsqueeze(0).expand_as(out)) < 1e-4          # O((mu/lambda_min)^2) damping bias at hull slivers


def test_ls_readout_tangent_on_surfaces(device):
    """On a curved surface the reconstruction lives in the vertex tangent plane (well conditioned), and for the
    line integrals of a tangent rotation field it recovers that field to discretisation accuracy."""
    K = make_complex("sphere", device, subdiv=3).to(dtype=F64)
    ctx = make_context(K, scaling="dec", B=1, dtype=F64)
    ro = _identity_readout(K)
    omega = torch.tensor([0.3, -0.5, 0.8], dtype=F64, device=device)
    e = K.cells[1]
    mid = 0.5 * (K.pos[e[:, 0]] + K.pos[e[:, 1]]).to(F64)
    x1 = torch.zeros(K.n[1], 1, 4, dtype=F64, device=device)
    x1[:, 0, 0] = (torch.linalg.cross(omega.expand_as(mid), mid) * K.edge_vectors().to(F64)).sum(-1)
    out = ro([None, x1, None], ctx)[:, 0].detach()                                  # (n0, 3)
    normal = K.pos.to(F64) / K.pos.to(F64).norm(dim=-1, keepdim=True)
    assert float((out * normal).sum(-1).abs().max()) < 0.05 * float(out.norm(dim=-1).max())
    target = torch.linalg.cross(omega.expand_as(K.pos.to(F64)), K.pos.to(F64))
    assert rel(out, target) < 0.05
    E = vector_geometry(K, F64, K.dim)["E"]
    assert E is not None and E.shape == (K.n[0], 3, 2)
    assert rel(E.transpose(1, 2) @ E, torch.eye(2, dtype=F64, device=device).expand(K.n[0], 2, 2)) < 1e-12


@pytest.mark.parametrize("name,kw", [("grid", dict(nx=4, ny=4)), ("sphere", dict(subdiv=1)), ("tets", dict(n=14))])
@pytest.mark.parametrize("mode", ["ls", "direct"])
def test_vector_readout_gradcheck(name, kw, mode):
    K = make_complex(name, "cpu", **kw).to(dtype=F64)
    ctx = make_context(K, scaling="dec", B=2, dtype=F64)
    ro = NodeVectorReadout(3, 2, K.geo_dim(1), mode).to(F64)
    randomize_(ro, scale=0.3, seed=1)
    params = {n: p.detach().clone().requires_grad_() for n, p in ro.named_parameters()}
    names = list(params)
    x1 = torch.randn(K.n[1], 2, 3, dtype=F64, generator=torch.Generator().manual_seed(2)).requires_grad_()
    rest = [None] * (K.dim - 1)

    def f(x1, *ps):
        return torch.func.functional_call(ro, dict(zip(names, ps)), ([None, x1] + rest, ctx))

    assert gradcheck(f, (x1, *params.values()), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_vector_geometry_cache_is_per_complex():
    K1 = make_complex("sphere", subdiv=1)
    K2 = make_complex("sphere", subdiv=2)
    g1 = vector_geometry(K1, torch.float32, 2)
    g2 = vector_geometry(K2, torch.float32, 2)
    assert g1["ev"].shape[0] == K1.n[1] and g2["ev"].shape[0] == K2.n[1]
    assert vector_geometry(K1, torch.float32, 2) is g1


def test_ls_solve_rank_deficient_and_well_posed(device):
    """Damped least squares: min-norm answer (no noise amplification) for collinear edge stars, and negligible
    bias for well-posed stars."""
    from rhmp.readout import ls_solve
    g = torch.Generator().manual_seed(0)
    th = torch.rand(64, generator=g, dtype=F64) * math.pi
    u = torch.stack([th.cos(), th.sin()], -1)                                     # (64, 2) edge direction
    M = 3.0 * u[:, :, None] * u[:, None, :]                                       # rank 1 (collinear star)
    v = torch.randn(64, 2, generator=g, dtype=F64)
    noise = 1e-7 * torch.stack([-th.sin(), th.cos()], -1)                         # rounding in the null direction
    b = (M @ v[:, :, None]).squeeze(-1) + noise
    out = ls_solve(M.to(device).float(), b.to(device).float().unsqueeze(1)).squeeze(1).double().cpu()
    proj = (v * u).sum(-1, keepdim=True) * u                                      # min-norm solution
    assert float((out - proj).norm(dim=-1).max()) < 1e-5
    A = torch.randn(64, 2, 2, generator=g, dtype=F64)
    M2 = A @ A.transpose(1, 2) + 0.2 * torch.eye(2, dtype=F64)                  # well posed
    b2 = (M2 @ v[:, :, None]).squeeze(-1)
    out2 = ls_solve(M2.to(device), b2.to(device).unsqueeze(1)).squeeze(1).cpu()
    assert rel(out2, v) < 1e-5                                                   # O((mu / lambda_min)^2) bias


@pytest.mark.parametrize("shape", [(40000, 16, 16), (70000, 64, 16), (50000, 5, 32), (33000, 32, 1)])
def test_row_linear_matches_linear(shape, device):
    """row_linear (chunked split-K weight gradient on CUDA for tall inputs) == F.linear, values and gradients."""
    from rhmp.layers import row_linear
    N, i, o = shape
    g = torch.Generator().manual_seed(0)
    x = torch.randn(N // 8, 8, i, generator=g).to(device).requires_grad_()
    W = torch.randn(o, i, generator=g).to(device).requires_grad_()
    b = torch.randn(o, generator=g).to(device).requires_grad_()
    w = torch.randn(N // 8, 8, o, generator=g).to(device)
    y = row_linear(x, W, b)
    gx, gW, gb = torch.autograd.grad((y * w).sum(), (x, W, b))
    y2 = torch.nn.functional.linear(x, W, b)
    gx2, gW2, gb2 = torch.autograd.grad((y2 * w).sum(), (x, W, b))
    assert rel(y, y2) < 1e-6
    for a, r in ((gx, gx2), (gW, gW2), (gb, gb2)):
        assert rel(a, r) < 1e-5


@pytest.mark.parametrize("e_shape", ["bias", "cell", "sample"])
def test_mlp2_gradcheck(e_shape, device):
    """The memory-lean fused MLP (hidden layer recomputed in the backward) has exact gradients."""
    from rhmp.layers import mlp2
    g = torch.Generator().manual_seed(0)
    n, B, i, h, o = 7, 3, 4, 5, 2
    shapes = {"bias": (h,), "cell": (n, 1, h), "sample": (n, B, h)}
    args = [torch.randn(n, B, i, generator=g, dtype=F64), torch.randn(h, i, generator=g, dtype=F64),
            torch.randn(*shapes[e_shape], generator=g, dtype=F64), torch.randn(o, h, generator=g, dtype=F64),
            torch.randn(o, generator=g, dtype=F64)]
    args = [a.to(device).requires_grad_() for a in args]
    assert gradcheck(mlp2, tuple(args), eps=1e-6, atol=1e-7, rtol=1e-6)
    ref = torch.nn.functional.linear(torch.nn.functional.silu(torch.nn.functional.linear(args[0], args[1]) + args[2]),
                                     args[3], args[4])
    assert rel(mlp2(*args), ref) < 1e-14


# ----------------------------------------------------------------------------------------------------------------
# resolvent (implicit) layers, DESIGN §9.2
# ----------------------------------------------------------------------------------------------------------------
def _block_ops(ctx, K, k, g, device, dtype, spread=1.0):
    """Random-metric operands of the up/down blocks of degree k (None where the block does not exist)."""
    B = ctx.B
    up = block_operands(ctx, k, True, (spread * torch.randn(K.n[k + 1], B, generator=g, dtype=dtype)).to(device)) \
        if k < K.dim else None
    dn = block_operands(ctx, k, False, (spread * torch.randn(K.n[k - 1], B, generator=g, dtype=dtype)).to(device)) \
        if k > 0 else None
    return up, dn


@pytest.mark.parametrize("tau", [1.0, 10.0, 100.0])
def test_resolvent_cg_residual_t6_size(tau, device):
    """CG residual <= 1e-3 after 32 iterations on a T6-size mesh (n0 = 1024), for every degree, checked against
    an independently computed residual |x - (I + tau L_up + tau L_dn) y| / |x|."""
    from rhmp.layers import resolvent_apply
    K = make_complex("delaunay", device, n=1024)
    B, C = 3, 8
    ctx = make_context(K, scaling="dec", B=B, dtype=torch.float32)
    g = torch.Generator().manual_seed(0)
    t = torch.tensor(tau, device=device)
    iters = 32 if tau <= 10 else 48                    # at the cap tau = 100 (cond <= 201), 32 iterations give ~1e-3
    for k in range(K.dim + 1):
        up, dn = _block_ops(ctx, K, k, g, device, torch.float32)
        x = torch.randn(K.n[k], B, C, generator=g).to(device)
        y, res = resolvent_apply(ctx, k, x, up, dn, t if up else None, t if dn else None, iters=iters)
        My = y + (t * apply_L(up, y) if up else 0) + (t * apply_L(dn, y) if dn else 0)
        true_res = float(((x - My).norm(dim=(0, 2)) / x.norm(dim=(0, 2))).max())
        assert float(res.max()) <= 1e-3 and true_res <= 1e-3, (k, float(res.max()), true_res)


def test_resolvent_matches_dense_solve(device):
    from rhmp.layers import resolvent_apply
    K = make_complex("sphere", device, subdiv=1).to(dtype=F64)
    B, C = 2, 3
    ctx = make_context(K, scaling="dec", B=B, dtype=F64)
    g = torch.Generator().manual_seed(1)
    tu, td = torch.tensor(3.0, dtype=F64, device=device), torch.tensor(0.7, dtype=F64, device=device)
    for k in range(K.dim + 1):
        up, dn = _block_ops(ctx, K, k, g, device, F64)
        n = K.n[k]
        M = torch.eye(n, dtype=F64, device=device).expand(B, n, n).clone()
        if up is not None:
            M = M + tu * _dense(lambda z: apply_L(up, z), n, B, M)
        if dn is not None:
            M = M + td * _dense(lambda z: apply_L(dn, z), n, B, M)
        x = torch.randn(n, B, C, generator=g, dtype=F64).to(device)
        ref = torch.linalg.solve(M, x.permute(1, 0, 2)).permute(1, 0, 2)
        y, _ = resolvent_apply(ctx, k, x, up, dn, tu if up else None, td if dn else None, iters=200)
        assert rel(y, ref) < 1e-9, (k, rel(y, ref))


@pytest.mark.parametrize("grad", ["implicit", "unrolled"])
def test_resolvent_gradcheck(grad, device):
    """Gradients w.r.t. the input, both block metrics and both time steps (fp64, converged CG)."""
    from rhmp.layers import resolvent_apply
    K = make_complex("grid", device, nx=4, ny=4).to(dtype=F64)
    B, C = 2, 1
    ctx = make_context(K, scaling="dec", B=B, dtype=F64)
    g = torch.Generator().manual_seed(2)

    def leaf(*shape, scale=1.0, shift=0.0):
        return (torch.randn(*shape, generator=g, dtype=F64) * scale + shift).to(device).requires_grad_()

    x, lu, ld = leaf(K.n[1], B, C), leaf(K.n[2], B), leaf(K.n[0], B)
    tu, td = leaf((), scale=0.1, shift=1.5), leaf((), scale=0.1, shift=0.5)

    def f(x, lu, ld, tu, td):
        up, dn = block_operands(ctx, 1, True, lu), block_operands(ctx, 1, False, ld)
        return resolvent_apply(ctx, 1, x, up, dn, tu, td, iters=40, grad=grad)[0]   # converged (cond <= 3)

    assert gradcheck(f, (x, lu, ld, tu, td), eps=1e-6, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("grad", ["implicit", "unrolled"])
def test_resolvent_layer_model(grad, device):
    """A model with a resolvent layer runs, all time steps receive finite gradients, diagnostics report tau and
    CG residuals, and the config round-trips."""
    import json
    from rhmp import RHMP, RHMPConfig
    K = make_complex("delaunay", device)
    cfg = small_cfg(layers=["resolvent", "poly", "resolvent"], resolvent_grad=grad, resolvent_iters=16,
                    readout="cochain:1")
    assert cfg.n_layers == 3 and RHMPConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
    m = build_model(cfg, K)
    y = m(make_inputs(cfg, K), K)
    loss_of(y).backward()
    assert all_finite(m, y)
    taus = [p for name, p in m.named_parameters() if "log_tau" in name]
    assert taus and all(p.grad is not None and torch.isfinite(p.grad) for p in taus[:4])
    d = m.diagnostics
    from rhmp.layers import TAU_MAX
    assert 0.0 < d["layer0.tau_up0"] <= TAU_MAX and d["layer0.cg_res1"] < 1e-3


# ----------------------------------------------------------------------------------------------------------------
# Whitney material-tensor metric, DESIGN §9.1
# ----------------------------------------------------------------------------------------------------------------
def _rand_tensor_params(K, km, B, g, device, dtype, a_scale=2.0):
    n_top, m = K.n[K.dim], K.whitney[km]["t"].shape[1]
    b = torch.exp(0.8 * torch.randn(n_top, B, generator=g, dtype=dtype)).to(device)
    a = (a_scale * torch.rand(n_top, B, m, generator=g, dtype=dtype)).to(device)
    return b, a


@pytest.mark.parametrize("name", ["grid", "sphere", "tets"])
def test_tensor_metric_spd_and_operator_bounds(name, device):
    """H_k(b, a) is SPD, rowsum|H| bounds lambda_max(H), and the tensor up block / cross-up transport keep
    ||L|| <= 1, ||T|| <= 1 for random material tensors."""
    from rhmp import dec
    from rhmp.layers import TensorMetric, apply_L_tensor, apply_T_tensor, tensor_operands
    K = make_complex(name, device).to(dtype=F64)
    B = 2
    ctx = make_context(K, scaling="dec", B=B, dtype=F64)
    g = torch.Generator().manual_seed(0)
    for km in range(1, K.dim):
        b, a = _rand_tensor_params(K, km, B, g, device, F64)
        H = _dense(lambda z: dec.apply_whitney_metric(K, km, b, a, z), K.n[km], B, b)
        assert (H - H.transpose(1, 2)).abs().max() <= 1e-12 * H.abs().max()
        ev = torch.linalg.eigvalsh(H)
        assert float(ev.min()) > 0
        rs = dec.whitney_rowsum_abs(K, km, b, a)                                   # (n_km, B)
        assert bool((ev.max(dim=1).values <= rs.max(0).values * (1 + 1e-12)).all())
        op = tensor_operands(ctx, km - 1, TensorMetric(ctx, km, b=b, a=a))
        L = _dense(lambda z: apply_L_tensor(ctx, op, z), K.n[km - 1], B, b)
        evL = torch.linalg.eigvalsh(L)
        assert float(evL.max()) <= 1 + 1e-10 and float(evL.min()) >= -1e-10
        T = _dense(lambda y: apply_T_tensor(ctx, op, y), K.n[km], B, b)
        assert float(torch.linalg.matrix_norm(T, ord=2).max()) <= 1 + 1e-10


def test_tensor_metric_at_init_is_p1_stiffness(device):
    """b = 1, a = 0: d_0^T H_1 d_0 is the P1 stiffness = cotan Laplacian (no clamp active on a jittered grid)."""
    from rhmp import dec
    K = make_complex("grid", device, nx=6, ny=6, jitter=0.1).to(dtype=F64)
    n2, m = K.n[2], K.whitney[1]["t"].shape[1]
    b = torch.ones(n2, 1, dtype=F64, device=device)
    a = torch.zeros(n2, 1, m, dtype=F64, device=device)
    u = torch.randn(K.n[0], 1, 3, dtype=F64, generator=torch.Generator().manual_seed(0)).to(device)
    lhs = K.apply_dT(0, dec.apply_whitney_metric(K, 1, b, a, K.apply_d(0, u)))
    w = K.geo[1][:, 2].to(F64)                                                     # raw cotan weight column
    rhs = K.apply_dT(0, w.view(-1, 1, 1) * K.apply_d(0, u))
    assert rel(lhs, rhs) < 1e-6


def test_anisotropy_recovery(device):
    """Fitting (b, a) alone (no MLP) by gradient-based CGLS reproduces the action of an anisotropic FEM operator
    d_0^T H_1(sigma*) d_0 generated from a random piecewise-constant material tensor to <= 1e-3 (reaches ~1e-7).

    H_1 is linear in (b, a), so the fit is a consistent linear least-squares problem; CGLS (conjugate gradients on the
    normal equations, gradients by autograd) is the gradient method that converges on it (L-BFGS line searches stall
    near 1e-3 because d_0^T H_1 d_0 only sees H_1 on the image of d_0: large null space, small curvatures).
    """
    from rhmp import dec
    K = make_complex("delaunay", device, n=50, seed=3).to(dtype=F64)
    n2, m = K.n[2], K.whitney[1]["t"].shape[1]
    g = torch.Generator().manual_seed(4)
    b_true = torch.exp(0.5 * torch.randn(n2, 1, generator=g, dtype=F64)).to(device)
    a_true = (3.0 * torch.rand(n2, 1, m, generator=g, dtype=F64)).to(device)
    U = torch.randn(K.n[0], 1, 24, generator=g, dtype=F64).to(device)

    def op(b, a):
        return K.apply_dT(0, dec.apply_whitney_metric(K, 1, b, a, K.apply_d(0, U)))

    def grad_adjoint(r):                                                          # J^T r by autograd
        zb = torch.zeros(n2, 1, dtype=F64, device=device, requires_grad=True)
        za = torch.zeros(n2, 1, m, dtype=F64, device=device, requires_grad=True)
        return torch.autograd.grad((op(zb, za) * r).sum(), (zb, za))

    target = op(b_true, a_true)
    b = torch.ones(n2, 1, dtype=F64, device=device)                               # isotropic start: b = 1, a = 0
    a = torch.zeros(n2, 1, m, dtype=F64, device=device)
    with torch.no_grad():
        r = target - op(b, a)
        err0 = rel(op(b, a), target)
    sb, sa = grad_adjoint(r)
    pb, pa, gam = sb.clone(), sa.clone(), sb.square().sum() + sa.square().sum()
    for _ in range(600):
        with torch.no_grad():
            q = op(pb, pa)
            alpha = gam / q.square().sum()
            b += alpha * pb
            a += alpha * pa
            r -= alpha * q
        sb, sa = grad_adjoint(r)
        with torch.no_grad():
            gnew = sb.square().sum() + sa.square().sum()
            pb, pa, gam = sb + (gnew / gam) * pb, sa + (gnew / gam) * pa, gnew
    with torch.no_grad():
        err = rel(op(b, a), target)
    assert err0 > 0.1                                                              # the isotropic start is far off
    assert err <= 1e-3, err


# ----------------------------------------------------------------------------------------------------------------
# stage 3: material columns, solve layers, operator identification
# ----------------------------------------------------------------------------------------------------------------
def _fem_problem(K, sigma_e=None, sigma_f=None, B=2, seed=0):
    """Reference FEM data on a triangle complex (fp64): Dirichlet u = 0 on K.boundary[0], lumped mass M = star_0.

    sigma_e: edge coefficients -> stiffness d0^T diag(star1 sigma_e) d0;  sigma_f: face coefficients -> P1 stiffness
    sum_T |T| sigma_T grad(phi_i).grad(phi_j) assembled independently from the positions.
    Returns (f (n0, B), u (n0, B)).
    """
    n0 = K.n[0]
    D0 = K.d[0].to_dense().to(F64)
    if sigma_e is not None:
        Kmat = D0.T @ torch.diag(K.star[1].to(F64) * sigma_e) @ D0
    else:
        P, T = K.pos.to(F64), K.cells[2].long()
        Kmat = torch.zeros(n0, n0, dtype=F64, device=P.device)
        p0, p1, p2 = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
        area = 0.5 * ((p1 - p0)[:, 0] * (p2 - p0)[:, 1] - (p1 - p0)[:, 1] * (p2 - p0)[:, 0])
        for (a, b, c) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):                 # grad(phi_a) = rot(p_c - p_b) / (2A)
            e = P[T[:, c]] - P[T[:, b]]
            g_a = torch.stack([-e[:, 1], e[:, 0]], 1) / (2 * area[:, None])
            for (a2, b2, c2) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
                e2 = P[T[:, c2]] - P[T[:, b2]]
                g_b = torch.stack([-e2[:, 1], e2[:, 0]], 1) / (2 * area[:, None])
                val = area.abs() * sigma_f * (g_a * g_b).sum(1)
                Kmat.index_put_((T[:, a], T[:, a2]), val, accumulate=True)
    M = K.star[0].to(F64)
    free = ~K.boundary[0]
    g = torch.Generator().manual_seed(seed)
    f = torch.randn(n0, B, generator=g, dtype=F64).to(K.pos.device)
    u = torch.zeros_like(f)
    I = free.nonzero().squeeze(1)
    u[I] = torch.linalg.solve(Kmat[I][:, I], M[I, None] * f[I])
    return f, u


def _set_solver_identity(m):
    """Unit lifting / readout / gate weights and lam -> 0: the solver-preset model then IS (Delta_H)^{-1}."""
    with torch.no_grad():
        m.lifting.lin0.weight.fill_(1.0)
        m.readout.lin.weight.fill_(1.0)
        for layer in m.layers:
            for g in layer.gates:
                g.gain.fill_(1.0)
            for p in layer.log_lam.values():
                p.fill_(-40.0)
    return m


def test_solver_mode_reproduces_fem_diag(device):
    """ACCEPTANCE: solver preset + true sigma as metric reference (learn_metric=False) reproduces the absolute FEM
    solution of d0^T diag(star1 sigma) d0 u = M f (Dirichlet) to <= 1e-4."""
    from rhmp import RHMP, RHMPConfig
    K = make_complex("grid", device, nx=9, ny=8, jitter=0.2).to(dtype=F64)
    g = torch.Generator().manual_seed(1)
    sigma = torch.exp(torch.randn(K.n[1], generator=g, dtype=F64)).to(device)          # 1-2 decades
    f, u = _fem_problem(K, sigma_e=sigma, B=3)
    cfg = RHMPConfig.solver_preset(in_dims={0: 1, 1: 1}, even_dims={1: 1}, material_dims={1: 1},
                                   metric_reference={1: 0}, learn_metric=False, C=1, solve_iters=4000,
                                   solve_tol=1e-13)
    m = _set_solver_identity(RHMP(cfg, K.geo_dims).to(device=device, dtype=F64))
    inp = {0: f.unsqueeze(-1), 1: torch.log(sigma).view(-1, 1, 1).expand(-1, 3, 1).contiguous()}
    y = m(inp, K)[..., 0]
    err = rel(y, u)
    assert err <= 1e-4, err
    fields = m.metric_fields(inp, K)                                                     # works in solver mode
    assert rel(fields[0]["log_ratio"][1], inp[1][..., 0]) < 1e-12
    assert m.diagnostics["layer0.solve_res0"] < 1e-8


def test_solver_mode_reproduces_p1_fem_tensor(device):
    """ACCEPTANCE (FEEC): tensor metric with b_f = sigma_f (face material as reference) reproduces the absolute P1
    FEM solution with element-wise sigma, assembled independently, to <= 1e-4."""
    from rhmp import RHMP, RHMPConfig
    K = make_complex("grid", device, nx=9, ny=8, jitter=0.2).to(dtype=F64)
    g = torch.Generator().manual_seed(2)
    sigma_f = torch.exp(1.5 * torch.randn(K.n[2], generator=g, dtype=F64)).to(device)
    f, u = _fem_problem(K, sigma_f=sigma_f, B=2)
    cfg = RHMPConfig.solver_preset(in_dims={0: 1, 2: 1}, even_dims={2: 1}, material_dims={2: 1},
                                   metric_reference={2: 0}, learn_metric=False, metric_type="tensor", C=1,
                                   solve_iters=4000, solve_tol=1e-13)
    m = _set_solver_identity(RHMP(cfg, K.geo_dims).to(device=device, dtype=F64))
    inp = {0: f.unsqueeze(-1), 2: torch.log(sigma_f).view(-1, 1, 1).expand(-1, 2, 1).contiguous()}
    y = m(inp, K)[..., 0]
    err = rel(y, u)
    assert err <= 1e-4, err
    fields = m.metric_fields(inp, K)[0]
    b, a = fields["tensor"][1]
    assert rel(b, sigma_f.view(-1, 1).expand_as(b)) < 1e-12 and float(a.abs().max()) == 0.0
    eye = torch.eye(2, dtype=F64, device=device)
    assert rel(fields["sigma"][1], sigma_f.view(-1, 1, 1, 1) * eye) < 1e-12


def test_trainer_solver_mode_material_aux_pde(tmp_path, device):
    """Trainer: --solver-mode --material --metric-ref --no-learn-metric --aux-pde on FEM data.  The config is the
    solver preset with the material columns; after the solver normalisation (mean-free source/target, raw reference)
    the preset with unit weights reproduces the normalised targets and the PDE residual of the true solution is ~0;
    nonlinear readouts are rejected."""
    import json
    from rhmp.data import TaskData
    from rhmp.train import build_config, parse_args, pde_residual, run
    K = make_complex("grid", device, nx=7, ny=6, jitter=0.2)
    g = torch.Generator().manual_seed(3)
    sigma = torch.exp(torch.randn(K.n[1], generator=g, dtype=F64)).to(device)
    N = 30
    f, u = _fem_problem(K.to(dtype=F64), sigma_e=sigma, B=N, seed=5)
    inputs = {0: f.T.unsqueeze(-1).float(), 1: torch.log(sigma).float().view(1, -1, 1).expand(N, -1, 1).contiguous()}
    target = u.T.unsqueeze(-1).float()

    def make_task():
        return TaskData.from_arrays(K, inputs, target, readout="node_scalar", even_dims={1: 1}, name="FEM",
                                    meta={"pde": dict(degree=0, source="inputs[0][...,0]", bc="dirichlet")})

    task = make_task()
    argv = ["--task", "FEM", "--out", str(tmp_path / "run"), "--epochs", "1", "--device", str(device), "--quiet",
            "--solver-mode", "--C", "1", "--material", "1:1", "--metric-ref", "1:0", "--no-learn-metric",
            "--aux-pde", "1.0", "--batch", "8", "--solve-iters", "400", "--lr", "0", "--wd", "0", "--no-tf32"]
    with _KeepTF32():
        res = run(parse_args(argv), task=task)
    assert res["complete"]
    cfg = json.load(open(tmp_path / "run" / "config.json"))["model"]
    assert cfg["layers"] == ["solve"] and cfg["gate"] == "none" and cfg["lifting"] == "linear"
    assert cfg["readout"] == "cochain:0" and cfg["C"] == 1 and not cfg["cross"]
    assert cfg["material_dims"] == {"1": 1} and cfg["metric_reference"] == {"1": 0}
    hist = json.load(open(tmp_path / "run" / "history.json"))
    assert hist[0]["train_aux_pde"] < 1e-6                                    # the true metric solves the PDE
    assert float(task.y_stats.mean.abs().max()) == 0.0 and float(task.x_stats[1].std[0]) == 1.0
    from rhmp import RHMP
    m = RHMP.from_checkpoint(tmp_path / "run" / "best.pt", map_location=device)
    _set_solver_identity(m)
    with torch.no_grad():
        m.readout.lin.weight.fill_(float(task.x_stats[0].std[0] / task.y_stats.std[0]))
    xb = {k: v.transpose(0, 1).contiguous() for k, v in task.inputs.items()}          # (n_k, N, F)
    yb = task.target.transpose(0, 1).contiguous()
    assert rel(m(xb, K), yb) < 1e-4                                            # exact after the normalisation
    assert float(pde_residual(m, task, xb, yb, K).max()) < 1e-6
    bad = make_task()
    bad.readout = "even:1"
    with pytest.raises(ValueError, match="linear readout"):
        build_config(bad, parse_args(argv))


def _hp_task_name():
    import os
    from rhmp.tasks import resolve_root
    for name in ("HP_k10000", "HP_k100"):                                  # smallest HP set first
        if os.path.exists(os.path.join(resolve_root(None), "v2", f"{name}.pt")):
            return name
    return None


def test_trainer_metric_ref_raw_on_loaded_hp(tmp_path):
    """--metric-ref columns arrive RAW in every mode: on a loaded HP task with --metric-ref 1:0 the frozen/zero-init
    metric is exactly H/star = exp(log sigma_raw) (physical log conductivity, not the loader-normalised column), and
    config.json records metric_reference_raw."""
    import json
    from rhmp import RHMP
    from rhmp.tasks import load_task
    from rhmp.train import make_batch, parse_args, run
    name = _hp_task_name()
    if name is None:
        pytest.skip("no HP data set (datasets/v2/HP_*.pt)")
    task = load_task(name, None, native=True, device="cpu", max_samples=6, fine=False, whitney=False)
    raw = [task.x_stats[1].denormalize(d[1])[:, 0].clone() for d in task.inputs]         # physical log sigma_e
    assert abs(float(task.x_stats[1].std[0]) - 1.0) > 0.1                                 # the loader normalises it
    argv = ["--task", name, "--out", str(tmp_path / "run"), "--epochs", "1", "--max-train-batches", "1", "--lr", "0",
            "--wd", "0", "--device", "cpu", "--quiet", "--metric-ref", "1:0", "--C", "8", "--layers", "2",
            "--bs-meshes", "2", "--max-samples", "6", "--no-fine"]
    run(parse_args(argv), task=task)
    conf = json.load(open(tmp_path / "run" / "config.json"))
    assert conf["metric_reference_raw"] is True and conf["raw_input_columns"] == {"1": [0]}
    st = conf["data"]["x_stats"]["1"]
    assert st["mean"][0] == 0.0 and st["std"][0] == 1.0
    for d, r in zip(task.inputs, raw):                                                    # raw values reach the model
        assert rel(d[1][:, 0], r) < 1e-6
    m = RHMP.from_checkpoint(tmp_path / "run" / "best.pt", map_location="cpu")
    idx = torch.arange(2)
    K, xb, _ = make_batch(task, idx, "cpu")
    for layer in m.metric_fields(xb, K):
        assert rel(layer["log_ratio"][1][:, 0], torch.cat(raw[:2])) < 1e-6                # H/star = exp(log sigma_raw)
        assert float(layer["phi"][1].abs().max()) < 1e-5                                  # zero-init heads (fp32)
    conf_frozen = argv[:]                                                                  # the frozen control too
    conf_frozen[conf_frozen.index("--out") + 1] = str(tmp_path / "frozen")
    task2 = load_task(name, None, native=True, device="cpu", max_samples=6, fine=False, whitney=False)
    run(parse_args(conf_frozen + ["--no-learn-metric"]), task=task2)
    mf = RHMP.from_checkpoint(tmp_path / "frozen" / "best.pt", map_location="cpu")
    K2, xb2, _ = make_batch(task2, idx, "cpu")
    for layer in mf.metric_fields(xb2, K2):
        assert rel(layer["log_ratio"][1][:, 0], torch.cat(raw[:2])) < 1e-6


def _hp_like_complex(n_tot, seed, device):
    """HP-style mesh: random Delaunay mesh of the unit square with jittered boundary nodes (as datasets/generators/gen_HP.py)."""
    import numpy as np
    from scipy.spatial import Delaunay
    rng = np.random.default_rng(seed)
    m = max(4, int(round(np.sqrt(n_tot))))
    sides = []
    for side in range(4):
        t = (np.arange(1, m) + rng.uniform(-0.25, 0.25, m - 1)) / m
        z, o = np.zeros_like(t), np.ones_like(t)
        sides.append([np.stack([t, z], 1), np.stack([o, t], 1), np.stack([t, o], 1), np.stack([z, t], 1)][side])
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    pts = np.concatenate([corners] + sides + [rng.uniform(0.3 / m, 1 - 0.3 / m, (n_tot - 4 * m, 2))], 0)
    faces = Delaunay(pts).simplices.astype(np.int64)
    return CochainComplex.from_triangles(torch.as_tensor(pts), torch.as_tensor(faces), device=device)


def _hp_like_material(K, seed, contrast=100.0):
    """Smooth random log-conductivity on the edges with max/min = contrast (random Fourier features)."""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(16, 2, generator=g, dtype=F64) * (2 * math.pi / 0.2)
    ph = torch.rand(16, generator=g, dtype=F64) * (2 * math.pi)
    e, P = K.cells[1].long().cpu(), K.pos.double().cpu()
    f = torch.cos((0.5 * (P[e[:, 0]] + P[e[:, 1]])) @ W.T + ph).sum(1)
    return (0.5 * math.log(contrast) * f / f.abs().max()).to(K.pos.device)


def _solver_model(K, precond, iters, tol, dtype, **kw):
    from rhmp import RHMP, RHMPConfig
    cfg = RHMPConfig.solver_preset(in_dims={0: 1, 1: 1}, even_dims={1: 1}, material_dims={1: 1}, metric_reference={1: 0},
                                   C=4, solve_precond=precond, solve_iters=iters, solve_tol=tol, **kw)
    torch.manual_seed(0)
    m = RHMP(cfg, K.geo_dims).to(device=K.pos.device, dtype=dtype)
    m.record_diagnostics = True
    return m


def test_twolevel_preconditioner_cuts_solve_residual(device):
    """solve_precond='twolevel' on an HP-size fixture (2 graphs of 1500 and 1900 vertices, contrast 100): the relative
    residual after 64 iterations drops by >= 5x (observed ~1e3), the converged solution and the implicit gradients
    agree with the Jacobi solve, and the preconditioned solve keeps O(C)-equivariance."""
    parts = [_hp_like_complex(1500, 1, device), _hp_like_complex(1900, 2, device)]
    K = CochainComplex.batch(parts).to(dtype=F64)
    ls = torch.cat([_hp_like_material(P, 7 + i) for i, P in enumerate(parts)]).to(F64)
    g = torch.Generator().manual_seed(0)
    inp = {0: torch.randn(K.n[0], 2, 1, generator=g, dtype=F64).to(device),
           1: ls.view(-1, 1, 1).expand(-1, 2, 1).contiguous()}
    res = {}
    for pc in ("none", "twolevel"):
        m = _solver_model(K, pc, 64, 1e-14, F64)
        with torch.no_grad():
            m(inp, K)
        res[pc] = m.diagnostics["layer0.solve_res0"]
        assert m.diagnostics["layer0.solve_it0"] == 64
    assert res["none"] >= 5.0 * res["twolevel"], res
    # same solution and implicit gradients at convergence
    out = {}
    for pc, iters in (("none", 3000), ("twolevel", 400)):
        m = _solver_model(K, pc, iters, 1e-12, F64, learn_metric=True)
        with torch.no_grad():                                           # a non-trivial metric head
            for p_ in m.layers[0].heads.parameters():
                p_.copy_(0.3 * torch.randn(p_.shape, generator=torch.Generator().manual_seed(p_.numel()), dtype=F64))
        y = m(inp, K)
        gr = torch.autograd.grad(loss_of(y), [p_ for p_ in m.parameters() if p_.requires_grad], allow_unused=True)
        out[pc] = (y.detach(), [t for t in gr if t is not None], m.diagnostics["layer0.solve_it0"])
    assert out["twolevel"][2] < out["none"][2] / 1.5                   # iterations to 1e-12 (random metric heads)
    assert rel(out["twolevel"][0], out["none"][0]) < 1e-8
    for a, b in zip(out["twolevel"][1], out["none"][1]):
        assert rel(a, b) < 1e-6
    # O(C): the coarse correction acts identically on every channel (unconverged iterates included)
    m = _solver_model(K, "twolevel", 20, 1e-14, F64)
    x = m.lift(inp, K)
    Q = torch.as_tensor(random_orthogonal(4, seed=3, reflect=True), dtype=F64, device=device)
    a = m.propagate({k: v @ Q for k, v in x.items()}, K, inp)
    b = m.propagate(x, K, inp)
    assert rel(a[0], b[0] @ Q) < 1e-10


def test_twolevel_preconditioner_is_per_sample_and_per_graph(device):
    """Unconverged (64 iterations) two-level solves: a sample's result does not depend on the other samples (shared
    mesh, B=1 vs B=6) nor on the other graphs of a block-diagonal batch.  Only rounding differs (batched sparse
    products / factorisations of other sizes), amplified by the unconverged Krylov iteration on an ill-conditioned
    operator (contrast 100): tolerance 1e-6 in fp64, while any coupling between samples or graphs gives O(1)."""
    parts = [_hp_like_complex(600, 3, device).to(dtype=F64), _hp_like_complex(900, 4, device).to(dtype=F64),
             _hp_like_complex(750, 5, device).to(dtype=F64)]
    lss = [_hp_like_material(P, 11 + i) for i, P in enumerate(parts)]
    g = torch.Generator().manual_seed(1)
    fs = [torch.randn(P.n[0], 6, 1, generator=g, dtype=F64).to(device) for P in parts]
    m = _solver_model(parts[0], "twolevel", 64, 1e-14, F64)
    inp0 = {0: fs[0], 1: lss[0].view(-1, 1, 1).expand(-1, 6, 1).contiguous()}
    y6 = m(inp0, parts[0])
    y1 = m({0: fs[0][:, :1].contiguous(), 1: inp0[1][:, :1].contiguous()}, parts[0])
    assert rel(y6[:, :1], y1) < 1e-6
    Kb = CochainComplex.batch(parts)
    yb = m({0: torch.cat([f[:, :1] for f in fs]), 1: torch.cat(lss).view(-1, 1, 1)}, Kb)
    ptr = Kb.meta["ptr"][0].tolist()
    for i, P in enumerate(parts):
        yi = m({0: fs[i][:, :1].contiguous(), 1: lss[i].view(-1, 1, 1)}, P)
        assert rel(yb[ptr[i]:ptr[i + 1]], yi) < 1e-6, i


def test_trainer_solver_frozen_tensor_faceref_reproduces_real_hp(tmp_path):
    """Real HP samples (P1 FEM with element-wise sigma, lumped mass, u = 0 on the boundary): the trainer's
    --solver-mode with the frozen tensor metric and the face conductivity as reference (--metric-ref 2:0, two-level
    PCG) reproduces the FEM targets to <= 1e-3 relative WITHOUT training (least-squares readout from the first
    training batch, lr 0) and logs the CG iterations / residual per epoch."""
    import json
    import os
    from rhmp import RHMP
    from rhmp.tasks import load_task, resolve_root
    from rhmp.train import make_batch, parse_args, run
    name = next((n for n in ("HP_k100", "HP_k10000")
                 if os.path.exists(os.path.join(resolve_root(None), "v2", f"{n}.pt"))), None)
    if name is None:
        pytest.skip("no HP data set (datasets/v2/HP_*.pt)")
    task = load_task(name, None, native=True, device="cpu", max_samples=24, fine=False, whitney=True)
    argv = ["--task", name, "--out", str(tmp_path / "run"), "--epochs", "1", "--max-train-batches", "1", "--lr", "0",
            "--wd", "0", "--device", "cpu", "--quiet", "--solver-mode", "--solve-precond", "twolevel", "--solve-iters",
            "128", "--no-learn-metric", "--metric-type", "tensor", "--metric-ref", "2:0", "--bs-meshes", "4",
            "--max-samples", "24", "--no-fine"]
    res = run(parse_args(argv), task=task)
    assert res["best_val_R2"] > 0.99999, res["best_val_R2"]
    hist = json.load(open(tmp_path / "run" / "history.json"))
    assert hist[0]["solve"]["layer0.solve_it0"] <= 128 and hist[0]["solve"]["layer0.solve_res0"] <= 1e-6
    m = RHMP.from_checkpoint(tmp_path / "run" / "best.pt", map_location="cpu")
    idx = torch.cat([task.split[1], task.split[2]])
    K, xb, yb = make_batch(task, idx, "cpu")
    with torch.no_grad():
        y = m(xb, K)
    ptr = K.meta["ptr"][0].tolist()
    errs = [float((y[a:b] - yb[a:b]).norm() / yb[a:b].norm()) for a, b in zip(ptr[:-1], ptr[1:])]
    assert max(errs) <= 1e-3, errs


class _KeepTF32:
    """Restore the global TF32 switches after ``rhmp.train.run`` (which sets them for the whole process)."""

    def __enter__(self):
        self.flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        return self

    def __exit__(self, *exc):
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = self.flags
        return False


def _solver_trainer_run(tmp_path, task, extra):
    """Trainer run (lr 0, one step) of a solver-mode model; returns (model, K, prediction, target) on all samples."""
    import os
    from rhmp import RHMP
    from rhmp.train import model_output, parse_args, run
    argv = ["--task", task.name, "--out", str(tmp_path / "run"), "--epochs", "1", "--max-train-batches", "1", "--lr",
            "0", "--wd", "0", "--device", str(task.K.pos.device), "--quiet", "--solver-mode", "--C", "2",
            "--batch", "8", "--no-tf32"] + extra
    with _KeepTF32():
        run(parse_args(argv), task=task)
    m = RHMP.from_checkpoint(os.path.join(tmp_path, "run", "best.pt"), map_location=task.K.pos.device)
    xb = {k: v.transpose(0, 1).contiguous() for k, v in task.inputs.items()}           # (n_k, N, F)
    with torch.no_grad():
        y = model_output(task, m, xb, task.K)
    return m, y, task.target.transpose(0, 1)


def test_solver_mode_grad_readout_reproduces_potential_differences(tmp_path, device):
    """--solver-mode with the grad readout (linear head phi = Lin(x_0)) and a frozen metric (edge reference): the
    least-squares readout init reproduces the FEM potential differences E = d0 u (Dirichlet Poisson) to <= 1e-3
    without training; the MLP head is kept outside solver mode."""
    from rhmp import RHMPConfig
    from rhmp.data import TaskData
    from rhmp.readout import GradReadout
    K = make_complex("grid", device, nx=8, ny=7, jitter=0.2)
    g = torch.Generator().manual_seed(4)
    sigma = torch.exp(torch.randn(K.n[1], generator=g, dtype=F64)).to(device)
    N = 20
    f, u = _fem_problem(K.to(dtype=F64), sigma_e=sigma, B=N, seed=6)
    E = K.to(dtype=F64).apply_d(0, u.contiguous())                                        # (n1, N)
    task = TaskData.from_arrays(K, {0: f.T.unsqueeze(-1).float(),
                                    1: torch.log(sigma).float().view(1, -1, 1).expand(N, -1, 1).contiguous()},
                                E.T.unsqueeze(-1).float(), readout="grad", even_dims={1: 1}, name="FEMgrad")
    m, y, t = _solver_trainer_run(tmp_path, task, ["--no-learn-metric", "--metric-ref", "1:0", "--material", "1:1",
                                                   "--solve-iters", "400"])
    assert isinstance(m.readout, GradReadout) and m.readout.linear and m.cfg.readout == "grad"
    assert rel(y, t) <= 1e-3, rel(y, t)
    assert not GradReadout(8, 1).linear and not RHMPConfig(in_dims={0: 1}, readout="grad").readout_head == "linear"


def test_solver_mode_grad_with_output_map_neumann_poisson(tmp_path, device):
    """T5g-like: charge -> potential of the pure Neumann Poisson problem ((L + 1e-6 I) phi = A rho, area mean
    removed, cotan L, lumped areas A) -> E = -d0 phi -> fixed v1 edge->node average (direct_vector output map).
    --solver-mode --no-learn-metric --solve-bc neumann (+ two-level PCG) reproduces the node vectors to <= 1e-3
    without training (least-squares readout through the composed map)."""
    from rhmp.data import TaskData
    from rhmp.tasks.paper import direct_vector_map
    K = make_complex("grid", device, nx=9, ny=8, jitter=0.2)
    Kd = K.to(dtype=F64)
    D0 = Kd.d[0].to_dense()
    L = D0.T @ torch.diag(Kd.star[1]) @ D0
    A = Kd.star[0]
    N = 20
    g = torch.Generator().manual_seed(8)
    rho = (torch.randn(K.n[0], N, generator=g, dtype=F64) + 0.3).to(device)             # nonzero total charge
    phi = torch.linalg.solve(L + 1e-6 * torch.eye(K.n[0], dtype=F64, device=device), A[:, None] * rho)
    phi = (phi - (A[:, None] * phi).sum(0) / A.sum()).contiguous()
    omap = direct_vector_map(K, "grad")
    target = omap((-(Kd.apply_d(0, phi))).float().unsqueeze(-1).contiguous())             # (n0, N, 2)
    task = TaskData.from_arrays(K, {0: rho.T.unsqueeze(-1).float()}, target.transpose(0, 1).contiguous(),
                                readout="grad", name="T5like", output_map=omap, stats="none")
    m, y, t = _solver_trainer_run(tmp_path, task, ["--no-learn-metric", "--solve-bc", "neumann", "--solve-precond",
                                                   "twolevel", "--solve-iters", "400"])
    assert m.cfg.solve_bc == "neumann" and m.readout.linear
    assert rel(y, t) <= 1e-3, rel(y, t)


def test_mdiv_readout_conserves_mass_in_block_diagonal_batches(tmp_path, device):
    """mdiv:1 = S0^{-1} d0^T b: sum_i star0_i y_i = 0 per graph (1e-6) in block-diagonal batches, per-graph results equal
    the single-graph ones, the output is S0^{-1} times the div:1 output of the same weights, the least-squares
    readout fit supports it, and checkpoints round-trip."""
    from conftest import make_batch
    from rhmp import RHMP, ops
    from rhmp.layers import make_context
    Kb, parts = make_batch(device)                  # fp64: CUDA reductions are not bitwise deterministic and a TF32
    Kb, parts = Kb.to(dtype=F64), [P.to(dtype=F64) for P in parts]     # setting left by earlier tests would blur fp32
    cfg = small_cfg(readout="mdiv:1", out_dim=2)
    m = build_model(cfg, parts[0], dtype=F64)
    ins = [make_inputs(cfg, P, B=1, seed=i, dtype=F64) for i, P in enumerate(parts)]
    inb = {k: torch.cat([x[k] for x in ins], 0) for k in ins[0]}
    y = m(inb, Kb)
    assert y.shape == (Kb.n[0], 1, 2) and m.output_degree == 0
    tot = ops.segment_sum((Kb.star[0].view(-1, 1, 1) * y).reshape(Kb.n[0], -1), Kb.batch[0], Kb.num_graphs)
    ref = ops.segment_sum((Kb.star[0].view(-1, 1, 1) * y).abs().reshape(Kb.n[0], -1), Kb.batch[0], Kb.num_graphs)
    assert float((tot.abs() / ref).max()) < 1e-6
    ptr = Kb.meta["ptr"][0].tolist()
    for i, (P, x) in enumerate(zip(parts, ins)):
        assert rel(y[ptr[i]:ptr[i + 1]], m(x, P)) < 1e-6
    md = build_model(dataclasses.replace(cfg, readout="div:1"), parts[0], randomize=False, dtype=F64)
    md.load_state_dict(m.state_dict())
    S0 = torch.exp(make_context(Kb, scaling="dec", B=1, dtype=y.dtype).log_star[0]).view(-1, 1, 1)
    assert rel(y * S0, md(inb, Kb)) < 1e-6
    m.fit_linear_readout_(inb, Kb, y.detach() * 2.0)                  # linear readout: LS fit supported
    assert rel(m(inb, Kb), 2.0 * y.detach()) < 1e-4
    torch.save(m.to_checkpoint(), tmp_path / "mdiv.pt")
    m2 = RHMP.from_checkpoint(tmp_path / "mdiv.pt", map_location=device).to(F64)
    assert m2.cfg.readout == "mdiv:1" and rel(m2(inb, Kb), m(inb, Kb)) < 1e-6


def test_material_columns_only_reach_the_metric(device):
    """material_dims: the lifting (all hidden features before the first Hodge operator) does not depend on the
    material column; the output does (through the metric heads)."""
    K = make_complex("delaunay", device).to(dtype=F64)
    for lifting in ("mlp", "linear"):
        cfg = small_cfg(in_dims={0: 3, 1: 2, 2: 2}, even_dims={0: 1, 1: 1, 2: 1}, material_dims={0: 1, 1: 1},
                        lifting=lifting, readout="cochain:1")
        m = build_model(cfg, K)
        inp = make_inputs(cfg, K, dtype=F64)
        leaves = {k: v.clone().requires_grad_() for k, v in inp.items()}
        x = m.lift(leaves, K)
        g = torch.autograd.grad(sum(loss_of(v, seed=i) for i, v in x.items()), [leaves[0], leaves[1]],
                                allow_unused=True)
        assert g[1] is not None and float(g[1][..., 1].abs().max()) == 0.0              # edge material column
        assert float(g[1][..., 0].abs().max()) > 0                                        # edge odd column is used
        assert float(g[0][..., 2].abs().max()) == 0.0                                     # vertex material column
        y = m(leaves, K)
        gy = torch.autograd.grad(loss_of(y), [leaves[0], leaves[1]])
        assert float(gy[1][..., 1].abs().max()) > 0 and float(gy[0][..., 2].abs().max()) > 0


def test_operator_residual_identifies_the_metric(device):
    """operator_residual: exactly 0 for the true metric, (c-1)^2 when sigma is scaled by c (linear in H: the global
    magnitude is identified), and it trains the metric heads."""
    from rhmp import RHMP, RHMPConfig
    K = make_complex("grid", device, nx=8, ny=8, jitter=0.2).to(dtype=F64)
    g = torch.Generator().manual_seed(3)
    sigma = torch.exp(torch.randn(K.n[1], generator=g, dtype=F64)).to(device)
    f, u = _fem_problem(K, sigma_e=sigma, B=2)
    cfg = RHMPConfig.solver_preset(in_dims={0: 1, 1: 1}, even_dims={1: 1}, material_dims={1: 1},
                                   metric_reference={1: 0}, learn_metric=False, C=1)
    m = RHMP(cfg, K.geo_dims).to(device=device, dtype=F64)
    ls = torch.log(sigma).view(-1, 1, 1).expand(-1, 2, 1)
    for c in (1.0, 2.0, 0.5):
        inp = {0: f.unsqueeze(-1), 1: (ls + math.log(c)).contiguous()}
        r = m.operator_residual(inp, K, u, f)
        assert r.shape == (1, 2) and float((r - (c - 1.0) ** 2).abs().max()) < 1e-10, (c, r)
    learned = RHMP(dataclasses.replace(cfg, learn_metric=True, metric_reference={}), K.geo_dims).to(device, F64)
    randomize_(learned, scale=0.3)
    r = learned.operator_residual({0: f.unsqueeze(-1), 1: ls.contiguous()}, K, u, f).mean()
    r.backward()
    grads = [p.grad for n, p in learned.named_parameters() if ".heads." in n]
    assert float(r.detach()) > 1e-3 and any(gr is not None and float(gr.abs().sum()) > 0 for gr in grads)


def test_unnormalised_resolvent_sees_the_metric_scale(device):
    """The beta-normalised resolvent is invariant to H -> c H (scale gauge); resolvent_normalize=False is not, and it
    equals (I + tau L^2 Delta_H)^{-1} computed densely."""
    from rhmp import RHMP
    K = make_complex("grid", device, nx=6, ny=6, jitter=0.2).to(dtype=F64)
    outs = {}
    for norm in (True, False):
        cfg = small_cfg(in_dims={0: 1, 1: 1}, even_dims={1: 1}, material_dims={1: 1}, metric_reference={1: 0},
                        learn_metric=False, layers=["resolvent"], resolvent_normalize=norm, resolvent_iters=400,
                        readout="cochain:0", lifting="linear", gate="none", cross=False)
        m = RHMP(cfg, K.geo_dims).to(device=device, dtype=F64)
        f = torch.randn(K.n[0], 1, 1, dtype=F64, generator=torch.Generator().manual_seed(4)).to(device)
        ys = [m({0: f, 1: torch.full((K.n[1], 1, 1), math.log(c), dtype=F64, device=device)}, K) for c in (1.0, 3.0)]
        outs[norm] = rel(ys[1], ys[0])
    assert outs[True] < 1e-10 and outs[False] > 1e-3


def test_solve_layer_batch_independence_and_gradients(device):
    from conftest import make_batch
    K = make_complex("delaunay", device)
    cfg = small_cfg(layers=["poly", "solve", "poly"], solve_iters=400, solve_tol=1e-5, readout="cochain:1")
    m = build_model(cfg, K)
    inp = make_inputs(cfg, K, B=6)
    y = m(inp, K)
    loss_of(y).backward()
    assert all_finite(m, y) and all(p.grad is not None for p in m.layers[1].log_lam.values())
    # fp32: reproducibility across batch compositions is bounded by the CG accuracy (per-sample scalars keep the
    # samples independent; exact equivariance at convergence is tested in fp64 by test_OC_equivariance_variants)
    alone = m({k: v[:, :1].contiguous() for k, v in inp.items()}, K)
    assert rel(y[:, :1].detach(), alone) < 1e-4
    Kb, parts = make_batch(device)
    ins = [make_inputs(cfg, P, B=1, seed=i) for i, P in enumerate(parts)]
    yb = m({k: torch.cat([x[k] for x in ins], 0) for k in ins[0]}, Kb)
    ptr = Kb.meta["ptr"][m.output_degree].tolist()
    for i, (P, x) in enumerate(zip(parts, ins)):
        assert rel(yb[ptr[i]:ptr[i + 1]], m(x, P)) < 1e-4


# ----------------------------------------------------------------------------------------------------------------
# full tensor parameterisation (sigma_f = b expm(S_f), signed dyad coordinates)
# ----------------------------------------------------------------------------------------------------------------
def _random_spd_2x2(n, ratio, g, device):
    """Random SPD 2x2 tensors with eigenvalue ratio `ratio` and random (misaligned) principal directions."""
    th = torch.rand(n, generator=g, dtype=F64) * math.pi
    R = torch.stack([torch.stack([th.cos(), -th.sin()], -1), torch.stack([th.sin(), th.cos()], -1)], -2)
    lam = torch.stack([torch.ones(n, dtype=F64), torch.full((n,), float(ratio), dtype=F64)], -1)
    lam = lam * torch.exp(torch.randn(n, 1, generator=g, dtype=F64))
    return (R * lam[:, None, :]) @ R.transpose(1, 2), R, lam


def _p1_stiffness_tensor(K, sig):
    """Independent P1 assembly K_ij = sum_T |T| grad(phi_i)^T sigma_T grad(phi_j) (2-D triangles)."""
    P, T = K.pos.to(F64), K.cells[2].long()
    n0 = K.n[0]
    Km = torch.zeros(n0, n0, dtype=F64, device=P.device)
    p0, p1, p2 = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    area = 0.5 * ((p1 - p0)[:, 0] * (p2 - p0)[:, 1] - (p1 - p0)[:, 1] * (p2 - p0)[:, 0])
    grads = []
    for (a, b, c) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        e = P[T[:, c]] - P[T[:, b]]
        grads.append(torch.stack([-e[:, 1], e[:, 0]], 1) / (2 * area[:, None]))
    for i in range(3):
        for j in range(3):
            val = area.abs() * torch.einsum("fd,fde,fe->f", grads[i], sig, grads[j])
            Km.index_put_((T[:, i], T[:, j]), val, accumulate=True)
    return Km


def test_full_tensor_representability_misaligned(device):
    """(i) Every SPD sigma_f (misaligned, ratio 100) is representable: b expm(sum_j s_j t_j t_j^T) with s = the matrix
    logarithm in dyad coordinates, assembled into the Galerkin blocks, reproduces the independent P1 stiffness."""
    from rhmp.layers import TensorMetric, _expm_sym, dyad_gram_inverse, galerkin_blocks, galerkin_geometry
    K = make_complex("grid", device, nx=7, ny=6, jitter=0.2).to(dtype=F64)
    ctx = make_context(K, scaling="dec", B=1, dtype=F64)
    g = torch.Generator().manual_seed(0)
    sig, R, lam = _random_spd_2x2(K.n[2], 100.0, g, device)
    sig, R, lam = sig.to(device), R.to(device), lam.to(device)
    logsig = (R * torch.log(lam)[:, None, :]) @ R.transpose(1, 2)                       # matrix logarithm
    t = K.whitney[1]["t"].to(F64)
    s = torch.einsum("fij,fj->fi", dyad_gram_inverse(K, 1, F64), torch.einsum("fjd,fde,fje->fj", t, logsig, t))
    S = torch.einsum("fj,fjd,fje->fde", s, t, t)
    assert rel(S, logsig) < 1e-10                                                       # the dyads span Sym(2)
    sig_m = _expm_sym(S).unsqueeze(1)                                                   # b = 1, (n2, 1, 2, 2)
    tm = TensorMetric(ctx, 1, blocks=galerkin_blocks(galerkin_geometry(K, 1, F64), sig_m), sigma=sig_m)
    D0 = K.d[0].to_dense().to(F64)
    H = _dense(tm.apply, K.n[1], 1, s)[0]
    assert rel(D0.T @ H @ D0, _p1_stiffness_tensor(K, sig)) < 1e-10


def test_full_tensor_spd_bounds_and_zero_init(device):
    """(ii) SPD metric, rowsum bound of lambda_max, exact diagonal, ||L|| <= 1 and ||T|| <= 1 for random full tensors
    (edge metrics of triangles, surfaces and tets, face metric of tets); (iii) s = 0 gives the Galerkin star b G0 of
    ``rhmp.dec`` (same slots and orientation; its blocks are stored in fp32)."""
    from rhmp import dec
    from rhmp.layers import (TensorMetric, _expm_sym, apply_L_tensor, apply_T_tensor, galerkin_blocks,
                             galerkin_geometry, tensor_operands)
    for name in ("grid", "sphere", "tets"):
        K = make_complex(name, device).to(dtype=F64)
        ctx = make_context(K, scaling="dec", B=2, dtype=F64)
        g = torch.Generator().manual_seed(1)
        for km in range(1, K.dim):
            t = K.whitney[km]["t"].to(F64)
            n_top, m = t.shape[0], t.shape[1]
            b = torch.exp(0.5 * torch.randn(n_top, 2, generator=g, dtype=F64)).to(device)
            s = (2.0 * torch.tanh(torch.randn(n_top, 2, m, generator=g, dtype=F64))).to(device)
            geom = galerkin_geometry(K, km, F64)

            def metric(s_):
                sig = b[..., None, None] * _expm_sym(torch.einsum("fbj,fjd,fje->fbde", s_, t, t))
                return TensorMetric(ctx, km, blocks=galerkin_blocks(geom, sig), sigma=sig)

            tm = metric(s)
            H = _dense(tm.apply, K.n[km], 2, b)
            assert (H - H.transpose(1, 2)).abs().max() <= 1e-12 * H.abs().max(), (name, km)
            ev = torch.linalg.eigvalsh(H)
            assert float(ev.min()) > 0, (name, km)
            assert bool((ev.max(dim=1).values <= tm.rowsum_abs().max(0).values * (1 + 1e-12)).all()), (name, km)
            assert rel(tm.diagonal(), torch.diagonal(H, dim1=1, dim2=2).T) < 1e-12, (name, km)
            op = tensor_operands(ctx, km - 1, tm)
            L = _dense(lambda z: apply_L_tensor(ctx, op, z), K.n[km - 1], 2, b)
            evL = torch.linalg.eigvalsh(L)
            assert float(evL.max()) <= 1 + 1e-10 and float(evL.min()) >= -1e-10, (name, km)
            T = _dense(lambda y: apply_T_tensor(ctx, op, y), K.n[km], 2, b)
            assert float(torch.linalg.matrix_norm(T, ord=2).max()) <= 1 + 1e-10, (name, km)
            z = torch.randn(K.n[km], 2, 3, generator=g, dtype=F64).to(device)
            err = rel(metric(torch.zeros_like(s)).apply(z), dec.apply_whitney_metric(K, km, b, None, z))
            assert err < 1e-5, (name, km, err)


def test_full_tensor_anisotropy_recovery_misaligned(device):
    """(iv) Misaligned anisotropy (ratio 100, random principal directions): fitting the full-tensor coordinates
    ``(log b, s)`` of ``sigma_f = b expm(sum_j s_j t_j t_j^T)`` from the isotropic start (Levenberg-Marquardt, exact
    Jacobian) reproduces the action of the anisotropic P1 operator to <= 1e-3; the model's Galerkin blocks with the
    fitted coordinates give the same operator.  Most of these tensors lie outside the cone ``b I + sum a t t^T``."""
    from rhmp.layers import TensorMetric, _expm_sym, dyad_gram_inverse, galerkin_blocks, galerkin_geometry
    K = make_complex("delaunay", device, n=50, seed=3).to(dtype=F64)
    n2, m = K.n[2], K.whitney[1]["t"].shape[1]
    t = K.whitney[1]["t"].to(F64)
    g = torch.Generator().manual_seed(4)
    sig_true = _random_spd_2x2(n2, 100.0, g, device)[0].to(device)
    U = torch.randn(K.n[0], 16, generator=g, dtype=F64).to(device)
    target = (_p1_stiffness_tensor(K, sig_true) @ U).reshape(-1)
    unit = torch.zeros(3, 2, 2, dtype=F64, device=device)                             # symmetric unit entries
    unit[0, 0, 0] = unit[1, 0, 1] = unit[1, 1, 0] = unit[2, 1, 1] = 1.0
    cols = []                                                                          # the operator is linear in sigma
    for f in range(n2):
        for e in range(3):
            sg = torch.zeros(n2, 2, 2, dtype=F64, device=device)
            sg[f] = unit[e]
            cols.append((_p1_stiffness_tensor(K, sg) @ U).reshape(-1))
    A = torch.stack(cols, 1).view(-1, n2, 3)

    def sym3(X):
        return torch.stack([X[..., 0, 0], X[..., 0, 1], X[..., 1, 1]], -1)

    def sigma(theta):                                                                  # theta = (log b, s): (n2, 1+m)
        return torch.exp(theta[:, :1, None]) * _expm_sym(torch.einsum("fj,fjd,fje->fde", theta[:, 1:], t, t))

    def resid(theta):
        return torch.einsum("nfe,fe->n", A, sym3(sigma(theta))) - target

    theta = torch.zeros(n2, 1 + m, dtype=F64, device=device)
    r = resid(theta)
    err0 = float(r.norm() / target.norm())
    mu, eye = 1e-2, torch.eye(n2 * (1 + m), dtype=F64, device=device)
    for _ in range(200):
        tang = [torch.zeros_like(theta).index_fill_(1, torch.tensor([j], device=device), 1.0) for j in range(1 + m)]
        dsig = torch.stack([torch.func.jvp(sigma, (theta,), (v,))[1] for v in tang], 1)   # (n2, 1+m, 2, 2)
        J = torch.einsum("nfe,fje->nfj", A, sym3(dsig)).reshape(A.shape[0], -1)
        JTJ, JTr = J.T @ J, J.T @ r
        scale = float(JTJ.diagonal().mean())
        while mu < 1e10:
            step = torch.linalg.solve(JTJ + mu * scale * eye, -JTr).view(n2, 1 + m)
            r_new = resid(theta + step)
            if r_new.norm() < r.norm():
                theta, r, mu = theta + step, r_new, max(mu / 3.0, 1e-12)
                break
            mu *= 4.0
        if float(r.norm() / target.norm()) < 1e-8 or mu >= 1e10:
            break
    err = float(r.norm() / target.norm())
    assert err0 > 0.1 and err <= 1e-3, (err0, err)
    ctx = make_context(K, scaling="dec", B=1, dtype=F64)
    sig_fit = sigma(theta).unsqueeze(1)
    tm = TensorMetric(ctx, 1, blocks=galerkin_blocks(galerkin_geometry(K, 1, F64), sig_fit), sigma=sig_fit)
    y = K.apply_dT(0, tm.apply(K.apply_d(0, U.unsqueeze(1).contiguous())))[:, 0]
    assert rel(y.reshape(-1), target) <= 1e-3
    coords = torch.einsum("fij,fj->fi", dyad_gram_inverse(K, 1, F64), torch.einsum("fjd,fde,fje->fj", t, sig_true, t))
    assert float((coords < 0).any(dim=1).float().mean()) > 0.3                           # cone would fail there
