"""Tests for rhmp.ops (DESIGN §5)."""
from __future__ import annotations

import pytest
import torch
from conftest import DEVICES, make_batch, make_complex

from rhmp.ops import (
    broadcast_to_cells,
    gather_apply,
    gershgorin_bound,
    power_iteration_norm,
    scatter_apply,
    segment_max,
    segment_mean,
    segment_sum,
    spmm,
    to_bnc,
    to_nbc,
)


def dense(A: torch.Tensor) -> torch.Tensor:
    return A.to_dense().double()


def gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


# ----------------------------------------------------------------------------------------------
# spmm
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["grid", "tets", "mixed"])
@pytest.mark.parametrize("tail", [(3, 5), (4,), ()])
def test_spmm_matches_dense(name, tail, device):
    K = make_complex(name, device)
    for k in range(K.dim):
        for A, AT in ((K.d[k], K.dT[k]), (K.dT[k], K.d[k]), (K.d_abs[k], K.dT_abs[k])):
            x = torch.randn((A.shape[1],) + tail, generator=gen(k)).to(device)
            ref = (dense(A) @ x.double().reshape(A.shape[1], -1)).reshape((A.shape[0],) + tail)
            for out in (spmm(A, x), spmm(A, x, AT)):
                assert out.shape == (A.shape[0],) + tail and out.dtype == torch.float32
                torch.testing.assert_close(out.double(), ref, rtol=1e-5, atol=1e-5)


def test_spmm_bitwise_matches_torch_sparse_mm(device):
    # spmm uses addmm(beta=0) into an uninitialised buffer; it must equal torch.sparse.mm bit for bit
    K = make_complex("delaunay", device)
    for A in (K.d[0], K.dT[0], K.d[1], K.dT[1], K.dT_abs[1]):
        for w in (1, 7, 64):
            x = torch.randn(A.shape[1], w, device=device)
            assert torch.equal(spmm(A, x), torch.sparse.mm(A, x))


def test_spmm_is_a_zero_copy_view_and_validates(device):
    K = make_complex("grid", device)
    x = torch.randn(K.n[0], 4, 8, device=device)
    y = spmm(K.d[0], x)
    assert y.is_contiguous() and y.shape == (K.n[1], 4, 8)
    with pytest.raises(ValueError, match="contiguous"):
        spmm(K.d[0], torch.randn(4, K.n[0], 8, device=device).transpose(0, 1))
    with pytest.raises(ValueError, match="shape"):
        spmm(K.d[0], torch.randn(K.n[1], 4, device=device))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_spmm_low_precision_computes_in_fp32(dtype, device):
    K = make_complex("sphere", device)
    x = torch.randn(K.n[1], 2, 16, device=device).to(dtype)
    y = spmm(K.d[1], x, K.dT[1])
    assert y.dtype == dtype
    assert torch.equal(y, spmm(K.d[1], x.float()).to(dtype))
    xr = x.clone().requires_grad_(True)
    spmm(K.d[1], xr, K.dT[1]).float().sum().backward()
    assert xr.grad.dtype == dtype
    torch.testing.assert_close(xr.grad.float(), spmm(K.dT[1], torch.ones(K.n[2], 2, 16, device=device)))


def test_spmm_under_autocast_stays_fp32(device):
    K = make_complex("grid", device)
    x = torch.randn(K.n[0], 3, 4, device=device)
    with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
        y = spmm(K.d[0], x, K.dT[0])
    assert y.dtype == torch.float32
    torch.testing.assert_close(y, spmm(K.d[0], x))


def test_spmm_fp64_input_upcasts_operator(device):
    K = make_complex("tets", device)
    x = torch.randn(K.n[2], 2, 3, dtype=torch.float64, device=device)
    y = spmm(K.d[2], x)
    assert y.dtype == torch.float64
    torch.testing.assert_close(y.cpu(), (dense(K.d[2]).cpu() @ x.cpu().view(K.n[2], -1)).view(-1, 2, 3))


@pytest.mark.parametrize("use_at", [True, False])
def test_spmm_gradcheck_fp64(use_at, device):
    K = make_complex("grid", device, nx=4, ny=4).to(dtype=torch.float64)
    for k in range(K.dim):
        x = torch.randn(K.n[k], 2, 3, dtype=torch.float64, device=device, requires_grad=True)
        f = (lambda x, k=k: spmm(K.d[k], x, K.dT[k])) if use_at else (lambda x, k=k: spmm(K.d[k], x))
        assert torch.autograd.gradcheck(f, (x,))
        if use_at:
            assert torch.autograd.gradgradcheck(f, (x,))
        y = torch.randn(K.n[k + 1], 2, 3, dtype=torch.float64, device=device, requires_grad=True)
        assert torch.autograd.gradcheck(lambda y, k=k: K.apply_dT(k, y), (y,))


def test_spmm_backward_uses_given_transpose(device):
    K = make_complex("delaunay", device)
    x = torch.randn(K.n[0], 2, 4, device=device, requires_grad=True)
    g = torch.randn(K.n[1], 2, 4, device=device)
    (spmm(K.d[0], x, K.dT[0]) * g).sum().backward()
    torch.testing.assert_close(x.grad, spmm(K.dT[0], g))
    fake_T = K.dT_abs[0]  # a deliberately different "transpose" must change the gradient
    x2 = x.detach().clone().requires_grad_(True)
    (spmm(K.d[0], x2, fake_T) * g).sum().backward()
    torch.testing.assert_close(x2.grad, spmm(fake_T, g))


@pytest.mark.parametrize("name,k,r", [("grid", 0, 2), ("sphere", 1, 3), ("tets", 2, 4), ("tets", 0, 2)])
def test_gather_scatter_match_csr(name, k, r, device):
    K = make_complex(name, device)
    A = K.d[k]
    col, val = A.col_indices().view(-1, r), A.values().view(-1, r)
    x = torch.randn(K.n[k], 3, 5, device=device, requires_grad=True)
    y = torch.randn(K.n[k + 1], 3, 5, device=device, requires_grad=True)
    torch.testing.assert_close(gather_apply(col, val, x), K.apply_d(k, x), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(scatter_apply(col, val, y, K.n[k]), K.apply_dT(k, y), rtol=1e-6, atol=1e-6)
    g1, = torch.autograd.grad(gather_apply(col, val, x).square().sum(), x)
    g2, = torch.autograd.grad(K.apply_d(k, x).square().sum(), x)
    torch.testing.assert_close(g1, g2, rtol=1e-5, atol=1e-5)


# ----------------------------------------------------------------------------------------------
# Gershgorin bounds
# ----------------------------------------------------------------------------------------------
def _dense_lambda_max(A, h, s_inv_sqrt):
    """Largest eigenvalue of S^{-1/2} A^T diag(h_b) A S^{-1/2} per column b (dense, fp64)."""
    A = dense(A).cpu()
    h = h.double().cpu()
    s = s_inv_sqrt.double().cpu()
    lam = []
    for b in range(h.shape[1]):
        sb = s[:, b] if s.dim() == 2 else s
        M = sb[:, None] * (A.T @ (h[:, b:b + 1] * A)) * sb[None, :]
        lam.append(torch.linalg.eigvalsh(M)[-1])
    return torch.stack(lam)


CASES = [("grid", "up", 0), ("grid", "up", 1), ("grid", "down", 1), ("grid", "down", 2), ("sphere", "up", 1),
         ("sphere", "down", 1), ("quads", "up", 0), ("quads", "down", 2), ("tets", "up", 2), ("tets", "down", 3),
         ("tets", "up", 1), ("sliver", "up", 0), ("sliver", "down", 2)]


@pytest.mark.parametrize("name,block,k", CASES)
@pytest.mark.parametrize("scaling", ["random", "dec", "per_sample"])
def test_gershgorin_bounds_lambda_max(name, block, k, scaling, device):
    K = make_complex(name, device)
    g = gen(100 + k)
    B = 3
    if block == "up":
        A_abs, AT_abs, A, m = K.d_abs[k], K.dT_abs[k], K.d[k], K.n[k + 1]
    else:
        A_abs, AT_abs, A, m = K.dT_abs[k - 1], K.d_abs[k - 1], K.dT[k - 1], K.n[k - 1]
    n = K.n[k]
    h = torch.exp(torch.randn(m, B, generator=g, dtype=torch.float64)).float().to(device)
    if scaling == "random":
        s = torch.exp(0.5 * torch.randn(n, generator=g, dtype=torch.float64)).float().to(device)
    elif scaling == "dec":  # S_up = star_k, S_down = 1/star_k (DESIGN §3.3); H = star of the other degree
        s = K.star[k].rsqrt() if block == "up" else K.star[k].sqrt()
        other = K.star[k + 1] if block == "up" else 1.0 / K.star[k - 1]
        h = h * other[:, None]
    else:
        s = torch.exp(0.5 * torch.randn(n, B, generator=g, dtype=torch.float64)).float().to(device)
    bound = gershgorin_bound(A_abs, AT_abs, h, s)
    assert bound.shape == (B,)
    lam = _dense_lambda_max(A, h, s)
    ratio = bound.double().cpu() / lam
    assert bool((ratio >= 1 - 1e-5).all()), ratio
    assert bool((ratio <= 3.0).all()), ratio


def test_gershgorin_block_diagonal_matches_parts(device):
    Kb, parts = make_batch(device)
    g = gen(0)
    for k, block in ((0, "up"), (1, "up"), (1, "down"), (2, "down")):
        if block == "up":
            m = Kb.n[k + 1]
            args = lambda K: (K.d_abs[k], K.dT_abs[k])  # noqa: E731
        else:
            m = Kb.n[k - 1]
            args = lambda K: (K.dT_abs[k - 1], K.d_abs[k - 1])  # noqa: E731
        h = torch.exp(torch.randn(m, 1, generator=g)).to(device)
        s = torch.exp(0.3 * torch.randn(Kb.n[k], generator=g)).to(device)
        bb = gershgorin_bound(*args(Kb), h, s, batch=Kb.batch[k], num_graphs=Kb.num_graphs)
        assert bb.shape == (3, 1)
        ptr_h = Kb.meta["ptr"][k + 1 if block == "up" else k - 1]
        ptr_s = Kb.meta["ptr"][k]
        for gi, P in enumerate(parts):
            hp = h[ptr_h[gi]:ptr_h[gi + 1]]
            sp = s[ptr_s[gi]:ptr_s[gi + 1]]
            torch.testing.assert_close(bb[gi], gershgorin_bound(*args(P), hp, sp), rtol=1e-6, atol=0)
        cells = broadcast_to_cells(bb, Kb.batch[k])
        assert cells.shape == (Kb.n[k], 1)
    assert broadcast_to_cells(torch.ones(5)).shape == (1, 5)


def test_gershgorin_is_differentiable(device):
    K = make_complex("grid", device, nx=4, ny=4).to(dtype=torch.float64)
    h = torch.rand(K.n[1], 2, dtype=torch.float64, device=device).add(0.5).requires_grad_(True)
    s = torch.rand(K.n[0], dtype=torch.float64, device=device).add(0.5)
    assert torch.autograd.gradcheck(lambda h: gershgorin_bound(K.d_abs[0], K.dT_abs[0], h, s), (h,))


# ----------------------------------------------------------------------------------------------
# segment ops, power iteration, layout helpers
# ----------------------------------------------------------------------------------------------
def test_segment_ops(device):
    g = gen(3)
    n, S = 50, 7
    idx = torch.randint(0, S - 1, (n,), generator=g).to(device)  # segment S-1 stays empty
    x = torch.randn(n, 2, 3, generator=g).to(device)
    ssum, smean, smax = segment_sum(x, idx, S), segment_mean(x, idx, S), segment_max(x, idx, S)
    for s in range(S):
        sel = x[idx == s]
        if sel.shape[0] == 0:
            assert bool((ssum[s] == 0).all() and (smean[s] == 0).all() and (smax[s] == 0).all())
            continue
        torch.testing.assert_close(ssum[s], sel.sum(0))
        torch.testing.assert_close(smean[s], sel.mean(0))
        torch.testing.assert_close(smax[s], sel.amax(0))
    with pytest.raises(ValueError):
        segment_sum(x, idx[:-1], S)
    xd = torch.randn(n, 3, dtype=torch.float64, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: segment_max(x, idx, S), (xd,))
    assert torch.autograd.gradcheck(lambda x: segment_mean(x, idx, S), (xd,))


def test_power_iteration_matches_eigvalsh(device):
    K = make_complex("sphere", device)
    h = torch.exp(torch.randn(K.n[1], 2, generator=gen(1))).to(device)

    def matvec(v):  # (n0, B, 1)
        return K.apply_dT(0, (h[:, :, None] * K.apply_d(0, v)).contiguous())

    est = power_iteration_norm(matvec, K.n[0], 2, iters=300, device=device)
    lam = _dense_lambda_max(K.d[0], h, torch.ones(K.n[0], device=device))
    ratio = est.double().cpu() / lam
    assert bool((ratio <= 1 + 1e-5).all() and (ratio >= 0.98).all()), ratio


def test_layout_helpers(device):
    x = torch.randn(4, 9, 3, device=device)
    y = to_nbc(x)
    assert y.shape == (9, 4, 3) and y.is_contiguous()
    assert torch.equal(to_bnc(y), x) and to_bnc(y).is_contiguous()
