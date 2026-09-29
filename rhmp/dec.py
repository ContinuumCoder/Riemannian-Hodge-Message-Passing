"""Discrete exterior calculus toolkit on cochain complexes (DESIGN §9.1, §9.3).

* Galerkin (Whitney) tensor metrics:
  :func:`whitney_blocks`, :func:`apply_whitney_metric`, :func:`whitney_rowsum_abs`, :func:`assemble_whitney_metric`.
  For a top cell f with material tensor ``sigma_f = b_f I + sum_j a_fj t_j t_j^T`` the degree-k metric block is
  ``M_f = b_f G0_f + sum_j a_fj Gk_fj`` (precomputed PSD blocks ``K.whitney[k]``); ``H_k = sum_f P_f^T M_f P_f``.
* Solvers: :func:`cg_solve` (batched CG on ``(n, B, C)``, per-sample Frobenius inner products -> O(C)-equivariant,
  unrolled autograd) and :func:`cg_solve_implicit` (adjoint-solve gradients).
* Metric Hodge Laplacians and decomposition: :func:`hodge_laplacian`, :func:`hodge_decompose`, :func:`harmonic_basis`.
* Whitney interpolation: :func:`sharp` (1-cochain -> vector field) and :func:`flat` (vector field -> 1-cochain).

Metric arguments ``H`` (Laplacians / decomposition) are indexed by degree (list, tuple or dict); an entry is a
diagonal metric ``(n_j,)`` or ``(n_j, B)``, a callable ``x -> H_j x`` (e.g. a Whitney metric), or ``None``
(-> ``K.star[j]``).  Inverse metrics are needed only on the "down" side and must be diagonal (lumped).
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import torch
from torch import Tensor

from .geometry import _TET_EDGES, _TRI_SLOTS, _simplex_measure, barycentric_gradients
from .ops import segment_sum, spmm

__all__ = [
    "whitney_blocks",
    "apply_whitney_metric",
    "whitney_rowsum_abs",
    "assemble_whitney_metric",
    "cg_solve",
    "cg_solve_implicit",
    "hodge_laplacian",
    "hodge_decompose",
    "harmonic_basis",
    "sharp",
    "flat",
]

_TINY = 1e-300  # guards divisions by exact zeros in rank detection
_CG_RTOL_FLOOR = 1e-12  # CG freezes a sample at ||r|| <= max(tol, floor) ||rhs|| (keeps unrolled backward finite)


# ----------------------------------------------------------------------------------------------
# Whitney (Galerkin) tensor metrics
# ----------------------------------------------------------------------------------------------
def _whitney(K, k: int) -> dict:
    if k not in K.whitney:
        raise ValueError(f"no Whitney blocks of degree {k} on this complex (available: {sorted(K.whitney)}); "
                         "Galerkin metrics need simplicial complexes (triangles: k=1; tets: k=1, 2)")
    return K.whitney[k]


def whitney_blocks(K, k: int, b: Tensor, a: Tensor | None) -> Tensor:
    """Per-top-cell metric blocks ``M_f = b_f G0_f + sum_j a_fj Gk_fj``.

    Computed with elementwise operations in a fixed order, so the block of sample ``b`` is bitwise independent of
    the other samples.

    Args:
        K: complex with ``K.whitney[k]``.
        k: metric degree (1: edges; 2: faces of tets).
        b: ``(n_top, B)`` or ``(n_top,)``, positive.
        a: ``(n_top, B, m)``, ``(n_top, m)`` or ``None``; nonnegative; ``m = K.whitney[k]['t'].shape[1]`` directions.
    Returns:
        ``(n_top, B, m_k, m_k)`` blocks in the canonically oriented basis of ``K.whitney[k]['cells']``.
    """
    W = _whitney(K, k)
    G0, Gk = W["G0"], W["Gk"]
    n_top, m = Gk.shape[0], Gk.shape[1]
    if b.dim() == 1:
        b = b[:, None]
    if b.shape[0] != n_top:
        raise ValueError(f"whitney_blocks: b has {b.shape[0]} rows, expected n_top = {n_top}")
    M = b[:, :, None, None] * G0.to(b.dtype)[:, None]
    if a is not None:
        if a.dim() == 2:
            a = a[:, None, :]
        if a.shape[0] != n_top or a.shape[-1] != m:
            raise ValueError(f"whitney_blocks: a has shape {tuple(a.shape)}, expected (n_top={n_top}, B, m={m})")
        Gk = Gk.to(a.dtype)
        for j in range(m):
            M = M + a[:, :, j, None, None] * Gk[:, None, j]
    return M


def _block_mm(Mt: Tensor, xl: Tensor) -> Tensor:
    """``y[i] = sum_j Mt[i, j, ..., None] * xl[j]`` for slot-major ``xl (mk, n_top, B, C)`` and ``Mt (mk, mk, n_top, B)``;
    fixed summation order, contiguous operands."""
    out = torch.empty_like(xl)
    for i in range(xl.shape[0]):
        torch.mul(Mt[i, 0, :, :, None], xl[0], out=out[i])
        for j in range(1, xl.shape[0]):
            out[i].addcmul_(Mt[i, j, :, :, None], xl[j])
    return out


class _WhitneyApply(torch.autograd.Function):
    """``y = S blockdiag(M) G x`` (G = gather, S = scatter = G^T); saves only ``M`` and ``x`` (the gathered slot values
    are recomputed in backward).  Bitwise per-sample forward; first-order backward."""

    @staticmethod
    def forward(ctx, Mt, x, gather, scatter):  # noqa: D401
        mk, n_top = Mt.shape[0], Mt.shape[2]
        B, C = x.shape[1], x.shape[2]
        xl = spmm(gather, x).view(mk, n_top, B, C)
        y = spmm(scatter, _block_mm(Mt, xl).view(mk * n_top, B, C))
        ctx.save_for_backward(Mt, x)
        ctx.maps = (gather, scatter)
        return y

    @staticmethod
    def backward(ctx, gy):
        Mt, x = ctx.saved_tensors
        gather, scatter = ctx.maps
        mk, n_top = Mt.shape[0], Mt.shape[2]
        B, C = x.shape[1], x.shape[2]
        gl = spmm(gather, gy.contiguous()).view(mk, n_top, B, C)  # scatter^T = gather
        gM = gx = None
        if ctx.needs_input_grad[0]:
            gM = torch.einsum("ifbc,jfbc->ijfb", gl, spmm(gather, x).view(mk, n_top, B, C))
        if ctx.needs_input_grad[1]:
            gx = spmm(scatter, _block_mm(Mt.transpose(0, 1).contiguous(), gl).view(mk * n_top, B, C))
        return gM, gx, None, None


def apply_whitney_metric(K, k: int, b: Tensor, a: Tensor | None, x: Tensor) -> Tensor:
    """``H_k(b, a) x``: gather the ``m_k`` cochain values of every top cell, one ``m_k x m_k`` block product per
    (cell, sample), scatter back.  Differentiable in ``b``, ``a`` and ``x`` (first order); cost ``O(n_top m_k^2 B C)``;
    the forward result of every sample is bitwise independent of the other samples; only ``x`` and the blocks are
    kept for backward.

    Args:
        K: complex with ``K.whitney[k]``; k: metric degree.
        b: ``(n_top, B)`` (or ``(n_top, 1)`` / ``(n_top,)`` shared by all samples); a: ``(n_top, B, m)`` or ``None``.
        x: ``(n_k, B, C)`` contiguous.
    Returns:
        ``(n_k, B, C)``.
    """
    W = _whitney(K, k)
    if x.dim() != 3 or x.shape[0] != K.n[k]:
        raise ValueError(f"apply_whitney_metric: x must be (n_{k}={K.n[k]}, B, C), got {tuple(x.shape)}")
    n_top, mk = W["cells"].shape
    B = x.shape[1]
    M = whitney_blocks(K, k, b, a).to(x.dtype)
    if M.shape[1] != B:
        M = M.expand(n_top, B, mk, mk)
    if not x.is_contiguous():
        raise ValueError("apply_whitney_metric: x must be contiguous")
    return _WhitneyApply.apply(M.permute(2, 3, 0, 1).contiguous(), x, W["gather"], W["scatter"])


def whitney_rowsum_abs(K, k: int, b: Tensor, a: Tensor | None) -> Tensor:
    """Row sums of ``|H_k(b, a)|`` bounded blockwise: ``sum_f sum_j |M_f|_ij`` scattered to the k-cells.

    An upper bound of the exact row sums (triangle inequality), hence valid in Gershgorin bounds; differentiable.

    Args:
        K, k, b, a: as in :func:`whitney_blocks`.
    Returns:
        ``(n_k, B)``.
    """
    W = _whitney(K, k)
    n_top, mk = W["cells"].shape
    r = whitney_blocks(K, k, b, a).abs().sum(-1)  # (n_top, B, mk)
    return spmm(W["scatter"], r.permute(2, 0, 1).reshape(mk * n_top, r.shape[1]), W["gather"])


def assemble_whitney_metric(K, k: int, b: Tensor | None = None, a: Tensor | None = None) -> Tensor:
    """Assembled sparse ``H_k`` for one sample (diagnostics, tests, direct solvers).

    Args:
        K, k: as above.
        b: ``(n_top,)`` (default 1); a: ``(n_top, m)`` (default 0).
    Returns:
        CSR ``(n_k, n_k)`` (dtype of ``b``, float32 by default).
    """
    W = _whitney(K, k)
    cells = W["cells"]
    n_top, mk = cells.shape
    if b is None:
        b = torch.ones(n_top, dtype=W["G0"].dtype, device=cells.device)
    M = whitney_blocks(K, k, b.reshape(n_top, 1), None if a is None else a.reshape(n_top, 1, -1))[:, 0]
    rows = cells[:, :, None].expand(n_top, mk, mk).reshape(-1)
    cols = cells[:, None, :].expand(n_top, mk, mk).reshape(-1)
    H = torch.sparse_coo_tensor(torch.stack([rows, cols]), M.reshape(-1), (K.n[k], K.n[k])).coalesce()
    return H.to_sparse_csr()


# ----------------------------------------------------------------------------------------------
# Conjugate gradients on the (n, B, C) layout
# ----------------------------------------------------------------------------------------------
def _dot(u: Tensor, v: Tensor, batch: Tensor | None, num_graphs: int) -> Tensor:
    """Per-sample Frobenius inner products over cells (and channels): ``(1, B)`` or per graph ``(G, B)``."""
    p = (u * v).flatten(2).sum(-1) if u.dim() > 2 else u * v  # (n, B)
    if batch is None:
        return p.sum(0, keepdim=True)
    return segment_sum(p, batch, num_graphs)


def _sqnorm(u: Tensor, batch: Tensor | None, num_graphs: int) -> Tensor:
    """Per-sample squared Frobenius norms (fused reduction): ``(1, B)`` or ``(G, B)``."""
    if batch is None:
        return torch.linalg.vector_norm(u, dim=(0,) + tuple(range(2, u.dim()))).square()[None]
    p = torch.linalg.vector_norm(u, dim=tuple(range(2, u.dim()))).square() if u.dim() > 2 else u * u
    return segment_sum(p, batch, num_graphs)


def _bcast(c: Tensor, batch: Tensor | None, like: Tensor) -> Tensor:
    cc = c if batch is None else c.index_select(0, batch)
    return cc.view(tuple(cc.shape) + (1,) * (like.dim() - 2))


def cg_solve(
    matvec: Callable[[Tensor], Tensor],
    rhs: Tensor,
    iters: int = 32,
    tol: float = 1e-6,
    x0: Tensor | None = None,
    batch: Tensor | None = None,
    num_graphs: int = 1,
    early_exit: bool = True,
) -> tuple[Tensor, Tensor]:
    """Batched conjugate gradients for symmetric positive (semi)definite systems ``A x = rhs``.

    Every sample (column ``b``; and every graph of a block-diagonal batch when ``batch`` is given) runs its own CG
    with Frobenius inner products over cells and channels, so the result for one sample never depends on the others,
    and ``cg_solve(A, rhs @ R) == cg_solve(A, rhs) @ R`` for orthogonal channel mixings ``R`` (O(C)-equivariance) when
    ``matvec`` acts identically on every channel.  The system is solved for ``rhs / ||rhs||`` per sample (rescaled at
    the end) and a sample is frozen once ``||r|| <= max(tol, 1e-12) ||rhs||``, so every division stays well scaled
    and the unrolled backward is finite in fp32 and fp64.  Differentiable by unrolling.

    Args:
        matvec: linear, symmetric PSD map ``(n, B, ...) -> (n, B, ...)`` (per sample).
        rhs: ``(n, B)`` or ``(n, B, C)``.
        iters: maximum number of iterations.
        tol: relative residual at which a sample is frozen (floored at ``1e-12``).
        x0: optional initial guess (same shape as ``rhs``).
        batch: optional ``(n,)`` graph id of every row (block-diagonal batch); num_graphs: number of graphs.
        early_exit: stop as soon as every sample is frozen (one host sync per iteration); ``False`` always runs
            ``iters`` iterations without host synchronisation (frozen samples simply stop changing).
    Returns:
        ``(x, res)``: solution like ``rhs`` and relative residual norms ``res (T+1, S, B)`` with ``T <= iters`` the
        number of iterations run and ``S = num_graphs`` if ``batch`` is given else 1.
    """
    if rhs.dim() < 2:
        raise ValueError(f"cg_solve: rhs must be (n, B) or (n, B, C), got {tuple(rhs.shape)}")
    bn = _sqnorm(rhs, batch, num_graphs).sqrt()
    nz = bn > 0
    scale = torch.where(nz, bn, torch.ones_like(bn))
    rhs_n = rhs / _bcast(scale, batch, rhs)
    if x0 is None:
        x, r = torch.zeros_like(rhs), rhs_n
    else:
        x = x0 / _bcast(scale, batch, x0)
        r = rhs_n - matvec(x)
    p = r
    rs = _sqnorm(r, batch, num_graphs)
    thr = max(float(tol), _CG_RTOL_FLOOR)
    hist = [rs.sqrt()]
    active = nz & (rs.sqrt() > thr)
    for _ in range(iters):
        if early_exit and not bool(active.any()):
            break
        Ap = matvec(p)
        pAp = _dot(p, Ap, batch, num_graphs)
        ok = active & (pAp > 0)
        alpha = torch.where(ok, rs / torch.where(ok, pAp, torch.ones_like(pAp)), torch.zeros_like(rs))
        x = x + _bcast(alpha, batch, x) * p
        r = r - _bcast(alpha, batch, r) * Ap
        rs_new = _sqnorm(r, batch, num_graphs)
        okb = ok & (rs > 0)
        beta = torch.where(okb, rs_new / torch.where(okb, rs, torch.ones_like(rs)), torch.zeros_like(rs))
        p = r + _bcast(beta, batch, p) * p
        rs = torch.where(ok, rs_new, rs)
        hist.append(rs.sqrt())
        active = ok & (rs.sqrt() > thr)
    return x * _bcast(scale, batch, x), torch.stack(hist)


class _CGImplicit(torch.autograd.Function):
    """x = A(theta)^{-1} rhs with adjoint gradients: d rhs = A^{-1} g, d theta = -<A^{-1} g, dA/dtheta x>."""

    @staticmethod
    def forward(ctx, rhs, cfg, *params):  # noqa: D401
        matvec, iters, tol, batch, G, _, early = cfg
        with torch.no_grad():
            x, res = cg_solve(matvec, rhs, iters, tol, None, batch, G, early)
        ctx.cfg = cfg
        ctx.save_for_backward(x)
        ctx.mark_non_differentiable(res)
        return x, res

    @staticmethod
    def backward(ctx, gx, gres):
        (x,) = ctx.saved_tensors
        matvec, iters, tol, batch, G, params, early = ctx.cfg
        with torch.no_grad():
            lam, _ = cg_solve(matvec, gx.contiguous(), iters, tol, None, batch, G, early)
        grads = [None] * len(params)
        need = [i for i in range(len(params)) if ctx.needs_input_grad[2 + i]]
        if need:
            with torch.enable_grad():
                Ax = matvec(x.detach())
                gp = torch.autograd.grad(Ax, [params[i] for i in need], grad_outputs=-lam, allow_unused=True)
            for i, g in zip(need, gp):
                grads[i] = g
        return (lam if ctx.needs_input_grad[0] else None, None, *grads)


def cg_solve_implicit(
    matvec: Callable[[Tensor], Tensor],
    rhs: Tensor,
    params: Sequence[Tensor] = (),
    iters: int = 32,
    tol: float = 1e-6,
    batch: Tensor | None = None,
    num_graphs: int = 1,
    early_exit: bool = True,
) -> tuple[Tensor, Tensor]:
    """:func:`cg_solve` with implicit (adjoint) gradients instead of unrolling: memory ``O(1)`` in ``iters``.

    Backward solves ``A lam = dL/dx`` with the same CG and returns ``dL/drhs = lam`` and
    ``dL/dtheta = -lam^T (dA/dtheta) x`` for every tensor in ``params``.

    Args:
        matvec: linear, symmetric PSD map (see :func:`cg_solve`), depending on the tensors in ``params``.
        rhs: ``(n, B)`` or ``(n, B, C)``.
        params: tensors used inside ``matvec`` that need gradients (e.g. the metric ``H``).
        iters, tol, batch, num_graphs, early_exit: as in :func:`cg_solve`.
    Returns:
        ``(x, res)`` as in :func:`cg_solve` (``res`` is not differentiable).
    """
    params = tuple(params)
    return _CGImplicit.apply(rhs, (matvec, iters, tol, batch, num_graphs, params, early_exit), *params)


# ----------------------------------------------------------------------------------------------
# Metric Hodge Laplacians, decomposition, harmonic forms
# ----------------------------------------------------------------------------------------------
def _metric(K, H, j: int):
    if H is None:
        return K.star[j]
    if isinstance(H, dict):
        h = H.get(j)
    else:
        h = H[j] if j < len(H) else None
    return K.star[j] if h is None else h


def _mul(h, x: Tensor, inverse: bool = False) -> Tensor:
    """Apply a diagonal metric ``(n,)``/``(n, B)`` (or its inverse), or a callable metric, to ``x (n, B, ...)``."""
    if callable(h):
        if inverse:
            raise ValueError("the inverse of a non-diagonal (Galerkin) metric is not supported here; use a diagonal "
                             "(lumped) metric on this degree")
        return h(x)
    hh = h.to(x.dtype)
    hh = 1.0 / hh if inverse else hh
    return x * hh.view(tuple(hh.shape) + (1,) * (x.dim() - hh.dim()))


def hodge_laplacian(K, k: int, H=None, weak: bool = True) -> Callable[[Tensor], Tensor]:
    """Metric Hodge Laplacian of degree ``k`` as a function on ``(n_k, B, ...)`` cochains.

    ``weak=True``: ``L_k = d_k^T H_{k+1} d_k + H_k d_{k-1} H_{k-1}^{-1} d_{k-1}^T H_k`` (symmetric PSD in the Euclidean
    inner product; usable with :func:`cg_solve`).  ``weak=False``: ``Delta_k = H_k^{-1} L_k`` (self-adjoint in the
    ``H_k`` inner product; ``H_k`` must be diagonal).  ``H_{k+1}`` and ``H_k`` may be callables (Galerkin metrics);
    ``H_{k-1}`` must be diagonal.

    Args:
        K: complex; k: degree; H: metrics by degree (default ``K.star``); weak: see above.
    Returns:
        ``f(x) -> (n_k, B, ...)``.
    """
    if not 0 <= k <= K.dim:
        raise ValueError(f"degree {k} out of range")
    Hk = _metric(K, H, k)

    def lap(x: Tensor) -> Tensor:
        y = torch.zeros_like(x)
        if k < K.dim:
            y = y + K.apply_dT(k, _mul(_metric(K, H, k + 1), K.apply_d(k, x)).contiguous())
        if k > 0:
            z = _mul(_metric(K, H, k - 1), K.apply_dT(k - 1, _mul(Hk, x).contiguous()), inverse=True)
            y = y + _mul(Hk, K.apply_d(k - 1, z.contiguous()))
        return y if weak else _mul(Hk, y, inverse=True)

    return lap


def hodge_decompose(K, k: int, x: Tensor, H=None, iters: int = 1000, tol: float = 1e-10
                    ) -> tuple[Tensor, Tensor, Tensor]:
    """Hodge decomposition ``x = d_{k-1} alpha + H_k^{-1} d_k^T gamma + h`` (orthogonal in the ``H_k`` inner product).

    ``alpha`` solves ``d_{k-1}^T H_k d_{k-1} alpha = d_{k-1}^T H_k x`` and ``gamma`` solves
    ``d_k H_k^{-1} d_k^T gamma = d_k x`` by :func:`cg_solve` (per sample, per graph for block-diagonal batches);
    the harmonic part ``h`` is closed (``d_k h = 0``) and co-closed (``d_{k-1}^T H_k h = 0``).

    Args:
        K: complex; k: degree; x: ``(n_k, B)`` or ``(n_k, B, C)``.
        H: metrics by degree; only the (diagonal) ``H_k`` matters (default ``K.star[k]``).
        iters, tol: CG controls (relative residual of the two normal equations).
    Returns:
        ``(exact, coexact, harmonic)``, each like ``x``.
    """
    Hk = _metric(K, H, k)
    if callable(Hk):
        raise ValueError("hodge_decompose needs a diagonal metric on degree k")
    x = x.contiguous()
    exact = torch.zeros_like(x)
    coexact = torch.zeros_like(x)
    idx = (lambda j: None) if K.batch is None else (lambda j: K.batch[j])
    if k > 0:
        def A_e(v):
            return K.apply_dT(k - 1, _mul(Hk, K.apply_d(k - 1, v)).contiguous())

        alpha, _ = cg_solve(A_e, K.apply_dT(k - 1, _mul(Hk, x).contiguous()), iters, tol, None, idx(k - 1),
                            K.num_graphs)
        exact = K.apply_d(k - 1, alpha.contiguous())
    if k < K.dim:
        def A_c(g):
            return K.apply_d(k, _mul(Hk, K.apply_dT(k, g), inverse=True).contiguous())

        gamma, _ = cg_solve(A_c, K.apply_d(k, x), iters, tol, None, idx(k + 1), K.num_graphs)
        coexact = _mul(Hk, K.apply_dT(k, gamma.contiguous()), inverse=True)
    return exact, coexact, x - exact - coexact


def harmonic_basis(K, k: int, H=None, n_probe: int = 8, iters: int = 2000, tol: float = 1e-12,
                   rank_tol: float = 1e-6, seed: int = 0) -> Tensor:
    """``H_k``-orthonormal basis of the discrete harmonic k-cochains of a single complex.

    Random probes are projected onto the harmonic space by :func:`hodge_decompose` (the infinite-shift limit of
    inverse iteration on ``L_k``), then orthonormalised in the ``H_k`` inner product with rank detection; the
    dimension equals the k-th Betti number as long as it is ``<= n_probe``.

    Args:
        K: single complex; k: degree; H: metrics by degree (diagonal ``H_k``, 1-D).
        n_probe: number of random probes (upper bound of the detectable dimension).
        iters, tol: CG controls; rank_tol: relative singular-value threshold; seed: probe seed.
    Returns:
        ``(n_k, h)`` float64 basis with ``basis^T diag(H_k) basis = I``.
    """
    if K.num_graphs != 1:
        raise ValueError("harmonic_basis expects a single complex")
    hk = _metric(K, H, k)
    if callable(hk) or hk.dim() != 1:
        raise ValueError("harmonic_basis needs a diagonal (n_k,) metric on degree k")
    hk = hk.to(torch.float64)
    g = torch.Generator(device="cpu").manual_seed(seed)
    X = torch.randn(K.n[k], n_probe, generator=g, dtype=torch.float64).to(K.device)
    _, _, Y = hodge_decompose(K, k, X, {k: hk}, iters, tol)
    Gm = Y.T @ (hk[:, None] * Y)
    evals, evecs = torch.linalg.eigh(Gm)
    ref = (X * X * hk[:, None]).sum(0).mean()
    keep = evals > (rank_tol ** 2) * ref
    return Y @ (evecs[:, keep] / evals[keep].clamp_min(_TINY).sqrt())


# ----------------------------------------------------------------------------------------------
# Whitney interpolation (sharp) and de Rham map (flat)
# ----------------------------------------------------------------------------------------------
def _whitney_frame(K):
    """Top cells, barycentric gradients (float64), measures and the local edge pairs of ``K.whitney[1]``."""
    W = _whitney(K, 1)
    T = K.cells[K.dim]
    V = K.pos.to(torch.float64)[T]
    pairs = torch.tensor(_TRI_SLOTS if K.dim == 2 else _TET_EDGES, device=T.device)
    return W, T, barycentric_gradients(V), _simplex_measure(V), pairs


def sharp(K, x1: Tensor, at: str = "vertex") -> Tensor:
    """Whitney interpolation of a 1-cochain into a vector field (exact for constant and rigid-rotation fields).

    Args:
        K: triangle or tet complex; x1: ``(n1, *tail)`` edge cochain (canonical orientation).
        at: ``'vertex'`` (value of the Whitney field of every incident top cell at the vertex, averaged with the
            cells' areas/volumes as weights) or ``'cell'`` (value at the barycentre of every top cell).
    Returns:
        ``(n0, *tail, D)`` or ``(n_top, *tail, D)``.
    """
    W, T, grads, meas, pairs = _whitney_frame(K)
    n_top, d1 = T.shape
    tail = x1.shape[1:]
    F = math.prod(tail)
    xe = (x1.reshape(K.n[1], F)[W["cells"]] * W["signs"].to(x1.dtype)[..., None]).to(torch.float64)  # (n_top, m, F)
    m = pairs.shape[0]
    a, b = pairs[:, 0], pairs[:, 1]
    if at == "cell":
        C = torch.zeros(m, d1, dtype=torch.float64, device=T.device)
        C[torch.arange(m), b] += 1.0 / d1
        C[torch.arange(m), a] -= 1.0 / d1
        out = torch.einsum("jl,fjF,flD->fFD", C, xe, grads)
        return out.reshape((n_top,) + tuple(tail) + (grads.shape[-1],)).to(x1.dtype)
    if at != "vertex":
        raise ValueError(f"sharp: unknown location {at!r}; use 'vertex' or 'cell'")
    C = torch.zeros(d1, m, d1, dtype=torch.float64, device=T.device)
    C[a, torch.arange(m), b] += 1.0  # w_ab(p_a) = grad(lam_b)
    C[b, torch.arange(m), a] -= 1.0  # w_ab(p_b) = -grad(lam_a)
    u = torch.einsum("ijl,fjF,flD->fiFD", C, xe, grads)  # (n_top, d+1, F, D)
    wgt = meas[:, None].expand(n_top, d1)
    num = torch.zeros((K.n[0],) + tuple(u.shape[2:]), dtype=torch.float64, device=T.device)
    num.index_add_(0, T.reshape(-1), (u * wgt[..., None, None]).reshape((-1,) + tuple(u.shape[2:])))
    den = torch.zeros(K.n[0], dtype=torch.float64, device=T.device).index_add_(0, T.reshape(-1), wgt.reshape(-1))
    out = num / den.clamp_min(_TINY)[:, None, None]
    return out.reshape((K.n[0],) + tuple(tail) + (grads.shape[-1],)).to(x1.dtype)


def flat(K, v: Tensor) -> Tensor:
    """De Rham map of a vertex vector field: ``x_e = (v_src + v_dst)/2 . (p_dst - p_src)`` (trapezoidal rule, exact
    for fields that are linear along every edge).

    Args:
        K: complex; v: ``(n0, *tail, D)``.
    Returns:
        ``(n1, *tail)``.
    """
    E = K.cells[1]
    D = v.shape[-1]
    if v.shape[0] != K.n[0] or D != K.pos.shape[1]:
        raise ValueError(f"flat: v must be (n0={K.n[0]}, ..., D={K.pos.shape[1]}), got {tuple(v.shape)}")
    ev = K.edge_vectors().to(v.dtype)
    ev = ev.view((K.n[1],) + (1,) * (v.dim() - 2) + (D,))
    return (0.5 * (v[E[:, 0]] + v[E[:, 1]]) * ev).sum(-1)
