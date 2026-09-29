"""Sparse, segment and diagnostic operations on cochain tensors.

Layout convention (DESIGN §2): cochain features are ``(n, B, C)`` contiguous tensors.  A sparse
operator ``A`` of shape ``(m, n)`` acts on them through a zero-copy ``(n, B*C)`` view, so one
sparse-dense product serves every sample and channel at once.

All sparse products run in fp32 (or fp64 in tests); bf16/fp16 inputs are cast up for the product
and the result is cast back.  Gradients of ``spmm`` use a *precomputed* transpose when one is given
(``AT``), so no ``.t()``/``.coalesce()`` happens during forward or backward.
"""
from __future__ import annotations

import warnings
from typing import Callable

import torch
from torch import Tensor

__all__ = [
    "spmm",
    "sparse_csr",
    "csr_from_coo",
    "csr_with_values",
    "gather_apply",
    "scatter_apply",
    "gershgorin_bound",
    "beta_unit",
    "broadcast_to_cells",
    "segment_sum",
    "segment_mean",
    "segment_max",
    "power_iteration_norm",
    "to_nbc",
    "to_bnc",
]

_LOW_PRECISION = (torch.bfloat16, torch.float16)
_NORM_EPS = 1e-30  # guards the division in power iteration against an all-zero iterate


# ----------------------------------------------------------------------------------------------
# CSR construction helpers
# ----------------------------------------------------------------------------------------------
def sparse_csr(crow: Tensor, col: Tensor, val: Tensor, shape: tuple[int, int]) -> Tensor:
    """Wrap ``(crow, col, val)`` into a CSR tensor without invariant checks.

    Args:
        crow: ``(m+1,)`` int64 row pointer.
        col: ``(nnz,)`` int64 column indices (sorted within each row).
        val: ``(nnz,)`` values.
        shape: ``(m, n)``.
    Returns:
        sparse CSR tensor ``(m, n)``.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support is in beta.*")
        return torch.sparse_csr_tensor(crow, col, val, size=tuple(int(s) for s in shape))


def csr_from_coo(row: Tensor, col: Tensor, val: Tensor, shape: tuple[int, int]) -> Tensor:
    """Build a CSR matrix from COO triplets (vectorised; entries are sorted by ``(row, col)``).

    Duplicate ``(row, col)`` pairs are *not* summed; callers guarantee uniqueness.

    Args:
        row, col: ``(nnz,)`` int64 indices.
        val: ``(nnz,)`` values.
        shape: ``(m, n)``.
    Returns:
        sparse CSR tensor ``(m, n)`` on the device of ``row``.
    """
    m, n = int(shape[0]), int(shape[1])
    key = row * max(n, 1) + col
    order = torch.argsort(key, stable=True)
    row_s, col_s, val_s = row[order], col[order], val[order]
    counts = torch.bincount(row_s, minlength=m)
    crow = torch.zeros(m + 1, dtype=torch.long, device=row.device)
    crow[1:] = torch.cumsum(counts, 0)
    return sparse_csr(crow, col_s.contiguous(), val_s.contiguous(), (m, n))


def csr_with_values(A: Tensor, val: Tensor) -> Tensor:
    """Same sparsity pattern as CSR ``A`` with new values ``val`` (index tensors are shared).

    Args:
        A: CSR ``(m, n)``.
        val: ``(nnz,)``.
    Returns:
        CSR ``(m, n)``.
    """
    return sparse_csr(A.crow_indices(), A.col_indices(), val, tuple(A.shape))


def _cast_csr(A: Tensor, dtype: torch.dtype) -> Tensor:
    return A if A.dtype == dtype else csr_with_values(A, A.values().to(dtype))


# ----------------------------------------------------------------------------------------------
# spmm
# ----------------------------------------------------------------------------------------------
def _addmm0(A: Tensor, x2: Tensor) -> Tensor:
    # addmm with beta=0 into an uninitialised output: bitwise identical to torch.sparse.mm but skips its
    # zero-fill and extra scaling pass (1.6-2.3x faster at T6 size on CUDA, see bench/RESULTS_ops.md);
    # beta=0 means the output's initial contents are never read (NaN-safe).
    out = torch.empty(A.shape[0], x2.shape[1], dtype=x2.dtype, device=x2.device)
    return torch.addmm(out, A, x2, beta=0.0)


def _mm(A: Tensor, x2: Tensor) -> Tensor:
    """Sparse ``(m, n)`` @ dense ``(n, F)`` -> ``(m, F)`` with autocast disabled (stays fp32/fp64)."""
    if torch.is_autocast_enabled(x2.device.type):
        with torch.autocast(device_type=x2.device.type, enabled=False):
            return _addmm0(A, x2)
    return _addmm0(A, x2)


class _SpMM(torch.autograd.Function):
    """``y = A x`` with ``dx = A^T dy`` computed from a precomputed transpose ``AT``."""

    @staticmethod
    def forward(ctx, A: Tensor, AT: Tensor, x2: Tensor) -> Tensor:  # noqa: D401
        ctx.A, ctx.AT = A, AT
        return _mm(A, x2)

    @staticmethod
    def backward(ctx, g: Tensor):
        if not ctx.needs_input_grad[2]:
            return None, None, None
        g = g.contiguous()
        if torch.is_grad_enabled():  # double backward: stay differentiable
            return None, None, _SpMM.apply(ctx.AT, ctx.A, g)
        return None, None, _mm(ctx.AT, g)


def spmm(A: Tensor, x: Tensor, AT: Tensor | None = None) -> Tensor:
    """Sparse-dense product on the cochain layout.

    ``x`` of shape ``(n, *tail)`` (typically ``(n, B, C)`` or ``(n, B)``) is viewed as
    ``(n, prod(tail))`` without copying, multiplied by ``A`` and viewed back.

    Args:
        A: sparse CSR (or COO) ``(m, n)``, fp32 (fp64 allowed in tests).
        x: dense ``(n, *tail)``; must be contiguous.  bf16/fp16 inputs are computed in fp32 and cast
            back; an fp64 ``x`` upcasts ``A`` on the fly (exact for incidence matrices).
        AT: optional precomputed ``A^T`` (CSR ``(n, m)``).  When given, the backward pass uses it
            (no transposition at run time).  Always pass it in training code.
    Returns:
        ``(m, *tail)`` tensor with the dtype of ``x``.
    Raises:
        ValueError: on shape mismatch or non-contiguous ``x``.
    """
    if x.dim() < 1 or x.shape[0] != A.shape[1]:
        raise ValueError(f"spmm: operator has shape {tuple(A.shape)} but x has shape {tuple(x.shape)}")
    if not x.is_contiguous():
        raise ValueError("spmm: x must be contiguous (layout (n, B, C)); call .contiguous() upstream")
    tail = x.shape[1:]
    x2 = x.view(x.shape[0], -1)
    in_dtype = x2.dtype
    if in_dtype in _LOW_PRECISION:
        x2 = x2.float()
    if A.dtype != x2.dtype:
        A = _cast_csr(A, x2.dtype) if A.layout == torch.sparse_csr else A.to(x2.dtype)
        if AT is not None:
            AT = _cast_csr(AT, x2.dtype) if AT.layout == torch.sparse_csr else AT.to(x2.dtype)
    if AT is not None and x2.requires_grad and torch.is_grad_enabled():
        out = _SpMM.apply(A, AT, x2)
    else:
        out = _mm(A, x2)
    if in_dtype in _LOW_PRECISION:
        out = out.to(in_dtype)
    return out.view(A.shape[0], *tail)


# ----------------------------------------------------------------------------------------------
# Gather / scatter application of incidence matrices with constant row length (ELL view of CSR)
# ----------------------------------------------------------------------------------------------
def gather_apply(col: Tensor, val: Tensor, x: Tensor) -> Tensor:
    """``y = A x`` for an operator whose rows all have ``r`` entries, via ``index_select``.

    Args:
        col: ``(m, r)`` int64 column index of every row entry.
        val: ``(m, r)`` values (same dtype as ``x`` or castable).
        x: ``(n, *tail)``.
    Returns:
        ``(m, *tail)``.
    """
    shape = (-1,) + (1,) * (x.dim() - 1)
    out = x.index_select(0, col[:, 0]) * val[:, 0].to(x.dtype).view(shape)
    for j in range(1, col.shape[1]):
        out = out + x.index_select(0, col[:, j]) * val[:, j].to(x.dtype).view(shape)
    return out


def scatter_apply(col: Tensor, val: Tensor, y: Tensor, n: int) -> Tensor:
    """``x = A^T y`` for the operator of :func:`gather_apply`, via ``index_add``.

    Args:
        col: ``(m, r)`` int64.
        val: ``(m, r)``.
        y: ``(m, *tail)``.
        n: number of columns of ``A``.
    Returns:
        ``(n, *tail)``.
    """
    shape = (-1,) + (1,) * (y.dim() - 1)
    out = y.new_zeros((n,) + tuple(y.shape[1:]))
    for j in range(col.shape[1]):
        out = out.index_add(0, col[:, j], y * val[:, j].to(y.dtype).view(shape))
    return out


# ----------------------------------------------------------------------------------------------
# Gershgorin bound for normalised metric Hodge operators (DESIGN §3.3)
# ----------------------------------------------------------------------------------------------
def gershgorin_bound(
    A_abs: Tensor,
    AT_abs: Tensor,
    h: Tensor,
    s_inv_sqrt: Tensor,
    batch: Tensor | None = None,
    num_graphs: int = 1,
) -> Tensor:
    """Per-sample upper bound on ``lambda_max(S^{-1/2} A^T diag(h) A S^{-1/2})``.

    Row sums of the entrywise absolute operator (Gershgorin):
    ``R_j = s_j^{-1/2} [ |A|^T ( h * (|A| s^{-1/2}) ) ]_j``, bound ``= max_j R_j`` per sample.
    Exact upper bound for any ``h >= 0`` and ``s > 0``; differentiable in ``h`` and ``s``.

    Args:
        A_abs: CSR ``|A|``, ``(m, n)``.  Up-block of degree k: ``|d_k|``; down-block: ``|d_{k-1}^T|``.
        AT_abs: CSR ``|A|^T``, ``(n, m)``.
        h: metric on the m-side cells, ``(m, B)`` or ``(m,)``; nonnegative.
        s_inv_sqrt: ``S^{-1/2}`` on the n-side cells, ``(n,)`` or ``(n, B)``; positive.
        batch: optional ``(n,)`` int64 sample id of each n-side cell (block-diagonal batch).
        num_graphs: number of samples in the block-diagonal batch.
    Returns:
        ``(B,)`` when ``batch is None`` (shared mesh), else ``(num_graphs, B)``.
    """
    if h.dim() == 1:
        h = h[:, None]
    s = s_inv_sqrt[:, None] if s_inv_sqrt.dim() == 1 else s_inv_sqrt
    if h.shape[0] != A_abs.shape[0] or s.shape[0] != A_abs.shape[1]:
        raise ValueError(
            f"gershgorin_bound: A has shape {tuple(A_abs.shape)}, h {tuple(h.shape)}, s {tuple(s.shape)}"
        )
    r = spmm(A_abs, s.contiguous(), AT_abs)  # (m, 1) or (m, B)
    y = spmm(AT_abs, (h * r).contiguous(), A_abs)  # (n, B)
    R = s * y
    if batch is None:
        return R.amax(0)
    return segment_max(R, batch, num_graphs)


def beta_unit(
    A_abs: Tensor,
    AT_abs: Tensor,
    s_inv_sqrt: Tensor | None = None,
    batch: Tensor | None = None,
    num_graphs: int = 1,
) -> Tensor:
    """Gershgorin bound of the unit-metric operator ``S^{-1/2} A^T A S^{-1/2}`` (i.e. ``h = 1``).

    For any SPD metric ``H`` (diagonal or Galerkin):
    ``lambda_max(S^{-1/2} A^T H A S^{-1/2}) <= lambda_max(H) beta_unit <= rowsum_max(|H|) beta_unit``,
    so a tensor (Whitney) metric can be normalised with ``rowsum_max(|H|) * beta_unit`` (DESIGN §9.1).

    Args:
        A_abs: CSR ``|A|`` ``(m, n)`` (may already carry a scaling folded into its values).
        AT_abs: CSR ``|A|^T`` ``(n, m)``.
        s_inv_sqrt: ``S^{-1/2}``: ``(n,)``, ``(n, B)`` or ``None`` (``S = 1``).
        batch, num_graphs: as in :func:`gershgorin_bound`.
    Returns:
        ``(B,)`` (``B = 1`` unless ``s_inv_sqrt`` is ``(n, B)``) or ``(num_graphs, B)``.
    """
    m, n = A_abs.shape
    h = torch.ones(m, 1, dtype=A_abs.dtype, device=A_abs.device)
    s = torch.ones(n, dtype=A_abs.dtype, device=A_abs.device) if s_inv_sqrt is None else s_inv_sqrt
    return gershgorin_bound(A_abs, AT_abs, h, s, batch, num_graphs)


def broadcast_to_cells(bound: Tensor, batch: Tensor | None = None) -> Tensor:
    """Expand a per-sample quantity to per-cell rows for broadcasting against ``(n, B, ...)``.

    Args:
        bound: ``(B,)`` (shared mesh) or ``(num_graphs, B)`` (block-diagonal).
        batch: ``None`` or ``(n,)`` sample ids.
    Returns:
        ``(1, B)`` if ``batch is None`` else ``(n, B)``.
    """
    if batch is None:
        return bound.reshape(1, -1)
    return bound.index_select(0, batch)


# ----------------------------------------------------------------------------------------------
# Segment reductions (pure torch; index need not be sorted)
# ----------------------------------------------------------------------------------------------
def _check_index(x: Tensor, index: Tensor) -> None:
    if index.dim() != 1 or index.shape[0] != x.shape[0]:
        raise ValueError(f"segment op: index {tuple(index.shape)} does not match x {tuple(x.shape)}")


def segment_sum(x: Tensor, index: Tensor, num_segments: int) -> Tensor:
    """Sum rows of ``x`` per segment.

    Args:
        x: ``(n, *tail)``.
        index: ``(n,)`` int64 segment id in ``[0, num_segments)``.
        num_segments: number of segments.
    Returns:
        ``(num_segments, *tail)``.
    """
    _check_index(x, index)
    return x.new_zeros((num_segments,) + tuple(x.shape[1:])).index_add(0, index, x)


def segment_mean(x: Tensor, index: Tensor, num_segments: int) -> Tensor:
    """Mean of rows of ``x`` per segment (empty segments give 0).

    Args:
        x: ``(n, *tail)``; index: ``(n,)``; num_segments: int.
    Returns:
        ``(num_segments, *tail)``.
    """
    s = segment_sum(x, index, num_segments)
    cnt = torch.bincount(index, minlength=num_segments).clamp_min(1).to(s.dtype)
    return s / cnt.view((-1,) + (1,) * (x.dim() - 1))


def segment_max(x: Tensor, index: Tensor, num_segments: int) -> Tensor:
    """Max of rows of ``x`` per segment (empty segments give 0); differentiable.

    Args:
        x: ``(n, *tail)``; index: ``(n,)``; num_segments: int.
    Returns:
        ``(num_segments, *tail)``.
    """
    _check_index(x, index)
    view = (-1,) + (1,) * (x.dim() - 1)
    idx = index.view(view).expand_as(x)
    out = x.new_full((num_segments,) + tuple(x.shape[1:]), float("-inf"))
    out = out.scatter_reduce(0, idx, x, reduce="amax", include_self=False)
    nonempty = (torch.bincount(index, minlength=num_segments) > 0).view(view)
    return torch.where(nonempty, out, torch.zeros_like(out))


# ----------------------------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def power_iteration_norm(
    matvec: Callable[[Tensor], Tensor],
    n: int,
    B: int,
    iters: int = 30,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> Tensor:
    """Per-sample largest-magnitude eigenvalue of a *symmetric* operator (diagnostics/tests).

    Args:
        matvec: maps ``(n, B, 1) -> (n, B, 1)``; column ``b`` must depend only on column ``b``.
        n: operator size.
        B: number of independent samples (columns).
        iters: number of iterations.
        device, dtype: of the iterate.
        seed: seed of the random start vector.
    Returns:
        ``(B,)`` Rayleigh-quotient estimate of ``|lambda|_max`` (a lower bound that converges up).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(n, B, 1, generator=g, dtype=dtype).to(device)
    v = v / v.norm(dim=0, keepdim=True).clamp_min(_NORM_EPS)
    lam = torch.zeros(B, dtype=dtype, device=device)
    for _ in range(iters):
        w = matvec(v)
        lam = (v * w).sum(dim=(0, 2))
        v = w / w.norm(dim=0, keepdim=True).clamp_min(_NORM_EPS)
    return lam.abs()


def to_nbc(x: Tensor) -> Tensor:
    """``(B, n, C) -> (n, B, C)`` contiguous (model layout)."""
    return x.transpose(0, 1).contiguous()


def to_bnc(x: Tensor) -> Tensor:
    """``(n, B, C) -> (B, n, C)`` contiguous (user layout)."""
    return x.transpose(0, 1).contiguous()
