"""Normalised metric Hodge operators and the RHMP message-passing layer (DESIGN §3.3).

Notation (per degree ``k``, all feature tensors ``(n_k, B, C)``, per-cell/per-sample scalars ``(n, B)``):

* Up block of degree ``k`` (exists if ``k < K``): ``A = d_k`` ``(n_{k+1}, n_k)``, metric ``H = H_{k+1}``.
* Down block of degree ``k`` (exists if ``k > 0``): ``A = d_{k-1}^T`` ``(n_{k-1}, n_k)``, metric
  ``H = H^down_{k-1}`` (``= 1 / H_{k-1}`` when metrics are tied).
* Both blocks share one form.  With the diagonal scaling ``s = S^{-1/2}`` on degree ``k``::

      L x      = s ⊙ A^T ( h ⊙ A ( s ⊙ x ) ),        h = H / beta          (normalised operator)
      T y      = s ⊙ A^T ( sqrt(h) ⊙ y )                                   (half operator, L = T T^T)
      beta     = max_j s_j [ |A|^T ( H ⊙ |A| s ) ]_j   per sample (Gershgorin bound of s A^T H A s)

  hence ``||L|| <= 1`` and ``||T|| <= 1`` for every sample and every metric.  The cross-degree transport is
  ``cross_up(x_{k+1}) = T_up x_{k+1}`` and ``cross_down(x_{k-1}) = T_down x_{k-1}``.
* Scalings (``scaling=``): ``'dec'`` uses the reference star in the symmetrised frame ``x = star^{1/2} u``,
  i.e. ``S_up = star_k`` and ``S_down = 1 / star_k``; at initialisation (``H = star``) the blocks are the
  DEC Hodge Laplacian pieces ``star^{-1/2} d^T star d star^{-1/2}`` and ``star^{1/2} d star^{-1} d^T star^{1/2}``
  and the cross terms are the DEC coboundary / codifferential in that frame.  ``'jacobi'`` uses
  ``S = diag(A^T H A)`` per sample; ``'none'`` uses ``S = 1``.  All operators are invariant under a per-sample
  rescaling of ``S`` and of ``H``, so the stars are normalised per sample by their geometric mean (numerical
  hygiene only).

Layer update (all degrees synchronously from the layer input)::

    m_k = sum_p c^up_p L_up^p x_k + sum_p c^dn_p L_dn^p x_k + w_cu T_up x_{k+1} + w_cd T_dn x_{k-1}
    x_k <- x_k + gamma_k * sigmoid(MLP(log1p(rms(m_k)))) * m_k / rms(m_k)      (radial norm gate)

Implementation notes (performance):

* For ``'dec'`` / ``'none'`` the constant scaling is folded into scaled copies of the CSR operators
  (``A diag(s)``, ``diag(s) A^T`` and their absolute values; same index arrays, new values), cached per complex,
  so no elementwise ``s ⊙ x`` pass is ever executed.  ``'jacobi'`` (per-sample ``s``) uses the explicit form.
* The first coboundary ``A (s ⊙ x)`` of every block is computed once per layer and shared by the block and by
  the metric invariants.
* The polynomial filter runs inside a fused autograd function (Horner-like recursion) that only keeps
  ``(A s)(s A^T) ... q`` for ``p >= 2`` for the backward and uses fused ``addcmul``/``vecdot`` passes.
"""
from __future__ import annotations

import math
import weakref
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd.function import once_differentiable
from torch.utils.checkpoint import checkpoint as _checkpoint

from . import ops
from .metric import MetricHead, TensorMetricHead, metric_feature_dim

__all__ = [
    "LayerContext", "BlockGeometry", "make_context", "RadialGate", "RHMPLayer", "BlockOperands", "block_operands",
    "apply_L", "apply_T", "poly_block", "poly_block_reference", "spmm_ad", "SharedCoboundaries", "mean_square",
    "residual_scale", "row_linear", "linear", "mlp2", "resolvent_matvec", "resolvent_apply", "TAU_MAX",
    "TensorOperands", "tensor_operands", "apply_L_tensor", "apply_T_tensor", "block_beta_unit",
    "PhysicalHodge", "physical_hodge", "physical_solve", "dirichlet_free", "dyad_gram_inverse", "TensorMetric",
    "galerkin_geometry", "galerkin_blocks",
]

# Floor added to the per-cell mean square of a message inside the radial gate normaliser (features are O(1)).
RMS_EPS = 1e-6
# |log H| is clamped to this value after adding the (normalised) reference star: keeps exp(.) finite in fp32
# even for absurd ``log_range`` values; never active for the default a = 2.
LOG_H_MAX = 60.0
_SCALINGS = ("dec", "jacobi", "none")
_GATES = ("norm", "relu", "none")


# --------------------------------------------------------------------------------------------------------------
# elementary fused ops
# --------------------------------------------------------------------------------------------------------------
def spmm_ad(A: Tensor, AT: Tensor, x: Tensor) -> Tensor:
    """Autograd-aware ``A @ x`` on ``(n, B, C)`` tensors (backward uses the precomputed transpose ``AT``).

    Args:
        A: CSR ``(m, n)``.
        AT: CSR ``(n, m)``, the transpose of ``A`` (used for the backward; no run-time transposes).
        x: ``(n, B, C)`` (any trailing shape).

    Returns:
        ``(m, B, C)``.
    """
    return ops.spmm(A, x.contiguous(), AT)


def _mul(x: Tensor, s: Tensor | None) -> Tensor:
    """``x * s`` with ``s = None`` meaning the identity."""
    return x if s is None else x * s


class _MeanSquare(torch.autograd.Function):
    """``|x|^2 / C`` over the last dim with a single-pass backward."""

    @staticmethod
    def forward(ctx: Any, x: Tensor) -> Tensor:  # noqa: D102
        ctx.save_for_backward(x)
        return torch.linalg.vector_norm(x, dim=-1).square() / x.shape[-1]   # fused reduction, no temporary

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, g: Tensor) -> Tensor:  # noqa: D102
        (x,) = ctx.saved_tensors
        return x * (g * (2.0 / x.shape[-1])).unsqueeze(-1)


def mean_square(x: Tensor) -> Tensor:
    """Per-cell mean square over channels ``|x|^2 / C``: ``(n, B, C) -> (n, B)``."""
    return _MeanSquare.apply(x)


_msq = mean_square  # short alias used by the lifting and readouts


class _ResidualScale(torch.autograd.Function):
    """``y = x + m ⊙ a`` with ``a`` ``(n, B, 1)``: one fused forward pass, ``vecdot`` reduction in the backward."""

    @staticmethod
    def forward(ctx: Any, x: Tensor, m: Tensor, a: Tensor) -> Tensor:  # noqa: D102
        ctx.save_for_backward(m, a)
        return torch.addcmul(x, m, a)

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, g: Tensor):  # noqa: D102
        m, a = ctx.saved_tensors
        need = ctx.needs_input_grad
        grad_m = g * a if need[1] else None
        grad_a = torch.linalg.vecdot(g, m).unsqueeze(-1) if need[2] else None
        return (g if need[0] else None), grad_m, grad_a


def residual_scale(x: Tensor, m: Tensor, a: Tensor) -> Tensor:
    """``x + m ⊙ a`` for ``x, m`` ``(n, B, C)`` and a per-cell scale ``a`` ``(n, B, 1)``."""
    return _ResidualScale.apply(x, m, a)


# Row-count threshold above which weight gradients of per-cell linear maps use a chunked (split-K) batched GEMM.
_CHUNK_MIN_ROWS = 32768
_CHUNK_ROWS = 1024


def _weight_grad(g: Tensor, x: Tensor) -> Tensor:
    """``g^T x`` for tall ``g`` ``(N, out)`` and ``x`` ``(N, in)``.

    cuBLAS picks a slow "large-K" kernel for small ``out x in`` with ``N ~ 1e5`` rows (up to 7x slower than
    necessary); splitting ``N`` into chunks and reducing a batched GEMM avoids it.
    """
    N, o = g.shape
    i = x.shape[1]
    if N < _CHUNK_MIN_ROWS or min(o, i) < 8 or not g.is_cuda:
        return g.t() @ x
    nc = N // _CHUNK_ROWS
    main = nc * _CHUNK_ROWS
    out = torch.bmm(g[:main].view(nc, _CHUNK_ROWS, o).transpose(1, 2), x[:main].view(nc, _CHUNK_ROWS, i)).sum(0)
    if main < N:
        out = out + g[main:].t() @ x[main:]
    return out


class _RowLinearFn(torch.autograd.Function):
    """``y = x W^T (+ b)`` over the rows of ``x`` with a chunked weight gradient (see :func:`_weight_grad`)."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx: Any, x: Tensor, W: Tensor, b: Tensor | None) -> Tensor:  # noqa: D102
        ctx.save_for_backward(x, W)
        ctx.has_bias = b is not None
        return F.linear(x, W, b)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    @once_differentiable
    def backward(ctx: Any, g: Tensor):  # noqa: D102
        x, W = ctx.saved_tensors
        need = ctx.needs_input_grad
        g2 = g.reshape(-1, g.shape[-1])
        gx = gW = gb = None
        if need[0]:
            gx = (g2 @ W.to(g2.dtype)).view(*g.shape[:-1], W.shape[1]).to(x.dtype)
        if need[1]:
            x2 = x.reshape(-1, x.shape[-1])
            gW = _weight_grad(g2, x2.to(g2.dtype)).to(W.dtype)
        if ctx.has_bias and need[2]:
            gb = g2.sum(0).to(W.dtype)
        return gx, gW, gb


# Recompute the hidden layer of per-cell MLPs in the backward (memory) instead of storing it (speed).
_MLP2_LEAN = True


class _MLP2Fn(torch.autograd.Function):
    """``y = W2 silu(x W1^T + e) + b2`` saving only ``x`` and ``e``; the hidden layer is recomputed in the backward.

    ``e`` is an additive pre-activation broadcastable to ``(..., hidden)`` (bias, per-cell geometry part ``(n, 1, h)``
    or a per-sample part ``(n, B, h)``).  This removes the two hidden-sized activations autograd would keep.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx: Any, x: Tensor, W1: Tensor, e: Tensor, W2: Tensor, b2: Tensor | None) -> Tensor:  # noqa: D102
        ctx.save_for_backward(x, W1, e, W2)
        ctx.has_b2 = b2 is not None
        return F.linear(F.silu(F.linear(x, W1) + e), W2, b2)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    @once_differentiable
    def backward(ctx: Any, g: Tensor):  # noqa: D102
        x, W1, e, W2 = ctx.saved_tensors
        need = ctx.needs_input_grad
        z = F.linear(x, W1) + e                                                  # recomputed pre-activation
        sg = torch.sigmoid(z)
        a = z * sg                                                              # silu(z)
        h = z.shape[-1]
        g2 = g.reshape(-1, g.shape[-1])
        gW2 = _weight_grad(g2, a.reshape(-1, h).to(g2.dtype)).to(W2.dtype) if need[3] else None
        gb2 = g2.sum(0).to(W2.dtype) if (ctx.has_b2 and need[4]) else None
        gz = torch.matmul(g, W2.to(g.dtype)) * (sg * (1.0 + z * (1.0 - sg)))    # silu'(z) = s (1 + z (1 - s))
        gx = torch.matmul(gz, W1.to(gz.dtype)).to(x.dtype) if need[0] else None
        gW1 = _weight_grad(gz.reshape(-1, h), x.reshape(-1, x.shape[-1]).to(gz.dtype)).to(W1.dtype) if need[1] else None
        ge = gz.sum_to_size(e.shape).to(e.dtype) if need[2] else None
        return gx, gW1, ge, gW2, gb2


def mlp2(x: Tensor, W1: Tensor, e: Tensor, W2: Tensor, b2: Tensor | None) -> Tensor:
    """Memory-lean two-layer MLP ``W2 silu(x W1^T + e) + b2`` for per-cell features (see :class:`_MLP2Fn`).

    Args:
        x: ``(..., in)``.
        W1: ``(hidden, in)`` (may be a column slice of a larger weight).
        e: additive pre-activation broadcastable to ``(..., hidden)`` (includes the first bias).
        W2: ``(out, hidden)``.
        b2: ``(out,)`` or ``None``.

    Returns:
        ``(..., out)``.
    """
    if (not _MLP2_LEAN or not torch.is_grad_enabled()
            or not any(t is not None and t.requires_grad for t in (x, W1, e, W2, b2))):
        return row_linear(F.silu(row_linear(x, W1) + e), W2, b2)
    return _MLP2Fn.apply(x, W1, e, W2, b2)


def row_linear(x: Tensor, W: Tensor, b: Tensor | None = None) -> Tensor:
    """``F.linear`` for per-cell features ``(n, B, in) -> (n, B, out)`` with a fast weight gradient.

    Args:
        x: ``(..., in)``.
        W: ``(out, in)`` (may be a column slice of a larger weight).
        b: ``(out,)`` or ``None``.
    """
    if not (x.requires_grad or W.requires_grad) or not torch.is_grad_enabled():
        return F.linear(x, W, b)
    return _RowLinearFn.apply(x, W, b)


def linear(mod: nn.Linear, x: Tensor) -> Tensor:
    """Apply an ``nn.Linear`` module through :func:`row_linear`."""
    return row_linear(x, mod.weight, mod.bias)


# --------------------------------------------------------------------------------------------------------------
# per-complex operator cache and per-forward context
# --------------------------------------------------------------------------------------------------------------
@dataclass
class BlockGeometry:
    """Operators of one block with the constant scaling folded in (cached per complex).

    Attributes:
        A: CSR ``(m, n_k)``: ``A diag(s)`` for ``'dec'``, the raw coboundary (or its transpose) otherwise.
        AT: CSR ``(n_k, m)``: transpose of ``A``.
        A_abs: CSR ``|A|`` (scaled like ``A``).
        AT_abs: CSR ``|A|^T``.
        r: ``|A| 1`` ``(m, 1, 1)`` (Gershgorin helper; ``= |A_raw| s``), ``None`` for Jacobi scaling.
        scale: the folded scaling ``s`` ``(n_k, 1, 1)`` (``'dec'``) or ``None``.
        nb: degree of the neighbour cells (``k+1`` for up blocks, ``k-1`` for down blocks).
    """

    A: Tensor
    AT: Tensor
    A_abs: Tensor
    AT_abs: Tensor
    r: Tensor | None
    scale: Tensor | None
    nb: int


_OP_CACHE: "weakref.WeakKeyDictionary[Any, dict]" = weakref.WeakKeyDictionary()


def _csr_scale_cols(A: Tensor, s: Tensor) -> Tensor:
    """``A diag(s)`` for CSR ``A`` ``(m, n)`` and ``s`` ``(n,)`` (same index arrays)."""
    return ops.csr_with_values(A, A.values() * s[A.col_indices()])


def _csr_scale_rows(A: Tensor, s: Tensor) -> Tensor:
    """``diag(s) A`` for CSR ``A`` ``(m, n)`` and ``s`` ``(m,)`` (same index arrays)."""
    crow = A.crow_indices()
    rows = torch.repeat_interleave(torch.arange(A.shape[0], device=crow.device), crow[1:] - crow[:-1])
    return ops.csr_with_values(A, A.values() * s[rows])


@torch.no_grad()
def _complex_operators(K: Any, scaling: str, dtype: torch.dtype) -> dict:
    """Geometry-only tensors of a complex for one scaling: normalised log-stars and all block operators."""
    try:
        store = _OP_CACHE.setdefault(K, {})
    except TypeError:  # not weak-referenceable: no caching
        store = {}
    key = (scaling, dtype, str(K.pos.device))
    if key in store:
        return store[key]
    top = int(K.dim)
    batch = K.batch
    G = int(K.num_graphs) if batch is not None else 1
    log_star, log_gm = [], []
    for k in range(top + 1):
        ls = torch.log(K.star[k].to(dtype))
        if batch is None:
            gm = ls.mean().reshape(1)                                      # (1,) log geometric mean of the star
            ls = ls - gm
        else:
            gm = ops.segment_mean(ls, batch[k], G)                         # (G,)
            ls = ls - gm[batch[k]]
        log_star.append(ls)
        log_gm.append(gm)
    vol = K.star[0].to(dtype)
    vol = vol.sum().reshape(1) if batch is None else ops.segment_sum(vol, batch[0], G)   # total dual measure
    blocks: dict[tuple[int, bool], BlockGeometry] = {}
    for k in range(top + 1):
        for up in (True, False):
            if (up and k >= top) or (not up and k <= 0):
                continue
            if up:
                A, AT, A_abs, AT_abs, nb = K.d[k], K.dT[k], K.d_abs[k], K.dT_abs[k], k + 1
            else:
                A, AT, A_abs, AT_abs, nb = K.dT[k - 1], K.d[k - 1], K.dT_abs[k - 1], K.d_abs[k - 1], k - 1
            scale = r = None
            if scaling == "dec":
                s = torch.exp((-0.5 if up else 0.5) * log_star[k])                 # S_up = star, S_down = 1/star
                A, AT = _csr_scale_cols(A, s), _csr_scale_rows(AT, s)
                A_abs, AT_abs = _csr_scale_cols(A_abs, s), _csr_scale_rows(AT_abs, s)
                scale = s.view(-1, 1, 1)
            if scaling != "jacobi":
                ones = torch.ones(K.n[k], 1, 1, dtype=dtype, device=log_star[k].device)
                r = ops.spmm(A_abs, ones)                                          # |A_raw| s
            blocks[(k, up)] = BlockGeometry(A=A, AT=AT, A_abs=A_abs, AT_abs=AT_abs, r=r, scale=scale, nb=nb)
    bcount = [None] + [ops.spmm(K.d_abs[k - 1], torch.ones(K.n[k - 1], 1, 1, dtype=dtype,
                                                          device=log_star[0].device)).clamp_min(1.0)
                       for k in range(1, top + 1)]
    out = dict(log_star=log_star, log_gm=log_gm, log_vol=torch.log(vol), blocks=blocks, bcount=bcount,
               geo=[g if g.dtype == dtype else g.to(dtype) for g in K.geo])
    store[key] = out
    return out


def block_beta_unit(ctx: "LayerContext", k: int) -> Tensor:
    """Gershgorin bound of ``A^T A`` for the (scaled) up-block operator of degree ``k`` (tensor metrics), cached.

    Returns:
        ``(1, 1)`` for a single complex or ``(num_graphs, 1)`` for a block-diagonal batch.
    """
    g: BlockGeometry = ctx.blocks[(k, True)]
    if not hasattr(g, "_beta_unit"):
        with torch.no_grad():
            bu = ops.beta_unit(g.A_abs, g.AT_abs, None, ctx.batch[k] if ctx.batch is not None else None,
                               ctx.num_graphs)
        g._beta_unit = bu.reshape(-1, 1)
    return g._beta_unit


@dataclass
class LayerContext:
    """Geometry-derived tensors shared by the lifting, all layers and the readout of one forward pass.

    Attributes:
        K: the ``CochainComplex``.
        top: top degree ``K.dim``.
        B: number of samples sharing the complex (1 for a block-diagonal batch of meshes).
        dtype: feature dtype (float32, or float64 in tests).
        batch: ``K.batch`` (list of ``(n_k,)`` int64 graph ids) or ``None`` for a single complex.
        num_graphs: number of graphs in the complex.
        log_star: ``log`` of the per-graph geometric-mean normalised reference star, ``(n_k,)``.
        blocks: ``{(k, is_up): BlockGeometry}`` operators of every block (scaling folded in).
        bcount: number of boundary ``(k-1)``-cells of each ``k``-cell, ``(n_k, 1, 1)`` (``None`` for k = 0).
        geo: ``K.geo[k]`` in ``dtype``, ``(n_k, G_k)``.
        even: even raw inputs per degree ``(n_k, B, E_k)`` or ``None``.
        scaling: ``'dec' | 'jacobi' | 'none'``.
        record: whether layers record diagnostics.
        ref: optional fixed log-metric references ``(n_k, B)`` per degree (``cfg.metric_reference``) or ``None``.
        resolvent_prev: ``{k: (n_k, B, C)}`` last resolvent-layer solution per degree (warm starts), or ``None``.
        log_gm: ``log`` of the geometric mean of ``star_k`` per graph, ``(1|G,)`` (restores physical units).
        log_vol: ``log`` of the total dual measure ``sum star_0`` per graph, ``(1|G,)`` (domain scale ``L^D``).
    """

    K: Any
    top: int
    B: int
    dtype: torch.dtype
    batch: list[Tensor] | None
    num_graphs: int
    log_star: list[Tensor]
    blocks: dict
    bcount: list[Tensor | None]
    geo: list[Tensor]
    even: list[Tensor | None]
    scaling: str
    record: bool = False
    ref: list | None = None                                                  # log-scale metric references per degree
    resolvent_prev: dict | None = None                                       # last resolvent solution per degree
    log_gm: list | None = None                                               # log geometric mean of star_k per graph
    log_vol: Tensor | None = None                                            # log total dual measure per graph

    def cell_mean(self, v: Tensor, k: int) -> Tensor:
        """Per-sample mean over ``k``-cells, broadcast back to cells.

        Args:
            v: ``(n_k, B)``.
            k: degree.

        Returns:
            ``(1, B)`` (single complex) or ``(n_k, B)`` (block-diagonal batch).
        """
        if self.batch is None:
            return v.mean(0, keepdim=True)
        idx = self.batch[k]
        return ops.segment_mean(v, idx, self.num_graphs)[idx]

    def cell_max(self, v: Tensor, k: int) -> Tensor:
        """Per-sample maximum over ``k``-cells.

        Args:
            v: ``(n_k, B)``.
            k: degree.

        Returns:
            ``(1, B)`` (single complex) or ``(num_graphs, B)`` (block-diagonal batch).
        """
        if self.batch is None:
            return v.amax(0, keepdim=True)
        return ops.segment_max(v, self.batch[k], self.num_graphs)

    def to_cells(self, stat: Tensor, k: int) -> Tensor:
        """Broadcast a per-sample statistic ``(1|num_graphs, B)`` to ``k``-cells: ``(1, B)`` or ``(n_k, B)``."""
        if self.batch is None:
            return stat
        return stat[self.batch[k]]


def make_context(K: Any, *, scaling: str, B: int, dtype: torch.dtype, even: list[Tensor | None] | None = None,
                 record: bool = False) -> LayerContext:
    """Build the per-forward :class:`LayerContext` (geometry only, no gradients; operators cached per complex).

    Args:
        K: ``CochainComplex`` (its float tensors must have dtype ``dtype``).
        scaling: ``'dec' | 'jacobi' | 'none'``.
        B: batch size of the features.
        dtype: feature dtype.
        even: even raw inputs per degree (``(n_k, B, E_k)`` or ``None``), length ``K.dim + 1``.
        record: record diagnostics in the layers.

    Returns:
        The context.
    """
    if scaling not in _SCALINGS:
        raise ValueError(f"scaling must be one of {_SCALINGS}, got {scaling!r}")
    top = int(K.dim)
    geom = _complex_operators(K, scaling, dtype)
    if even is None:
        even = [None] * (top + 1)
    return LayerContext(K=K, top=top, B=int(B), dtype=dtype, batch=K.batch,
                        num_graphs=int(K.num_graphs) if K.batch is not None else 1, log_star=geom["log_star"],
                        blocks=geom["blocks"], bcount=geom["bcount"], geo=geom["geo"], even=list(even),
                        scaling=scaling, record=record, log_gm=geom["log_gm"], log_vol=geom["log_vol"])


# --------------------------------------------------------------------------------------------------------------
# normalised block operators
# --------------------------------------------------------------------------------------------------------------
class BlockOperands(NamedTuple):
    """Normalised operands of one metric Hodge block (see module docstring).

    Attributes:
        A: CSR ``(m, n_k)`` used in the products (``A diag(s)`` when the scaling is folded in).
        AT: CSR ``(n_k, m)``, transpose of ``A``.
        s: scaling that still has to be applied explicitly (Jacobi: ``(n_k, B, 1)``), else ``None``.
        scale: the block scaling ``S^{-1/2}`` (``(n_k, 1, 1)``, ``(n_k, B, 1)``) or ``None`` (identity).
        log_h: ``log(H / beta)`` on the ``m`` neighbour cells, ``(m, B)``.
        beta: per-sample Gershgorin bound, ``(1, B)`` or ``(num_graphs, B)``.
        nb: degree of the neighbour cells (``k+1`` for up, ``k-1`` for down).
    """

    A: Tensor
    AT: Tensor
    s: Tensor | None
    scale: Tensor | None
    log_h: Tensor
    beta: Tensor
    nb: int


def block_operands(ctx: LayerContext, k: int, up: bool, log_H: Tensor) -> BlockOperands:
    """Scaling, Gershgorin bound and normalised metric of the up (``up=True``) or down block of degree ``k``.

    Args:
        ctx: forward context.
        k: degree of the block's cochains.
        up: up block (``A = d_k``, metric on ``(k+1)``-cells) or down block (``A = d_{k-1}^T``, metric on
            ``(k-1)``-cells).
        log_H: ``log H`` of the block metric, ``(m, B)``.

    Returns:
        :class:`BlockOperands`.
    """
    if (up and k >= ctx.top) or (not up and k <= 0):
        raise ValueError(f"no {'up' if up else 'down'} block for degree {k}")
    g: BlockGeometry = ctx.blocks[(k, up)]
    H = torch.exp(log_H).unsqueeze(-1)                                      # (m, B, 1)
    if ctx.scaling == "jacobi":
        S = spmm_ad(g.AT_abs, g.A_abs, H)                                   # (n_k, B, 1) = diag(A^T H A)
        pos = S > 0
        s = torch.where(pos, torch.where(pos, S, torch.ones_like(S)).rsqrt(), torch.zeros_like(S))
        r = spmm_ad(g.A_abs, g.AT_abs, s)                                   # (m, B, 1)
        rb = (spmm_ad(g.AT_abs, g.A_abs, (H * r).contiguous()) * s).squeeze(-1)
        s_explicit, scale = s, s
    else:                                                                   # |A| already carries the scaling
        rb = spmm_ad(g.AT_abs, g.A_abs, (H * g.r).contiguous()).squeeze(-1)  # (n_k, B) Gershgorin row bounds
        s_explicit, scale = None, g.scale
    beta = ctx.cell_max(rb, k)
    beta = torch.where(beta > 0, beta, torch.ones_like(beta))               # empty operator -> beta := 1
    log_h = log_H - ctx.to_cells(torch.log(beta), g.nb)
    return BlockOperands(A=g.A, AT=g.AT, s=s_explicit, scale=scale, log_h=log_h, beta=beta, nb=g.nb)


def apply_L(op: BlockOperands, z: Tensor) -> Tensor:
    """Normalised block operator ``L z = s ⊙ A^T (h ⊙ A (s ⊙ z))``: ``(n_k, B, C) -> (n_k, B, C)``."""
    h = torch.exp(op.log_h).unsqueeze(-1)
    return _mul(spmm_ad(op.AT, op.A, h * spmm_ad(op.A, op.AT, _mul(z, op.s))), op.s)


def apply_T(op: BlockOperands, y: Tensor) -> Tensor:
    """Half operator ``T y = s ⊙ A^T (sqrt(h) ⊙ y)`` (cross-degree transport): ``(m, B, C) -> (n_k, B, C)``."""
    hs = torch.exp(0.5 * op.log_h).unsqueeze(-1)
    return _mul(spmm_ad(op.AT, op.A, hs * y), op.s)


def _sym_eigvals(M: Tensor) -> Tensor:
    """Ascending eigenvalues of symmetric ``(..., D, D)`` matrices, ``D in {2, 3}``, in closed form (diagnostics;
    batched cuSOLVER rejects very large batches of tiny matrices)."""
    D = M.shape[-1]
    if D == 2:
        a, b, d = M[..., 0, 0], M[..., 0, 1], M[..., 1, 1]
        m, r = 0.5 * (a + d), torch.sqrt((0.5 * (a - d)) ** 2 + b * b)
        return torch.stack([m - r, m + r], -1)
    if D != 3:
        return torch.linalg.eigvalsh(M)
    q = torch.diagonal(M, dim1=-2, dim2=-1).sum(-1) / 3.0
    p1 = M[..., 0, 1] ** 2 + M[..., 0, 2] ** 2 + M[..., 1, 2] ** 2
    p2 = (M[..., 0, 0] - q) ** 2 + (M[..., 1, 1] - q) ** 2 + (M[..., 2, 2] - q) ** 2 + 2.0 * p1
    p = torch.sqrt(p2 / 6.0)
    ps = torch.where(p > 0, p, torch.ones_like(p))
    Bm = (M - q[..., None, None] * torch.eye(3, dtype=M.dtype, device=M.device)) / ps[..., None, None]
    r = (torch.linalg.det(Bm) / 2.0).clamp(-1.0, 1.0)
    phi = torch.acos(r) / 3.0
    e1 = q + 2.0 * p * torch.cos(phi)
    e3 = q + 2.0 * p * torch.cos(phi + 2.0 * math.pi / 3.0)
    e2 = 3.0 * q - e1 - e3
    return torch.stack([e3, e2, e1], -1)


_GRAM_CACHE: "weakref.WeakKeyDictionary[Any, dict]" = weakref.WeakKeyDictionary()


@torch.no_grad()
def dyad_gram_inverse(K: Any, km: int, dtype: torch.dtype) -> Tensor:
    """Inverse Gram matrices of the edge dyads ``t_j t_j^T`` of every top cell, ``(n_top, m, m)`` (cached).

    ``G_f[i, j] = <t_i t_i^T, t_j t_j^T>_F = (t_i . t_j)^2``; the dyads span Sym(2) on triangles (their tangent plane on
    surfaces) and Sym(3) on tets, so ``G_f`` is invertible for non-degenerate cells.
    """
    try:
        store = _GRAM_CACHE.setdefault(K, {})
    except TypeError:
        store = {}
    key = (km, dtype, str(K.pos.device))
    if key not in store:
        t = K.whitney[km]["t"].to(torch.float64)
        G = (t @ t.transpose(-1, -2)).square()
        store[key] = torch.linalg.inv(G).to(dtype)
    return store[key]


def _expm_sym(S: Tensor) -> Tensor:
    """Matrix exponential of symmetric ``(..., D, D)`` matrices: closed form for D = 2, scaling-and-squaring Taylor
    otherwise (no eigendecomposition: smooth gradients at repeated eigenvalues, e.g. ``S = 0``)."""
    D = S.shape[-1]
    if D == 2:
        a, b, c = S[..., 0, 0], S[..., 0, 1], S[..., 1, 1]
        m, h = 0.5 * (a + c), 0.5 * (a - c)
        q = h * h + b * b
        small = q < 1e-8
        d = torch.sqrt(torch.where(small, torch.ones_like(q), q))
        ch = torch.where(small, 1.0 + q / 2.0 + q * q / 24.0, torch.cosh(d))
        shc = torch.where(small, 1.0 + q / 6.0 + q * q / 120.0, torch.sinh(d) / d)
        em = torch.exp(m)
        e00, e11, e01 = em * (ch + shc * h), em * (ch - shc * h), em * shc * b
        return torch.stack([torch.stack([e00, e01], -1), torch.stack([e01, e11], -1)], -2)
    squarings = 6                                                            # ||S|| <= ~12 -> ||S / 64|| <= 0.2
    X = S / (2.0 ** squarings)
    eye = torch.eye(D, dtype=S.dtype, device=S.device).expand_as(S)
    E, term = eye, eye
    for n in range(1, 11):                                                   # Taylor order 10
        term = term @ X / n
        E = E + term
    for _ in range(squarings):
        E = E @ E
    return E


_GALERKIN_CACHE: "weakref.WeakKeyDictionary[Any, dict]" = weakref.WeakKeyDictionary()


@torch.no_grad()
def galerkin_geometry(K: Any, km: int, dtype: torch.dtype) -> dict:
    """Quadrature data of the Whitney ``km``-form Galerkin star of every top cell (cached per complex).

    The basis functions are those of ``K.whitney[km]`` (same local slots and orientation signs; ``rhmp.geometry``),
    evaluated in float64 from ``K.pos`` at the degree-2 rule (exact for ``int w_i . sigma_f w_j`` with constant
    ``sigma_f``): edges of triangles/surfaces (3 x 3 blocks), edges (6 x 6) and faces (4 x 4) of tets.

    Returns:
        dict with ``W (n_top, Q, D, m)`` basis values and ``wq (n_top, Q)`` weights (cell measure included).
    """
    try:
        store = _GALERKIN_CACHE.setdefault(K, {})
    except TypeError:
        store = {}
    key = (km, dtype, str(K.pos.device))
    if key in store:
        return store[key]
    from . import geometry as geo
    d = int(K.dim)
    Wk = K.whitney[km]
    T = K.cells[d][:, :d + 1].long()
    V = K.pos.to(torch.float64)[T]
    grads = geo.barycentric_gradients(V)
    meas = geo._simplex_measure(V)
    if d == 2:
        lam = torch.tensor(geo._TRI_Q, dtype=torch.float64, device=V.device)
        W = geo.whitney1_values(lam, grads, torch.tensor(geo._TRI_SLOTS, device=V.device),
                                Wk["signs"].to(torch.float64))
    elif km == 1:
        lam = torch.tensor(geo._TET_Q, dtype=torch.float64, device=V.device)
        W = geo.whitney1_values(lam, grads, torch.tensor(geo._TET_EDGES, device=V.device),
                                Wk["signs"].to(torch.float64))
    else:
        lam = torch.tensor(geo._TET_Q, dtype=torch.float64, device=V.device)
        loc = torch.tensor(geo._TET_FACES, device=V.device)
        face_local = torch.gather(loc.expand(T.shape[0], 4, 3), 2, T[:, loc].argsort(dim=2))
        W = geo.whitney2_values_tet(lam, grads, face_local)
    Q = lam.shape[0]
    out = dict(W=W.to(dtype), wq=(meas / Q)[:, None].expand(-1, Q).to(dtype).contiguous())
    store[key] = out
    return out


def galerkin_blocks(geom: dict, sigma: Tensor) -> Tensor:
    """Galerkin star blocks ``M_f[i, j] = int_f w_i . sigma_f w_j`` for per-cell tensors (differentiable in sigma).

    Args:
        geom: :func:`galerkin_geometry`.
        sigma: ``(n_top, B, D, D)`` symmetric.

    Returns:
        ``(n_top, B, m, m)`` symmetric blocks in the slot basis of ``K.whitney[km]``.
    """
    W = geom["W"].to(sigma.dtype)
    SW = torch.einsum("fbde,fqej->fbqdj", sigma, W)                         # sigma w_j at the quadrature points
    M = torch.einsum("fq,fqdi,fbqdj->fbij", geom["wq"].to(sigma.dtype), W, SW)
    return 0.5 * (M + M.transpose(-1, -2))


class TensorMetric:
    """Galerkin material-tensor metric ``H_km`` of one forward pass (per top cell and sample).

    ``cone``: coefficients ``(b, a)`` of the precomputed Whitney blocks (``rhmp.dec``; ``sigma = b I + sum a t t^T``).
    ``blocks``: explicit element matrices ``M_f (n_top, B, m, m)`` of full tensors ``sigma_f`` (:func:`galerkin_blocks`;
    exact for any SPD tensor and well conditioned for any cell shape).  Both use the slot maps of ``K.whitney[km]``.
    """

    def __init__(self, ctx: LayerContext, km: int, b: Tensor | None = None, a: Tensor | None = None,
                 blocks: Tensor | None = None, sigma: Tensor | None = None) -> None:
        self.ctx, self.km, self.b, self.a, self.blocks, self.sigma = ctx, km, b, a, blocks, sigma
        self._mt: dict = {}                     # slot-major blocks per (grad mode, dtype, B): reused by every apply

    def scale(self, f: Tensor) -> "TensorMetric":
        """Metric multiplied by a positive per-top-cell factor ``f`` ``(n_top|1, 1|B)``."""
        if self.blocks is not None:
            return TensorMetric(self.ctx, self.km, blocks=self.blocks * f[..., None, None],
                                sigma=None if self.sigma is None else self.sigma * f[..., None, None])
        return TensorMetric(self.ctx, self.km, b=self.b * f, a=None if self.a is None else self.a * f.unsqueeze(-1))

    def _slots_to_cells(self, v: Tensor) -> Tensor:                          # (n_top, B, m) -> (n_km, B)
        W = self.ctx.K.whitney[self.km]
        n_top, B, m = v.shape
        return ops.spmm(W["scatter"], v.permute(2, 0, 1).reshape(m * n_top, B).contiguous(), W["gather"])

    def apply(self, x: Tensor) -> Tensor:
        """``H x``: ``(n_km, B, C) -> (n_km, B, C)`` (per-sample bitwise independent forward).

        The slot-major blocks are built once per grad mode (e.g. once for all CG iterations under ``no_grad`` and once
        for the gradient pass) instead of on every application."""
        from . import dec
        W = self.ctx.K.whitney[self.km]
        B = x.shape[1]
        key = (torch.is_grad_enabled(), x.dtype, B)
        Mt = self._mt.get(key)
        if Mt is None:
            M = dec.whitney_blocks(self.ctx.K, self.km, self.b, self.a) if self.blocks is None else self.blocks
            M = M.to(x.dtype)
            if M.shape[1] != B:
                M = M.expand(M.shape[0], B, M.shape[2], M.shape[3])
            Mt = self._mt[key] = M.permute(2, 3, 0, 1).contiguous()
        return dec._WhitneyApply.apply(Mt, x.contiguous(), W["gather"], W["scatter"])

    def rowsum_abs(self) -> Tensor:
        """Block-wise upper bound of the row sums of ``|H|`` ``(n_km, B)`` (Gershgorin; differentiable)."""
        if self.blocks is None:
            from . import dec
            return dec.whitney_rowsum_abs(self.ctx.K, self.km, self.b, self.a)
        return self._slots_to_cells(self.blocks.abs().sum(-1))

    @torch.no_grad()
    def diagonal(self) -> Tensor:
        """Diagonal of ``H`` ``(n_km, B)``."""
        if self.blocks is None:
            from . import dec
            M = dec.whitney_blocks(self.ctx.K, self.km, self.b, self.a)
        else:
            M = self.blocks
        return self._slots_to_cells(torch.diagonal(M, dim1=-2, dim2=-1).contiguous())

    def params(self) -> list[Tensor]:
        """Tensors inside :meth:`apply` that require gradients."""
        cand = [self.blocks] if self.blocks is not None else [self.b, self.a]
        return [t for t in cand if isinstance(t, Tensor) and t.requires_grad]

    def tensor(self) -> Tensor:
        """``sigma_f`` ``(n_top, B, D, D)``."""
        if self.sigma is not None:
            return self.sigma
        t = self.ctx.K.whitney[self.km]["t"].to(self.b.dtype)
        eye = torch.eye(t.shape[-1], dtype=self.b.dtype, device=self.b.device)
        out = self.b[..., None, None] * eye
        return out if self.a is None else out + torch.einsum("fbj,fjd,fje->fbde", self.a, t, t)

    def as_dyads(self) -> tuple[Tensor, Tensor]:
        """``(b, a)`` with ``sigma = b I + sum_j a_j t_j t_j^T`` (reporting).

        Full tensors: the minimum-norm coordinates in the basis ``{I, t_j t_j^T}`` (pseudo-inverse in orthonormal
        coordinates of Sym(D)); they stay bounded on needle-shaped cells, while on flat caps any dyad representation is
        ill-conditioned -- prefer :meth:`tensor` (``metric_fields(...)['sigma']``) for per-cell material fields.
        """
        if self.blocks is None:
            return self.b, self.a
        f64 = torch.float64
        t = self.ctx.K.whitney[self.km]["t"].to(f64)                             # (n_top, m, D)
        D = t.shape[-1]
        iu = torch.triu_indices(D, D, device=t.device)
        w = torch.where(iu[0] == iu[1], 1.0, math.sqrt(2.0)).to(f64)

        def coords(X: Tensor) -> Tensor:                                         # orthonormal Sym(D) coordinates
            return X[..., iu[0], iu[1]] * w

        eye = torch.eye(D, dtype=f64, device=t.device)
        Y = torch.cat([coords(eye).expand(t.shape[0], 1, -1), coords(t.unsqueeze(-1) * t.unsqueeze(-2))], 1)
        c = torch.einsum("fkp,fbp->fbk", torch.linalg.pinv(Y.transpose(1, 2), rtol=1e-12),
                         coords(self.sigma.to(f64)))                              # (n_top, B, 1 + m)
        return c[..., 0].to(self.sigma.dtype), c[..., 1:].to(self.sigma.dtype)


class TensorOperands(NamedTuple):
    """Normalised operands of an up block with a Whitney material-tensor metric (DESIGN §9.1).

    Attributes:
        A, AT: (scaled) operator ``d_k diag(s)`` and its transpose.
        tm: the :class:`TensorMetric` ``H_{k+1}``.
        km: metric degree ``k + 1``.
        inv_beta: ``1 / beta`` broadcast to the ``km``-cells ``(1|n_km, B, 1)``; ``beta = rho * beta_unit``.
        cross_scale: ``1 / (sqrt(beta_unit) rho)`` broadcast likewise (normalises the cross-up transport).
        beta: ``(S, B)``.
    """

    A: Tensor
    AT: Tensor
    tm: "TensorMetric"
    km: int
    inv_beta: Tensor
    cross_scale: Tensor
    beta: Tensor


def tensor_operands(ctx: LayerContext, k: int, tm: "TensorMetric") -> TensorOperands:
    """Operands of the up block of degree ``k`` with the tensor metric ``H_{k+1}`` (``tm``).

    ``rho = max_e rowsum|H|_e >= lambda_max(H)`` (per sample), ``beta_unit >= lambda_max(A^T A)``, hence
    ``||A^T H A / (rho beta_unit)|| <= 1`` and ``||A^T H / (sqrt(beta_unit) rho)|| <= 1``.
    """
    km = k + 1
    g: BlockGeometry = ctx.blocks[(k, True)]
    rho = ctx.cell_max(tm.rowsum_abs(), km)                                  # (S, B)
    bu = block_beta_unit(ctx, k).to(rho.dtype)                               # (S, 1)
    beta = rho * bu
    return TensorOperands(A=g.A, AT=g.AT, tm=tm, km=km, inv_beta=(1.0 / ctx.to_cells(beta, km)).unsqueeze(-1),
                          cross_scale=(1.0 / ctx.to_cells(torch.sqrt(bu) * rho, km)).unsqueeze(-1), beta=beta)


def _whitney(ctx: LayerContext, op: TensorOperands, z: Tensor) -> Tensor:
    return op.tm.apply(z.contiguous())


def apply_L_tensor(ctx: LayerContext, op: TensorOperands, z: Tensor) -> Tensor:
    """``L z = A^T H A z / beta``: ``(n_k, B, C) -> (n_k, B, C)``."""
    return spmm_ad(op.AT, op.A, _whitney(ctx, op, spmm_ad(op.A, op.AT, z)) * op.inv_beta)


def apply_T_tensor(ctx: LayerContext, op: TensorOperands, y: Tensor) -> Tensor:
    """Cross-up transport ``T y = A^T H y / (sqrt(beta_unit) rho)``: ``(n_{k+1}, B, C) -> (n_k, B, C)``."""
    return spmm_ad(op.AT, op.A, _whitney(ctx, op, y * op.cross_scale))


class _PolyBlockFn(torch.autograd.Function):
    """Fused ``out = A^T( sum_p (c_p h) ⊙ q_p + u ⊙ X )`` with ``q_{p+1} = A( A^T( h ⊙ q_p ) )`` (scaling folded
    into ``A``).

    ``q_1 = A x`` is an input (shared with the metric invariants), so ``out = sum_p c_p L^p x + T(w X)`` with
    ``u = w sqrt(h)``.  Only ``q_2 .. q_P`` are saved for the backward.
    """

    @staticmethod
    def forward(ctx: Any, q1: Tensor, h: Tensor, c: Tensor, u: Tensor | None, X: Tensor | None,
                A: Tensor, AT: Tensor) -> Tensor:  # noqa: D102
        P = c.shape[0]
        qs = [q1]
        for _ in range(1, P):
            qs.append(ops.spmm(A, ops.spmm(AT, h * qs[-1])))
        hc = h * c                                                          # (m, B, P): c_p h
        z = qs[0] * hc[..., 0:1]
        for i in range(1, P):
            z.addcmul_(qs[i], hc[..., i:i + 1])
        if u is not None:
            z.addcmul_(X, u)
        out = ops.spmm(AT, z)
        ctx.A, ctx.AT = A, AT
        ctx.save_for_backward(h, c, u, X, *qs)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, g: Tensor):  # noqa: D102
        h, c, u, X, *qs = ctx.saved_tensors
        A, AT = ctx.A, ctx.AT
        P = len(qs)
        need = ctx.needs_input_grad
        G = ops.spmm(A, g.contiguous())                                     # dL/dz   (m, B, C)
        grad_u = torch.linalg.vecdot(G, X).unsqueeze(-1) if (u is not None and need[3]) else None
        vd = [torch.linalg.vecdot(G, q) for q in qs]                         # <G, q_p>  (m, B)
        hs = h.squeeze(-1)
        grad_c = torch.stack([(hs * v).sum() for v in vd]) if need[2] else None
        grad_h = vd[0] * c[0]
        for i in range(1, P):
            grad_h = grad_h + vd[i] * c[i]
        hc = h * c
        gamma = G * hc[..., P - 1:P]                                         # dL/dq_P
        for i in range(P - 2, -1, -1):
            t = ops.spmm(A, ops.spmm(AT, gamma))                             # dL/d(h ⊙ q_i) via q_{i+1}
            grad_h = grad_h + torch.linalg.vecdot(t, qs[i])
            gamma = G * hc[..., i:i + 1]
            gamma.addcmul_(h, t)                                             # c_i h G + h t
            del t
        grad_X = G * u if (u is not None and need[4]) else None
        return gamma, (grad_h.unsqueeze(-1) if need[1] else None), grad_c, grad_u, grad_X, None, None


def poly_block(q1: Tensor, h: Tensor, c: Tensor, u: Tensor | None, X: Tensor | None, op: BlockOperands) -> Tensor:
    """Fused polynomial block (see :class:`_PolyBlockFn`); requires the scaling to be folded into ``op.A``.

    Args:
        q1: ``A x`` (with the folded scaling), ``(m, B, C)``.
        h: ``exp(op.log_h)`` as ``(m, B, 1)``.
        c: polynomial coefficients ``(P,)``.
        u: ``w * sqrt(h)`` ``(m, B, 1)`` or ``None`` (no cross term).
        X: neighbour-degree features ``(m, B, C)`` (used iff ``u`` is given).
        op: block operands.

    Returns:
        ``sum_p c_p L^p x + w T X``, ``(n_k, B, C)``.
    """
    if op.s is not None:
        raise ValueError("poly_block needs a scaling folded into the operator; use poly_block_reference for "
                         "'jacobi'")
    if u is None:
        X = None
    return _PolyBlockFn.apply(q1.contiguous(), h.contiguous(), c, u, X, op.A, op.AT)


def poly_block_reference(x: Tensor, h: Tensor, c: Tensor, u: Tensor | None, X: Tensor | None, op: BlockOperands,
                         q1: Tensor | None = None) -> Tensor:
    """Plain-autograd reference of :func:`poly_block` (also used for ``scaling='jacobi'``).

    Args:
        x: block input ``(n_k, B, C)``.
        h, c, u, X, op: as in :func:`poly_block`.
        q1: optional precomputed ``A (s ⊙ x)``.

    Returns:
        ``(n_k, B, C)``.
    """
    A, AT, s = op.A, op.AT, op.s
    if q1 is None:
        q1 = spmm_ad(A, AT, _mul(x, s))
    qs = [q1]
    for _ in range(1, c.shape[0]):
        y = _mul(spmm_ad(AT, A, h * qs[-1]), s)                              # L^p x
        qs.append(spmm_ad(A, AT, _mul(y, s)))
    z = h * c[0] * qs[0]
    for i in range(1, c.shape[0]):
        z = z + h * c[i] * qs[i]
    if u is not None:
        z = z + u * X
    return _mul(spmm_ad(AT, A, z), s)


# --------------------------------------------------------------------------------------------------------------
# resolvent (implicit) blocks (DESIGN §9.2)
# --------------------------------------------------------------------------------------------------------------
# Cap of the resolvent time steps: cond(I + tau_up L_up + tau_dn L_dn) <= 1 + tau_up + tau_dn since ||L|| <= 1.
TAU_MAX = 100.0
_RESOLVENT_GRADS = ("implicit", "unrolled")


def _apply_L_h(A: Tensor, AT: Tensor, s: Tensor | None, h: Tensor, z: Tensor) -> Tensor:
    """``s ⊙ A^T (h ⊙ A (s ⊙ z))`` with a precomputed ``h`` ``(m, B, 1)``."""
    return _mul(spmm_ad(AT, A, h * spmm_ad(A, AT, _mul(z, s))), s)


def resolvent_matvec(y: Tensor, ops_up: tuple | None, ops_dn: tuple | None, h_up, h_dn, s_up, s_dn, tau_up,
                     tau_dn) -> Tensor:
    """``(I + tau_up L_up + tau_dn L_dn) y`` with ``tau`` folded into the (normalised) metrics.

    Args:
        y: ``(n_k, B, C)``.
        ops_up, ops_dn: ``(A, AT)`` of the blocks or ``None``.
        h_up, h_dn: ``exp(log_h)`` ``(m, B, 1)``; s_up, s_dn: explicit scalings (Jacobi) or ``None``.
        tau_up, tau_dn: scalar tensors.
    """
    out = y
    if ops_up is not None:
        out = out + _apply_L_h(ops_up[0], ops_up[1], s_up, tau_up * h_up, y)
    if ops_dn is not None:
        out = out + _apply_L_h(ops_dn[0], ops_dn[1], s_dn, tau_dn * h_dn, y)
    return out


def resolvent_apply(ctx: LayerContext, k: int, x: Tensor, op_up: BlockOperands | None, op_dn: BlockOperands | None,
                    tau_up: Tensor | None, tau_dn: Tensor | None, iters: int, grad: str = "implicit",
                    x0: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """``y = (I + tau_up L_up + tau_dn L_dn)^{-1} x`` for degree ``k`` by batched CG (``rhmp.dec.cg_solve``).

    Per-sample (per-graph) Frobenius inner products over (cells, channels) make the solve O(C)-equivariant and
    batch-independent.  A fixed number of iterations runs without host synchronisation; samples freeze at a relative
    residual of ``1e-12``.  ``cond <= 1 + tau_up + tau_dn`` since ``||L|| <= 1``.

    Warm start: with an initial guess ``x0`` (e.g. the previous resolvent layer's solution) the correction
    ``delta = M^{-1} (x - M x0)`` is solved from zero and ``y = x0 + delta`` (``x0`` detached).  Autograd through the
    right-hand side makes the implicit gradients exact: ``-lam^T (dM) (delta + x0) = -lam^T (dM) y``.

    Channels: the solve acts identically on every channel and commutes with channel mixing, ``M^{-1}(x Q) =
    (M^{-1} x) Q``, so there is no equivariant way to reduce its cost by mixing or compressing channels first; the
    cost is ``iters`` matvecs on ``(n_k, B, C)`` (plus the adjoint solve in the backward).

    Args:
        ctx: context.
        k: degree.
        x: ``(n_k, B, C)``.
        op_up, op_dn: normalised block operands (``None`` if the block does not exist).
        tau_up, tau_dn: scalar tensors ``> 0``.
        iters: CG iterations.
        grad: ``'implicit'`` (adjoint CG solve, memory independent of ``iters``) or ``'unrolled'`` (autograd through
            the iterations, recomputed in the backward via checkpointing; memory grows with ``iters``).
        x0: optional initial guess ``(n_k, B, C)`` (treated as a constant).

    Returns:
        ``(y, rel_res)``: ``(n_k, B, C)`` and the per-sample relative residual ``|x - M y| / |x|`` ``(S, B)`` (no grad).
    """
    from . import dec
    if grad not in _RESOLVENT_GRADS:
        raise ValueError(f"resolvent grad must be one of {_RESOLVENT_GRADS}, got {grad!r}")
    h_up = s_up = h_dn = s_dn = None
    ops_up = ops_dn = None
    if op_up is not None:
        h_up, s_up, ops_up = torch.exp(op_up.log_h).unsqueeze(-1), op_up.s, (op_up.A, op_up.AT)
    if op_dn is not None:
        h_dn, s_dn, ops_dn = torch.exp(op_dn.log_h).unsqueeze(-1), op_dn.s, (op_dn.A, op_dn.AT)
    params = (h_up, h_dn, s_up, s_dn, tau_up, tau_dn)
    batch = ctx.batch[k] if ctx.batch is not None else None
    G = ctx.num_graphs
    x = x.contiguous()
    rhs = x
    if x0 is not None:
        x0 = x0.detach()
        rhs = (x - resolvent_matvec(x0, ops_up, ops_dn, *params)).contiguous()
    if grad == "implicit":
        leaves = [p for p in params if p is not None and p.requires_grad]
        y, res = dec.cg_solve_implicit(lambda v: resolvent_matvec(v, ops_up, ops_dn, *params), rhs, params=leaves,
                                       iters=int(iters), tol=0.0, batch=batch, num_graphs=G, early_exit=False)
    else:
        def solve(xx: Tensor, *pp) -> tuple[Tensor, Tensor]:
            return dec.cg_solve(lambda v: resolvent_matvec(v, ops_up, ops_dn, *pp), xx, int(iters), 0.0, None,
                                batch, G, early_exit=False)

        if torch.is_grad_enabled():
            y, res = _checkpoint(solve, rhs, *params, use_reentrant=False)
        else:
            y, res = solve(rhs, *params)
    rel = res[-1].detach()
    if x0 is not None:
        y = x0 + y
        with torch.no_grad():                                               # residual relative to |x|, not |rhs|
            nr = torch.linalg.vecdot(rhs, rhs).sum(0, keepdim=True) if batch is None else \
                ops.segment_sum(torch.linalg.vecdot(rhs, rhs), batch, G)
            nx = torch.linalg.vecdot(x, x).sum(0, keepdim=True) if batch is None else \
                ops.segment_sum(torch.linalg.vecdot(x, x), batch, G)
            rel = rel * torch.sqrt(nr / torch.where(nx > 0, nx, torch.ones_like(nx)))
    return y, rel


# --------------------------------------------------------------------------------------------------------------
# physical (un-normalised) metric Hodge operators and solve layers
# --------------------------------------------------------------------------------------------------------------
_SOLVE_BCS = ("dirichlet", "none", "neumann")


def dirichlet_free(ctx: LayerContext, k: int, bc: str) -> Tensor | None:
    """Mask of the free k-cells ``(n_k, 1, 1)`` (1 free, 0 fixed) or ``None`` (``bc='none'`` / ``'neumann'``).

    Fixed cells: ``K.meta['dirichlet'][k]`` (bool ``(n_k,)``, list or dict per degree) when the task provides it,
    else the boundary cells ``K.boundary[k]``.
    """
    if bc in ("none", "neumann"):
        return None
    if bc not in _SOLVE_BCS:
        raise ValueError(f"solve_bc must be one of {_SOLVE_BCS}, got {bc!r}")
    K = ctx.K
    fixed = None
    dm = K.meta.get("dirichlet") if isinstance(K.meta, dict) else None
    if isinstance(dm, dict):
        fixed = dm.get(k)
    elif isinstance(dm, (list, tuple)) and k < len(dm):
        fixed = dm[k]
    if fixed is None:
        fixed = K.boundary[k]
    return (~fixed.to(device=ctx.log_star[k].device, dtype=torch.bool)).to(ctx.dtype).view(-1, 1, 1)


class PhysicalHodge:
    """Un-normalised metric Hodge operator of degree ``k`` in the symmetric frame (physical units, no ``beta``)::

        A_sym = star_k^{-1/2} d_k^T H_{k+1} d_k star_k^{-1/2}  +  star_k^{1/2} d_{k-1} H^down_{k-1} d_{k-1}^T star_k^{1/2}

    so that the metric Hodge Laplacian on cochain values is ``Delta_H = star_k^{-1/2} A_sym star_k^{1/2}`` and the
    weak (FEM) stiffness is ``star_k^{1/2} A_sym star_k^{1/2}`` (``= d_0^T H_1 d_0`` for k = 0).  Built from the
    cached DEC-scaled operators (``ctx.blocks``); the per-graph star normalisation constants are folded into the
    effective metrics.  ``up`` is ``('diag', geometry, H (m, B, 1))`` or ``('tensor', geometry, TensorMetric)``; ``dn`` is
    ``('diag', geometry, H (m, B, 1))``.
    """

    def __init__(self, ctx: LayerContext, up: tuple | None, dn: tuple | None) -> None:
        self.ctx, self.up, self.dn = ctx, up, dn

    def apply(self, z: Tensor) -> Tensor:
        """``A_sym z``: ``(n_k, B, C) -> (n_k, B, C)``."""
        from . import dec
        out = None
        if self.up is not None:
            g = self.up[1]
            q = spmm_ad(g.A, g.AT, z)
            q = self.up[2] * q if self.up[0] == "diag" else self.up[2].apply(q.contiguous())
            out = spmm_ad(g.AT, g.A, q)
        if self.dn is not None:
            g = self.dn[1]
            t = spmm_ad(g.AT, g.A, self.dn[2] * spmm_ad(g.A, g.AT, z))
            out = t if out is None else out + t
        return out

    @torch.no_grad()
    def diagonal(self) -> Tensor:
        """Diagonal of ``A_sym`` per cell and sample ``(n_k, B, 1)`` (exact for diagonal metrics; for tensor metrics the
        within-block couplings are dropped).  Used as a (detached) Jacobi preconditioner."""
        out = None
        for part in (self.up, self.dn):
            if part is None:
                continue
            g = part[1]
            if not hasattr(g, "_AT_sq"):
                g._AT_sq = ops.csr_with_values(g.AT, g.AT.values().square())       # (Ã ∘ Ã)^T
            hd = part[2] if part[0] == "diag" else part[2].diagonal().unsqueeze(-1)
            t = ops.spmm(g._AT_sq, hd.to(g._AT_sq.dtype).contiguous())
            out = t if out is None else out + t
        return out

    def params(self) -> list[Tensor]:
        """Tensors inside :meth:`apply` that require gradients (for implicit CG gradients)."""
        ps = []
        for part in (self.up, self.dn):
            if part is None:
                continue
            if part[0] == "diag":
                ps += [part[2]] if part[2].requires_grad else []
            else:
                ps += part[2].params()
        return ps


def physical_hodge(ctx: LayerContext, k: int, logH_up: list, logH_dn: list, tensors: dict | None = None,
                   w_up: Tensor | None = None, w_dn: Tensor | None = None) -> PhysicalHodge:
    """:class:`PhysicalHodge` of degree ``k`` from a layer's log-metrics (``logH_up[k+1]``, ``logH_dn[k-1]``,
    normalised-star units) or its tensor metric ``tensors[k+1]`` (:class:`TensorMetric`); optional scalar block
    weights."""
    if ctx.scaling != "dec":
        raise ValueError("physical (un-normalised) operators need scaling='dec'")
    top, batch = ctx.top, ctx.batch

    def cells(v: Tensor, deg: int) -> Tensor:                                # (S,) per graph -> (n_deg|1, 1)
        return (v if batch is None else v[batch[deg]]).view(-1, 1)

    up = dn = None
    tensors = tensors or {}
    if k < top:
        g = ctx.blocks[(k, True)]
        fu = torch.exp(-ctx.log_gm[k])                                       # 1 / gm_k per graph
        if (k + 1) in tensors:
            f_top = cells(fu, top)
            if w_up is not None:
                f_top = f_top * w_up
            up = ("tensor", g, tensors[k + 1].scale(f_top))
        else:
            H = torch.exp(logH_up[k + 1] + cells(ctx.log_gm[k + 1], k + 1)) * cells(fu, k + 1)
            if w_up is not None:
                H = H * w_up
            up = ("diag", g, H.unsqueeze(-1))
    if k > 0:
        g = ctx.blocks[(k, False)]
        H = torch.exp(logH_dn[k - 1] - cells(ctx.log_gm[k - 1], k - 1)) * cells(torch.exp(ctx.log_gm[k]), k - 1)
        if w_dn is not None:
            H = H * w_dn
        dn = ("diag", g, H.unsqueeze(-1))
    return PhysicalHodge(ctx, up, dn)


# Target number of cells per aggregate of the two-level solve preconditioner (coarse space per graph).
COARSE_AGG_SIZE = 32
_COARSE_CACHE: "weakref.WeakKeyDictionary[Any, dict]" = weakref.WeakKeyDictionary()


@torch.no_grad()
def coarse_aggregates(K: Any, k: int, target: int | None = None) -> dict:
    """Spatial aggregates of the ``k``-cells of every graph (cached per complex): the cells (vertices, or barycentres
    of higher cells) are binned on a per-graph grid over the graph's bounding box with about ``target`` cells per bin
    (``ceil((n_g / target)^(1/d))`` bins per axis, ``d`` = intrinsic dimension); empty bins are dropped.

    Deterministic and local to each graph, so the aggregates of a graph do not depend on the other graphs of a
    block-diagonal batch.

    Returns:
        dict with ``local (n_k,)`` aggregate index within the graph, ``n_c`` (max aggregates per graph), ``G``,
        ``n_agg`` and the CSR maps between cells and padded coarse slots ``graph * n_c + local``:
        ``R ((G n_c), n_k)`` (sum over the aggregate) and ``RT (n_k, (G n_c))``.
    """
    target = COARSE_AGG_SIZE if target is None else int(target)
    try:
        store = _COARSE_CACHE.setdefault(K, {})
    except TypeError:
        store = {}
    key = (k, target, str(K.pos.device))
    if key in store:
        return store[key]
    P = K.pos.to(torch.float64)
    if k == 0:
        X = P
    else:
        cells = K.cells[k].long()
        valid = cells >= 0
        X = (P[cells.clamp_min(0)] * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
    n, D = X.shape
    dev = X.device
    G = int(K.num_graphs)
    g = K.batch[k].long() if (K.batch is not None and G > 1) else torch.zeros(n, dtype=torch.long, device=dev)
    lo = torch.full((G, D), float("inf"), dtype=torch.float64, device=dev).scatter_reduce(
        0, g[:, None].expand(n, D), X, "amin", include_self=True)
    hi = torch.full((G, D), -float("inf"), dtype=torch.float64, device=dev).scatter_reduce(
        0, g[:, None].expand(n, D), X, "amax", include_self=True)
    cnt = torch.bincount(g, minlength=G).to(torch.float64)
    d_int = max(1, min(int(K.dim), D))
    nb = torch.ceil((cnt / float(target)).clamp_min(1.0) ** (1.0 / d_int)).long().clamp_min(1)   # (G,)
    xi = (X - lo[g]) / (hi[g] - lo[g]).clamp_min(1e-300)
    c = torch.minimum((xi * nb[g, None]).floor().long().clamp_min(0), nb[g, None] - 1)             # (n, D)
    mult = nb[g, None] ** torch.arange(D, device=dev)[None, :]
    b = (c * mult).sum(1)
    span = int((nb.max() ** D).item())
    uniq, agg = torch.unique(g * span + b, return_inverse=True)                                    # graph-major
    agg_graph = uniq // span
    first = torch.searchsorted(agg_graph, torch.arange(G, device=dev))
    local_agg = torch.arange(uniq.numel(), device=dev) - first[agg_graph]
    n_c = int(local_agg.max().item()) + 1
    n_agg = int(uniq.numel())
    ones = torch.ones(n, dtype=torch.float32, device=dev)
    cols = torch.arange(n, device=dev)
    slot = (agg_graph * n_c + local_agg)[agg]                                       # padded coarse slot of every cell
    out = dict(local=local_agg[agg], n_c=n_c, G=G, n_agg=n_agg,
               R=ops.csr_from_coo(slot, cols, ones, (G * n_c, n)), RT=ops.csr_from_coo(cols, slot, ones, (n, G * n_c)))
    store[key] = out
    return out


@torch.no_grad()
def two_level_preconditioner(ctx: LayerContext, k: int, matvec: Any, q: Tensor, target: int | None = None):
    """Additive two-level preconditioner ``M^{-1} = I + Z (Z^T A Z)^{-1} Z^T`` of a Jacobi-scaled operator ``A``.

    ``Z`` has one column per aggregate (:func:`coarse_aggregates`) and graph: ``q`` restricted to the aggregate, where
    ``q (n_k, B, 1)`` is the near-kernel vector of ``A`` (the constants of the physical variable in the scaled frame;
    zero on fixed cells).  The coarse matrices ``A_c = Z^T A Z`` are assembled per sample and graph from one
    application of ``A`` to all aggregate indicators (packed into the channel dimension) and factorised in float64;
    the correction acts identically on every channel, so PCG stays O(C)-equivariant and per sample.

    Returns:
        ``apply(r) -> z`` on ``(n_k, B, C)``.
    """
    agg = coarse_aggregates(ctx.K, k, target)
    n, B = q.shape[0], q.shape[1]
    n_c, G = agg["n_c"], agg["G"]
    R, RT = agg["R"], agg["RT"]
    Zq = q.new_zeros(n, B, n_c).scatter_(2, agg["local"].view(n, 1, 1).expand(n, B, 1), q.expand(n, B, 1))
    S = ops.spmm(R, (q * matvec(Zq.contiguous())).contiguous())                   # ((G n_c), B, n_c) = Z^T A Z
    Ac = S.double().view(G, n_c, B, n_c).permute(0, 2, 1, 3)                       # (G, B, n_c, n_c)
    Ac = 0.5 * (Ac + Ac.transpose(-1, -2))
    dg = torch.diagonal(Ac, dim1=-2, dim2=-1)
    scale = dg.amax(-1, keepdim=True).clamp_min(1e-300)
    fix = torch.where(dg > 1e-12 * scale, 1e-13 * scale, torch.ones_like(dg))      # empty / padded slots -> 1
    Ac = Ac + torch.diag_embed(fix)
    L, info = torch.linalg.cholesky_ex(Ac)
    if bool((info != 0).any()):                                                    # fall back to the coarse diagonal
        bad = (info != 0)[..., None, None]
        L = torch.linalg.cholesky(torch.where(bad, torch.diag_embed(torch.diagonal(Ac, dim1=-2, dim2=-1)), Ac))
    Ainv = torch.cholesky_inverse(L)                                               # (G, B, n_c, n_c) float64
    Ainv = 0.5 * (Ainv + Ainv.transpose(-1, -2))

    def apply(r: Tensor) -> Tensor:
        C = r.shape[-1]
        rc = ops.spmm(R, (q * r).contiguous()).double().view(G, n_c, B, C).permute(0, 2, 1, 3)   # (G, B, n_c, C)
        sol = (Ainv @ rc).permute(0, 2, 1, 3).reshape(G * n_c, B, C).to(r.dtype).contiguous()
        return r + q * ops.spmm(RT, sol)
    return apply


def pcg_solve(matvec: Any, rhs: Tensor, precond: Any = None, iters: int = 32, tol: float = 1e-6,
              batch: Tensor | None = None, num_graphs: int = 1, early_exit: bool = True, check_every: int = 8
              ) -> tuple[Tensor, Tensor]:
    """``rhmp.dec.cg_solve`` with an optional symmetric positive definite preconditioner ``z = precond(r)`` (``None``:
    plain CG): same per-sample (and per-graph) Frobenius inner products, rhs normalisation and freezing rule
    (``||r|| <= max(tol, 1e-12) ||rhs||``), so per-sample independence and O(C)-equivariance carry over when
    ``precond`` acts per sample and identically on every channel.  The early exit is checked every ``check_every``
    iterations (one host synchronisation each; frozen samples do not change in between, so the result is the same).

    Returns:
        ``(x, res)`` with ``res (T+1, S, B)`` the relative residual norms (constant after a sample froze).
    """
    from .dec import _CG_RTOL_FLOOR, _bcast, _dot, _sqnorm
    G = num_graphs
    bn = _sqnorm(rhs, batch, G).sqrt()
    nz = bn > 0
    scale = torch.where(nz, bn, torch.ones_like(bn))
    r = rhs / _bcast(scale, batch, rhs)
    x = torch.zeros_like(r)
    rs = _sqnorm(r, batch, G)
    z = r if precond is None else precond(r)
    p = z
    rz = rs if precond is None else _dot(r, z, batch, G)
    thr = max(float(tol), _CG_RTOL_FLOOR)
    hist = [rs.sqrt()]
    active = nz & (rs.sqrt() > thr)
    every = max(1, int(check_every))
    for it in range(int(iters)):
        if early_exit and it % every == 0 and not bool(active.any()):
            break
        Ap = matvec(p)
        pAp = _dot(p, Ap, batch, G)
        ok = active & (pAp > 0)
        alpha = torch.where(ok, rz / torch.where(ok, pAp, torch.ones_like(pAp)), torch.zeros_like(rz))
        x = x + _bcast(alpha, batch, x) * p
        r = r - _bcast(alpha, batch, r) * Ap
        rs_new = _sqnorm(r, batch, G)
        z = r if precond is None else precond(r)
        rz_new = rs_new if precond is None else _dot(r, z, batch, G)
        okb = ok & (rz > 0)
        beta = torch.where(okb, rz_new / torch.where(okb, rz, torch.ones_like(rz)), torch.zeros_like(rz))
        p = z + _bcast(beta, batch, p) * p
        rz = torch.where(ok, rz_new, rz)
        rs = torch.where(ok, rs_new, rs)
        hist.append(rs.sqrt())
        active = ok & (rs.sqrt() > thr)
    return x * _bcast(scale, batch, x), torch.stack(hist)


class _PCGImplicit(torch.autograd.Function):
    """``x = A(theta)^{-1} rhs`` by PCG with adjoint gradients (the preconditioner is a constant of the solve)."""

    @staticmethod
    def forward(ctx: Any, rhs: Tensor, cfg: tuple, *params: Tensor):  # noqa: D102
        matvec, precond, iters, tol, batch, G, _, early = cfg
        with torch.no_grad():
            x, res = pcg_solve(matvec, rhs, precond, iters, tol, batch, G, early)
        ctx.cfg = cfg
        ctx.save_for_backward(x)
        ctx.mark_non_differentiable(res)
        return x, res

    @staticmethod
    def backward(ctx: Any, gx: Tensor, gres: Tensor):  # noqa: D102
        (x,) = ctx.saved_tensors
        matvec, precond, iters, tol, batch, G, params, early = ctx.cfg
        with torch.no_grad():
            lam, _ = pcg_solve(matvec, gx.contiguous(), precond, iters, tol, batch, G, early)
        grads = [None] * len(params)
        need = [i for i in range(len(params)) if ctx.needs_input_grad[2 + i]]
        if need:
            with torch.enable_grad():
                Ax = matvec(x.detach())
                gp = torch.autograd.grad(Ax, [params[i] for i in need], grad_outputs=-lam, allow_unused=True)
            for i, gi in zip(need, gp):
                grads[i] = gi
        return (lam if ctx.needs_input_grad[0] else None, None, *grads)


def pcg_solve_implicit(matvec: Any, rhs: Tensor, precond: Any, params: Any = (), iters: int = 32, tol: float = 1e-6,
                       batch: Tensor | None = None, num_graphs: int = 1, early_exit: bool = True
                       ) -> tuple[Tensor, Tensor]:
    """:func:`pcg_solve` with implicit (adjoint) gradients (as ``rhmp.dec.cg_solve_implicit``)."""
    params = tuple(params)
    return _PCGImplicit.apply(rhs, (matvec, precond, iters, tol, batch, num_graphs, params, early_exit), *params)


def physical_solve(ctx: LayerContext, k: int, x: Tensor, op: PhysicalHodge, shift: Tensor, free: Tensor | None,
                   iters: int, tol: float, x0: Tensor | None = None, precond: str = "none", mean_free: bool = False
                   ) -> tuple[Tensor, Tensor, Tensor, int]:
    """``y = (Delta_H + shift)^{-1} x`` on the free cells (``x = 0`` imposed on fixed cells) by batched (P)CG with
    implicit gradients (:func:`pcg_solve_implicit`, per-sample / per-graph inner products).

    Solved in the symmetric frame, ``y = star^{-1/2} (P (A_sym + shift) P)^{-1} P star^{1/2} x``; on the free cells this
    is ``(K + shift M)^{-1} M x`` with the weak stiffness ``K`` and the lumped mass ``M = star_k``.  Jacobi
    preconditioned (symmetric diagonal change of variables, per cell and sample), which removes the spread of the
    physical metric (slivers, material contrast) from the CG condition number.

    Args:
        ctx, k: context and degree; x: ``(n_k, B, C)`` (cochain values).
        op: the physical operator; shift: ``(1|n_k, 1|B, 1)`` identity shift (physical units).
        free: free-cell mask ``(n_k, 1, 1)`` or ``None``; iters, tol: CG budget; x0: optional warm start in the
            symmetric frame (detached).
        precond: ``'none'`` (Jacobi) or ``'twolevel'`` (Jacobi + additive coarse correction on spatial aggregates,
            :func:`two_level_preconditioner`; degree 0 only, where the near-kernel is the constants -- other degrees
            use Jacobi).
        mean_free: vertex solves with the natural boundary condition (``solve_bc='neumann'``): the lumped source
            ``M x`` is made compatible by removing its mean per graph, sample and channel (the ``eps -> 0`` limit of
            ``(K + eps I) y = M x``, as in the T5 generator), so the solution is the zero-mean (``1^T M y = 0``)
            solution of the pure Neumann problem instead of carrying a ``1/shift``-sized constant.

    Returns:
        ``(y, y_sym, rel_res, n_iter)``: solution ``(n_k, B, C)``, its symmetric-frame form (for warm starts), the final
        relative CG residual ``(S, B)`` and the number of CG iterations run.
    """
    sh = torch.exp(0.5 * ctx.log_star[k]).view(-1, 1, 1)                    # normalised star^{1/2}
    if mean_free:                  # compatible source: M x <- M x - mean(M x) per graph (eps -> 0 of (K + eps I))
        wm = torch.exp(ctx.log_star[k]).view(-1, 1, 1).to(x.dtype)
        if ctx.batch is None:
            mu = (wm * x).sum(0, keepdim=True) / x.shape[0]
        else:
            bk = ctx.batch[k]
            num = ops.segment_sum((wm * x).reshape(x.shape[0], -1), bk, ctx.num_graphs)
            cnt = torch.bincount(bk, minlength=ctx.num_graphs).to(x.dtype).clamp_min(1).view(-1, 1)
            mu = (num / cnt).index_select(0, bk).view_as(x)
        x = x - mu / wm
    # Jacobi preconditioning as a symmetric change of variables w = D^{1/2} v (D = diag(A_sym + shift), detached:
    # the exact solution does not depend on it), per cell and sample -> per-sample and O(C)-equivariant.
    with torch.no_grad():
        dg = op.diagonal() + shift.detach()
        dm = torch.where(dg > 0, dg, torch.ones_like(dg)).rsqrt()           # (n_k, B, 1) = D^{-1/2}
        if free is not None:
            dm = torch.where(free > 0, dm, torch.ones_like(dm))
        dmf = dm if free is None else dm * free                              # D^{-1/2} P (one multiply per side)

    def matvec(v: Tensor) -> Tensor:                                         # D^{-1/2} P (A + shift) P D^{-1/2}
        vp = v * dmf
        return (op.apply(vp) + shift * vp) * dmf

    rhs = x * sh * dm
    if free is not None:
        rhs = rhs * free
    if x0 is not None:
        x0 = x0.detach()
        rhs = rhs - matvec(x0 / dm)
    params = op.params() + ([shift] if shift.requires_grad else [])
    batch = ctx.batch[k] if ctx.batch is not None else None
    pc = None
    if precond == "twolevel" and k == 0:
        with torch.no_grad():
            q = sh / dm if free is None else sh / dm * free                      # constants of u in the scaled frame
            pc = two_level_preconditioner(ctx, k, matvec, q.expand(-1, dm.shape[1], 1).contiguous())
    w, res = pcg_solve_implicit(matvec, rhs.contiguous(), pc, params=params, iters=int(iters), tol=float(tol),
                                batch=batch, num_graphs=ctx.num_graphs, early_exit=True)
    if free is not None:                         # w = 0 on fixed cells: masking keeps the adjoint rhs in range(P)
        w = w * free
    y_sym = w * dm
    if x0 is not None:
        y_sym = x0 + y_sym
    rel = res[-1].detach()
    if x0 is not None:                                                        # residual relative to |x|
        with torch.no_grad():
            nr = sample_norm2(ctx, k, rhs)
            nx = sample_norm2(ctx, k, x * sh * dm if free is None else x * sh * dm * free)
            rel = rel * torch.sqrt(nr / torch.where(nx > 0, nx, torch.ones_like(nx)))
    with torch.no_grad():                                                     # iterations of the slowest sample
        n_it = int((res[1:] != res[:-1]).sum(0).max().item()) if res.shape[0] > 1 else 0
    return y_sym / sh, y_sym, rel, n_it


def sample_norm2(ctx: LayerContext, k: int, v: Tensor) -> Tensor:
    """Per-sample squared Frobenius norm over (cells, channels): ``(1|G, B)``."""
    n = torch.linalg.vecdot(v, v)
    return n.sum(0, keepdim=True) if ctx.batch is None else ops.segment_sum(n, ctx.batch[k], ctx.num_graphs)


# --------------------------------------------------------------------------------------------------------------
# gate
# --------------------------------------------------------------------------------------------------------------
class RadialGate(nn.Module):
    """Radial norm gate with scalar-gain RMS normalisation and the residual update (O(C)-equivariant).

    ``gate='norm'``: ``x + gamma * sigmoid(MLP(log1p(rms(m)))) * m / rms(m)`` with ``rms`` over channels per cell
    (``MLP: 1 -> hidden -> SiLU -> 1``).  The gate is computed from the rms *before* normalisation, so the update
    magnitude is a learned bounded function of the message magnitude and its direction is the message direction.

    ``gate='relu'`` (ablation, breaks O(C) and orientation symmetry): ``x + gamma * relu(m) / rms(relu(m))``.

    ``gate='none'`` (linear models, e.g. the solver preset): ``x + gamma * m`` (no nonlinearity, no normalisation).

    Args:
        mode: ``'norm' | 'relu' | 'none'``.
        hidden: hidden width of the gate MLP.
    """

    def __init__(self, mode: str = "norm", hidden: int = 16) -> None:
        super().__init__()
        if mode not in _GATES:
            raise ValueError(f"gate must be one of {_GATES}, got {mode!r}")
        self.mode = mode
        self.mlp = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, 1)) if mode == "norm" else None
        self.gain = nn.Parameter(torch.ones(()))

    def scale(self, m: Tensor) -> Tensor:
        """Per-cell factor ``a`` ``(n, B, 1)`` such that the update is ``m ⊙ a``."""
        rms = torch.sqrt(mean_square(m).unsqueeze(-1) + RMS_EPS)          # (n, B, 1)
        a = self.gain / rms
        if self.mlp is not None:
            l1, l2 = self.mlp[0], self.mlp[2]
            z = mlp2(torch.log1p(rms), l1.weight, l1.bias, l2.weight, l2.bias)
            a = a * torch.sigmoid(z.to(rms.dtype))
        return a

    def forward(self, x: Tensor, m: Tensor) -> Tensor:
        """Residual update ``x + m ⊙ a(m)``: ``(n, B, C), (n, B, C) -> (n, B, C)``."""
        if self.mode == "none":
            return x + self.gain * m
        if self.mode == "relu":
            m = F.relu(m)
        return residual_scale(x, m, self.scale(m))


# --------------------------------------------------------------------------------------------------------------
# layer
# --------------------------------------------------------------------------------------------------------------
class SharedCoboundaries:
    """Lazily computed, memoised first coboundary products of one layer input.

    ``up(k) = (d_k diag(s_up)) x_k`` ``(n_{k+1}, B, C)`` and ``dn(k) = (d_{k-1}^T diag(s_dn)) x_k``
    ``(n_{k-1}, B, C)`` (raw coboundaries for Jacobi scaling); each is the first sparse product of the
    corresponding block and also feeds the metric invariants.
    """

    def __init__(self, x: list[Tensor], ctx: LayerContext) -> None:
        self.x, self.ctx = x, ctx
        self._up: dict[int, Tensor] = {}
        self._dn: dict[int, Tensor] = {}
        self.feats: dict[int, Tensor] = {}                                  # metric invariants per degree

    def up(self, k: int) -> Tensor:
        """First coboundary of the up block of degree ``k``."""
        if k not in self._up:
            g = self.ctx.blocks[(k, True)]
            self._up[k] = spmm_ad(g.A, g.AT, self.x[k])
        return self._up[k]

    def dn(self, k: int) -> Tensor:
        """First coboundary of the down block of degree ``k``."""
        if k not in self._dn:
            g = self.ctx.blocks[(k, False)]
            self._dn[k] = spmm_ad(g.A, g.AT, self.x[k])
        return self._dn[k]


class RHMPLayer(nn.Module):
    """One RHMP message-passing layer on a complex of top degree ``top``.

    Args:
        top: top degree ``K``.
        geo_dims: ``{k: G_k}``.
        even_dims: ``{k: E_k}`` even raw input columns per degree (fed to the metric heads).
        C: channels.
        poly_order: ``P >= 1``.
        metric_hidden: hidden width of the metric MLPs.
        log_range: bound ``a`` of the log-metric.
        tie_metrics: one metric per degree (up uses ``H_m``, down uses ``1/H_m``) or separate up/down heads.
        scaling: ``'dec' | 'jacobi' | 'none'``.
        cross: cross-degree transport terms.
        gate: ``'norm' | 'relu' | 'none'``.
        identity_metric: every metric ``H = 1`` (ablation of the learned metric; with ``scaling='dec'`` the DEC stars
            of the block's own degree remain through the scaling ``S``; see ``RHMPConfig.identity_metric``).
        fused: use the fused polynomial kernels (``False`` = plain autograd reference path).
        kind: ``'poly'`` (polynomial filters) or ``'resolvent'`` (self term ``(I + tau_up L_up + tau_dn L_dn)^{-1} x``
            by batched CG, DESIGN §9.2; learnable ``tau = exp(log_tau) <= TAU_MAX`` per degree and block, init 1).
        resolvent_iters: CG iterations of resolvent layers.
        resolvent_grad: ``'implicit'`` (adjoint solve) or ``'unrolled'`` (checkpointed unrolled iterations).
        metric_type: ``'diag'`` or ``'tensor'`` (Whitney material-tensor metric for the up blocks and cross-up terms of
            the intermediate degrees, DESIGN §9.1).
        resolvent_warm_start: start the CG of a resolvent layer from the previous resolvent layer's solution of the
            same degree (stored in the forward context) when the shapes match.
        learn_metric: ``False`` removes the metric heads: ``H_m = star_m exp(ref_m)`` exactly (tensor metric:
            ``b = 1, a = 0``), the fixed DEC/FEEC operators.  ``identity_metric`` takes precedence.
        resolvent_normalize: ``False`` makes resolvent layers use the un-normalised physical operator
            ``(I + tau L^2 Delta_H)^{-1}`` (``L^D`` = domain measure) instead of ``(I + tau L_hat)^{-1}``: the metric's
            global magnitude then matters (the ``beta``-normalised ``L_hat`` is invariant to ``H -> c H``).
        solve_iters, solve_tol, solve_bc: ``kind='solve'``: CG budget and boundary condition (``'dirichlet'``: fixed
            ``K.meta['dirichlet'][k]`` or ``K.boundary[k]`` cells; ``'none'``; ``'neumann'``: no fixed cells and
            mean-free vertex sources, see :func:`physical_solve`).  A solve layer computes
            ``y_k = (Delta_H + lam / L^2)^{-1} x_k`` with the un-normalised metric Hodge Laplacian (physical units),
            ``lam = softplus(log_lam)`` (init 1e-3), and updates ``x_k <- gate(x_k, y_k - x_k [+ cross terms])``;
            with ``gate='none'`` (gain 1) this is ``x_k <- y_k``.
        tensor_param: ``'full'`` or ``'cone'`` parameterisation of the tensor-metric heads
            (:class:`rhmp.metric.TensorMetricHead`).
        solve_precond: ``'none'`` (Jacobi) or ``'twolevel'`` (Jacobi plus a coarse correction for vertex solves;
            :func:`physical_solve`).
    """

    def __init__(self, top: int, geo_dims: dict[int, int], even_dims: dict[int, int], *, C: int, poly_order: int = 2,
                 metric_hidden: int = 32, log_range: float = 2.0, tie_metrics: bool = True, scaling: str = "dec",
                 cross: bool = True, gate: str = "norm", identity_metric: bool = False, fused: bool = True,
                 kind: str = "poly", resolvent_iters: int = 20, resolvent_grad: str = "implicit",
                 metric_type: str = "diag", resolvent_warm_start: bool = True, learn_metric: bool = True,
                 resolvent_normalize: bool = True, solve_iters: int = 64, solve_tol: float = 1e-6,
                 solve_bc: str = "dirichlet", tensor_param: str = "full", solve_precond: str = "none") -> None:
        super().__init__()
        if top < 1:
            raise ValueError("RHMPLayer needs a complex with at least one edge degree (top >= 1)")
        if poly_order < 1:
            raise ValueError("poly_order must be >= 1")
        if scaling not in _SCALINGS:
            raise ValueError(f"scaling must be one of {_SCALINGS}, got {scaling!r}")
        if kind not in ("poly", "resolvent", "solve"):
            raise ValueError(f"layer kind must be 'poly', 'resolvent' or 'solve', got {kind!r}")
        if solve_bc not in _SOLVE_BCS:
            raise ValueError(f"solve_bc must be one of {_SOLVE_BCS}, got {solve_bc!r}")
        if solve_precond not in ("none", "twolevel"):
            raise ValueError(f"solve_precond must be 'none' or 'twolevel', got {solve_precond!r}")
        if (kind == "solve" or (kind == "resolvent" and not resolvent_normalize)) and scaling != "dec":
            raise ValueError("solve layers and un-normalised resolvent layers need scaling='dec'")
        if resolvent_grad not in _RESOLVENT_GRADS:
            raise ValueError(f"resolvent_grad must be one of {_RESOLVENT_GRADS}, got {resolvent_grad!r}")
        if metric_type not in ("diag", "tensor"):
            raise ValueError(f"metric_type must be 'diag' or 'tensor', got {metric_type!r}")
        if metric_type == "tensor" and (scaling == "jacobi" or identity_metric):
            raise ValueError("metric_type='tensor' needs scaling 'dec' or 'none' and identity_metric=False")
        self.top = int(top)
        self.C = int(C)
        self.poly_order = int(poly_order)
        self.tie = bool(tie_metrics)
        self.scaling = scaling
        self.cross = bool(cross)
        self.identity_metric = bool(identity_metric)
        self.fused = bool(fused)
        self.kind = kind
        self.resolvent_iters = int(resolvent_iters)
        self.resolvent_grad = resolvent_grad
        self.resolvent_warm_start = bool(resolvent_warm_start)
        self.resolvent_normalize = bool(resolvent_normalize)
        self.solve_iters, self.solve_tol, self.solve_bc = int(solve_iters), float(solve_tol), solve_bc
        self.solve_precond = solve_precond
        self.metric_type = metric_type
        self.learn_metric = bool(learn_metric)
        if tensor_param not in ("cone", "full"):
            raise ValueError(f"tensor_param must be 'cone' or 'full', got {tensor_param!r}")
        self.tensor_param = tensor_param
        # degrees whose metric is a Whitney material tensor in tensor mode (up blocks + cross-up terms)
        self.tensor_degrees = list(range(1, self.top)) if metric_type == "tensor" else []
        self.heads = nn.ModuleDict()
        if not self.identity_metric and self.learn_metric:
            for m in range(self.top + 1):
                args = (geo_dims[m], even_dims.get(m, 0), metric_feature_dim(m, self.top), metric_hidden, log_range)
                if self.tie:
                    self.heads[f"H{m}"] = MetricHead(*args)
                else:
                    if m >= 1:
                        self.heads[f"Hu{m}"] = MetricHead(*args)
                    if m < self.top:
                        self.heads[f"Hd{m}"] = MetricHead(*args)
        self.theads = nn.ModuleDict()
        for km in (self.tensor_degrees if self.learn_metric else []):
            top_ = self.top
            in_top = geo_dims[top_] + metric_feature_dim(top_, top_) + even_dims.get(top_, 0)
            in_edge = geo_dims[1] + metric_feature_dim(1, top_) + even_dims.get(1, 0)
            self.theads[f"T{km}"] = TensorMetricHead(in_top, in_edge, metric_hidden, log_range, tensor_param)
        init = torch.zeros(self.poly_order)
        init[0] = 1.0
        self.coef = nn.ParameterDict()
        self.log_tau = nn.ParameterDict()
        self.log_lam = nn.ParameterDict()
        self.wcross = nn.ParameterDict()
        for k in range(self.top + 1):
            keys = ([f"up{k}"] if k < self.top else []) + ([f"dn{k}"] if k > 0 else [])
            for key in keys:
                if self.kind == "poly":
                    self.coef[key] = nn.Parameter(init.clone())
                elif self.kind == "resolvent":
                    self.log_tau[key] = nn.Parameter(torch.zeros(()))
                if self.cross:
                    self.wcross[key] = nn.Parameter(torch.tensor(0.5))
            if self.kind == "solve":
                self.log_lam[str(k)] = nn.Parameter(torch.tensor(math.log(math.expm1(1e-3))))   # lam = 1e-3
        self.gates = nn.ModuleList([RadialGate(gate) for _ in range(self.top + 1)])
        self.last_diagnostics: dict[str, Tensor] = {}

    # ---------------------------------------------------------------------------------------------------------
    def _metric_features(self, m: int, x: list[Tensor], q: SharedCoboundaries, ctx: LayerContext) -> Tensor:
        """O(C)-, E(n)-, orientation- and permutation-invariant per-sample features, ``(n_m, B, F_m)`` (memoised)."""
        if m in q.feats:
            return q.feats[m]
        q.feats[m] = self._metric_features_impl(m, x, q, ctx)
        return q.feats[m]

    def _metric_features_impl(self, m: int, x: list[Tensor], q: SharedCoboundaries, ctx: LayerContext) -> Tensor:
        xm = x[m]
        C = xm.shape[-1]
        nx = torch.log1p(mean_square(xm))                                   # (n_m, B)
        feats = [nx]
        if m >= 1:
            qm = q.up(m - 1)                                                # d_{m-1} (s ⊙ x_{m-1})
            feats.append(torch.log1p(mean_square(qm)))
            feats.append(torch.asinh(torch.linalg.vecdot(xm, qm) / C))
        if m < self.top:
            feats.append(torch.log1p(mean_square(q.dn(m + 1))))             # d_m^T (s' ⊙ x_{m+1})
        feats.append(ctx.cell_mean(nx, m).expand_as(nx))
        return torch.stack(feats, dim=-1)

    def metrics(self, x: list[Tensor], q: SharedCoboundaries, ctx: LayerContext, need_up: list[int] | None = None,
                need_dn: list[int] | None = None) -> tuple[list[Tensor | None], list[Tensor | None]]:
        """Log-metrics of the blocks.

        Args:
            x: layer input features.
            q: shared coboundaries of ``x``.
            ctx: context.
            need_up: degrees ``m`` whose up-usage metric is needed (default: ``1..K``).
            need_dn: degrees ``m`` whose down-usage metric is needed (default: ``0..K-1``).

        Returns:
            ``(logH_up, logH_dn)``: ``logH_up[m]`` ``(n_m, B)`` is used by the up block of degree ``m-1`` and
            ``logH_dn[m]`` by the down block of degree ``m+1`` (``None`` where not requested).
        """
        top = self.top
        need_up = list(range(1, top + 1)) if need_up is None else list(need_up)
        need_dn = list(range(top)) if need_dn is None else list(need_dn)
        logH_up: list[Tensor | None] = [None] * (top + 1)
        logH_dn: list[Tensor | None] = [None] * (top + 1)
        diag = self.last_diagnostics if ctx.record else None
        for m in range(top + 1):
            want_up, want_dn = m in need_up, m in need_dn
            if not (want_up or want_dn):
                continue
            if self.identity_metric:
                z = x[m].new_zeros(x[m].shape[0], x[m].shape[1])
                logH_up[m] = z if want_up else None
                logH_dn[m] = z if want_dn else None
                continue
            ls = ctx.log_star[m].unsqueeze(1)                               # (n_m, 1)
            if ctx.ref is not None and ctx.ref[m] is not None:              # fixed material reference (even input)
                ls = ls + ctx.ref[m]                                        # (n_m, B)
            if not self.learn_metric:                                       # fixed DEC metric: H = star exp(ref)
                lh = ls.expand(x[m].shape[0], x[m].shape[1]).clamp(-LOG_H_MAX, LOG_H_MAX)
                logH_up[m] = lh if want_up else None
                logH_dn[m] = -lh if want_dn else None
                if diag is not None:
                    self._record_metric(diag, f"H{m}", torch.zeros_like(lh), lh, m, ctx, 1.0)
                continue
            feats = self._metric_features(m, x, q, ctx)
            if self.tie:
                phi = self.heads[f"H{m}"](ctx.geo[m], ctx.even[m], feats)
                lh = (ls + phi).clamp(-LOG_H_MAX, LOG_H_MAX)
                logH_up[m] = lh if want_up else None
                logH_dn[m] = -lh if want_dn else None
                if diag is not None:
                    self._record_metric(diag, f"H{m}", phi, lh, m, ctx, self.heads[f"H{m}"].log_range)
            else:
                if want_up:
                    phi = self.heads[f"Hu{m}"](ctx.geo[m], ctx.even[m], feats)
                    logH_up[m] = (ls + phi).clamp(-LOG_H_MAX, LOG_H_MAX)
                    if diag is not None:
                        self._record_metric(diag, f"H{m}.up", phi, logH_up[m], m, ctx, self.heads[f"Hu{m}"].log_range)
                if want_dn:
                    phi = self.heads[f"Hd{m}"](ctx.geo[m], ctx.even[m], feats)
                    logH_dn[m] = (phi - ls).clamp(-LOG_H_MAX, LOG_H_MAX)
                    if diag is not None:
                        self._record_metric(diag, f"H{m}.down", phi, logH_dn[m], m, ctx,
                                            self.heads[f"Hd{m}"].log_range)
        return logH_up, logH_dn

    @staticmethod
    @torch.no_grad()
    def _record_metric(diag: dict, name: str, phi: Tensor, logH: Tensor, m: int, ctx: LayerContext,
                       log_range: float) -> None:
        """Store ``[mean, std, min, max]`` of the bounded part ``phi``, the log condition numbers of ``exp(phi)`` and of
        ``H`` (max over samples) and the fractions of saturated cells ``|tanh(.)| = |phi| / a > 0.99`` (clamp) and
        ``> 0.95`` (sat).  ``RHMP.diagnostics`` turns the entry ``name`` (``'H{m}'``, ``'H{m}.up'``, ``'H{m}.down'``)
        into ``layer{l}.H{m}.<stat>`` keys."""
        phi = phi.detach().float()
        lh = logH.detach().float()
        logcond_learned = (ctx.cell_max(phi, m) + ctx.cell_max(-phi, m)).max()
        logcond_total = (ctx.cell_max(lh, m) + ctx.cell_max(-lh, m)).max()
        clamp = (phi.abs() > 0.99 * log_range).float().mean()
        sat = (phi.abs() > 0.95 * log_range).float().mean()
        diag[name] = torch.stack([phi.mean(), phi.std(correction=0), phi.min(), phi.max(), logcond_learned,
                                  logcond_total, clamp, sat])

    def _descriptors(self, m: int, x: list[Tensor], q: SharedCoboundaries, ctx: LayerContext) -> Tensor:
        """``[geo_m, invariants_m, even_m]`` per cell and sample, ``(n_m, B, P_m)``."""
        f = self._metric_features(m, x, q, ctx)
        parts = [ctx.geo[m].unsqueeze(1).expand(-1, f.shape[1], -1), f]
        if ctx.even[m] is not None:
            parts.append(ctx.even[m])
        return torch.cat(parts, dim=-1)

    def tensor_metric(self, km: int, x: list[Tensor], q: SharedCoboundaries, ctx: LayerContext) -> "TensorMetric":
        """The Galerkin material-tensor metric of degree ``km`` (:class:`TensorMetric`).

        Learned full tensors (``tensor_param='full'``): ``sigma_f = b_f expm(sum_j s_j t_j t_j^T)`` (any SPD tensor;
        ``s = log_range tanh(.)``, zero-init -> ``b_f I``) assembled into explicit Galerkin element matrices; learned cone
        tensors: ``b_f I + sum_j a_j t_j t_j^T`` with ``a >= 0`` (``rhmp.dec`` blocks); fixed metrics
        (``learn_metric=False``): ``b = 1, a = 0``.  Material references scale ``sigma_f`` (see
        ``RHMPConfig.metric_reference``).
        """
        whitney = getattr(ctx.K, "whitney", None)
        if not whitney or km not in whitney:
            raise ValueError("metric_type='tensor' needs Whitney blocks (K.whitney); they exist for triangle and "
                             "tetrahedral complexes, not for polygon meshes")
        W = whitney[km]
        de = W["dir_edges"]                                                   # (n_top, m)
        dt = ctx.dtype
        B = x[0].shape[1]
        full = self.learn_metric and self.tensor_param == "full"
        if self.learn_metric:
            psi_top = self._descriptors(self.top, x, q, ctx)                  # (n_top, B, P_top)
            psi_1 = self._descriptors(1, x, q, ctx)                           # (n_1, B, P_1)
            psi_e = psi_1.index_select(0, de.reshape(-1)).view(de.shape[0], de.shape[1], *psi_1.shape[1:])
            head = self.theads[f"T{km}"]
            b, a = head(psi_top, psi_e)
            b, a = b.to(dt), a.to(dt)
            log_range = head.log_range
        else:                                                                 # fixed Galerkin/Whitney star
            b = x[0].new_ones(de.shape[0], B)
            a = x[0].new_zeros(de.shape[0], B, de.shape[1])
            log_range = 1.0
        if full:                                                              # sigma_f = b expm(S_f)
            t = W["t"].to(dt)
            sig = b[..., None, None] * _expm_sym(torch.einsum("fbj,fjd,fje->fbde", a, t, t))
            if ctx.record:
                self._record_tensor(km, sig, b, a, ctx, log_range, "s")
            tm = TensorMetric(ctx, km, blocks=galerkin_blocks(galerkin_geometry(ctx.K, km, dt), sig), sigma=sig)
        else:
            if ctx.record:
                self._record_tensor(km, None, b, a, ctx, log_range, "a")
            tm = TensorMetric(ctx, km, b=b, a=a)
        ref = ctx.ref[km] if ctx.ref is not None else None
        if ref is not None:                                                   # material reference on the km-cells
            cells = W["cells"]                                                # (n_top, m_k)
            rc = ref.index_select(0, cells.reshape(-1)).view(cells.shape[0], cells.shape[1], -1).mean(1)
            if full or km != 1:
                tm = tm.scale(torch.exp(rc))                                  # scalar scaling keeps sigma_f's shape
            else:                                                             # cone edge metric: per direction
                rd = ref.index_select(0, de.reshape(-1)).view(de.shape[0], de.shape[1], -1)
                tm = TensorMetric(ctx, km, b=tm.b * torch.exp(rc), a=tm.a * torch.exp(rd).permute(0, 2, 1))
        rt = ctx.ref[self.top] if (ctx.ref is not None and km != self.top) else None
        if rt is not None:                                                    # per-cell material on the top cells
            tm = tm.scale(torch.exp(rt))
        return tm

    @torch.no_grad()
    def _record_tensor(self, km: int, sig: Tensor | None, b: Tensor, a: Tensor, ctx: LayerContext, log_range: float,
                       kind: str) -> None:
        """``log b`` stats, the learned coefficients, the anisotropy (eigenvalue ratio) of the learned ``sigma_f`` and
        the fractions of saturated heads (``|tanh| > 0.99``)."""
        if sig is None:
            sig = TensorMetric(ctx, km, b=b, a=a).tensor()
        sig = sig.double()
        if sig.shape[-1] > self.top:                                          # surfaces: restrict to the tangent plane
            t = ctx.K.whitney[km]["t"].to(torch.float64)
            Qt = torch.linalg.qr(t[:, :self.top].transpose(1, 2))[0]          # (n_top, D, top) orthonormal
            sig = Qt.transpose(1, 2).unsqueeze(1) @ sig @ Qt.unsqueeze(1)
        ev = _sym_eigvals(sig)
        ratio = (ev[..., -1] / ev[..., 0].clamp_min(1e-300)).float()
        d = self.last_diagnostics
        p = f"H{km}.tensor_"
        d[p + "logb_mean"] = torch.log(b).float().mean()
        d[p + "aniso_mean"] = ratio.mean()
        d[p + "aniso_max"] = ratio.max()
        d[p + "clamp_b"] = (torch.log(b).abs() > 0.99 * log_range).float().mean()
        if kind == "s":
            d[p + "s_absmax"] = a.float().abs().max()
            d[p + "clamp_s"] = (a.abs() > 0.99 * log_range).float().mean()
        else:
            d[p + "a_mean"] = a.float().mean()
            d[p + "a_max"] = a.float().max()
            d[p + "clamp_a"] = (a > 0.99 * math.exp(log_range)).float().mean()

    def tensor_block(self, k: int, x: list[Tensor], q1: Tensor, tm: "TensorMetric", ctx: LayerContext) -> Tensor:
        """Up block of degree ``k`` with the Whitney tensor metric ``H_{k+1}(b, a)`` (DESIGN §9.1).

        ``sum_p c_p L^p x_k + w T x_{k+1}`` with ``L = A^T H A / beta`` and ``T = A^T H / (sqrt(beta_unit) rho)``
        (:func:`tensor_operands`; scaling folded into ``A``).  One metric application per ``L`` application: the
        polynomial and the cross term share the final ``A^T H (.)``.
        """
        op = tensor_operands(ctx, k, tm)
        key = f"up{k}"
        c = self.coef[key]
        qs = [q1]
        for _ in range(1, c.shape[0]):
            y = spmm_ad(op.AT, op.A, _whitney(ctx, op, qs[-1]) * op.inv_beta)
            qs.append(spmm_ad(op.A, op.AT, y))
        z = qs[0] * (c[0] * op.inv_beta)
        for i in range(1, c.shape[0]):
            z = z + qs[i] * (c[i] * op.inv_beta)
        if self.cross:
            z = z + x[op.km] * (self.wcross[key] * op.cross_scale)
        if ctx.record:
            self.last_diagnostics[f"beta_up{k}"] = op.beta.detach().float().mean()
        return spmm_ad(op.AT, op.A, _whitney(ctx, op, z))

    def block(self, up: bool, k: int, x: list[Tensor], q1: Tensor | None, log_H: Tensor, ctx: LayerContext) -> Tensor:
        """Message of the up (``up=True``) or down block of degree ``k``, including its cross term.

        Args:
            up: block kind.
            k: degree.
            x: layer input features per degree.
            q1: shared first coboundary ``A (s ⊙ x_k)`` (``None`` for Jacobi scaling).
            log_H: block metric ``(m, B)``.
            ctx: context.

        Returns:
            ``(n_k, B, C)``.
        """
        op = block_operands(ctx, k, up, log_H)
        key = f"up{k}" if up else f"dn{k}"
        h = torch.exp(op.log_h).unsqueeze(-1)                               # (m, B, 1)
        u = X = None
        if self.cross:
            u = (self.wcross[key] * torch.exp(0.5 * op.log_h)).unsqueeze(-1)
            X = x[op.nb]
        c = self.coef[key]
        if ctx.record:
            self.last_diagnostics[f"beta_{'up' if up else 'down'}{k}"] = op.beta.detach().float().mean()
        if op.s is not None:                                                # Jacobi: explicit per-sample scaling
            return poly_block_reference(x[k], h, c, u, X, op)
        if self.fused:
            return poly_block(q1, h, c, u, X, op)
        return poly_block_reference(x[k], h, c, u, X, op, q1=q1)

    def forward(self, x: list[Tensor], ctx: LayerContext, degrees: list[int] | None = None) -> list[Tensor]:
        """One synchronous update of all degrees (or only of ``degrees``).

        Args:
            x: ``[x_0, .., x_K]`` with ``x_k`` ``(n_k, B, C)``.
            ctx: forward context.
            degrees: degrees to update (default: all).  Features of other degrees are returned unchanged; the
                model uses this in its last layer, where only the readout degree matters.

        Returns:
            Updated features, same shapes.
        """
        top = self.top
        if len(x) != top + 1:
            raise ValueError(f"expected {top + 1} feature tensors, got {len(x)}")
        degrees = list(range(top + 1)) if degrees is None else [k for k in range(top + 1) if k in degrees]
        if ctx.record:
            self.last_diagnostics = {}
        q = SharedCoboundaries(x, ctx)
        need_up = [k + 1 for k in degrees if k < top]
        need_dn = [k - 1 for k in degrees if k > 0]
        tensor_up = [k for k in degrees if (k + 1) in self.tensor_degrees]  # up blocks with a Whitney metric
        need_up = [m for m in need_up if (m - 1) not in tensor_up]
        logH_up, logH_dn = self.metrics(x, q, ctx, need_up, need_dn)
        tens = {k + 1: self.tensor_metric(k + 1, x, q, ctx) for k in tensor_up}
        if self.kind == "solve":
            return self._solve_forward(x, ctx, degrees, logH_up, logH_dn, tens)
        if self.kind == "resolvent":
            if tens and self.resolvent_normalize:
                raise NotImplementedError("normalised resolvent layers use diagonal metrics; set metric_type='diag' "
                                          "or resolvent_normalize=False")
            return self._resolvent_forward(x, ctx, degrees, logH_up, logH_dn, tens)
        jacobi = self.scaling == "jacobi"
        out = list(x)
        for k in degrees:
            msg = None
            if k in tensor_up:
                msg = self.tensor_block(k, x, q.up(k), tens[k + 1], ctx)
            elif k < top:
                msg = self.block(True, k, x, None if jacobi else q.up(k), logH_up[k + 1], ctx)
            if k > 0:
                md = self.block(False, k, x, None if jacobi else q.dn(k), logH_dn[k - 1], ctx)
                msg = md if msg is None else msg + md
            out[k] = self.gates[k](x[k], msg)
        return out

    @torch.no_grad()
    def metric_fields(self, x: list[Tensor], ctx: LayerContext) -> dict:
        """The metrics this layer would use on input ``x`` (no update, no grad).

        Returns:
            ``{'log_ratio': {m: (n_m, B)}, 'phi': {m: (n_m, B)}, 'tensor': {km: (b, a)}, 'sigma': {km: (n_top, B, D,
            D)}}`` where ``log_ratio`` is
            ``log(H_m / star_m)`` of the up-usage (tied) metric including the reference offset (up to a per-sample
            constant: stars are normalised per sample), ``phi`` its learned bounded part, and ``tensor`` the material
            tensor parameters ``b`` ``(n_top, B)``, ``a`` ``(n_top, B, m)`` (after the reference scaling).  Untied
            metrics report the up heads for ``m >= 1``; for ``m = 0`` (down usage only) ``log_ratio = -log(H_down
            star_0)``, i.e. the equivalent tied metric (``phi = -phi_down``).
        """
        q = SharedCoboundaries(x, ctx)
        top = self.top
        logH_up, logH_dn = self.metrics(x, q, ctx, list(range(1, top + 1)), list(range(top)))
        out: dict = {"log_ratio": {}, "phi": {}, "tensor": {}, "sigma": {}}
        for m in range(top + 1):
            ref = ctx.ref[m] if (ctx.ref is not None and ctx.ref[m] is not None) else 0.0
            if logH_up[m] is not None:
                lr = logH_up[m] - ctx.log_star[m].unsqueeze(1)
            else:                                                           # down-only degree: H_down = 1 / H
                lr = -(logH_dn[m] + ctx.log_star[m].unsqueeze(1))
            out["log_ratio"][m] = lr
            out["phi"][m] = lr - ref
        for km in self.tensor_degrees:
            tm = self.tensor_metric(km, x, q, ctx)
            out["tensor"][km] = tm.as_dyads()
            out["sigma"][km] = tm.tensor()
        return out

    def tau(self, key: str) -> Tensor:
        """Resolvent time step ``tau = exp(log_tau) <= TAU_MAX`` of block ``key`` (``'up{k}'`` / ``'dn{k}'``)."""
        return torch.exp(self.log_tau[key].clamp(max=math.log(TAU_MAX)))

    def _resolvent_forward(self, x: list[Tensor], ctx: LayerContext, degrees: list[int],
                           logH_up: list[Tensor | None], logH_dn: list[Tensor | None],
                           tens: dict | None = None) -> list[Tensor]:
        """Resolvent layer: ``m_k = (I + tau_up L_up + tau_dn L_dn)^{-1} x_k + w_cu T_up x_{k+1} + w_cd T_dn x_{k-1}``
        (``resolvent_normalize=False``: ``(I + L^2 (tau_up Delta_up + tau_dn Delta_dn))^{-1} x_k`` with the physical
        metric Hodge Laplacian)."""
        top = self.top
        out = list(x)
        for k in degrees:
            op_up = block_operands(ctx, k, True, logH_up[k + 1]) if (k < top and (k + 1) not in (tens or {})) else None
            op_dn = block_operands(ctx, k, False, logH_dn[k - 1]) if k > 0 else None
            tau_up = self.tau(f"up{k}") if k < top else None
            tau_dn = self.tau(f"dn{k}") if k > 0 else None
            store = ctx.resolvent_prev
            if not self.resolvent_normalize:                                  # physical operator, CG in sym frame
                L2 = torch.exp((2.0 / top) * ctx.log_vol)                    # domain length scale squared per graph

                def l2_on(deg: int) -> Tensor:                               # (n_deg|1, 1)
                    return (L2 if ctx.batch is None else L2[ctx.batch[deg]]).view(-1, 1)

                w_up = None if tau_up is None else tau_up * l2_on(top if (k + 1) in (tens or {}) else k + 1)
                w_dn = None if tau_dn is None else tau_dn * l2_on(k - 1)
                op = physical_hodge(ctx, k, logH_up, logH_dn, tens, w_up=w_up, w_dn=w_dn)
                key = ("phys", k)
                prev = store.get(key) if (self.resolvent_warm_start and store is not None) else None
                if prev is not None and prev.shape != x[k].shape:
                    prev = None
                one = torch.ones((), dtype=ctx.dtype, device=x[k].device).view(1, 1, 1)
                msg, y_sym, rel, _ = physical_solve(ctx, k, x[k], op, one, None, self.resolvent_iters, 0.0, x0=prev)
                if store is not None:
                    store[key] = y_sym.detach()
            else:
                prev = store.get(k) if (self.resolvent_warm_start and store is not None) else None
                if prev is not None and prev.shape != x[k].shape:
                    prev = None
                msg, rel = resolvent_apply(ctx, k, x[k], op_up, op_dn, tau_up, tau_dn, self.resolvent_iters,
                                           self.resolvent_grad, x0=prev)
                if store is not None:
                    store[k] = msg.detach()
            if self.cross:
                if op_up is not None:
                    msg = msg + self.wcross[f"up{k}"] * apply_T(op_up, x[k + 1])
                if op_dn is not None:
                    msg = msg + self.wcross[f"dn{k}"] * apply_T(op_dn, x[k - 1])
            if ctx.record:
                d = self.last_diagnostics
                d[f"cg_res{k}"] = rel.detach().float().max()
                for key, op, t in ((f"up{k}", op_up, tau_up), (f"down{k}", op_dn, tau_dn)):
                    if t is not None:
                        d[f"tau_{key}"] = t.detach().float()
                    if op is not None:
                        d[f"beta_{key}"] = op.beta.detach().float().mean()
            out[k] = self.gates[k](x[k], msg)
        return out

    def _solve_forward(self, x: list[Tensor], ctx: LayerContext, degrees: list[int],
                       logH_up: list[Tensor | None], logH_dn: list[Tensor | None], tens: dict) -> list[Tensor]:
        """Solve layer: ``y_k = (Delta_H + lam / L^2)^{-1} x_k`` (Dirichlet mask, physical units, implicit CG
        gradients, warm start from the previous solve layer), ``x_k <- gate(x_k, y_k - x_k [+ cross terms])``."""
        top = self.top
        out = list(x)
        L2 = torch.exp((2.0 / top) * ctx.log_vol)                            # (S,)
        store = ctx.resolvent_prev
        for k in degrees:
            op = physical_hodge(ctx, k, logH_up, logH_dn, tens)
            lam = F.softplus(self.log_lam[str(k)])
            shift = (lam / (L2 if ctx.batch is None else L2[ctx.batch[k]])).view(-1, 1, 1)
            free = dirichlet_free(ctx, k, self.solve_bc)
            key = ("solve", k)
            prev = store.get(key) if (self.resolvent_warm_start and store is not None) else None
            if prev is not None and prev.shape != x[k].shape:
                prev = None
            y, y_sym, rel, n_it = physical_solve(ctx, k, x[k], op, shift, free, self.solve_iters, self.solve_tol,
                                                 x0=prev, precond=self.solve_precond,
                                                 mean_free=(self.solve_bc == "neumann" and k == 0))
            if store is not None:
                store[key] = y_sym.detach()
            msg = y - x[k]
            if self.cross:
                if k < top and (k + 1) not in tens:
                    msg = msg + self.wcross[f"up{k}"] * apply_T(block_operands(ctx, k, True, logH_up[k + 1]), x[k + 1])
                if k > 0:
                    msg = msg + self.wcross[f"dn{k}"] * apply_T(block_operands(ctx, k, False, logH_dn[k - 1]), x[k - 1])
            if ctx.record:
                d = self.last_diagnostics
                d[f"solve_res{k}"] = rel.detach().float().max()
                d[f"solve_it{k}"] = torch.tensor(float(n_it))
                d[f"lam{k}"] = lam.detach().float()
            out[k] = self.gates[k](x[k], msg)
        return out
