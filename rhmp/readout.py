"""Readouts from hidden cochains (DESIGN §3.5).

All readouts are E(n)-invariant (``node_vector``: E(n)-equivariant) and permutation-equivariant; they pick a
frame of the hidden channels, so O(C) is a property of the message-passing stack only.  Orientation behaviour:

* ``node_scalar``   ``MLP(x_0)`` -> ``(n_0, B, out_dim)`` (vertex scalars, even).
* ``cochain:k``     bias-free linear map of the odd features ``x_k`` -> ``(n_k, B, out_dim)`` (odd for k >= 1:
  fluxes, circulations, curvatures).
* ``even:k``        ``MLP([log1p(x_k^2), log1p(|x_k|^2/C), geo_k])`` (``x_0`` itself for k = 0) -> ``(n_k, B, out_dim)``
  (orientation-invariant cell scalars, e.g. magnitudes).
* ``grad``          ``E = d_0 phi`` with ``phi = MLP(x_0)`` (``readout_head='linear'``: ``phi = Lin_nobias(x_0)``): odd
  edge cochain ``(n_1, B, out_dim)``, exactly curl-free (``d_1 E = 0``) and invariant under ``phi -> phi + const``.
* ``curl``          ``F = d_1 a`` with ``a = Lin_nobias(x_1)``: odd face cochain ``(n_2, B, out_dim)``, exactly closed
  (``d_2 F = 0`` on volumes) and, in connection mode, exactly gauge invariant (every hidden feature is).
* ``div`` / ``div:k``  ``y = d_{k-1}^T b`` with ``b = Lin_nobias(x_k)`` (``k = K`` for ``div``): output on
  ``(k-1)``-cells ``(n_{k-1}, B, out_dim)``, exactly co-closed (``d_{k-2}^T y = 0`` for k >= 2); for ``k = 1`` it is
  a vertex divergence (even) whose sum over the vertices of every graph is exactly 0 (discrete Gauss law).
  The operators are purely combinatorial (no stars), so the constraints hold to rounding.
* ``mdiv`` / ``mdiv:k``  mass-weighted divergence ``y = S_{k-1}^{-1} d_{k-1}^T b`` with ``b = Lin_nobias(x_k)`` and
  ``S_{k-1} = exp(log_star_{k-1})`` the per-graph geometric-mean-normalised star of the output degree (outputs of
  order one).  For ``k = 1`` (``S_0`` = normalised lumped vertex mass) ``sum_i star0_i y_i = 0`` exactly per graph:
  exactly mass-conserving density increments on variable meshes.  Same orientation behaviour as ``div:k``.
* ``node_vector``   odd per-edge scalars ``w_e = Lin(x_1)_e`` (``out_dim`` vector fields) turned into vertex vectors,
  ``(n_0, B, out_dim * D)`` with ``D = K.pos.shape[1]``:

  - ``vector_mode='ls'`` (least squares, exact for constant fields when ``w_e = l_e t_e . v``)::

        M_i = sum_{e ∋ i} omega_e l_e^2 t_e t_e^T ,      b_i = sum_{e ∋ i} omega_e l_e w_e t_e
        v_i = (M_i^2 + mu_i^2 I)^{-1} M_i b_i ,         mu_i = 1e-4 * trace(M_i)

    i.e. the minimiser of ``sum_e omega_e (l_e t_e . v - w_e)^2`` computed as damped least squares on the normal
    equations ``M v = b``.  Unlike adding ``1e-6 * trace`` to ``M``, the damped form has *zero* response in
    rank-deficient directions (vertices whose edges are collinear, e.g. on straight polygon sides) instead of
    amplifying fp32 rounding by ``1e6`` there; its bias on well-posed vertices is ``O((mu / lambda_min)^2)``.
    The small ``D x D`` solves run in float64 with autocast disabled.  ``omega_e =
    softplus(MLP([log1p(x_1^2), log1p(|x_1|^2/C), geo_1])) > 0`` is a learned even weight (1 at initialisation).
    For surfaces in 3-D (top degree 2, ``D = 3``) the system is solved in the vertex tangent plane (orthonormal
    basis of the two smallest eigenvectors of ``sum_f a_f n_f n_f^T``, which does not depend on the face
    orientation), so the output is tangent and the solve stays well conditioned on curved surfaces (the full 3x3
    system is nearly singular in the normal direction there).
  - ``vector_mode='direct'`` (v1 style): ``v_i = sum_{e ∋ i} w_e t_e / deg_i``.

  Both are E(n)-equivariant (rotations, reflections, translations) and invariant to the edge-orientation
  convention (``w_e`` and ``t_e`` flip together).
"""
from __future__ import annotations

import math
import weakref
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import ops
from .layers import LayerContext, _msq, linear, mlp2, spmm_ad
from .lifting import GeoMLP

__all__ = [
    "NodeScalarReadout", "CochainReadout", "EvenCellReadout", "NodeVectorReadout", "GradReadout", "CurlReadout",
    "DivReadout", "MassDivReadout", "build_readout", "parse_readout", "readout_degrees", "vector_geometry", "ls_solve",
]

# Damping of the least-squares vector reconstruction, relative to trace(M): v = (M^2 + (LS_REG tr M)^2)^{-1} M b.
LS_REG = 1e-4
# softplus^{-1}(1): bias giving omega = 1 at initialisation.
_SOFTPLUS_INV_ONE = math.log(math.e - 1.0)


_READOUT_HELP = ("use 'node_scalar', 'node_vector', 'cochain:k', 'even:k', 'grad', 'curl', 'div', 'div:k', 'mdiv' or "
                 "'mdiv:k'")


def parse_readout(spec: str) -> tuple[str, int | None]:
    """Parse a readout name.

    Args:
        spec: ``'node_scalar' | 'node_vector' | 'cochain:k' | 'even:k' | 'grad' | 'curl' | 'div' | 'div:k' | 'mdiv' |
            'mdiv:k'``.

    Returns:
        ``(kind, k)``: ``k`` is the degree of the hidden features the readout reads (``None`` for ``'div'`` /
        ``'mdiv'``, meaning the top degree).
    """
    fixed = {"node_scalar": ("node_scalar", 0), "node_vector": ("node_vector", 1), "grad": ("grad", 0),
             "curl": ("curl", 1), "div": ("div", None), "mdiv": ("mdiv", None)}
    if spec in fixed:
        return fixed[spec]
    if ":" in spec:
        kind, _, k = spec.partition(":")
        if kind in ("cochain", "even", "div", "mdiv") and k.strip().isdigit():
            return kind, int(k)
    raise ValueError(f"unknown readout {spec!r}; {_READOUT_HELP}")


def readout_degrees(spec: str, top: int) -> tuple[int, int]:
    """``(feature_degree, output_degree)`` of a readout on a complex of top degree ``top``.

    Raises:
        ValueError: if the readout does not exist on such a complex.
    """
    kind, k = parse_readout(spec)
    if kind in ("div", "mdiv"):
        k = top if k is None else k
        if not 1 <= k <= top:
            raise ValueError(f"readout {spec!r} needs 1 <= k <= {top}")
        return k, k - 1
    if kind == "curl" and top < 2:
        raise ValueError("readout 'curl' needs faces (top degree >= 2)")
    if k > top:
        raise ValueError(f"readout {spec!r}: degree {k} exceeds the top degree {top}")
    out = {"node_scalar": 0, "node_vector": 0, "grad": 1, "curl": 2}.get(kind, k)
    return k, out


class NodeScalarReadout(nn.Module):
    """``MLP(x_0)``: ``Linear(C, C) -> SiLU -> Linear(C, out_dim)``, output ``(n_0, B, out_dim)``."""

    def __init__(self, C: int, out_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(C, C)
        self.fc2 = nn.Linear(C, out_dim)

    def forward(self, x: list[Tensor], ctx: LayerContext) -> Tensor:
        """``x[0]`` ``(n_0, B, C)`` -> ``(n_0, B, out_dim)``."""
        return mlp2(x[0], self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias)


class CochainReadout(nn.Module):
    """Bias-free linear map of ``x_k`` (odd for k >= 1): ``(n_k, B, C) -> (n_k, B, out_dim)``."""

    def __init__(self, C: int, out_dim: int, k: int) -> None:
        super().__init__()
        self.degree = int(k)
        self.lin = nn.Linear(C, out_dim, bias=False)

    def forward(self, x: list[Tensor], ctx: LayerContext) -> Tensor:
        """Readout on degree ``k``."""
        return linear(self.lin, x[self.degree])


class EvenCellReadout(nn.Module):
    """Orientation-invariant readout on degree ``k``: ``MLP([f(x_k), log1p(|x_k|^2/C), geo_k])``.

    ``f(x_k) = log1p(x_k^2)`` (per channel) for k >= 1 and ``f(x_0) = x_0``.
    """

    def __init__(self, C: int, out_dim: int, k: int, geo_dim: int) -> None:
        super().__init__()
        self.degree = int(k)
        self.mlp = GeoMLP(geo_dim, C + 1, C, out_dim)

    def forward(self, x: list[Tensor], ctx: LayerContext) -> Tensor:
        """``(n_k, B, C) -> (n_k, B, out_dim)``."""
        xk = x[self.degree]
        f = xk if self.degree == 0 else torch.log1p(xk.square())
        feats = torch.cat([f, torch.log1p(_msq(xk)).unsqueeze(-1)], dim=-1)
        return self.mlp(feats, ctx.geo[self.degree])


# --------------------------------------------------------------------------------------------------------------
# vector reconstruction
# --------------------------------------------------------------------------------------------------------------
_VEC_CACHE: "weakref.WeakKeyDictionary[Any, dict]" = weakref.WeakKeyDictionary()


@torch.no_grad()
def _tangent_bases(K: Any) -> Tensor:
    """Orthonormal tangent bases of a surface in 3-D, ``(n_0, 3, 2)`` (float64).

    Two smallest eigenvectors of ``N_i = sum_{f ∋ i} a_f n_f n_f^T`` (area-weighted face normal tensor; invariant
    under face orientation flips).  Polygons use their fan-triangulated vector area.
    """
    faces = K.cells[2]
    P = K.pos.to(torch.float64)
    n0 = P.shape[0]
    valid = faces >= 0
    f = faces.clamp_min(0)
    p0 = P[f[:, 0]]
    va = torch.zeros(faces.shape[0], 3, dtype=P.dtype, device=P.device)
    for j in range(1, faces.shape[1] - 1):
        cr = torch.linalg.cross(P[f[:, j]] - p0, P[f[:, j + 1]] - p0)
        va = va + torch.where(valid[:, j + 1, None], cr, torch.zeros_like(cr))
    va = 0.5 * va
    area = va.norm(dim=-1)
    Nf = va[:, :, None] * va[:, None, :] / area.clamp_min(1e-300)[:, None, None]
    N = torch.zeros(n0, 3, 3, dtype=P.dtype, device=P.device)
    for j in range(faces.shape[1]):
        w = valid[:, j]
        N.index_add_(0, f[w, j], Nf[w])
    _, evecs = torch.linalg.eigh(N)
    return evecs[..., :2].contiguous()


@torch.no_grad()
def vector_geometry(K: Any, dtype: torch.dtype, top: int) -> dict[str, Tensor | None]:
    """Geometry of the vector readout, cached per complex (weakly referenced), device and dtype.

    Args:
        K: ``CochainComplex``.
        dtype: feature dtype.
        top: top degree of the complex.

    Returns:
        dict with ``ev`` ``(n_1, D)`` edge vectors, ``t`` unit edge vectors, ``eet`` ``(n_1, 1, D*D)`` = ``ev ev^T``,
        ``deg`` ``(n_0, 1, 1)`` vertex valences (>= 1), ``E`` ``(n_0, 3, 2)`` tangent bases (surfaces in 3-D) or
        ``None``.
    """
    try:
        store = _VEC_CACHE.setdefault(K, {})
    except TypeError:  # complex not weak-referenceable: no caching
        store = {}
    key = (str(K.pos.device), dtype, top)
    if key not in store:
        ev = K.edge_vectors().to(dtype)
        n1, D = ev.shape
        ell = ev.norm(dim=-1, keepdim=True)
        t = ev / ell.clamp_min(torch.finfo(dtype).tiny)
        eet = (ev[:, :, None] * ev[:, None, :]).reshape(n1, 1, D * D).contiguous()
        ones = torch.ones(n1, 1, 1, dtype=dtype, device=ev.device)
        deg = ops.spmm(K.dT_abs[0], ones).clamp_min(1.0)
        E = _tangent_bases(K).to(dtype) if (D == 3 and top == 2) else None
        store[key] = dict(ev=ev, t=t, eet=eet, deg=deg, E=E)
    return store[key]


def _solve_spd(A: Tensor, r: Tensor, ok: Tensor) -> Tensor:
    """Closed-form solve of ``A v = r`` for SPD ``A`` ``(..., d, d)`` and ``r`` ``(..., V, d)`` (``d <= 3``)."""
    d = A.shape[-1]
    if d == 1:
        den = torch.where(ok, A[..., 0, 0], torch.ones_like(A[..., 0, 0]))
        return r / den[..., None, None]
    if d == 2:
        a, bb, c, e = A[..., 0, 0], A[..., 0, 1], A[..., 1, 0], A[..., 1, 1]
        det = torch.where(ok, a * e - bb * c, torch.ones_like(a))[..., None]
        r0, r1 = r[..., 0], r[..., 1]
        return torch.stack([(e[..., None] * r0 - bb[..., None] * r1) / det,
                            (a[..., None] * r1 - c[..., None] * r0) / det], dim=-1)
    if d == 3:
        m = [[A[..., i, j][..., None] for j in range(3)] for i in range(3)]
        cof = [[m[1][1] * m[2][2] - m[1][2] * m[2][1], m[1][2] * m[2][0] - m[1][0] * m[2][2],
                m[1][0] * m[2][1] - m[1][1] * m[2][0]],
               [m[0][2] * m[2][1] - m[0][1] * m[2][2], m[0][0] * m[2][2] - m[0][2] * m[2][0],
                m[0][1] * m[2][0] - m[0][0] * m[2][1]],
               [m[0][1] * m[1][2] - m[0][2] * m[1][1], m[0][2] * m[1][0] - m[0][0] * m[1][2],
                m[0][0] * m[1][1] - m[0][1] * m[1][0]]]
        det = m[0][0] * cof[0][0] + m[0][1] * cof[0][1] + m[0][2] * cof[0][2]
        det = torch.where(ok[..., None], det, torch.ones_like(det))
        rr = [r[..., j] for j in range(3)]
        return torch.stack([(cof[0][i] * rr[0] + cof[1][i] * rr[1] + cof[2][i] * rr[2]) / det for i in range(3)], -1)
    eye = torch.eye(d, dtype=A.dtype, device=A.device)
    Ab = torch.where(ok[..., None, None], A, eye.expand_as(A))
    return torch.linalg.solve(Ab.unsqueeze(-3), r.unsqueeze(-1)).squeeze(-1)


def ls_solve(M: Tensor, b: Tensor, reg: float = LS_REG) -> Tensor:
    """Damped least squares ``v = (M^2 + (reg * tr M)^2 I)^{-1} M b`` per cell (closed form for ``d <= 3``).

    Computed in float64 with autocast disabled; zero response along null directions of ``M``.

    Args:
        M: ``(..., d, d)`` symmetric positive semi-definite.
        b: ``(..., V, d)`` right-hand sides.
        reg: relative damping ``mu / tr(M)``.

    Returns:
        ``(..., V, d)`` in the dtype of ``b``; zero where ``tr(M) = 0`` (isolated vertices).
    """
    out_dtype = b.dtype
    with torch.autocast(device_type=M.device.type, enabled=False):
        M64, b64 = M.double(), b.double()
        d = M64.shape[-1]
        tr = torch.diagonal(M64, dim1=-2, dim2=-1).sum(-1)
        ok = tr > 0
        eye = torch.eye(d, dtype=M64.dtype, device=M64.device)
        A = M64 @ M64 + ((reg * tr) ** 2)[..., None, None] * eye
        r = torch.einsum("...ij,...vj->...vi", M64, b64)
        v = _solve_spd(A, r, ok)
        v = torch.where(ok[..., None, None], v, torch.zeros_like(v))
    return v.to(out_dtype)


class NodeVectorReadout(nn.Module):
    """Vertex vector fields from odd edge scalars (see module docstring).

    Args:
        C: channels.
        n_fields: number of vector fields ``V`` (``cfg.out_dim``); output ``(n_0, B, V * D)``.
        geo_dim1: ``G_1``.
        mode: ``'ls' | 'direct'``.
    """

    def __init__(self, C: int, n_fields: int, geo_dim1: int, mode: str = "ls") -> None:
        super().__init__()
        if mode not in ("ls", "direct"):
            raise ValueError(f"vector_mode must be 'ls' or 'direct', got {mode!r}")
        self.mode = mode
        self.V = int(n_fields)
        self.w = nn.Linear(C, self.V, bias=False)
        if mode == "ls":
            self.omega = GeoMLP(geo_dim1, C + 1, 32, 1, zero_last=True)
            with torch.no_grad():
                self.omega.fc2.bias.fill_(_SOFTPLUS_INV_ONE)

    def forward(self, x: list[Tensor], ctx: LayerContext) -> Tensor:
        """``x[1]`` ``(n_1, B, C)`` -> ``(n_0, B, V * D)``."""
        K = ctx.K
        x1 = x[1]
        dt = ctx.dtype
        n1, B, _ = x1.shape
        n0 = K.n[0]
        geom = vector_geometry(K, dt, ctx.top)
        ev = geom["ev"]
        D, V = ev.shape[1], self.V
        w = linear(self.w, x1).to(dt)                                           # (n1, B, V)  odd
        if self.mode == "direct":
            contrib = (w.unsqueeze(-1) * geom["t"].view(n1, 1, 1, D)).reshape(n1, B, V * D)
            return spmm_ad(K.dT_abs[0], K.d_abs[0], contrib) / geom["deg"]
        feats = torch.cat([torch.log1p(x1.square()), torch.log1p(_msq(x1)).unsqueeze(-1)], dim=-1)
        omega = F.softplus(self.omega(feats, ctx.geo[1]).to(dt))               # (n1, B, 1)  even, > 0
        with torch.autocast(device_type=x1.device.type, enabled=False):         # geometric part in full precision
            M = spmm_ad(K.dT_abs[0], K.d_abs[0], (omega * geom["eet"]).contiguous()).view(n0, B, D, D)
            rhs = ((omega * w).unsqueeze(-1) * ev.view(n1, 1, 1, D)).reshape(n1, B, V * D)
            b = spmm_ad(K.dT_abs[0], K.d_abs[0], rhs).view(n0, B, V, D)
            E = geom["E"]
            if E is not None:                                                   # surface in 3-D: tangent plane
                M2 = torch.einsum("nia,nbij,njc->nbac", E, M, E)
                b2 = torch.einsum("nia,nbvi->nbva", E, b)
                v = torch.einsum("nia,nbva->nbvi", E, ls_solve(M2, b2))
            else:
                v = ls_solve(M, b)
        return v.reshape(n0, B, V * D)


class GradReadout(nn.Module):
    """``E = d_0 phi``, ``phi = MLP(x_0)`` (``head='mlp'``) or ``phi = Lin_nobias(x_0)`` (``head='linear'``, linear
    models such as the solver preset): odd edge cochain ``(n_1, B, out_dim)`` with ``d_1 E = 0`` exactly."""

    def __init__(self, C: int, out_dim: int, head: str = "mlp") -> None:
        super().__init__()
        if head not in ("mlp", "linear"):
            raise ValueError(f"grad readout head must be 'mlp' or 'linear', got {head!r}")
        self.linear = head == "linear"
        if self.linear:
            self.lin = nn.Linear(C, out_dim, bias=False)
        else:
            self.phi = NodeScalarReadout(C, out_dim)

    def potential(self, x: list[Tensor], ctx: LayerContext) -> Tensor:
        """The predicted vertex potential ``phi`` ``(n_0, B, out_dim)``."""
        if self.linear:
            return linear(self.lin, x[0]).to(ctx.dtype)
        return self.phi(x, ctx).to(ctx.dtype)

    def forward(self, x: list[Tensor], ctx: LayerContext) -> Tensor:
        """``(n_1, B, out_dim)``."""
        K = ctx.K
        return spmm_ad(K.d[0], K.dT[0], self.potential(x, ctx))


class CurlReadout(nn.Module):
    """``F = d_1 a``, ``a = Lin_nobias(x_1)``: odd face cochain ``(n_2, B, out_dim)``, closed (``d_2 F = 0``)."""

    def __init__(self, C: int, out_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(C, out_dim, bias=False)

    def forward(self, x: list[Tensor], ctx: LayerContext) -> Tensor:
        """``(n_2, B, out_dim)``."""
        K = ctx.K
        return spmm_ad(K.d[1], K.dT[1], linear(self.lin, x[1]).to(ctx.dtype))


class DivReadout(nn.Module):
    """``y = d_{k-1}^T b``, ``b = Lin_nobias(x_k)``: ``(n_{k-1}, B, out_dim)``, co-closed (``d_{k-2}^T y = 0``); for
    ``k = 1`` a vertex divergence with zero sum per graph."""

    def __init__(self, C: int, out_dim: int, k: int) -> None:
        super().__init__()
        if k < 1:
            raise ValueError("div readout needs k >= 1")
        self.k = int(k)
        self.lin = nn.Linear(C, out_dim, bias=False)

    def forward(self, x: list[Tensor], ctx: LayerContext) -> Tensor:
        """``(n_{k-1}, B, out_dim)``."""
        K = ctx.K
        return spmm_ad(K.dT[self.k - 1], K.d[self.k - 1], linear(self.lin, x[self.k]).to(ctx.dtype))


class MassDivReadout(DivReadout):
    """``y = S_{k-1}^{-1} d_{k-1}^T b``, ``b = Lin_nobias(x_k)``, with ``S`` the normalised star of the output degree
    (``(n_{k-1}, B, out_dim)``); for ``k = 1`` ``sum_i star0_i y_i = 0`` exactly per graph (mass conservation)."""

    def forward(self, x: list[Tensor], ctx: LayerContext) -> Tensor:
        """``(n_{k-1}, B, out_dim)``."""
        y = super().forward(x, ctx)
        return y * torch.exp(-ctx.log_star[self.k - 1]).to(y.dtype).view(-1, 1, 1)


def build_readout(cfg: Any, geo_dims: dict[int, int]) -> nn.Module:
    """Readout module selected by ``cfg.readout`` (``cfg.out_dim``, ``cfg.vector_mode``, ``cfg.C``).

    The module carries ``feature_degree`` (hidden degree it reads) and ``output_degree`` (cells of its output).
    """
    top = max(geo_dims)
    kind, _ = parse_readout(cfg.readout)
    fdeg, odeg = readout_degrees(cfg.readout, top)
    if kind == "node_scalar":
        mod = NodeScalarReadout(cfg.C, cfg.out_dim)
    elif kind == "node_vector":
        mod = NodeVectorReadout(cfg.C, cfg.out_dim, geo_dims[1], cfg.vector_mode)
    elif kind == "cochain":
        mod = CochainReadout(cfg.C, cfg.out_dim, fdeg)
    elif kind == "even":
        mod = EvenCellReadout(cfg.C, cfg.out_dim, fdeg, geo_dims[fdeg])
    elif kind == "grad":
        mod = GradReadout(cfg.C, cfg.out_dim, getattr(cfg, "readout_head", "mlp"))
    elif kind == "curl":
        mod = CurlReadout(cfg.C, cfg.out_dim)
    elif kind == "mdiv":
        mod = MassDivReadout(cfg.C, cfg.out_dim, fdeg)
    else:
        mod = DivReadout(cfg.C, cfg.out_dim, fdeg)
    mod.feature_degree = fdeg
    mod.output_degree = odeg
    return mod
