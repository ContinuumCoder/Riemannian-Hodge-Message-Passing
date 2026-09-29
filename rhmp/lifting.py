"""Cochain lifting: raw inputs on any degree -> hidden cochains (DESIGN §3.4, §3.6).

Input layout on degree ``k`` (``inputs[k]``: ``(n_k, B, F_k)``)::

    [ connection columns (c_k) | ordinary odd columns (O_k) | even columns (E_k) ]
      cfg.connection_dims[k]     F_k - c_k - E_k               cfg.even_dims[k]

Hidden cochains (all ``(n_k, B, C)``)::

    x_0 = MLP([f_0, geo_0])                                   (connection and material columns excluded)
    for k = 1..K:
        o_k = Lin_a(d_{k-1} x_{k-1}) + Lin_b(f_k^odd) + Lin_c(d_{k-1} A_{k-1})         (bias-free: odd)
        e_k = MLP([f_k^even, geo_k, log1p(|f_k^odd|^2), log1p(|d_{k-1} x_{k-1}|^2 / C),
                   mean over boundary (k-1)-cells of log1p(|x_{k-1}|^2 / C)])         (even, C-dim)
        x_k = o_k ⊙ (1 + e_k)                                                        (odd x even = odd)

*Material* columns (``cfg.material_dims[k]``: the last even columns) never enter the lifting: they only feed the
metric heads (and ``metric_reference``).  ``cfg.lifting='linear'`` drops the geometry, the biases and the even gates
(``x_0 = W f_0``, ``x_k = o_k``), so the lifting is linear in the field inputs.

``A_{k-1}`` are the connection columns of degree ``k-1``; they enter *only* through ``d_{k-1} A_{k-1}``, which
makes the network exactly invariant under ``A -> A + d_{k-2} lambda`` (``d d = 0``).  With
``cfg.connection_odd=True`` (non-Abelian connections) they additionally enter the ordinary odd path.
The last layer of every ``e_k`` MLP is zero-initialised (``x_k = o_k`` at initialisation).
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .layers import LayerContext, _msq, linear, mlp2, spmm_ad

__all__ = ["GeoMLP", "CochainLifting", "input_layout"]


class GeoMLP(nn.Module):
    """Two-layer MLP on ``[feats, geo]`` where the ``geo`` part of the first layer is computed once per cell.

    ``y = fc2(SiLU(W_f feats + W_g geo + b))``.

    Args:
        geo_dim: ``G`` (per-cell descriptors, broadcast over the batch); may be 0.
        in_dim: ``P`` (per-sample features); may be 0.
        hidden: hidden width.
        out_dim: output width.
        zero_last: zero-initialise the last layer.
    """

    def __init__(self, geo_dim: int, in_dim: int, hidden: int, out_dim: int, zero_last: bool = False) -> None:
        super().__init__()
        if geo_dim + in_dim == 0:
            raise ValueError("GeoMLP needs at least one input feature")
        self.geo_dim = int(geo_dim)
        self.in_dim = int(in_dim)
        self.fc1 = nn.Linear(self.in_dim + self.geo_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        if zero_last:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, feats: Tensor | None, geo: Tensor | None) -> Tensor:
        """Evaluate the MLP.

        Args:
            feats: ``(n, B, P)`` (or ``None`` if ``P == 0``).
            geo: ``(n, G)`` (or ``None`` if ``G == 0``).

        Returns:
            ``(n, B, out_dim)``, or ``(n, 1, out_dim)`` when ``P == 0``.
        """
        W, b = self.fc1.weight, self.fc1.bias
        if self.in_dim and (feats is None or feats.shape[-1] != self.in_dim):
            got = None if feats is None else tuple(feats.shape)
            raise ValueError(f"GeoMLP expected {self.in_dim} features, got {got}")
        if self.geo_dim:
            if geo is None or geo.shape[-1] != self.geo_dim:
                got = None if geo is None else tuple(geo.shape)
                raise ValueError(f"GeoMLP expected {self.geo_dim} geometric descriptors, got {got}")
            e = F.linear(geo, W[:, self.in_dim:], b).unsqueeze(1)                       # (n, 1, hidden)
        else:
            e = b
        if not self.in_dim:                                                             # geometry only: (n, 1, out)
            return linear(self.fc2, F.silu(e))
        return mlp2(feats, W[:, :self.in_dim], e, self.fc2.weight, self.fc2.bias)


def input_layout(cfg: Any, k: int) -> dict[str, int]:
    """Column layout of ``inputs[k]`` for a config.

    Returns:
        ``dict(F, conn, E, M, E_lift, odd_lo, odd_hi, O)``: total width, number of connection columns, number of even
        columns, number of *material* columns (the last ``M`` even columns: metric heads / reference only), number of
        even columns seen by the lifting gates (``E - M``, columns ``[F - E, F - M)``), and the ``[odd_lo, odd_hi)``
        range of the columns entering the ordinary odd path (width ``O``).  For degree 0 the odd range is the range
        fed to the vertex map (odd/even is meaningless on vertices; material columns excluded).
    """
    Fk = int(cfg.in_dims.get(k, 0))
    conn = int(cfg.connection_dims.get(k, 0))
    E = int(cfg.even_dims.get(k, 0))
    M = int(getattr(cfg, "material_dims", {}).get(k, 0))
    odd_lo = 0 if cfg.connection_odd else conn
    odd_hi = Fk - M if k == 0 else Fk - E
    return dict(F=Fk, conn=conn, E=E, M=M, E_lift=E - M, odd_lo=odd_lo, odd_hi=odd_hi, O=max(odd_hi - odd_lo, 0))


class CochainLifting(nn.Module):
    """Lift raw inputs on any degree to hidden cochains ``x_k`` ``(n_k, B, C)`` (see module docstring).

    Args:
        cfg: ``RHMPConfig`` (uses ``in_dims, even_dims, material_dims, connection_dims, connection_odd, lifting, C,
            lift_hidden``).
        geo_dims: ``{k: G_k}``.
    """

    def __init__(self, cfg: Any, geo_dims: dict[int, int]) -> None:
        super().__init__()
        self.top = max(geo_dims)
        self.C = int(cfg.C)
        self.layout = {k: input_layout(cfg, k) for k in range(self.top + 1)}
        self.linear = getattr(cfg, "lifting", "mlp") == "linear"
        if self.layout[self.top]["conn"] > 0:
            raise ValueError(f"connection columns on the top degree {self.top} have no coboundary")
        C = self.C
        if self.linear:                                   # linear in the field inputs: no geometry, no bias, no gates
            self.lin0 = nn.Linear(self.layout[0]["O"], C, bias=False) if self.layout[0]["O"] > 0 else None
        else:
            self.mlp0 = GeoMLP(geo_dims[0], self.layout[0]["O"], C, C)
        self.lin_a = nn.ModuleDict()
        self.lin_b = nn.ModuleDict()
        self.lin_conn = nn.ModuleDict()
        self.gate = nn.ModuleDict()
        for k in range(1, self.top + 1):
            lay = self.layout[k]
            self.lin_a[str(k)] = nn.Linear(C, C, bias=False)
            if lay["O"] > 0:
                self.lin_b[str(k)] = nn.Linear(lay["O"], C, bias=False)
            if self.layout[k - 1]["conn"] > 0:
                self.lin_conn[str(k)] = nn.Linear(self.layout[k - 1]["conn"], C, bias=False)
            if not self.linear:
                n_inv = lay["E_lift"] + (1 if lay["O"] > 0 else 0) + 2
                self.gate[str(k)] = GeoMLP(geo_dims[k], n_inv, int(cfg.lift_hidden), C, zero_last=True)

    def forward(self, inputs: dict[int, Tensor], ctx: LayerContext) -> list[Tensor]:
        """Lift.

        Args:
            inputs: ``{k: (n_k, B, F_k)}`` (validated by the model, already in ``ctx.dtype``).
            ctx: forward context.

        Returns:
            ``[x_0, .., x_K]``, ``x_k``: ``(n_k, B, C)`` contiguous in ``ctx.dtype``.
        """
        K = ctx.K
        B, C, dt = ctx.B, self.C, ctx.dtype
        lay0 = self.layout[0]
        f0 = inputs[0][..., lay0["odd_lo"]:lay0["odd_hi"]] if lay0["O"] > 0 else None
        if self.linear:
            x0 = linear(self.lin0, f0).to(dt) if self.lin0 is not None else \
                torch.zeros(K.n[0], B, C, dtype=dt, device=ctx.log_star[0].device)
        else:
            x0 = self.mlp0(f0, ctx.geo[0]).to(dt)
        xs = [x0.expand(K.n[0], B, C).contiguous()]
        for k in range(1, self.top + 1):
            lay = self.layout[k]
            xp = xs[k - 1]
            dx = spmm_ad(K.d[k - 1], K.dT[k - 1], xp)                                   # (n_k, B, C)
            o = linear(self.lin_a[str(k)], dx)
            if self.linear:
                if lay["O"] > 0:
                    o = o + linear(self.lin_b[str(k)], inputs[k][..., lay["odd_lo"]:lay["odd_hi"]])
                if str(k) in self.lin_conn:
                    a = inputs[k - 1][..., :self.layout[k - 1]["conn"]].contiguous()
                    o = o + linear(self.lin_conn[str(k)], spmm_ad(K.d[k - 1], K.dT[k - 1], a))
                xs.append(o.to(dt).contiguous())
                continue
            feats = []
            if lay["E_lift"] > 0:                                                       # material columns excluded
                feats.append(inputs[k][..., lay["F"] - lay["E"]:lay["F"] - lay["M"]])
            if lay["O"] > 0:
                fo = inputs[k][..., lay["odd_lo"]:lay["odd_hi"]]
                o = o + linear(self.lin_b[str(k)], fo)
                feats.append(torch.log1p(fo.square().sum(-1, keepdim=True)))
            if str(k) in self.lin_conn:
                a = inputs[k - 1][..., :self.layout[k - 1]["conn"]].contiguous()
                o = o + linear(self.lin_conn[str(k)], spmm_ad(K.d[k - 1], K.dT[k - 1], a))  # d_{k-1} A only
            feats.append(torch.log1p(_msq(dx)).unsqueeze(-1))
            v = torch.log1p(_msq(xp)).unsqueeze(-1)                                      # (n_{k-1}, B, 1)
            feats.append(spmm_ad(K.d_abs[k - 1], K.dT_abs[k - 1], v) / ctx.bcount[k])
            e = self.gate[str(k)](torch.cat([f.to(dt) for f in feats], dim=-1), ctx.geo[k])
            xs.append(torch.addcmul(o, o, e).to(dt).contiguous())                      # o ⊙ (1 + e)
        return xs
