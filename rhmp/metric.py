"""Bounded log-metric heads (DESIGN §3.2).

For degree ``m`` the learned diagonal cochain metric is, per sample and per cell,

    H_m = ref_m * exp(phi_m),      phi_m = a * tanh(MLP_m(psi_m)),      a = log_range,

so that ``H_m / ref_m`` lies in ``[e^-a, e^a]``.  ``ref_m`` is the reference DEC Hodge star of the
complex (normalised per sample by its geometric mean inside the model, see ``layers.make_context``).
The last MLP layer is zero-initialised, hence ``H_m = ref_m`` (pure DEC) at initialisation.

``psi_m`` is built only from quantities that are invariant under the cochain-frame group O(C), under
E(n), under cell relabelling and under orientation changes, and it depends on one sample only:

* ``geo[m]``            ``(n_m, G_m)``   E(n)-invariant cell descriptors of the complex (broadcast over B),
* even raw inputs       ``(n_m, B, E_m)`` (e.g. a conductivity),
* feature invariants    ``(n_m, B, F_m)`` (built by the layer, see :func:`metric_feature_dim`).

The first linear layer is applied group-wise so that the ``geo`` part is computed once per cell and
broadcast over the batch (no ``(n_m, B, P_m)`` concatenation is ever materialised).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def row_linear(x: Tensor, W: Tensor, b: Tensor | None = None) -> Tensor:
    """Per-cell linear map with a fast weight gradient (see ``rhmp.layers.row_linear``; imported lazily)."""
    from .layers import row_linear as _rl

    return _rl(x, W, b)


def mlp2(x: Tensor, W1: Tensor, e: Tensor, W2: Tensor, b2: Tensor | None) -> Tensor:
    """Memory-lean two-layer MLP (see ``rhmp.layers.mlp2``; imported lazily)."""
    from .layers import mlp2 as _m

    return _m(x, W1, e, W2, b2)

__all__ = ["MetricHead", "TensorMetricHead", "metric_feature_dim"]


def metric_feature_dim(m: int, top: int) -> int:
    """Number of feature invariants fed to the metric head of degree ``m``.

    Layout (all computed per cell and per sample, see ``RHMPLayer._metric_features``)::

        log1p(|x_m|^2 / C)                                   always
        log1p(|q_m^-|^2 / C), asinh(<x_m, q_m^-> / C)       if m >= 1   (q_m^- = d_{m-1} S^-1/2 x_{m-1})
        log1p(|q_m^+|^2 / C)                                 if m < top  (q_m^+ = d_m^T S'^-1/2 x_{m+1})
        per-sample mean over m-cells of log1p(|x_m|^2 / C)   always

    Args:
        m: cochain degree.
        top: top degree ``K`` of the complex.

    Returns:
        ``F_m`` (3..5).
    """
    return 1 + (2 if m >= 1 else 0) + (1 if m < top else 0) + 1


class MetricHead(nn.Module):
    """Bounded log-metric head ``phi = a * tanh(MLP(psi))`` for one degree.

    MLP: ``P -> hidden -> SiLU -> 1`` with ``P = geo_dim + even_dim + feat_dim``; last layer zero-init.

    Args:
        geo_dim: ``G_m``, number of geometric descriptors of the degree.
        even_dim: ``E_m``, number of even raw input columns of the degree (0 if none).
        feat_dim: ``F_m``, number of feature invariants (:func:`metric_feature_dim`).
        hidden: hidden width (DESIGN default 32).
        log_range: ``a``; ``H/ref`` is confined to ``[e^-a, e^a]``.
    """

    def __init__(self, geo_dim: int, even_dim: int, feat_dim: int, hidden: int = 32,
                 log_range: float = 2.0) -> None:
        super().__init__()
        if feat_dim < 1:
            raise ValueError("MetricHead needs at least one feature invariant")
        self.geo_dim = int(geo_dim)
        self.even_dim = int(even_dim)
        self.feat_dim = int(feat_dim)
        self.log_range = float(log_range)
        self.fc1 = nn.Linear(self.geo_dim + self.even_dim + self.feat_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, geo: Tensor, even: Tensor | None, feats: Tensor) -> Tensor:
        """Bounded log-ratio ``phi = log(H / ref)``.

        Args:
            geo: ``(n, G)`` geometric descriptors (ignored if ``G == 0``).
            even: ``(n, B, E)`` even raw inputs, or ``None`` if ``E == 0``.
            feats: ``(n, B, F)`` feature invariants.

        Returns:
            ``phi``: ``(n, B)`` float32 (or the dtype of ``feats`` if it is float64), in ``[-a, a]``.
        """
        n, B, Fd = feats.shape
        if Fd != self.feat_dim:
            raise ValueError(f"MetricHead expected {self.feat_dim} feature invariants, got {Fd}")
        G, E = self.geo_dim, self.even_dim
        W, b = self.fc1.weight, self.fc1.bias
        if G:
            if geo.shape != (n, G):
                raise ValueError(f"MetricHead expected geo of shape {(n, G)}, got {tuple(geo.shape)}")
            e = F.linear(geo, W[:, :G], b).unsqueeze(1)                # (n, 1, hidden) broadcast over B
        else:
            e = b
        if E:
            if even is None or even.shape != (n, B, E):
                got = None if even is None else tuple(even.shape)
                raise ValueError(f"MetricHead expected even inputs of shape {(n, B, E)}, got {got}")
            e = e + row_linear(even, W[:, G:G + E])                    # (n, B, hidden)
        z = mlp2(feats, W[:, G + E:], e, self.fc2.weight, self.fc2.bias).squeeze(-1)   # (n, B)
        z = z.double() if feats.dtype == torch.float64 else z.float()  # bounded map in full precision
        return self.log_range * torch.tanh(z)


class TensorMetricHead(nn.Module):
    """Whitney-consistent material tensor per top cell (DESIGN §9.1).

    ``param='full'`` (the ``RHMPConfig`` default): ``sigma_f = b_f expm(S_f)``, ``S_f = sum_j s_{f,j} t_j t_j^T`` with
    signed ``s_{f,j} = a tanh(z_{f,j})`` (zero-init): every SPD tensor with bounded condition number (the edge dyads
    span Sym(2) on triangles and Sym(3) on tets).  ``param='cone'`` (the default of this class and the
    parameterisation of configurations saved without ``tensor_param``; the cone spanned by ``I`` and the edge dyads,
    i.e. anisotropy along cell edges only):
    ``sigma_f = b_f I + sum_j a_{f,j} t_j t_j^T`` in the cell's own edge frame, with

        b_f     = exp(a * tanh(z_b))              in [e^-a, e^a]       z_b     = MLP_b(psi_f)
        a_{f,j} = e^a * tanh(max(z_{f,j}, 0))     in [0, e^a)          z_{f,j} = MLP_a([psi_f, psi_{e_j}])

    where ``psi_f`` are the (O(C)-, E(n)-invariant, per-sample) descriptors of the top cell and ``psi_{e_j}`` those of
    its direction edge ``j`` (so that ``a_{f,j}`` can align anisotropy with edge ``j``).  Both last layers are
    zero-initialised: ``b = 1``, ``a = 0`` (the Galerkin/Whitney star of the reference metric) at initialisation;
    the clamp passes gradients at 0 so ``a`` can grow from there.  The induced cochain metric is assembled by
    ``rhmp.dec.apply_whitney_metric``.

    Args:
        in_top: width of ``psi_f``.
        in_edge: width of ``psi_{e_j}``.
        hidden: hidden width of both MLPs.
        log_range: ``a``.
        param: ``'full'`` or ``'cone'`` (see above).
    """

    def __init__(self, in_top: int, in_edge: int, hidden: int = 32, log_range: float = 2.0, param: str = "cone") -> None:
        super().__init__()
        if param not in ("cone", "full"):
            raise ValueError(f"tensor_param must be 'cone' or 'full', got {param!r}")
        self.in_top, self.in_edge = int(in_top), int(in_edge)
        self.log_range = float(log_range)
        self.param = param
        self.b1 = nn.Linear(self.in_top, hidden)
        self.b2 = nn.Linear(hidden, 1)
        self.a1 = nn.Linear(self.in_top + self.in_edge, hidden)
        self.a2 = nn.Linear(hidden, 1)
        for lin in (self.b2, self.a2):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, psi_top: Tensor, psi_edge: Tensor) -> tuple[Tensor, Tensor]:
        """Tensor parameters of every top cell.

        Args:
            psi_top: ``(n_top, B, in_top)``.
            psi_edge: ``(n_top, m, B, in_edge)`` descriptors of the ``m`` direction edges of each top cell.

        Returns:
            ``(b, a)``: ``b`` ``(n_top, B)`` in ``[e^-a, e^a]`` and ``a`` ``(n_top, B, m)`` in ``[0, e^a)`` (float32 or
            the dtype of the inputs if float64).  With ``param='full'`` the second output is the signed log-tensor
            coordinate ``s`` ``(n_top, B, m)`` in ``(-a, a)`` (``sigma_f = b_f expm(sum_j s_j t_j t_j^T)``, assembled into
            explicit Galerkin blocks by ``rhmp.layers.galerkin_blocks``).
        """
        full = torch.float64 if psi_top.dtype == torch.float64 else torch.float32
        zb = mlp2(psi_top, self.b1.weight, self.b1.bias, self.b2.weight, self.b2.bias).squeeze(-1)
        b = torch.exp(self.log_range * torch.tanh(zb.to(full)))
        Wt, We = self.a1.weight[:, :self.in_top], self.a1.weight[:, self.in_top:]
        e = F.linear(psi_top, Wt, self.a1.bias).unsqueeze(1)                         # (n_top, 1, B, hidden)
        za = mlp2(psi_edge, We, e, self.a2.weight, self.a2.bias).squeeze(-1)          # (n_top, m, B)
        if self.param == "full":                                                      # signed log-tensor coordinates
            return b, (self.log_range * torch.tanh(za.to(full))).permute(0, 2, 1).contiguous()
        a = math.exp(self.log_range) * torch.tanh(za.to(full).clamp_min(0.0))
        return b, a.permute(0, 2, 1).contiguous()
