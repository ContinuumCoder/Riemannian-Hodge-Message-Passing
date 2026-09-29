# Vendored v1 code.  Original: module `gauge_hodge_mp.hodge_mp` of the RHMP v1 code base,
#   https://github.com/ContinuumCoder/Riemannian-Hodge-Message-Passing
# which accompanies "Learning Discrete Riemannian Metrics for Physical Fields with Cochain-Frame Equivariance"
# (Zheng & Allen-Blanchette, arXiv:2608.14556).  Kept verbatim (only this header was added).
# Used by rhmp.baselines (the v1 baselines and the v1 model `ours_v1`) and by `python -m rhmp.train --eval-v1`.
"""
Hodge Message Passing layer.

Two operations per layer:
1. Hodge Laplacian (self-interaction): d_{k-1} H d_{k-1}^T x_k + d_k^T H d_k x_k
2. Cross-dimension transfer: d_{k-1} H x_{k-1} + d_k^T H x_{k+1}

Guarantees: d^2=0, O(C)-equivariance, E(n)-invariance, SPD metric.

Metric variants:
- "full": H in R^{n x n}, full SPD (O(n^2) parameters)
- "diagonal": H = diag(h) + eps*I (O(n) parameters, element-wise)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import build_spd, scatter_mean, spmm, batch_spmm


class HodgeMPLayer(nn.Module):
    """
    Single Gauge-Hodge Message Passing layer.

    Supports full SPD and diagonal metric variants:
    - full: H = LL^T + eps*I, O(n^2) params
    - diagonal: H = softplus(h) + eps*I, O(n) params
    """

    def __init__(self, C: int, n_lower: int, n_upper: int, hidden: int = 64,
                 eps: float = 1e-3, metric_type: str = "diagonal",
                 metric_rank: int = 0, geo_dim: int = 2,
                 identity_metric: bool = False,
                 no_cross: bool = False,
                 relu_gate: bool = False):
        """
        Args:
            C: cochain feature channels
            n_lower: lower-adjacency dimension (0 = none)
            n_upper: upper-adjacency dimension (0 = none)
            hidden: MLP hidden dimension
            eps: diagonal regularisation for the metric
            metric_type: "full" (SPD) or "diagonal" (efficient)
            metric_rank: 0 = standard, >0 = low-rank (k basis vectors)
        """
        super().__init__()
        self.C = C
        self.n_lower = n_lower
        self.n_upper = n_upper
        self.eps = eps
        self.has_lower = n_lower > 0
        self.has_upper = n_upper > 0
        self.metric_type = metric_type
        self.metric_rank = metric_rank

        self.identity_metric = identity_metric
        self.no_cross = no_cross
        self.relu_gate = relu_gate

        inv_dim = 2  # ||x||^2, mean(x^T x_neighbors)
        self.geo_dim = geo_dim

        if self.has_lower:
            if metric_type == "local_rich":
                self.mlp_H_down = nn.Sequential(
                    nn.Linear(self.geo_dim, hidden), nn.SiLU(),
                    nn.Linear(hidden, hidden // 2), nn.SiLU(),
                    nn.Linear(hidden // 2, 1)
                )
            elif metric_type == "local":
                self.mlp_H_down = nn.Sequential(
                    nn.Linear(2, hidden), nn.SiLU(),
                    nn.Linear(hidden, 1)
                )
            elif metric_type == "scalar":
                self.mlp_H_down = nn.Sequential(
                    nn.Linear(inv_dim, hidden), nn.SiLU(),
                    nn.Linear(hidden, 1)
                )
            elif metric_rank > 0 and metric_type == "diagonal":
                k = metric_rank
                self.basis_down = nn.Parameter(torch.randn(n_lower, k) * 0.01)
                self.mlp_H_down = nn.Sequential(
                    nn.Linear(inv_dim, hidden), nn.SiLU(),
                    nn.Linear(hidden, k)
                )
            else:
                out_dim = (n_lower * (n_lower + 1) // 2) if metric_type == "full" else n_lower
                self.mlp_H_down = nn.Sequential(
                    nn.Linear(inv_dim, hidden), nn.SiLU(),
                    nn.Linear(hidden, out_dim)
                )
            self.alpha_lower = nn.Parameter(torch.tensor(0.5))

        if self.has_upper:
            if metric_type == "local_rich":
                self.mlp_H_up = nn.Sequential(
                    nn.Linear(self.geo_dim, hidden), nn.SiLU(),
                    nn.Linear(hidden, hidden // 2), nn.SiLU(),
                    nn.Linear(hidden // 2, 1)
                )
            elif metric_type == "local":
                self.mlp_H_up = nn.Sequential(
                    nn.Linear(2, hidden), nn.SiLU(),
                    nn.Linear(hidden, 1)
                )
            elif metric_type == "scalar":
                self.mlp_H_up = nn.Sequential(
                    nn.Linear(inv_dim, hidden), nn.SiLU(),
                    nn.Linear(hidden, 1)
                )
            elif metric_rank > 0 and metric_type == "diagonal":
                k = metric_rank
                self.basis_up = nn.Parameter(torch.randn(n_upper, k) * 0.01)
                self.mlp_H_up = nn.Sequential(
                    nn.Linear(inv_dim, hidden), nn.SiLU(),
                    nn.Linear(hidden, k)
                )
            else:
                out_dim = (n_upper * (n_upper + 1) // 2) if metric_type == "full" else n_upper
                self.mlp_H_up = nn.Sequential(
                    nn.Linear(inv_dim, hidden), nn.SiLU(),
                    nn.Linear(hidden, out_dim)
                )
            self.alpha_upper = nn.Parameter(torch.tensor(0.5))

        self.gate_mlp = nn.Sequential(
            nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 1), nn.Sigmoid()
        )

        # Channel mixing is placed before the readout in network.py (not here)
        # to preserve strict O(C)-equivariance within MP layers.

    def _build_metric(self, mlp_out: torch.Tensor, n: int, basis: torch.Tensor = None):
        """Build metric matrix (full) or diagonal vector."""
        if self.metric_type == "full":
            return build_spd(mlp_out, n, self.eps).squeeze(0)  # (n, n)
        elif self.metric_type == "scalar":
            return F.softplus(mlp_out.squeeze()) + self.eps
        elif self.metric_type in ("local", "local_rich"):
            return F.softplus(mlp_out.squeeze(-1)) + self.eps  # (n,)
        else:
            if basis is not None:
                coeffs = mlp_out.squeeze(0)  # (k,)
                h = basis @ coeffs  # (n, k) @ (k,) -> (n,)
            else:
                h = mlp_out.squeeze(0)  # (n,)
            return F.softplus(h) + self.eps  # (n,)

    def _metric_matmul(self, H, x: torch.Tensor) -> torch.Tensor:
        """Metric-feature product: full -> matmul, diagonal -> element-wise."""
        if self.identity_metric:
            return x  # H = I ablation
        if self.metric_type == "full":
            return H @ x  # (n, C)
        else:
            return H.unsqueeze(-1) * x  # (n,1)*(n,C) -> (n,C)

    def compute_invariants(self, x: torch.Tensor, adj: torch.Tensor | None
                           ) -> torch.Tensor:
        """O(C)-invariants: ||x||^2 and mean(x^T x_neighbors)."""
        norm_sq = (x * x).sum(dim=-1, keepdim=True)
        if adj is not None and adj.numel() > 0:
            inner = (x[adj[0]] * x[adj[1]]).sum(dim=-1, keepdim=True)
            neighbor_ip = scatter_mean(
                inner.squeeze(-1), adj[0], dim=0, dim_size=x.size(0)
            ).unsqueeze(-1)
        else:
            neighbor_ip = torch.zeros_like(norm_sq)
        return torch.cat([norm_sq, neighbor_ip], dim=-1)

    def forward(self, x_k: torch.Tensor,
                d_lower: torch.Tensor | None,
                d_upper: torch.Tensor | None,
                adj: torch.Tensor | None,
                x_lower: torch.Tensor | None = None,
                x_upper: torch.Tensor | None = None,
                geo_lower: torch.Tensor | None = None,
                geo_upper: torch.Tensor | None = None) -> torch.Tensor:
        inv = self.compute_invariants(x_k, adj)
        inv_global = inv.mean(dim=0, keepdim=True)

        msg = torch.zeros_like(x_k)

        # Lower adjacency
        if self.has_lower and d_lower is not None:
            if self.metric_type == "local_rich":
                geo = geo_lower if geo_lower is not None else torch.zeros(d_lower.size(1), self.geo_dim, device=x_k.device)
                H_down = self._build_metric(self.mlp_H_down(geo), geo.size(0))
            elif self.metric_type == "local":
                x_src = x_lower if x_lower is not None else spmm(d_lower.t(), x_k)
                norm_sq = (x_src * x_src).sum(-1, keepdim=True)
                geo = geo_lower if geo_lower is not None else torch.zeros_like(norm_sq)
                local_inv = torch.cat([norm_sq, geo], dim=-1)
                H_down = self._build_metric(self.mlp_H_down(local_inv), x_src.size(0))
            else:
                basis_down = getattr(self, 'basis_down', None)
                H_down = self._build_metric(self.mlp_H_down(inv_global), self.n_lower, basis_down)

            # Hodge Laplacian: d H d^T x_k
            agg = spmm(d_lower.t(), x_k)
            msg_hodge = spmm(d_lower, self._metric_matmul(H_down, agg))

            if x_lower is not None and not self.no_cross:
                msg_cross = spmm(d_lower, self._metric_matmul(H_down, x_lower))
                alpha = torch.sigmoid(self.alpha_lower)
                msg = msg + alpha * msg_hodge + (1 - alpha) * msg_cross
            else:
                msg = msg + msg_hodge

        # Upper adjacency
        if self.has_upper and d_upper is not None:
            if self.metric_type == "local_rich":
                geo = geo_upper if geo_upper is not None else torch.zeros(d_upper.size(0), self.geo_dim, device=x_k.device)
                H_up = self._build_metric(self.mlp_H_up(geo), geo.size(0))
            elif self.metric_type == "local":
                x_dst = x_upper if x_upper is not None else spmm(d_upper, x_k)
                norm_sq = (x_dst * x_dst).sum(-1, keepdim=True)
                geo = geo_upper if geo_upper is not None else torch.zeros_like(norm_sq)
                local_inv = torch.cat([norm_sq, geo], dim=-1)
                H_up = self._build_metric(self.mlp_H_up(local_inv), x_dst.size(0))
            else:
                basis_up = getattr(self, 'basis_up', None)
                H_up = self._build_metric(self.mlp_H_up(inv_global), self.n_upper, basis_up)

            agg = spmm(d_upper, x_k)
            msg_hodge = spmm(d_upper.t(), self._metric_matmul(H_up, agg))

            if x_upper is not None and not self.no_cross:
                msg_cross = spmm(d_upper.t(), self._metric_matmul(H_up, x_upper))
                alpha = torch.sigmoid(self.alpha_upper)
                msg = msg + alpha * msg_hodge + (1 - alpha) * msg_cross
            else:
                msg = msg + msg_hodge

        # Nonlinearity
        if self.relu_gate:
            return x_k + F.relu(msg)
        msg_norm = msg.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        gate = self.gate_mlp(msg_norm)
        return x_k + gate * msg

    def _batch_metric_matmul(self, H, x: torch.Tensor) -> torch.Tensor:
        """Batched metric-feature product: x is (B, n, C)."""
        if self.identity_metric:
            return x  # H = I ablation
        if self.metric_type == "full":
            return torch.einsum("mn,bnc->bmc", H, x)
        else:
            return H.unsqueeze(0).unsqueeze(-1) * x

    def forward_batch(self, x_k: torch.Tensor,
                      d_lower: torch.Tensor | None,
                      d_upper: torch.Tensor | None,
                      adj: torch.Tensor | None,
                      x_lower: torch.Tensor | None = None,
                      x_upper: torch.Tensor | None = None) -> torch.Tensor:
        """
        Batched forward: x_k (B, n_k, C) -> (B, n_k, C).
        Uses batch-mean invariants for shared metric construction.
        """
        B = x_k.size(0)

        x_mean = x_k.mean(dim=0)
        inv = self.compute_invariants(x_mean, adj)
        inv_global = inv.mean(dim=0, keepdim=True)

        msg = torch.zeros_like(x_k)

        if self.has_lower and d_lower is not None:
            if self.metric_type == "local":
                x_src = x_lower if x_lower is not None else batch_spmm(d_lower.t(), x_k)
                local_inv = (x_src.mean(dim=0) * x_src.mean(dim=0)).sum(-1, keepdim=True)
                H_down = self._build_metric(self.mlp_H_down(local_inv), x_src.size(1))
            else:
                basis_down = getattr(self, 'basis_down', None)
                H_down = self._build_metric(self.mlp_H_down(inv_global), self.n_lower, basis_down)
            agg = batch_spmm(d_lower.t(), x_k)
            h_agg = self._batch_metric_matmul(H_down, agg)
            msg_hodge = batch_spmm(d_lower, h_agg)

            if x_lower is not None and not self.no_cross:
                h_lower = self._batch_metric_matmul(H_down, x_lower)
                msg_cross = batch_spmm(d_lower, h_lower)
                alpha = torch.sigmoid(self.alpha_lower)
                msg = msg + alpha * msg_hodge + (1 - alpha) * msg_cross
            else:
                msg = msg + msg_hodge

        if self.has_upper and d_upper is not None:
            if self.metric_type == "local":
                x_dst = x_upper if x_upper is not None else batch_spmm(d_upper, x_k)
                local_inv = (x_dst.mean(dim=0) * x_dst.mean(dim=0)).sum(-1, keepdim=True)
                H_up = self._build_metric(self.mlp_H_up(local_inv), x_dst.size(1))
            else:
                basis_up = getattr(self, 'basis_up', None)
                H_up = self._build_metric(self.mlp_H_up(inv_global), self.n_upper, basis_up)
            agg = batch_spmm(d_upper, x_k)
            h_agg = self._batch_metric_matmul(H_up, agg)
            msg_hodge = batch_spmm(d_upper.t(), h_agg)

            if x_upper is not None and not self.no_cross:
                h_upper = self._batch_metric_matmul(H_up, x_upper)
                msg_cross = batch_spmm(d_upper.t(), h_upper)
                alpha = torch.sigmoid(self.alpha_upper)
                msg = msg + alpha * msg_hodge + (1 - alpha) * msg_cross
            else:
                msg = msg + msg_hodge

        if self.relu_gate:
            return x_k + F.relu(msg)
        msg_norm = msg.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        gate = self.gate_mlp(msg_norm)
        return x_k + gate * msg
