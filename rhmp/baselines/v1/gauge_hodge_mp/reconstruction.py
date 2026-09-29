# Vendored v1 code.  Original: module `gauge_hodge_mp.reconstruction` of the RHMP v1 code base,
#   https://github.com/ContinuumCoder/Riemannian-Hodge-Message-Passing
# which accompanies "Learning Discrete Riemannian Metrics for Physical Fields with Cochain-Frame Equivariance"
# (Zheng & Allen-Blanchette, arXiv:2608.14556).  Kept verbatim (only this header was added).
# Used by rhmp.baselines (the v1 baselines and the v1 model `ours_v1`) and by `python -m rhmp.train --eval-v1`.
"""
Equivariant reconstruction (readout stage).

Recovers E(n)-equivariant vector/tensor fields from E(n)-invariant k-cochains.

Key principle:
  x_1(e) is an E(n)-invariant scalar cochain,
  e_hat_{vw} is an E(n)-equivariant unit edge direction,
  their product x_1 * e_hat is E(n)-equivariant.
"""

import torch
import torch.nn as nn
from .utils import scatter_sum


class EquivariantReconstruction(nn.Module):
    """
    1-cochain -> vertex vector field (E(n)-equivariant).

    v_hat(v) = sum_{e containing v} w(x_1(e)) * e_hat_{vw}

    where w(.) is a scalar weight MLP (E(n)-invariant) and
    e_hat_{vw} = (r_w - r_v) / ||r_w - r_v|| is the unit edge direction (E(n)-equivariant).
    """

    def __init__(self, C: int, spatial_dim: int = 3):
        """
        Args:
            C: cochain feature dimension
            spatial_dim: spatial dimension (2D or 3D)
        """
        super().__init__()
        self.spatial_dim = spatial_dim
        # Weight MLP: outputs 2 scalars (w_src, w_dst), asymmetric for the two endpoints
        # Input = [x1_e, ||r_ij||^2, x0_src_norm, x0_dst_norm], all E(n)-invariant
        self.weight_mlp = nn.Sequential(
            nn.Linear(C + 3, C), nn.SiLU(), nn.Linear(C, 2)
        )

    def forward(self, x1: torch.Tensor, pos: torch.Tensor,
                edges: torch.Tensor, n_vertices: int,
                x0: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x1: (n1, C) 1-cochain features (E(n)-invariant)
            pos: (n0, dim) vertex coordinates
            edges: (n1, 2) directed edges
            n_vertices: number of vertices
            x0: (n0, C) 0-cochain features (optional, for endpoint information)

        Returns:
            (n0, dim) vertex vector field (E(n)-equivariant)
        """
        src, dst = edges[:, 0], edges[:, 1]

        edge_dirs = pos[dst] - pos[src]  # E(n)-equivariant
        edge_len_sq = (edge_dirs * edge_dirs).sum(-1, keepdim=True)  # E(n)-invariant

        if x0 is not None:
            src_norm = (x0[src] * x0[src]).sum(-1, keepdim=True)
            dst_norm = (x0[dst] * x0[dst]).sum(-1, keepdim=True)
        else:
            src_norm = dst_norm = torch.zeros_like(edge_len_sq)

        # Asymmetric scalar weights (w_src, w_dst)
        mlp_input = torch.cat([x1, edge_len_sq, src_norm, dst_norm], dim=-1)
        weights = self.weight_mlp(mlp_input)
        w_src = weights[:, 0]
        w_dst = weights[:, 1]

        # Asymmetric aggregation to vertices
        v_src = scatter_sum((w_src.unsqueeze(-1) * edge_dirs), src, dim=0, dim_size=n_vertices)
        v_dst = scatter_sum((w_dst.unsqueeze(-1) * (-edge_dirs)), dst, dim=0, dim_size=n_vertices)
        v_field = v_src + v_dst

        # Degree normalisation
        deg = scatter_sum(torch.ones(len(edges), device=edges.device), src, dim=0, dim_size=n_vertices) + \
              scatter_sum(torch.ones(len(edges), device=edges.device), dst, dim=0, dim_size=n_vertices)
        v_field = v_field / deg.unsqueeze(-1).clamp(min=1)

        return v_field


class ScalarReadout(nn.Module):
    """
    k-cochain -> per-cell scalar prediction.

    The readout is a task-specific decoder; the MP layers maintain
    strict O(C)-equivariance internally.
    """

    def __init__(self, C: int, out_dim: int = 1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(C, C), nn.SiLU(), nn.Linear(C, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (n, C) cochain features

        Returns:
            (n, out_dim)
        """
        return self.mlp(x)
