"""
E(n)-invariant lifting encoder.

Maps node features + geometric invariants to k-cochain features.
All inputs are E(n)-invariant (distances, angles, areas).
1-cochain lifting satisfies Z_2 orientation compatibility (flipping
an edge orientation negates the output).
"""

import torch
import torch.nn as nn
from .utils import (
    compute_edge_lengths, compute_face_areas, compute_face_angles,
    compute_node_degrees, scatter_mean,
)


class InvariantLifting(nn.Module):
    """
    E(n)-invariant lifting:
    - 0-cochain: node features + degree + mean neighbour distance
    - 1-cochain: symmetric/antisymmetric endpoint features + edge length
      + endpoint degrees (Z_2 compatible)
    - 2-cochain: sum of edge features + area + three interior angles
    """

    def __init__(self, f_in: int, C: int):
        """
        Args:
            f_in: raw node feature dimension
            C: cochain feature dimension (shared across all ranks)
        """
        super().__init__()
        self.C = C

        # 0-cochain MLP: node_feat(f_in) + degree(1) + avg_neighbour_dist(1)
        self.node_mlp = nn.Sequential(
            nn.Linear(f_in + 2, C), nn.SiLU(), nn.Linear(C, C)
        )

        # 1-cochain MLP: symmetric + antisymmetric branches -> Z_2 compatible
        # Symmetric: (x_i + x_j)(C) + edge_len(1) + (deg_i + deg_j)(1) = C+2
        # Antisymmetric: (x_i - x_j)(C) + edge_len(1) + (deg_i - deg_j)(1) = C+2
        self.edge_mlp_sym = nn.Sequential(
            nn.Linear(C + 2, C), nn.SiLU(), nn.Linear(C, C)
        )
        self.edge_mlp_asym = nn.Sequential(
            nn.Linear(C + 2, C), nn.SiLU(), nn.Linear(C, C)
        )

        # 2-cochain MLP: edge_feat_sum(C) + area(1) + 3 angles(3) = C+4
        self.face_mlp = nn.Sequential(
            nn.Linear(C + 4, C), nn.SiLU(), nn.Linear(C, C)
        )

    def forward(self, f0: torch.Tensor, K) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            f0: (n0, f_in) raw node features
            K: CellComplex instance

        Returns:
            x0: (n0, C) 0-cochain
            x1: (n1, C) 1-cochain
            x2: (n2, C) 2-cochain
        """
        pos, edges, faces = K.pos, K.edges, K.faces

        # Geometric invariants (E(n)-invariant, computed once)
        edge_lengths = compute_edge_lengths(pos, edges)
        degrees = compute_node_degrees(edges, K.n0)
        avg_dist_src = scatter_mean(
            edge_lengths, edges[:, 0], dim=0, dim_size=K.n0
        )
        avg_dist_dst = scatter_mean(
            edge_lengths, edges[:, 1], dim=0, dim_size=K.n0
        )
        avg_dist = (avg_dist_src + avg_dist_dst) / 2

        # 0-cochain lifting
        x0_input = torch.cat([
            f0,
            degrees.unsqueeze(-1),
            avg_dist.unsqueeze(-1),
        ], dim=-1)
        x0 = self.node_mlp(x0_input)

        # 1-cochain lifting (Z_2 compatible)
        x0_i = x0[edges[:, 0]]
        x0_j = x0[edges[:, 1]]
        deg_i = degrees[edges[:, 0]].unsqueeze(-1)
        deg_j = degrees[edges[:, 1]].unsqueeze(-1)
        el = edge_lengths.unsqueeze(-1)

        feat_sym = torch.cat([x0_i + x0_j, el, deg_i + deg_j], dim=-1)
        feat_asym = torch.cat([x0_i - x0_j, el, deg_i - deg_j], dim=-1)

        # Orientation flip: sym invariant, asym negates -> x1 negates
        x1 = self.edge_mlp_sym(feat_sym) + self.edge_mlp_asym(feat_asym)

        # 2-cochain lifting
        edge_feats_sum = self._gather_face_edge_features(x1, faces, edges, K)
        areas = compute_face_areas(pos, faces).unsqueeze(-1)
        angles = compute_face_angles(pos, faces)

        feat_face = torch.cat([edge_feats_sum, areas, angles], dim=-1)
        x2 = self.face_mlp(feat_face)

        return x0, x1, x2

    def forward_batch(self, f0: torch.Tensor, K) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batched lifting: f0 (B, n0, f_in) -> (B, n0, C), (B, n1, C), (B, n2, C).
        Geometric invariants depend only on K (shared); MLPs broadcast over batch.
        """
        pos, edges, faces = K.pos, K.edges, K.faces
        B = f0.size(0)

        edge_lengths = compute_edge_lengths(pos, edges)
        degrees = compute_node_degrees(edges, K.n0)
        avg_dist_src = scatter_mean(edge_lengths, edges[:, 0], dim=0, dim_size=K.n0)
        avg_dist_dst = scatter_mean(edge_lengths, edges[:, 1], dim=0, dim_size=K.n0)
        avg_dist = (avg_dist_src + avg_dist_dst) / 2

        x0_input = torch.cat([
            f0,
            degrees.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1),
            avg_dist.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1),
        ], dim=-1)
        x0 = self.node_mlp(x0_input)

        x0_i = x0[:, edges[:, 0]]
        x0_j = x0[:, edges[:, 1]]
        deg_i = degrees[edges[:, 0]].unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)
        deg_j = degrees[edges[:, 1]].unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)
        el = edge_lengths.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)

        feat_sym = torch.cat([x0_i + x0_j, el, deg_i + deg_j], dim=-1)
        feat_asym = torch.cat([x0_i - x0_j, el, deg_i - deg_j], dim=-1)
        x1 = self.edge_mlp_sym(feat_sym) + self.edge_mlp_asym(feat_asym)

        edge_feats_sum = self._gather_face_edge_features_batch(x1, K)
        areas = compute_face_areas(pos, faces).unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)
        angles = compute_face_angles(pos, faces).unsqueeze(0).expand(B, -1, -1)
        feat_face = torch.cat([edge_feats_sum, areas, angles], dim=-1)
        x2 = self.face_mlp(feat_face)

        return x0, x1, x2

    def _gather_face_edge_features(
        self, x1: torch.Tensor, faces: torch.Tensor,
        edges: torch.Tensor, K
    ) -> torch.Tensor:
        """
        Sum the three oriented edge features for each face,
        using the sparsity pattern of d1 to locate edge-face incidence.
        """
        d1 = K.d1.coalesce()
        indices = d1.indices()
        values = d1.values()

        face_idx = indices[0]
        edge_idx = indices[1]
        signs = values  # +/-1

        # Signed edge features (orientation-consistent with the face)
        edge_feats = x1[edge_idx] * signs.unsqueeze(-1)

        result = torch.zeros(K.n2, self.C, dtype=x1.dtype, device=x1.device)
        result.scatter_add_(0, face_idx.unsqueeze(-1).expand_as(edge_feats), edge_feats)
        return result

    def _gather_face_edge_features_batch(self, x1: torch.Tensor, K) -> torch.Tensor:
        """Batched version: x1 (B, n1, C) -> (B, n2, C)."""
        d1 = K.d1.coalesce()
        indices = d1.indices()
        face_idx = indices[0]
        edge_idx = indices[1]
        signs = d1.values()

        B = x1.size(0)
        edge_feats = x1[:, edge_idx] * signs.unsqueeze(0).unsqueeze(-1)

        result = torch.zeros(B, K.n2, self.C, dtype=x1.dtype, device=x1.device)
        fi_exp = face_idx.unsqueeze(0).unsqueeze(-1).expand(B, -1, self.C)
        result.scatter_add_(1, fi_exp, edge_feats)
        return result
