# Vendored v1 code.  Original: module `gauge_hodge_mp.cell_complex` of the RHMP v1 code base,
#   https://github.com/ContinuumCoder/Riemannian-Hodge-Message-Passing
# which accompanies "Learning Discrete Riemannian Metrics for Physical Fields with Cochain-Frame Equivariance"
# (Zheng & Allen-Blanchette, arXiv:2608.14556).  Kept verbatim (only this header was added).
# Used by rhmp.baselines (the v1 baselines and the v1 model `ours_v1`) and by `python -m rhmp.train --eval-v1`.
"""
CW complex data structure.

Stores vertices, edges, faces (optionally volumes), builds coboundary
operators d0, d1 as sparse matrices, and verifies the Hodge axiom d1 @ d0 = 0.
"""

import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class CellComplex:
    """
    Regular CW complex.

    Attributes:
        pos: (n0, dim) vertex coordinates
        edges: (n1, 2) directed edges [src, dst]
        faces: (n2, 3) triangular faces [v0, v1, v2]
        d0: (n1, n0) sparse coboundary operator  d_0: 0-cochains -> 1-cochains
        d1: (n2, n1) sparse coboundary operator  d_1: 1-cochains -> 2-cochains
        n0, n1, n2: number of cells per dimension
    """
    pos: torch.Tensor
    edges: torch.Tensor
    faces: torch.Tensor
    d0: torch.Tensor          # (n1, n0) sparse
    d1: torch.Tensor          # (n2, n1) sparse
    n0: int
    n1: int
    n2: int

    @staticmethod
    def from_triangulation(pos: torch.Tensor, faces: torch.Tensor,
                           device: str = "cpu") -> "CellComplex":
        """
        Build a CW complex from a triangular mesh.

        Args:
            pos: (n0, dim) vertex coordinates
            faces: (n2, 3) triangle face indices
            device: target device

        Returns:
            CellComplex with d0, d1 satisfying d1 @ d0 = 0
        """
        pos = pos.to(device)
        faces = faces.to(device)
        n0 = pos.size(0)
        n2 = faces.size(0)

        # Extract directed edges (vectorised, no Python loop)
        # Half-edges from triangles: (v0,v1), (v1,v2), (v2,v0)
        v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
        all_src = torch.cat([v0, v1, v2])
        all_dst = torch.cat([v1, v2, v0])
        # Canonical orientation: min -> max
        e_min = torch.min(all_src, all_dst)
        e_max = torch.max(all_src, all_dst)
        # Unique key per edge: min * n0 + max
        edge_keys = e_min * n0 + e_max
        unique_keys, half_to_edge = torch.unique(edge_keys, return_inverse=True)
        n1 = unique_keys.size(0)
        edges = torch.stack([unique_keys // n0, unique_keys % n0], dim=1).to(device)

        # Build d0: (n1, n0)
        row_d0 = torch.arange(n1, device=device).repeat_interleave(2)
        col_d0 = edges.flatten()
        val_d0 = torch.tensor([-1.0, 1.0], device=device).repeat(n1)
        d0 = torch.sparse_coo_tensor(
            torch.stack([row_d0, col_d0]), val_d0, size=(n1, n0)
        ).coalesce()

        # Build d1: (n2, n1), vectorised
        face_indices = torch.arange(n2, device=device).repeat(3)
        edge_indices = half_to_edge.to(device)
        # Sign: +1 if half-edge agrees with stored orientation (src < dst), else -1
        signs = torch.where(all_src.to(device) == e_min.to(device),
                           torch.ones(3*n2, device=device),
                           -torch.ones(3*n2, device=device))

        d1 = torch.sparse_coo_tensor(
            torch.stack([face_indices, edge_indices]),
            signs.float(),
            size=(n2, n1)
        ).coalesce()

        # Verify Hodge axiom d1 @ d0 = 0
        check = torch.sparse.mm(d1, d0)
        if check.is_sparse:
            check_dense = check.to_dense()
        else:
            check_dense = check
        assert check_dense.abs().max().item() < 1e-6, \
            f"Hodge axiom violated! d1@d0 max={check_dense.abs().max().item():.2e}"

        return CellComplex(
            pos=pos, edges=edges, faces=faces,
            d0=d0, d1=d1,
            n0=n0, n1=n1, n2=n2,
        )

    def to(self, device: str) -> "CellComplex":
        """Move all tensors to the specified device."""
        return CellComplex(
            pos=self.pos.to(device),
            edges=self.edges.to(device),
            faces=self.faces.to(device),
            d0=self.d0.to(device),
            d1=self.d1.to(device),
            n0=self.n0, n1=self.n1, n2=self.n2,
        )

    def get_adjacency(self, dim_from: int, dim_to: int) -> torch.Tensor:
        """
        Return adjacency as a COO edge_index tensor.
        Adjacency dim_from -> dim_to via shared (dim-1)-cells or coboundary.
        """
        if dim_from == 0 and dim_to == 0:
            # Node-node adjacency (via shared edges)
            src = self.edges[:, 0]
            dst = self.edges[:, 1]
            return torch.stack([
                torch.cat([src, dst]),
                torch.cat([dst, src])
            ], dim=0)
        elif dim_from == 1 and dim_to == 1:
            # Edge-edge adjacency (via shared faces)
            d1_dense = self.d1.to_dense()
            adj = (d1_dense.T @ d1_dense).abs()
            adj.fill_diagonal_(0)
            idx = adj.nonzero(as_tuple=False).T
            return idx
        else:
            raise NotImplementedError(f"Adjacency {dim_from}->{dim_to} not implemented")
