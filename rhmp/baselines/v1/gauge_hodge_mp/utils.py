# Vendored v1 code.  Original: module `gauge_hodge_mp.utils` of the RHMP v1 code base,
#   https://github.com/ContinuumCoder/Riemannian-Hodge-Message-Passing
# which accompanies "Learning Discrete Riemannian Metrics for Physical Fields with Cochain-Frame Equivariance"
# (Zheng & Allen-Blanchette, arXiv:2608.14556).  Kept verbatim (only this header was added).
# Used by rhmp.baselines (the v1 baselines and the v1 model `ours_v1`) and by `python -m rhmp.train --eval-v1`.
"""
Utility functions:
- SPD matrix construction (Cholesky parameterisation)
- Sparse matrix operations (spmm)
- Geometric invariant computation (edge lengths, areas, angles)
- Scatter operations (pure PyTorch, no PyG dependency)
"""

import torch
import torch.nn.functional as F


# ============================================================
# Scatter operations (replacing torch_scatter)
# ============================================================

def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                dim_size: int | None = None) -> torch.Tensor:
    """Grouped sum along dim according to index."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    shape = list(src.shape)
    shape[dim] = dim_size
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    idx = index.unsqueeze(-1).expand_as(src) if src.dim() > 1 else index
    return out.scatter_add_(dim, idx, src)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                 dim_size: int | None = None) -> torch.Tensor:
    """Grouped mean along dim according to index."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    s = scatter_sum(src, index, dim, dim_size)
    ones = torch.ones(src.shape[dim], dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, dim=0, dim_size=dim_size).clamp(min=1)
    if s.dim() > 1:
        count = count.unsqueeze(-1)
    return s / count


# ============================================================
# Sparse matrix multiplication
# ============================================================

def spmm(sparse: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
    """Sparse x dense matrix multiply (CSR or COO)."""
    return torch.sparse.mm(sparse, dense)


def batch_spmm(sparse: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
    """
    Batched sparse x dense: sparse (m, n) @ dense (B, n, C) -> (B, m, C).
    Folds the batch into the feature dimension via spmm linearity.
    """
    if dense.dim() == 2:
        return spmm(sparse, dense)
    B, n, C = dense.shape
    flat = dense.permute(1, 0, 2).reshape(n, B * C)  # (n, B*C)
    out = spmm(sparse, flat)  # (m, B*C)
    m = out.size(0)
    return out.reshape(m, B, C).permute(1, 0, 2)  # (B, m, C)


# ============================================================
# SPD matrix construction
# ============================================================

def unflatten_lower_triangular(flat: torch.Tensor, n: int) -> torch.Tensor:
    """
    Fill a flat vector into an n x n lower-triangular matrix L.
    flat: (..., n*(n+1)//2) -> (..., n, n)
    """
    batch_shape = flat.shape[:-1]
    L = torch.zeros(*batch_shape, n, n, dtype=flat.dtype, device=flat.device)
    idx = torch.tril_indices(n, n, device=flat.device)
    L[..., idx[0], idx[1]] = flat
    return L


def build_spd(L_flat: torch.Tensor, n: int, eps: float = 1e-3) -> torch.Tensor:
    """
    Construct an SPD matrix from a flat vector: H = L L^T + eps * I.
    L_flat: (..., n*(n+1)//2) -> (..., n, n) positive-definite matrix.

    L values are clamped via tanh to [-1, 1] for numerical stability.
    """
    L = unflatten_lower_triangular(torch.tanh(L_flat), n)
    H = L @ L.transpose(-2, -1) + eps * torch.eye(n, dtype=L.dtype, device=L.device)
    return H


# ============================================================
# Geometric invariants (all E(n)-invariant)
# ============================================================

def compute_edge_lengths(pos: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """
    Edge lengths ||r_j - r_i|| (E(n)-invariant).
    pos: (n_vertices, dim), edges: (n_edges, 2) -> (n_edges,)
    """
    diff = pos[edges[:, 1]] - pos[edges[:, 0]]
    return diff.norm(dim=-1)


def compute_edge_vectors(pos: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """Edge direction vectors r_j - r_i (E(n)-equivariant, used in reconstruction)."""
    return pos[edges[:, 1]] - pos[edges[:, 0]]


def compute_face_areas(pos: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Triangle area = 0.5 * ||(r_j - r_i) x (r_k - r_i)|| (E(n)-invariant).
    faces: (n_faces, 3) -> (n_faces,)
    """
    v0, v1, v2 = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    # 2D support: pad to 3D for cross product
    if e1.shape[-1] == 2:
        z = torch.zeros(e1.shape[0], 1, device=e1.device, dtype=e1.dtype)
        e1 = torch.cat([e1, z], dim=-1)
        e2 = torch.cat([e2, z], dim=-1)
    cross = torch.cross(e1, e2, dim=-1)
    return 0.5 * cross.norm(dim=-1)


def compute_face_angles(pos: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Three interior angles of each triangle (E(n)-invariant).
    faces: (n_faces, 3) -> (n_faces, 3) in radians.
    """
    v0, v1, v2 = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]

    def _angle(a, b, c):
        ba = a - b
        bc = c - b
        cos = (ba * bc).sum(dim=-1) / (ba.norm(dim=-1) * bc.norm(dim=-1) + 1e-8)
        return torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))

    a0 = _angle(v1, v0, v2)
    a1 = _angle(v0, v1, v2)
    a2 = _angle(v0, v2, v1)
    return torch.stack([a0, a1, a2], dim=-1)


def compute_node_degrees(edges: torch.Tensor, n_vertices: int) -> torch.Tensor:
    """Node degree (E(n)-invariant)."""
    deg = torch.zeros(n_vertices, dtype=torch.float, device=edges.device)
    deg.scatter_add_(0, edges[:, 0], torch.ones(edges.size(0), device=edges.device))
    deg.scatter_add_(0, edges[:, 1], torch.ones(edges.size(0), device=edges.device))
    return deg
