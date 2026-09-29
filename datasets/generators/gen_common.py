"""Shared helpers for the v2 dataset generators (numpy / scipy / torch only; no rhmp imports).

* ``delaunay_1024()``          : the seeded 1024-node Delaunay mesh shared by T5/T6/T7 (``np.random.seed(42)``).
* ``edge_face_incidence()``    : canonical edges (src < dst, lexicographically sorted, as the v1 generators) and the
                                  face -> edge incidence (index, sign) in the face orientation given by ``faces``.
* ``encode_edges_to_nodes()``  : vectorised version of the v1 node encoding ``(avg, avg*dx, avg*dy)``.
* ``encode_faces_to_nodes()``  : vectorised version of the v1 face -> node averaging.
* ``p1_stiffness_2d/3d``        : vectorised P1 FEM stiffness assembly with per-element (optionally anisotropic)
                                  conductivity, lumped mass.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
from scipy.spatial import Delaunay


# ------------------------------------------------------------------------------------------------------------------
# meshes / incidence
# ------------------------------------------------------------------------------------------------------------------
def delaunay_1024(seed: int = 42, n_pts: int = 1024):
    """Seeded random Delaunay mesh of the unit square used by the v1 T5/T6/T7 generators.

    Returns:
        pts ``(n_pts, 2)`` float64, faces ``(n2, 3)`` int64 (scipy simplex order and orientation).
    """
    np.random.seed(seed)
    pts = np.random.rand(n_pts, 2).astype(np.float64)
    faces = Delaunay(pts).simplices.astype(np.int64)
    return pts, faces


def edge_face_incidence(faces: np.ndarray, n_pts: int):
    """Canonical edges and oriented face-edge incidence (vectorised, identical to the v1 generator loops).

    Args:
        faces: ``(n2, 3)`` int.
    Returns:
        edges ``(n1, 2)`` int64 (src < dst, sorted lexicographically), face_edge_idx ``(n2, 3)`` int64 (edge of the
        half-edge ``(f[k], f[k+1])``), face_edge_sign ``(n2, 3)`` float32 (+1 if the half-edge agrees with src->dst).
    """
    a = faces
    b = np.roll(faces, -1, axis=1)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    keys = lo.astype(np.int64) * n_pts + hi
    uniq, inv = np.unique(keys.ravel(), return_inverse=True)
    edges = np.stack([uniq // n_pts, uniq % n_pts], axis=1).astype(np.int64)
    face_edge_idx = inv.reshape(faces.shape).astype(np.int64)
    face_edge_sign = np.where(a == lo, 1.0, -1.0).astype(np.float32)
    return edges, face_edge_idx, face_edge_sign


def encode_edges_to_nodes(A_edges: torch.Tensor, edges: np.ndarray, pts: np.ndarray, n_pts: int) -> torch.Tensor:
    """v1 node encoding of edge values: per node the mean over incident edges of ``(A, A*dx, A*dy)``.

    ``(dx, dy)`` is the unit edge direction src->dst (float32).  Vectorised (index_add) version of the v1 loop:
    identical up to float32 summation order (~1e-7).

    Args:
        A_edges: ``(N, n1)`` or ``(N, n1, c)``.
    Returns:
        ``(N, n_pts, 3c)`` ordered ``[avg(c), avg*dx(c), avg*dy(c)]`` as in v1.
    """
    if A_edges.dim() == 2:
        A_edges = A_edges.unsqueeze(-1)
    dev = A_edges.device
    N, n1, c = A_edges.shape
    edge_dir = pts[edges[:, 1]] - pts[edges[:, 0]]
    edge_len = np.linalg.norm(edge_dir, axis=1, keepdims=True) + 1e-12
    edge_dir_norm = edge_dir / edge_len
    dx = torch.tensor(edge_dir_norm[:, 0], dtype=torch.float32, device=dev)
    dy = torch.tensor(edge_dir_norm[:, 1], dtype=torch.float32, device=dev)
    src = torch.as_tensor(edges[:, 0], device=dev)
    dst = torch.as_tensor(edges[:, 1], device=dev)
    feats = torch.cat([A_edges, A_edges * dx[None, :, None], A_edges * dy[None, :, None]], dim=-1)  # (N, n1, 3c)
    out = torch.zeros(N, n_pts, 3 * c, device=dev)
    out.index_add_(1, src, feats)
    out.index_add_(1, dst, feats)
    count = torch.zeros(n_pts, device=dev)
    count.index_add_(0, src, torch.ones(n1, device=dev))
    count.index_add_(0, dst, torch.ones(n1, device=dev))
    count[count == 0] = 1
    return out / count[None, :, None]


def encode_faces_to_nodes(F_faces: torch.Tensor, faces: np.ndarray, n_pts: int) -> torch.Tensor:
    """v1 face -> node averaging (mean over incident faces).

    Args:
        F_faces: ``(N, n2)`` or ``(N, n2, c)``.
    Returns:
        ``(N, n_pts, c)``.
    """
    if F_faces.dim() == 2:
        F_faces = F_faces.unsqueeze(-1)
    dev = F_faces.device
    N, n2, c = F_faces.shape
    out = torch.zeros(N, n_pts, c, device=dev)
    count = torch.zeros(n_pts, device=dev)
    for k in range(faces.shape[1]):
        v = torch.as_tensor(faces[:, k], device=dev)
        out.index_add_(1, v, F_faces)
        count.index_add_(0, v, torch.ones(n2, device=dev))
    count[count == 0] = 1
    return out / count[None, :, None]


# ------------------------------------------------------------------------------------------------------------------
# P1 finite elements
# ------------------------------------------------------------------------------------------------------------------
def p1_stiffness_2d(pts: np.ndarray, tris: np.ndarray, sigma_T: np.ndarray, A_T: np.ndarray | None = None):
    """P1 stiffness ``K_ij = sum_T |T| sigma_T grad(phi_i)^T A_T grad(phi_j)`` and lumped mass (vectorised).

    Args:
        pts ``(n, 2)``, tris ``(m, 3)``, sigma_T ``(m,)`` > 0, A_T optional ``(m, 2, 2)`` SPD anisotropy tensors.
    Returns:
        K (csr ``(n, n)``), lumped mass ``(n,)``, triangle areas ``(m,)``.
    """
    n = pts.shape[0]
    P = pts[tris]                                   # (m, 3, 2)
    J = np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]], axis=2)   # (m, 2, 2) columns e1, e2
    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    area = 0.5 * np.abs(det)
    Jinv = np.linalg.inv(J)                         # (m, 2, 2)
    G = np.zeros((tris.shape[0], 3, 2))
    G[:, 1:, :] = Jinv                              # rows: grad(lambda_1), grad(lambda_2)
    G[:, 0, :] = -G[:, 1, :] - G[:, 2, :]
    if A_T is None:
        Kloc = np.einsum("mid,mjd->mij", G, G)
    else:
        Kloc = np.einsum("mid,mde,mje->mij", G, A_T, G)
    Kloc *= (area * sigma_T)[:, None, None]
    rows = np.repeat(tris, 3, axis=1).ravel()
    cols = np.tile(tris, (1, 3)).ravel()
    K = sp.csr_matrix((Kloc.ravel(), (rows, cols)), shape=(n, n))
    mass = np.zeros(n)
    np.add.at(mass, tris.ravel(), np.repeat(area / 3.0, 3))
    return K, mass, area


def p1_stiffness_3d(pts: np.ndarray, tets: np.ndarray, sigma_T: np.ndarray):
    """P1 stiffness on tetrahedra with per-element conductivity, lumped mass (vectorised).

    Args:
        pts ``(n, 3)``, tets ``(m, 4)``, sigma_T ``(m,)`` > 0.
    Returns:
        K (csr ``(n, n)``), lumped mass ``(n,)``, tet volumes ``(m,)``.
    """
    n = pts.shape[0]
    P = pts[tets]                                   # (m, 4, 3)
    J = np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]], axis=2)  # (m, 3, 3)
    vol = np.abs(np.linalg.det(J)) / 6.0
    Jinv = np.linalg.inv(J)
    G = np.zeros((tets.shape[0], 4, 3))
    G[:, 1:, :] = Jinv
    G[:, 0, :] = -G[:, 1:, :].sum(1)
    Kloc = np.einsum("mid,mjd->mij", G, G) * (vol * sigma_T)[:, None, None]
    rows = np.repeat(tets, 4, axis=1).ravel()
    cols = np.tile(tets, (1, 4)).ravel()
    K = sp.csr_matrix((Kloc.ravel(), (rows, cols)), shape=(n, n))
    mass = np.zeros(n)
    np.add.at(mass, tets.ravel(), np.repeat(vol / 4.0, 4))
    return K, mass, vol


def mesh_edges(cells: np.ndarray, n: int) -> np.ndarray:
    """Unique canonical edges (src < dst, lexicographic) of triangles ``(m, 3)`` or tetrahedra ``(m, 4)``."""
    k = cells.shape[1]
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    a = np.concatenate([cells[:, i] for i, _ in pairs])
    b = np.concatenate([cells[:, j] for _, j in pairs])
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    uniq = np.unique(lo.astype(np.int64) * n + hi)
    return np.stack([uniq // n, uniq % n], axis=1).astype(np.int64)


def mesh_faces_of_tets(tets: np.ndarray, n: int) -> np.ndarray:
    """Unique triangles (sorted vertex triples, lexicographic) of a tetrahedral mesh ``(m, 4)``."""
    tri = np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]])
    tri = np.sort(tri, axis=1).astype(np.int64)
    return np.unique(tri, axis=0)
