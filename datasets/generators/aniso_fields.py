"""Anisotropic material fields, Whitney/Galerkin (FEEC) element matrices and representability diagnostics for the
anisotropy task generator ``gen_aniso.py`` (numpy / scipy only; no rhmp imports).

Random fields (mesh independent, so that the 4x-finer test meshes see the same physical instance):
  * scalar GRFs by random Fourier features, squashed by ``tanh(g / sigma_g)`` with the exact standard deviation of the
    drawn feature sum, so no mesh-dependent min/max normalisation is needed;
  * smooth random vector fields ``V(x)`` (shared frequencies, vector amplitudes) that define the principal direction of
    the material tensors: ``tau = V / |V|`` (3-D, planar) or the normalised tangential projection ``P_T V`` (surfaces).
    Near zeros of ``V`` the anisotropy fades out (``w = |V|^2 / (|V|^2 + delta^2)``, eigenvalue ratio
    ``r_eff = 1 + (r - 1) w``), which makes the tensor field continuous everywhere (defects of line fields are
    unavoidable on spheres).

Material tensors (unit determinant times a scalar magnitude):
  * planar / surface (tangent plane): ``A = r_eff^{-1/2} (P + (r_eff - 1) tau tau^T)``, eigenvalues ``sqrt(r_eff)``
    along ``tau`` and ``1/sqrt(r_eff)`` across (``P`` = identity in 2-D, tangent projector on surfaces);
  * volumes: ``A = r_eff^{-1/3} (I + (r_eff - 1) tau tau^T)``, eigenvalues ``r_eff^{2/3}, r_eff^{-1/3}, r_eff^{-1/3}``.

FEEC element matrices (float64, exact degree-2 quadrature, the same bases and orientations as ``rhmp.geometry``):
  * Whitney 1-forms ``w_ab = lam_a grad lam_b - lam_b grad lam_a`` (triangles: slots (0,1),(1,2),(2,0); tets: the six
    pairs (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)), signed to the canonical edge orientation ``src < dst``;
  * Whitney 2-forms of tets ``w_abc = 2 (lam_a grad lam_b x grad lam_c + cyclic)`` with ``(a, b, c)`` the face's vertices
    in increasing global id (face i omits local vertex i);
  * P1 stiffness ``K_T = |T| G^T Sigma_T G`` (= ``d0^T M1(Sigma) d0``), lumped masses.

Representability diagnostics of a per-cell tensor ``S`` against the model's tensor families
(``t_j`` = unit edge vectors of the cell, whose dyads ``t_j t_j^T`` span Sym(2) on a triangle / Sym(3) on a tet):
  * ``dyad_coeffs``: the unique coefficients ``S = sum_j alpha_j t_j t_j^T``;
  * ``cone_membership``: ``S`` is in the cone ``{b I + sum_j a_j t_j t_j^T : b > 0, a >= 0}`` (the ``tensor_param='cone'``
    family) iff an interval of ``b > 0`` with ``alpha - b gamma >= 0`` exists (``I = sum_j gamma_j t_j t_j^T``);
  * ``full_param_range``: the smallest ``max_j |s_j|`` with ``S = b expm(sum_j s_j t_j t_j^T)`` (the
    ``tensor_param='full'`` family; ``b`` absorbs the isotropic part);
  * ``cone_project``: Frobenius-nearest cone member (NNLS per cell; analysis only).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

N_RFF = 64                     # random Fourier features per field
TRI_SLOTS = ((0, 1), (1, 2), (2, 0))                            # local edge j of a triangle joins vertices j, j+1
TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
TET_FACES = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))        # face i omits local vertex i
TRI_Q = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])            # edge midpoints, weights |f|/3
_QA, _QB = (5.0 + 3.0 * 5.0 ** 0.5) / 20.0, (5.0 - 5.0 ** 0.5) / 20.0
TET_Q = np.array([[_QA if j == i else _QB for j in range(4)] for i in range(4)])   # 4-point degree-2 rule, |T|/4
TINY = 1e-300


# ======================================================================================================================
# random fields
# ======================================================================================================================
def rff_draw(rng: np.random.Generator, dim: int, ell: float, ncomp: int = 1) -> dict:
    """Random Fourier features of a squared-exponential GRF (length scale ``ell``) with ``ncomp`` components sharing
    frequencies and phases.  Each component has unit variance in expectation."""
    k = rng.normal(0.0, 1.0 / (2 * np.pi * ell), size=(N_RFF, dim))
    ph = rng.uniform(0.0, 2 * np.pi, N_RFF)
    a = rng.normal(0.0, 1.0, (N_RFF, ncomp)) * np.sqrt(2.0 / N_RFF)
    return dict(k=k, ph=ph, a=a, ell=float(ell))


def rff_eval(p: dict, x: np.ndarray) -> np.ndarray:
    """``(M, dim) -> (M, ncomp)``."""
    return np.cos(2 * np.pi * x @ p["k"].T + p["ph"]) @ p["a"]


def rff_std(p: dict) -> float:
    """Spatial standard deviation of one component of the drawn feature sum (``sqrt(sum a^2 / 2)``, averaged over
    components): a mesh-independent normalisation."""
    return float(np.sqrt((p["a"] ** 2).sum(0).mean() / 2.0))


def squashed(p: dict, x: np.ndarray) -> np.ndarray:
    """Scalar field in (-1, 1): ``tanh(g / sigma_g)``, ``(M,)``."""
    return np.tanh(rff_eval(p, x)[:, 0] / max(rff_std(p), 1e-12))


def log_magnitude(p: dict, x: np.ndarray, contrast: float) -> np.ndarray:
    """Log-normal-type magnitude ``log s = 0.5 ln(contrast) tanh(g / sigma_g)``: ``s`` in (contrast^-1/2, contrast^1/2)."""
    return 0.5 * np.log(contrast) * squashed(p, x)


def unit_vector_field(p: dict, x: np.ndarray) -> np.ndarray:
    """Smooth random vector field normalised to unit component variance, ``(M, ncomp)``."""
    return rff_eval(p, x) / max(rff_std(p), 1e-12)


# ======================================================================================================================
# material tensors
# ======================================================================================================================
def uniaxial_planar(V: np.ndarray, r: float, delta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit-determinant 2-D tensors with major axis ``V / |V|`` (anisotropy fading out where ``|V| << delta``).

    Args:
        V: ``(M, 2)``.
    Returns:
        ``A (M, 2, 2)``, ``tau (M, 2)`` unit major axis, ``r_eff (M,)`` eigenvalue ratio.
    """
    v2 = (V * V).sum(1)
    w = v2 / (v2 + delta ** 2)
    tau = V / np.sqrt(np.maximum(v2, TINY))[:, None]
    r_eff = 1.0 + (r - 1.0) * w
    dyad = tau[:, :, None] * tau[:, None, :]
    A = (np.eye(2)[None] + (r_eff - 1.0)[:, None, None] * dyad) / np.sqrt(r_eff)[:, None, None]
    return A, tau, r_eff


def uniaxial_tangent(V: np.ndarray, nrm: np.ndarray, r: float, delta: float
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tangent-plane tensors of a surface: ``A = r_eff^{-1/2} (P + (r_eff - 1) tau tau^T)`` with ``tau`` the normalised
    tangential projection of ``V`` (``P = I - n n^T``); tangential determinant 1, zero normal-normal part.

    Args:
        V: ``(M, 3)``; nrm: ``(M, 3)`` unit normals.
    Returns:
        ``A (M, 3, 3)``, ``tau (M, 3)``, ``r_eff (M,)``.
    """
    P = np.eye(3)[None] - nrm[:, :, None] * nrm[:, None, :]
    Vt = np.einsum("mij,mj->mi", P, V)
    v2 = (Vt * Vt).sum(1)
    w = v2 / (v2 + delta ** 2)
    tau = Vt / np.sqrt(np.maximum(v2, TINY))[:, None]
    r_eff = 1.0 + (r - 1.0) * w
    A = (P + (r_eff - 1.0)[:, None, None] * tau[:, :, None] * tau[:, None, :]) / np.sqrt(r_eff)[:, None, None]
    return A, tau, r_eff


def uniaxial_3d(V: np.ndarray, r: float, delta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit-determinant prolate 3-D tensors along ``tau = V / |V|``: ``A = r_eff^{-1/3} (I + (r_eff - 1) tau tau^T)``.

    Returns:
        ``A (M, 3, 3)``, ``tau (M, 3)``, ``r_eff (M,)``.
    """
    v2 = (V * V).sum(1)
    w = v2 / (v2 + delta ** 2)
    tau = V / np.sqrt(np.maximum(v2, TINY))[:, None]
    r_eff = 1.0 + (r - 1.0) * w
    dyad = tau[:, :, None] * tau[:, None, :]
    A = (np.eye(3)[None] + (r_eff - 1.0)[:, None, None] * dyad) / np.cbrt(r_eff)[:, None, None]
    return A, tau, r_eff


def sym_pack(S: np.ndarray) -> np.ndarray:
    """``(M, D, D)`` symmetric -> ``(M, 3)`` ``[xx, xy, yy]`` (D = 2) or ``(M, 6)`` ``[xx, xy, xz, yy, yz, zz]``."""
    if S.shape[-1] == 2:
        return np.stack([S[:, 0, 0], S[:, 0, 1], S[:, 1, 1]], 1)
    return np.stack([S[:, 0, 0], S[:, 0, 1], S[:, 0, 2], S[:, 1, 1], S[:, 1, 2], S[:, 2, 2]], 1)


def sym_unpack(v: np.ndarray) -> np.ndarray:
    """Inverse of :func:`sym_pack`."""
    v = np.asarray(v, dtype=np.float64)
    if v.shape[-1] == 3:
        return np.stack([np.stack([v[:, 0], v[:, 1]], -1), np.stack([v[:, 1], v[:, 2]], -1)], -2)
    return np.stack([np.stack([v[:, 0], v[:, 1], v[:, 2]], -1), np.stack([v[:, 1], v[:, 3], v[:, 4]], -1),
                     np.stack([v[:, 2], v[:, 4], v[:, 5]], -1)], -2)


# ======================================================================================================================
# simplex geometry and topology
# ======================================================================================================================
def bary_grads(V: np.ndarray) -> np.ndarray:
    """Barycentric gradients of simplices ``V (n, d+1, D)`` in their affine hull -> ``(n, d+1, D)``."""
    E = V[:, 1:] - V[:, :1]
    G = np.einsum("nid,njd->nij", E, E)
    rest = np.linalg.solve(G, E)
    return np.concatenate([-rest.sum(1, keepdims=True), rest], 1)


def simplex_measure(V: np.ndarray) -> np.ndarray:
    """d-volume of simplices ``V (n, d+1, D)`` via the Gram determinant."""
    E = V[:, 1:] - V[:, :1]
    d = E.shape[1]
    return np.sqrt(np.maximum(np.linalg.det(np.einsum("nid,njd->nij", E, E)), 0.0)) / float(np.prod(range(1, d + 1)))


def canonical_edges(cells: np.ndarray, n: int) -> np.ndarray:
    """Unique canonical edges ``(n1, 2)`` (src < dst, lexicographic) of triangles or tets."""
    p = cells.shape[1]
    pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]
    a = np.concatenate([cells[:, i] for i, _ in pairs])
    b = np.concatenate([cells[:, j] for _, j in pairs])
    key = np.unique(np.minimum(a, b).astype(np.int64) * n + np.maximum(a, b))
    return np.stack([key // n, key % n], 1)


def edge_lookup(edges: np.ndarray, n: int, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Edge id and orientation sign (+1 if ``a -> b`` agrees with ``src -> dst``) of vertex pairs ``(a, b)``."""
    key = edges[:, 0].astype(np.int64) * n + edges[:, 1]
    q = np.minimum(a, b).astype(np.int64) * n + np.maximum(a, b)
    idx = np.searchsorted(key, q)
    if not np.array_equal(key[np.minimum(idx, len(key) - 1)], q):
        raise ValueError("edge_lookup: pair not in the edge list")
    return idx, np.where(a < b, 1.0, -1.0)


def tri_topology(faces: np.ndarray, n: int) -> dict:
    """Canonical edges and the local slot -> edge maps of a triangle mesh (faces keep their orientation).

    Returns:
        dict ``edges (n1, 2)``, ``face_edges (n2, 3)`` (slot j joins local vertices j, j+1), ``face_signs (n2, 3)``,
        ``d0`` csr ``(n1, n)``, ``d1`` csr ``(n2, n1)`` (oriented faces).
    """
    edges = canonical_edges(faces, n)
    fe, fs = [], []
    for a, b in TRI_SLOTS:
        e, s = edge_lookup(edges, n, faces[:, a], faces[:, b])
        fe.append(e)
        fs.append(s)
    fe, fs = np.stack(fe, 1), np.stack(fs, 1)
    m = len(edges)
    d0 = sp.csr_matrix((np.r_[-np.ones(m), np.ones(m)], (np.r_[np.arange(m), np.arange(m)], np.r_[edges[:, 0],
                                                                                                    edges[:, 1]])),
                       shape=(m, n))
    n2 = len(faces)
    d1 = sp.csr_matrix((fs.ravel(), (np.repeat(np.arange(n2), 3), fe.ravel())), shape=(n2, m))
    return dict(edges=edges, face_edges=fe, face_signs=fs, d0=d0, d1=d1)


def _perm_parity(p: np.ndarray) -> np.ndarray:
    """Parity (+1 even / -1 odd) of each row of a batch of permutations of ``0..k-1`` (inversion count)."""
    k = p.shape[1]
    inv = np.zeros(len(p), dtype=np.int64)
    for i in range(k):
        for j in range(i + 1, k):
            inv += (p[:, i] > p[:, j])
    return np.where(inv % 2 == 0, 1.0, -1.0)


def tet_topology(tets: np.ndarray, n: int) -> dict:
    """Canonical edges / faces and incidence of a positively oriented tet mesh.

    Returns:
        dict ``edges (n1, 2)``, ``faces (n2, 3)`` (sorted triples, lexicographic), ``tet_edges (n3, 6)`` and
        ``tet_edge_signs (n3, 6)`` for the local pairs ``TET_EDGES``, ``tet_faces (n3, 4)`` (face i omits local vertex i)
        and ``tet_face_signs (n3, 4)`` (orientation of face i in the boundary of the tet relative to its canonical
        sorted orientation), ``face_local (n3, 4, 3)`` local vertex ids of face i in increasing global id,
        ``d0 (n1, n)``, ``d1 (n2, n1)``, ``d2 (n3, n2)`` csr, ``boundary_faces`` / ``boundary_edges`` /
        ``boundary_nodes`` bool masks (faces with one tet, their edges and vertices).
    """
    n3 = len(tets)
    edges = canonical_edges(tets, n)
    te, ts = [], []
    for a, b in TET_EDGES:
        e, s = edge_lookup(edges, n, tets[:, a], tets[:, b])
        te.append(e)
        ts.append(s)
    te, ts = np.stack(te, 1), np.stack(ts, 1)
    loc = np.array(TET_FACES)                                     # (4, 3)
    glob = tets[:, loc]                                           # (n3, 4, 3) in local (increasing local id) order
    order = np.argsort(glob, axis=2)
    face_local = np.take_along_axis(np.broadcast_to(loc, glob.shape), order, 2)
    srt = np.take_along_axis(glob, order, 2)                      # sorted global ids
    key = (srt[..., 0].astype(np.int64) * n + srt[..., 1]) * n + srt[..., 2]
    ukey, inv = np.unique(key.ravel(), return_inverse=True)
    faces = np.stack([ukey // (n * n), (ukey // n) % n, ukey % n], 1)
    tf = inv.reshape(n3, 4)
    # boundary orientation: face i appears with sign (-1)^i in the order of TET_FACES[i]; relative to the sorted order
    # the sign changes by the parity of the sorting permutation
    par = _perm_parity(order.reshape(-1, 3)).reshape(n3, 4)
    tsgn = par * np.array([1.0, -1.0, 1.0, -1.0])[None]
    m, n2 = len(edges), len(faces)
    d0 = sp.csr_matrix((np.r_[-np.ones(m), np.ones(m)], (np.r_[np.arange(m), np.arange(m)], np.r_[edges[:, 0],
                                                                                                    edges[:, 1]])),
                       shape=(m, n))
    fe = np.stack([edge_lookup(edges, n, faces[:, a], faces[:, b])[0] for a, b in ((0, 1), (1, 2), (0, 2))], 1)
    d1 = sp.csr_matrix((np.tile([1.0, 1.0, -1.0], n2), (np.repeat(np.arange(n2), 3), fe.ravel())), shape=(n2, m))
    d2 = sp.csr_matrix((tsgn.ravel(), (np.repeat(np.arange(n3), 4), tf.ravel())), shape=(n3, n2))
    nf = np.bincount(tf.ravel(), minlength=n2)
    bf = nf == 1
    be = np.zeros(m, bool)
    be[fe[bf].ravel()] = True
    bn = np.zeros(n, bool)
    bn[faces[bf].ravel()] = True
    return dict(edges=edges, faces=faces, face_edge=fe, tet_edges=te, tet_edge_signs=ts, tet_faces=tf,
                tet_face_signs=tsgn, face_local=face_local, d0=d0, d1=d1, d2=d2, boundary_faces=bf,
                boundary_edges=be, boundary_nodes=bn)


# ======================================================================================================================
# FEEC element matrices and assembly
# ======================================================================================================================
def whitney1_values(lam: np.ndarray, grads: np.ndarray, pairs, signs: np.ndarray) -> np.ndarray:
    """Canonically signed Whitney 1-form basis at quadrature points: ``(n, Q, m, D)``."""
    pr = np.asarray(pairs)
    a, b = pr[:, 0], pr[:, 1]
    W = lam[None, :, a, None] * grads[:, None, b] - lam[None, :, b, None] * grads[:, None, a]
    return W * signs[:, None, :, None]


def whitney2_values_tet(lam: np.ndarray, grads: np.ndarray, face_local: np.ndarray) -> np.ndarray:
    """Whitney 2-form vector proxies of the 4 faces of every tet (canonical sorted orientation): ``(n, Q, 4, 3)``."""
    n = grads.shape[0]
    out = []
    for i in range(4):
        c = face_local[:, i]
        g = np.take_along_axis(grads, c[:, :, None].repeat(3, 2), 1)          # (n, 3, 3): grads of a, b, c
        lc = lam[:, c].transpose(1, 0, 2)                                     # (n, Q, 3)
        cr = np.stack([np.cross(g[:, 1], g[:, 2]), np.cross(g[:, 2], g[:, 0]), np.cross(g[:, 0], g[:, 1])], 1)
        out.append(2.0 * np.einsum("nqj,njd->nqd", lc, cr))
    return np.stack(out, 2)


def element_mass(W: np.ndarray, wq: np.ndarray, S: np.ndarray | None) -> np.ndarray:
    """``M_ij = sum_q wq_q W_qi^T S W_qj`` per cell: ``W (n, Q, m, D)``, ``wq (n, Q)``, ``S (n, D, D)`` or ``None`` (I)."""
    if S is None:
        M = np.einsum("nq,nqid,nqjd->nij", wq, W, W)
    else:
        M = np.einsum("nq,nqid,nde,nqje->nij", wq, W, S, W)
    return 0.5 * (M + M.transpose(0, 2, 1))


def assemble(local: np.ndarray, idx: np.ndarray, n: int) -> sp.csr_matrix:
    """Sum of per-cell blocks ``local (n_c, m, m)`` scattered to the global ids ``idx (n_c, m)``: csr ``(n, n)``."""
    m = idx.shape[1]
    rows = np.repeat(idx, m, axis=1).ravel()
    cols = np.tile(idx, (1, m)).ravel()
    return sp.csr_matrix((local.ravel(), (rows, cols)), shape=(n, n))


def tri_whitney1_mass(pos: np.ndarray, faces: np.ndarray, topo: dict, S: np.ndarray | None) -> np.ndarray:
    """Per-face Whitney 1-form Galerkin blocks ``(n2, 3, 3)`` (slot order of ``topo['face_edges']``)."""
    V = pos[faces]
    g = bary_grads(V)
    area = simplex_measure(V)
    W = whitney1_values(TRI_Q, g, TRI_SLOTS, topo["face_signs"])
    return element_mass(W, np.repeat((area / 3.0)[:, None], 3, 1), S)


def tet_whitney1_mass(pos: np.ndarray, tets: np.ndarray, topo: dict, S: np.ndarray | None) -> np.ndarray:
    """Per-tet Whitney 1-form Galerkin blocks ``(n3, 6, 6)`` (``TET_EDGES`` order)."""
    V = pos[tets]
    g = bary_grads(V)
    vol = simplex_measure(V)
    W = whitney1_values(TET_Q, g, TET_EDGES, topo["tet_edge_signs"])
    return element_mass(W, np.repeat((vol / 4.0)[:, None], 4, 1), S)


def tet_whitney2_mass(pos: np.ndarray, tets: np.ndarray, topo: dict, S: np.ndarray | None) -> np.ndarray:
    """Per-tet Whitney 2-form Galerkin blocks ``(n3, 4, 4)`` (face i omits local vertex i, canonical orientation)."""
    V = pos[tets]
    g = bary_grads(V)
    vol = simplex_measure(V)
    W = whitney2_values_tet(TET_Q, g, topo["face_local"])
    return element_mass(W, np.repeat((vol / 4.0)[:, None], 4, 1), S)


def p1_stiffness(pos: np.ndarray, cells: np.ndarray, S: np.ndarray) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """P1 stiffness ``sum_T |T| grad(phi)^T S_T grad(phi)`` (triangles in 2-D/3-D, tets), barycentric lumped mass and
    cell measures.  Equals ``d0^T M1(S) d0`` with the Whitney 1-form mass ``M1(S)``.

    Returns:
        ``K`` csr ``(n, n)``, ``mass (n,)``, ``measure (n_c,)``.
    """
    n = len(pos)
    V = pos[cells]
    g = bary_grads(V)
    meas = simplex_measure(V)
    Kl = np.einsum("nid,nde,nje->nij", g, S, g) * meas[:, None, None]
    Kl = 0.5 * (Kl + Kl.transpose(0, 2, 1))
    p = cells.shape[1]
    mass = np.zeros(n)
    np.add.at(mass, cells.ravel(), np.repeat(meas / p, p))
    return assemble(Kl, cells, n), mass, meas


def edge_weights(K: sp.csr_matrix, edges: np.ndarray) -> np.ndarray:
    """Graph-Laplacian edge weights ``w_e = -K_{src, dst}`` of a zero-row-sum stiffness matrix."""
    return -np.asarray(K[edges[:, 0], edges[:, 1]]).ravel()


def cell_edge_average(values: np.ndarray, cell_edges: np.ndarray, n1: int) -> np.ndarray:
    """Mean over the cells incident to every edge of per-(cell, local edge) values ``(n_c, m)`` -> ``(n1,)``."""
    s = np.zeros(n1)
    c = np.zeros(n1)
    np.add.at(s, cell_edges.ravel(), values.ravel())
    np.add.at(c, cell_edges.ravel(), 1.0)
    return s / np.maximum(c, 1.0)


# ======================================================================================================================
# representability diagnostics
# ======================================================================================================================
def local_frame(t: np.ndarray, nrm: np.ndarray | None) -> np.ndarray | None:
    """Orthonormal tangent frame ``(n, 3, 2)`` of surface cells (``e1`` = first direction, ``e2 = n x e1``); ``None``
    for planar / volume cells (their coordinates are used directly)."""
    if nrm is None:
        return None
    e1 = t[:, 0] / np.linalg.norm(t[:, 0], axis=1, keepdims=True)
    e2 = np.cross(nrm, e1)
    return np.stack([e1, e2], 2)


def _coords(S: np.ndarray, frame: np.ndarray | None) -> np.ndarray:
    """Tensors in the cell frame: ``(n, d, d)``."""
    return S if frame is None else np.einsum("nda,nde,neb->nab", frame, S, frame)


def dyad_matrix(t: np.ndarray, frame: np.ndarray | None = None) -> np.ndarray:
    """Columns = packed dyads ``t_j t_j^T`` in the cell frame: ``(n, P, m)`` with ``P = 3`` (2-D) or ``6`` (3-D)."""
    tc = t if frame is None else np.einsum("nda,njd->nja", frame, t)       # (n, m, d)
    dd = tc[:, :, :, None] * tc[:, :, None, :]                             # (n, m, d, d)
    n, m = dd.shape[:2]
    return np.stack([sym_pack(dd[:, j]) for j in range(m)], 2)


def dyad_coeffs(S: np.ndarray, t: np.ndarray, frame: np.ndarray | None = None) -> np.ndarray:
    """Unique coefficients ``alpha (n, m)`` with ``S = sum_j alpha_j t_j t_j^T`` (m = 3 on triangles, 6 on tets)."""
    A = dyad_matrix(t, frame)
    return np.linalg.solve(A, sym_pack(_coords(S, frame))[..., None])[..., 0]


def identity_coeffs(t: np.ndarray, frame: np.ndarray | None = None) -> np.ndarray:
    """``gamma (n, m)`` with ``I = sum_j gamma_j t_j t_j^T`` (tangent identity on surfaces)."""
    d = 2 if frame is not None else t.shape[-1]
    I = np.broadcast_to(np.eye(d), (t.shape[0], d, d))
    A = dyad_matrix(t, frame)
    return np.linalg.solve(A, sym_pack(np.ascontiguousarray(I))[..., None])[..., 0]


def cone_membership(alpha: np.ndarray, gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Is ``S = sum alpha_j t_j t_j^T`` of the form ``b I + sum a_j t_j t_j^T`` with ``b > 0, a >= 0``?

    Returns:
        ``(inside (n,) bool, b_lo, b_hi)``: the feasible interval ``b in (max(b_lo, 0), b_hi]``.
    """
    lo = np.zeros(len(alpha))
    hi = np.full(len(alpha), np.inf)
    for j in range(alpha.shape[1]):
        g, a = gamma[:, j], alpha[:, j]
        pos, neg = g > 0, g < 0
        hi[pos] = np.minimum(hi[pos], a[pos] / g[pos])
        lo[neg] = np.maximum(lo[neg], a[neg] / g[neg])
        bad = (g == 0) & (a < 0)
        hi[bad] = -np.inf
    return hi > np.maximum(lo, 0.0), lo, hi


def _sym_logm(S: np.ndarray) -> np.ndarray:
    ev, U = np.linalg.eigh(S)
    return np.einsum("nij,nj,nkj->nik", U, np.log(np.maximum(ev, TINY)), U)


def _sym_expm(S: np.ndarray) -> np.ndarray:
    ev, U = np.linalg.eigh(S)
    return np.einsum("nij,nj,nkj->nik", U, np.exp(ev), U)


def full_param_coeffs(S: np.ndarray, t: np.ndarray, frame: np.ndarray | None = None
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``S = b expm(sum_j s_j t_j t_j^T)`` with the isotropic part chosen to minimise ``max_j |s_j|``.

    Returns:
        ``(s (n, m), log_b (n,), s_max (n,))``.
    """
    Sc = _coords(S, frame)
    d = Sc.shape[-1]
    logdet = np.linalg.slogdet(Sc)[1]
    L = _sym_logm(Sc / np.exp(logdet / d)[:, None, None])                  # traceless log of the unit-det part
    A = dyad_matrix(t, frame)
    s0 = np.linalg.solve(A, sym_pack(L)[..., None])[..., 0]
    g = identity_coeffs(t, frame)
    # minimise f(c) = max_j |s0_j + c g_j| over c: the optimum is a crossing of two of the lines +-(s0_j + c g_j)
    m = s0.shape[1]
    cands = [np.zeros(len(s0))]
    for i in range(m):                                                     # zeros of single lines
        with np.errstate(divide="ignore", invalid="ignore"):
            cands.append(np.where(np.abs(g[:, i]) > 1e-14, -s0[:, i] / g[:, i], 0.0))
    for i in range(m):
        for j in range(i + 1, m):
            for sgn in (1.0, -1.0):
                den = g[:, i] - sgn * g[:, j]
                with np.errstate(divide="ignore", invalid="ignore"):
                    cands.append(np.where(np.abs(den) > 1e-14, -(s0[:, i] - sgn * s0[:, j]) / den, 0.0))
    C = np.stack(cands, 1)                                                 # (n, n_c)
    f = np.abs(s0[:, None, :] + C[:, :, None] * g[:, None, :]).max(-1)    # (n, n_c)
    k = f.argmin(1)
    c = C[np.arange(len(C)), k]
    s = s0 + c[:, None] * g
    return s, logdet / d - c, f[np.arange(len(f)), k]


def full_param_tensor(s: np.ndarray, log_b: np.ndarray, t: np.ndarray, frame: np.ndarray | None = None
                      ) -> np.ndarray:
    """Inverse of :func:`full_param_coeffs`: ``b expm(sum_j s_j t_j t_j^T)`` in world coordinates ``(n, D, D)``."""
    tc = t if frame is None else np.einsum("nda,njd->nja", frame, t)
    Sm = np.einsum("nj,nja,njb->nab", s, tc, tc)
    T = np.exp(log_b)[:, None, None] * _sym_expm(Sm)
    if frame is None:
        return T
    return np.einsum("nda,nab,neb->nde", frame, T, frame)


def cone_project(S: np.ndarray, t: np.ndarray, frame: np.ndarray | None = None) -> np.ndarray:
    """Frobenius-nearest member of the cone ``{b I + sum_j a_j t_j t_j^T : b, a >= 0}`` per cell (NNLS; slow, analysis
    only).  Returns world-coordinate tensors ``(n, D, D)``."""
    from scipy.optimize import nnls
    Sc = _coords(S, frame)
    d = Sc.shape[-1]
    w = np.array([1.0, np.sqrt(2.0), 1.0]) if d == 2 else np.array([1, np.sqrt(2), np.sqrt(2), 1, np.sqrt(2), 1.0])
    A = dyad_matrix(t, frame) * w[None, :, None]
    Ivec = sym_pack(np.broadcast_to(np.eye(d), Sc.shape).copy())[0] * w
    bvec = sym_pack(Sc) * w[None]
    tc = t if frame is None else np.einsum("nda,njd->nja", frame, t)
    out = np.empty_like(Sc)
    for i in range(len(Sc)):
        x, _ = nnls(np.concatenate([Ivec[:, None], A[i]], 1), bvec[i])
        out[i] = x[0] * np.eye(d) + np.einsum("j,ja,jb->ab", x[1:], tc[i], tc[i])
    if frame is None:
        return out
    return np.einsum("nda,nab,neb->nde", frame, out, frame)


def representability(S: np.ndarray, t: np.ndarray, nrm: np.ndarray | None = None) -> dict:
    """Per-cell diagnostics of the tensors ``S (n, D, D)`` in the edge frames ``t (n, m, D)`` (``nrm``: surface normals).

    Returns:
        dict ``in_cone (n,) bool`` (tensor_param='cone' can represent it), ``s_max (n,)`` (smallest ``max |s_j|`` of the
        tensor_param='full' family), ``alpha (n, m)`` signed dyad coefficients, ``gamma (n, m)``.
    """
    fr = local_frame(t, nrm)
    alpha = dyad_coeffs(S, t, fr)
    gamma = identity_coeffs(t, fr)
    inside, _, _ = cone_membership(alpha, gamma)
    _, _, smax = full_param_coeffs(S, t, fr)
    return dict(in_cone=inside, s_max=smax, alpha=alpha, gamma=gamma)
