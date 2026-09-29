"""Parametric closed-surface meshes and DEC/FEM operators for the SURF generator (numpy / scipy; no rhmp imports).

Surface families (all closed, consistently oriented counter-clockwise w.r.t. the outward normal, no duplicate
vertices, every grid quad split along a random diagonal so that connectivity differs from sample to sample):

* genus 0 (:func:`genus0_mesh`): equal-angle *cube-sphere* (six UV grids on the faces of a cube, seam vertices
  merged, ``V = 6 n^2 + 2``), randomly rotated, mapped radially onto a star-shaped surface ``x = r(w) w``
  (``w`` unit direction) and relaxed by tangential umbrella smoothing with radial re-projection (uniform edge
  lengths whatever the shape map).  Radial functions: :func:`ellipsoid_radius`, :func:`superquadric_radius`,
  :func:`sh_radius` (sphere with a low-order real spherical-harmonic perturbation).
* genus 1 (:func:`torus_mesh`): periodic ``nu x nv`` UV grid of a torus (ring radius ``R``, tube radius ``r``),
  optionally with a smooth low-order Fourier perturbation of the tube radius.
* genus 2 (:func:`double_torus_mesh`): two mirror-image torus grids, each with a ``2k x 2k``-cell hole at its
  outer equator, joined by a neck of rings; the neck rings follow cubic Hermite curves that leave both hole
  boundaries tangentially to the tori (G1 join) and the neck plus a few grid rings around each hole are relaxed by
  umbrella smoothing.  Pure grid construction: the mesh style (split quads, valence ~6) is the same as for the
  genus-0/1 families, so the genus-2 test isolates topology/geometry rather than meshing style.

Operators (:func:`cotan_operators`): cotan edge weights ``w_e = (cot a + cot b) / 2`` (unclamped; ``a, b`` the angles
opposite the edge), stiffness ``L = d0^T diag(w) d0`` (identical to the P1 FEM Laplace-Beltrami stiffness) and the
barycentric lumped mass ``M_ii = sum_{T ni i} |T| / 3`` (= the barycentric dual area, i.e. rhmp's ``star0``).
"""
from __future__ import annotations

from math import factorial

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.special import lpmv

MERGE_TOL = 1e-9        # seam vertices of the cube-sphere closer than this (unit sphere) are merged
DEGENERATE_AREA = 1e-12  # triangles with 2*area < DEGENERATE_AREA * longest_edge^2 count as degenerate


# ======================================================================================================================
# basic mesh utilities
# ======================================================================================================================
def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniformly random proper rotation ``(3, 3)`` (QR of a Gaussian matrix, sign-fixed, det +1)."""
    Q, Rm = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q * np.sign(np.diag(Rm))[None, :]
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def face_cross(pts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """``(m, 3)`` un-normalised face normals ``(p1 - p0) x (p2 - p0)`` (length = 2 * area)."""
    P = pts[faces]
    return np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])


def triangle_areas(pts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """``(m,)`` triangle areas."""
    return 0.5 * np.linalg.norm(face_cross(pts, faces), axis=1)


def vertex_normals(pts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """``(n, 3)`` unit vertex normals (area-weighted face normals)."""
    fn = face_cross(pts, faces)
    vn = np.zeros_like(pts)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-300)


def unique_edges(faces: np.ndarray, n: int) -> np.ndarray:
    """Canonical edges ``(n1, 2)`` (``src < dst``, lexicographic order) of a triangle mesh."""
    a = faces.ravel()
    b = np.roll(faces, -1, axis=1).ravel()
    key = np.unique(np.minimum(a, b).astype(np.int64) * n + np.maximum(a, b))
    return np.stack([key // n, key % n], axis=1)


def adjacency(faces: np.ndarray, n: int) -> sp.csr_matrix:
    """Symmetric 0/1 vertex adjacency ``(n, n)``."""
    e = unique_edges(faces, n)
    A = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n))
    return A.tocsr()


def merge_close_vertices(pts: np.ndarray, faces: np.ndarray, tol: float = MERGE_TOL):
    """Merge vertices closer than ``tol`` (seams of patch-wise parametrisations) and drop unused vertices.

    Returns:
        ``(pts', faces')`` with faces re-indexed (vertex order = order of first occurrence of each cluster).
    """
    n = len(pts)
    pairs = cKDTree(pts).query_pairs(tol, output_type="ndarray")
    G = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else \
        sp.coo_matrix((n, n))
    _, lab = connected_components(G, directed=False)
    # representative = first vertex of each cluster, renumbered in order of first occurrence
    first = np.full(lab.max() + 1, n, dtype=np.int64)
    np.minimum.at(first, lab, np.arange(n))
    order = np.argsort(first)
    new_id = np.empty_like(order)
    new_id[order] = np.arange(len(order))
    vid = new_id[lab]
    return pts[first[order]], vid[faces]


def orient_faces_outward_convex(pts: np.ndarray, faces: np.ndarray, center: np.ndarray | None = None) -> np.ndarray:
    """Flip every triangle whose normal points towards ``center`` (valid for star-shaped surfaces only)."""
    c = pts.mean(0) if center is None else center
    fn = face_cross(pts, faces)
    flip = (fn * (pts[faces].mean(1) - c)).sum(1) < 0
    out = faces.copy()
    out[flip] = out[flip][:, [0, 2, 1]]
    return out


def signed_volume(pts: np.ndarray, faces: np.ndarray) -> float:
    """Signed enclosed volume ``sum_f det(p0, p1, p2) / 6`` (> 0 for outward-oriented closed surfaces)."""
    P = pts[faces]
    return float(np.einsum("ij,ij->i", P[:, 0], np.cross(P[:, 1], P[:, 2])).sum() / 6.0)


def triangle_quality(pts: np.ndarray, faces: np.ndarray) -> dict:
    """Angles (degrees) and aspect ratios ``R / (2 r)`` (1 = equilateral) of all triangles."""
    P = pts[faces]
    la = np.linalg.norm(P[:, 1] - P[:, 2], axis=1)   # opposite vertex 0
    lb = np.linalg.norm(P[:, 2] - P[:, 0], axis=1)   # opposite vertex 1
    lc = np.linalg.norm(P[:, 0] - P[:, 1], axis=1)   # opposite vertex 2
    area = triangle_areas(pts, faces)
    s = 0.5 * (la + lb + lc)
    aspect = la * lb * lc * s / np.maximum(8.0 * area ** 2, 1e-300)
    ca = np.clip((lb ** 2 + lc ** 2 - la ** 2) / np.maximum(2 * lb * lc, 1e-300), -1, 1)
    cb = np.clip((la ** 2 + lc ** 2 - lb ** 2) / np.maximum(2 * la * lc, 1e-300), -1, 1)
    cc = np.clip((la ** 2 + lb ** 2 - lc ** 2) / np.maximum(2 * la * lb, 1e-300), -1, 1)
    ang = np.degrees(np.arccos(np.stack([ca, cb, cc], 1)))
    return dict(angles=ang, aspect=aspect, area=area, lmax=np.maximum(np.maximum(la, lb), lc))


def mesh_report(pts: np.ndarray, faces: np.ndarray) -> dict:
    """Topology / orientation / quality summary of a triangle mesh (closed-surface checks).

    Keys: ``n0, n1, n2, euler, genus, components, closed`` (every edge has exactly two faces), ``manifold``
    (no edge with > 2 faces), ``oriented`` (every half-edge appears once and its reverse exists), ``n_unused``,
    ``n_degenerate``, ``n_duplicate_faces``, ``volume`` (signed), ``area``, ``edge_mean / edge_median / edge_min /
    edge_max``, ``min_angle, max_angle, max_aspect, p99_aspect``, ``valence_min / valence_max``.
    """
    n = len(pts)
    F = np.asarray(faces, dtype=np.int64)
    src = F.ravel()
    dst = np.roll(F, -1, axis=1).ravel()
    kdir = src * n + dst
    _, cdir = np.unique(kdir, return_counts=True)
    kund = np.minimum(src, dst) * n + np.maximum(src, dst)
    und, cund = np.unique(kund, return_counts=True)
    has_rev = np.isin(dst * n + src, kdir)
    used = np.unique(F)
    comps = connected_components(adjacency(F, n)[used][:, used], directed=False)[0] if len(used) else 0
    chi = len(used) - len(und) + len(F)
    q = triangle_quality(pts, F)
    e = unique_edges(F, n)
    el = np.linalg.norm(pts[e[:, 1]] - pts[e[:, 0]], axis=1)
    val = np.bincount(e.ravel(), minlength=n)[used]
    dup_faces = len(F) - len(np.unique(np.sort(F, axis=1), axis=0))
    return dict(
        n0=int(len(used)), n1=int(len(und)), n2=int(len(F)), euler=int(chi), genus=(2 - chi) / 2.0,
        components=int(comps), closed=bool((cund == 2).all()), manifold=bool((cund <= 2).all()),
        oriented=bool((cdir == 1).all() and has_rev.all()), n_unused=int(n - len(used)),
        n_degenerate=int((2 * q["area"] < DEGENERATE_AREA * q["lmax"] ** 2).sum()),
        n_duplicate_faces=int(dup_faces), volume=signed_volume(pts, F), area=float(q["area"].sum()),
        edge_mean=float(el.mean()), edge_median=float(np.median(el)), edge_min=float(el.min()),
        edge_max=float(el.max()), min_angle=float(q["angles"].min()), max_angle=float(q["angles"].max()),
        max_aspect=float(q["aspect"].max()), p99_aspect=float(np.percentile(q["aspect"], 99)),
        valence_min=int(val.min()), valence_max=int(val.max()),
    )


def assert_valid_closed(rep: dict, genus: int, where: str = "") -> None:
    """Raise ``RuntimeError`` unless ``rep`` (from :func:`mesh_report`) is a valid closed oriented genus-``genus``
    surface with positive (outward) volume and no degenerate/duplicate faces."""
    ok = (rep["closed"] and rep["manifold"] and rep["oriented"] and rep["components"] == 1 and rep["n_unused"] == 0
          and rep["n_degenerate"] == 0 and rep["n_duplicate_faces"] == 0 and rep["genus"] == genus
          and rep["volume"] > 0)
    if not ok:
        raise RuntimeError(f"invalid surface mesh {where}: {rep}")


def split_quads(q: np.ndarray, rng: np.random.Generator | None) -> np.ndarray:
    """Split quads ``(m, 4)`` (cyclic vertex order) into triangles along a random diagonal (``rng=None``: 0-2).

    Returns ``(2m, 3)`` triangles with the orientation of the quads.
    """
    diag = rng.random(len(q)) < 0.5 if rng is not None else np.zeros(len(q), bool)
    t1 = np.where(diag[:, None], q[:, [0, 1, 2]], q[:, [0, 1, 3]])
    t2 = np.where(diag[:, None], q[:, [0, 2, 3]], q[:, [1, 2, 3]])
    return np.concatenate([t1, t2], 0)


# ======================================================================================================================
# DEC / FEM operators
# ======================================================================================================================
def d0_matrix(edges: np.ndarray, n: int) -> sp.csr_matrix:
    """Coboundary ``d0`` ``(n1, n)``: ``(d0 u)_e = u_dst - u_src``."""
    m = len(edges)
    rows = np.r_[np.arange(m), np.arange(m)]
    return sp.csr_matrix((np.r_[-np.ones(m), np.ones(m)], (rows, np.r_[edges[:, 0], edges[:, 1]])), shape=(m, n))


def cotan_weights(pts: np.ndarray, faces: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Unclamped cotan weights ``w_e = sum_{T ni e} cot(angle of T opposite e) / 2`` for canonical ``edges``."""
    n = len(pts)
    key = edges[:, 0].astype(np.int64) * n + edges[:, 1]
    w = np.zeros(len(edges))
    P = pts[faces]
    for k in range(3):
        i, j = faces[:, (k + 1) % 3], faces[:, (k + 2) % 3]       # edge opposite local vertex k
        a = P[:, (k + 1) % 3] - P[:, k]
        b = P[:, (k + 2) % 3] - P[:, k]
        cot = (a * b).sum(1) / np.maximum(np.linalg.norm(np.cross(a, b), axis=1), 1e-300)
        idx = np.searchsorted(key, np.minimum(i, j).astype(np.int64) * n + np.maximum(i, j))
        np.add.at(w, idx, 0.5 * cot)
    return w


def cotan_operators(pts: np.ndarray, faces: np.ndarray) -> dict:
    """DEC/FEM Laplace-Beltrami operators of a triangle mesh (2-D or 3-D positions).

    Returns:
        dict with ``edges (n1, 2)``, ``w (n1,)`` unclamped cotan weights, ``d0`` csr ``(n1, n)``,
        ``L`` csr ``(n, n)`` = ``d0^T diag(w) d0`` (P1 stiffness, PSD, ``L 1 = 0``), ``mass (n,)`` barycentric lumped
        mass, ``area (m,)`` triangle areas.
    """
    P3 = pts if pts.shape[1] == 3 else np.concatenate([pts, np.zeros((len(pts), 1))], 1)
    n = len(P3)
    edges = unique_edges(faces, n)
    w = cotan_weights(P3, faces, edges)
    d0 = d0_matrix(edges, n)
    L = (d0.T @ sp.diags(w) @ d0).tocsr()
    area = triangle_areas(P3, faces)
    mass = np.zeros(n)
    np.add.at(mass, faces.ravel(), np.repeat(area / 3.0, 3))
    return dict(edges=edges, w=w, d0=d0, L=L, mass=mass, area=area)


# ======================================================================================================================
# genus 0: cube-sphere + radial shape maps
# ======================================================================================================================
def cube_sphere(n: int, rng: np.random.Generator | None = None):
    """Equal-angle cube-sphere on the unit sphere.

    Args:
        n: cells per cube edge (``V = 6 n^2 + 2``, ``F = 12 n^2``).
        rng: random diagonal per quad (``None``: fixed diagonals).
    Returns:
        ``pts (V, 3)`` unit vectors, ``faces (F, 3)`` int64 CCW w.r.t. the outward normal.
    """
    t = np.tan(np.linspace(-np.pi / 4, np.pi / 4, n + 1))
    A, B = np.meshgrid(t, t, indexing="ij")
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    blocks, quads = [], []
    off = 0
    for axis in range(3):
        o1, o2 = (axis + 1) % 3, (axis + 2) % 3
        for sign in (1.0, -1.0):
            P = np.empty((n + 1, n + 1, 3))
            P[..., axis] = sign
            P[..., o1] = A
            P[..., o2] = B
            blocks.append(P.reshape(-1, 3))
            vid = lambda i, j: off + i * (n + 1) + j  # noqa: E731
            quads.append(np.stack([vid(ii, jj), vid(ii + 1, jj), vid(ii + 1, jj + 1), vid(ii, jj + 1)], 1))
            off += (n + 1) ** 2
    pts = np.concatenate(blocks, 0)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    faces = split_quads(np.concatenate(quads, 0), rng)
    pts, faces = merge_close_vertices(pts, faces)
    if len(pts) != 6 * n * n + 2:
        raise RuntimeError(f"cube-sphere seam merge failed: {len(pts)} != {6 * n * n + 2}")
    return pts, orient_faces_outward_convex(pts, faces, np.zeros(3))


def ellipsoid_radius(w: np.ndarray, axes) -> np.ndarray:
    """Radial function of the ellipsoid with semi-axes ``axes`` for unit directions ``w (M, 3)``."""
    a = np.asarray(axes, dtype=np.float64)
    return 1.0 / np.sqrt(((w / a) ** 2).sum(1))


def superquadric_radius(w: np.ndarray, axes, e1: float, e2: float) -> np.ndarray:
    """Radial function of the superellipsoid ``(|x/a|^(2/e2) + |y/b|^(2/e2))^(e2/e1) + |z/c|^(2/e1) = 1``.

    The implicit function is homogeneous of degree ``2/e1``, hence ``r(w) = F(w)^(-e1/2)``.
    """
    a = np.asarray(axes, dtype=np.float64)
    x, y, z = np.abs(w[:, 0] / a[0]), np.abs(w[:, 1] / a[1]), np.abs(w[:, 2] / a[2])
    Fv = (x ** (2.0 / e2) + y ** (2.0 / e2)) ** (e2 / e1) + z ** (2.0 / e1)
    return Fv ** (-e1 / 2.0)


def real_sph_harm(l: int, m: int, w: np.ndarray) -> np.ndarray:
    """Real orthonormal spherical harmonic ``Y_lm`` at unit directions ``w (M, 3)`` (via ``scipy.special.lpmv``)."""
    theta = np.arccos(np.clip(w[:, 2], -1.0, 1.0))
    phi = np.arctan2(w[:, 1], w[:, 0])
    am = abs(m)
    N = np.sqrt((2 * l + 1) / (4 * np.pi) * factorial(l - am) / factorial(l + am))
    P = lpmv(am, l, np.cos(theta))
    if m == 0:
        return N * P
    if m > 0:
        return np.sqrt(2.0) * N * P * np.cos(m * phi)
    return np.sqrt(2.0) * N * P * np.sin(am * phi)


def fibonacci_sphere(M: int) -> np.ndarray:
    """``(M, 3)`` quasi-uniform unit vectors (used to normalise random radial perturbations)."""
    k = np.arange(M) + 0.5
    z = 1.0 - 2.0 * k / M
    phi = np.pi * (1.0 + 5 ** 0.5) * k
    s = np.sqrt(1.0 - z ** 2)
    return np.stack([s * np.cos(phi), s * np.sin(phi), z], 1)


def draw_sh_perturbation(rng: np.random.Generator, lmin: int = 2, lmax: int = 4, amp: float = 0.25,
                         decay: float = 1.0) -> dict:
    """Random real-SH coefficients ``c_lm ~ N(0, l^-2 decay)`` for ``lmin <= l <= lmax``, rescaled so that
    ``max_w |sum c_lm Y_lm(w)| = amp`` (maximum over 4000 quasi-uniform directions)."""
    coeffs = {(l, m): rng.normal() * l ** (-decay) for l in range(lmin, lmax + 1) for m in range(-l, l + 1)}
    probe = fibonacci_sphere(4000)
    v = sum(c * real_sph_harm(l, m, probe) for (l, m), c in coeffs.items())
    s = amp / max(np.abs(v).max(), 1e-12)
    return {k: c * s for k, c in coeffs.items()}


def sh_radius(w: np.ndarray, coeffs: dict) -> np.ndarray:
    """Radial function ``1 + sum_lm c_lm Y_lm(w)`` of a perturbed unit sphere."""
    return 1.0 + sum(c * real_sph_harm(l, m, w) for (l, m), c in coeffs.items())


def radial_project(x: np.ndarray, radius_fn) -> np.ndarray:
    """Project points radially (from the origin) onto the star-shaped surface ``r(w) w``."""
    w = x / np.linalg.norm(x, axis=1, keepdims=True)
    return w * radius_fn(w)[:, None]


def tangential_smooth(pts: np.ndarray, faces: np.ndarray, radius_fn, iters: int = 30, lam: float = 0.5):
    """Tangential umbrella smoothing with radial re-projection onto a star-shaped surface.

    Each iteration moves every vertex by ``lam`` times the tangential part of ``mean(neighbours) - x`` and
    projects it back onto ``r(w) w``; this equalises edge lengths without changing the surface.
    """
    A = adjacency(faces, len(pts))
    deg = np.asarray(A.sum(1)).ravel()
    x = pts.copy()
    for _ in range(iters):
        d = (A @ x) / deg[:, None] - x
        nv = vertex_normals(x, faces)
        d -= (d * nv).sum(1, keepdims=True) * nv
        x = radial_project(x + lam * d, radius_fn)
    return x


def genus0_mesh(radius_fn, area_target: float, h: float, rng: np.random.Generator, smooth_iters: int = 30,
                calibrate: bool = True):
    """Genus-0 star-shaped surface with total area ``area_target`` and median edge length close to ``h``.

    The unit-scale shape ``r(w) w`` is meshed with a randomly rotated cube-sphere of ``n`` cells per cube edge
    (``n`` from ``6 n^2 h^2 = area``), smoothed tangentially, scaled to the target area; if the median edge length
    deviates by more than 4 % from ``h`` the resolution is corrected once.

    Returns:
        ``pts (V, 3)``, ``faces (F, 3)`` (CCW outward), info dict (``n``, ``scale``).
    """
    n = max(3, int(round(np.sqrt(area_target / (6.0 * h * h)))))
    Q = random_rotation(rng)
    diag_seed = int(rng.integers(2 ** 31))
    for attempt in range(2):
        s_pts, faces = cube_sphere(n, np.random.default_rng(diag_seed))
        x = radial_project(s_pts @ Q.T, radius_fn)
        if smooth_iters:
            x = tangential_smooth(x, faces, radius_fn, iters=smooth_iters)
        scale = np.sqrt(area_target / triangle_areas(x, faces).sum())
        x = x * scale
        e = unique_edges(faces, len(x))
        med = float(np.median(np.linalg.norm(x[e[:, 1]] - x[e[:, 0]], axis=1)))
        if not calibrate or abs(med / h - 1.0) < 0.04 or attempt == 1:
            break
        n = max(3, int(round(n * med / h)))
    # star-shaped check: every face normal has a positive radial component (no fold-overs)
    if ((face_cross(x, faces) * x[faces].mean(1)).sum(1) <= 0).any():
        raise RuntimeError("genus-0 mesh folded during smoothing")
    return x, faces, dict(n=n, scale=float(scale))


# ======================================================================================================================
# genus 1: torus
# ======================================================================================================================
def torus_point(u, v, R, r, center=(0.0, 0.0, 0.0), rho=None):
    """Point of the torus ``c + ((R + rho cos v) cos u, (R + rho cos v) sin u, rho sin v)`` (``rho = r`` default)."""
    rho = r if rho is None else rho
    c = np.asarray(center, dtype=np.float64)
    ring = R + rho * np.cos(v)
    return np.stack([c[0] + ring * np.cos(u), c[1] + ring * np.sin(u), c[2] + rho * np.sin(v)], -1)


def torus_normal(u, v):
    """Outward unit normal of an (unperturbed) torus at parameters ``(u, v)``."""
    return np.stack([np.cos(v) * np.cos(u), np.cos(v) * np.sin(u), np.sin(v)], -1)


def draw_tube_perturbation(rng: np.random.Generator, amp: float, n_terms: int = 4, max_freq: int = 2) -> dict:
    """Random smooth periodic relative perturbation ``delta(u, v) = sum_j a_j cos(m_j u + n_j v + phi_j)`` with
    ``max |delta| <= amp`` (integer frequencies ``|m|, |n| <= max_freq``, not both 0)."""
    freqs = []
    while len(freqs) < n_terms:
        m, k = rng.integers(-max_freq, max_freq + 1, 2)
        if m or k:
            freqs.append((int(m), int(k)))
    a = rng.normal(size=n_terms)
    a *= amp / max(np.abs(a).sum(), 1e-12)          # sum |a_j| = amp  =>  |delta| <= amp
    return dict(freqs=freqs, a=a, phi=rng.uniform(0, 2 * np.pi, n_terms))


def tube_perturbation(p: dict | None, U, V):
    if p is None:
        return np.zeros_like(U)
    return sum(a * np.cos(m * U + k * V + ph) for (m, k), a, ph in zip(p["freqs"], p["a"], p["phi"]))


def torus_mesh(R: float, r: float, nu: int, nv: int, rng: np.random.Generator | None, pert: dict | None = None):
    """Periodic UV-grid torus (``nu`` cells around the ring, ``nv`` around the tube), random quad diagonals.

    Tube radius ``rho(u, v) = r (1 + delta(u, v))`` with an optional perturbation from
    :func:`draw_tube_perturbation`.  The ``(u, v)`` frame is positively oriented w.r.t. the outward normal, so the
    faces ``(v00, v10, v11), (v00, v11, v01)`` are CCW outward.

    Returns:
        ``pts (nu*nv, 3)``, ``faces (2 nu nv, 3)`` int64.
    """
    U, V = np.meshgrid(2 * np.pi * np.arange(nu) / nu, 2 * np.pi * np.arange(nv) / nv, indexing="ij")
    rho = r * (1.0 + tube_perturbation(pert, U, V))
    pts = torus_point(U, V, R, r, rho=rho).reshape(-1, 3)
    a, b = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
    a, b = a.ravel(), b.ravel()
    vid = lambda i, j: (i % nu) * nv + (j % nv)  # noqa: E731
    quads = np.stack([vid(a, b), vid(a + 1, b), vid(a + 1, b + 1), vid(a, b + 1)], 1)
    return pts, split_quads(quads, rng)


# ======================================================================================================================
# genus 2: two tori joined by a neck
# ======================================================================================================================
def _hermite(t):
    t2, t3 = t * t, t * t * t
    return 2 * t3 - 3 * t2 + 1, t3 - 2 * t2 + t, -2 * t3 + 3 * t2, t3 - t2


def double_torus_mesh(R: float, r: float, gap: float, h: float, k: int, rng: np.random.Generator,
                      tau_factor: float = 1.2, smooth_iters: int = 12, smooth_rings: int = 2, lam: float = 0.5):
    """Genus-2 surface: two mirror-image tori (ring radius ``R``, tube radius ``r``, axes along z) centred at
    ``(-/+ D, 0, 0)`` with ``D = R + r + gap/2``, each with a ``2k x 2k``-cell hole around its point facing the other
    torus, joined by a neck of rings.

    Neck: loop vertex ``p_A`` (grid index ``(a, b)``) of torus A is connected to its mirror image ``p_B`` on torus
    B by the cubic Hermite curve with end tangents ``tau w_A`` and ``-tau w_B``, where ``w`` is the unit tangent
    direction (in the torus tangent plane) from the loop vertex towards its hole centre and
    ``tau = tau_factor |p_A - c_A|`` (so the neck leaves both holes tangentially and has a waist of about
    ``1 - tau_factor / 4`` of the hole size).  Rings are placed at equal arc length (spacing ~h).  Finally the neck
    vertices and the torus vertices within ``smooth_rings`` grid rings of each hole are relaxed by
    ``smooth_iters`` umbrella iterations.

    Returns:
        ``pts (V, 3)``, ``faces (F, 3)`` int64 (CCW outward), info dict.
    """
    nu = max(4 * k + 8, int(round(2 * np.pi * R / h)))
    nv = max(4 * k + 6, int(round(2 * np.pi * r / h)))
    D = R + r + 0.5 * gap
    U, V = np.meshgrid(2 * np.pi * np.arange(nu) / nu, 2 * np.pi * np.arange(nv) / nv, indexing="ij")
    PA = torus_point(U, V, R, r, center=(-D, 0.0, 0.0))            # (nu, nv, 3)
    mirror = np.array([-1.0, 1.0, 1.0])
    PB = PA * mirror
    # removed vertices (strict interior of the hole block) and removed cells
    removed = np.zeros((nu, nv), bool)
    removed[np.ix_(np.arange(-(k - 1), k) % nu, np.arange(-(k - 1), k) % nv)] = True
    cell_removed = np.zeros((nu, nv), bool)
    cell_removed[np.ix_(np.arange(-k, k) % nu, np.arange(-k, k) % nv)] = True
    keep = ~removed
    nA = int(keep.sum())
    idx = -np.ones((nu, nv), dtype=np.int64)
    idx[keep] = np.arange(nA)
    pts = [PA[keep], PB[keep]]                                     # torus A: [0, nA), torus B: [nA, 2 nA)
    a, b = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
    live = ~cell_removed[a, b]
    a, b = a[live], b[live]
    q = np.stack([idx[a % nu, b % nv], idx[(a + 1) % nu, b % nv], idx[(a + 1) % nu, (b + 1) % nv],
                  idx[a % nu, (b + 1) % nv]], 1)
    if (q < 0).any():
        raise RuntimeError("double torus: a live cell references a removed vertex")
    tri_A = split_quads(q, rng)
    tri_B = split_quads(q + nA, rng)[:, [0, 2, 1]]                  # mirror image: reverse orientation
    # hole boundary loop (grid indices), cyclic
    loop = [(i, -k) for i in range(-k, k)] + [(k, j) for j in range(-k, k)] + \
           [(i, k) for i in range(k, -k, -1)] + [(-k, j) for j in range(k, -k, -1)]
    la = np.array([p[0] for p in loop]) % nu
    lb = np.array([p[1] for p in loop]) % nv
    ring_A = idx[la, lb]
    ring_B = ring_A + nA
    Lp = len(loop)
    # geometry of the neck curves
    pA = PA[la, lb]
    nA_vec = torus_normal(U[la, lb], V[la, lb])
    cA = torus_point(0.0, 0.0, R, r, center=(-D, 0.0, 0.0))
    dvec = cA - pA
    w = dvec - (dvec * nA_vec).sum(1, keepdims=True) * nA_vec
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    tau = tau_factor * np.linalg.norm(dvec, axis=1, keepdims=True)
    pB, wB = pA * mirror, w * mirror
    ts = np.linspace(0.0, 1.0, 401)
    H = _hermite(ts[:, None, None])
    curves = H[0] * pA + H[1] * tau * w + H[2] * pB + H[3] * tau * (-wB)      # (401, Lp, 3)
    seg = np.linalg.norm(np.diff(curves, axis=0), axis=2).mean(1)             # mean arc length per t-step
    arc = np.r_[0.0, np.cumsum(seg)]
    m = max(1, int(round(arc[-1] / h)) - 1)                                    # interior rings
    t_ring = np.interp(np.linspace(0, arc[-1], m + 2)[1:-1], arc, ts)
    Hr = _hermite(t_ring[:, None, None])
    neck = Hr[0] * pA + Hr[1] * tau * w + Hr[2] * pB + Hr[3] * tau * (-wB)    # (m, Lp, 3)
    base = 2 * nA
    rings = [ring_A] + [base + j * Lp + np.arange(Lp) for j in range(m)] + [ring_B]
    pts.append(neck.reshape(-1, 3))
    pts = np.concatenate(pts, 0)
    # orientation of the neck strip: opposite to torus A's traversal of its hole loop
    he = set(zip(tri_A[:, 0].tolist(), tri_A[:, 1].tolist())) | set(zip(tri_A[:, 1].tolist(), tri_A[:, 2].tolist())) \
        | set(zip(tri_A[:, 2].tolist(), tri_A[:, 0].tolist()))
    forward = (int(ring_A[0]), int(ring_A[1])) in he
    if not forward and (int(ring_A[1]), int(ring_A[0])) not in he:
        raise RuntimeError("double torus: hole loop is not a boundary of torus A")
    quads = []
    i0 = np.arange(Lp)
    i1 = (i0 + 1) % Lp
    for j in range(m + 1):
        Rj, Rk = rings[j], rings[j + 1]
        qd = np.stack([Rj[i0], Rk[i0], Rk[i1], Rj[i1]], 1)   # traverses ring j backwards, ring j+1 forwards
        quads.append(qd if forward else qd[:, [0, 3, 2, 1]])
    tri_N = split_quads(np.concatenate(quads, 0), rng)
    faces = np.concatenate([tri_A, tri_B, tri_N], 0)
    # relax the neck and the hole neighbourhoods (umbrella smoothing, other vertices fixed)
    da = np.minimum(np.arange(nu), nu - np.arange(nu))
    db = np.minimum(np.arange(nv), nv - np.arange(nv))
    near = (np.maximum(da[:, None], db[None, :]) <= k + smooth_rings) & keep
    movable = np.zeros(len(pts), bool)
    movable[idx[near]] = True
    movable[idx[near] + nA] = True
    movable[base:] = True
    Adj = adjacency(faces, len(pts))
    deg = np.asarray(Adj.sum(1)).ravel()
    x = pts.copy()
    for _ in range(smooth_iters):
        x_new = (1 - lam) * x + lam * (Adj @ x) / deg[:, None]
        x[movable] = x_new[movable]
    info = dict(nu=nu, nv=nv, k=k, neck_rings=m, loop_len=Lp, D=float(D))
    return x, faces, info
