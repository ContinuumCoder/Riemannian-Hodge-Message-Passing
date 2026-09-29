"""E(n)-invariant geometry of cochain complexes: reference Hodge stars and per-cell descriptors.

Everything here is computed once per complex under ``torch.no_grad()`` in float64 (vectorised with
``index_add``/``scatter_reduce``; no loops over cells) and cached on the complex as float32.
Only intrinsic quantities (lengths, areas, volumes, angles) are used, so every output is invariant
under rotations, reflections and translations of ``pos``.  Scale quantities enter the descriptors
as ``log(x / median_x)`` with the median taken over *this* complex (mesh-transferable, DESIGN §3.1).

Reference stars (DESIGN §3.1):
  * ``cotan``: ``star0`` barycentric dual area, ``star1 = sum_f w_fe`` with ``w_fe = cot(opposite
    angle)/2`` for triangular faces (``|c_f - m_e| / |e|``, the barycentric half dual edge ratio, for
    polygonal faces; this equals the circumcentric value on rectangles) clamped below at
    ``KAPPA * median``; ``star2 = 1/area``.  Surfaces only.
  * ``barycentric``: ``star1 = |dual edge| / |edge|`` (dual edge = polyline edge-midpoint -> face
    centroids); tets: ``star0`` dual volume, ``star1 = |dual face|/|edge|``, ``star2 = |dual
    edge|/|face|``, ``star3 = 1/vol``.
  * ``unit``: all ones.

Descriptor columns per degree are listed in :data:`GEO_FEATURE_NAMES` (identical for all cell
types so that a model can move between complexes; columns that do not apply are constant).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = [
    "GEO_FEATURE_NAMES",
    "STAR_TYPES",
    "KAPPA",
    "COT_CLAMP",
    "face_measures",
    "surface_geometry",
    "volume_geometry",
    "barycentric_gradients",
    "whitney1_values",
    "whitney2_values_tet",
    "galerkin_blocks",
    "whitney_triangles",
    "whitney_tets",
]

KAPPA = 1e-2  # cotan star1 is clamped below at KAPPA * median(star1) (DESIGN §3.1)
COT_CLAMP = 10.0  # the raw (dimensionless) cotan-weight descriptor is clamped to [-COT_CLAMP, COT_CLAMP]
_LOG_FLOOR = 1e-12  # log-features use max(x, _LOG_FLOOR * median) -> descriptor values >= log(1e-12)
_SIN_FLOOR = 1e-12  # |sin| of an angle is floored at _SIN_FLOOR * |u||w| before forming cotangents
_TINY = 1e-300  # absolute guard against division by exact zero (degenerate input with validate=False)

STAR_TYPES = ("cotan", "barycentric", "unit")

GEO_FEATURE_NAMES: dict[int, tuple[str, ...]] = {
    # vertices: dual area (surfaces) / dual volume (tets); mean incident edge length; log #edges;
    # boundary flag; angle defect 2pi - sum(angles) (pi - sum at boundary vertices; 0 for tets)
    0: ("log_dual_measure", "log_mean_edge_length", "log_valence", "boundary", "angle_defect"),
    # edges: length; reference star1; raw DEC cotan weight (dimensionless, clamped); boundary flag;
    # log #cofaces; cos of the dihedral angle between the two cofaces (1 = flat; 1 if != 2 cofaces or tets)
    1: ("log_length", "log_star", "cotan_weight", "boundary", "log_cofaces", "cos_dihedral"),
    # faces: area; reference star2; smallest interior angle (rad); log aspect (triangles: R/(2r),
    # polygons: perimeter^2 / (4 L tan(pi/L) area); 0 = equilateral / regular); boundary flag
    2: ("log_area", "log_star", "min_angle", "log_aspect", "boundary"),
    # tets: volume; log aspect R/(3r) (0 = regular tet; the inradius/circumradius quality proxy); boundary flag
    3: ("log_volume", "log_aspect", "boundary"),
}


# ----------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------
def _pad3(P: Tensor) -> Tensor:
    """(n, 2|3) -> (n, 3) (2-D positions embedded at z = 0)."""
    if P.shape[1] == 3:
        return P
    if P.shape[1] == 2:
        return torch.cat([P, P.new_zeros(P.shape[0], 1)], dim=1)
    raise ValueError(f"positions must be 2-D or 3-D, got shape {tuple(P.shape)}")


def _cross(a: Tensor, b: Tensor) -> Tensor:
    return torch.linalg.cross(a, b, dim=-1)


def _median_pos(x: Tensor) -> Tensor:
    """Median of the strictly positive entries (1 if there are none)."""
    pos = x[x > 0]
    return pos.median() if pos.numel() else x.new_tensor(1.0)


def _log_rel(x: Tensor) -> Tensor:
    """``log(x / median(x))`` with a relative floor; E(n)- and scale-invariant."""
    med = _median_pos(x)
    return torch.log(x.clamp_min(med * _LOG_FLOOR) / med)


def _scatter_add(n: int, index: Tensor, src: Tensor) -> Tensor:
    return src.new_zeros((n,) + tuple(src.shape[1:])).index_add_(0, index, src)


def _scatter_any(n: int, index: Tensor, flag: Tensor) -> Tensor:
    return _scatter_add(n, index, flag.to(torch.float64)) > 0


def _check_star(star: str, allowed: tuple[str, ...]) -> None:
    if star not in allowed:
        raise ValueError(f"unknown/unsupported star type {star!r}; expected one of {allowed}")


# ----------------------------------------------------------------------------------------------
# faces
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def face_measures(P: Tensor, faces: Tensor, face_len: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Area, unit normal and area centroid of (possibly polygonal) faces via fan triangulation.

    Args:
        P: ``(n0, 3)`` float64 positions.
        faces: ``(n2, M)`` int64 vertex ids, ``-1`` padded (padding trailing).
        face_len: ``(n2,)`` number of valid vertices (>= 3).
    Returns:
        area ``(n2,)``, normal ``(n2, 3)`` (zero for degenerate faces; orientation follows the vertex
        order), centroid ``(n2, 3)`` (vertex mean for triangles; signed-fan area centroid otherwise).
    """
    n2, M = faces.shape
    F = faces.clamp_min(0)
    p0 = P[F[:, 0]]
    fan = []
    vec_area = P.new_zeros(n2, 3)
    for j in range(1, M - 1):
        valid = ((j + 1) < face_len).to(P.dtype)[:, None]
        pj, pj1 = P[F[:, j]], P[F[:, j + 1]]
        cr = 0.5 * _cross(pj - p0, pj1 - p0) * valid
        vec_area += cr
        fan.append((cr, (p0 + pj + pj1) / 3.0))
    area = vec_area.norm(dim=-1)
    normal = vec_area / area.clamp_min(_TINY)[:, None]
    valid_v = (faces >= 0).to(P.dtype)
    vmean = (P[F] * valid_v[..., None]).sum(1) / face_len.to(P.dtype)[:, None]
    if M == 3:
        return area, normal, vmean
    num = P.new_zeros(n2, 3)
    den = P.new_zeros(n2)
    for cr, c in fan:
        a = (cr * normal).sum(-1)
        num += a[:, None] * c
        den += a
    ok = (face_len > 3) & (den.abs() > 0)
    centroid = torch.where(ok[:, None], num / torch.where(den.abs() > 0, den, 1.0)[:, None], vmean)
    return area, normal, centroid


@torch.no_grad()
def _face_block(P: Tensor, faces: Tensor, face_len: Tensor, corner: dict[str, Tensor], ell: Tensor) -> dict:
    """Face measures + corner angles + per-face descriptors shared by surfaces and tet faces."""
    n2 = faces.shape[0]
    cf, cv, cn, cp, ce = corner["face"], corner["v"], corner["next"], corner["prev"], corner["edge"]
    area, normal, centroid = face_measures(P, faces, face_len)
    tri_c = (face_len == 3)[cf]
    Pv = P[cv]
    u = P[cn] - Pv
    w = P[cp] - Pv
    cr = _cross(u, w)
    dot = (u * w).sum(-1)
    sin_u = cr.norm(dim=-1)
    sin = torch.where(tri_c, sin_u, (cr * normal[cf]).sum(-1))
    theta = torch.atan2(sin, dot)
    theta = torch.where(theta < 0, theta + 2.0 * math.pi, theta)
    min_angle = P.new_full((n2,), float("inf")).scatter_reduce(0, cf, theta, reduce="amin", include_self=True)
    # aspect
    le = ell[ce]
    perim = _scatter_add(n2, cf, le)
    log_prod = _scatter_add(n2, cf, torch.log(le.clamp_min(_TINY)))
    logA = torch.log(area.clamp_min(_TINY))
    L = face_len.to(P.dtype)
    log_asp_tri = log_prod + torch.log(0.5 * perim.clamp_min(_TINY)) - math.log(8.0) - 2.0 * logA
    reg = 4.0 * L * torch.tan(math.pi / L)
    log_asp_poly = 2.0 * torch.log(perim.clamp_min(_TINY)) - torch.log(reg) - logA
    log_aspect = torch.where(face_len == 3, log_asp_tri, log_asp_poly)
    uw = u.norm(dim=-1) * w.norm(dim=-1)
    return dict(
        area=area, normal=normal, centroid=centroid, theta=theta, dot=dot, sin_u=sin_u, uw=uw,
        min_angle=min_angle, log_aspect=log_aspect, tri_c=tri_c, u=u, w=w,
    )


# ----------------------------------------------------------------------------------------------
# surfaces (top degree 2): triangles and polygons
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def surface_geometry(
    pos: Tensor,
    faces: Tensor,
    face_len: Tensor,
    edges: Tensor,
    corner: dict[str, Tensor],
    star: str = "cotan",
    kappa: float = KAPPA,
) -> dict:
    """Stars, descriptors and boundary flags of a 2-complex (triangles and/or polygons).

    Args:
        pos: ``(n0, D)`` positions, D in {2, 3} (any float dtype; computed in float64).
        faces: ``(n2, M)`` int64 vertex ids (``-1`` padded), oriented as given.
        face_len: ``(n2,)`` int64 number of vertices per face.
        edges: ``(n1, 2)`` int64 canonical edges (src < dst).
        corner: corner/half-edge table, each ``(H,)`` with ``H = sum(face_len)``: ``face``, ``v`` (corner
            vertex), ``next``, ``prev`` (neighbouring vertices in the face), ``edge`` (edge id of the
            half-edge ``v -> next``), ``sign`` (+1 if ``v < next``), ``prev_idx`` (corner index of ``prev``
            in the same face; for triangles this is the corner opposite the half-edge).
        star: ``'cotan' | 'barycentric' | 'unit'``.
        kappa: cotan clamp factor.
    Returns:
        dict with ``star`` (3 tensors ``(n_k,)`` float32 > 0), ``geo`` (3 tensors ``(n_k, G_k)`` float32),
        ``boundary`` (3 bool tensors) and ``stats`` (python numbers).
    """
    _check_star(star, STAR_TYPES)
    P = _pad3(pos.to(torch.float64))
    n0, n1, n2 = P.shape[0], edges.shape[0], faces.shape[0]
    cf, cv, cn, cp, ce = corner["face"], corner["v"], corner["next"], corner["prev"], corner["edge"]
    cprev, csign = corner["prev_idx"], corner["sign"]

    e0, e1 = edges[:, 0], edges[:, 1]
    ev = P[e1] - P[e0]
    ell = ev.norm(dim=-1)
    emid = 0.5 * (P[e0] + P[e1])

    fb = _face_block(P, faces, face_len, corner, ell)
    area, centroid, theta, tri_c = fb["area"], fb["centroid"], fb["theta"], fb["tri_c"]

    # ---- per half-edge DEC weights
    q = centroid[cf] - emid[ce]
    bary_half = q.norm(dim=-1) / ell[ce].clamp_min(_TINY)
    sin_opp = fb["sin_u"][cprev].clamp_min(_SIN_FLOOR * fb["uw"][cprev] + _TINY)
    cot_half = torch.where(tri_c, 0.5 * fb["dot"][cprev] / sin_opp, bary_half)
    n_cof = torch.bincount(ce, minlength=n1)
    raw_cot = _scatter_add(n1, ce, cot_half)
    star1_bary = _scatter_add(n1, ce, bary_half)

    # ---- boundary flags
    bnd1 = n_cof == 1
    bnd0 = _scatter_any(n0, e0, bnd1) | _scatter_any(n0, e1, bnd1)
    bnd2 = _scatter_any(n2, cf, bnd1[ce])

    # ---- vertex quantities
    valence = torch.bincount(e0, minlength=n0) + torch.bincount(e1, minlength=n0)
    isolated = valence == 0
    corner_area = torch.where(
        tri_c, area[cf] / 3.0, 0.25 * _cross(centroid[cf] - P[cv], P[cn] - P[cp]).norm(dim=-1)
    )
    dual_area = _scatter_add(n0, cv, corner_area)
    dual_area = torch.where(isolated, _median_pos(dual_area), dual_area)
    mean_len = (_scatter_add(n0, e0, ell) + _scatter_add(n0, e1, ell)) / valence.clamp_min(1).to(P.dtype)
    mean_len = torch.where(isolated, _median_pos(mean_len), mean_len)
    ang_sum = _scatter_add(n0, cv, theta)
    defect = torch.where(bnd0, math.pi - ang_sum, 2.0 * math.pi - ang_sum)
    defect = torch.where(isolated, torch.zeros_like(defect), defect)

    # ---- stars
    n_clamped = 0
    if star == "cotan":
        med = raw_cot.median()
        if not bool(med > 0):
            raise ValueError(
                "cotan star: the median cotan edge weight is not positive (pathological mesh); "
                "use star='barycentric'"
            )
        floor = kappa * med
        n_clamped = int((raw_cot < floor).sum())
        star1 = raw_cot.clamp_min(floor)
        star0 = dual_area
        star2 = 1.0 / area.clamp_min(_TINY)
    elif star == "barycentric":
        star0, star1, star2 = dual_area, star1_bary, 1.0 / area.clamp_min(_TINY)
    else:
        star0, star1, star2 = P.new_ones(n0), P.new_ones(n1), P.new_ones(n2)

    # ---- dihedral angle between the two cofaces of manifold edges (orientation independent)
    t = ev / ell.clamp_min(_TINY)[:, None]
    te = t[ce]
    q_perp = q - (q * te).sum(-1, keepdim=True) * te
    q_hat = q_perp / q_perp.norm(dim=-1).clamp_min(_TINY)[:, None]
    S = _scatter_add(n1, ce, q_hat)
    cos_dih = (1.0 - 0.5 * (S * S).sum(-1)).clamp(-1.0, 1.0)
    cos_dih = torch.where(n_cof == 2, cos_dih, torch.ones_like(cos_dih))

    f32 = torch.float32
    geo0 = torch.stack(
        [_log_rel(dual_area), _log_rel(mean_len), torch.log(valence.clamp_min(1).to(P.dtype)),
         bnd0.to(P.dtype), defect], dim=1).to(f32)
    geo1 = torch.stack(
        [_log_rel(ell), _log_rel(star1), raw_cot.clamp(-COT_CLAMP, COT_CLAMP), bnd1.to(P.dtype),
         torch.log(n_cof.clamp_min(1).to(P.dtype)), cos_dih], dim=1).to(f32)
    geo2 = torch.stack(
        [_log_rel(area), _log_rel(star2), fb["min_angle"], fb["log_aspect"], bnd2.to(P.dtype)], dim=1).to(f32)

    sign_sum = _scatter_add(n1, ce, csign.to(P.dtype))
    stats = dict(
        n_boundary_edges=int(bnd1.sum()),
        n_nonmanifold_edges=int((n_cof > 2).sum()),
        n_isolated_vertices=int(isolated.sum()),
        n_inconsistent_orientation_edges=int(((n_cof == 2) & (sign_sum != 0)).sum()),
        n_star1_clamped=n_clamped,
        max_log_aspect=float(fb["log_aspect"].max()) if n2 else 0.0,
        min_angle=float(fb["min_angle"].min()) if n2 else 0.0,
    )
    return dict(
        star=[s.to(f32) for s in (star0, star1, star2)],
        geo=[geo0, geo1, geo2],
        boundary=[bnd0, bnd1, bnd2],
        stats=stats,
    )


# ----------------------------------------------------------------------------------------------
# volumes (top degree 3): tetrahedra
# ----------------------------------------------------------------------------------------------
_TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_TET_OPP = ((2, 3), (1, 3), (1, 2), (0, 3), (0, 2), (0, 1))
_TET_FACES = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))  # face i omits vertex i


@torch.no_grad()
def volume_geometry(
    pos: Tensor,
    tets: Tensor,
    faces: Tensor,
    edges: Tensor,
    tet_face: Tensor,
    tet_edge: Tensor,
    face_edge: Tensor,
    face_corner: dict[str, Tensor],
    star: str = "barycentric",
) -> dict:
    """Stars, descriptors and boundary flags of a tetrahedral 3-complex.

    Args:
        pos: ``(n0, 3)`` positions.
        tets: ``(n3, 4)`` int64 vertex ids (input order = orientation).
        faces: ``(n2, 3)`` canonical (sorted) triangles.
        edges: ``(n1, 2)`` canonical edges.
        tet_face: ``(n3, 4)`` face id of the face opposite local vertex i.
        tet_edge: ``(n3, 6)`` edge id of local vertex pairs ``(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)``.
        face_edge: ``(n2, 3)`` edge ids of face edges ``(a,b),(b,c),(a,c)``.
        face_corner: corner table of the canonical faces (see :func:`surface_geometry`).
        star: ``'barycentric' | 'unit'``.
    Returns:
        dict with ``star``/``geo``/``boundary`` lists of length 4 and ``stats``.
    """
    _check_star(star, ("barycentric", "unit"))
    P = pos.to(torch.float64)
    if P.shape[1] != 3:
        raise ValueError(f"tetrahedral complexes need 3-D positions, got {tuple(P.shape)}")
    n0, n1, n2, n3 = P.shape[0], edges.shape[0], faces.shape[0], tets.shape[0]
    V = P[tets]  # (n3, 4, 3)
    a, b, c = V[:, 1] - V[:, 0], V[:, 2] - V[:, 0], V[:, 3] - V[:, 0]
    bxc = _cross(b, c)
    vol = (a * bxc).sum(-1).abs() / 6.0
    ct = V.mean(1)

    e0, e1 = edges[:, 0], edges[:, 1]
    ell = (P[e1] - P[e0]).norm(dim=-1)
    face_len = torch.full((n2,), 3, dtype=torch.long, device=P.device)
    fb = _face_block(P, faces, face_len, face_corner, ell)
    f_area, f_cen = fb["area"], fb["centroid"]

    # ---- stars (barycentric duals)
    valence = torch.bincount(e0, minlength=n0) + torch.bincount(e1, minlength=n0)
    isolated = valence == 0
    dual_vol = _scatter_add(n0, tets.reshape(-1), (vol / 4.0).repeat_interleave(4))
    dual_vol = torch.where(isolated, _median_pos(dual_vol), dual_vol)
    dual_face = P.new_zeros(n1)
    w_cot3 = P.new_zeros(n1)
    for idx, ((i, j), (k, l)) in enumerate(zip(_TET_EDGES, _TET_OPP)):
        pi, pj, pk, pl = V[:, i], V[:, j], V[:, k], V[:, l]
        m = 0.5 * (pi + pj)
        c1 = (pi + pj + pk) / 3.0
        c2 = (pi + pj + pl) / 3.0
        piece = 0.5 * _cross(c1 - m, ct - m).norm(dim=-1) + 0.5 * _cross(ct - m, c2 - m).norm(dim=-1)
        dual_face.index_add_(0, tet_edge[:, idx], piece)
        # 3-D cotan weight of edge (i,j): |e_kl| cot(dihedral angle at kl) / 6
        ekl = pl - pk
        lkl = ekl.norm(dim=-1)
        tkl = ekl / lkl.clamp_min(_TINY)[:, None]
        qi, qj = pi - pk, pj - pk
        qi = qi - (qi * tkl).sum(-1, keepdim=True) * tkl
        qj = qj - (qj * tkl).sum(-1, keepdim=True) * tkl
        s = _cross(qi, qj).norm(dim=-1)
        s = s.clamp_min(_SIN_FLOOR * qi.norm(dim=-1) * qj.norm(dim=-1) + _TINY)
        w_cot3.index_add_(0, tet_edge[:, idx], lkl * (qi * qj).sum(-1) / s / 6.0)
    dual_edge = P.new_zeros(n2)
    for i, fv in enumerate(_TET_FACES):
        cfi = V[:, list(fv)].mean(1)
        dual_edge.index_add_(0, tet_face[:, i], (ct - cfi).norm(dim=-1))
    if star == "barycentric":
        star0 = dual_vol
        star1 = dual_face / ell.clamp_min(_TINY)
        star2 = dual_edge / f_area.clamp_min(_TINY)
        star3 = 1.0 / vol.clamp_min(_TINY)
    else:
        star0, star1, star2, star3 = P.new_ones(n0), P.new_ones(n1), P.new_ones(n2), P.new_ones(n3)

    # ---- boundary flags and counts
    n_tet_face = torch.bincount(tet_face.reshape(-1), minlength=n2)
    bnd2 = n_tet_face == 1
    bnd1 = _scatter_any(n1, face_edge.reshape(-1), bnd2.repeat_interleave(3))
    bnd0 = _scatter_any(n0, faces.reshape(-1), bnd2.repeat_interleave(3))
    bnd3 = bnd2[tet_face].any(dim=1)
    n_face_edge = torch.bincount(face_edge.reshape(-1), minlength=n1)
    mean_len = (_scatter_add(n0, e0, ell) + _scatter_add(n0, e1, ell)) / valence.clamp_min(1).to(P.dtype)
    mean_len = torch.where(isolated, _median_pos(mean_len), mean_len)

    # ---- tet aspect R/(3r)
    R_num = ((a * a).sum(-1, keepdim=True) * bxc + (b * b).sum(-1, keepdim=True) * _cross(c, a)
             + (c * c).sum(-1, keepdim=True) * _cross(a, b)).norm(dim=-1)
    R = R_num / (12.0 * vol).clamp_min(_TINY)
    r = 3.0 * vol / f_area[tet_face].sum(1).clamp_min(_TINY)
    log_aspect3 = torch.log(R.clamp_min(_TINY)) - torch.log((3.0 * r).clamp_min(_TINY))

    f32 = torch.float32
    geo0 = torch.stack(
        [_log_rel(dual_vol), _log_rel(mean_len), torch.log(valence.clamp_min(1).to(P.dtype)),
         bnd0.to(P.dtype), P.new_zeros(n0)], dim=1).to(f32)
    geo1 = torch.stack(
        [_log_rel(ell), _log_rel(star1), (w_cot3 / ell.clamp_min(_TINY)).clamp(-COT_CLAMP, COT_CLAMP),
         bnd1.to(P.dtype), torch.log(n_face_edge.clamp_min(1).to(P.dtype)), P.new_ones(n1)], dim=1).to(f32)
    geo2 = torch.stack(
        [_log_rel(f_area), _log_rel(star2), fb["min_angle"], fb["log_aspect"], bnd2.to(P.dtype)], dim=1).to(f32)
    geo3 = torch.stack([_log_rel(vol), log_aspect3, bnd3.to(P.dtype)], dim=1).to(f32)
    stats = dict(
        n_boundary_faces=int(bnd2.sum()),
        n_nonmanifold_faces=int((n_tet_face > 2).sum()),
        n_isolated_vertices=int(isolated.sum()),
        max_log_aspect=float(log_aspect3.max()) if n3 else 0.0,
        n_negative_cotan_edges=int((w_cot3 < 0).sum()),
    )
    return dict(
        star=[s.to(f32) for s in (star0, star1, star2, star3)],
        geo=[geo0, geo1, geo2, geo3],
        boundary=[bnd0, bnd1, bnd2, bnd3],
        stats=stats,
    )


# ----------------------------------------------------------------------------------------------
# Whitney forms and Galerkin (consistent) metric blocks (DESIGN §9.1)
# ----------------------------------------------------------------------------------------------
# Degree-2-exact quadrature (the Whitney mass-matrix integrands are quadratic in the barycentrics).
_TRI_Q = ((0.5, 0.5, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5))  # edge midpoints, weights |f|/3
_TET_QA, _TET_QB = (5.0 + 3.0 * 5.0 ** 0.5) / 20.0, (5.0 - 5.0 ** 0.5) / 20.0  # 4-point rule, weights |t|/4
_TET_Q = tuple(tuple(_TET_QA if j == i else _TET_QB for j in range(4)) for i in range(4))
_TRI_SLOTS = ((0, 1), (1, 2), (2, 0))  # local edge j of a triangle joins local vertices j and j+1


@torch.no_grad()
def barycentric_gradients(V: Tensor) -> Tensor:
    """Gradients of the barycentric coordinates of simplices (in their affine hull; any ambient dim).

    Args:
        V: ``(n, d+1, D)`` vertex positions (float64 recommended), ``d <= D``.
    Returns:
        ``(n, d+1, D)`` with ``grad(lambda_i) . (V_j - V_0) = delta_ij - delta_i0`` and zero normal part.
    """
    E = V[:, 1:] - V[:, :1]
    G = E @ E.transpose(1, 2)
    rest = torch.linalg.solve(G, E)
    return torch.cat([-rest.sum(1, keepdim=True), rest], dim=1)


def _simplex_measure(V: Tensor) -> Tensor:
    """(n, d+1, D) -> (n,) d-volume via the Gram determinant."""
    E = V[:, 1:] - V[:, :1]
    d = E.shape[1]
    return torch.linalg.det(E @ E.transpose(1, 2)).clamp_min(0).sqrt() / math.factorial(d)


def whitney1_values(lam: Tensor, grads: Tensor, pairs: Tensor, signs: Tensor) -> Tensor:
    """Whitney 1-form basis ``w_ab = lam_a grad(lam_b) - lam_b grad(lam_a)`` at quadrature points.

    Args:
        lam: ``(Q, d+1)`` barycentric coordinates of the quadrature points.
        grads: ``(n, d+1, D)`` barycentric gradients.
        pairs: ``(m, 2)`` local vertex pairs ``a -> b`` of the edges.
        signs: ``(n, m)`` +-1 orientation of ``a -> b`` relative to the complex's canonical edge.
    Returns:
        ``(n, Q, D, m)`` values of the canonically oriented basis functions.
    """
    a, b = pairs[:, 0], pairs[:, 1]
    W = lam[None, :, a, None] * grads[:, None, b] - lam[None, :, b, None] * grads[:, None, a]  # (n, Q, m, D)
    return (W * signs[:, None, :, None]).transpose(2, 3)


def whitney2_values_tet(lam: Tensor, grads: Tensor, face_local: Tensor) -> Tensor:
    """Whitney 2-form basis (vector proxy) of the 4 faces of tets at quadrature points.

    ``w_abc = 2 (lam_a grad_b x grad_c + lam_b grad_c x grad_a + lam_c grad_a x grad_b)`` with ``(a, b, c)``
    the face's vertices in canonical (sorted global id) order, so the basis is consistent with ``d_1``/``d_2``.

    Args:
        lam: ``(Q, 4)``; grads: ``(n, 4, 3)``; face_local: ``(n, 4, 3)`` local vertex ids of face i in canonical order.
    Returns:
        ``(n, Q, 3, 4)``.
    """
    n = grads.shape[0]
    out = []
    for i in range(4):
        c = face_local[:, i]  # (n, 3)
        g = torch.gather(grads, 1, c[:, :, None].expand(n, 3, 3))  # (n, 3, 3) grads of a, b, c
        lc = lam[:, c].permute(1, 0, 2)  # (n, Q, 3)
        cr = torch.stack([_cross(g[:, 1], g[:, 2]), _cross(g[:, 2], g[:, 0]), _cross(g[:, 0], g[:, 1])], 1)  # (n,3,3)
        out.append(2.0 * torch.einsum("nqj,njd->nqd", lc, cr))
    return torch.stack(out, dim=-1)


def galerkin_blocks(W: Tensor, wq: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    """PSD blocks of the Galerkin star ``int w_i . sigma w_j`` for ``sigma = b I + sum_k a_k t_k t_k^T``.

    Args:
        W: ``(n, Q, D, mk)`` basis values; wq: ``(n, Q)`` quadrature weights (measure included);
        t: ``(n, m, D)`` unit direction vectors.
    Returns:
        ``G0 (n, mk, mk) = sum_q wq W_q^T W_q`` and ``Gk (n, m, mk, mk) = sum_q wq (W_q^T t_k)(W_q^T t_k)^T``.
    """
    G0 = torch.einsum("nq,nqdi,nqdj->nij", wq, W, W)
    Pk = torch.einsum("nqdi,nkd->nqki", W, t)
    Gk = torch.einsum("nq,nqki,nqkj->nkij", wq, Pk, Pk)
    return 0.5 * (G0 + G0.transpose(1, 2)), 0.5 * (Gk + Gk.transpose(2, 3))


def _pack_whitney(G0: Tensor, Gk: Tensor, t: Tensor, cells: Tensor, signs: Tensor, dir_edges: Tensor) -> dict:
    f32 = torch.float32
    G0, Gk = G0.to(f32), Gk.to(f32)
    return dict(
        cells=cells, signs=signs.to(f32), dir_edges=dir_edges, t=t.to(f32), G0=G0, Gk=Gk,
        rowsum_G0=G0.abs().sum(-1), rowsum_Gk=Gk.abs().sum(-1),
    )


@torch.no_grad()
def whitney_triangles(pos: Tensor, faces: Tensor, face_edges: Tensor, face_edge_signs: Tensor) -> dict:
    """Whitney 1-form Galerkin blocks of a triangle complex (2-D or surface in 3-D).

    Args:
        pos: ``(n0, D)``; faces: ``(n2, 3)`` (given orientation); face_edges/face_edge_signs: ``(n2, 3)`` edge id and
            orientation of local edge j (joining local vertices j and j+1).
    Returns:
        dict (float32 tensors): ``cells (n2, 3)``, ``signs (n2, 3)``, ``dir_edges (n2, 3)``, ``t (n2, 3, D)`` unit
        vectors of the local edges, ``G0 (n2, 3, 3)``, ``Gk (n2, 3, 3, 3)``, ``rowsum_G0 (n2, 3)``,
        ``rowsum_Gk (n2, 3, 3)``.
    """
    P = pos.to(torch.float64)
    V = P[faces]
    grads = barycentric_gradients(V)
    area = _simplex_measure(V)
    pairs = torch.tensor(_TRI_SLOTS, device=P.device)
    lam = torch.tensor(_TRI_Q, dtype=torch.float64, device=P.device)
    W = whitney1_values(lam, grads, pairs, face_edge_signs.to(torch.float64))
    t = V[:, pairs[:, 1]] - V[:, pairs[:, 0]]
    t = t / t.norm(dim=-1, keepdim=True).clamp_min(_TINY)
    G0, Gk = galerkin_blocks(W, (area / 3.0)[:, None].expand(-1, 3), t)
    return _pack_whitney(G0, Gk, t, face_edges, face_edge_signs, face_edges)


@torch.no_grad()
def whitney_tets(pos: Tensor, tets: Tensor, tet_edge: Tensor, tet_faces: Tensor) -> tuple[dict, dict]:
    """Whitney 1-form (edges, 6x6) and 2-form (faces, 4x4) Galerkin blocks of a tetrahedral complex.

    Both use the 6 edge directions of the tet as the tensor frame (their ``t t^T`` span Sym(3); the 4 face normals
    do not) and the 4-point degree-2 quadrature rule.

    Args:
        pos: ``(n0, 3)``; tets: ``(n3, 4)``; tet_edge: ``(n3, 6)`` edge ids of local pairs
            ``(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)``; tet_faces: ``(n3, 4)`` face ids (face i omits local vertex i).
    Returns:
        ``(whitney1, whitney2)`` dicts as in :func:`whitney_triangles` (``cells`` = edges resp. faces, ``dir_edges``
        = ``tet_edge`` for both, ``t (n3, 6, 3)``).
    """
    P = pos.to(torch.float64)
    V = P[tets]
    dev = P.device
    grads = barycentric_gradients(V)
    wq = (_simplex_measure(V) / 4.0)[:, None].expand(-1, 4)
    lam = torch.tensor(_TET_Q, dtype=torch.float64, device=dev)
    pairs = torch.tensor(_TET_EDGES, device=dev)
    esign = torch.where(tets[:, pairs[:, 0]] < tets[:, pairs[:, 1]], 1.0, -1.0).to(torch.float64)
    t = V[:, pairs[:, 1]] - V[:, pairs[:, 0]]
    t = t / t.norm(dim=-1, keepdim=True).clamp_min(_TINY)
    W1 = whitney1_values(lam, grads, pairs, esign)
    G0, Gk = galerkin_blocks(W1, wq, t)
    w1 = _pack_whitney(G0, Gk, t, tet_edge, esign, tet_edge)
    loc = torch.tensor(_TET_FACES, device=dev)  # (4, 3) local ids of face i (increasing)
    glob = tets[:, loc]  # (n3, 4, 3)
    face_local = torch.gather(loc.expand(tets.shape[0], 4, 3), 2, glob.argsort(dim=2))
    W2 = whitney2_values_tet(lam, grads, face_local)
    G0, Gk = galerkin_blocks(W2, wq, t)
    w2 = _pack_whitney(G0, Gk, t, tet_faces, torch.ones_like(tet_faces, dtype=torch.float64), tet_edge)
    return w1, w2
