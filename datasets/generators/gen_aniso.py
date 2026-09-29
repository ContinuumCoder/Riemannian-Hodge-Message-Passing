"""Anisotropy task suite: PDEs whose operator is a Whitney/Galerkin (FEEC) Hodge star of a misaligned SPD material
tensor field (tasks AHP, ASURF, ACURL, ADARCY; loaders in ``rhmp/tasks/aniso.py``, details in docs/ANISO_TASKS.md).

Every sample draws a new mesh and new smooth random fields; the principal direction of the material tensor follows a
smooth random vector field that is independent of the mesh (misaligned with the edges), the eigenvalue ratio is
``r_eff = 1 + (r - 1) w`` with ``w -> 1`` away from the zeros of the direction field (maximal ratio ``r`` in
{10, 100}: sets ``*_r10`` / ``*_r100``; both ratios share meshes, fields and sources), and a scalar magnitude varies
with contrast 10.  Targets are exact FEEC solutions (sparse direct solves, float64, relative residuals <= 1e-8):

AHP     2-D, random Delaunay mesh of [0,1]^2 (n0 ~ U{1000..2000}, the HP meshes of ``gen_HP.draw_mesh_2d``):
        -div(Sigma grad u) = f, u = 0 on the boundary; P1 stiffness = d0^T M1(Sigma) d0 (M1 = Whitney 1-form mass with
        the per-face tensor), lumped right-hand side.  Target u (nodes).
ASURF   closed surfaces of ``gen_surf`` (ellipsoids, SH-perturbed spheres, perturbed tori; median edge 0.07):
        screened Poisson (M + K(Sigma)) u = M f and heat flow (10 implicit steps of (M + K(Sigma)/10) v' = M v) with the
        tangent-plane fibre tensor Sigma_f = eps s(x) [P + (r_eff - 1) tau tau^T] / sqrt(r_eff), eps = 0.05.
        Targets u, heat (nodes).
ACURL   3-D, random Delaunay tetrahedra of [0,1]^3 (n0 ~ U{2000..3000}, ``gen_TET.draw_mesh_3d``): anisotropic
        curl-curl / eddy-current problem  curl(nu curl A) + eps A = J  with lowest-order Nedelec (Whitney 1-form)
        elements: (d1^T M2(nu) d1 + eps M1) A = M1 j, tangential A = 0 on the boundary (PEC), nu = s(x) A_prolate(x),
        eps = 10; J = curl W (divergence free): exact line integrals j_e = int_e J.dl minus their discrete gradient
        part (d0^T M1 j = 0 at interior nodes, so A is weakly divergence free).  Targets A (edges) and B = d1 A
        (faces, derived by the loader).
ADARCY  3-D Darcy flow -div(K grad p) = f, p = 0 on the boundary, K = s(x) A_prolate(x):
        primal P1 pressure (d0^T M1(K) d0 p = M0 f; target p at nodes, task ADARCYp) and mixed RT0-P0 = Whitney 2-/3-form
        discretisation [[M2(K^-1), -d2^T], [-d2, 0]] [J; P] = [0; -F] with F_T = int_T f: face fluxes J (target of
        ADARCY; per-tet conservation d2 J = F holds exactly) and tet pressures P.

Model inputs (all E(n)-invariant; directions only through edge projections, whose values on the edges of a cell
determine the cell tensor): nodes f; edges log(t_e^T Sigma t_e) (mean over the incident cells); faces / tets tensor
invariants (log det, log eigenvalue ratio), 3-D faces log(n_f^T nu n_f) (ACURL) or log(n_f^T K^-1 n_f) (ADARCY);
odd cochain sources: ACURL j (edges), ADARCY F_T (tets).  Evaluation-only ground truth: the per-cell tensors
(packed symmetric, their eigenvectors are the principal directions) and representability diagnostics (see
``aniso_fields.representability``): ``in_cone`` = the cell tensor is reachable by the tensor family
{b I + sum_j a_j t_j t_j^T : b > 0, a >= 0}, ``s_max`` = smallest max |s_j| of the family b expm(sum_j s_j t_j t_j^T).

Packing: flat concatenations with offsets ptr0..ptr3 (float32 fields, int32 cells) as ``gen_HP.pack``; canonical edges
(src < dst, lexicographic) and, for tets, canonical faces (sorted triples, lexicographic); triangle faces keep their
orientation.  ``*_fine.pt`` re-solves the first ``n_fine`` samples of the sequential 70/15/15 test split on meshes with
4x the nodes (same fields and sources).

Usage (CPU only):
  CUDA_VISIBLE_DEVICES= python3 -u datasets/generators/gen_aniso.py --workers 24                  # all tasks, r = 10, 100
  CUDA_VISIBLE_DEVICES= python3 -u datasets/generators/gen_aniso.py --tasks AHP ACURL --n 300 --n-fine 20
  CUDA_VISIBLE_DEVICES= python3 -u datasets/generators/gen_aniso.py --analyse 12 --workers 24     # representability oracles
  CUDA_VISIBLE_DEVICES= python3 -u datasets/generators/gen_aniso.py --selftest                    # convergence checks
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aniso_fields as AF  # noqa: E402
import gen_HP  # noqa: E402
import gen_surf  # noqa: E402
import gen_TET  # noqa: E402
import surfaces as S  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASKS = ("AHP", "ASURF", "ACURL", "ADARCY")
RATIOS = (10.0, 100.0)
RES_TOL = 1e-8          # maximal accepted relative residual of every direct solve
CONFIG = {
    # N / n_fine: samples of the main / 4x-finer set; seed: base seed of the task (fields, meshes);
    # contrast: range of the scalar magnitude s(x); delta: isotropic-core width of the direction field;
    # ell: length scale of the direction field; ell_s: of the magnitude field
    "AHP": dict(N=5000, n_fine=500, nmin=1000, nmax=2000, seed=2050, contrast=10.0, delta=0.3, ell=(0.2, 0.5),
                ell_s=(0.1, 0.3), src_w=(0.05, 0.2), fine_factor=4),
    "ASURF": dict(N=4000, n_fine=400, seed=2051, h=0.07, eps=0.05, T=0.05, heat_steps=10, contrast=10.0, delta=0.3,
                  ell=(0.4, 0.8), ell_s=(0.3, 0.8), fine_factor=4),
    "ACURL": dict(N=3000, n_fine=200, nmin=2000, nmax=3000, seed=2052, eps=10.0, contrast=10.0, delta=0.3,
                  ell=(0.3, 0.6), ell_s=(0.3, 0.6), ell_src=(0.25, 0.5), fine_factor=4),
    "ADARCY": dict(N=3000, n_fine=200, nmin=2000, nmax=3000, seed=2053, contrast=10.0, delta=0.3, ell=(0.3, 0.6),
                   ell_s=(0.3, 0.6), src_w=(0.1, 0.25), fine_factor=4),
}


def rtag(r: float) -> str:
    return f"r{int(r)}"


# ======================================================================================================================
# shared helpers
# ======================================================================================================================
def draw_sources(rng: np.random.Generator, dim: int, widths=(0.05, 0.2)) -> dict:
    """2-5 Gaussian sources (the ``gen_HP`` recipe): centres U[0.1, 0.9]^dim, amplitudes +-U[1, 3]."""
    n_src = int(rng.integers(2, 6))
    return dict(c=rng.uniform(0.1, 0.9, (n_src, dim)), w=rng.uniform(*widths, n_src),
                a=rng.uniform(1.0, 3.0, n_src) * rng.choice([-1.0, 1.0], n_src))


def eval_sources(src: dict, x: np.ndarray) -> np.ndarray:
    d2 = ((x[:, None, :] - src["c"][None]) ** 2).sum(-1)
    return (src["a"][None] * np.exp(-d2 / (2 * src["w"][None] ** 2))).sum(-1)


def solve_spd(A: sp.spmatrix, b: np.ndarray) -> tuple[np.ndarray, float]:
    """Sparse direct solve (SuperLU) and relative residual."""
    A = A.tocsc()
    x = spla.splu(A).solve(b)
    return x, float(np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300))


def _check(res: float, what: str, idx) -> None:
    if not np.isfinite(res) or res > RES_TOL:
        raise RuntimeError(f"sample {idx}: {what} residual {res:.3e} > {RES_TOL:.0e}")


def cell_dirs(pos: np.ndarray, cells: np.ndarray, pairs) -> np.ndarray:
    """Unit edge vectors ``(n_c, m, D)`` of the local vertex pairs of every cell (dyad frame of the cell)."""
    V = pos[cells]
    pr = np.asarray(pairs)
    t = V[:, pr[:, 1]] - V[:, pr[:, 0]]
    return t / np.linalg.norm(t, axis=-1, keepdims=True)


def rep_stats(rep: dict, prefix: str = "") -> dict:
    """Summary statistics of :func:`aniso_fields.representability`."""
    s = rep["s_max"]
    return {f"{prefix}cone_frac": float(rep["in_cone"].mean()), f"{prefix}smax_median": float(np.median(s)),
            f"{prefix}smax_p90": float(np.percentile(s, 90)), f"{prefix}smax_max": float(s.max()),
            f"{prefix}frac_smax_le2": float((s <= 2).mean()), f"{prefix}frac_smax_le3": float((s <= 3).mean()),
            f"{prefix}frac_smax_le5": float((s <= 5).mean())}


def interior_neg_weight_frac(K: sp.csr_matrix, edges: np.ndarray, bnd_nodes: np.ndarray | None) -> float:
    """Fraction of edges (not joining two boundary nodes) whose P1 graph-Laplacian weight is negative."""
    w = AF.edge_weights(K, edges)
    keep = np.ones(len(edges), bool) if bnd_nodes is None else ~(bnd_nodes[edges[:, 0]] & bnd_nodes[edges[:, 1]])
    return float((w[keep] < 0).mean())


# ======================================================================================================================
# AHP: 2-D anisotropic Poisson on the HP meshes
# ======================================================================================================================
def ahp_fields(seed: int, idx: int, cfg: dict) -> dict:
    rng = np.random.default_rng([seed, 1, idx])
    return dict(V=AF.rff_draw(rng, 2, rng.uniform(*cfg["ell"]), 2), s=AF.rff_draw(rng, 2, rng.uniform(*cfg["ell_s"])),
                src=draw_sources(rng, 2, cfg["src_w"]))


def ahp_tensors(F: dict, pts: np.ndarray, faces: np.ndarray, r: float, cfg: dict):
    cent = pts[faces].mean(1)
    V = AF.unit_vector_field(F["V"], cent)
    logs = AF.log_magnitude(F["s"], cent, cfg["contrast"])
    A, tau, r_eff = AF.uniaxial_planar(V, r, cfg["delta"])
    return np.exp(logs)[:, None, None] * A, tau, r_eff, logs


def ahp_mesh(seed: int, idx: int, n_tot: int) -> tuple[np.ndarray, np.ndarray]:
    """HP mesh (``gen_HP.draw_mesh_2d``) with positions rounded to float32 first (stored precision = solved
    precision, so the stored mesh reproduces the operators exactly)."""
    pts, faces = gen_HP.draw_mesh_2d(seed, idx, n_tot)
    pts = pts.astype(np.float32).astype(np.float64)
    P = pts[faces]
    sa = (P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) - (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0])
    if (sa <= 0).any():
        raise RuntimeError(f"sample {idx}: triangle orientation lost after float32 rounding")
    return pts, faces.astype(np.int64)


def ahp_sample(args) -> dict:
    """One AHP mesh with the targets of every ratio (``out[rtag(r)]``)."""
    idx, n_tot, seed, cfg, ratios = args
    t0 = time.time()
    F = ahp_fields(seed, idx, cfg)
    pts, faces = ahp_mesh(seed, idx, n_tot)
    n = len(pts)
    topo = AF.tri_topology(faces, n)
    edges = topo["edges"]
    f = eval_sources(F["src"], pts)
    bnd = (pts[:, 0] < 1e-12) | (pts[:, 0] > 1 - 1e-12) | (pts[:, 1] < 1e-12) | (pts[:, 1] > 1 - 1e-12)
    I = np.where(~bnd)[0]
    t_slot = cell_dirs(pts, faces, AF.TRI_SLOTS)
    out = dict(pos=pts.astype(np.float32), faces=faces.astype(np.int32), f=f.astype(np.float32), bnd=bnd,
               stats=dict(idx=int(idx), n0=n, n1=len(edges), n2=len(faces)))
    for r in ratios:
        Sig, tau, r_eff, logs = ahp_tensors(F, pts, faces, r, cfg)
        K, mass, _ = AF.p1_stiffness(pts, faces, Sig)
        b = mass * f
        u = np.zeros(n)
        u[I], res = solve_spd(K[I][:, I], b[I])
        _check(res, "AHP", idx)
        proj = np.einsum("fjd,fde,fje->fj", t_slot, Sig, t_slot)
        rep = AF.representability(Sig, t_slot)
        out[rtag(r)] = dict(
            u=u.astype(np.float32), logproj_edge=np.log(AF.cell_edge_average(proj, topo["face_edges"], len(edges)))
            .astype(np.float32), logdet_face=(2.0 * logs).astype(np.float32),
            logratio_face=np.log(r_eff).astype(np.float32), sigma_face=AF.sym_pack(Sig).astype(np.float32),
            in_cone_face=rep["in_cone"], smax_face=rep["s_max"].astype(np.float32),
            stats=dict(residual=res, neg_weight_frac=interior_neg_weight_frac(K, edges, bnd),
                       ratio_median=float(np.median(r_eff)), frac_ratio_gt_half=float((r_eff > 0.5 * r).mean()),
                       u_std=float(u.std()), **rep_stats(rep)))
    out["stats"]["seconds"] = time.time() - t0
    return out


# ======================================================================================================================
# ASURF: fibre-aligned conduction on closed surfaces
# ======================================================================================================================
def asurf_surface(idx: int, family: str, seed: int, h: float):
    """Mesh sample ``idx`` of ``family`` at edge length ``h``: the shape, its rigid motion and hence the continuous
    surface do not depend on ``h`` (the 4x-finer test mesh is the same surface).  Returns ``pts, faces, params``."""
    fam = gen_surf.FAMILIES.index(family)
    cfg = gen_surf.default_config(h=h)
    last = None
    for attempt in range(10):
        try:
            pts, faces, params = gen_surf.make_surface(family, np.random.default_rng([seed, 21, fam, idx, attempt]),
                                                       cfg)
            rm = np.random.default_rng([seed, 22, fam, idx])
            Q = S.random_rotation(rm)
            pts = pts @ Q.T + rm.uniform(-0.5, 0.5, 3)
            pts = pts.astype(np.float32).astype(np.float64)       # stored precision = solved precision
            rep = S.mesh_report(pts, faces)
            S.assert_valid_closed(rep, gen_surf.GENUS[family], f"({family} #{idx})")
            params.update(attempts=attempt + 1, **{k: rep[k] for k in ("edge_median", "min_angle", "max_angle",
                                                                        "p99_aspect", "area")})
            return pts, faces.astype(np.int64), params
        except RuntimeError as e:
            last = e
    raise RuntimeError(f"ASURF sample {idx}: no valid surface: {last}")


def asurf_fields(seed: int, idx: int, family: str, coarse_pts: np.ndarray) -> dict:
    """Direction / magnitude / source fields in the embedding space; bump centres are vertices of the coarse mesh."""
    fam = gen_surf.FAMILIES.index(family)
    rng = np.random.default_rng([seed, 23, fam, idx])
    cfg = CONFIG["ASURF"]
    F = dict(V=AF.rff_draw(rng, 3, rng.uniform(*cfg["ell"]), 3), s=AF.rff_draw(rng, 3, rng.uniform(*cfg["ell_s"])))
    F["f_grf"] = AF.rff_draw(rng, 3, rng.uniform(0.25, 0.6))
    nb = int(rng.integers(1, 5))
    F["bumps"] = dict(c=coarse_pts[rng.choice(len(coarse_pts), nb, replace=False)], w=rng.uniform(0.1, 0.3, nb),
                      a=rng.uniform(1.0, 2.5, nb) * rng.choice([-1.0, 1.0], nb))
    return F


def asurf_source(F: dict, pts: np.ndarray) -> np.ndarray:
    """Embedding-space GRF (unit variance) + Gaussian bumps (the ``gen_surf.surface_source`` recipe)."""
    f = AF.rff_eval(F["f_grf"], pts)[:, 0]
    b = F["bumps"]
    d2 = ((pts[:, None, :] - b["c"][None]) ** 2).sum(-1)
    return f + (b["a"][None] * np.exp(-d2 / (2 * b["w"][None] ** 2))).sum(1)


def asurf_tensors(F: dict, pts: np.ndarray, faces: np.ndarray, r: float, cfg: dict):
    P = pts[faces]
    nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    cent = P.mean(1)
    V = AF.unit_vector_field(F["V"], cent)
    logs = AF.log_magnitude(F["s"], cent, cfg["contrast"])
    A, tau, r_eff = AF.uniaxial_tangent(V, nrm, r, cfg["delta"])
    return cfg["eps"] * np.exp(logs)[:, None, None] * A, tau, r_eff, logs, nrm


def asurf_sample(args) -> dict:
    """One ASURF surface (``fine``: the same surface meshed at h/2) with the targets of every ratio."""
    idx, family, seed, cfg, ratios, fine = args
    t0 = time.time()
    coarse, faces_c, params = asurf_surface(idx, family, seed, cfg["h"])
    F = asurf_fields(seed, idx, family, coarse)
    if fine:
        pts, faces, params = asurf_surface(idx, family, seed, cfg["h"] / 2.0)
    else:
        pts, faces = coarse, faces_c
    n = len(pts)
    topo = AF.tri_topology(faces, n)
    edges = topo["edges"]
    f = asurf_source(F, pts)
    t_slot = cell_dirs(pts, faces, AF.TRI_SLOTS)
    out = dict(pos=pts.astype(np.float32), faces=faces.astype(np.int32), f=f.astype(np.float32),
               stats=dict(idx=int(idx), family=family, genus=gen_surf.GENUS[family], n0=n, n1=len(edges),
                          n2=len(faces), params=params))
    for r in ratios:
        Sig, tau, r_eff, logs, nrm = asurf_tensors(F, pts, faces, r, cfg)
        K, mass, _ = AF.p1_stiffness(pts, faces, Sig)
        M = sp.diags(mass)
        u, res_u = solve_spd(M + K, mass * f)
        _check(res_u, "ASURF screened Poisson", idx)
        Ah = (M + K / cfg["heat_steps"]).tocsc()
        lu = spla.splu(Ah)
        v, res_h = f.copy(), 0.0
        for _ in range(cfg["heat_steps"]):
            rhs = mass * v
            v = lu.solve(rhs)
            res_h = max(res_h, float(np.linalg.norm(Ah @ v - rhs) / max(np.linalg.norm(rhs), 1e-300)))
        _check(res_h, "ASURF heat", idx)
        drift = float(abs((mass * v).sum() - (mass * f).sum()) / max(np.abs(mass * f).sum(), 1e-300))
        proj = np.einsum("fjd,fde,fje->fj", t_slot, Sig, t_slot)
        rep = AF.representability(Sig, t_slot, nrm)
        out[rtag(r)] = dict(
            u=u.astype(np.float32), heat=v.astype(np.float32),
            logproj_edge=np.log(AF.cell_edge_average(proj, topo["face_edges"], len(edges))).astype(np.float32),
            logdet_face=(2.0 * (np.log(cfg["eps"]) + logs)).astype(np.float32),
            logratio_face=np.log(r_eff).astype(np.float32), sigma_face=AF.sym_pack(Sig).astype(np.float32),
            in_cone_face=rep["in_cone"], smax_face=rep["s_max"].astype(np.float32),
            stats=dict(res_u=res_u, res_heat=res_h, heat_mass_drift=drift,
                       neg_weight_frac=interior_neg_weight_frac(K, edges, None),
                       ratio_median=float(np.median(r_eff)), frac_ratio_gt_half=float((r_eff > 0.5 * r).mean()),
                       u_std=float(u.std()), heat_std=float(v.std()), **rep_stats(rep)))
    out["stats"]["seconds"] = time.time() - t0
    return out


# ======================================================================================================================
# 3-D helpers
# ======================================================================================================================
def tet_quad_points(pts: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """``(n3, 4, 3)`` points of the 4-point degree-2 rule (weights |T|/4)."""
    return np.einsum("qi,tid->tqd", AF.TET_Q, pts[tets])


def face_normals(pts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    P = pts[faces]
    nf = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    return nf / np.linalg.norm(nf, axis=1, keepdims=True)


def tet_face_average(values: np.ndarray, tet_faces: np.ndarray, n2: int) -> np.ndarray:
    """Mean over the (1-2) tets of every face of per-(tet, local face) values ``(n3, 4)``."""
    return AF.cell_edge_average(values, tet_faces, n2)


def curl_line_integrals(p: dict, pts: np.ndarray, edges: np.ndarray, norm: float) -> np.ndarray:
    """Exact line integrals ``int_e J . dl`` of ``J = curl W / norm``, ``W = sum_k a_k cos(2 pi k.x + ph_k)``.

    ``curl W = -2 pi sum_k sin(2 pi k.x + ph_k) (k x a_k)``; along ``x = p + s e``, ``s in [0, 1]``:
    ``int_0^1 sin(al + be s) ds = sin(al + be/2) sinc(be / 2pi)`` (numpy's normalised sinc).
    """
    P0 = pts[edges[:, 0]]
    e = pts[edges[:, 1]] - P0
    kxa = np.cross(p["k"], p["a"])                                  # (N_RFF, 3)
    al = 2 * np.pi * P0 @ p["k"].T + p["ph"]                        # (n1, N_RFF)
    be = 2 * np.pi * e @ p["k"].T
    proj = e @ kxa.T                                                # (n1, N_RFF)
    return (-2 * np.pi * proj * np.sin(al + be / 2) * np.sinc(be / (2 * np.pi))).sum(1) / norm


def curl_field(p: dict, x: np.ndarray, norm: float) -> np.ndarray:
    """``J(x) = curl W / norm``, ``(M, 3)`` (evaluation / tests)."""
    kxa = np.cross(p["k"], p["a"])
    return (-2 * np.pi * np.sin(2 * np.pi * x @ p["k"].T + p["ph"]) @ kxa) / norm


def curl_norm(p: dict) -> float:
    """RMS of ``curl W``: ``sqrt(sum_k (2 pi)^2 |k x a_k|^2 / 2)`` (mesh independent)."""
    kxa = np.cross(p["k"], p["a"])
    return float(np.sqrt(((2 * np.pi) ** 2 * (kxa ** 2).sum(1) / 2.0).sum()))


def tet_mesh(seed: int, idx: int, n_tot: int) -> tuple[np.ndarray, np.ndarray, int]:
    pts, tets, n_drop = gen_TET.draw_mesh_3d(seed, idx, n_tot)
    pts = pts.astype(np.float32).astype(np.float64)                # stored precision = solved precision
    X = pts[tets]
    det = np.einsum("mi,mi->m", np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]), X[:, 3] - X[:, 0])
    if (det <= 0).any():
        raise RuntimeError(f"sample {idx}: non-positive tet after float32 rounding")
    return pts, tets.astype(np.int64), n_drop


def tet_proj_inputs(pts, tets, topo, T: np.ndarray, inverse_normal: bool = False):
    """Edge projections ``log(t_e^T T t_e)`` (mean over incident tets) and face normal projections
    ``log(n_f^T T n_f)`` (``inverse_normal``: of ``T^-1``; mean over the adjacent tets)."""
    t6 = cell_dirs(pts, tets, AF.TET_EDGES)
    pe = np.einsum("tjd,tde,tje->tj", t6, T, t6)
    n1, n2 = len(topo["edges"]), len(topo["faces"])
    loge = np.log(AF.cell_edge_average(pe, topo["tet_edges"], n1))
    nf = face_normals(pts, topo["faces"])[topo["tet_faces"]]        # (n3, 4, 3)
    Tn = np.linalg.inv(T) if inverse_normal else T
    pf = np.einsum("tjd,tde,tje->tj", nf, Tn, nf)
    logf = np.log(tet_face_average(pf, topo["tet_faces"], n2))
    return loge, logf, t6


# ======================================================================================================================
# ACURL: anisotropic curl-curl (Nedelec)
# ======================================================================================================================
def acurl_fields(seed: int, idx: int, cfg: dict) -> dict:
    rng = np.random.default_rng([seed, 1, idx])
    F = dict(V=AF.rff_draw(rng, 3, rng.uniform(*cfg["ell"]), 3), s=AF.rff_draw(rng, 3, rng.uniform(*cfg["ell_s"])),
             W=AF.rff_draw(rng, 3, rng.uniform(*cfg["ell_src"]), 3))
    F["J_norm"] = curl_norm(F["W"])
    return F


def acurl_tensors(F: dict, pts: np.ndarray, tets: np.ndarray, r: float, cfg: dict):
    cent = pts[tets].mean(1)
    V = AF.unit_vector_field(F["V"], cent)
    logs = AF.log_magnitude(F["s"], cent, cfg["contrast"])
    A, tau, r_eff = AF.uniaxial_3d(V, r, cfg["delta"])
    return np.exp(logs)[:, None, None] * A, tau, r_eff, logs


def acurl_operator(pts, tets, topo, nu: np.ndarray, eps: float, M1I: sp.csr_matrix | None = None):
    """``S = d1^T M2(nu) d1 + eps M1`` and the identity Whitney 1-form mass ``M1``."""
    n1, n2 = len(topo["edges"]), len(topo["faces"])
    if M1I is None:
        M1I = AF.assemble(AF.tet_whitney1_mass(pts, tets, topo, None), topo["tet_edges"], n1)
    M2 = AF.assemble(AF.tet_whitney2_mass(pts, tets, topo, nu), topo["tet_faces"], n2)
    d1 = topo["d1"]
    return (d1.T @ M2 @ d1 + eps * M1I).tocsr(), M1I


def acurl_source(F: dict, pts: np.ndarray, topo: dict, M1I: sp.csr_matrix) -> tuple[np.ndarray, float]:
    """Discretely divergence-free edge source: the exact line integrals ``j = int_e J.dl`` of ``J = curl W`` minus
    their M1-orthogonal gradient part, ``j <- j - d0 chi`` with ``(d0^T M1 d0) chi = d0^T M1 j`` on the interior nodes
    (``chi = 0`` on the boundary).  The Whitney interpolant of a divergence-free field is not weakly divergence-free
    (normal jumps across faces), and because ``eps`` damps gradients far less than the curl-curl term damps the rest,
    that 7-10 % gradient residue would make up ~40 % of A.  After the projection ``d0^T M1 j = 0`` exactly at interior
    nodes, hence ``d0^T M1 A = 0`` (weak Coulomb gauge) for the solution: pure curl-curl physics.

    Returns:
        ``j (n1,)`` and the removed fraction ``|d0 chi|_M1 / |j_exact|_M1``.
    """
    j = curl_line_integrals(F["W"], pts, topo["edges"], F["J_norm"])
    In = np.where(~topo["boundary_nodes"])[0]
    D = topo["d0"][:, In]
    chi = spla.splu((D.T @ M1I @ D).tocsc()).solve(D.T @ (M1I @ j))
    g = D @ chi
    nrm = lambda v: float(np.sqrt(v @ (M1I @ v)))  # noqa: E731
    return j - g, nrm(g) / max(nrm(j), 1e-300)


def acurl_sample(args) -> dict:
    idx, n_tot, seed, cfg, ratios = args
    t0 = time.time()
    F = acurl_fields(seed, idx, cfg)
    pts, tets, n_drop = tet_mesh(seed, idx, n_tot)
    n = len(pts)
    topo = AF.tet_topology(tets, n)
    edges, n1 = topo["edges"], len(topo["edges"])
    M1I = AF.assemble(AF.tet_whitney1_mass(pts, tets, topo, None), topo["tet_edges"], n1)
    j, removed = acurl_source(F, pts, topo, M1I)
    b = M1I @ j
    I = np.where(~topo["boundary_edges"])[0]
    out = dict(pos=pts.astype(np.float32), tets=tets.astype(np.int32), j=j.astype(np.float32),
               bnd_edge=topo["boundary_edges"],
               stats=dict(idx=int(idx), n0=n, n1=n1, n2=len(topo["faces"]), n3=len(tets), dropped_tets=n_drop,
                          grad_removed=removed,
                          div_j_rel=float(np.linalg.norm((topo["d0"].T @ b)[~topo["boundary_nodes"]]) /
                                          max(np.linalg.norm(b), 1e-300))))
    for r in ratios:
        nu, tau, r_eff, logs = acurl_tensors(F, pts, tets, r, cfg)
        Sop, _ = acurl_operator(pts, tets, topo, nu, cfg["eps"], M1I)
        A = np.zeros(n1)
        A[I], res = solve_spd(Sop[I][:, I], b[I])
        _check(res, "ACURL", idx)
        loge, logf, t6 = tet_proj_inputs(pts, tets, topo, nu)
        rep = AF.representability(nu, t6)
        B = topo["d1"] @ A
        MA = M1I @ A
        gauge = float(np.linalg.norm((topo["d0"].T @ MA)[~topo["boundary_nodes"]]) / max(np.linalg.norm(MA), 1e-300))
        out[rtag(r)] = dict(
            A=A.astype(np.float32), logproj_edge=loge.astype(np.float32), lognn_face=logf.astype(np.float32),
            logdet_tet=(3.0 * logs).astype(np.float32), logratio_tet=np.log(r_eff).astype(np.float32),
            nu_tet=AF.sym_pack(nu).astype(np.float32), in_cone_tet=rep["in_cone"],
            smax_tet=rep["s_max"].astype(np.float32),
            stats=dict(residual=res, ratio_median=float(np.median(r_eff)),
                       frac_ratio_gt_half=float((r_eff > 0.5 * r).mean()), A_rms=float(np.sqrt((A ** 2).mean())),
                       B_rms=float(np.sqrt((B ** 2).mean())), divB_max=float(np.abs(topo["d2"] @ B).max()),
                       gauge=gauge,
                       **rep_stats(rep)))
    out["stats"]["seconds"] = time.time() - t0
    return out


# ======================================================================================================================
# ADARCY: anisotropic Darcy flow, primal P1 pressure and mixed RT0 face fluxes
# ======================================================================================================================
def adarcy_fields(seed: int, idx: int, cfg: dict) -> dict:
    rng = np.random.default_rng([seed, 1, idx])
    return dict(V=AF.rff_draw(rng, 3, rng.uniform(*cfg["ell"]), 3), s=AF.rff_draw(rng, 3, rng.uniform(*cfg["ell_s"])),
                src=draw_sources(rng, 3, cfg["src_w"]))


def mixed_solve(M2: sp.spmatrix, d2: sp.spmatrix, F_T: np.ndarray, rtol: float = 1e-13):
    """Mixed RT0-P0 system ``[[M2, -d2^T], [-d2, 0]] [J; P] = [0; -F]``.

    Pressure Schur complement ``d2 M2^-1 d2^T P = F`` by PCG (``M2`` factorised once; preconditioner = the two-point
    flux operator ``d2 diag(M2)^-1 d2^T``, factorised), then ``J = M2^-1 d2^T P``; falls back to a sparse LU of the full
    saddle-point matrix if PCG does not reach ``RES_TOL``.  ~3x faster and ~4x less memory than the saddle-point LU on
    10K-node meshes (identical solutions to 2e-14).

    Returns:
        ``J (n2,)``, ``P (n3,)``, relative residual of the saddle-point system, conservation residual
        ``max |d2 J - F| / max |F|``.
    """
    n2, n3 = d2.shape[1], d2.shape[0]
    M2 = M2.tocsc()
    rhs = np.r_[np.zeros(n2), -F_T]
    KKT = sp.bmat([[M2, -d2.T], [-d2, None]], format="csc")
    luM = spla.splu(M2)
    luS = spla.splu((d2 @ sp.diags(1.0 / M2.diagonal()) @ d2.T).tocsc())
    Sop = spla.LinearOperator((n3, n3), matvec=lambda q: d2 @ luM.solve(d2.T @ q), dtype=np.float64)
    Pre = spla.LinearOperator((n3, n3), matvec=luS.solve, dtype=np.float64)
    P, info = spla.cg(Sop, F_T, M=Pre, rtol=rtol, maxiter=10000)
    J = luM.solve(d2.T @ P)
    res = float(np.linalg.norm(KKT @ np.r_[J, P] - rhs) / max(np.linalg.norm(rhs), 1e-300))
    if info != 0 or not np.isfinite(res) or res > RES_TOL:
        x, res = solve_spd(KKT, rhs)
        J, P = x[:n2], x[n2:]
    cons = float(np.abs(d2 @ J - F_T).max() / max(np.abs(F_T).max(), 1e-300))
    return J, P, res, cons


def mixed_darcy(pts, tets, topo, Kinv: np.ndarray, F_T: np.ndarray):
    """Mixed RT0-P0 Darcy (Whitney 2-/3-forms) with the resistivity ``K^-1`` per tet, p = 0 on the boundary (natural
    condition): :func:`mixed_solve` with ``M2 = M2(K^-1)``."""
    M2 = AF.assemble(AF.tet_whitney2_mass(pts, tets, topo, Kinv), topo["tet_faces"], len(topo["faces"]))
    return mixed_solve(M2, topo["d2"], F_T)


def adarcy_tensors(F: dict, pts: np.ndarray, tets: np.ndarray, r: float, cfg: dict):
    cent = pts[tets].mean(1)
    V = AF.unit_vector_field(F["V"], cent)
    logs = AF.log_magnitude(F["s"], cent, cfg["contrast"])
    A, tau, r_eff = AF.uniaxial_3d(V, r, cfg["delta"])
    Kt = np.exp(logs)[:, None, None] * A
    # exact inverse of the uniaxial tensor: A^-1 = r_eff^{1/3} (I - (r_eff - 1)/r_eff tau tau^T)
    Ainv = np.cbrt(r_eff)[:, None, None] * (np.eye(3)[None] - ((r_eff - 1.0) / r_eff)[:, None, None]
                                            * tau[:, :, None] * tau[:, None, :])
    return Kt, np.exp(-logs)[:, None, None] * Ainv, tau, r_eff, logs


def adarcy_sample(args) -> dict:
    idx, n_tot, seed, cfg, ratios = args
    t0 = time.time()
    F = adarcy_fields(seed, idx, cfg)
    pts, tets, n_drop = tet_mesh(seed, idx, n_tot)
    n = len(pts)
    topo = AF.tet_topology(tets, n)
    n1, n2, n3 = len(topo["edges"]), len(topo["faces"]), len(tets)
    f = eval_sources(F["src"], pts)
    vol = AF.simplex_measure(pts[tets])
    F_T = vol / 4.0 * eval_sources(F["src"], tet_quad_points(pts, tets).reshape(-1, 3)).reshape(n3, 4).sum(1)
    bnd = topo["boundary_nodes"]
    I = np.where(~bnd)[0]
    out = dict(pos=pts.astype(np.float32), tets=tets.astype(np.int32), f=f.astype(np.float32), bnd=bnd,
               F_tet=F_T.astype(np.float32),
               stats=dict(idx=int(idx), n0=n, n1=n1, n2=n2, n3=n3, dropped_tets=n_drop))
    for r in ratios:
        Kt, Kinv, tau, r_eff, logs = adarcy_tensors(F, pts, tets, r, cfg)
        Kp, mass, _ = AF.p1_stiffness(pts, tets, Kt)
        rhs = mass * f
        p = np.zeros(n)
        p[I], res_p = solve_spd(Kp[I][:, I], rhs[I])
        _check(res_p, "ADARCY primal", idx)
        J, P, res_m, cons = mixed_darcy(pts, tets, topo, Kinv, F_T)
        _check(res_m, "ADARCY mixed", idx)
        _check(cons, "ADARCY conservation", idx)
        loge, logf, t6 = tet_proj_inputs(pts, tets, topo, Kt, inverse_normal=True)
        rep1 = AF.representability(Kt, t6)
        rep2 = AF.representability(Kinv, t6)
        out[rtag(r)] = dict(
            p=p.astype(np.float32), J=J.astype(np.float32), p_tet=P.astype(np.float32),
            logproj_edge=loge.astype(np.float32), lognninv_face=logf.astype(np.float32),
            logdet_tet=(3.0 * logs).astype(np.float32), logratio_tet=np.log(r_eff).astype(np.float32),
            K_tet=AF.sym_pack(Kt).astype(np.float32),
            in_cone_tet=rep1["in_cone"], smax_tet=rep1["s_max"].astype(np.float32),
            in_cone_inv_tet=rep2["in_cone"], smax_inv_tet=rep2["s_max"].astype(np.float32),
            stats=dict(res_p=res_p, res_mixed=res_m, conservation=cons,
                       neg_weight_frac=interior_neg_weight_frac(Kp, topo["edges"], bnd),
                       ratio_median=float(np.median(r_eff)), frac_ratio_gt_half=float((r_eff > 0.5 * r).mean()),
                       p_std=float(p.std()), J_rms=float(np.sqrt((J ** 2).mean())), **rep_stats(rep1),
                       **rep_stats(rep2, "inv_")))
    out["stats"]["seconds"] = time.time() - t0
    return out


# ======================================================================================================================
# packing / driver
# ======================================================================================================================
SAMPLE_FN = {"AHP": ahp_sample, "ASURF": asurf_sample, "ACURL": acurl_sample, "ADARCY": adarcy_sample}
TOP_KEY = {"AHP": "faces", "ASURF": "faces", "ACURL": "tets", "ADARCY": "tets"}


def pack(samples: list[dict], r: float, meta: dict) -> dict:
    """Flat concatenation (offsets ``ptr0..ptr3``) of the shared keys and the keys of ratio ``r``."""
    import torch
    tag = rtag(r)
    st = [dict(s["stats"], **s[tag]["stats"]) for s in samples]
    ptr = lambda v: torch.from_numpy(np.concatenate([[0], np.cumsum(v)]).astype(np.int64))  # noqa: E731
    out = {"meta": meta}
    for k in ("n0", "n1", "n2", "n3"):
        if k in st[0]:
            out[f"ptr{k[1]}"] = ptr([s[k] for s in st])
    shared = [k for k in samples[0] if k not in ("stats",) and not k.startswith("r")]
    for key in shared:
        out[key] = torch.from_numpy(np.concatenate([s[key] for s in samples], 0))
    for key in samples[0][tag]:
        if key != "stats":
            out[key] = torch.from_numpy(np.concatenate([s[tag][key] for s in samples], 0))
    if "family" in st[0]:
        out["family"] = torch.tensor([gen_surf.FAMILIES.index(s["family"]) for s in st], dtype=torch.int64)
        out["genus"] = torch.tensor([s["genus"] for s in st], dtype=torch.int64)
    out["sample_stats"] = st
    return out


def job_args(task: str, ids, cfg: dict, seed: int, ratios, fine: bool) -> list:
    if task == "ASURF":
        return [(int(i), gen_surf.MAIN_FAMILIES[int(i) % 3], seed, cfg, ratios, fine) for i in ids]
    rng_n = np.random.default_rng([seed, 4])
    n_tots = rng_n.integers(cfg["nmin"], cfg["nmax"] + 1, cfg["N"])     # node counts of the main set (fixed by seed)
    fac = cfg["fine_factor"] if fine else 1
    return [(int(i), int(fac * n_tots[int(i)]), seed, cfg, ratios) for i in ids]


def run_jobs(fn, jobs: list, workers: int) -> list:
    if workers <= 1:
        return [fn(j) for j in jobs]
    with Pool(workers) as pool:
        return list(pool.imap(fn, jobs, chunksize=1 if len(jobs) < 4 * workers else 4))


def summary_of(samples: list[dict], r: float) -> dict:
    tag = rtag(r)
    st = [dict(s["stats"], **s[tag]["stats"]) for s in samples]
    keys = [k for k, v in st[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "idx"]
    out = {}
    for k in keys:
        v = np.array([s[k] for s in st], dtype=np.float64)
        out[k] = dict(mean=float(v.mean()), min=float(v.min()), max=float(v.max()))
    return out


def write_task(task: str, ratios, workers: int, out_dir: str, n=None, n_fine=None, seed=None, fine_only=False,
               overrides: dict | None = None) -> dict:
    """Generate ``<task>_r<R>.pt`` (+ ``_fine.pt``) for every ratio; returns the JSON summaries.

    ``overrides`` updates ``CONFIG[task]`` (e.g. ``nmin``/``nmax`` or ``h`` for tiny test sets)."""
    import torch
    cfg = dict(CONFIG[task], **(overrides or {}))
    if n is not None:
        cfg["N"] = int(n)
    if n_fine is not None:
        cfg["n_fine"] = int(n_fine)
    seed = cfg["seed"] if seed is None else int(seed)
    os.makedirs(out_dir, exist_ok=True)
    fn = SAMPLE_FN[task]
    sets = []
    if not fine_only:
        sets.append(("", np.arange(cfg["N"]), False))
    nt, nv = int(0.7 * cfg["N"]), int(0.15 * cfg["N"])
    fine_ids = np.arange(nt + nv, min(nt + nv + cfg["n_fine"], cfg["N"]))
    if cfg["n_fine"] > 0 and len(fine_ids):
        sets.append(("_fine", fine_ids, True))
    summaries = {}
    for suffix, ids, fine in sets:
        t0 = time.time()
        fw = max(1, workers // 3) if (fine and task in ("ACURL", "ADARCY")) else workers   # memory of 4x 3-D solves
        samples = run_jobs(fn, job_args(task, ids, cfg, seed, ratios, fine), fw)
        t_gen = time.time() - t0
        for r in ratios:
            name = f"{task}_{rtag(r)}{suffix}"
            path = os.path.join(out_dir, name + ".pt")
            meta = dict(task=task, ratio=r, set=name, N=len(samples), seed=seed, gen_seconds=t_gen,
                        generator="datasets/generators/gen_aniso.py", config={k: v for k, v in cfg.items()},
                        fine=fine, fine_factor=cfg.get("fine_factor"), coarse_ids=ids.tolist() if fine else None,
                        ratios_generated=list(ratios), workers=fw,
                        edge_order="unique src<dst pairs, lexicographic",
                        face_order=("sorted triples, lexicographic" if TOP_KEY[task] == "tets"
                                    else "as generated (oriented)"))
            torch.save(pack(samples, r, meta), path)
            summ = dict({k: v for k, v in meta.items() if k not in ("coarse_ids", "config")}, file=path,
                        file_MB=os.path.getsize(path) / 1e6, stats=summary_of(samples, r))
            json.dump(summ, open(path.replace(".pt", ".json"), "w"), indent=2)
            summaries[name] = summ
            s = summ["stats"]
            print(f"[{name}] N={len(samples)} gen {t_gen:.1f}s ({fw} workers) {summ['file_MB']:.0f} MB  n0 "
                  f"{s['n0']['mean']:.0f} [{s['n0']['min']:.0f}, {s['n0']['max']:.0f}]  "
                  + "  ".join(f"{k}={s[k]['max']:.1e}" for k in s if k.startswith("res") or k == "conservation")
                  + "".join(f"  {k}={s[k]['mean']:.3f}" for k in ("cone_frac", "inv_cone_frac", "neg_weight_frac",
                                                                   "smax_median") if k in s), flush=True)
        del samples                          # before the next set forks its workers (copy-on-write pages)
        gc.collect()
    return summaries


# ======================================================================================================================
# representability oracles (analysis; imports rhmp for the model's reference stars)
# ======================================================================================================================
def _rel(a: np.ndarray, b: np.ndarray, w: np.ndarray | None = None) -> float:
    w = np.ones_like(b) if w is None else w
    return float(np.sqrt((w * (a - b) ** 2).sum() / max((w * b ** 2).sum(), 1e-300)))


def edge_reconstruction(proj_edge: np.ndarray, cell_edges: np.ndarray, t: np.ndarray, nrm=None,
                        inverse: bool = False, floor: float = 1e-3) -> np.ndarray:
    """Cell tensors rebuilt from the model's edge inputs.

    The edge input of edge e is ``log p_e`` with ``p_e = t_e^T T t_e`` averaged over the cells incident to e.  Per
    cell, the coefficients ``alpha`` of ``T_f = sum_j alpha_j t_j t_j^T`` solve the Gram system
    ``sum_j alpha_j (t_i . t_j)^2 = p_i`` over the cell's edges (unique: the edge dyads span Sym).  Averaging can make
    the rebuilt tensor indefinite, so eigenvalues are clipped at ``floor x`` their mean.  ``inverse`` returns the
    inverse (ADARCY's 2-form metric is K^-1).

    Args:
        proj_edge: ``(n1,)`` edge projections; cell_edges: ``(n_c, m)`` edge ids of the cell's dyad directions;
        t: ``(n_c, m, D)`` unit directions; nrm: surface normals (tangent frame) or ``None``.
    Returns:
        ``(n_c, D, D)`` world-coordinate tensors.
    """
    fr = AF.local_frame(t, nrm)
    tc = t if fr is None else np.einsum("nda,njd->nja", fr, t)
    Gm = np.einsum("nid,njd->nij", tc, tc) ** 2
    alpha = np.linalg.solve(Gm, proj_edge[cell_edges][..., None])[..., 0]
    Tc = np.einsum("nj,nja,njb->nab", alpha, tc, tc)
    ev, U = np.linalg.eigh(Tc)
    ev = np.maximum(ev, floor * np.abs(ev).mean(1, keepdims=True))
    if inverse:
        ev = 1.0 / ev
    Tc = np.einsum("nij,nj,nkj->nik", U, ev, U)
    if fr is None:
        return Tc
    return np.einsum("nda,nab,neb->nde", fr, Tc, fr)


def _oracle_tensors(T: np.ndarray, t: np.ndarray, nrm=None, clips=(2.0, 3.0, 5.0)) -> dict:
    """Per-cell replacements of the true tensors: cone projection, isotropic part, full family with |s| <= L."""
    fr = AF.local_frame(t, nrm)
    out = {"cone": AF.cone_project(T, t, fr)}
    if nrm is None:
        d = T.shape[-1]
        iso = np.exp(np.linalg.slogdet(T)[1] / d)
        out["iso"] = iso[:, None, None] * np.eye(d)[None]
    else:
        Tc = np.einsum("nda,nde,neb->nab", fr, T, fr)
        iso = np.exp(np.linalg.slogdet(Tc)[1] / 2)
        P = np.eye(3)[None] - nrm[:, :, None] * nrm[:, None, :]
        out["iso"] = iso[:, None, None] * P
    s, logb, _ = AF.full_param_coeffs(T, t, fr)
    for L in clips:
        out[f"full_L{L:g}"] = AF.full_param_tensor(np.clip(s, -L, L), logb, t, fr)
    return out


def _rhmp_path() -> None:
    """The analysis uses rhmp (reference stars); make the repository importable when run as a script."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)


def _complex(pts, cells):
    import torch
    _rhmp_path()
    from rhmp.complex import CochainComplex
    if cells.shape[1] == 3:
        return CochainComplex.from_triangles(torch.as_tensor(pts), torch.as_tensor(cells), star="cotan", device="cpu")
    return CochainComplex.from_tetrahedra(torch.as_tensor(pts), torch.as_tensor(cells), star="barycentric",
                                          device="cpu")


def _star_on(Kc, k: int, gen_cells: np.ndarray, n0: int) -> np.ndarray:
    """Reference star of degree ``k`` of the rhmp complex, re-ordered to the generator's canonical cells."""
    import torch
    _rhmp_path()
    from rhmp.data import cell_alignment, edge_alignment
    st = Kc.star[k].double().numpy()
    if k == 1:
        perm, _ = edge_alignment(torch.as_tensor(gen_cells), Kc.cells[1], n0)
    else:
        perm, _ = cell_alignment(torch.as_tensor(gen_cells), Kc.cells[k], n0, oriented=False)
    return st[perm.numpy()]


def analyse_sample(args) -> dict:
    """Oracle solution errors of one test sample (every ratio): the exact Galerkin target vs the same solve with the
    per-cell tensors replaced by the cone projection / isotropic part / full family with bounded |s|, and vs diagonal
    stars (the model's reference star times the projected-material input = the diag + metric-reference prior)."""
    task, idx, cfg, ratios = args
    import torch
    torch.set_num_threads(1)
    res = {"idx": int(idx)}
    if task in ("AHP", "ASURF"):
        if task == "AHP":
            (_, n_tot, seed, _, _), = job_args(task, [idx], cfg, cfg["seed"], ratios, False)
            F = ahp_fields(seed, idx, cfg)
            pts, faces = ahp_mesh(seed, idx, n_tot)
            f = eval_sources(F["src"], pts)
            bnd = (pts.min(1) < 1e-12) | (pts.max(1) > 1 - 1e-12)
        else:
            family = gen_surf.MAIN_FAMILIES[idx % 3]
            pts, faces, _ = asurf_surface(idx, family, cfg["seed"], cfg["h"])
            F = asurf_fields(cfg["seed"], idx, family, pts)
            f = asurf_source(F, pts)
            bnd = np.zeros(len(pts), bool)
        n = len(pts)
        topo = AF.tri_topology(faces, n)
        edges = topo["edges"]
        t_slot = cell_dirs(pts, faces, AF.TRI_SLOTS)
        Kc = _complex(pts, faces)
        star1 = _star_on(Kc, 1, edges, n)
        I = np.where(~bnd)[0]
        for r in ratios:
            if task == "AHP":
                Sig, _, _, _ = ahp_tensors(F, pts, faces, r, cfg)
                nrm = None
            else:
                Sig, _, _, _, nrm = asurf_tensors(F, pts, faces, r, cfg)

            def solve(Kop, mass):
                if task == "AHP":
                    u = np.zeros(n)
                    u[I] = spla.splu(Kop[I][:, I].tocsc()).solve((mass * f)[I])
                    return u
                return spla.splu((sp.diags(mass) + Kop).tocsc()).solve(mass * f)
            Kt, mass, _ = AF.p1_stiffness(pts, faces, Sig)
            ut = solve(Kt, mass)
            row = {}
            for name, T in _oracle_tensors(Sig, t_slot, nrm).items():
                row[name] = _rel(solve(AF.p1_stiffness(pts, faces, T)[0], mass), ut, mass)
            proj = AF.cell_edge_average(np.einsum("fjd,fde,fje->fj", t_slot, Sig, t_slot), topo["face_edges"],
                                        len(edges))
            Trec = edge_reconstruction(proj, topo["face_edges"], t_slot, nrm)
            row["edge_recon"] = _rel(solve(AF.p1_stiffness(pts, faces, Trec)[0], mass), ut, mass)
            d0 = topo["d0"]
            row["diag_ref"] = _rel(solve((d0.T @ sp.diags(star1 * proj) @ d0).tocsr(), mass), ut, mass)
            w = AF.edge_weights(Kt, edges)
            row["diag_pospart"] = _rel(solve((d0.T @ sp.diags(np.maximum(w, 0.0)) @ d0).tocsr(), mass), ut, mass)
            res[rtag(r)] = row
        return res
    (_, n_tot, seed, _, _), = job_args(task, [idx], cfg, cfg["seed"], ratios, False)
    pts, tets, _ = tet_mesh(seed, idx, n_tot)
    n = len(pts)
    topo = AF.tet_topology(tets, n)
    n1, n2 = len(topo["edges"]), len(topo["faces"])
    t6 = cell_dirs(pts, tets, AF.TET_EDGES)
    Kc = _complex(pts, tets)
    d0, d1 = topo["d0"], topo["d1"]
    if task == "ACURL":
        F = acurl_fields(seed, idx, cfg)
        M1I = AF.assemble(AF.tet_whitney1_mass(pts, tets, topo, None), topo["tet_edges"], n1)
        j, _ = acurl_source(F, pts, topo, M1I)
        b = M1I @ j
        I = np.where(~topo["boundary_edges"])[0]
        star2 = _star_on(Kc, 2, topo["faces"], n)

        def solve(Sop):
            A = np.zeros(n1)
            A[I] = spla.splu(Sop[I][:, I].tocsc()).solve(b[I])
            return A
        for r in ratios:
            nu, _, _, _ = acurl_tensors(F, pts, tets, r, cfg)
            At = solve(acurl_operator(pts, tets, topo, nu, cfg["eps"], M1I)[0])
            Bt = d1 @ At
            row = {}
            for name, T in _oracle_tensors(nu, t6).items():
                A_o = solve(acurl_operator(pts, tets, topo, T, cfg["eps"], M1I)[0])
                row[name] = _rel(A_o, At)
                row[name + "_B"] = _rel(d1 @ A_o, Bt)
            loge, logf, _ = tet_proj_inputs(pts, tets, topo, nu)
            A_r = solve(acurl_operator(pts, tets, topo, edge_reconstruction(np.exp(loge), topo["tet_edges"], t6),
                                       cfg["eps"], M1I)[0])
            row["edge_recon"], row["edge_recon_B"] = _rel(A_r, At), _rel(d1 @ A_r, Bt)
            A_d = solve((d1.T @ sp.diags(star2 * np.exp(logf)) @ d1 + cfg["eps"] * M1I).tocsr())
            row["diag_ref"], row["diag_ref_B"] = _rel(A_d, At), _rel(d1 @ A_d, Bt)
            res[rtag(r)] = row
        return res
    # ADARCY
    F = adarcy_fields(seed, idx, cfg)
    f = eval_sources(F["src"], pts)
    vol = AF.simplex_measure(pts[tets])
    F_T = vol / 4.0 * eval_sources(F["src"], tet_quad_points(pts, tets).reshape(-1, 3)).reshape(len(tets), 4).sum(1)
    bnd = topo["boundary_nodes"]
    I = np.where(~bnd)[0]
    star1 = _star_on(Kc, 1, topo["edges"], n)
    star2 = _star_on(Kc, 2, topo["faces"], n)

    def solve_p(Kop, mass):
        p = np.zeros(n)
        p[I] = spla.splu(Kop[I][:, I].tocsc()).solve((mass * f)[I])
        return p

    def solve_J(M2):
        return mixed_solve(M2, topo["d2"], F_T)[0]
    for r in ratios:
        Kt, Kinv, _, _, _ = adarcy_tensors(F, pts, tets, r, cfg)
        Kp, mass, _ = AF.p1_stiffness(pts, tets, Kt)
        pt = solve_p(Kp, mass)
        M2t = AF.assemble(AF.tet_whitney2_mass(pts, tets, topo, Kinv), topo["tet_faces"], n2)
        Jt = solve_J(M2t)
        row = {}
        for name, T in _oracle_tensors(Kt, t6).items():
            row["p_" + name] = _rel(solve_p(AF.p1_stiffness(pts, tets, T)[0], mass), pt, mass)
        for name, T in _oracle_tensors(Kinv, t6).items():
            M2o = AF.assemble(AF.tet_whitney2_mass(pts, tets, topo, T), topo["tet_faces"], n2)
            row["J_" + name] = _rel(solve_J(M2o), Jt)
        loge, logf, _ = tet_proj_inputs(pts, tets, topo, Kt, inverse_normal=True)
        row["p_edge_recon"] = _rel(solve_p(AF.p1_stiffness(pts, tets, edge_reconstruction(
            np.exp(loge), topo["tet_edges"], t6))[0], mass), pt, mass)
        M2r = AF.assemble(AF.tet_whitney2_mass(pts, tets, topo, edge_reconstruction(
            np.exp(loge), topo["tet_edges"], t6, inverse=True)), topo["tet_faces"], n2)
        row["J_edge_recon"] = _rel(solve_J(M2r), Jt)
        row["p_diag_ref"] = _rel(solve_p((d0.T @ sp.diags(star1 * np.exp(loge)) @ d0).tocsr(), mass), pt, mass)
        w = AF.edge_weights(Kp, topo["edges"])
        row["p_diag_pospart"] = _rel(solve_p((d0.T @ sp.diags(np.maximum(w, 0.0)) @ d0).tocsr(), mass), pt, mass)
        row["J_diag_ref"] = _rel(solve_J(sp.diags(star2 * np.exp(logf)).tocsr()), Jt)
        res[rtag(r)] = row
    return res


def analyse(tasks, ratios, n_samples: int, workers: int, out_path: str, data_dir: str | None = None) -> dict:
    """Oracle errors on the first ``n_samples`` test-split samples of every task (see :func:`analyse_sample`) plus the
    dataset-wide representability statistics from the generated ``.json`` summaries in ``data_dir`` (if present)."""
    data_dir = data_dir or os.path.dirname(out_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    report = {}
    for task in tasks:
        cfg = dict(CONFIG[task])
        nt, nv = int(0.7 * cfg["N"]), int(0.15 * cfg["N"])
        ids = list(range(nt + nv, nt + nv + n_samples))
        t0 = time.time()
        rows = run_jobs(analyse_sample, [(task, i, cfg, ratios) for i in ids], workers)
        for r in ratios:
            keys = rows[0][rtag(r)].keys()
            agg = {k: dict(mean=float(np.mean([x[rtag(r)][k] for x in rows])),
                           std=float(np.std([x[rtag(r)][k] for x in rows]))) for k in keys}
            entry = dict(n_samples=len(rows), oracle_rel_err=agg, seconds=time.time() - t0)
            js = os.path.join(data_dir, f"{task}_{rtag(r)}.json")
            if os.path.exists(js):
                st = json.load(open(js))["stats"]
                entry["dataset"] = {k: st[k] for k in st if any(k.startswith(p) for p in (
                    "cone_frac", "inv_cone_frac", "neg_weight_frac", "smax", "inv_smax", "frac_smax", "inv_frac_smax",
                    "ratio_median", "frac_ratio"))}
            report[f"{task}_{rtag(r)}"] = entry
            print(f"[{task}_{rtag(r)}] " + "  ".join(f"{k}={v['mean']:.3f}" for k, v in agg.items()), flush=True)
    json.dump(report, open(out_path, "w"), indent=2)
    return report


# ======================================================================================================================
# convergence self-test (manufactured solutions, constant anisotropic tensors)
# ======================================================================================================================
def _const_tensor_3d(r: float) -> np.ndarray:
    tau = np.array([1.0, 2.0, 2.0]) / 3.0
    return (np.eye(3) + (r - 1) * np.outer(tau, tau)) / np.cbrt(r)


def selftest(sizes_2d=(500, 2000, 8000), sizes_3d=(600, 2400, 9600)) -> dict:
    """P1 (2-D), mixed RT0 and Nedelec curl-curl (3-D) errors against manufactured solutions decrease with h at (at
    least 3/4 of) the expected rates: nodal O(h^2) for P1, O(h) for the RT0 fluxes and the Nedelec edge values."""
    import torch
    # --- 2-D P1: u* = sin(pi x) sin(pi y), constant Sigma (ratio 10, misaligned)
    th = 0.4
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    Sg = R @ np.diag([np.sqrt(10.0), 1 / np.sqrt(10.0)]) @ R.T
    errs, nodes = [], []
    for n_tot in sizes_2d:
        pts, faces = gen_HP.draw_mesh_2d(0, 0, n_tot)
        nodes.append(len(pts))
        K, mass, _ = AF.p1_stiffness(pts, faces, np.broadcast_to(Sg, (len(faces), 2, 2)).copy())
        x, y = pts[:, 0], pts[:, 1]
        us = np.sin(np.pi * x) * np.sin(np.pi * y)
        f = np.pi ** 2 * (Sg[0, 0] + Sg[1, 1]) * us - 2 * np.pi ** 2 * Sg[0, 1] * np.cos(np.pi * x) * np.cos(np.pi * y)
        bnd = (pts.min(1) < 1e-12) | (pts.max(1) > 1 - 1e-12)
        I = np.where(~bnd)[0]
        u = np.zeros(len(pts))
        u[I] = spla.spsolve(K[I][:, I].tocsc(), (mass * f)[I])
        errs.append(float(np.abs(u - us).max()))
    print("selftest AHP P1 max nodal errors", ["%.2e" % e for e in errs])
    out = {"AHP": errs}
    assert errs[-1] < errs[0] * (nodes[-1] / nodes[0]) ** -0.75, errs          # O(h^2) = O(n^-1)
    # --- 3-D mixed Darcy: p* = sin sin sin, j* = -K grad p*, exact face fluxes (7-point rule on the faces)
    Kc = _const_tensor_3d(10.0)
    errs, nodes = [], []
    for n_tot in sizes_3d:
        pts, tets, _ = tet_mesh(1, 0, n_tot)
        nodes.append(len(pts))
        topo = AF.tet_topology(tets, len(pts))

        def grad_p(X):
            s, c = np.sin(np.pi * X), np.cos(np.pi * X)
            return np.pi * np.stack([c[:, 0] * s[:, 1] * s[:, 2], s[:, 0] * c[:, 1] * s[:, 2],
                                     s[:, 0] * s[:, 1] * c[:, 2]], 1)

        def f_src(X):
            s, c = np.sin(np.pi * X), np.cos(np.pi * X)
            H = -np.pi ** 2 * np.stack([
                np.stack([s[:, 0] * s[:, 1] * s[:, 2], -c[:, 0] * c[:, 1] * s[:, 2], -c[:, 0] * s[:, 1] * c[:, 2]], 1),
                np.stack([-c[:, 0] * c[:, 1] * s[:, 2], s[:, 0] * s[:, 1] * s[:, 2], -s[:, 0] * c[:, 1] * c[:, 2]], 1),
                np.stack([-c[:, 0] * s[:, 1] * c[:, 2], -s[:, 0] * c[:, 1] * c[:, 2], s[:, 0] * s[:, 1] * s[:, 2]],
                         1)], 1)
            return -np.einsum("ij,nij->n", Kc, H)
        vol = AF.simplex_measure(pts[tets])
        F_T = vol / 4.0 * f_src(tet_quad_points(pts, tets).reshape(-1, 3)).reshape(len(tets), 4).sum(1)
        Kinv = np.broadcast_to(np.linalg.inv(Kc), (len(tets), 3, 3)).copy()
        J, _, res, cons = mixed_darcy(pts, tets, topo, Kinv, F_T)
        fc = topo["faces"]
        P = pts[fc]
        nA = 0.5 * np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])        # area vector of the canonical orientation
        lam = np.array([[1 / 3, 1 / 3, 1 / 3], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
        wts = np.array([-27 / 48, 25 / 48, 25 / 48, 25 / 48])
        Xq = np.einsum("qi,fid->fqd", lam, P)
        jq = -np.einsum("de,fqe->fqd", Kc, grad_p(Xq.reshape(-1, 3)).reshape(Xq.shape))
        Jex = np.einsum("q,fqd,fd->f", wts, jq, nA)
        errs.append(float(np.linalg.norm(J - Jex) / np.linalg.norm(Jex)))
        assert res < RES_TOL and cons < 1e-10, (res, cons)
    print("selftest ADARCY mixed relative flux errors", ["%.2e" % e for e in errs])
    out["ADARCY"] = errs
    assert errs[-1] < errs[0] * (nodes[-1] / nodes[0]) ** -0.25 and errs[-1] < 0.3, errs    # O(h) = O(n^-1/3)
    # --- 3-D curl-curl: A* tangential-free on the cube, J = curl(nu curl A*) + eps A* by autograd
    nu_c = _const_tensor_3d(10.0)
    eps = 10.0
    nu_t = torch.tensor(nu_c)

    def A_star(X):
        s = torch.sin(np.pi * X)
        return torch.stack([s[:, 1] * s[:, 2], s[:, 2] * s[:, 0], s[:, 0] * s[:, 1]], 1)

    def curl(Ffn, X):
        X = X.requires_grad_(True)
        V = Ffn(X)
        g = [torch.autograd.grad(V[:, i].sum(), X, create_graph=True)[0] for i in range(3)]
        return torch.stack([g[2][:, 1] - g[1][:, 2], g[0][:, 2] - g[2][:, 0], g[1][:, 0] - g[0][:, 1]], 1)

    def J_star(X):
        Xt = torch.tensor(X, dtype=torch.float64)
        Jv = curl(lambda Y: curl(A_star, Y) @ nu_t.T, Xt) + eps * A_star(Xt)
        return Jv.detach().numpy()
    errs, nodes = [], []
    gl = np.array([0.5 - np.sqrt(15) / 10, 0.5, 0.5 + np.sqrt(15) / 10])
    gw = np.array([5, 8, 5]) / 18.0
    for n_tot in sizes_3d:
        pts, tets, _ = tet_mesh(2, 0, n_tot)
        nodes.append(len(pts))
        topo = AF.tet_topology(tets, len(pts))
        n1 = len(topo["edges"])
        nu = np.broadcast_to(nu_c, (len(tets), 3, 3)).copy()
        Sop, _ = acurl_operator(pts, tets, topo, nu, eps)
        # load b_e = int J . w_e (4-point rule per tet)
        V = pts[tets]
        g = AF.bary_grads(V)
        vol = AF.simplex_measure(V)
        W = AF.whitney1_values(AF.TET_Q, g, AF.TET_EDGES, topo["tet_edge_signs"])      # (n3, Q, 6, 3)
        Jq = J_star(tet_quad_points(pts, tets).reshape(-1, 3)).reshape(len(tets), 4, 3)
        bl = np.einsum("tq,tqjd,tqd->tj", np.repeat((vol / 4)[:, None], 4, 1), W, Jq)
        b = np.zeros(n1)
        np.add.at(b, topo["tet_edges"].ravel(), bl.ravel())
        I = np.where(~topo["boundary_edges"])[0]
        A = np.zeros(n1)
        A[I], res = solve_spd(Sop[I][:, I], b[I])
        E0, E1 = pts[topo["edges"][:, 0]], pts[topo["edges"][:, 1]]
        Aex = np.zeros(n1)
        for q, wq in zip(gl, gw):
            Xq = E0 + q * (E1 - E0)
            Aex += wq * (A_star(torch.tensor(Xq)).numpy() * (E1 - E0)).sum(1)
        errs.append(float(np.linalg.norm(A - Aex) / np.linalg.norm(Aex)))
        assert res < RES_TOL, res
    print("selftest ACURL Nedelec relative edge errors", ["%.2e" % e for e in errs])
    out["ACURL"] = errs
    assert errs[-1] < errs[0] * (nodes[-1] / nodes[0]) ** -0.25 and errs[-1] < 0.3, errs
    print("selftest OK")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--ratios", nargs="+", type=float, default=list(RATIOS))
    ap.add_argument("--n", type=int, default=None, help="samples per set (default: CONFIG[task]['N'])")
    ap.add_argument("--n-fine", type=int, default=None, help="4x-finer test samples (default: CONFIG[task])")
    ap.add_argument("--fine-only", action="store_true", help="only (re)generate the *_fine sets")
    ap.add_argument("--seed", type=int, default=None, help="override the task seed")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "datasets", "v2"))
    ap.add_argument("--analyse", type=int, default=0, metavar="K",
                    help="only compute the representability oracles on K test samples per task (no data written "
                         "except <out-dir>/ANISO_representability.json)")
    ap.add_argument("--analyse-out", default=None,
                    help="report path of --analyse (default <out-dir>/ANISO_representability.json)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        selftest()
        return None
    if a.analyse:
        return analyse(a.tasks, a.ratios, a.analyse, a.workers,
                       a.analyse_out or os.path.join(a.out_dir, "ANISO_representability.json"), data_dir=a.out_dir)
    out = {}
    for task in a.tasks:
        out.update(write_task(task, a.ratios, a.workers, a.out_dir, n=a.n, n_fine=a.n_fine, seed=a.seed,
                              fine_only=a.fine_only))
    return out


if __name__ == "__main__":
    main()
