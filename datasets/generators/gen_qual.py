"""Mesh-quality shift test sets for HP models (tasks HP_qual_graded / HP_qual_sliver; test only).

The physical instances are those of the first ``n_base`` *test* samples of ``HP_k100`` (same seed, same sample ids,
same conductivity / source fields from ``gen_HP.draw_fields`` with kappa = 100, same node counts); only the mesh
changes.  Every base sample is re-meshed at every quality level, so accuracy-vs-quality curves compare identical
physical problems:

  HP_qual_graded (levels):
    base              the original HP mesh (``gen_HP.draw_mesh_2d``), re-solved: reproduces the HP_k100 sample
    corner_r{4,16,64} HP nodes mapped by x -> (exp(b x) - 1)/(exp(b) - 1) per coordinate, b = ln(ratio): mesh
                      refined towards the corner (0,0) (spacing ratio ~ ``ratio`` between the corners), re-Delaunay
    circle_a{10,30,100} interior nodes resampled with density 1 + a exp(-(|x - c| - 0.3)^2 / (2 * 0.04^2)),
                      c = (0.5, 0.5) (refined around a circle; spacing ratio ~ sqrt(1 + a)), boundary nodes as HP
  HP_qual_sliver (levels):
    base              as above
    a{2,3,4,6,8}      the HP nodes re-triangulated by the Delaunay triangulation of the anisotropically scaled
                      point cloud  x -> R(phi) diag(a, 1) R(phi)^T x  (random phi): same nodes, sliver connectivity
                      (triangles elongated by ~a perpendicular to phi).  Per-sample max aspect ratio R/(2r) ~ 1e2 (a=2)
                      to ~1.5e3 (a=8), 99th percentile 17 -> 210, share of triangles with aspect > 10: 3 % -> 50 %
                      (the HP base meshes themselves reach aspect ~50 and angles of ~1 deg: Delaunay of uniform
                      random points).  Larger stretches (a >= 16) make the P1 solution itself meaningless (max-angle
                      condition): its error w.r.t. the reference below grows from 0.6 % (base) to ~18 % (a = 8).
Physics and solver: identical to ``gen_HP.solve_sample`` (isotropic, kappa = 100): P1 FEM of -div(sigma grad u) = f,
u = 0 on the boundary, element-wise sigma at centroids, lumped-mass right-hand side (``solve_on_mesh``; a test
checks equality with ``gen_HP.solve_sample`` on the HP mesh).  Stored keys = HP keys (pos, faces, f, u, bnd,
sigma_edge, logsigma_face, flux, flux_fem) + ``u_ref``: the P1 solution on the 4x-finer HP reference mesh (the mesh of
``HP_k100_fine``) interpolated linearly to the nodes (the discretisation error of the quality-shifted mesh is
``|u - u_ref|``, stored per sample) + per-sample ``level`` codes and mesh-quality statistics (angles, aspect ratios
R/(2r), edge-length ratio) in ``sample_stats``.

Usage:
  python3 -u datasets/generators/gen_qual.py [--n-base 100] [--workers 32]
  -> datasets/v2/HP_qual_graded.pt, datasets/v2/HP_qual_sliver.pt (+ .json summaries)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import scipy.sparse.linalg as spla
from scipy.spatial import Delaunay

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_common import mesh_edges, p1_stiffness_2d  # noqa: E402
from gen_HP import BND_TOL, MIN_AREA, draw_fields, draw_mesh_2d, log_sigma, pack, source  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRADED_LEVELS = ("base", "corner_r4", "corner_r16", "corner_r64", "circle_a10", "circle_a30", "circle_a100")
SLIVER_LEVELS = ("base", "a2", "a3", "a4", "a6", "a8")
DEGEN_REL = 1e-8        # a triangle with 2*area < DEGEN_REL * lmax^2 is rejected (rhmp drops < 1e-10)


# ======================================================================================================================
# solver (identical to gen_HP.solve_sample for the isotropic case, on a given mesh)
# ======================================================================================================================
def solve_on_mesh(F: dict, pts: np.ndarray, faces: np.ndarray, kappa: float) -> dict:
    """P1 FEM solve of the HP problem with fields ``F`` on the mesh ``(pts, faces)`` (CCW triangles).

    Returns the HP sample dict (``pos, faces, f, u, bnd, sigma_edge, logsigma_face, flux, flux_fem``) with float64
    arrays and ``residual`` / ``conservation`` diagnostics.
    """
    n = len(pts)
    cent = pts[faces].mean(1)
    ls_T = log_sigma(F, cent, kappa)
    K, mass, _ = p1_stiffness_2d(pts, faces, np.exp(ls_T), None)
    f = source(F, pts)
    b = mass * f
    bnd = (pts[:, 0] < BND_TOL) | (pts[:, 0] > 1 - BND_TOL) | (pts[:, 1] < BND_TOL) | (pts[:, 1] > 1 - BND_TOL)
    I = np.where(~bnd)[0]
    KII = K[I][:, I].tocsc()
    u = np.zeros(n)
    u[I] = spla.spsolve(KII, b[I])
    res = float(np.linalg.norm(KII @ u[I] - b[I]) / max(np.linalg.norm(b[I]), 1e-30))
    edges = mesh_edges(faces, n)
    sig_e = np.exp(log_sigma(F, pts[edges].mean(1), kappa))
    du = u[edges[:, 1]] - u[edges[:, 0]]
    w = -np.asarray(K[edges[:, 0], edges[:, 1]]).ravel()
    j_fem = -w * du
    div = np.zeros(n)
    np.add.at(div, edges[:, 0], -j_fem)
    np.add.at(div, edges[:, 1], j_fem)
    cons = float(np.abs(div[I] + b[I]).max() / max(np.abs(b[I]).max(), 1e-30))
    return dict(pos=pts, faces=faces, f=f, u=u, bnd=bnd, sigma_edge=sig_e, logsigma_face=ls_T, flux=-sig_e * du,
                flux_fem=j_fem, residual=res, conservation=cons, edges=edges)


# ======================================================================================================================
# meshes
# ======================================================================================================================
def triangulate(tri_pts: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Delaunay connectivity of ``tri_pts`` used with the positions ``pts`` (same order); zero-area triangles (collinear
    boundary points) dropped, CCW orientation w.r.t. ``pts``.  Raises if a node is lost or a triangle is degenerate."""
    tri = Delaunay(tri_pts)
    if len(tri.coplanar):
        tri = Delaunay(tri_pts, qhull_options="QJ Qbb Qc Qz")
    faces = tri.simplices.astype(np.int64)
    P = pts[faces]
    sa = 0.5 * ((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) -
                (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0]))
    lmax = np.max(np.stack([np.linalg.norm(P[:, a] - P[:, b], axis=1) for a, b in ((0, 1), (1, 2), (2, 0))], 1), 1)
    keep = np.abs(sa) > MIN_AREA * 1e-3
    faces, sa, lmax = faces[keep], sa[keep], lmax[keep]
    neg = sa < 0
    faces[neg] = faces[neg][:, [0, 2, 1]]
    if len(np.unique(faces)) != len(pts):
        raise RuntimeError("triangulation lost nodes")
    if abs(np.abs(sa).sum() - 1.0) > 1e-9:
        raise RuntimeError(f"triangulation does not cover the unit square (area {np.abs(sa).sum()})")
    if (2 * np.abs(sa) < DEGEN_REL * lmax ** 2).any():
        raise RuntimeError("near-degenerate triangle")
    return faces


def corner_graded(pts: np.ndarray, ratio: float) -> np.ndarray:
    """Map ``[0,1]^2 -> [0,1]^2``, ``x -> (exp(b x) - 1)/(exp(b) - 1)`` per coordinate with ``b = ln ratio``."""
    b = np.log(ratio)
    return np.expm1(b * pts) / np.expm1(b)


def circle_graded_points(rng: np.random.Generator, pts_hp: np.ndarray, n_bnd: int, amp: float,
                         center=(0.5, 0.5), R0: float = 0.3, w: float = 0.04) -> np.ndarray:
    """Keep the HP boundary nodes (first ``n_bnd`` rows) and resample the interior nodes with the density
    ``1 + amp exp(-(|x - c| - R0)^2 / (2 w^2))`` (rejection sampling, same margin to the boundary as HP)."""
    n_int = len(pts_hp) - n_bnd
    m = max(4, int(round(np.sqrt(len(pts_hp)))))
    delta = 0.3 / m
    c = np.asarray(center)
    out = []
    while sum(len(o) for o in out) < n_int:
        x = rng.uniform(delta, 1.0 - delta, (4 * n_int, 2))
        dens = 1.0 + amp * np.exp(-(np.linalg.norm(x - c, axis=1) - R0) ** 2 / (2 * w * w))
        out.append(x[rng.random(len(x)) * (1.0 + amp) < dens])
    return np.concatenate([pts_hp[:n_bnd], np.concatenate(out, 0)[:n_int]], 0)


def sliver_faces(pts: np.ndarray, a: float, phi: float) -> np.ndarray:
    """Delaunay connectivity of the point cloud stretched by ``a`` along the direction ``phi``."""
    c, s = np.cos(phi), np.sin(phi)
    Rm = np.array([[c, -s], [s, c]])
    T = Rm @ np.diag([a, 1.0]) @ Rm.T
    return triangulate(pts @ T.T, pts)


def quality(pts: np.ndarray, faces: np.ndarray) -> dict:
    """Angles, aspect ratios ``R/(2r)`` and the edge-length ratio of a planar triangle mesh."""
    P = np.concatenate([pts, np.zeros((len(pts), 1))], 1)[faces]
    l = np.stack([np.linalg.norm(P[:, 1] - P[:, 2], axis=1), np.linalg.norm(P[:, 2] - P[:, 0], axis=1),
                  np.linalg.norm(P[:, 0] - P[:, 1], axis=1)], 1)
    area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    s = 0.5 * l.sum(1)
    aspect = l.prod(1) * s / np.maximum(8 * area ** 2, 1e-300)
    cosang = np.stack([(l[:, 1] ** 2 + l[:, 2] ** 2 - l[:, 0] ** 2) / (2 * l[:, 1] * l[:, 2]),
                       (l[:, 0] ** 2 + l[:, 2] ** 2 - l[:, 1] ** 2) / (2 * l[:, 0] * l[:, 2]),
                       (l[:, 0] ** 2 + l[:, 1] ** 2 - l[:, 2] ** 2) / (2 * l[:, 0] * l[:, 1])], 1)
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    e = mesh_edges(faces, len(pts))
    el = np.linalg.norm(pts[e[:, 1]] - pts[e[:, 0]], axis=1)
    return dict(min_angle=float(ang.min()), max_angle=float(ang.max()), aspect_max=float(aspect.max()),
                aspect_p50=float(np.median(aspect)), aspect_p90=float(np.percentile(aspect, 90)),
                aspect_p99=float(np.percentile(aspect, 99)), frac_aspect_gt10=float((aspect > 10).mean()),
                frac_aspect_gt100=float((aspect > 100).mean()), frac_obtuse=float((ang.max(1) > 90).mean()),
                edge_ratio=float(el.max() / el.min()), edge_median=float(np.median(el)))


def level_mesh(level: str, pts_hp: np.ndarray, faces_hp: np.ndarray, rng: np.random.Generator):
    """Mesh of one quality level from the HP base mesh.  Returns ``(pts, faces, info)``."""
    if level == "base":
        return pts_hp, faces_hp, {}
    if level.startswith("corner_r"):
        ratio = float(level[len("corner_r"):])
        p = corner_graded(pts_hp, ratio)
        return p, triangulate(p, p), dict(ratio=ratio)
    if level.startswith("circle_a"):
        amp = float(level[len("circle_a"):])
        n_bnd = 4 * max(4, int(round(np.sqrt(len(pts_hp)))))      # corners + 4 (m-1) side nodes (gen_HP layout)
        for _ in range(10):
            p = circle_graded_points(rng, pts_hp, n_bnd, amp)
            try:
                return p, triangulate(p, p), dict(amp=amp)
            except RuntimeError:
                continue
        raise RuntimeError(f"could not triangulate level {level}")
    if level.startswith("a"):
        a = float(level[1:])
        for _ in range(20):
            phi = float(rng.uniform(0, np.pi))
            try:
                return pts_hp, sliver_faces(pts_hp, a, phi), dict(stretch=a, phi=phi)
            except RuntimeError:
                continue
        raise RuntimeError(f"could not build a non-degenerate sliver mesh at level {level}")
    raise ValueError(f"unknown level {level!r}")


# ======================================================================================================================
# one base sample: all levels
# ======================================================================================================================
def interpolate_p1(pts_f: np.ndarray, u_f: np.ndarray, q: np.ndarray, bnd_q: np.ndarray) -> np.ndarray:
    """Linear interpolation of the P1 field ``u_f`` (on the Delaunay mesh of ``pts_f``) at ``q``; boundary -> 0."""
    tri = Delaunay(pts_f)
    out = np.zeros(len(q))
    I = np.where(~bnd_q)[0]
    s = tri.find_simplex(q[I], tol=1e-12)
    miss = s < 0
    if miss.any():                                   # not expected for interior nodes: nearest-vertex fallback
        d = ((q[I][miss][:, None, :] - pts_f[None]) ** 2).sum(-1)
        out[I[miss]] = u_f[d.argmin(1)]
    ok = ~miss
    T = tri.transform[s[ok]]
    bc = np.einsum("mij,mj->mi", T[:, :2], q[I][ok] - T[:, 2])
    bary = np.concatenate([bc, 1.0 - bc.sum(1, keepdims=True)], 1)
    out[I[ok]] = (bary * u_f[tri.simplices[s[ok]]]).sum(1)
    return out


def make_base(args) -> list[dict]:
    """All quality levels of HP sample ``idx`` (fields of HP seed ``seed``, ``n_tot`` nodes)."""
    idx, n_tot, seed, kappa, levels, fine_factor, qseed = args
    t0 = time.time()
    F = draw_fields(seed, idx, 2, False)
    pts_hp, faces_hp = draw_mesh_2d(seed, idx, n_tot)
    pts_f, faces_f = draw_mesh_2d(seed, idx, fine_factor * n_tot)
    ref = solve_on_mesh(F, pts_f, faces_f, kappa)
    rng = np.random.default_rng([qseed, idx])
    out = []
    base_err = None
    for lev in levels:
        pts, faces, info = level_mesh(lev, pts_hp, faces_hp, rng)
        sol = solve_on_mesh(F, pts, faces, kappa)
        u_ref = interpolate_p1(pts_f, ref["u"], pts, sol["bnd"])
        err = float(np.linalg.norm(sol["u"] - u_ref) / max(np.linalg.norm(u_ref), 1e-30))
        if lev == "base":
            base_err = err
        q = quality(pts, faces)
        st = dict(idx=int(idx), level=lev, n0=len(pts), n1=len(sol["edges"]), n2=len(faces), residual=sol["residual"],
                  conservation=sol["conservation"], rel_err_vs_ref=err, base_rel_err_vs_ref=base_err, **info, **q)
        out.append(dict(pos=pts.astype(np.float32), faces=faces.astype(np.int32), f=sol["f"].astype(np.float32),
                        u=sol["u"].astype(np.float32), bnd=sol["bnd"], sigma_edge=sol["sigma_edge"].astype(np.float32),
                        logsigma_face=sol["logsigma_face"].astype(np.float32), flux=sol["flux"].astype(np.float32),
                        flux_fem=sol["flux_fem"].astype(np.float32), u_ref=u_ref.astype(np.float32), stats=st))
    for s in out:
        s["stats"]["t_base"] = time.time() - t0
    return out


def generate(jobs, workers):
    if workers <= 1:
        return [s for j in jobs for s in make_base(j)]
    with Pool(workers) as pool:
        return [s for group in pool.imap(make_base, jobs, chunksize=1) for s in group]


def write_set(name: str, levels, a, out_dir: str) -> dict:
    import torch
    n_tots = np.random.default_rng([a.hp_seed, 4]).integers(a.hp_nmin, a.hp_nmax + 1, a.hp_n)
    nt, nv = int(0.7 * a.hp_n), int(0.15 * a.hp_n)
    ids = np.arange(nt + nv, min(nt + nv + a.n_base, a.hp_n))
    jobs = [(int(i), int(n_tots[i]), a.hp_seed, a.kappa, tuple(levels), a.fine_factor, a.seed) for i in ids]
    t0 = time.time()
    samples = generate(jobs, a.workers)
    t_gen = time.time() - t0
    # group by level (level-major order: all base samples, then level 1, ...) for readable subsets
    order = sorted(range(len(samples)), key=lambda i: (levels.index(samples[i]["stats"]["level"]), i))
    samples = [samples[i] for i in order]
    meta = dict(task="HP_qual", set=name, levels=list(levels), N=len(samples), n_base=len(ids),
                hp_ids=ids.tolist(), hp_seed=a.hp_seed, kappa=a.kappa, seed=a.seed, gen_seconds=t_gen,
                fine_factor=a.fine_factor, generator="datasets/generators/gen_qual.py",
                edge_order="unique src<dst pairs of the triangles, lexicographic",
                pde="-div(sigma grad u) = f, u=0 on the boundary of [0,1]^2, P1 FEM (as gen_HP.py, kappa=100)")
    blob = pack(samples, meta)
    blob["level"] = torch.tensor([levels.index(s["stats"]["level"]) for s in samples], dtype=torch.int64)
    path = os.path.join(out_dir, name + ".pt")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(blob, path)
    st = [s["stats"] for s in samples]
    per_level = {}
    for lev in levels:
        ss = [s for s in st if s["level"] == lev]
        per_level[lev] = dict(N=len(ss), n0_mean=float(np.mean([s["n0"] for s in ss])),
                              min_angle=float(min(s["min_angle"] for s in ss)),
                              max_angle=float(max(s["max_angle"] for s in ss)),
                              aspect_p50=float(np.median([s["aspect_p50"] for s in ss])),
                              aspect_p99_mean=float(np.mean([s["aspect_p99"] for s in ss])),
                              aspect_max=float(max(s["aspect_max"] for s in ss)),
                              frac_aspect_gt10=float(np.mean([s["frac_aspect_gt10"] for s in ss])),
                              edge_ratio_mean=float(np.mean([s["edge_ratio"] for s in ss])),
                              rel_err_vs_ref_mean=float(np.mean([s["rel_err_vs_ref"] for s in ss])),
                              max_residual=float(max(s["residual"] for s in ss)))
    summary = dict(meta, hp_ids=None, file=path, file_MB=os.path.getsize(path) / 1e6, per_level=per_level,
                   max_residual=float(max(s["residual"] for s in st)),
                   max_conservation=float(max(s["conservation"] for s in st)))
    # consistency with the stored HP_k100 set (base level == HP test sample), when available
    hp_path = os.path.join(out_dir, "HP_k100.pt")
    if os.path.exists(hp_path) and "base" in levels:
        hp = torch.load(hp_path, map_location="cpu", weights_only=False)
        diffs = []
        for s in [s for s in samples if s["stats"]["level"] == "base"][:20]:
            i = s["stats"]["idx"]
            p0, p1 = int(hp["ptr0"][i]), int(hp["ptr0"][i + 1])
            if p1 - p0 != len(s["u"]):
                diffs.append(float("inf"))
                continue
            diffs.append(float(np.abs(hp["u"][p0:p1].numpy() - s["u"]).max()))
        summary["base_vs_HP_k100_max_abs_u_diff"] = max(diffs) if diffs else None
    json.dump(summary, open(path.replace(".pt", ".json"), "w"), indent=2)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-base", type=int, default=100, help="HP test samples re-meshed at every level")
    ap.add_argument("--kappa", type=float, default=100.0)
    ap.add_argument("--hp-seed", type=int, default=2026, help="seed of the HP_k100 set (gen_HP.py default)")
    ap.add_argument("--hp-n", type=int, default=5000, help="size of the HP_k100 set (defines its test ids)")
    ap.add_argument("--hp-nmin", type=int, default=1000)
    ap.add_argument("--hp-nmax", type=int, default=2000)
    ap.add_argument("--fine-factor", type=int, default=4, help="node factor of the reference mesh")
    ap.add_argument("--seed", type=int, default=2050, help="seed of the re-meshing randomness")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--sets", default="graded,sliver")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "datasets", "v2"))
    a = ap.parse_args(argv)
    out = {}
    for s in a.sets.split(","):
        levels = {"graded": GRADED_LEVELS, "sliver": SLIVER_LEVELS}[s]
        summ = write_set(f"HP_qual_{s}", levels, a, a.out_dir)
        print(json.dumps(summ, indent=2), flush=True)
        out[s] = summ
    return out


if __name__ == "__main__":
    main()
