"""3-D variable-conductivity Poisson on random tetrahedral Delaunay meshes of the unit cube (tasks TET / TETflux).

Same recipe as ``gen_HP.py`` in 3-D:
  * mesh     : random Delaunay tetrahedralisation of [0,1]^3 with n0 ~ U{nmin..nmax} nodes: 8 corners, jittered
               points on the 12 cube edges and on the 6 faces (m per side), the rest uniform in the interior;
               zero-volume tetrahedra (possible with coplanar boundary points) are dropped, tets oriented positively;
  * sigma(x) : log-normal conductivity, contrast kappa (GRF rescaled to [-1,1] on a fixed 25^3 reference grid);
  * f(x)     : 2-5 Gaussian sources;
  * solve    : -div(sigma grad u) = f, u = 0 on the cube boundary, P1 FEM (element-wise sigma at tet centroids),
               lumped mass, sparse direct solve.
Stored per sample (canonical edges = unique src<dst pairs, lexicographic; faces = unique sorted vertex triples,
lexicographic):
  pos (n0,3) f32, tets (n3,4) i32, f (n0), u (n0), bnd (n0) bool,
  sigma_edge (n1)        sigma at edge midpoints (even edge input)
  logsigma_face (n2)     log sigma at triangle centroids (even face input)
  logsigma_tet (n3)      log sigma at tet centroids (even degree-3 input; the value used by the FEM)
  flux (n1)              -sigma_e (u_dst - u_src)   (TETflux degree-1 target)
  flux_fem (n1)          -w_e (u_dst - u_src), w_e = -K_{src,dst}: conservative flux through the dual faces of the
                         edges, d0^T flux_fem = -M f exactly at interior nodes (TETfluxfem)
Packed with offsets ptr0..ptr3.  ``*_fine.pt``: first ``n_fine`` test-split samples re-solved on meshes with
``fine_factor`` x the nodes (same fields).

Usage:  python3 -u datasets/generators/gen_TET.py --kappa 100 [--n 3000] [--n-fine 200] [--workers 32]
        python3 -u datasets/generators/gen_TET.py --selftest
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
from gen_common import mesh_edges, mesh_faces_of_tets, p1_stiffness_3d  # noqa: E402
from gen_HP import draw_fields, log_sigma, source  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF_GRID_3D = 25
BND_TOL = 1e-12
MIN_VOL_REL = 1e-6    # tets with volume below MIN_VOL_REL * mean volume are dropped (flat boundary slivers)


def draw_mesh_3d(seed, idx, n_tot, max_tries=20):
    """Random Delaunay tetrahedralisation of the unit cube with about ``n_tot`` nodes.

    Returns:
        pts ``(n0, 3)`` float64, tets ``(n3, 4)`` int64 (positively oriented), n_dropped (flat tets removed).
    """
    rng = np.random.default_rng([seed, 2, idx, n_tot])
    m = max(3, int(round(0.95 * n_tot ** (1.0 / 3.0))))
    for _ in range(max_tries):
        P = [np.array([[i, j, k] for i in (0.0, 1.0) for j in (0.0, 1.0) for k in (0.0, 1.0)])]
        # 12 cube edges: axis a varies, the two other coordinates in {0,1}
        for a in range(3):
            for u in (0.0, 1.0):
                for v in (0.0, 1.0):
                    t = (np.arange(1, m) + rng.uniform(-0.25, 0.25, m - 1)) / m
                    X = np.empty((m - 1, 3))
                    X[:, a] = t
                    o = [b for b in range(3) if b != a]
                    X[:, o[0]], X[:, o[1]] = u, v
                    P.append(X)
        # 6 faces: coordinate a fixed to 0/1, jittered (m-1)^2 grid on the face
        g = np.arange(1, m)
        for a in range(3):
            for w in (0.0, 1.0):
                s, t = np.meshgrid(g, g, indexing="ij")
                s = (s.ravel() + rng.uniform(-0.3, 0.3, s.size)) / m
                t = (t.ravel() + rng.uniform(-0.3, 0.3, t.size)) / m
                X = np.empty((s.size, 3))
                o = [b for b in range(3) if b != a]
                X[:, a], X[:, o[0]], X[:, o[1]] = w, s, t
                P.append(X)
        n_bnd = sum(len(x) for x in P)
        n_int = max(n_tot - n_bnd, 1)
        delta = 0.3 / m
        P.append(rng.uniform(delta, 1.0 - delta, (n_int, 3)))
        pts = np.concatenate(P, 0)
        tri = Delaunay(pts)
        if len(tri.coplanar):
            continue
        tets = tri.simplices.astype(np.int64)
        X = pts[tets]
        det = np.einsum("mi,mi->m", np.cross(X[:, 1] - X[:, 0], X[:, 2] - X[:, 0]), X[:, 3] - X[:, 0])
        vol = np.abs(det) / 6.0
        keep = vol > MIN_VOL_REL * vol.mean()
        n_drop = int((~keep).sum())
        tets, det = tets[keep], det[keep]
        neg = det < 0
        tets[neg] = tets[neg][:, [0, 2, 1, 3]]
        used = np.unique(tets)
        if len(used) != len(pts):            # drop vertices that only belonged to flat tets
            remap = -np.ones(len(pts), dtype=np.int64)
            remap[used] = np.arange(len(used))
            pts, tets = pts[used], remap[tets]
        return pts, tets, n_drop
    raise RuntimeError(f"could not build a valid tet mesh for sample {idx}")


def solve_sample(args):
    idx, n_tot, seed, kappa = args
    F = draw_fields(seed, idx, 3, False, ref_grid=REF_GRID_3D)
    pts, tets, n_drop = draw_mesh_3d(seed, idx, n_tot)
    n = len(pts)
    ls_T = log_sigma(F, pts[tets].mean(1), kappa)
    K, mass, vol = p1_stiffness_3d(pts, tets, np.exp(ls_T))
    f = source(F, pts)
    b = mass * f
    bnd = (pts.min(1) < BND_TOL) | (pts.max(1) > 1 - BND_TOL)
    I = np.where(~bnd)[0]
    KII = K[I][:, I].tocsc()
    u = np.zeros(n)
    u[I] = spla.spsolve(KII, b[I])
    res = float(np.linalg.norm(KII @ u[I] - b[I]) / max(np.linalg.norm(b[I]), 1e-30))
    edges = mesh_edges(tets, n)
    faces = mesh_faces_of_tets(tets, n)
    ls_e = log_sigma(F, pts[edges].mean(1), kappa)
    ls_f = log_sigma(F, pts[faces].mean(1), kappa)
    sig_e = np.exp(ls_e)
    du = u[edges[:, 1]] - u[edges[:, 0]]
    w = -np.asarray(K[edges[:, 0], edges[:, 1]]).ravel()
    j_fem = -w * du
    div = np.zeros(n)
    np.add.at(div, edges[:, 0], -j_fem)
    np.add.at(div, edges[:, 1], j_fem)
    cons = float(np.abs(div[I] + b[I]).max() / max(np.abs(b[I]).max(), 1e-30))
    out = dict(pos=pts.astype(np.float32), tets=tets.astype(np.int32), f=f.astype(np.float32),
               u=u.astype(np.float32), bnd=bnd, sigma_edge=sig_e.astype(np.float32),
               logsigma_face=ls_f.astype(np.float32), logsigma_tet=ls_T.astype(np.float32),
               flux=(-sig_e * du).astype(np.float32), flux_fem=j_fem.astype(np.float32))
    out["stats"] = dict(idx=idx, n0=n, n1=len(edges), n2=len(faces), n3=len(tets), residual=res,
                        conservation=cons, dropped_tets=n_drop, min_vol=float(vol.min()), ell=F["ell"])
    return out


def pack(samples, meta):
    import torch
    cnt = {k: np.array([s["stats"][k] for s in samples]) for k in ("n0", "n1", "n2", "n3")}
    ptr = lambda n: torch.from_numpy(np.concatenate([[0], np.cumsum(n)]).astype(np.int64))
    out = {f"ptr{i}": ptr(cnt[f"n{i}"]) for i in range(4)}
    out["meta"] = meta
    for key in samples[0]:
        if key != "stats":
            out[key] = torch.from_numpy(np.concatenate([s[key] for s in samples], 0))
    out["sample_stats"] = [s["stats"] for s in samples]
    return out


def generate(ids, n_tots, seed, kappa, workers):
    args = [(int(i), int(n), seed, kappa) for i, n in zip(ids, n_tots)]
    with Pool(workers) as pool:
        return list(pool.imap(solve_sample, args, chunksize=4))


def selftest():
    """P1 FEM convergence on u* = sin(pi x) sin(pi y) sin(pi z), sigma = 1."""
    errs = []
    for n_tot in (1000, 4000, 16000):
        pts, tets, _ = draw_mesh_3d(0, 0, n_tot)
        K, mass, _ = p1_stiffness_3d(pts, tets, np.ones(len(tets)))
        us = np.prod(np.sin(np.pi * pts), axis=1)
        b = mass * 3 * np.pi ** 2 * us
        bnd = (pts.min(1) < BND_TOL) | (pts.max(1) > 1 - BND_TOL)
        I = np.where(~bnd)[0]
        u = np.zeros(len(pts))
        u[I] = spla.spsolve(K[I][:, I].tocsc(), b[I])
        errs.append(float(np.abs(u - us).max()))
        print(f"selftest n0={len(pts):6d} n3={len(tets):6d}: max nodal error {errs[-1]:.3e}")
    assert errs[-1] < errs[0] / 3 and errs[-1] < 3e-2, errs
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kappa", type=float, default=100.0)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--n-fine", type=int, default=200)
    ap.add_argument("--fine-factor", type=int, default=4)
    ap.add_argument("--nmin", type=int, default=2000)
    ap.add_argument("--nmax", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    import torch
    name = f"TET_k{int(a.kappa)}"
    out = a.out or os.path.join(ROOT, "datasets", "v2", name + ".pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n_tots = np.random.default_rng([a.seed, 4]).integers(a.nmin, a.nmax + 1, a.n)
    t0 = time.time()
    samples = generate(np.arange(a.n), n_tots, a.seed, a.kappa, a.workers)
    t_gen = time.time() - t0
    meta = dict(task="TET", kappa=a.kappa, N=a.n, seed=a.seed, nmin=a.nmin, nmax=a.nmax, gen_seconds=t_gen,
                generator="datasets/generators/gen_TET.py",
                edge_order="unique src<dst pairs, lexicographic", face_order="unique sorted triples, lexicographic",
                pde="-div(sigma grad u) = f, u=0 on the boundary of [0,1]^3, P1 FEM")
    torch.save(pack(samples, meta), out)
    st = [s["stats"] for s in samples]
    summary = dict(meta, file=out, file_MB=os.path.getsize(out) / 1e6,
                   n0_mean=float(np.mean([s["n0"] for s in st])), n3_mean=float(np.mean([s["n3"] for s in st])),
                   dropped_tets_total=int(sum(s["dropped_tets"] for s in st)),
                   max_conservation_residual=float(max(s["conservation"] for s in st)),
                   max_residual=float(max(s["residual"] for s in st)))
    print(json.dumps(summary, indent=2))
    json.dump(summary, open(out.replace(".pt", ".json"), "w"), indent=2)
    if a.n_fine > 0:
        nt, nv = int(0.7 * a.n), int(0.15 * a.n)
        ids = np.arange(nt + nv, min(nt + nv + a.n_fine, a.n))
        t0 = time.time()
        fine = generate(ids, a.fine_factor * n_tots[ids], a.seed, a.kappa, a.workers)
        t_fine = time.time() - t0
        out_f = out.replace(".pt", "_fine.pt")
        meta_f = dict(meta, N=len(ids), fine_factor=a.fine_factor, coarse_ids=ids.tolist(), gen_seconds=t_fine)
        torch.save(pack(fine, meta_f), out_f)
        stf = [s["stats"] for s in fine]
        summary_f = dict(file=out_f, file_MB=os.path.getsize(out_f) / 1e6, N=len(ids), gen_seconds=t_fine,
                         n0_mean=float(np.mean([s["n0"] for s in stf])),
                         max_residual=float(max(s["residual"] for s in stf)))
        print(json.dumps(summary_f, indent=2))
        json.dump(dict(meta_f, **summary_f, coarse_ids=None), open(out_f.replace(".pt", ".json"), "w"), indent=2)


if __name__ == "__main__":
    main()
