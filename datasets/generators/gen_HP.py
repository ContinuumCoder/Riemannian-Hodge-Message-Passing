"""Hetero(-anisotropic) Poisson on random 2-D Delaunay meshes of the unit square (tasks HP / HPflux).

For every sample:
  * mesh     : new random Delaunay mesh of [0,1]^2 with n0 ~ U{1000..2000} nodes (4m jittered boundary nodes, the
               rest uniform in the interior), triangles oriented counter-clockwise;
  * sigma(x) : log-normal conductivity  log sigma = 0.5 ln(kappa) * g_n(x),  g_n a Gaussian random field (random
               Fourier features, length scale U[0.1, 0.3]) rescaled to [-1, 1] on a fixed 65x65 reference grid, so
               that max sigma / min sigma = kappa (contrast), independently of the mesh;
  * aniso    : optional SPD conductivity *tensor* Sigma(x) = sigma(x) A(x), A = R(theta(x)) diag(sqrt r, 1/sqrt r) R^T,
               det A = 1, with a random principal-direction field theta(x) = pi * g2_n(x) and
               - ``--aniso-max R`` (sets ``HP_k<kappa>_aniso<R>``): a spatially varying eigenvalue ratio
                 r(x) = R^{(g3_n(x)+1)/2} in [1, R] (third random field);
               - ``--aniso`` (legacy ``HP_k<kappa>_aniso``): a constant per-sample ratio r ~ logU[2, 20];
  * f(x)     : 2-5 Gaussian sources (centres U[0.1,0.9]^2, widths U[0.05,0.2], amplitudes +-U[1,3]);
  * solve    : -div(sigma A grad u) = f in the square, u = 0 on the boundary; P1 FEM with element-wise
               sigma/A at the triangle centroid, lumped mass right-hand side, sparse direct solve (SuperLU).
Stored per sample (canonical edges = unique src<dst pairs of the triangles, lexicographic order):
  pos (n0,2) f32, faces (n2,3) i32, f (n0) f32, u (n0) f32, bnd (n0) bool,
  sigma_edge (n1) f32         sigma at edge midpoints (even edge input)
  sigma_dir_edge (n1) f32     (legacy aniso) sigma t^T A t at the midpoint = conductivity along the edge
  sigma_proj_edge (n1) f32    (tensor aniso) t^T Sigma t at the midpoint: the E(n)-invariant per-edge projection
                              of the conductivity tensor (even edge input; 3 edges of a face determine Sigma_f)
  logsigma_face (n2) f32      log sigma at triangle centroids (even face input)
  logdet_face, logratio_face  (tensor aniso) tensor invariants log det Sigma, log(lambda_max/lambda_min) at centroids
  theta_face, sigma_tensor_face (tensor aniso, evaluation only) major-axis angle and (S_xx, S_xy, S_yy) at centroids
  flux (n1) f32               -sigma_e (u_dst - u_src)   (HPflux; sigma_e = the edge conductivity input)
  flux_fem (n1) f32           -w_e (u_dst - u_src) with w_e = -K_{src,dst} the FEM edge weight: the conservative
                              discrete flux, d0^T flux_fem = -M f exactly at interior nodes (HPfluxfem)
Samples are packed as flat concatenations with offsets ``ptr0/ptr1/ptr2`` (N+1).

Random streams are independent of kappa: the HP_k10 / HP_k100 / HP_k1000 sets share meshes, g and f (only the contrast
differs), and the aniso set adds the tensor field on top.  A second file ``*_fine.pt`` re-solves the first ``n_fine``
samples of the (sequential 70/15/15) *test* split on meshes with 4x as many nodes (same sigma/A/f fields): zero-shot
resolution transfer on identical physical instances.

Usage:
  python3 -u datasets/generators/gen_HP.py --kappa 100 [--aniso | --aniso-max 10] [--n 5000] [--n-fine 500] [--workers 32]
  -> datasets/v2/HP_k100[_aniso|_aniso10].pt, ..._fine.pt (+ .json summaries)
  python3 -u datasets/generators/gen_HP.py --selftest     (FEM convergence check against a manufactured solution)
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

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
N_RFF = 64            # random Fourier features per random field
REF_GRID = 65         # reference grid for the mesh-independent [-1, 1] normalisation of the random fields
BND_TOL = 1e-12       # boundary node detection tolerance
MIN_AREA = 1e-12      # triangles below this area are dropped (never observed with the jittered boundary)


# ------------------------------------------------------------------------------------------------------------------
# random fields
# ------------------------------------------------------------------------------------------------------------------
def rff_draw(rng, dim, ell):
    """Random Fourier features of a squared-exponential GRF with length scale ``ell``."""
    k = rng.normal(0.0, 1.0 / (2 * np.pi * ell), size=(N_RFF, dim))
    ph = rng.uniform(0.0, 2 * np.pi, N_RFF)
    a = rng.normal(0.0, 1.0, N_RFF) * np.sqrt(2.0 / N_RFF)
    return k, ph, a


def rff_eval(p, x):
    """Evaluate the random field ``p`` at points ``x (M, dim)`` -> ``(M,)``."""
    k, ph, a = p
    return np.cos(2 * np.pi * x @ k.T + ph) @ a


def ref_minmax(p, dim, n_grid):
    g = np.linspace(0.0, 1.0, n_grid)
    X = np.stack(np.meshgrid(*([g] * dim), indexing="ij"), -1).reshape(-1, dim)
    v = rff_eval(p, X)
    return float(v.min()), float(v.max())


def draw_fields(seed, idx, dim=2, aniso=False, ref_grid=REF_GRID, aniso_max=None):
    """Conductivity / anisotropy / source parameters of sample ``idx`` (independent of kappa and of the mesh)."""
    rng = np.random.default_rng([seed, 1, idx])
    ell = rng.uniform(0.1, 0.3)
    p_sig = rff_draw(rng, dim, ell)
    gmin, gmax = ref_minmax(p_sig, dim, ref_grid)
    n_src = int(rng.integers(2, 6))
    src = dict(c=rng.uniform(0.1, 0.9, (n_src, dim)), w=rng.uniform(0.05, 0.2, n_src),
               a=rng.uniform(1.0, 3.0, n_src) * rng.choice([-1.0, 1.0], n_src))
    F = dict(p_sig=p_sig, gmin=gmin, gmax=gmax, ell=ell, src=src, aniso=None)
    if aniso:
        rng_a = np.random.default_rng([seed, 3, idx])
        p_th = rff_draw(rng_a, dim, rng_a.uniform(0.2, 0.5))
        tmin, tmax = ref_minmax(p_th, dim, ref_grid)
        F["aniso"] = dict(p_th=p_th, tmin=tmin, tmax=tmax, ratio=float(np.exp(rng_a.uniform(np.log(2), np.log(20)))))
    if aniso_max:
        rng_t = np.random.default_rng([seed, 5, idx])
        p_th = rff_draw(rng_t, dim, rng_t.uniform(0.2, 0.5))
        tmin, tmax = ref_minmax(p_th, dim, ref_grid)
        p_r = rff_draw(rng_t, dim, rng_t.uniform(0.2, 0.5))
        rmin, rmax = ref_minmax(p_r, dim, ref_grid)
        F["tensor"] = dict(p_th=p_th, tmin=tmin, tmax=tmax, p_r=p_r, rmin=rmin, rmax=rmax, rmax_ratio=float(aniso_max))
    return F


def log_sigma(F, x, kappa):
    g = rff_eval(F["p_sig"], x)
    gn = 2.0 * (g - F["gmin"]) / max(F["gmax"] - F["gmin"], 1e-12) - 1.0
    return 0.5 * np.log(kappa) * gn


def aniso_tensor(F, x):
    """``(M, 2, 2)`` SPD tensors with det 1 at points ``x``."""
    A = F["aniso"]
    t = rff_eval(A["p_th"], x)
    th = np.pi * (2.0 * (t - A["tmin"]) / max(A["tmax"] - A["tmin"], 1e-12) - 1.0)
    c, s = np.cos(th), np.sin(th)
    R = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)          # (M, 2, 2)
    D = np.array([np.sqrt(A["ratio"]), 1.0 / np.sqrt(A["ratio"])])
    return np.einsum("mij,j,mkj->mik", R, D, R)


def tensor_field(F, x, return_theta=False):
    """Spatially varying unit-determinant anisotropy ``(A (M, 2, 2), log_ratio (M,))`` at points ``x``
    (``return_theta``: also the angle of the major principal axis)."""
    T = F["tensor"]
    t = rff_eval(T["p_th"], x)
    th = np.pi * (2.0 * (t - T["tmin"]) / max(T["tmax"] - T["tmin"], 1e-12) - 1.0)
    g = rff_eval(T["p_r"], x)
    un = np.clip((g - T["rmin"]) / max(T["rmax"] - T["rmin"], 1e-12), 0.0, 1.0)     # in [0, 1]
    log_r = un * np.log(T["rmax_ratio"])
    c, s = np.cos(th), np.sin(th)
    R = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
    D = np.stack([np.exp(0.5 * log_r), np.exp(-0.5 * log_r)], -1)                   # (M, 2)
    A = np.einsum("mij,mj,mkj->mik", R, D, R)
    return (A, log_r, th) if return_theta else (A, log_r)


def source(F, x):
    s = F["src"]
    d2 = ((x[:, None, :] - s["c"][None]) ** 2).sum(-1)
    return (s["a"][None] * np.exp(-d2 / (2 * s["w"][None] ** 2))).sum(-1)


# ------------------------------------------------------------------------------------------------------------------
# mesh
# ------------------------------------------------------------------------------------------------------------------
def draw_mesh_2d(seed, idx, n_tot, max_tries=20):
    """Random Delaunay mesh of the unit square with ``n_tot`` nodes, CCW triangles.

    Returns:
        pts ``(n_tot, 2)`` float64, faces ``(n2, 3)`` int64.
    """
    rng = np.random.default_rng([seed, 2, idx, n_tot])
    m = max(4, int(round(np.sqrt(n_tot))))           # boundary nodes per side (incl. one corner)
    for _ in range(max_tries):
        sides = []
        for s in range(4):
            t = (np.arange(1, m) + rng.uniform(-0.25, 0.25, m - 1)) / m
            z, o = np.zeros_like(t), np.ones_like(t)
            sides.append([np.stack([t, z], 1), np.stack([o, t], 1), np.stack([t, o], 1), np.stack([z, t], 1)][s])
        corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        n_int = n_tot - 4 * m
        delta = 0.3 / m
        interior = rng.uniform(delta, 1.0 - delta, (n_int, 2))
        pts = np.concatenate([corners] + sides + [interior], 0)
        tri = Delaunay(pts)
        if len(tri.coplanar):
            continue
        faces = tri.simplices.astype(np.int64)
        P = pts[faces]
        sa = 0.5 * ((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) -
                    (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0]))
        keep = np.abs(sa) > MIN_AREA
        faces, sa = faces[keep], sa[keep]
        neg = sa < 0
        faces[neg] = faces[neg][:, [0, 2, 1]]
        if len(np.unique(faces)) != n_tot:
            continue
        return pts, faces
    raise RuntimeError(f"could not build a valid mesh for sample {idx}")


# ------------------------------------------------------------------------------------------------------------------
# one sample
# ------------------------------------------------------------------------------------------------------------------
def solve_sample(args):
    idx, n_tot, seed, kappa, aniso, aniso_max = args
    F = draw_fields(seed, idx, 2, aniso, aniso_max=aniso_max)
    pts, faces = draw_mesh_2d(seed, idx, n_tot)
    n = len(pts)
    cent = pts[faces].mean(1)
    ls_T = log_sigma(F, cent, kappa)
    A_T, lr_T = None, None
    if aniso:
        A_T = aniso_tensor(F, cent)
    elif aniso_max:
        A_T, lr_T, th_T = tensor_field(F, cent, return_theta=True)
    K, mass, _ = p1_stiffness_2d(pts, faces, np.exp(ls_T), A_T)
    f = source(F, pts)
    b = mass * f
    bnd = (pts[:, 0] < BND_TOL) | (pts[:, 0] > 1 - BND_TOL) | (pts[:, 1] < BND_TOL) | (pts[:, 1] > 1 - BND_TOL)
    I = np.where(~bnd)[0]
    KII = K[I][:, I].tocsc()
    u = np.zeros(n)
    u[I] = spla.spsolve(KII, b[I])
    res = float(np.linalg.norm(KII @ u[I] - b[I]) / max(np.linalg.norm(b[I]), 1e-30))
    edges = mesh_edges(faces, n)
    mid = pts[edges].mean(1)
    ls_e = log_sigma(F, mid, kappa)
    sig_e = np.exp(ls_e)
    du = u[edges[:, 1]] - u[edges[:, 0]]
    out = dict(pos=pts.astype(np.float32), faces=faces.astype(np.int32), f=f.astype(np.float32),
               u=u.astype(np.float32), bnd=bnd, sigma_edge=sig_e.astype(np.float32),
               logsigma_face=ls_T.astype(np.float32))
    sig_flux = sig_e
    t = pts[edges[:, 1]] - pts[edges[:, 0]]
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    if aniso:
        sig_dir = sig_e * np.einsum("ei,eij,ej->e", t, aniso_tensor(F, mid), t)
        out["sigma_dir_edge"] = sig_dir.astype(np.float32)
        sig_flux = sig_dir
    elif aniso_max:
        A_e, _ = tensor_field(F, mid)
        sig_proj = sig_e * np.einsum("ei,eij,ej->e", t, A_e, t)
        out["sigma_proj_edge"] = sig_proj.astype(np.float32)
        out["logdet_face"] = (2.0 * ls_T).astype(np.float32)             # det(sigma A) = sigma^2
        out["logratio_face"] = lr_T.astype(np.float32)
        # evaluation-only ground truth (frame dependent, never a model input): major-axis angle and Sigma_f
        out["theta_face"] = th_T.astype(np.float32)
        S = np.exp(ls_T)[:, None, None] * A_T
        out["sigma_tensor_face"] = np.stack([S[:, 0, 0], S[:, 0, 1], S[:, 1, 1]], 1).astype(np.float32)
        sig_flux = sig_proj
    out["flux"] = (-sig_flux * du).astype(np.float32)
    # conservative FEM flux on the dual faces of the edges: w_e = -K_{src,dst};  d0^T j = -M f at interior nodes
    w = -np.asarray(K[edges[:, 0], edges[:, 1]]).ravel()
    j_fem = -w * du
    div = np.zeros(n)
    np.add.at(div, edges[:, 0], -j_fem)
    np.add.at(div, edges[:, 1], j_fem)
    cons = float(np.abs(div[I] + b[I]).max() / max(np.abs(b[I]).max(), 1e-30))
    out["flux_fem"] = j_fem.astype(np.float32)
    ratio = F["aniso"]["ratio"] if aniso else (float(np.exp(lr_T.max())) if aniso_max else 1.0)
    out["stats"] = dict(idx=idx, n0=n, n1=len(edges), n2=len(faces), residual=res, conservation=cons,
                        ell=F["ell"], ratio=ratio)
    return out


# ------------------------------------------------------------------------------------------------------------------
# packing
# ------------------------------------------------------------------------------------------------------------------
def pack(samples, meta):
    import torch
    n0 = np.array([s["stats"]["n0"] for s in samples])
    n1 = np.array([s["stats"]["n1"] for s in samples])
    n2 = np.array([len(s["faces"]) for s in samples])
    ptr = lambda n: torch.from_numpy(np.concatenate([[0], np.cumsum(n)]).astype(np.int64))
    out = {"ptr0": ptr(n0), "ptr1": ptr(n1), "ptr2": ptr(n2), "meta": meta}
    for key in samples[0]:
        if key == "stats":
            continue
        out[key] = torch.from_numpy(np.concatenate([s[key] for s in samples], 0))
    out["sample_stats"] = [s["stats"] for s in samples]
    return out


def generate(ids, n_tots, seed, kappa, aniso, workers, aniso_max=None):
    args = [(int(i), int(n), seed, kappa, aniso, aniso_max) for i, n in zip(ids, n_tots)]
    with Pool(workers) as pool:
        return list(pool.imap(solve_sample, args, chunksize=8))


def selftest():
    """P1 FEM convergence on u* = sin(pi x) sin(pi y), sigma = 1 (expect ~O(h^2) nodal error)."""
    errs = []
    for n_tot in (500, 2000, 8000):
        pts, faces = draw_mesh_2d(0, 0, n_tot)
        K, mass, _ = p1_stiffness_2d(pts, faces, np.ones(len(faces)))
        us = np.sin(np.pi * pts[:, 0]) * np.sin(np.pi * pts[:, 1])
        b = mass * 2 * np.pi ** 2 * us
        bnd = (pts.min(1) < BND_TOL) | (pts.max(1) > 1 - BND_TOL)
        I = np.where(~bnd)[0]
        u = np.zeros(len(pts))
        u[I] = spla.spsolve(K[I][:, I].tocsc(), b[I])
        errs.append(float(np.abs(u - us).max()))
        print(f"selftest n0={n_tot:5d}: max nodal error {errs[-1]:.3e}")
    assert errs[-1] < errs[0] / 8 and errs[-1] < 5e-3, errs
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kappa", type=float, default=100.0)
    ap.add_argument("--aniso", action="store_true", help="legacy anisotropy: constant per-sample ratio in [2, 20]")
    ap.add_argument("--aniso-max", type=float, default=None,
                    help="tensor anisotropy with a spatially varying eigenvalue ratio in [1, R] (HP_k<kappa>_aniso<R>)")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--n-fine", type=int, default=500)
    ap.add_argument("--fine-factor", type=int, default=4, help="node-count factor of the resolution-transfer set")
    ap.add_argument("--nmin", type=int, default=1000)
    ap.add_argument("--nmax", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    import torch
    if a.aniso and a.aniso_max:
        raise SystemExit("--aniso and --aniso-max are exclusive")
    name = f"HP_k{int(a.kappa)}" + ("_aniso" if a.aniso else "") + (f"_aniso{int(a.aniso_max)}" if a.aniso_max else "")
    out = a.out or os.path.join(ROOT, "datasets", "v2", name + ".pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n_tots = np.random.default_rng([a.seed, 4]).integers(a.nmin, a.nmax + 1, a.n)
    t0 = time.time()
    samples = generate(np.arange(a.n), n_tots, a.seed, a.kappa, a.aniso, a.workers, a.aniso_max)
    t_gen = time.time() - t0
    meta = dict(task="HP", kappa=a.kappa, aniso=a.aniso, aniso_max=a.aniso_max, N=a.n, seed=a.seed, nmin=a.nmin,
                nmax=a.nmax,
                gen_seconds=t_gen, generator="datasets/generators/gen_HP.py",
                edge_order="unique src<dst pairs of the triangles, lexicographic",
                pde="-div(sigma A grad u) = f, u=0 on the boundary of [0,1]^2, P1 FEM")
    torch.save(pack(samples, meta), out)
    st = [s["stats"] for s in samples]
    summary = dict(meta, file=out, file_MB=os.path.getsize(out) / 1e6,
                   n0_mean=float(np.mean([s["n0"] for s in st])), n0_min=int(min(s["n0"] for s in st)),
                   n0_max=int(max(s["n0"] for s in st)), max_residual=float(max(s["residual"] for s in st)),
                   max_conservation_residual=float(max(s["conservation"] for s in st)),
                   max_ratio=float(max(s["ratio"] for s in st)),
                   u_std=float(np.std(np.concatenate([s["u"] for s in samples]))))
    print(json.dumps(summary, indent=2))
    json.dump(summary, open(out.replace(".pt", ".json"), "w"), indent=2)

    if a.n_fine > 0:
        nt, nv = int(0.7 * a.n), int(0.15 * a.n)
        ids = np.arange(nt + nv, min(nt + nv + a.n_fine, a.n))
        t0 = time.time()
        fine = generate(ids, a.fine_factor * n_tots[ids], a.seed, a.kappa, a.aniso, a.workers, a.aniso_max)
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
