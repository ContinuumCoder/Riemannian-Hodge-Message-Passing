"""Advection-diffusion rollouts with a divergence-free velocity 1-form (tasks DYN / DYNfix).

For every trajectory:
  * mesh     : DYN: new random Delaunay mesh of [0,1]^2 with n0 ~ U{1300..1700} nodes (``gen_HP.draw_mesh_2d``:
               jittered boundary nodes, CCW triangles); DYNfix: one fixed mesh (1500 nodes) for all trajectories;
  * velocity : stream function psi(x,y) = sum_{m,n=1..4} a_mn sin(m pi x) sin(n pi y), a_mn ~ N(0,1) (m^2+n^2)^-1
               (psi = 0 on the boundary), v = rot psi = (d psi/dy, -d psi/dx) (div v = 0, v.n = 0 on the
               boundary), rescaled so that max |v| (129^2 grid) = U ~ U[0.6, 1.0]  ->  Pe = U L / nu in [60, 100];
  * model input (odd degree-1): theta_e = (v(x_i) + v(x_j))/2 . (x_j - x_i) on the canonical edges i < j,
               plus the current scalar u_t at the nodes;
  * initial u: 2-5 Gaussian blobs (centres U[0.15, 0.85]^2, widths U[0.06, 0.15], amplitudes +-U[0.5, 1.5]).
Discretisation (finite volumes on the barycentric dual = DEC with the lumped mass, float64):
  * mass      M = diag(barycentric dual areas) (rhmp star0);
  * diffusion L = P1/cotan stiffness d0^T diag(w) d0 (zero row/column sums: no-flux boundary);
  * advection: dual-edge fluxes from the stream function, Phi_{i->j} = psi(b_left) - psi(b_right) (b = barycentres
    of the triangles left/right of the edge i->j; psi := 0 at boundary-edge midpoints).  Phi is *exactly*
    discretely divergence-free (the flux sum around every dual cell telescopes to 0), which is what makes u = const
    a discrete steady state.  Central (skew) flux form  (A u)_i = sum_j Phi_{i->j} (u_i + u_j)/2  (1^T A = 0 exactly:
    discrete mass conservation for any u), optionally blended with first-order upwinding A + alpha D,
    D = graph Laplacian with weights |Phi|/2 (symmetric PSD, zero row/column sums: conservative and dissipative);
  * time: Crank-Nicolson  (M + dt/2 (A + nu L)) u^{n+1} = (M - dt/2 (A + nu L)) u^n,  dt = 0.005, 8 sub-steps per
    stored step (Delta t = 0.04, stored-step CFL U Delta t / h ~ 1.1), 100 stored steps (T = 4).  Unconditionally stable (the symmetric part of A + nu L
    is PSD in the M inner product) and mass conserving to round-off; the per-step mass residual
    |1^T M (u^{n+1} - u^n)| / 1^T M |u^n| is stored (max over steps) together with the total drift.
Stored (DYN, packed like gen_HP.py with ptr0/ptr1/ptr2 over trajectories): pos (n0,2) f32, faces (n2,3) i32,
theta (n1) f32 (canonical lexicographic edges), vel (n0,2) f32 (node velocities, for node-based baselines),
traj (n0, T+1) f32 (u at the stored steps), mass (n0) f32, flux (n1) f32 (dual-edge fluxes Phi of the solver).
DYNfix stores the shared mesh once: pos, faces, mass, and per trajectory theta (N, n1), vel (N, n0, 2),
traj (N, T+1, n0), flux (N, n1).

Usage (server):
  python3 -u datasets/generators/gen_dyn.py [--n 600] [--n-fix 500] [--workers 32]
  -> datasets/v2/DYN.pt, datasets/v2/DYNfix.pt (+ .json summaries)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_HP import draw_mesh_2d  # noqa: E402
import surfaces as S  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF_GRID = 129          # grid for the max-|v| normalisation of the velocity


def default_config(**over) -> dict:
    cfg = dict(n_modes=4, spec_decay=1.0, U=(0.6, 1.0), nu=0.01, dt=0.005, substeps=8, steps=100, theta_cn=0.5,
               upwind=0.0, blobs=(2, 5), blob_w=(0.06, 0.15), blob_amp=(0.5, 1.5), nmin=1300, nmax=1700,
               n_fix=1500)
    cfg.update(over)
    return cfg


# ======================================================================================================================
# velocity field
# ======================================================================================================================
def draw_stream(rng: np.random.Generator, cfg: dict) -> dict:
    """Random stream-function coefficients ``a_mn`` (``psi = sum a_mn sin(m pi x) sin(n pi y)``), max |v| = U."""
    M = cfg["n_modes"]
    m, n = np.meshgrid(np.arange(1, M + 1), np.arange(1, M + 1), indexing="ij")
    a = rng.normal(size=(M, M)) * (m ** 2 + n ** 2) ** (-cfg["spec_decay"])
    g = np.linspace(0.0, 1.0, REF_GRID)
    X = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    vmax = np.linalg.norm(velocity(dict(a=a), X), axis=1).max()
    U = float(rng.uniform(*cfg["U"]))
    return dict(a=a * U / vmax, U=U)


def stream(p: dict, x: np.ndarray) -> np.ndarray:
    """``psi`` at points ``x (P, 2)``."""
    a = p["a"]
    M = a.shape[0]
    k = np.pi * np.arange(1, M + 1)
    sx, sy = np.sin(x[:, :1] * k), np.sin(x[:, 1:] * k)        # (P, M)
    return np.einsum("pm,mn,pn->p", sx, a, sy)


def velocity(p: dict, x: np.ndarray) -> np.ndarray:
    """``v = (d psi / dy, -d psi / dx)`` at points ``x (P, 2)`` -> ``(P, 2)``."""
    a = p["a"]
    M = a.shape[0]
    k = np.pi * np.arange(1, M + 1)
    sx, sy = np.sin(x[:, :1] * k), np.sin(x[:, 1:] * k)
    cx, cy = np.cos(x[:, :1] * k) * k, np.cos(x[:, 1:] * k) * k
    return np.stack([np.einsum("pm,mn,pn->p", sx, a, cy), -np.einsum("pm,mn,pn->p", cx, a, sy)], 1)


def edge_one_form(v_nodes: np.ndarray, pts: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """``theta_e = (v_i + v_j)/2 . (x_j - x_i)`` for edges ``(i, j)`` -> ``(n1,)`` (trapezoidal line integral)."""
    i, j = edges[:, 0], edges[:, 1]
    return 0.5 * ((v_nodes[i] + v_nodes[j]) * (pts[j] - pts[i])).sum(1)


def dual_fluxes(p: dict, pts: np.ndarray, faces: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Exact fluxes of ``v = rot psi`` through the barycentric dual edges, from the stream function.

    For the canonical edge ``i -> j`` the dual edge runs from the barycentre of the triangle on its right to the
    barycentre of the triangle on its left (through the edge midpoint); ``int v.n ds = psi(end) - psi(start)``
    with ``n`` pointing from the dual cell of ``i`` into that of ``j``.  Boundary edges have one triangle; the
    missing side is the edge midpoint on the boundary where ``psi = 0``.  Faces must be CCW.

    Returns:
        ``(n1,)`` fluxes ``Phi_{i->j}``.
    """
    n = len(pts)
    key = edges[:, 0].astype(np.int64) * n + edges[:, 1]
    psi_b = stream(p, pts[faces].mean(1))
    left = np.zeros(len(edges))
    right = np.zeros(len(edges))
    for k in range(3):
        a, b = faces[:, k], faces[:, (k + 1) % 3]                 # half-edge a -> b, triangle on its left
        idx = np.searchsorted(key, np.minimum(a, b).astype(np.int64) * n + np.maximum(a, b))
        same = a < b                                               # half-edge agrees with the canonical direction
        np.add.at(left, idx[same], psi_b[same])
        np.add.at(right, idx[~same], psi_b[~same])
    # boundary edges: the missing side contributes psi(midpoint on the boundary) = 0 (already zero-initialised)
    return left - right


def advection_matrix(Phi: np.ndarray, edges: np.ndarray, n: int, upwind: float = 0.0) -> sp.csr_matrix:
    """Conservative flux-form advection ``(A u)_i = sum_j Phi_{i->j} (u_i + u_j)/2`` (+ ``upwind * D``).

    ``1^T A = 0`` exactly (each edge flux leaves one dual cell and enters the other); ``A 1 = div Phi = 0`` for the
    stream-function fluxes.  ``D = sum_e |Phi_e|/2 (e_i - e_j)(e_i - e_j)^T`` (first-order upwind dissipation).
    """
    i, j = edges[:, 0], edges[:, 1]
    h = 0.5 * Phi
    rows = np.r_[i, i, j, j]
    cols = np.r_[i, j, j, i]
    vals = np.r_[h, h, -h, -h]                                     # row i: +Phi_ij/2 (u_i + u_j); row j: -Phi_ij/2 (..)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    if upwind:
        g = 0.5 * np.abs(Phi)
        D = sp.csr_matrix((np.r_[g, g, -g, -g], (np.r_[i, j, i, j], np.r_[i, j, j, i])), shape=(n, n))
        A = A + upwind * D
    return A.tocsr()


def blobs(rng: np.random.Generator, x: np.ndarray, cfg: dict) -> np.ndarray:
    nb = int(rng.integers(cfg["blobs"][0], cfg["blobs"][1] + 1))
    c = rng.uniform(0.15, 0.85, (nb, 2))
    w = rng.uniform(*cfg["blob_w"], nb)
    a = rng.uniform(*cfg["blob_amp"], nb) * rng.choice([-1.0, 1.0], nb)
    d2 = ((x[:, None, :] - c[None]) ** 2).sum(-1)
    return (a[None] * np.exp(-d2 / (2 * w[None] ** 2))).sum(1)


# ======================================================================================================================
# one trajectory
# ======================================================================================================================
def simulate(pts: np.ndarray, faces: np.ndarray, p: dict, u0: np.ndarray, cfg: dict) -> dict:
    """Crank-Nicolson advection-diffusion on a planar CCW triangle mesh (see module docstring).

    Returns:
        dict with ``traj (steps+1, n0)``, ``theta (n1)``, ``vel (n0, 2)``, ``flux (n1)``, ``mass (n0)``, ``edges``,
        diagnostics ``div_rel`` (max |d0^T Phi| / max |Phi|), ``mass_res`` (max per-substep relative mass residual),
        ``mass_drift`` (total), ``overshoot`` (max-principle violation / initial range), ``cfl``, ``pe_h``.
    """
    n = len(pts)
    ops = S.cotan_operators(pts, faces)
    edges, L, mass = ops["edges"], ops["L"], ops["mass"]
    Phi = dual_fluxes(p, pts, faces, edges)
    div = np.zeros(n)
    np.add.at(div, edges[:, 0], Phi)
    np.add.at(div, edges[:, 1], -Phi)
    A = advection_matrix(Phi, edges, n, cfg["upwind"])
    Mm = sp.diags(mass)
    dt, th, nu = cfg["dt"], cfg["theta_cn"], cfg["nu"]
    Op = A + nu * L
    lhs = (Mm + th * dt * Op).tocsc()
    rhs_m = (Mm - (1 - th) * dt * Op).tocsr()
    lu = spla.splu(lhs)
    u = u0.astype(np.float64).copy()
    traj = [u.copy()]
    m0 = float((mass * u).sum())
    scale = float((mass * np.abs(u)).sum())
    mres = 0.0
    for _ in range(cfg["steps"]):
        for _ in range(cfg["substeps"]):
            u_new = lu.solve(rhs_m @ u)
            mres = max(mres, abs(float((mass * (u_new - u)).sum())) / max(float((mass * np.abs(u)).sum()), 1e-300))
            u = u_new
        traj.append(u.copy())
    traj = np.stack(traj, 0)
    rng0 = float(u0.max() - u0.min())
    over = max(0.0, float(traj.max() - u0.max()), float(u0.min() - traj.min())) / max(rng0, 1e-300)
    v_nodes = velocity(p, pts)
    elen = np.linalg.norm(pts[edges[:, 1]] - pts[edges[:, 0]], axis=1)
    h_med = float(np.median(elen))
    return dict(traj=traj, theta=edge_one_form(v_nodes, pts, edges), vel=v_nodes, flux=Phi, mass=mass, edges=edges,
                div_rel=float(np.abs(div).max() / max(np.abs(Phi).max(), 1e-300)), mass_res=mres,
                mass_drift=abs(float((mass * traj[-1]).sum()) - m0) / max(scale, 1e-300), overshoot=over,
                cfl=float(p["U"] * dt / h_med), cfl_stored=float(p["U"] * dt * cfg["substeps"] / h_med),
                pe=float(p["U"] / nu), pe_h=float(p["U"] * h_med / nu), h_median=h_med)


def make_trajectory(args) -> dict:
    """Trajectory ``idx``: mesh (unless ``fixed`` = (pts, faces)), stream function, initial blobs, simulation."""
    idx, n_tot, seed, cfg, fixed = args
    t0 = time.time()
    if fixed is None:
        pts, faces = draw_mesh_2d(seed, idx, n_tot)
    else:
        pts, faces = fixed
    pts = pts.astype(np.float32).astype(np.float64)                 # stored precision = simulated precision
    rng = np.random.default_rng([seed, 21, idx])
    p = draw_stream(rng, cfg)
    u0 = blobs(rng, pts, cfg)
    sim = simulate(pts, faces, p, u0, cfg)
    stats = dict(idx=int(idx), n0=len(pts), n1=len(sim["edges"]), n2=len(faces), U=p["U"], t_total=time.time() - t0,
                 **{k: sim[k] for k in ("div_rel", "mass_res", "mass_drift", "overshoot", "cfl", "cfl_stored", "pe",
                                        "pe_h", "h_median")},
                 u_std_t0=float(sim["traj"][0].std()), u_std_tT=float(sim["traj"][-1].std()))
    return dict(pos=pts.astype(np.float32), faces=faces.astype(np.int32), theta=sim["theta"].astype(np.float32),
                vel=sim["vel"].astype(np.float32), traj=sim["traj"].T.astype(np.float32),
                mass=sim["mass"].astype(np.float32), flux=sim["flux"].astype(np.float32), stats=stats)


# ======================================================================================================================
# packing / driver
# ======================================================================================================================
def pack_variable(samples: list[dict], meta: dict) -> dict:
    import torch
    st = [s["stats"] for s in samples]
    ptr = lambda n: torch.from_numpy(np.concatenate([[0], np.cumsum(n)]).astype(np.int64))  # noqa: E731
    out = {"ptr0": ptr([s["n0"] for s in st]), "ptr1": ptr([s["n1"] for s in st]),
           "ptr2": ptr([s["n2"] for s in st]), "meta": meta}
    for key in ("pos", "faces", "theta", "vel", "traj", "mass", "flux"):
        out[key] = torch.from_numpy(np.concatenate([s[key] for s in samples], 0))
    out["sample_stats"] = st
    return out


def pack_fixed(samples: list[dict], meta: dict) -> dict:
    import torch
    s0 = samples[0]
    out = {"pos": torch.from_numpy(s0["pos"]), "faces": torch.from_numpy(s0["faces"]),
           "mass": torch.from_numpy(s0["mass"]), "meta": meta}
    out["theta"] = torch.from_numpy(np.stack([s["theta"] for s in samples], 0))
    out["vel"] = torch.from_numpy(np.stack([s["vel"] for s in samples], 0))
    out["traj"] = torch.from_numpy(np.stack([s["traj"].T for s in samples], 0))     # (N, T+1, n0)
    out["flux"] = torch.from_numpy(np.stack([s["flux"] for s in samples], 0))
    out["sample_stats"] = [s["stats"] for s in samples]
    return out


def generate(jobs: list, workers: int) -> list[dict]:
    if workers <= 1:
        return [make_trajectory(j) for j in jobs]
    with Pool(workers) as pool:
        return list(pool.imap(make_trajectory, jobs, chunksize=4))


def write_set(name: str, n: int, seed: int, cfg: dict, workers: int, out_dir: str, fixed: bool) -> dict:
    import torch
    if fixed:
        mesh = draw_mesh_2d(seed, 0, cfg["n_fix"])
        jobs = [(i, cfg["n_fix"], seed, cfg, mesh) for i in range(n)]
    else:
        n_tots = np.random.default_rng([seed, 4]).integers(cfg["nmin"], cfg["nmax"] + 1, n)
        jobs = [(i, int(n_tots[i]), seed, cfg, None) for i in range(n)]
    t0 = time.time()
    samples = generate(jobs, workers)
    t_gen = time.time() - t0
    meta = dict(task="DYN", set=name, N=n, seed=seed, fixed_mesh=fixed, gen_seconds=t_gen, config=cfg,
                generator="datasets/generators/gen_dyn.py", dt_stored=cfg["dt"] * cfg["substeps"],
                edge_order="unique src<dst pairs of the triangles, lexicographic",
                pde="du/dt + v.grad u = nu Lap u on [0,1]^2, no-flux boundary, v = rot psi (psi = 0 on the boundary)",
                scheme="finite volumes on the barycentric dual: stream-function dual fluxes (exactly div-free), central "
                       "flux form (+ upwind*D), cotan diffusion, lumped mass, Crank-Nicolson")
    path = os.path.join(out_dir, name + ".pt")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(pack_fixed(samples, meta) if fixed else pack_variable(samples, meta), path)
    st = [s["stats"] for s in samples]
    summary = dict({k: v for k, v in meta.items() if k != "config"}, file=path, file_MB=os.path.getsize(path) / 1e6,
                   n0_mean=float(np.mean([s["n0"] for s in st])), n0_min=int(min(s["n0"] for s in st)),
                   n0_max=int(max(s["n0"] for s in st)), max_div_rel=float(max(s["div_rel"] for s in st)),
                   max_mass_res=float(max(s["mass_res"] for s in st)),
                   max_mass_drift=float(max(s["mass_drift"] for s in st)),
                   overshoot_mean=float(np.mean([s["overshoot"] for s in st])),
                   overshoot_max=float(max(s["overshoot"] for s in st)),
                   cfl_stored_mean=float(np.mean([s["cfl_stored"] for s in st])),
                   pe_range=[float(min(s["pe"] for s in st)), float(max(s["pe"] for s in st))],
                   pe_h_max=float(max(s["pe_h"] for s in st)),
                   std_ratio_T_over_0=float(np.mean([s["u_std_tT"] / max(s["u_std_t0"], 1e-12) for s in st])),
                   config=cfg)
    json.dump(summary, open(path.replace(".pt", ".json"), "w"), indent=2)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=600, help="DYN trajectories (one random mesh each)")
    ap.add_argument("--n-fix", type=int, default=500, help="DYNfix trajectories (one shared mesh)")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--substeps", type=int, default=8)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--nu", type=float, default=0.01)
    ap.add_argument("--upwind", type=float, default=0.0, help="upwind blending alpha in [0, 1]")
    ap.add_argument("--nmin", type=int, default=1300)
    ap.add_argument("--nmax", type=int, default=1700)
    ap.add_argument("--n-fix-nodes", type=int, default=1500, help="nodes of the DYNfix mesh")
    ap.add_argument("--seed", type=int, default=2040)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "datasets", "v2"))
    ap.add_argument("--prefix", default="DYN")
    a = ap.parse_args(argv)
    cfg = default_config(steps=a.steps, substeps=a.substeps, dt=a.dt, nu=a.nu, upwind=a.upwind, nmin=a.nmin,
                         nmax=a.nmax, n_fix=a.n_fix_nodes)
    out = {}
    for name, n, seed, fixed in ((a.prefix, a.n, a.seed, False), (a.prefix + "fix", a.n_fix, a.seed + 1, True)):
        if n <= 0:
            continue
        s = write_set(name, n, seed, cfg, a.workers, a.out_dir, fixed)
        print(json.dumps({k: v for k, v in s.items() if k != "config"}, indent=2), flush=True)
        out[name] = s
    return out


if __name__ == "__main__":
    main()
