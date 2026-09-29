"""Scalar PDEs on variable closed surfaces (tasks SURF / SURF_heat, test sets SURF_geo / SURF_topo).

Every sample is a new closed surface meshed from a parametric UV grid (see ``surfaces.py``):
  (a) ellipsoid, semi-axes U[0.5, 1.5]^3                                  genus 0   [train/val/test]
  (b) superquadric (superellipsoid), semi-axes U[0.5,1.5]^3, exponents e1, e2 ~ U[0.3, 1.4]
                                                                          genus 0   [SURF_geo only]
  (c) sphere with a real spherical-harmonic radial perturbation, degrees 2..4, max |dr| ~ U[0.1, 0.35]
                                                                          genus 0   [train/val/test]
  (d) torus, r/R ~ U[0.25, 0.5], tube radius perturbed by <= U[0, 0.15] (low-order Fourier modes)
                                                                          genus 1   [train/val/test]
  (e) double torus: two tori (r/R ~ U[0.3, 0.5]) joined by a neck      genus 2   [SURF_topo only]
Resolution: the shape (unit scale) is rescaled to a surface area A ~ U[7.5, 15] (double torus: each lobe like a
torus of family (d), total area ~2A) and meshed with a median edge length close to h = 0.07 (V ~ 1.5K-3K; genus 2:
~3K-6K).  A *fixed physical resolution* is deliberate: the v2 model is exactly invariant to the length unit
(stars are normalised per sample, geometry descriptors are log(x / median)), so a PDE with an absolute length scale
(eps, T below) is only learnable if the ratio (PDE length) / (mesh spacing) is the same for all samples.
Every surface gets a random rotation and a translation U[-0.5, 0.5]^3 (coordinates never enter the v2 model;
coordinate-based baselines have to learn the invariance).

Physics (DEC / P1-FEM on the mesh, float64, positions rounded to float32 first so that the stored mesh reproduces
the operators exactly):
  L    = d0^T diag(w) d0,  w_e = (cot a + cot b)/2  (unclamped cotan Laplace-Beltrami stiffness)
  M    = diag(barycentric dual areas)  (lumped mass = rhmp star0)
  f    = GRF in the embedding space (64 random Fourier features, length scale U[0.25, 0.6], unit variance)
         + 1..4 Gaussian bumps (centres at random vertices, widths U[0.1, 0.3], amplitudes +-U[1, 2.5]),
         evaluated at the vertices
  u    : screened Poisson   (M + eps L) u = M f,  eps = 0.05                       -> target of SURF
  heat : u_T = exp(T Lap) f (heat flow du/dt = Lap u, Lap <= 0 the Laplace-Beltrami operator) by 10
         implicit-Euler steps (M + dt L) v^{k+1} = M v^k, dt = T/10, T = 0.05
                                                                                    -> target of SURF_heat
Stored per sample (packed with offsets ptr0/ptr1/ptr2 like gen_HP.py): pos (n0,3) f32, faces (n2,3) i32 (CCW w.r.t.
the outward normal), f, u, heat (n0) f32; per-sample ``family`` / ``genus`` (N,) int64; ``sample_stats`` (mesh
report, residuals, shape parameters).

Usage:
  python3 -u datasets/generators/gen_surf.py [--n 5700] [--n-geo 500] [--n-topo 500] [--workers 32]
  -> datasets/v2/SURF.pt (families a, c, d interleaved; sequential 70/15/15 split in the loader),
     datasets/v2/SURF_geo.pt (b), datasets/v2/SURF_topo.pt (e)   (+ .json summaries)
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
import surfaces as S  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FAMILIES = ("ellipsoid", "superquadric", "sphere_pert", "torus", "double_torus")
GENUS = {"ellipsoid": 0, "superquadric": 0, "sphere_pert": 0, "torus": 1, "double_torus": 2}
MAIN_FAMILIES = ("ellipsoid", "sphere_pert", "torus")     # interleaved in SURF.pt (sample i -> MAIN_FAMILIES[i % 3])
N_RFF = 64


# ======================================================================================================================
# configuration
# ======================================================================================================================
def default_config(**over) -> dict:
    cfg = dict(h=0.07, area_min=7.5, area_max=15.0, eps=0.05, T=0.05, heat_steps=10, smooth_iters=30,
               translate=0.5, grf_ell=(0.25, 0.6), bump_w=(0.1, 0.3), bump_amp=(1.0, 2.5), bump_n=(1, 4))
    cfg.update(over)
    return cfg


# ======================================================================================================================
# surfaces
# ======================================================================================================================
def make_surface(family: str, rng: np.random.Generator, cfg: dict):
    """Mesh one random surface of ``family`` (before the random rigid motion).

    Returns:
        ``pts (V, 3)`` float64, ``faces (F, 3)`` int64, ``params`` (JSON-friendly shape parameters).
    """
    h = cfg["h"]
    A = float(rng.uniform(cfg["area_min"], cfg["area_max"]))
    if family in ("ellipsoid", "superquadric", "sphere_pert"):
        if family == "ellipsoid":
            axes = rng.uniform(0.5, 1.5, 3)
            radius = lambda w: S.ellipsoid_radius(w, axes)  # noqa: E731
            params = dict(axes=axes.tolist())
        elif family == "superquadric":
            axes = rng.uniform(0.5, 1.5, 3)
            e1, e2 = (float(x) for x in rng.uniform(0.3, 1.4, 2))
            radius = lambda w: S.superquadric_radius(w, axes, e1, e2)  # noqa: E731
            params = dict(axes=axes.tolist(), e1=e1, e2=e2)
        else:
            amp = float(rng.uniform(0.1, 0.35))
            coeffs = S.draw_sh_perturbation(rng, 2, 4, amp)
            radius = lambda w: S.sh_radius(w, coeffs)  # noqa: E731
            params = dict(amp=amp, lmax=4)
        pts, faces, info = S.genus0_mesh(radius, A, h, rng, smooth_iters=cfg["smooth_iters"])
        params.update(info)
    elif family == "torus":
        rho = float(rng.uniform(0.25, 0.5))
        R = float(np.sqrt(A / (4 * np.pi ** 2 * rho)))
        r = rho * R
        nu, nv = max(8, int(round(2 * np.pi * R / h))), max(6, int(round(2 * np.pi * r / h)))
        pamp = float(rng.uniform(0.0, 0.15))
        pert = S.draw_tube_perturbation(rng, pamp)
        pts, faces = S.torus_mesh(R, r, nu, nv, rng, pert)
        params = dict(R=R, r=r, nu=nu, nv=nv, pert_amp=pamp)
    elif family == "double_torus":
        rho = float(rng.uniform(0.3, 0.5))
        R = float(np.sqrt(A / (4 * np.pi ** 2 * rho)))
        r = rho * R
        nv = max(6, int(round(2 * np.pi * r / h)))
        k = int(min(rng.integers(2, 5), max(2, (nv - 6) // 4)))
        gap = float(rng.uniform(0.6, 1.6) * k * h)
        tau = float(rng.uniform(0.8, 1.4))
        pts, faces, info = S.double_torus_mesh(R, r, gap, h, k, rng, tau_factor=tau)
        params = dict(R=R, r=r, gap=gap, tau_factor=tau, lobe_area=A, **info)
    else:
        raise ValueError(f"unknown family {family!r}")
    params["area_target"] = A
    return pts, faces, params


def surface_source(rng: np.random.Generator, pts: np.ndarray, cfg: dict) -> tuple[np.ndarray, dict]:
    """Random smooth source: embedding-space GRF (random Fourier features) + Gaussian bumps, at the vertices."""
    ell = float(rng.uniform(*cfg["grf_ell"]))
    k = rng.normal(0.0, 1.0 / (2 * np.pi * ell), (N_RFF, 3))
    ph = rng.uniform(0, 2 * np.pi, N_RFF)
    a = rng.normal(0.0, 1.0, N_RFF) * np.sqrt(2.0 / N_RFF)
    f = np.cos(2 * np.pi * pts @ k.T + ph) @ a
    nb = int(rng.integers(cfg["bump_n"][0], cfg["bump_n"][1] + 1))
    c = pts[rng.choice(len(pts), nb, replace=False)]
    w = rng.uniform(*cfg["bump_w"], nb)
    amp = rng.uniform(*cfg["bump_amp"], nb) * rng.choice([-1.0, 1.0], nb)
    d2 = ((pts[:, None, :] - c[None]) ** 2).sum(-1)
    f = f + (amp[None] * np.exp(-d2 / (2 * w[None] ** 2))).sum(1)
    return f, dict(grf_ell=ell, n_bumps=nb)


# ======================================================================================================================
# solvers
# ======================================================================================================================
def solve_surface(pts: np.ndarray, faces: np.ndarray, f: np.ndarray, eps: float, T: float, steps: int) -> dict:
    """Screened Poisson and implicit-Euler heat flow with the cotan Laplacian and the barycentric lumped mass.

    Returns:
        dict with ``u`` ((M + eps L) u = M f), ``heat`` (steps x (M + dt L) v' = M v from v = f), relative residuals
        ``res_u``, ``res_heat`` (max over steps), ``heat_mass_drift`` = |sum M v_T - sum M f| / sum M |f|,
        ``neg_cotan_frac`` (fraction of edges with negative cotan weight), operators ``L``, ``mass``.
    """
    ops = S.cotan_operators(pts, faces)
    L, M = ops["L"], ops["mass"]
    Mf = M * f
    Au = (sp.diags(M) + eps * L).tocsc()
    u = spla.splu(Au).solve(Mf)
    res_u = float(np.linalg.norm(Au @ u - Mf) / max(np.linalg.norm(Mf), 1e-300))
    dt = T / steps
    Ah = (sp.diags(M) + dt * L).tocsc()
    lu = spla.splu(Ah)
    v = f.copy()
    res_h = 0.0
    for _ in range(steps):
        rhs = M * v
        v = lu.solve(rhs)
        res_h = max(res_h, float(np.linalg.norm(Ah @ v - rhs) / max(np.linalg.norm(rhs), 1e-300)))
    drift = float(abs((M * v).sum() - Mf.sum()) / max(np.abs(Mf).sum(), 1e-300))
    return dict(u=u, heat=v, res_u=res_u, res_heat=res_h, heat_mass_drift=drift,
                neg_cotan_frac=float((ops["w"] < 0).mean()), L=L, mass=M)


# ======================================================================================================================
# one sample
# ======================================================================================================================
def make_sample(args) -> dict:
    """Generate sample ``idx`` of ``family`` (seeded by ``(seed, family code, idx)``)."""
    idx, family, seed, cfg = args
    t0 = time.time()
    fam = FAMILIES.index(family)
    rng = np.random.default_rng([seed, 11, fam, idx])
    for attempt in range(10):
        try:
            pts, faces, params = make_surface(family, rng, cfg)
            Q = S.random_rotation(rng)
            pts = pts @ Q.T + rng.uniform(-cfg["translate"], cfg["translate"], 3)
            pts = pts.astype(np.float32).astype(np.float64)       # stored precision = solved precision
            rep = S.mesh_report(pts, faces)
            S.assert_valid_closed(rep, GENUS[family], f"({family} #{idx})")
            break
        except RuntimeError:
            if attempt == 9:
                raise
    t_mesh = time.time() - t0
    f, fparams = surface_source(rng, pts, cfg)
    sol = solve_surface(pts, faces, f, cfg["eps"], cfg["T"], cfg["heat_steps"])
    stats = dict(idx=int(idx), family=family, genus=GENUS[family], n0=rep["n0"], n1=rep["n1"], n2=rep["n2"],
                 attempts=attempt + 1, res_u=sol["res_u"], res_heat=sol["res_heat"],
                 heat_mass_drift=sol["heat_mass_drift"], neg_cotan_frac=sol["neg_cotan_frac"],
                 t_mesh=t_mesh, t_total=time.time() - t0, params=params, source=fparams,
                 **{k: rep[k] for k in ("area", "volume", "edge_mean", "edge_median", "edge_min", "edge_max",
                                        "min_angle", "max_angle", "max_aspect", "p99_aspect", "valence_min",
                                        "valence_max", "euler")})
    return dict(pos=pts.astype(np.float32), faces=faces.astype(np.int32), f=f.astype(np.float32),
                u=sol["u"].astype(np.float32), heat=sol["heat"].astype(np.float32), stats=stats)


# ======================================================================================================================
# packing / driver
# ======================================================================================================================
def pack(samples: list[dict], meta: dict) -> dict:
    """Flat concatenation with offsets (``ptr0/ptr1/ptr2``), as ``gen_HP.pack``."""
    import torch
    st = [s["stats"] for s in samples]
    ptr = lambda n: torch.from_numpy(np.concatenate([[0], np.cumsum(n)]).astype(np.int64))  # noqa: E731
    out = {"ptr0": ptr([s["n0"] for s in st]), "ptr1": ptr([s["n1"] for s in st]),
           "ptr2": ptr([s["n2"] for s in st]), "meta": meta}
    for key in ("pos", "faces", "f", "u", "heat"):
        out[key] = torch.from_numpy(np.concatenate([s[key] for s in samples], 0))
    out["family"] = torch.tensor([FAMILIES.index(s["family"]) for s in st], dtype=torch.int64)
    out["genus"] = torch.tensor([s["genus"] for s in st], dtype=torch.int64)
    out["sample_stats"] = st
    return out


def generate(jobs: list, workers: int) -> list[dict]:
    if workers <= 1:
        return [make_sample(j) for j in jobs]
    with Pool(workers) as pool:
        return list(pool.imap(make_sample, jobs, chunksize=4))


def write_set(name: str, families: list[str], seed: int, cfg: dict, workers: int, out_dir: str) -> dict:
    """Generate and save ``<out_dir>/<name>.pt`` (+ ``.json`` summary); returns the summary."""
    import torch
    jobs = [(i, fam, seed, cfg) for i, fam in enumerate(families)]
    t0 = time.time()
    samples = generate(jobs, workers)
    t_gen = time.time() - t0
    meta = dict(task="SURF", set=name, N=len(samples), seed=seed, gen_seconds=t_gen, generator="datasets/generators/gen_surf.py",
                families=sorted(set(families)), family_codes=list(FAMILIES), config=cfg,
                pde="u: (M + eps L) u = M f; heat: 10 implicit Euler steps of (M + dt L) v' = M v, dt = T/10",
                operators="L = d0^T diag(cotan weights, unclamped) d0 (P1 FEM stiffness), M = barycentric lumped mass")
    path = os.path.join(out_dir, name + ".pt")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(pack(samples, meta), path)
    st = [s["stats"] for s in samples]
    n0 = np.array([s["n0"] for s in st])
    summary = dict(meta, file=path, file_MB=os.path.getsize(path) / 1e6, n0_mean=float(n0.mean()),
                   n0_min=int(n0.min()), n0_max=int(n0.max()),
                   edge_median_mean=float(np.mean([s["edge_median"] for s in st])),
                   edge_median_range=[float(min(s["edge_median"] for s in st)),
                                      float(max(s["edge_median"] for s in st))],
                   min_angle=float(min(s["min_angle"] for s in st)), max_aspect=float(max(s["max_aspect"] for s in st)),
                   max_res_u=float(max(s["res_u"] for s in st)), max_res_heat=float(max(s["res_heat"] for s in st)),
                   max_heat_mass_drift=float(max(s["heat_mass_drift"] for s in st)),
                   neg_cotan_frac_mean=float(np.mean([s["neg_cotan_frac"] for s in st])),
                   family_counts={f: int(sum(s["family"] == f for s in st)) for f in sorted(set(families))},
                   max_attempts=int(max(s["attempts"] for s in st)),
                   u_std=float(np.std(np.concatenate([s["u"] for s in samples]))),
                   heat_std=float(np.std(np.concatenate([s["heat"] for s in samples]))),
                   f_std=float(np.std(np.concatenate([s["f"] for s in samples]))))
    summary.pop("config", None)
    json.dump(summary, open(path.replace(".pt", ".json"), "w"), indent=2)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5700, help="SURF samples (families a, c, d interleaved)")
    ap.add_argument("--n-geo", type=int, default=500, help="SURF_geo samples (superquadrics)")
    ap.add_argument("--n-topo", type=int, default=500, help="SURF_topo samples (double tori)")
    ap.add_argument("--seed", type=int, default=2030)
    ap.add_argument("--h", type=float, default=0.07, help="target median edge length")
    ap.add_argument("--area-min", type=float, default=7.5)
    ap.add_argument("--area-max", type=float, default=15.0)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--T", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "datasets", "v2"))
    ap.add_argument("--prefix", default="SURF")
    a = ap.parse_args(argv)
    cfg = default_config(h=a.h, area_min=a.area_min, area_max=a.area_max, eps=a.eps, T=a.T)
    sets = [(a.prefix, [MAIN_FAMILIES[i % 3] for i in range(a.n)], a.seed),
            (a.prefix + "_geo", ["superquadric"] * a.n_geo, a.seed + 1),
            (a.prefix + "_topo", ["double_torus"] * a.n_topo, a.seed + 2)]
    out = {}
    for name, fams, seed in sets:
        if not fams:
            continue
        s = write_set(name, fams, seed, cfg, a.workers, a.out_dir)
        print(json.dumps(s, indent=2), flush=True)
        out[name] = s
    return out


if __name__ == "__main__":
    main()
