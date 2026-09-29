"""Adapters for the v1 paper tasks T1, T2, T3, T5, T6, T7, T8 (+ T1q, T6f, T7f, T6_100K).

Legacy mode (``native=False``) reproduces v1's inputs/outputs and normalisation exactly (the v1 training script
``formal_benchmark.py``): node inputs ``X (N, n0, F)``, targets ``Y (N, n0, O)``, per-feature mean/std of the first
70 % (torch unbiased std; numpy biased std for T8), sequential 70/15/15 split.
Native mode puts inputs/outputs on the cells where the physics lives (see ``rhmp/tasks/__init__.py``).
"""
from __future__ import annotations

import json
import os
import pickle
import time
import warnings

import numpy as np
import torch
from torch import Tensor

from rhmp.data import (OutputMap, Stats, TaskData, cell_alignment, edge_alignment, feature_stats, holdout_split,
                       scale_stats, sequential_split)
from rhmp.tasks import TASK_DEFAULTS

V1_FILES = {
    "T1": ("T1_cns_vorticity.pkl", "T1_cns_vorticity"),
    "T2": ("T2_torus_advection_diffusion.pkl", "T2_torus_advection_diffusion"),
    "T3": ("T3_ellipsoid_coexact.pkl", "T3_ellipsoid_surface_flow"),
    "T5": ("T5_maxwell_poisson.pkl", "T5_maxwell_poisson"),
    "T6": ("T6_wilson_loop.pkl", "T6_wilson_loop"),
    "T7": ("T7_yang_mills_su2.pkl", "T7_yang_mills_su2"),
    "T8": ("T8_airfoil_pressure_persample.pkl", "T8_airfoil_pressure"),
    "T6_100K": ("T6_100K_wilson_loop.pkl", "T6_100K_wilson_loop"),
}
PAPER_EVAL_N = 100  # compute_all_metrics.py: first 100 samples of the test split


# ----------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------
def _read_pickle(droot: str, task: str) -> dict:
    path = os.path.join(droot, V1_FILES[task][0])
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: download the v1 datasets with python3 datasets/download_v1.py "
                                f"(see datasets/README.md)")
    with open(path, "rb") as f:
        return pickle.load(f)


def _planar(points: np.ndarray) -> np.ndarray:
    """Drop an all-zero third coordinate (2-D meshes stored with z = 0)."""
    if points.shape[1] == 3 and np.abs(points[:, 2]).max() == 0.0:
        return points[:, :2]
    return points


def _xy(d: dict) -> tuple[Tensor, Tensor]:
    X = torch.as_tensor(np.asarray(d["X_data"], dtype=np.float32))
    Y = torch.as_tensor(np.asarray(d["Y_data"], dtype=np.float32))
    if X.dim() == 2:
        X = X.unsqueeze(-1)
    if Y.dim() == 2:
        Y = Y.unsqueeze(-1)
    return X, Y


def _complex(points: np.ndarray, faces: np.ndarray, star: str, device):
    from rhmp.complex import CochainComplex
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        K = CochainComplex.from_triangles(torch.as_tensor(points, dtype=torch.float64),
                                          torch.as_tensor(np.asarray(faces, dtype=np.int64)),
                                          star=star, device=device)
    if K.n[2] != len(faces):
        raise RuntimeError(f"complex dropped {len(faces) - K.n[2]} faces; face-indexed data would misalign")
    return K


def _paper_eval(N: int) -> Tensor:
    nt, nv = int(0.7 * N), int(0.15 * N)
    return torch.arange(nt + nv, min(nt + nv + PAPER_EVAL_N, N))


def _edge_one_form(V: Tensor, pos: Tensor, edges: Tensor) -> Tensor:
    """Midpoint-rule line integral of a node vector field along oriented edges.

    Args:
        V: ``(N, n0, D)`` node vectors; pos ``(n0, D)``; edges ``(n1, 2)`` (src, dst).
    Returns:
        ``(N, n1, 1)`` with ``V_e = (V_src + V_dst)/2 . (x_dst - x_src)``.
    """
    t = (pos[edges[:, 1]] - pos[edges[:, 0]]).to(V)
    Vm = 0.5 * (V[:, edges[:, 0]] + V[:, edges[:, 1]])
    return (Vm * t[None]).sum(-1, keepdim=True)


def oriented_face_to_node(K, out_dim: int) -> OutputMap:
    """Oriented face -> node average ``y_i = mean_{f ∋ i} sigma_f x_f`` for a planar 2-D complex.

    ``sigma_f = sign(signed area)`` (+1 for counter-clockwise faces): the physical orientation of the plane, which
    turns an odd face 2-cochain (flux / vorticity in the face's own orientation) into the node pseudo-scalar field of
    the v1 targets.  Model readout: ``cochain:2``.
    """
    from rhmp.ops import csr_from_coo
    F = K.cells[2].long().cpu()
    if (F < 0).any():
        raise ValueError("oriented_face_to_node: padded polygons not supported")
    P = K.pos.double().cpu()
    if P.shape[1] != 2:
        raise ValueError("oriented_face_to_node needs a planar complex with 2-D positions")
    Q = P[F]                                                    # (n2, m, 2), shoelace signed area
    sa = 0.5 * (Q[..., 0] * Q.roll(-1, 1)[..., 1] - Q.roll(-1, 1)[..., 0] * Q[..., 1]).sum(1)
    sigma = torch.sign(sa)
    if (sigma == 0).any():
        raise ValueError("degenerate face in oriented_face_to_node")
    n0, n2, m = K.n[0], K.n[2], F.shape[1]
    rows = F.reshape(-1)
    cols = torch.arange(n2).repeat_interleave(m)
    cnt = torch.bincount(rows, minlength=n0).double()
    vals = (sigma.repeat_interleave(m) / cnt[rows]).float()
    dev = K.device
    M = csr_from_coo(rows.to(dev), cols.to(dev), vals.to(dev), (n0, n2))
    MT = csr_from_coo(cols.to(dev), rows.to(dev), vals.to(dev), (n2, n0))
    ccw = int((sigma > 0).sum())
    return OutputMap("sparse", "cochain:2", out_dim, M=M, MT=MT,
                     description=f"oriented face->node mean (sigma=+1 for CCW; {ccw}/{n2} faces CCW)")


def direct_vector_map(K, model_readout: str = "grad") -> OutputMap:
    """Fixed edge -> vertex vector map ``v_i = sum_{e ∋ i} w_e t_e / deg_i`` (``t_e`` = unit vector src->dst, the same
    for both endpoints): the v1 edge-to-node average that *defines* the T5 targets (``w_e = -(d0 phi)_e``, the raw
    potential differences).  Composed with the ``grad`` readout (``w = d0 phi``) it represents the T5 targets exactly.
    """
    from rhmp.ops import csr_from_coo
    e = K.cells[1].long().cpu()
    ev = K.edge_vectors().double().cpu()
    t = ev / ev.norm(dim=1, keepdim=True)
    n0, n1, D = K.n[0], K.n[1], ev.shape[1]
    deg = torch.bincount(e.reshape(-1), minlength=n0).double().clamp(min=1)
    ends = torch.cat([e[:, 0], e[:, 1]])                               # vertex of each (edge, endpoint)
    eid = torch.arange(n1).repeat(2)
    rows = (ends[:, None] * D + torch.arange(D)[None, :]).reshape(-1)
    cols = eid[:, None].expand(-1, D).reshape(-1)
    vals = (t[eid] / deg[ends][:, None]).reshape(-1).float()
    dev = K.device
    M = csr_from_coo(rows.to(dev), cols.to(dev), vals.to(dev), (D * n0, n1))
    MT = csr_from_coo(cols.to(dev), rows.to(dev), vals.to(dev), (n1, D * n0))
    return OutputMap("direct_vector", model_readout, 1, M=M, MT=MT, D=D,
                     description="edge -> vertex vectors v_i = sum_e w_e t_e / deg_i (v1 average)")


def _t5_poisson_check(d: dict, K, omap: OutputMap, idx, n: int = 4) -> dict:
    """``omap(-d0 phi)`` vs the stored T5 target, ``phi`` = the generator's Poisson solve (cotan Laplacian with
    lumped areas, ``(L + 1e-6 I) phi = areas rho``, area-mean removed) of the stored ``rho``."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import factorized
    P = np.asarray(d["points"], dtype=np.float64)[:, :2]
    Fc = np.asarray(d["faces"], dtype=np.int64)
    n0 = len(P)
    i, j, k = Fc[:, 0], Fc[:, 1], Fc[:, 2]
    eij, eki, ejk = P[j] - P[i], P[i] - P[k], P[k] - P[j]
    area = np.abs(eij[:, 0] * (-eki[:, 1]) - eij[:, 1] * (-eki[:, 0])) / 2
    ok = area >= 1e-12

    def cot2d(a, b):
        return (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]) / (np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) + 1e-12)
    rows, cols, vals = [], [], []
    for a_, b_, cv in ((j, k, cot2d(eij, -eki)), (k, i, cot2d(ejk, -eij)), (i, j, cot2d(eki, -ejk))):
        a_, b_, w = a_[ok], b_[ok], cv[ok] / 2
        rows += [a_, b_, a_, b_]
        cols += [b_, a_, a_, b_]
        vals += [w, w, -w, -w]
    L = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n0, n0))
    areas = np.zeros(n0)
    for v in (i, j, k):
        np.add.at(areas, v[ok], area[ok] / 3)
    solve = factorized((L + 1e-6 * sp.eye(n0)).tocsc())
    e = K.cells[1].long().cpu().numpy()
    errs = []
    for s in list(idx)[:n]:
        rho = np.asarray(d["X_data"][s], dtype=np.float64).reshape(-1)
        phi = solve(areas * rho)
        phi -= np.sum(phi * areas) / np.sum(areas)
        w = torch.as_tensor(-(phi[e[:, 1]] - phi[e[:, 0]]), dtype=torch.float32)[:, None, None].to(K.device)
        v = omap(w)[:, 0].double().cpu().numpy()
        y = np.asarray(d["Y_data"][s], dtype=np.float64)
        errs.append(float(np.linalg.norm(v - y) / np.linalg.norm(y)))
    return {"max_rel_err_map_of_minus_d0_phi_vs_target": max(errs), "samples": len(errs)}


def surface_rotation(K, faces: np.ndarray, points: np.ndarray) -> OutputMap:
    """``v_i = n_i x g_i`` with the area-weighted vertex normals of the (consistently oriented) surface, exactly as
    the T3 generator builds its target ``v = n x grad psi``.  Model readout: ``node_vector`` (predicts ``g``)."""
    n = np.zeros_like(points)
    nm = np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]])
    for k in range(3):
        np.add.at(n, faces[:, k], nm)
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    return OutputMap("cross_normal", "node_vector", 1, normals=torch.as_tensor(n, dtype=torch.float32).to(K.device),
                     description="v = n x g (rotation by +90 deg about the outward vertex normal)")


def _finish(name, K, inputs_raw: dict, target_raw: Tensor, *, target_degree, target_kind, in_dims, even_dims,
            connection_dims, x_kinds: dict, y_kind: str, spatial_dim, native, device, meta, split=None,
            unbiased=True, y_v1: Tensor | None = None, output_map: OutputMap | None = None) -> TaskData:
    """Normalise with train-part statistics and move everything to ``device``.

    ``x_kinds[k]`` / ``y_kind``: ``'v1'`` (mean/std), ``'scale'`` (odd cochains: mean 0, RMS per column),
    ``'isotropic'`` (vector fields: mean 0, one RMS for all components).
    """
    N = target_raw.shape[0]
    split = split or sequential_split(N)
    tr = split[0]

    def stats(x, kind):
        if kind == "v1":
            return feature_stats(x[tr], unbiased=unbiased)
        if kind == "scale":
            return scale_stats(x[tr])
        if kind == "isotropic":
            return scale_stats(x[tr], isotropic=True)
        raise ValueError(kind)

    x_stats = {k: stats(v, x_kinds[k]) for k, v in inputs_raw.items()}
    y_stats = stats(target_raw, y_kind)
    inputs = {k: x_stats[k].normalize(v).to(device).contiguous() for k, v in inputs_raw.items()}
    target = y_stats.normalize(target_raw).to(device).contiguous()
    meta = dict(meta)
    v1_std = feature_stats((y_v1 if y_v1 is not None else target_raw)[tr], unbiased=unbiased).std
    meta.setdefault("r2_std", v1_std)
    meta.setdefault("paper_eval", _paper_eval(N))
    return TaskData(name=name, K=K, inputs=inputs, target=target, target_degree=target_degree,
                    target_kind=target_kind, in_dims=in_dims, even_dims=even_dims, connection_dims=connection_dims,
                    split=split, x_stats={k: s.to(device) for k, s in x_stats.items()}, y_stats=y_stats.to(device),
                    spatial_dim=spatial_dim, out_dim=int(target.shape[-1]), native=native, meta=meta,
                    output_map=output_map)


def _defaults(name: str) -> dict:
    d = TASK_DEFAULTS[name]
    return dict(v1_C=d["C"], batch_size=d["batch"], layers=d["layers"])


# ----------------------------------------------------------------------------------------------------------------
# T1 / T1q: CNS vorticity on the 32x32 grid
# ----------------------------------------------------------------------------------------------------------------
def load_T1(droot, *, native, device, star, quad=False, output_map=None, **_):
    """T1 CNS vorticity (32x32 grid; ``quad=True``: quad CW complex T1q).  Legacy: (rho,Vx,Vy,p) nodes -> vorticity
    nodes.  Native: (rho,p) nodes + velocity 1-form on edges; ``cochain:2`` readout + oriented face->node map."""
    from rhmp.complex import CochainComplex
    d = _read_pickle(droot, "T1")
    pts = np.asarray(d["points"], dtype=np.float64)
    X, Y = _xy(d)
    gs = int(d.get("grid_size", 32))
    name = "T1q" if quad else "T1"
    if quad:
        K = CochainComplex.from_grid((gs, gs), 1.0 / (gs - 1), cell="quad", star=star, device=device)
        # vertex id i*ny+j at (i h, j h) is the v1 ordering (meshgrid indexing='ij'); checked here
        if not torch.allclose(K.pos.double().cpu(), torch.as_tensor(pts), atol=1e-6):
            raise RuntimeError("from_grid vertex order does not match the T1 points")
    else:
        K = _complex(pts, d["faces"], star, device)
    meta = _defaults(name)
    meta["v1_name"] = V1_FILES["T1"][1]
    if not native:
        return _finish(name, K, {0: X}, Y, target_degree=0, target_kind="node_scalar", in_dims={0: 4},
                       even_dims={}, connection_dims={}, x_kinds={0: "v1"}, y_kind="v1", spatial_dim=2,
                       native=False, device=device, meta=meta)
    edges = K.cells[1].cpu()
    V1form = _edge_one_form(X[..., 1:3], torch.as_tensor(pts, dtype=torch.float32), edges)
    meta["native_inputs"] = "node (rho, p); edge velocity 1-form (V_s+V_d)/2.(x_d-x_s)"
    omap = oriented_face_to_node(K, 1) if output_map is not False else None     # vorticity = pseudo-scalar
    return _finish(name, K, {0: X[..., [0, 3]].contiguous(), 1: V1form}, Y, target_degree=0,
                   target_kind="node_scalar", in_dims={0: 2, 1: 1}, even_dims={}, connection_dims={},
                   x_kinds={0: "v1", 1: "scale"}, y_kind="v1", spatial_dim=2, native=True, device=device, meta=meta,
                   output_map=omap)


# ----------------------------------------------------------------------------------------------------------------
# T2: torus advection-diffusion (scalar -> scalar)
# ----------------------------------------------------------------------------------------------------------------
def load_T2(droot, *, native, device, star, **_):
    """T2 torus advection-diffusion: scalar ``(N, 1711, 1)`` -> scalar at nodes (legacy == native)."""
    d = _read_pickle(droot, "T2")
    pts = np.asarray(d["points"], dtype=np.float64)
    X, Y = _xy(d)
    K = _complex(pts, d["faces"], star, device)
    meta = _defaults("T2")
    meta["v1_name"] = V1_FILES["T2"][1]
    return _finish("T2", K, {0: X}, Y, target_degree=0, target_kind="node_scalar", in_dims={0: 1}, even_dims={},
                   connection_dims={}, x_kinds={0: "v1"}, y_kind="v1", spatial_dim=3, native=native,
                   device=device, meta=meta)


# ----------------------------------------------------------------------------------------------------------------
# T3: ellipsoid coexact flow (psi -> tangent vector field)
# ----------------------------------------------------------------------------------------------------------------
def load_T3(droot, *, native, device, star, output_map=None, **_):
    """T3 ellipsoid coexact flow psi -> v (node_vector).  Native: isotropic target scaling, the model predicts
    g = -n x v (true vector) and v = n x g is reported; legacy: v1 per-component normalisation, direct readout."""
    d = _read_pickle(droot, "T3")
    pts = np.asarray(d["points"], dtype=np.float64)
    X, Y = _xy(d)
    K = _complex(pts, d["faces"], star, device)
    meta = _defaults("T3")
    meta["v1_name"] = V1_FILES["T3"][1]
    meta["vector_mode"] = "ls" if native else "direct"
    # v = n x grad(psi) is a pseudo-vector: the model predicts the true vector g = -n x v, reported as v = n x g
    omap = surface_rotation(K, np.asarray(d["faces"], dtype=np.int64), pts) \
        if (native and output_map is not False) else None
    return _finish("T3", K, {0: X}, Y, target_degree=0, target_kind="node_vector", in_dims={0: 1}, even_dims={},
                   connection_dims={}, x_kinds={0: "v1"}, y_kind="isotropic" if native else "v1", spatial_dim=3,
                   native=native, device=device, meta=meta, output_map=omap)


# ----------------------------------------------------------------------------------------------------------------
# T5: Maxwell / Poisson (rho -> E)
# ----------------------------------------------------------------------------------------------------------------
def load_T5(droot, *, native, device, star, grad=False, name="T5", **_):
    """T5 Maxwell/Poisson rho -> E.  Legacy: 2 node scalars (v1).  Native: node_vector (direct), isotropic scaling."""
    d = _read_pickle(droot, "T5")
    pts = _planar(np.asarray(d["points"], dtype=np.float64))
    X, Y = _xy(d)
    K = _complex(pts, d["faces"], star, device)
    meta = _defaults(name)
    meta["v1_name"] = V1_FILES["T5"][1]
    if grad:                                   # T5g: predict a potential, E_e = d0 phi, fixed v1 edge->node average
        if not native:
            raise ValueError("T5g has no legacy variant")
        omap = direct_vector_map(K, "grad")
        meta["consistency"] = _t5_poisson_check(d, K, omap, _paper_eval(X.shape[0]))
        if meta["consistency"]["max_rel_err_map_of_minus_d0_phi_vs_target"] > 1e-3:
            raise RuntimeError(f"T5g: output map does not reproduce the generator: {meta['consistency']}")
        meta["native_inputs"] = "node rho; readout grad (E = d0 phi on edges) + fixed v1 edge->node average"
        return _finish(name, K, {0: X}, Y, target_degree=0, target_kind="node_vector", in_dims={0: 1},
                       even_dims={}, connection_dims={}, x_kinds={0: "v1"}, y_kind="isotropic", spatial_dim=2,
                       native=True, device=device, meta=meta, output_map=omap)
    if not native:
        return _finish("T5", K, {0: X}, Y, target_degree=0, target_kind="node_scalar", in_dims={0: 1},
                       even_dims={}, connection_dims={}, x_kinds={0: "v1"}, y_kind="v1", spatial_dim=2,
                       native=False, device=device, meta=meta)
    # the generator defines E_i = mean_{e ∋ i} E_e t_e, i.e. the 'direct' vector readout
    meta["vector_mode"] = "direct"
    return _finish("T5", K, {0: X}, Y, target_degree=0, target_kind="node_vector", in_dims={0: 1}, even_dims={},
                   connection_dims={}, x_kinds={0: "v1"}, y_kind="isotropic", spatial_dim=2, native=True,
                   device=device, meta=meta)


# ----------------------------------------------------------------------------------------------------------------
# T6 / T6f / T6_100K: U(1) Wilson loop
# ----------------------------------------------------------------------------------------------------------------
def _native_file(droot: str, task: str) -> dict:
    path = os.path.join(droot, "v2", f"{task}_native.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run python3 -u datasets/generators/gen_{task}_native.py (see datasets/README.md)")
    return torch.load(path, map_location="cpu", weights_only=False)


def load_T6(droot, *, native, device, star, face_target=False, name="T6", output_map=None, **_):
    """T6 U(1) Wilson loop.  Legacy: node encoding ``(N, n0, 3)``.  Native: theta on edges in exact gauge-connection
    mode; node target through ``cochain:2`` + oriented face->node map, or face target (``face_target``: T6f)."""
    d = _read_pickle(droot, "T6")
    pts = _planar(np.asarray(d["points"], dtype=np.float64))
    faces = np.asarray(d["faces"], dtype=np.int64)
    X, Y = _xy(d)
    K = _complex(pts, faces, star, device)
    meta = _defaults(name)
    meta["v1_name"] = V1_FILES["T6"][1]
    if not native:
        if face_target:
            raise ValueError("T6f has no legacy variant (face targets need the native edge inputs)")
        return _finish("T6", K, {0: X}, Y, target_degree=0, target_kind="node_scalar", in_dims={0: 3},
                       even_dims={}, connection_dims={}, x_kinds={0: "v1"}, y_kind="v1", spatial_dim=2,
                       native=False, device=device, meta=meta)
    nat = _native_file(droot, "T6")
    perm, sign = edge_alignment(K.cells[1], nat["edges"], K.n[0])
    theta = (nat["theta_edges"][:, perm] * sign[None]).unsqueeze(-1)         # (N, n1, 1), K orientation
    fperm, fsign = cell_alignment(K.cells[2], nat["faces"], K.n[0])
    plaq = (nat["plaq"][:, fperm] * fsign[None]).unsqueeze(-1)              # (N, n2, 1), K orientation
    # consistency: d1 theta == plaq (both in the orientation of the complex)
    th = theta[:16].permute(1, 0, 2).contiguous().to(K.device)
    err = (K.apply_d(1, th).permute(1, 0, 2).cpu() - plaq[:16]).abs().max().item()
    meta["consistency"] = {"d1_theta_minus_plaq_maxabs": err, "generator": nat["meta"]}
    if err > 1e-4:
        raise RuntimeError(f"T6 native alignment failed: max|d1 theta - plaq| = {err:.3e}")
    meta["native_inputs"] = "theta on edges (U(1) connection, exact gauge mode)"
    if face_target:
        return _finish(name, K, {1: theta}, plaq, target_degree=2, target_kind="cochain", in_dims={1: 1},
                       even_dims={}, connection_dims={1: 1}, x_kinds={1: "scale"}, y_kind="scale", spatial_dim=2,
                       native=True, device=device, meta=meta)
    omap = oriented_face_to_node(K, 1) if output_map is not False else None       # node flux = pseudo-scalar
    return _finish(name, K, {1: theta}, Y, target_degree=0, target_kind="node_scalar", in_dims={1: 1},
                   even_dims={}, connection_dims={1: 1}, x_kinds={1: "scale"}, y_kind="v1", spatial_dim=2,
                   native=True, device=device, meta=meta, output_map=omap)


def load_T6_100K(droot, *, native, device, star, **_):
    """T6 at 100K faces (legacy node encoding only; step benchmark)."""
    d = _read_pickle(droot, "T6_100K")
    pts = _planar(np.asarray(d["points"], dtype=np.float64))
    X, Y = _xy(d)
    K = _complex(pts, d["faces"], star, device)
    meta = _defaults("T6_100K")
    meta["v1_name"] = V1_FILES["T6_100K"][1]
    if native:
        meta["note"] = "no raw edge fields for T6_100K; legacy node encoding used"
    return _finish("T6_100K", K, {0: X}, Y, target_degree=0, target_kind="node_scalar", in_dims={0: 3},
                   even_dims={}, connection_dims={}, x_kinds={0: "v1"}, y_kind="v1", spatial_dim=2, native=False,
                   device=device, meta=meta)


# ----------------------------------------------------------------------------------------------------------------
# T7 / T7f: SU(2) Yang-Mills
# ----------------------------------------------------------------------------------------------------------------
def load_T7(droot, *, native, device, star, face_target=False, name="T7", connection=True, output_map=None, **_):
    """T7 SU(2) Yang-Mills.  Legacy: node encoding ``(N, n0, 9)``.  Native: A on edges (3 odd columns, connection +
    odd path); node target via ``cochain:2`` + oriented face->node map, or face target F (T7f)."""
    d = _read_pickle(droot, "T7")
    pts = _planar(np.asarray(d["points"], dtype=np.float64))
    faces = np.asarray(d["faces"], dtype=np.int64)
    X, Y = _xy(d)
    K = _complex(pts, faces, star, device)
    meta = _defaults(name)
    meta["v1_name"] = V1_FILES["T7"][1]
    if not native:
        if face_target:
            raise ValueError("T7f has no legacy variant")
        return _finish("T7", K, {0: X}, Y, target_degree=0, target_kind="node_scalar", in_dims={0: 9},
                       even_dims={}, connection_dims={}, x_kinds={0: "v1"}, y_kind="v1", spatial_dim=2,
                       native=False, device=device, meta=meta)
    nat = _native_file(droot, "T7")
    perm, sign = edge_alignment(K.cells[1], nat["edges"], K.n[0])
    A = nat["A_edges"][:, perm] * sign[None, :, None]                        # (N, n1, 3)
    fperm, fsign = cell_alignment(K.cells[2], nat["faces"], K.n[0])
    F = nat["F_faces"][:, fperm] * fsign[None, :, None]                      # (N, n2, 3)
    del nat["A_edges"], nat["F_faces"]
    # consistency: the abelian part d1 A agrees with F up to the commutator (|dA| / |[A,A]| ~ 5.7)
    a = A[:8].permute(1, 0, 2).contiguous().to(K.device)
    dA = K.apply_d(1, a).permute(1, 0, 2).cpu()
    rel = ((dA - F[:8]).norm() / F[:8].norm()).item()
    meta["consistency"] = {"rel_norm_F_minus_dA": rel, "generator": nat["meta"]}
    if rel > 0.5:
        raise RuntimeError(f"T7 native alignment failed: |F - d1 A| / |F| = {rel:.3f}")
    meta["native_inputs"] = "su(2) connection A on edges (3 odd columns)"
    cdims = {1: 3} if connection else {}
    if face_target:
        return _finish(name, K, {1: A}, F, target_degree=2, target_kind="cochain", in_dims={1: 3}, even_dims={},
                       connection_dims=cdims, x_kinds={1: "scale"}, y_kind="scale", spatial_dim=2, native=True,
                       device=device, meta=meta)
    omap = oriented_face_to_node(K, 3) if output_map is not False else None       # node F = pseudo-scalars
    return _finish(name, K, {1: A}, Y, target_degree=0, target_kind="node_scalar", in_dims={1: 3}, even_dims={},
                   connection_dims=cdims, x_kinds={1: "scale"}, y_kind="v1", spatial_dim=2, native=True,
                   device=device, meta=meta, output_map=omap)


# ----------------------------------------------------------------------------------------------------------------
# T8: AirfRANS pressure, one mesh per sample
# ----------------------------------------------------------------------------------------------------------------
def complex_code_version() -> str:
    """Hash of the complex/geometry implementation (cached complexes are rebuilt when it changes)."""
    import hashlib
    import rhmp.complex
    import rhmp.geometry
    h = hashlib.sha1()
    for mod in (rhmp.complex, rhmp.geometry):
        with open(mod.__file__, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def _t8_complexes(droot: str, samples: list, star: str, cache: bool):
    """Build (or load from ``datasets/v2/T8_complexes_<star>.pt``) the CPU complexes of all T8 samples.

    The cache is keyed on the number of samples and on :func:`complex_code_version` (a hash of ``rhmp/complex.py``
    and ``rhmp/geometry.py``), so it is rebuilt automatically (about 8 s) whenever the complex implementation changes.
    """
    from rhmp.complex import CochainComplex
    path = os.path.join(droot, "v2", f"T8_complexes_{star}.pt")
    version = complex_code_version()
    if cache and os.path.exists(path):
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001 - unpickling an outdated class layout
            blob = {}
        if blob.get("n") == len(samples) and blob.get("version") == version:
            return blob["complexes"], dict(blob.get("info", {}), cache="hit", path=path)
    t0 = time.time()
    Ks = []
    for s in samples:
        pts = _planar(np.asarray(s["pts_3d"], dtype=np.float64))
        Ks.append(CochainComplex.from_triangles(torch.as_tensor(pts), torch.as_tensor(np.asarray(s["faces"])),
                                                star=star, device="cpu"))
    info = dict(build_seconds=time.time() - t0, n=len(Ks))
    if cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"n": len(Ks), "version": version, "complexes": Ks, "info": info}, path)
        info = dict(info, cache="written", path=path, file_MB=os.path.getsize(path) / 1e6)
    return Ks, info


def _t8_inflow_forms(Ks, x_all):
    """Uniform inflow ``V = U (cos a, sin a)`` of every T8 sample as an odd edge 1-form ``theta_e = V . (x_d - x_s)``.

    ``x_pres`` columns are ``(sdf, U, a)`` with ``U`` (m/s) and the angle of attack ``a`` (radians, AirfRANS) constant
    per sample; the positions are in the airfoil frame (chord along +x), the frame ``a`` is measured in.  For a
    uniform field the midpoint rule ``(V_s + V_d)/2 . (x_d - x_s)`` is exact.

    Returns:
        list of ``(n1_i, 1)`` float32 tensors (edge order and orientation of the complexes).
    """
    out = []
    for K, x in zip(Ks, x_all):
        U, a = x[:, 1].double(), x[:, 2].double()
        if float(U.max() - U.min()) > 1e-4 * float(U.abs().max()) or float(a.max() - a.min()) > 1e-6:
            raise RuntimeError("T8: inlet velocity / angle of attack are not constant within a sample")
        V = torch.stack([U[0] * torch.cos(a[0]), U[0] * torch.sin(a[0])])
        ev = K.edge_vectors().double().cpu()
        out.append((ev @ V).float()[:, None])
    return out


def _t8_inflow_check(samples, n: int = 50) -> dict:
    """Consistency of the inflow convention with the stored flow field: far-field (top 10 % sdf) mean velocity of
    ``x_vort[:, :2]`` vs ``U (cos a, sin a)`` (a in radians) on the first ``n`` samples."""
    rel, cosang = [], []
    for s in samples[:n]:
        xp, xv = np.asarray(s["x_pres"], dtype=np.float64), np.asarray(s.get("x_vort", []), dtype=np.float64)
        if xv.ndim != 2 or xv.shape[1] < 2:
            return {"available": False}
        U, a = xp[0, 1], xp[0, 2]
        far = xp[:, 0] > np.quantile(xp[:, 0], 0.9)
        v = xv[far, :2].mean(0)
        d = np.array([np.cos(a), np.sin(a)])
        rel.append(float(np.linalg.norm(v - U * d) / U))
        cosang.append(float(v @ d / np.linalg.norm(v)))
    return {"available": True, "far_field_rel_err_median": float(np.median(rel)), "cos_angle_min": float(min(cosang))}


def load_T8(droot, *, native, device, star, cache=True, max_samples=None, whitney=True, inflow=False, name="T8",
            **_):
    """T8 AirfRANS pressure, one mesh per sample: list of complexes (cached), v1 70/30 split with the last 10 % of
    the train part as validation set, numpy-style (biased) normalisation from the first 700 samples as v1.

    ``inflow=True`` (task ``T8v``): the node inputs ``(sdf, U, a)`` are kept and the inflow velocity is added as an
    odd edge 1-form ``theta_e = V_inf . (x_d - x_s)`` (scale-normalised with the RMS of the first 700 samples, like
    the node statistics).  A reflection-invariant model cannot tell the pressure side from the suction side from the
    scalar angle ``a`` alone (a symmetric airfoil at ``a != 0`` has mirror-symmetric geometry but no mirror-symmetric
    pressure); the 1-form carries the flow direction equivariantly.
    """
    samples = _read_pickle(droot, "T8")
    N_all = len(samples)
    Ks, info = _t8_complexes(droot, samples, star, cache)
    x_all = [torch.as_tensor(np.asarray(s["x_pres"], dtype=np.float32)) for s in samples]
    y_all = [torch.as_tensor(np.asarray(s["y_pres"], dtype=np.float32)) for s in samples]
    x_all = [x[:, None] if x.dim() == 1 else x for x in x_all]
    y_all = [y[:, None] if y.dim() == 1 else y for y in y_all]
    for K, x in zip(Ks, x_all):
        if K.n[0] != x.shape[0]:
            raise RuntimeError("T8 complex / sample size mismatch")
    # v1 normalisation: numpy (biased) statistics over the first int(0.7 N) samples (formal_benchmark.py)
    n_v1_train = int(0.7 * N_all)
    x_stats = feature_stats(x_all[:n_v1_train], unbiased=False)
    y_stats = feature_stats(y_all[:n_v1_train], unbiased=False)
    e_all, e_stats, inflow_check = None, None, None
    if inflow:
        if not native:
            raise ValueError("T8v has no legacy variant (the inflow 1-form is a native cochain input)")
        e_all = _t8_inflow_forms(Ks, x_all)
        e_stats = scale_stats(e_all[:n_v1_train])
        inflow_check = _t8_inflow_check(samples)
        if inflow_check.get("available") and inflow_check["cos_angle_min"] < 0.99:
            raise RuntimeError(f"T8v: inflow direction disagrees with the stored flow field: {inflow_check}")
    split = holdout_split(N_all)
    idx_keep = None
    if max_samples is not None and max_samples < N_all:     # smoke tests: keep a few samples of every part
        per = max(1, max_samples // 3)
        idx_keep = torch.cat([s[:per] for s in split])
        remap = {int(i): j for j, i in enumerate(idx_keep.tolist())}
        split = tuple(torch.tensor([remap[int(i)] for i in s[:per].tolist()]) for s in split)
        Ks = [Ks[i] for i in idx_keep.tolist()]
        x_all = [x_all[i] for i in idx_keep.tolist()]
        y_all = [y_all[i] for i in idx_keep.tolist()]
        if e_all is not None:
            e_all = [e_all[i] for i in idx_keep.tolist()]
    if not whitney:                              # only tensor metrics use the Whitney blocks
        from rhmp.data import strip_whitney
        strip_whitney(Ks)
    from rhmp.data import pack_to            # one packed copy per dtype instead of ~40 per complex
    if e_all is None:
        raw_inputs = [{0: x_stats.normalize(x)} for x in x_all]
    else:
        raw_inputs = [{0: x_stats.normalize(x), 1: e_stats.normalize(e)} for x, e in zip(x_all, e_all)]
    Ks, inputs, target = pack_to((Ks, raw_inputs, [y_stats.normalize(y) for y in y_all]), device)
    meta = _defaults(name)
    meta.update(v1_name=V1_FILES["T8"][1], complexes=info, r2_std=y_stats.std,
                split_note="v1 70/30 (first 700 train, last 300 test); v2 selects on the last 10% of the train "
                           "part (630..699), v1 selected on the test part; normalisation from the first 700 (v1)",
                paper_eval=(torch.arange(850, 950) if idx_keep is None else split[2]))
    in_dims = {0: int(x_all[0].shape[-1])}
    x_stats_d = {0: x_stats.to(device)}
    if e_all is not None:
        in_dims[1] = 1
        x_stats_d[1] = e_stats.to(device)
        meta.update(native_inputs="node (sdf, U, angle of attack); edge inflow 1-form V_inf . (x_d - x_s)",
                    consistency={"inflow": inflow_check})
    return TaskData(name=name, K=Ks, inputs=inputs, target=target, target_degree=0, target_kind="node_scalar",
                    in_dims=in_dims, even_dims={}, connection_dims={}, split=split, x_stats=x_stats_d,
                    y_stats=y_stats.to(device), spatial_dim=2, out_dim=int(y_all[0].shape[-1]), native=native,
                    meta=meta)


# ----------------------------------------------------------------------------------------------------------------
# dispatcher
# ----------------------------------------------------------------------------------------------------------------
def load(name: str, droot: str, *, native: bool, device, star: str, **kw) -> TaskData:
    """Dispatch a paper task name to its loader."""
    if name == "T1":
        return load_T1(droot, native=native, device=device, star=star, **kw)
    if name == "T1q":
        return load_T1(droot, native=native, device=device, star=star, quad=True, **kw)
    if name == "T2":
        return load_T2(droot, native=native, device=device, star=star, **kw)
    if name == "T3":
        return load_T3(droot, native=native, device=device, star=star, **kw)
    if name == "T5":
        return load_T5(droot, native=native, device=device, star=star, **kw)
    if name == "T5g":
        return load_T5(droot, native=native, device=device, star=star, grad=True, name="T5g", **kw)
    if name == "T6":
        return load_T6(droot, native=native, device=device, star=star, **kw)
    if name == "T6f":
        return load_T6(droot, native=True, device=device, star=star, face_target=True, name="T6f", **kw)
    if name == "T7":
        return load_T7(droot, native=native, device=device, star=star, **kw)
    if name == "T7f":
        return load_T7(droot, native=True, device=device, star=star, face_target=True, name="T7f", **kw)
    if name == "T8":
        return load_T8(droot, native=native, device=device, star=star, **kw)
    if name == "T8v":
        return load_T8(droot, native=native, device=device, star=star, inflow=True, name="T8v", **kw)
    if name == "T6_100K":
        return load_T6_100K(droot, native=native, device=device, star=star, **kw)
    raise KeyError(name)
