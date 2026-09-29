"""Adapters for the new synthetic variable-mesh tasks (generators in ``datasets/generators/gen_HP.py``, ``gen_TET.py``).

HP  (2-D, triangles):  inputs  f (nodes) | log sigma_e (edges, even) | log sigma_f (faces, even)
TET (3-D, tets):       inputs  f (nodes) | log sigma_e (edges, even) | log sigma_f (faces, even) | log sigma_t (tets)
targets: u at nodes (``node_scalar``) or, for the ``flux`` variants, the edge flux ``-sigma_e d0 u``
(orientation-odd ``cochain`` readout on degree 1); ``fluxd`` divides it by the edge length (flux density, resolution
independent: the flux cochain itself halves on a 2x finer mesh, which a scale-normalised model cannot anticipate).  ``_aniso`` HP sets use the directional conductivity
``sigma t^T A t`` along each edge as the edge input (E(n)-invariant).  Conductivities enter in log form (DESIGN §7:
log-domain for scale features) and are standardised like every other even input.

Legacy mode (``native=False``) moves everything to the nodes (f and the mean log sigma of the incident edges) so
that node-input models (v1) can run on the same samples.

Splits: sequential 70/15/15.  Normalisation: pooled per-feature statistics of the training samples (v1 style);
odd targets (flux) are only scaled.  The ``fine`` extra test set (4x nodes, same fields as the first test samples)
is normalised with the same statistics and kept on the CPU (moved per batch at evaluation time).
"""
from __future__ import annotations

import os
import time

import torch
from torch import Tensor

from rhmp.data import (Stats, TaskData, cell_alignment, edge_alignment, feature_stats, pack_to, scale_stats,
                       sequential_split)
from rhmp.tasks import TASK_DEFAULTS

# Variable-mesh complexes are kept on the GPU when their total size is below this (override: RHMP_GPU_BUDGET_GB,
# trainer --data-on), otherwise on the CPU; each block-diagonal batch is then moved with one packed, pinned copy
# per dtype (rhmp.data.pack_to).  Measured on HP_k100 (C=128, 8 meshes/step, GPU shared with other jobs): CPU-resident
# 18.5 s / 80 steps at 1.0 GB peak vs GPU-resident 17.6 s at 8.5 GB (7.5 GB of complexes without Whitney blocks), so
# the 4 GB default keeps HP/TET-size sets on the CPU and small sets (T8: 1.5 GB) on the GPU.
GPU_BUDGET_GB = float(os.environ.get("RHMP_GPU_BUDGET_GB", 4.0))


def _load_packed(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run the generators in datasets/generators (scripts/gen_datasets.sh)")
    return torch.load(path, map_location="cpu", weights_only=False)


def _slices(blob: dict, i: int, key: str, deg: int) -> Tensor:
    p = blob[f"ptr{deg}"]
    return blob[key][p[i]:p[i + 1]]


def _mesh_edges_torch(cells: Tensor, n0: int) -> Tensor:
    """Canonical lexicographic edges (as ``gen_common.mesh_edges``) of triangles/tets ``(m, p)``."""
    p = cells.shape[1]
    pairs = [(i, j) for i in range(p) for j in range(i + 1, p)]
    a = torch.cat([cells[:, i] for i, _ in pairs]).long()
    b = torch.cat([cells[:, j] for _, j in pairs]).long()
    key = torch.unique(torch.minimum(a, b) * n0 + torch.maximum(a, b))
    return torch.stack([key // n0, key % n0], 1)


def _tet_faces_torch(tets: Tensor, n0: int) -> Tensor:
    """Unique sorted triangles (lexicographic) of a tet mesh, as ``gen_common.mesh_faces_of_tets``."""
    t = tets.long()
    tri = torch.cat([t[:, [0, 1, 2]], t[:, [0, 1, 3]], t[:, [0, 2, 3]], t[:, [1, 2, 3]]])
    tri = torch.sort(tri, 1).values
    key = torch.unique((tri[:, 0] * n0 + tri[:, 1]) * n0 + tri[:, 2])
    return torch.stack([key // (n0 * n0), (key // n0) % n0, key % n0], 1)


def _complex_bytes(K) -> int:
    """All tensor bytes of a complex (operators, geometry, meta, Whitney blocks)."""
    from rhmp.data import tensor_nbytes
    return tensor_nbytes(K)


def _node_mean(values: Tensor, edges: Tensor, n0: int) -> Tensor:
    """Mean of edge values over the incident edges of every node: ``(n1, F) -> (n0, F)``."""
    s = torch.zeros(n0, values.shape[1]).index_add_(0, edges[:, 0], values).index_add_(0, edges[:, 1], values)
    c = torch.zeros(n0).index_add_(0, edges[:, 0], torch.ones(len(edges))).index_add_(0, edges[:, 1],
                                                                                     torch.ones(len(edges)))
    return s / c.clamp(min=1)[:, None]


def _build_set(blob: dict, *, dim: int, star: str, flux, aniso: bool, native: bool, ids=None, whitney: bool = True):
    """Complexes (CPU) and raw per-sample inputs/targets aligned to the complexes.

    ``whitney=False`` drops each complex's Whitney blocks right after it is built (only tensor metrics use them; this
    keeps the host-memory peak at ~60 % for triangles and ~30 % for tetrahedra).

    Returns:
        Ks (list), inputs (list of {k: (n_k, F)}), targets (list of (n_t, 1)), build info.
    """
    from rhmp.complex import CochainComplex
    N = len(blob["ptr0"]) - 1
    ids = range(N) if ids is None else ids
    Ks, inputs, targets = [], [], []
    t0 = time.time()
    top_key = "faces" if dim == 2 else "tets"
    for i in ids:
        pos = _slices(blob, i, "pos", 0).double()
        top = _slices(blob, i, top_key, dim).long()
        n0 = pos.shape[0]
        if dim == 2:
            K = CochainComplex.from_triangles(pos, top, star=star, device="cpu")
        else:
            K = CochainComplex.from_tetrahedra(pos, top, star=star, device="cpu")
        if K.n[dim] != top.shape[0]:            # the generators never produce degenerate cells
            raise RuntimeError(f"sample {i}: the complex dropped {top.shape[0] - K.n[dim]} cells")
        if not whitney:
            K.whitney = {}
        gen_edges = _mesh_edges_torch(top, n0)
        eperm, esign = edge_alignment(K.cells[1], gen_edges, n0)
        f = _slices(blob, i, "f", 0)[:, None]
        tensor = "sigma_proj_edge" in blob
        e_key = "sigma_proj_edge" if tensor else ("sigma_dir_edge" if aniso else "sigma_edge")
        ls_e = torch.log(_slices(blob, i, e_key, 1))[eperm][:, None]
        if dim == 2:
            fperm, _ = cell_alignment(K.cells[2], top, n0, oriented=False)
        else:
            fperm, _ = cell_alignment(K.cells[2], _tet_faces_torch(top, n0), n0, oriented=False)
        if tensor:                                  # tensor invariants (log det, log anisotropy ratio)
            ls_f = torch.stack([_slices(blob, i, "logdet_face", 2), _slices(blob, i, "logratio_face", 2)], 1)[fperm]
        else:
            ls_f = _slices(blob, i, "logsigma_face", 2)[fperm][:, None]
        if native:
            x = {0: f, 1: ls_e, 2: ls_f}
            if dim == 3:
                tperm, _ = cell_alignment(K.cells[3], top, n0, oriented=False)
                x[3] = _slices(blob, i, "logsigma_tet", 3)[tperm][:, None]
        else:
            x = {0: torch.cat([f, _node_mean(ls_e, K.cells[1].long(), n0)], 1)}
        e = K.cells[1].long()
        if flux == "grad":                          # E = -d0 u in the orientation of the complex
            u = _slices(blob, i, "u", 0)
            y = -(u[e[:, 1]] - u[e[:, 0]])[:, None]
        elif flux == "fluxfem":
            if "flux_fem" not in blob:
                raise KeyError("dataset has no flux_fem: regenerate it with the current generators in datasets/generators")
            y = (_slices(blob, i, "flux_fem", 1)[eperm] * esign)[:, None]
        elif flux:
            y = (_slices(blob, i, "flux", 1)[eperm] * esign)[:, None]
            if flux == "fluxd":                     # flux density: resolution independent
                ev = pos[e[:, 1]] - pos[e[:, 0]]
                y = y / ev.norm(dim=1, keepdim=True).to(y.dtype)
        else:
            y = _slices(blob, i, "u", 0)[:, None]
        Ks.append(K)
        inputs.append(x)
        targets.append(y)
    info = dict(build_seconds=time.time() - t0, n=len(Ks), mean_MB=sum(_complex_bytes(K) for K in Ks[:20]) /
                min(20, len(Ks)) / 1e6)
    return Ks, inputs, targets, info


def _load_variable(name: str, droot: str, base: str, *, dim: int, flux, aniso: bool, native: bool, device,
                   star: str, fine: bool, max_samples, keep_on, meta_extra: dict, whitney: bool = True) -> TaskData:
    path = os.path.join(droot, "v2", base + ".pt")
    blob = _load_packed(path)
    N = len(blob["ptr0"]) - 1
    split = sequential_split(N)
    ids = None
    if max_samples is not None and max_samples < N:          # smoke tests: a few samples of every split part
        per = max(1, max_samples // 3)
        parts = [s[:per] for s in split]                       # a part may be shorter than ``per`` (val/test)
        ids = torch.cat(parts).tolist()
        offsets = [0, len(parts[0]), len(parts[0]) + len(parts[1])]
        split = tuple(torch.arange(offsets[j], offsets[j] + len(parts[j])) for j in range(3))
    # Whitney blocks are only needed by tensor metrics (41 % of an HP complex, 70 % of a tet complex): dropped
    # per complex while building when not requested
    Ks, xs, ys, info = _build_set(blob, dim=dim, star=star, flux=flux, aniso=aniso, native=native, ids=ids,
                                  whitney=whitney)
    tr = split[0].tolist()
    degs = sorted(xs[0])
    x_stats = {k: feature_stats([xs[i][k] for i in tr]) for k in degs}
    y_stats = scale_stats([ys[i] for i in tr]) if flux else feature_stats([ys[i] for i in tr])
    est_gb = info["mean_MB"] * len(Ks) / 1e3
    on = keep_on if keep_on != "auto" else (device if est_gb < GPU_BUDGET_GB else "cpu")
    # one packed copy per dtype (per-tensor copies of ~200K small tensors are very slow on a shared GPU)
    Ks, inputs, target = pack_to((Ks, [{k: x_stats[k].normalize(x[k]) for k in degs} for x in xs],
                                  [y_stats.normalize(y) for y in ys]), on)
    extra = {}
    if fine:
        fpath = os.path.join(droot, "v2", base + "_fine.pt")
        if os.path.exists(fpath):
            fb = _load_packed(fpath)
            nf = len(fb["ptr0"]) - 1
            fids = range(min(nf, max_samples)) if max_samples is not None else None
            fK, fx, fy, finfo = _build_set(fb, dim=dim, star=star, flux=flux, aniso=aniso, native=native, ids=fids,
                                           whitney=whitney)
            extra["fine"] = dict(K=fK, inputs=[{k: x_stats[k].normalize(x[k]) for k in degs} for x in fx],
                                 target=[y_stats.normalize(y) for y in fy], info=finfo,
                                 coarse_ids=fb["meta"].get("coarse_ids"), fine_factor=fb["meta"].get("fine_factor"))
    in_dims = {k: int(xs[0][k].shape[-1]) for k in degs}
    even_dims = {k: in_dims[k] for k in degs if k >= 1}
    meta = dict(TASK_DEFAULTS["HP" if dim == 2 else "TET"])
    meta = dict(v1_C=meta["C"], batch_size=meta["batch"], layers=meta["layers"], source=path,
                generator=blob["meta"], complexes=dict(info, estimated_GB=est_gb, stored_on=str(on)),
                r2_std=y_stats.std, star=star, target=flux or "u", whitney=whitney,
                sample_ids=list(range(N)) if ids is None else list(ids), **meta_extra)
    if not flux:                          # -div(sigma grad u) = f, u = 0 on K.boundary[0]; P1, lumped mass = star_0
        meta["pde"] = dict(degree=0, source="inputs[0][...,0]", bc="dirichlet")   # trainer --aux-pde
    return TaskData(name=name, K=Ks, inputs=inputs, target=target, target_degree=1 if flux else 0,
                    target_kind="cochain" if flux else "node_scalar", in_dims=in_dims, even_dims=even_dims,
                    connection_dims={}, split=split, x_stats={k: s.to(device) for k, s in x_stats.items()},
                    y_stats=y_stats.to(device), spatial_dim=dim, out_dim=1, native=native, meta=meta,
                    extra_tests=extra, readout="grad" if flux == "grad" else None)   # grad: E = d0 phi exactly


def load_hp(name: str, droot: str, *, flux, kappa: int, aniso, native: bool, device, star: str,
            fine: bool = True, max_samples: int | None = None, keep_on: str = "auto", whitney: bool = True,
            **_) -> TaskData:
    """Hetero(-anisotropic) 2-D Poisson (``datasets/v2/HP_k<kappa>[_aniso|_aniso<R>].pt``).

    ``aniso``: False (isotropic), True (legacy constant-ratio set ``_aniso``) or the maximal eigenvalue ratio R of
    the tensor sets ``_aniso<R>``."""
    base = f"HP_k{kappa}" + ("" if aniso is False else ("_aniso" if aniso is True else f"_aniso{int(aniso)}"))
    return _load_variable(name, droot, base, dim=2, flux=flux, aniso=aniso is not False, native=native,
                          device=device, star=star, fine=fine, max_samples=max_samples, keep_on=keep_on,
                          meta_extra=dict(kappa=kappa, aniso=aniso), whitney=whitney)


def load_tet(name: str, droot: str, *, flux, kappa: int, native: bool, device, star: str, fine: bool = True,
             max_samples: int | None = None, keep_on: str = "auto", whitney: bool = True, **_) -> TaskData:
    """3-D tetrahedral Poisson (``datasets/v2/TET_k<kappa>.pt``); ``cotan`` falls back to ``barycentric``."""
    star_used = "barycentric" if star == "cotan" else star
    return _load_variable(name, droot, f"TET_k{kappa}", dim=3, flux=flux, aniso=False, native=native,
                          device=device, star=star_used, fine=fine, max_samples=max_samples, keep_on=keep_on,
                          meta_extra=dict(kappa=kappa, star_requested=star), whitney=whitney)
