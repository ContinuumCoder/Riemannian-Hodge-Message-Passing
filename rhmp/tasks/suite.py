"""Extension task suite: SURF (closed surfaces), DYN (advection-diffusion rollouts), QUAL (mesh-quality shift).

Datasets are produced by ``datasets/generators/gen_surf.py``, ``gen_dyn.py`` and ``gen_qual.py`` (details, equations and
sizes in ``docs/TASK_SUITE_DETAILS.md``).  Every loader returns a :class:`rhmp.data.TaskData` built exactly like the
HP/TET loaders of :mod:`rhmp.tasks.synthetic` (normalisation from the training part, ``node_scalar`` readout,
complexes on the GPU when they fit ``GPU_BUDGET_GB``):

==================  ==================================================================================================
name                content
==================  ==================================================================================================
SURF                screened Poisson ``(M + eps L) u = M f`` on variable closed surfaces (ellipsoids, SH-perturbed
                    spheres, perturbed tori), one mesh per sample; input f (nodes), target u (nodes), spatial_dim 3.
                    Sequential 70/15/15 split.  Extra test sets (same normalisation): ``geo`` (superquadrics,
                    geometry transfer), ``topo`` (genus-2 double tori, topology transfer), and the per-family
                    in-distribution test subsets ``test_ellipsoid``, ``test_sphere_pert``, ``test_torus``.
SURF_heat           same inputs/meshes, target = heat flow ``exp(T Lap) f`` (T = 0.05, 10 implicit Euler steps).
SURF_geo, SURF_topo test-only views of the extra sets (split = (), (), all) for ``rhmp.train --eval-ckpt``;
SURF_heat_geo, ...  likewise for the heat target.
DYN                 one-step windows ``(u_t, theta) -> u_{t+1}`` of advection-diffusion trajectories on random
                    planar meshes (one mesh per trajectory); inputs u_t (nodes) and the velocity 1-form theta
                    (edges, odd); target u_{t+1} (nodes).  Split by trajectory (70/15/15); windows at every
                    ``T / windows`` steps.  ``task.rollout`` holds the full test trajectories for
                    :func:`rollout_eval`.
DYNfix              the same on one fixed mesh (shared-mesh TaskData, batches of 64 windows).
DYN_delta,          as DYN / DYNfix with the increment u_{t+1} - u_t as target (one-step persistence has R2 ~0.99 on
DYNfix_delta        the state target; rollouts add the predicted increment).
DYNfix_cons         exactly conservative variant on the fixed mesh: target du = u_{t+1} - u_t (scale-only normalisation,
                    loss in u-space) predicted through a fixed conservative output map: default a finite-volume update
                    du = (mean M / M) d0^T (l* . b) from a learned edge flux density b (readout ``cochain:1``), or
                    (``cons_param='div'``) du_i = (mean M / M_i) (d0^T b)_i (readout ``div:1``); in both cases
                    sum_i M_i du_i = 0 exactly for any weights.
DYN_cons            the same on variable meshes, with the model readout ``mdiv:1`` that divides the divergence by the
                    lumped mass inside the model (``rhmp.readout.MassDivReadout``).
DYN_cons_mass,      the first conservative convention (kept to evaluate the runs trained with it): target M du (the
DYNfix_cons_mass    mass change per dual cell) with the plain ``div:1`` readout.  Ill-conditioned for rollouts: the
                    node-uniform loss on M du leaves errors ~1/M_i in du (M varies ~900:1 on the DYN meshes), which
                    explode in a few autoregressive steps.
HP_qual_graded      HP_k100 physics on graded meshes (corner / circle refinement) and the original meshes, test only;
HP_qual_sliver      ... on sliver meshes (Delaunay of anisotropically scaled point clouds); one extra test set per
                    quality level (``base``, ``corner_r16``, ``circle_a30``, ``a8``, ...) -> accuracy-vs-quality curves.
HP_qual_*_ref       target = reference solution from the 4x-finer HP mesh interpolated to the nodes (instead of the
                    P1 solution on the quality-shifted mesh itself).
==================  ==================================================================================================

Test-only tasks use statistics of their own samples; ``python3 scripts/eval_on.py --run RUN --task HP_qual_sliver``
(or ``python3 -m rhmp.train --task HP_qual_sliver --eval-ckpt RUN`` once these names are registered in
``rhmp.tasks.load_task``) re-expresses inputs/targets in the normalisation of the training run before evaluating
(``train._renormalize``).

Rollouts: ``rollout_eval(model, task, steps)`` feeds the model's (denormalised) prediction back as the next input and
reports R2 / NRMSE per horizon and the drift of the discrete mass ``sum_i M_i u_i`` (lumped mass of the generator).
"""
from __future__ import annotations

import dataclasses
import functools
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch import Tensor

from rhmp.data import Stats, TaskData, edge_alignment, feature_stats, scale_stats, sequential_split

__all__ = ["SUITE_TASKS", "SUITE_DEFAULTS", "suite_task_defaults", "load_surf", "load_dyn", "load_dynfix",
           "load_qual", "rollout_eval", "surf_integral_error", "mass_weighted_errors", "mass_inverse_map",
           "mass_div_readout_available"]

# variable-mesh complexes are kept on the GPU when their estimated size is below this (override: RHMP_GPU_BUDGET_GB)
GPU_BUDGET_GB = float(os.environ.get("RHMP_GPU_BUDGET_GB", 24.0))
BUILD_THREADS = int(os.environ.get("RHMP_BUILD_THREADS", 4))   # complexes are built in a thread pool (x3 faster)
SURF_FAMILIES = ("ellipsoid", "superquadric", "sphere_pert", "torus", "double_torus")   # codes of gen_surf.py

SUITE_DEFAULTS: dict[str, dict] = {
    "SURF": dict(C=128, layers=4, batch=8),
    "DYN": dict(C=128, layers=4, batch=16),
    "DYNfix": dict(C=128, layers=4, batch=64),
    "HP_qual": dict(C=128, layers=4, batch=8),
}


def suite_task_defaults(name: str) -> dict:
    """Training defaults (``C``, ``layers``, ``batch``) of a suite task."""
    if name.startswith("SURF"):
        return dict(SUITE_DEFAULTS["SURF"])
    if name.startswith("DYNfix"):
        return dict(SUITE_DEFAULTS["DYNfix"])
    if name.startswith("DYN"):
        return dict(SUITE_DEFAULTS["DYN"])
    if name.startswith("HP_qual"):
        return dict(SUITE_DEFAULTS["HP_qual"])
    raise KeyError(f"unknown suite task {name!r}")


# ======================================================================================================================
# helpers
# ======================================================================================================================
def _droot(root) -> str:
    from rhmp.tasks import resolve_root
    return resolve_root(root)


def _load_packed(droot: str, base: str) -> dict:
    path = os.path.join(droot, "v2", base + ".pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run the generators in datasets/generators (gen_surf.py / gen_dyn.py / "
                                f"gen_qual.py)")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    blob["_path"] = path
    return blob


def _sl(blob: dict, key: str, i: int, deg: int) -> Tensor:
    p = blob[f"ptr{deg}"]
    return blob[key][p[i]:p[i + 1]]


def _nbytes(o) -> int:
    """Bytes of all tensors reachable from ``o`` (tensors incl. CSR, dicts, lists, dataclasses such as complexes)."""
    if isinstance(o, Tensor):
        if o.layout == torch.sparse_csr:
            return sum(t.numel() * t.element_size() for t in (o.crow_indices(), o.col_indices(), o.values()))
        if o.is_sparse:
            return o._indices().numel() * 8 + o._values().numel() * o._values().element_size()
        return o.numel() * o.element_size()
    if isinstance(o, dict):
        return sum(_nbytes(v) for v in o.values())
    if isinstance(o, (list, tuple)):
        return sum(_nbytes(v) for v in o)
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return sum(_nbytes(getattr(o, f.name)) for f in dataclasses.fields(o))
    return 0


def _placement(Ks: list, device, keep_on: str) -> tuple[str, float]:
    """Where to keep a list of complexes: the target device if their total size (every tensor, incl. the Whitney
    blocks) fits ``GPU_BUDGET_GB``, else the CPU (the trainer then moves each block-diagonal batch)."""
    uniq = list({id(K): K for K in Ks}.values())
    n = min(50, len(uniq))
    est_gb = sum(_nbytes(K) for K in uniq[:n]) / max(n, 1) * len(uniq) / 1e9
    if keep_on != "auto":
        return keep_on, est_gb
    return (str(device) if est_gb < GPU_BUDGET_GB else "cpu"), est_gb


def _triangle_complex(pos: Tensor, faces: Tensor, star: str):
    from rhmp.complex import CochainComplex
    K = CochainComplex.from_triangles(pos.double(), faces.long(), star=star, device="cpu")
    if K.n[2] != faces.shape[0]:            # the generators never produce degenerate or duplicate faces
        raise RuntimeError(f"the complex dropped {faces.shape[0] - K.n[2]} faces")
    return K


def _build_complexes(meshes: list, star: str) -> list:
    """CPU complexes of ``[(pos, faces), ...]``, built in a thread pool (bitwise identical to a serial build)."""
    if BUILD_THREADS <= 1 or len(meshes) < 8:
        return [_triangle_complex(p, f, star) for p, f in meshes]
    nt = torch.get_num_threads()
    torch.set_num_threads(1)                 # one intra-op thread per worker (avoids oversubscription)
    try:
        with ThreadPoolExecutor(BUILD_THREADS) as ex:
            return list(ex.map(lambda m: _triangle_complex(m[0], m[1], star), meshes))
    finally:
        torch.set_num_threads(nt)


def _pinned_copy(t: Tensor, dev: torch.device) -> Tensor:
    """Asynchronous host->GPU copy through pinned memory (synchronous pageable copies wait for the other processes'
    kernels on a shared GPU: ~5 ms per small tensor at 100 % utilisation)."""
    if t.device == dev:
        return t
    if t.device.type == "cpu" and dev.type == "cuda":
        return t.pin_memory().to(dev, non_blocking=True)
    return t.to(dev)


def _move_fast(obj, dev: torch.device):
    """``rhmp.complex._move`` (dense and CSR tensors in lists/tuples/dicts) with pinned non-blocking copies."""
    if isinstance(obj, Tensor):
        if obj.layout == torch.sparse_csr:
            from rhmp.ops import sparse_csr
            return sparse_csr(_pinned_copy(obj.crow_indices(), dev), _pinned_copy(obj.col_indices(), dev),
                              _pinned_copy(obj.values(), dev), tuple(obj.shape))
        return _pinned_copy(obj, dev)
    if isinstance(obj, list):
        return [_move_fast(v, dev) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_fast(v, dev) for v in obj)
    if isinstance(obj, dict):
        return {k: _move_fast(v, dev) for k, v in obj.items()}
    return obj


def _complexes_to(Ks: list, device) -> list:
    """``[K.to(device) for K in Ks]`` with one synchronisation (every tensor field of the complexes is moved;
    shared complexes are moved once)."""
    dev = torch.device(device)
    if dev.type != "cuda":
        return [K.to(dev) for K in Ks]
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    done: dict[int, object] = {}
    out = []
    for K in Ks:
        if id(K) not in done:
            done[id(K)] = K if K.pos.device == dev else dataclasses.replace(
                K, **{f.name: _move_fast(getattr(K, f.name), dev) for f in dataclasses.fields(K)})
        out.append(done[id(K)])
    torch.cuda.synchronize(dev)
    return out


def _to_views(tensors: list, device) -> list:
    """Move a list of small tensors to ``device`` with *one* (pinned, asynchronous) copy: views into one buffer."""
    if not tensors:
        return []
    dev = torch.device(device)
    if tensors[0].device.type == dev.type and (dev.type != "cuda" or dev.index in (None, tensors[0].device.index)):
        return list(tensors)
    sizes = [t.shape[0] for t in tensors]
    buf = torch.cat(tensors, 0)
    buf = buf.pin_memory().to(dev, non_blocking=True) if dev.type == "cuda" else buf.to(dev)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    return list(buf.split(sizes, 0))


def _truncated_split(split: tuple, max_samples: int | None):
    """Smoke-test truncation (as the HP loader): the first ``max_samples // 3`` samples of every split part."""
    N = sum(len(s) for s in split)
    if max_samples is None or max_samples >= N:
        return None, split
    per = max(1, max_samples // 3)
    ids = torch.cat([s[:per] for s in split]).tolist()
    sizes = [min(per, len(s)) for s in split]
    off = np.cumsum([0] + sizes)
    return ids, tuple(torch.arange(off[j], off[j + 1]) for j in range(3))


def _subset(src: dict, idx) -> dict:
    """View of an evaluation set (``K``, ``inputs``, ``target`` lists) restricted to the indices ``idx``."""
    idx = [int(i) for i in idx]
    return dict(K=[src["K"][i] for i in idx], inputs=[src["inputs"][i] for i in idx],
                target=[src["target"][i] for i in idx])


def _empty_split(N: int) -> tuple[Tensor, Tensor, Tensor]:
    e = torch.zeros(0, dtype=torch.long)
    return e, e.clone(), torch.arange(N)


def _meta_defaults(key: str) -> dict:
    d = SUITE_DEFAULTS[key]
    return dict(v1_C=d["C"], batch_size=d["batch"], layers=d["layers"])


# ======================================================================================================================
# SURF
# ======================================================================================================================
def _surf_build(blob: dict, target: str, star: str, ids=None):
    """Complexes (CPU), raw inputs ``{0: f (n0, 1)}``, raw targets ``(n0, 1)`` and family codes of samples ``ids``."""
    N = len(blob["ptr0"]) - 1
    ids = list(range(N)) if ids is None else list(ids)
    if target not in ("u", "heat"):
        raise ValueError(f"SURF target must be 'u' or 'heat', got {target!r}")
    t0 = time.time()
    Ks = _build_complexes([(_sl(blob, "pos", i, 0), _sl(blob, "faces", i, 2)) for i in ids], star)
    xs = [{0: _sl(blob, "f", i, 0).float()[:, None]} for i in ids]
    ys = [_sl(blob, target, i, 0).float()[:, None] for i in ids]
    fam = [int(blob["family"][i]) for i in ids]
    info = dict(build_seconds=time.time() - t0, n=len(Ks),
                n0_mean=float(np.mean([K.n[0] for K in Ks])) if Ks else 0.0)
    return Ks, xs, ys, fam, info


def load_surf(name: str, droot: str | None = None, *, target: str = "u", test_set: str | None = None,
              native: bool = True, device="cuda", star: str = "cotan", fine: bool = True, extra: bool = True,
              max_samples: int | None = None, keep_on: str = "auto", **_) -> TaskData:
    """SURF / SURF_heat (``test_set=None``) or the test-only views ``SURF[_heat]_geo`` / ``_topo``.

    Args:
        target: ``'u'`` (screened Poisson) or ``'heat'``.
        test_set: ``None`` (train/val/test on ``SURF.pt`` + extra test sets) or ``'geo'`` / ``'topo'``.
        fine / extra: ``False`` skips the extra test sets (``fine=False`` is what ``rhmp.train --no-fine`` passes).
        max_samples: smoke tests (first ``max_samples // 3`` samples of each split part; extra sets truncated to
            ``max_samples``).
    """
    droot = _droot(droot)
    t_all = time.time()
    if test_set is not None:
        blob = _load_packed(droot, f"SURF_{test_set}")
        N = len(blob["ptr0"]) - 1
        n_keep = N if max_samples is None else min(N, max_samples)
        Ks, xs, ys, fam, info = _surf_build(blob, target, star, range(n_keep))
        x_stats = {0: feature_stats([x[0] for x in xs])}
        y_stats = feature_stats(ys)
        on, est = _placement(Ks, device, keep_on)
        meta = dict(_meta_defaults("SURF"), source=blob["_path"], generator=blob["meta"], target=target,
                    complexes=dict(info, estimated_GB=est, stored_on=on), r2_std=y_stats.std, star=star,
                    families=[SURF_FAMILIES[c] for c in sorted(set(fam))], test_only=True,
                    note="test-only set: statistics from its own samples; evaluate a trained SURF model with "
                         "scripts/eval_on.py or rhmp.train --eval-ckpt (both re-normalise with the training "
                         "statistics)")
        return TaskData(name=name, K=_complexes_to(Ks, on),
                        inputs=[{0: v} for v in _to_views([x_stats[0].normalize(x[0]) for x in xs], on)],
                        target=_to_views([y_stats.normalize(y) for y in ys], on), target_degree=0,
                        target_kind="node_scalar",
                        in_dims={0: 1}, even_dims={}, connection_dims={}, split=_empty_split(len(Ks)),
                        x_stats={0: x_stats[0].to(device)}, y_stats=y_stats.to(device), spatial_dim=3, out_dim=1,
                        native=native, meta=meta)
    blob = _load_packed(droot, "SURF")
    N = len(blob["ptr0"]) - 1
    split = sequential_split(N)
    ids, split = _truncated_split(split, max_samples)
    Ks, xs, ys, fam, info = _surf_build(blob, target, star, ids)
    tr = split[0].tolist()
    x_stats = {0: feature_stats([xs[i][0] for i in tr])}
    y_stats = feature_stats([ys[i] for i in tr])
    on, est = _placement(Ks, device, keep_on)
    Ks = _complexes_to(Ks, on)
    inputs = [{0: v} for v in _to_views([x_stats[0].normalize(x[0]) for x in xs], on)]
    tgt = _to_views([y_stats.normalize(y) for y in ys], on)
    main = dict(K=Ks, inputs=inputs, target=tgt)
    extra_tests = {}
    te = split[2].tolist()
    for code in sorted({fam[i] for i in te}):
        extra_tests[f"test_{SURF_FAMILIES[code]}"] = _subset(main, [i for i in te if fam[i] == code])
    if fine and extra:
        for sub in ("geo", "topo"):
            path = os.path.join(droot, "v2", f"SURF_{sub}.pt")
            if not os.path.exists(path):
                continue
            b = _load_packed(droot, f"SURF_{sub}")
            nb = len(b["ptr0"]) - 1
            eK, ex, ey, efam, einfo = _surf_build(b, target, star, range(nb if max_samples is None else
                                                                          min(nb, max_samples)))
            extra_tests[sub] = dict(K=eK, inputs=[{0: x_stats[0].normalize(x[0])} for x in ex],
                                    target=[y_stats.normalize(y) for y in ey], info=einfo,
                                    families=[SURF_FAMILIES[c] for c in sorted(set(efam))])
    meta = dict(_meta_defaults("SURF"), source=blob["_path"], generator=blob["meta"], target=target,
                complexes=dict(info, estimated_GB=est, stored_on=on), r2_std=y_stats.std, star=star,
                family_of_sample=fam, load_seconds=time.time() - t_all,
                split_note="sequential 70/15/15 over SURF.pt (families ellipsoid / sphere_pert / torus interleaved)")
    return TaskData(name=name, K=Ks, inputs=inputs, target=tgt, target_degree=0, target_kind="node_scalar",
                    in_dims={0: 1}, even_dims={}, connection_dims={}, split=split,
                    x_stats={0: x_stats[0].to(device)}, y_stats=y_stats.to(device), spatial_dim=3, out_dim=1,
                    native=native, meta=meta, extra_tests=extra_tests)


@torch.no_grad()
def surf_integral_error(model, task: TaskData, *, source: dict | None = None, idx=None, batch_size: int = 8,
                        device=None) -> dict:
    """Structure metric of SURF / SURF_heat: relative error of the discrete integral ``sum_i M_i u_i``.

    Both targets conserve it exactly (``1^T L = 0``: ``sum M u = sum M f`` for the screened Poisson solve and for every
    implicit heat step), so ``e_i = |sum M u_pred - sum M f| / sum M |f|`` measures how far a prediction is from the
    constraint (``M`` = lumped mass = ``K.star[0]``, float32).  The same quantity for the stored targets is returned as
    a sanity reference (float32 round-off, ~1e-7).

    Args:
        source: evaluation set ``{'K', 'inputs', 'target'}`` (default: the task itself, split ``idx`` = test split).
    Returns:
        ``{'mean', 'max', 'median', 'target_max', 'N'}``.
    """
    from rhmp.train import predict
    device = torch.device(device) if device is not None else next(model.parameters()).device
    src = source if source is not None else dict(K=task.K, inputs=task.inputs, target=task.target)
    if idx is None:
        idx = torch.arange(len(src["target"])) if source is not None else task.split[2]
    idx = torch.as_tensor(idx)
    was = model.training
    model.eval()
    pred, _ = predict(model, task, idx, batch_size, device, source=source)
    model.train(was)
    errs, terr = [], []
    for j, i in enumerate(idx.tolist()):
        K = src["K"][i]
        m = K.star[0].double().cpu()
        f = task.x_stats[0].denormalize(src["inputs"][i][0]).double().cpu()[:, 0]
        u = task.y_stats.denormalize(pred[j]).double().cpu()[:, 0]
        y = task.y_stats.denormalize(src["target"][i]).double().cpu()[:, 0]
        scale = float((m * f.abs()).sum())
        errs.append(abs(float((m * u).sum() - (m * f).sum())) / scale)
        terr.append(abs(float((m * y).sum() - (m * f).sum())) / scale)
    e = np.array(errs)
    return dict(mean=float(e.mean()), median=float(np.median(e)), max=float(e.max()), target_max=float(max(terr)),
                N=len(errs))


def mass_weighted_errors(task: TaskData, pred, *, source: dict | None = None, idx=None) -> dict:
    """Mesh-density-independent errors of node predictions: ``||u_pred - u||_M / ||u||_M`` per sample, with the lumped
    mass ``M = K.star[0]`` (barycentric dual areas), in physical units.

    Node-pooled R2 / NRMSE weight every node equally, so on graded meshes (nodes clustered near a Dirichlet corner
    where u ~ 0) they change with the mesh density even for a perfect solver; the mass-weighted error approximates
    the continuous relative L2 error and is comparable across quality levels.

    Args:
        pred: normalised predictions for the samples ``idx`` of ``source`` (as returned by ``rhmp.train.predict``:
            list of ``(n_i, 1)`` or a stacked ``(N, n, 1)`` tensor).
        source: evaluation set ``{'K', 'target'}`` (default: the task itself; ``idx`` default = test split).
    Returns:
        ``{'relL2M_mean', 'relL2M_median', 'relL2M_pooled'}`` (pooled = sqrt(sum_i ||e_i||_M^2 / sum_i ||u_i||_M^2)).
    """
    src = source if source is not None else dict(K=task.K, target=task.target)
    if idx is None:
        idx = torch.arange(len(src["target"])) if source is not None else task.split[2]
    num, den, rel = 0.0, 0.0, []
    for j, i in enumerate(torch.as_tensor(idx).tolist()):
        m = src["K"][i].star[0].double().cpu()
        p = task.y_stats.denormalize(pred[j]).double().cpu()[:, 0]
        y = task.y_stats.denormalize(src["target"][i]).double().cpu()[:, 0]
        e2, y2 = float((m * (p - y) ** 2).sum()), float((m * y ** 2).sum())
        num, den = num + e2, den + y2
        rel.append((e2 / max(y2, 1e-300)) ** 0.5)
    return dict(relL2M_mean=float(np.mean(rel)), relL2M_median=float(np.median(rel)),
                relL2M_pooled=float((num / max(den, 1e-300)) ** 0.5))


# ======================================================================================================================
# DYN / DYNfix
# ======================================================================================================================
def _window_times(T: int, windows) -> np.ndarray:
    """Start times of the one-step windows: every ``T // windows`` steps (``'all'``: every step)."""
    if windows in (None, "all", 0):
        return np.arange(T)
    stride = max(1, T // int(windows))
    return np.arange(0, T, stride)[: int(windows)]


DYN_TARGETS = ("state", "delta", "cons", "cons_mass")
MASS_DIV_READOUT = "mdiv:1"     # y_i = (d0^T b)_i / star0_i inside the model (requested from the model owner)


def _check_dyn_target(target: str) -> None:
    if target not in DYN_TARGETS:
        raise ValueError(f"DYN target must be one of {DYN_TARGETS}, got {target!r}")


def _dyn_target(traj: Tensor, t: int, target: str, mass: Tensor | None = None) -> Tensor:
    """Training target of the window starting at ``t`` (``traj (T+1, n0)``): ``u_{t+1}`` (state), the density
    increment ``u_{t+1} - u_t`` (delta, cons) or the mass change per dual cell ``M (u_{t+1} - u_t)`` (cons_mass;
    ``mass (n0,)``)."""
    if target in ("delta", "cons"):
        return traj[t + 1] - traj[t]
    if target == "cons_mass":
        return mass * (traj[t + 1] - traj[t])
    return traj[t + 1]


def _dyn_target_stats(values, target: str) -> Stats:
    """Scale-only statistics for the conservative targets (a mean shift would break sum_i M_i du_i = 0)."""
    return scale_stats(values) if target in ("cons", "cons_mass") else feature_stats(values)


def mass_div_readout_available() -> bool:
    """True if the model provides the mass-weighted divergence readout ``mdiv:1`` (``y = star0^{-1} d0^T b``)."""
    try:
        from rhmp.readout import parse_readout
        parse_readout(MASS_DIV_READOUT)
        return True
    except Exception:  # noqa: BLE001 - unknown readout spec (ValueError/KeyError) or no readout module
        return False


def dual_edge_lengths(K) -> Tensor:
    """Barycentric dual edge lengths ``l*_e = sum_{f ni e} |c_f - m_e|`` (face centroid to edge midpoint), ``(n1,)``."""
    P = K.pos.double().cpu()
    F = K.cells[2].long().cpu()
    E = K.cells[1].long().cpu()
    fe = K.meta["face_edges"].long().cpu()                          # (n2, 3): edge of vertices (j, j+1)
    cf = P[F].mean(1)                                               # (n2, D)
    mid = P[E].mean(1)                                              # (n1, D)
    out = torch.zeros(len(E), dtype=torch.float64)
    for j in range(fe.shape[1]):
        e = fe[:, j]
        ok = e >= 0
        out.index_add_(0, e[ok], (cf[ok] - mid[e[ok]]).norm(dim=1))
    return out


def flux_density_map(K, mass: Tensor, device) -> "OutputMap":
    """Fixed output map ``du = (mean M / M) d0^T (l*/mean l* . b)`` from an odd edge cochain ``b`` (readout
    ``cochain:1``): a learned *flux density* integrated over the dual edges and divided by the dual area (a finite-
    volume update).  Exactly conservative (``sum_i M_i du_i = mean M sum_i (d0^T x)_i = 0``); the node-to-node
    dynamic range of the map is ~perimeter/area ~ 1/r_i instead of 1/M_i for the ``div:1`` form."""
    from rhmp.data import OutputMap
    from rhmp.ops import csr_from_coo
    E = K.cells[1].long().cpu()
    n0, n1 = K.n[0], K.n[1]
    ls = dual_edge_lengths(K)
    w = ls / ls.mean()
    s = mass.double().mean() / mass.double()
    rows = torch.cat([E[:, 0], E[:, 1]])
    cols = torch.cat([torch.arange(n1), torch.arange(n1)])
    vals = torch.cat([-s[E[:, 0]] * w, s[E[:, 1]] * w]).float()
    order = torch.argsort(rows * n1 + cols)
    rows, cols, vals = rows[order], cols[order], vals[order]
    A = csr_from_coo(rows.to(device), cols.to(device), vals.to(device), (n0, n1))
    o2 = torch.argsort(cols * n0 + rows)
    AT = csr_from_coo(cols[o2].to(device), rows[o2].to(device), vals[o2].to(device), (n1, n0))
    return OutputMap("sparse", "cochain:1", 1, M=A, MT=AT,
                     description="du = (mean M / M) d0^T (l*/mean l* . b): finite-volume update from a learned edge "
                                 "flux density (exactly conservative)")


def mass_inverse_map(mass: Tensor, device) -> "OutputMap":
    """Fixed output map ``du_i = (mean M / M_i) z_i`` on one mesh (model readout ``div:1``, ``z = d0^T b``).

    Makes a zero-sum divergence output an exactly conservative density increment (``sum_i M_i du_i = mean M *
    sum_i z_i = 0``) while the loss is taken on ``du`` itself (node-uniform in u-space).
    """
    from rhmp.data import OutputMap
    from rhmp.ops import csr_from_coo
    n = int(mass.numel())
    idx = torch.arange(n)
    val = (mass.double().mean() / mass.double()).float()
    M = csr_from_coo(idx.to(device), idx.to(device), val.to(device), (n, n))
    return OutputMap("sparse", "div:1", 1, M=M, MT=M,
                     description="du_i = (mean M / M_i) (d0^T b)_i: exactly conservative density increment "
                                 f"(M_min/M_mean = {float(mass.min() / mass.mean()):.3g})")


def _traj_split(n_traj: int, max_samples: int | None, per_traj: int):
    split = sequential_split(n_traj)
    if max_samples is None:
        return split
    per = max(1, max_samples // (3 * per_traj))
    return tuple(s[:per] for s in split)


def load_dyn(name: str = "DYN", droot: str | None = None, *, native: bool = True, device="cuda",
             star: str = "cotan", windows=25, target: str = "state", max_samples: int | None = None,
             keep_on: str = "auto", **_) -> TaskData:
    """DYN: one-step windows on random meshes (variable-mesh TaskData) + ``task.rollout`` (test trajectories).

    Args:
        windows: windows per trajectory (every ``T // windows`` steps; ``'all'`` = all ``T``).
        target: ``'state'`` (u_{t+1}, task ``DYN``), ``'delta'`` (u_{t+1} - u_t, task ``DYN_delta``; the rollout
            adds the predicted increment), ``'cons'`` (task ``DYN_cons``: du with scale-only normalisation and the
            mass-weighted divergence readout ``mdiv:1``, exactly conservative) or ``'cons_mass'`` (task ``DYN_cons_mass``: the mass change per dual
            cell ``M du`` with the plain ``div:1`` readout; the first, ill-conditioned conservative convention).
        max_samples: smoke tests: about ``max_samples`` windows in total (whole trajectories, 1/3 per split part).
    Inputs (native): ``{0: u_t (n0, 1), 1: theta (n1, 1)}`` (theta odd); legacy: ``{0: [u_t, v_x, v_y] (n0, 3)}``.
    """
    from rhmp.tasks.synthetic import _mesh_edges_torch
    _check_dyn_target(target)
    if target == "cons" and not mass_div_readout_available():
        raise NotImplementedError(
            "DYN_cons (variable meshes) needs a model readout that divides the divergence by the lumped mass inside "
            f"the model ({MASS_DIV_READOUT!r}: y_i = (d0^T b)_i / star0_i). "
            "With the plain 'div:1' readout the loss can only be put on the mass change M*du (DYN_cons_mass), whose "
            "node-uniform errors become ~1/M_i errors in du (M_min/M_median = 0.008 on the DYN meshes) and explode "
            "in autoregressive rollouts.  Use DYNfix_cons (fixed mesh: exactly conservative through a fixed output "
            "map) or DYN_delta with rollout_eval(..., project_mass=True).")
    droot = _droot(droot)
    t_all = time.time()
    blob = _load_packed(droot, "DYN")
    n_traj = len(blob["ptr0"]) - 1
    T = int(blob["traj"].shape[1]) - 1
    times = _window_times(T, windows)
    tsplit = _traj_split(n_traj, max_samples, len(times))
    t0 = time.time()
    trajK, trajU, trajTh, trajV, trajM = {}, {}, {}, {}, {}
    jj = torch.cat(tsplit).tolist()
    built = _build_complexes([(_sl(blob, "pos", j, 0), _sl(blob, "faces", j, 2)) for j in jj], star)
    for j, K in zip(jj, built):
        pos, faces = _sl(blob, "pos", j, 0), _sl(blob, "faces", j, 2)
        n0 = pos.shape[0]
        perm, sign = edge_alignment(K.cells[1], _mesh_edges_torch(faces.long(), n0), n0)
        trajK[j] = K
        trajU[j] = _sl(blob, "traj", j, 0).float().T.contiguous()                    # (T+1, n0) physical
        trajTh[j] = (_sl(blob, "theta", j, 1).float()[perm] * sign)[:, None]         # (n1, 1) odd, complex order
        trajV[j] = _sl(blob, "vel", j, 0).float()                                    # (n0, 2)
        trajM[j] = _sl(blob, "mass", j, 0).float()
    t_build = time.time() - t0
    tr_traj = tsplit[0].tolist()
    # statistics (training windows / training trajectories)
    u_in = [trajU[j][t][:, None] for j in tr_traj for t in times]
    u_out = [_dyn_target(trajU[j], t, target, trajM[j])[:, None] for j in tr_traj for t in times]
    if native:
        x_stats = {0: feature_stats(u_in), 1: scale_stats([trajTh[j] for j in tr_traj])}
    else:
        x_stats = {0: feature_stats([torch.cat([trajU[j][t][:, None], trajV[j]], 1) for j in tr_traj
                                     for t in times])}
    y_stats = _dyn_target_stats(u_out, target)
    on, est = _placement([trajK[j] for j in trajK], device, keep_on)
    Kdev = dict(zip(trajK, _complexes_to(list(trajK.values()), on)))
    if native:
        th_n = dict(zip(jj, _to_views([x_stats[1].normalize(trajTh[j]) for j in jj], on)))
    Ks, x0_list, x1_list, tgt_list, win = [], [], [], [], []
    split_idx = []
    for part in tsplit:
        start = len(Ks)
        for j in part.tolist():
            for t in times:
                u_t = trajU[j][t][:, None]
                if native:
                    x0_list.append(x_stats[0].normalize(u_t))
                    x1_list.append(th_n[j])
                else:
                    x0_list.append(x_stats[0].normalize(torch.cat([u_t, trajV[j]], 1)))
                Ks.append(Kdev[j])
                tgt_list.append(y_stats.normalize(_dyn_target(trajU[j], t, target, trajM[j])[:, None]))
                win.append((j, int(t)))
        split_idx.append(torch.arange(start, len(Ks)))
    x0_list = _to_views(x0_list, on)
    tgt_list = _to_views(tgt_list, on)
    inputs = [{0: a, 1: b} for a, b in zip(x0_list, x1_list)] if native else [{0: a} for a in x0_list]
    te_traj = tsplit[2].tolist()
    rollout = dict(shared=False, native=native, target=target, steps=T, dt=float(blob["meta"].get("dt_stored", 0.0)),
                   traj_ids=te_traj, K=[trajK[j] for j in te_traj], traj=[trajU[j] for j in te_traj],
                   theta=[trajTh[j] for j in te_traj], vel=[trajV[j] for j in te_traj],
                   mass=[trajM[j] for j in te_traj])
    in_dims = {0: 1, 1: 1} if native else {0: 3}
    meta = dict(_meta_defaults("DYN"), source=blob["_path"], generator=blob["meta"], r2_std=y_stats.std, star=star,
                windows_per_traj=len(times), window_times=times.tolist(), window_of_sample=win,
                traj_split=[len(s) for s in tsplit], rollout_steps=T, target=target,
                complexes=dict(build_seconds=t_build, n=len(trajK), estimated_GB=est, stored_on=on),
                load_seconds=time.time() - t_all,
                split_note="sequential 70/15/15 over trajectories; windows (u_t, theta) -> u_{t+1}; task.rollout = "
                           "full test trajectories for rhmp.tasks.suite.rollout_eval")
    readout = None
    if target == "cons_mass":
        readout = "div:1"                        # y = d0^T b: exactly zero sum per mesh (discrete conservation)
    elif target == "cons":
        readout = MASS_DIV_READOUT               # du = star0^{-1} d0^T b: exactly conservative density increment
        meta["cons_convention"] = "u-space"
    td = TaskData(name=name, K=Ks, inputs=inputs, target=tgt_list, target_degree=0, target_kind="node_scalar",
                  in_dims=in_dims, even_dims={}, connection_dims={}, split=tuple(split_idx),
                  x_stats={k: s.to(device) for k, s in x_stats.items()}, y_stats=y_stats.to(device), spatial_dim=2,
                  out_dim=1, native=native, meta=meta, readout=readout)
    td.rollout = rollout
    return td


def load_dynfix(name: str = "DYNfix", droot: str | None = None, *, native: bool = True, device="cuda",
                star: str = "cotan", windows=25, target: str = "state", max_samples: int | None = None,
                cons_param: str = "flux", **_) -> TaskData:
    """DYNfix: one-step windows on one fixed mesh (shared-mesh TaskData: ``inputs[k]`` is ``(N, n_k, F_k)``).

    ``target='delta'`` = task ``DYNfix_delta`` (increments); ``'cons'`` = ``DYNfix_cons``: du (scale-only
    normalisation, loss in u-space) predicted through a fixed exactly-conservative output map, ``cons_param='flux'``
    (default; :func:`flux_density_map`, readout ``cochain:1``: a learned edge flux density integrated over the dual
    edges and divided by the dual area) or ``'div'`` (:func:`mass_inverse_map`, readout ``div:1``); ``'cons_mass'`` =
    ``DYNfix_cons_mass`` (see :func:`load_dyn`)."""
    from rhmp.tasks.synthetic import _mesh_edges_torch
    _check_dyn_target(target)
    droot = _droot(droot)
    t_all = time.time()
    blob = _load_packed(droot, "DYNfix")
    traj = blob["traj"].float()                                    # (N, T+1, n0)
    n_traj, T = traj.shape[0], traj.shape[1] - 1
    times = _window_times(T, windows)
    tsplit = _traj_split(n_traj, max_samples, len(times))
    pos, faces = blob["pos"], blob["faces"]
    K = _triangle_complex(pos, faces, star)
    n0 = pos.shape[0]
    perm, sign = edge_alignment(K.cells[1], _mesh_edges_torch(faces.long(), n0), n0)
    theta = (blob["theta"].float()[:, perm] * sign)[..., None]     # (N, n1, 1) complex order
    vel = blob["vel"].float()                                      # (N, n0, 2)
    tr = tsplit[0]
    ti = torch.as_tensor(times)
    u_in_tr = traj[tr][:, ti].reshape(-1, n0, 1)
    u_out_tr = traj[tr][:, ti + 1].reshape(-1, n0, 1)
    mass = blob["mass"].float()
    if target in ("delta", "cons"):
        u_out_tr = u_out_tr - u_in_tr
    elif target == "cons_mass":
        u_out_tr = mass[None, :, None] * (u_out_tr - u_in_tr)
    if native:
        x_stats = {0: feature_stats(u_in_tr), 1: scale_stats(theta[tr])}
    else:
        v_rep = vel[tr][:, None].expand(len(tr), len(ti), n0, 2).reshape(-1, n0, 2)
        x_stats = {0: feature_stats(torch.cat([u_in_tr, v_rep], -1))}
    y_stats = _dyn_target_stats(u_out_tr, target)
    order = torch.cat(tsplit)                                      # trajectories in split order
    j_rep = order.repeat_interleave(len(ti))
    t_rep = ti.repeat(len(order))
    u_t = traj[j_rep, t_rep][..., None]                            # (Nw, n0, 1)
    u_n = traj[j_rep, t_rep + 1][..., None]
    if target in ("delta", "cons"):
        u_n = u_n - u_t
    elif target == "cons_mass":
        u_n = mass[None, :, None] * (u_n - u_t)
    if native:
        inputs = {0: x_stats[0].normalize(u_t), 1: x_stats[1].normalize(theta[j_rep])}
    else:
        inputs = {0: x_stats[0].normalize(torch.cat([u_t, vel[j_rep]], -1))}
    inputs = {k: v.to(device).contiguous() for k, v in inputs.items()}
    tgt = y_stats.normalize(u_n).to(device).contiguous()
    sizes = [len(s) * len(ti) for s in tsplit]
    off = np.cumsum([0] + sizes)
    split = tuple(torch.arange(off[j], off[j + 1]) for j in range(3))
    te = tsplit[2]
    rollout = dict(shared=True, native=native, target=target, steps=T, dt=float(blob["meta"].get("dt_stored", 0.0)),
                   traj_ids=te.tolist(), K=K, traj=traj[te], theta=theta[te], vel=vel[te], mass=mass)
    meta = dict(_meta_defaults("DYNfix"), source=blob["_path"], generator=blob["meta"], r2_std=y_stats.std, star=star,
                windows_per_traj=len(times), window_times=times.tolist(), traj_split=[len(s) for s in tsplit],
                rollout_steps=T, target=target, load_seconds=time.time() - t_all,
                split_note="sequential 70/15/15 over trajectories on one fixed mesh; task.rollout = test trajectories")
    readout, omap = None, None
    if target == "cons_mass":
        readout = "div:1"
    elif target == "cons":                       # loss on du; exactly conservative through a fixed output map
        if cons_param == "div":
            omap = mass_inverse_map(mass, device)            # readout div:1: du = (mean M / M) d0^T b
        elif cons_param == "flux":
            omap = flux_density_map(K, mass, device)         # readout cochain:1: du = (mean M/M) d0^T (l* . b)
        else:
            raise ValueError(f"cons_param must be 'div' or 'flux', got {cons_param!r}")
        meta["cons_convention"] = "u-space"
        meta["cons_param"] = cons_param
    td = TaskData(name=name, K=K.to(device), inputs=inputs, target=tgt, target_degree=0, target_kind="node_scalar",
                  in_dims={0: 1, 1: 1} if native else {0: 3}, even_dims={}, connection_dims={}, split=split,
                  x_stats={k: s.to(device) for k, s in x_stats.items()}, y_stats=y_stats.to(device), spatial_dim=2,
                  out_dim=1, native=native, meta=meta, output_map=omap, readout=readout)
    td.rollout = rollout
    return td


# ======================================================================================================================
# rollouts
# ======================================================================================================================
def _model_step(model, task: TaskData, u: Tensor, static: dict, K, xs: dict, ys: Stats, native: bool,
                mode: str = "state", minv: Tensor | None = None) -> Tensor:
    """One autoregressive step: physical ``u (n0, B)`` -> physical prediction ``(n0, B)``.

    ``mode`` = the task's target: ``'state'`` (the model predicts u_{t+1}), ``'delta'`` / ``'cons'`` (the density
    increment, after the task's output map if any; added to ``u``) or ``'cons_mass'`` (the mass change per dual cell:
    ``u + y / M`` with ``minv = 1 / M``)."""
    x0 = xs[0].normalize(u[..., None]) if native else None
    if native:
        inputs = {0: x0.contiguous(), 1: static[1]}
    else:
        inputs = {0: xs[0].normalize(torch.cat([u[..., None], static["vel"]], -1)).contiguous()}
    y = model(inputs, K)
    if task.output_map is not None:
        y = task.output_map(y)
    y = ys.denormalize(y)[..., 0]
    if mode in ("delta", "cons"):
        return u + y
    if mode == "cons_mass":
        return u + y * minv
    return y


def _project_mass(u_new: Tensor, u_old: Tensor, mass: Tensor, seg: Tensor | None, n_seg: int) -> Tensor:
    """Shift every trajectory by a constant so that ``sum M u_new = sum M u_old`` (M-orthogonal projection onto the
    conservative states).  Shared mesh: ``u (n0, B)``, ``mass (n0, 1)``, ``seg=None``; block-diagonal batch:
    ``u (sum n0, 1)``, ``mass (sum n0, 1)``, ``seg (sum n0,)`` trajectory ids."""
    d = mass * (u_old - u_new)
    if seg is None:
        return u_new + d.sum(0, keepdim=True) / mass.sum()
    num = torch.zeros(n_seg, 1, dtype=u_new.dtype, device=u_new.device).index_add_(0, seg, d)
    den = torch.zeros(n_seg, 1, dtype=u_new.dtype, device=u_new.device).index_add_(0, seg, mass)
    return u_new + (num / den)[seg]


@torch.no_grad()
def rollout_eval(model, task: TaskData, steps: int | None = None, *, device=None, batch_size: int | None = None,
                 horizons=(1, 5, 10, 25, 50, 100), return_pred: bool = False, project_mass: bool = False) -> dict:
    """Autoregressive rollouts of a one-step DYN/DYNfix model on the full test trajectories (``task.rollout``).

    Starting from the true ``u_0`` the model's denormalised prediction is fed back as the next input for ``steps``
    steps (default: the stored trajectory length).  Metrics per horizon ``t = 1..steps``:

    * ``R2``: v1 formula in the task's normalised target space (shared mesh: per-node mean over trajectories as for
      stacked tensors; variable meshes: pooled mean), ``NRMSE``: RMSE / range of the true fields at ``t``;
    * ``mass_drift_mean`` / ``_max`` over trajectories of ``|sum M u_pred(t) - sum M u_0| / sum M |u_0|`` with the
      generator's lumped mass ``M`` (the true dynamics conserve it to round-off: ``true_mass_drift_max``);
    * ``persistence_R2``: the trivial prediction ``u(t) = u_0`` for reference.

    Args:
        model: a module called as ``model(inputs, K) -> (n0, B, 1)`` (normalised output).
        task: DYN or DYNfix :class:`TaskData` (native or legacy) with the ``rollout`` attribute.
        batch_size: trajectories per forward (default 16 on variable meshes, 64 on the shared mesh).
        return_pred: also return the predicted trajectories (list of ``(steps+1, n0)`` CPU tensors).
        project_mass: after every step shift each trajectory by a constant so that ``sum M u`` keeps its initial
            value (post-hoc exact conservation for any model, e.g. DYN / DYN_delta; no effect on the ``_cons``
            models, which conserve by construction).
    Returns:
        dict with lists over horizons and ``summary`` = ``{'R2@h', 'NRMSE@h', 'mass_drift@h'}`` for ``h`` in
        ``horizons``, ``first_nonfinite_step`` (``None`` if the rollout stayed finite).
    """
    from rhmp.complex import CochainComplex
    from rhmp import metrics as M
    roll = getattr(task, "rollout", None)
    if roll is None:
        raise ValueError(f"task {task.name!r} has no rollout data (only DYN / DYNfix)")
    T = int(roll["steps"])
    steps = T if steps is None else min(int(steps), T)
    device = torch.device(device) if device is not None else next(model.parameters()).device
    was = model.training
    model.eval()
    native = bool(roll["native"])
    mode = roll.get("target", "state")
    xs = {k: v.to(device) for k, v in task.x_stats.items()}
    ys = task.y_stats.to(device)
    t0 = time.time()
    preds: list[Tensor] = []                      # per trajectory (steps+1, n0), CPU
    if roll["shared"]:
        K = _complexes_to([roll["K"]], device)[0]
        n_traj = roll["traj"].shape[0]
        bs = batch_size or 64
        minv = (1.0 / roll["mass"]).to(device)[:, None]                          # (n0, 1)
        for s in range(0, n_traj, bs):
            sl = slice(s, min(n_traj, s + bs))
            u = roll["traj"][sl, 0].to(device).T.contiguous()                  # (n0, B)
            B = u.shape[1]
            if native:
                static = {1: xs[1].normalize(roll["theta"][sl].to(device)).permute(1, 0, 2).contiguous()}
            else:
                static = {"vel": roll["vel"][sl].to(device).permute(1, 0, 2).contiguous()}
            out = [u.T.cpu()]
            mcol = roll["mass"].to(device)[:, None]
            for _ in range(steps):
                u_new = _model_step(model, task, u, static, K, xs, ys, native, mode, minv)
                u = _project_mass(u_new, u, mcol, None, 0) if project_mass else u_new
                out.append(u.T.cpu())
            P = torch.stack(out, 1)                                           # (B, steps+1, n0)
            preds.extend(P[b] for b in range(B))
    else:
        n_traj = len(roll["traj"])
        bs = batch_size or 16
        for s in range(0, n_traj, bs):
            ids = list(range(s, min(n_traj, s + bs)))
            Kb = _complexes_to([CochainComplex.batch([roll["K"][i] for i in ids])], device)[0]
            sizes = [int(roll["traj"][i].shape[1]) for i in ids]
            u = torch.cat([roll["traj"][i][0] for i in ids]).to(device)[:, None]   # (sum n0, 1)
            minv = torch.cat([1.0 / roll["mass"][i] for i in ids]).to(device)[:, None]
            if native:
                th = torch.cat([roll["theta"][i] for i in ids]).to(device)          # (sum n1, 1)
                static = {1: xs[1].normalize(th)[:, None, :].contiguous()}
            else:
                static = {"vel": torch.cat([roll["vel"][i] for i in ids]).to(device)[:, None, :]}
            out = [u[:, 0].cpu()]
            mcol = torch.cat([roll["mass"][i] for i in ids]).to(device)[:, None]
            seg = torch.arange(len(ids), device=device).repeat_interleave(torch.tensor(sizes, device=device))
            for _ in range(steps):
                u_new = _model_step(model, task, u, static, Kb, xs, ys, native, mode, minv)
                u = _project_mass(u_new, u, mcol, seg, len(ids)) if project_mass else u_new
                out.append(u[:, 0].cpu())
            P = torch.stack(out, 0)                                           # (steps+1, sum n0)
            preds.extend(P.split(sizes, 1))
    t_roll = time.time() - t0
    model.train(was)
    trajs = [roll["traj"][i] for i in range(len(preds))] if roll["shared"] else roll["traj"]
    mass = roll["mass"]
    st_state = ys if mode == "state" else xs[0]   # statistics of the *state* u (first input column otherwise)
    mean, std = st_state.mean.cpu().double()[:1], st_state.std.cpu().double()[:1]

    def norm(x):
        return (x.double() - mean) / std

    res = dict(task=task.name, target=mode, project_mass=project_mass, steps=steps, n_traj=len(preds),
               horizon=list(range(1, steps + 1)), R2=[], NRMSE=[], mass_drift_mean=[], mass_drift_max=[],
               persistence_R2=[], seconds=t_roll)
    m0 = []
    for i, P in enumerate(preds):
        m = mass if roll["shared"] else mass[i]
        m0.append((float((m.double() * trajs[i][0].double()).sum()), float((m.double() * trajs[i][0].double().abs()).sum())))
    true_drift = 0.0
    first_bad = None
    for t in range(1, steps + 1):
        p_t = [P[t] for P in preds]
        y_t = [trajs[i][t] for i in range(len(preds))]
        finite = all(bool(torch.isfinite(p).all()) for p in p_t)
        if not finite and first_bad is None:
            first_bad = t
        if roll["shared"]:
            pn, yn = norm(torch.stack(p_t))[..., None], norm(torch.stack(y_t))[..., None]
            p0 = norm(torch.stack([trajs[i][0] for i in range(len(preds))]))[..., None]
        else:
            pn, yn = [norm(p)[:, None] for p in p_t], [norm(y)[:, None] for y in y_t]
            p0 = [norm(trajs[i][0])[:, None] for i in range(len(preds))]
        res["R2"].append(M.r2_score(pn, yn) if finite else float("nan"))
        res["persistence_R2"].append(M.r2_score(p0, yn))
        res["NRMSE"].append(M.nrmse([p[:, None] for p in p_t], [y[:, None] for y in y_t]) if finite else float("nan"))
        drifts, tdr = [], []
        for i in range(len(preds)):
            m = (mass if roll["shared"] else mass[i]).double()
            drifts.append(abs(float((m * p_t[i].double()).sum()) - m0[i][0]) / max(m0[i][1], 1e-30))
            tdr.append(abs(float((m * y_t[i].double()).sum()) - m0[i][0]) / max(m0[i][1], 1e-30))
        res["mass_drift_mean"].append(float(np.mean(drifts)))
        res["mass_drift_max"].append(float(np.max(drifts)))
        true_drift = max(true_drift, max(tdr))
    res["true_mass_drift_max"] = true_drift
    res["first_nonfinite_step"] = first_bad
    res["summary"] = {}
    for h in horizons:
        if h <= steps:
            res["summary"][f"R2@{h}"] = res["R2"][h - 1]
            res["summary"][f"NRMSE@{h}"] = res["NRMSE"][h - 1]
            res["summary"][f"mass_drift@{h}"] = res["mass_drift_mean"][h - 1]
            res["summary"][f"persistence_R2@{h}"] = res["persistence_R2"][h - 1]
    if return_pred:
        res["pred"] = preds
    return res


# ======================================================================================================================
# QUAL
# ======================================================================================================================
def load_qual(name: str, droot: str | None = None, *, kind: str = "graded", target: str = "u", native: bool = True,
              device="cuda", star: str = "cotan", max_samples: int | None = None, keep_on: str = "auto",
              **_) -> TaskData:
    """HP_qual_graded / HP_qual_sliver (test only): HP_k100 physics on quality-shifted meshes.

    Built with the HP loader's own per-sample code (``rhmp.tasks.synthetic._build_set``: same inputs f / log sigma
    on edges / log sigma on faces, same alignment), so an HP model can be evaluated unchanged.  ``split[2]`` = all
    samples; ``extra_tests`` = one subset per quality level (``meta['levels']``).

    Args:
        kind: ``'graded'`` or ``'sliver'``.
        target: ``'u'`` (P1 solution on the shifted mesh; the HP definition) or ``'ref'`` (4x-finer reference
            solution interpolated to the nodes).
        max_samples: keep the first ``max(1, max_samples // n_levels)`` base samples of every level.
    """
    from rhmp.tasks.synthetic import _build_set
    droot = _droot(droot)
    t_all = time.time()
    blob = _load_packed(droot, f"HP_qual_{kind}")
    levels = list(blob["meta"]["levels"])
    lev = blob["level"].tolist()
    N = len(lev)
    ids = list(range(N))
    if max_samples is not None and max_samples < N:
        per = max(1, max_samples // len(levels))
        ids = [i for L in range(len(levels)) for i in [j for j in range(N) if lev[j] == L][:per]]
    Ks, xs, ys, info = _build_set(blob, dim=2, star=star, flux=False, aniso=False, native=native, ids=ids)
    if target == "ref":
        ys = [_sl(blob, "u_ref", i, 0).float()[:, None] for i in ids]
    elif target != "u":
        raise ValueError(f"QUAL target must be 'u' or 'ref', got {target!r}")
    degs = sorted(xs[0])
    x_stats = {k: feature_stats([x[k] for x in xs]) for k in degs}
    y_stats = feature_stats(ys)
    on, est = _placement(Ks, device, keep_on)
    Ks = _complexes_to(Ks, on)
    per_deg = {k: _to_views([x_stats[k].normalize(x[k]) for x in xs], on) for k in degs}
    inputs = [{k: per_deg[k][i] for k in degs} for i in range(len(xs))]
    tgt = _to_views([y_stats.normalize(y) for y in ys], on)
    main = dict(K=Ks, inputs=inputs, target=tgt)
    lev_ids = [lev[i] for i in ids]
    extra_tests = {levels[L]: _subset(main, [j for j, l in enumerate(lev_ids) if l == L]) for L in sorted(set(lev_ids))}
    st = blob["sample_stats"]
    qkeys = ("min_angle", "max_angle", "aspect_p50", "aspect_p99", "aspect_max", "frac_aspect_gt10", "edge_ratio",
             "rel_err_vs_ref")
    quality = {levels[L]: {k: float(np.mean([st[i][k] for i in ids if lev[i] == L])) for k in qkeys}
               for L in sorted(set(lev_ids))}
    in_dims = {k: int(xs[0][k].shape[-1]) for k in degs}
    meta = dict(_meta_defaults("HP_qual"), source=blob["_path"], generator=blob["meta"], r2_std=y_stats.std,
                star=star, target=target, levels=levels, level_of_sample=lev_ids, quality=quality, test_only=True,
                complexes=dict(info, estimated_GB=est, stored_on=on), load_seconds=time.time() - t_all,
                note="test only (HP_k100 physics): evaluate an HP_k100 run with rhmp.train --task %s --eval-ckpt RUN "
                     "(re-normalises with the training statistics); extra_tests = quality levels" % name)
    return TaskData(name=name, K=Ks, inputs=inputs, target=tgt, target_degree=0, target_kind="node_scalar",
                    in_dims=in_dims, even_dims={k: in_dims[k] for k in degs if k >= 1}, connection_dims={},
                    split=_empty_split(len(Ks)), x_stats={k: s.to(device) for k, s in x_stats.items()},
                    y_stats=y_stats.to(device), spatial_dim=2, out_dim=1, native=native, meta=meta,
                    extra_tests=extra_tests)


# ======================================================================================================================
# registry
# ======================================================================================================================
def _loader(fn, **fixed):
    """``loader(droot, *, native, device, star, **kw) -> TaskData`` with the task name and options bound."""
    @functools.wraps(fn)
    def load(droot=None, **kw):
        return fn(droot=droot, **{**fixed, **kw})
    return load


SUITE_TASKS: dict = {
    "SURF": _loader(load_surf, name="SURF", target="u"),
    "SURF_heat": _loader(load_surf, name="SURF_heat", target="heat"),
    "SURF_geo": _loader(load_surf, name="SURF_geo", target="u", test_set="geo"),
    "SURF_topo": _loader(load_surf, name="SURF_topo", target="u", test_set="topo"),
    "SURF_heat_geo": _loader(load_surf, name="SURF_heat_geo", target="heat", test_set="geo"),
    "SURF_heat_topo": _loader(load_surf, name="SURF_heat_topo", target="heat", test_set="topo"),
    "DYN": _loader(load_dyn, name="DYN"),
    "DYNfix": _loader(load_dynfix, name="DYNfix"),
    "DYN_delta": _loader(load_dyn, name="DYN_delta", target="delta"),
    "DYNfix_delta": _loader(load_dynfix, name="DYNfix_delta", target="delta"),
    "DYN_cons": _loader(load_dyn, name="DYN_cons", target="cons"),
    "DYNfix_cons": _loader(load_dynfix, name="DYNfix_cons", target="cons"),
    "DYN_cons_mass": _loader(load_dyn, name="DYN_cons_mass", target="cons_mass"),
    "DYNfix_cons_mass": _loader(load_dynfix, name="DYNfix_cons_mass", target="cons_mass"),
    "HP_qual_graded": _loader(load_qual, name="HP_qual_graded", kind="graded", target="u"),
    "HP_qual_sliver": _loader(load_qual, name="HP_qual_sliver", kind="sliver", target="u"),
    "HP_qual_graded_ref": _loader(load_qual, name="HP_qual_graded_ref", kind="graded", target="ref"),
    "HP_qual_sliver_ref": _loader(load_qual, name="HP_qual_sliver_ref", kind="sliver", target="ref"),
}


# ======================================================================================================================
# anisotropy tasks (rhmp/tasks/aniso.py: AHP / ASURF / ACURL / ADARCY) resolve through this registry as well; a failing
# import leaves the suite unchanged
# ======================================================================================================================
try:
    from rhmp.tasks.aniso import ANISO_TASKS as _ANISO_TASKS, aniso_task_defaults as _aniso_task_defaults
    SUITE_TASKS.update(_ANISO_TASKS)
    _suite_task_defaults = suite_task_defaults

    def suite_task_defaults(name: str) -> dict:  # noqa: F811 - extended by the anisotropy tasks
        """Training defaults (``C``, ``layers``, ``batch``) of a suite or anisotropy task."""
        return _aniso_task_defaults(name) if name in _ANISO_TASKS else _suite_task_defaults(name)
except Exception:  # noqa: BLE001 - a broken optional module must not break the suite
    pass
