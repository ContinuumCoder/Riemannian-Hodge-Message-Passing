"""Robustness transforms and the E4 robustness table (exact symmetries, gauge noise, input noise, batch composition).

Library API (used by ``scripts/eval_robustness.py`` and ``tests/test_train.py``)::

    from rhmp.robustness import robustness_table, transformed_task, gauge_task, noise_task, random_rotation
    rows = robustness_table(model, task, te, relabel=True, flip_orient=True, rotate=True, gauge_noise=[0.1, 1],
                            noise=[0.01], batch_sizes=[1, 64], eval_batch=64, device="cuda")

Transforms (each returns a test-only :class:`~rhmp.data.TaskData` view with the task's statistics and readout):

* vertex relabelling / face-orientation flips / E(n) transforms -> :func:`transformed_task`: the complexes are rebuilt
  from the transformed mesh and the data of every degree are remapped (odd cochains with the orientation sign of the
  new canonical edges/faces, node vectors rotated, orientation output maps rebuilt from the new geometry);
* gauge noise ``theta -> theta + d0 lambda`` on connection inputs -> :func:`gauge_task`;
* Gaussian input noise on the normalised inputs -> :func:`noise_task`.

An exactly equivariant/invariant model gives the same metrics on every transformed view (up to the float32 rounding
of the rebuilt geometry, ~1e-7); ``tests/test_train.py::test_robustness_transforms_are_exact`` checks this.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from rhmp import metrics as M
from rhmp.data import TaskData, cell_alignment, edge_alignment

__all__ = ["remap_complex", "transformed_task", "gauge_task", "noise_task", "random_rotation", "robustness_table",
           "SYMMETRY_CONDITIONS"]

SYMMETRY_CONDITIONS = ("relabel", "flip-orient", "rotate", "reflect")   # exact symmetries (plus gauge=<s>)


def _orientation_sign(a: torch.Tensor, b: torch.Tensor, polygon: bool) -> torch.Tensor:
    """+1 where the rows of ``b`` (same vertex sets as ``a``) have the orientation of ``a``.

    Triangles and tetrahedra: permutation parity.  Polygons with >3 vertices: same cyclic order vs reversed.
    """
    if not polygon or a.shape[1] == 3:
        from rhmp.data import _parity_to
        return _parity_to(a, b)
    m = a.shape[1]
    pos = (b == a[:, :1]).float().argmax(1)
    nxt = b.gather(1, ((pos + 1) % m)[:, None])[:, 0]
    return torch.where(nxt == a[:, 1], 1.0, -1.0)


def rebuild(K, pos_new: torch.Tensor, top_new: torch.Tensor):
    """Complex of the same kind and reference star as ``K`` from new positions and top cells."""
    from rhmp.complex import CochainComplex
    kind, star = K.meta.get("cell_type"), K.meta.get("star_type", "cotan")
    dev = K.device
    if K.dim == 3:
        return CochainComplex.from_tetrahedra(pos_new.double(), top_new, star=star, device=dev)
    if kind == "triangle":
        return CochainComplex.from_triangles(pos_new.double(), top_new, star=star, device=dev)
    return CochainComplex.from_polygons(pos_new.double(), top_new, star=star, device=dev)


def remap_complex(K, *, perm=None, flip_frac=0.0, Q=None, shift=None, seed=0):
    """Transformed complex and per-degree data maps.

    Args:
        K: source complex (single mesh).
        perm: ``(n0,)`` new vertex i = old vertex ``perm[i]``.
        flip_frac: fraction of top cells whose orientation is reversed.
        Q, shift: E(n) transform ``x -> x Q^T + shift`` of the positions.
    Returns:
        ``(K_new, maps)``, ``maps[k] = (idx, sign)``: ``new_k = old_k[idx] * sign`` (the sign only for odd data).
    """
    n0 = K.n[0]
    pos = K.pos.double().cpu()
    top = K.cells[K.dim].long().cpu()
    perm = torch.arange(n0) if perm is None else perm.cpu()
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n0)
    pos_new = pos[perm]
    if Q is not None:
        pos_new = pos_new @ torch.as_tensor(Q, dtype=torch.float64).T + torch.as_tensor(shift, dtype=torch.float64)
    top_new = torch.where(top >= 0, inv[top.clamp(min=0)], top)
    if flip_frac > 0:
        g = torch.Generator().manual_seed(seed)
        fl = torch.rand(len(top_new), generator=g) < flip_frac
        t = top_new[fl]
        if K.dim == 3:
            top_new[fl] = t[:, [0, 2, 1, 3]]
        elif (t >= 0).all():
            top_new[fl] = torch.flip(t, [1])
    K_new = rebuild(K, pos_new, top_new)
    maps = {0: (perm.to(K.device), None)}
    idx, sgn = edge_alignment(perm[K_new.cells[1].long().cpu()], K.cells[1].cpu(), n0)
    maps[1] = (idx.to(K.device), sgn.to(K.device))
    for k in range(2, K.dim + 1):
        c_new = K_new.cells[k].long().cpu()
        c_old = torch.where(c_new >= 0, perm[c_new.clamp(min=0)], c_new)
        idx, _ = cell_alignment(c_old, K.cells[k].long().cpu(), n0, oriented=False)
        sgn = _orientation_sign(c_old, K.cells[k].long().cpu()[idx], polygon=(k == 2))
        maps[k] = (idx.to(K.device), sgn.to(K.device))
    return K_new, maps


def _apply(x: torch.Tensor, m, odd: bool, cell_dim: int = 0) -> torch.Tensor:
    idx, sgn = m
    y = x.index_select(cell_dim, idx)
    if odd and sgn is not None:
        shape = [1] * y.dim()
        shape[cell_dim] = -1
        y = y * sgn.view(shape).to(y)
    return y


def _odd_columns(task: TaskData, k: int) -> int:
    return 0 if k == 0 else task.in_dims.get(k, 0) - task.even_dims.get(k, 0)


def _remap_inputs(task, inputs: dict, maps: dict, cell_dim: int) -> dict:
    out = {}
    for k, v in inputs.items():
        no = _odd_columns(task, k)
        parts = []
        if no:
            parts.append(_apply(v[..., :no], maps[k], True, cell_dim))
        if v.shape[-1] > no:
            parts.append(_apply(v[..., no:], maps[k], False, cell_dim))
        out[k] = torch.cat(parts, -1).contiguous()
    return out


def _target_odd(task) -> bool:
    return task.target_kind == "cochain" and task.target_degree >= 1


def _rebuild_map(task, K_new, *, Q=None, perm=None):
    """Output map of the transformed complex (oriented face->node from the new geometry, or rotated normals)."""
    om = task.output_map
    if om is None:
        return None
    from rhmp.data import OutputMap
    from rhmp.tasks.paper import oriented_face_to_node
    if om.kind == "sparse":
        return oriented_face_to_node(K_new, om.model_out_dim)
    if om.kind == "direct_vector":
        from rhmp.tasks.paper import direct_vector_map
        return direct_vector_map(K_new, om.model_readout)
    if om.kind == "cross_normal":
        n = om.normals
        if perm is not None:
            n = n.index_select(0, perm.to(n.device))
        if Q is not None:
            n = n @ torch.as_tensor(Q, dtype=n.dtype, device=n.device).T
        return OutputMap("cross_normal", om.model_readout, om.model_out_dim, normals=n, description=om.description)
    raise ValueError(om.kind)


def _view(task, K, inputs, target, om, n):
    """Test-only view (``split = ([], [], arange(n))``) sharing the task's statistics and readout."""
    return TaskData(name=task.name, K=K, inputs=inputs, target=target, target_degree=task.target_degree,
                    target_kind=task.target_kind, in_dims=task.in_dims, even_dims=task.even_dims,
                    connection_dims=task.connection_dims,
                    split=(torch.empty(0, dtype=torch.long),) * 2 + (torch.arange(n),),
                    x_stats=task.x_stats, y_stats=task.y_stats, spatial_dim=task.spatial_dim, out_dim=task.out_dim,
                    native=task.native, meta=dict(task.meta), output_map=om, readout=task.readout)


def transformed_task(task: TaskData, idx: torch.Tensor, *, perm_seed=None, flip_frac=0.0, Q=None, shift=None,
                     seed=0) -> TaskData:
    """Test-split view of ``task`` under a mesh transform (one transform per mesh for variable-mesh tasks)."""
    rot_vec = Q is not None and task.target_kind == "node_vector"
    g = torch.Generator().manual_seed(seed)
    if not task.variable_mesh:
        K = task.K
        perm = torch.randperm(K.n[0], generator=g) if perm_seed is not None else None
        K_new, maps = remap_complex(K, perm=perm, flip_frac=flip_frac, Q=Q, shift=shift, seed=seed)
        sel = idx.to(task.target.device)
        inputs = _remap_inputs(task, {k: v.index_select(0, sel) for k, v in task.inputs.items()}, maps, 1)
        tgt = _apply(task.target.index_select(0, sel), maps[task.target_degree], _target_odd(task), 1)
        if rot_vec:
            tgt = tgt @ torch.as_tensor(Q, dtype=tgt.dtype, device=tgt.device).T
        return _view(task, K_new, inputs, tgt.contiguous(), _rebuild_map(task, K_new, Q=Q, perm=perm), len(sel))
    Ks, xs, ys = [], [], []
    for j, i in enumerate(idx.tolist()):
        K = task.K[i]
        perm = torch.randperm(K.n[0], generator=g) if perm_seed is not None else None
        K_new, maps = remap_complex(K, perm=perm, flip_frac=flip_frac, Q=Q, shift=shift, seed=seed + j)
        Ks.append(K_new)
        xs.append(_remap_inputs(task, task.inputs[i], maps, 0))
        y = _apply(task.target[i], maps[task.target_degree], _target_odd(task), 0)
        if rot_vec:
            y = y @ torch.as_tensor(Q, dtype=y.dtype, device=y.device).T
        ys.append(y.contiguous())
    return _view(task, Ks, xs, ys, None, len(ys))


def gauge_task(task: TaskData, idx: torch.Tensor, s: float, seed: int = 0) -> TaskData:
    """theta -> theta + d0 lambda on the connection columns (raw units), lambda ~ N(0, s^2) per node and sample."""
    c = task.connection_dims.get(1, 0)
    if c == 0:
        raise ValueError("task has no connection inputs")
    if task.variable_mesh:
        raise NotImplementedError("gauge noise for variable meshes")
    st = task.x_stats[1]
    g = torch.Generator().manual_seed(seed)
    sel = idx.to(task.target.device)
    x1 = st.denormalize(task.inputs[1].index_select(0, sel).transpose(0, 1))         # (n1, N, F) raw
    lam = (torch.randn((task.K.n[0], x1.shape[1], c), generator=g) * s).to(x1)
    e = task.K.cells[1].long()
    x1 = x1.clone()
    x1[..., :c] += lam[e[:, 1]] - lam[e[:, 0]]
    x1 = st.normalize(x1).transpose(0, 1).contiguous()
    inputs = {k: (x1 if k == 1 else v.index_select(0, sel)) for k, v in task.inputs.items()}
    return _view(task, task.K, inputs, task.target.index_select(0, sel), task.output_map, len(sel))


def noise_task(task: TaskData, idx: torch.Tensor, s: float, seed: int = 0) -> TaskData:
    """Gaussian noise of std ``s`` on all normalised inputs."""
    g = torch.Generator().manual_seed(seed)
    if not task.variable_mesh:
        sel = idx.to(task.target.device)
        inputs = {}
        for k, v in task.inputs.items():
            vs = v.index_select(0, sel)
            inputs[k] = vs + s * torch.randn(vs.shape, generator=g).to(vs)
        return _view(task, task.K, inputs, task.target.index_select(0, sel), task.output_map, len(sel))
    ids = idx.tolist()
    inputs = [{k: v + s * torch.randn(v.shape, generator=g).to(v) for k, v in task.inputs[i].items()} for i in ids]
    return _view(task, [task.K[i] for i in ids], inputs, [task.target[i] for i in ids], None, len(ids))


def random_rotation(D: int, seed: int, proper: bool = True) -> np.ndarray:
    """Random orthogonal ``(D, D)`` matrix with det +1 (``proper``) or -1."""
    rng = np.random.default_rng(seed)
    Q, R = np.linalg.qr(rng.normal(size=(D, D)))
    Q = Q * np.sign(np.diag(R))
    if (np.linalg.det(Q) > 0) != proper:
        Q[:, 0] = -Q[:, 0]
    return Q


def _maxdiff(a, b) -> float:
    if isinstance(a, torch.Tensor):
        return float((a - b).abs().max().item())
    return float(max((x - y).abs().max().item() for x, y in zip(a, b)))


def robustness_table(model, task: TaskData, te: torch.Tensor, *, eval_batch: int, device, relabel: bool = False,
                     flip_orient: bool = False, rotate: bool = False, reflect: bool = False,
                     gauge_noise=(), noise=(), batch_sizes=(), extra: bool = True, seed: int = 0,
                     log=print) -> list[dict]:
    """Rows of the E4 table for ``model`` on the test samples ``te`` of ``task``.

    Args:
        model: trained ``RHMP`` in eval mode (``record_diagnostics = False`` recommended).
        task: normalised task (statistics of the training run, see ``rhmp.train._renormalize``).
        te: test sample indices.
        eval_batch: evaluation batch size of the reference row ``base``.
        relabel, flip_orient, rotate, reflect: exact-symmetry rows (``reflect`` is skipped for tasks with an
            orientation output map, whose targets are pseudo-quantities, and for legacy frame-dependent inputs).
        gauge_noise: lambda stds of the gauge rows (tasks with connection inputs, shared meshes).
        noise: input-noise stds.
        batch_sizes: evaluation batch sizes (batch independence, with ``max_dpred`` vs the reference).
        extra: also evaluate the task's extra test sets (``fine``, quality splits...).
    Returns:
        list of dicts ``{condition, R2, MSE, NRMSE, SSIM, Pearson, N, [max_dpred], dR2}``.
    """
    from rhmp.train import evaluate, predict
    keys = ("R2", "MSE", "NRMSE", "SSIM", "Pearson", "N")
    center = task.target_kind != "cochain"
    rows: list[dict] = []

    def run_view(view, name):
        t = time.time()
        vidx = view.split[2] if view is not task else te
        m = evaluate(model, view, vidx, eval_batch, device)
        row = {"condition": name, **{k: m[k] for k in keys}, "seconds": time.time() - t}
        rows.append(row)
        log(f"  {name:18s} R2={row['R2']:.6f} NRMSE={row['NRMSE']:.6f}")
        return row

    base = run_view(task, "base")
    if extra:
        for name, src in task.extra_tests.items():
            m = evaluate(model, task, torch.arange(len(src["target"])), eval_batch, device, source=src)
            rows.append({"condition": f"extra:{name}", **{k: m[k] for k in keys}})
            log(f"  extra:{name:12s} R2={m['R2']:.6f}")
    if batch_sizes:
        p_base, _ = predict(model, task, te, eval_batch, device)
        for b in batch_sizes:
            t = time.time()
            p, tg = predict(model, task, te, int(b), device)
            m = M.summarize(p, tg, task.y_stats.as_tuple(), r2_std=task.meta.get("r2_std"), center=center)
            rows.append({"condition": f"batch={b}", **{k: m[k] for k in keys}, "max_dpred": _maxdiff(p, p_base),
                         "seconds": time.time() - t})
            log(f"  batch={int(b):<12d} R2={m['R2']:.6f} max|dpred|={rows[-1]['max_dpred']:.2e}")
    D = task.spatial_dim
    if relabel:
        run_view(transformed_task(task, te, perm_seed=seed, seed=seed), "relabel")
    if flip_orient:
        run_view(transformed_task(task, te, flip_frac=0.5, seed=seed), "flip-orient")
    if rotate:
        if not task.native:
            log("  rotate: skipped (legacy inputs are frame components)")
        else:
            run_view(transformed_task(task, te, Q=random_rotation(D, seed), shift=np.full(D, 0.37), seed=seed),
                     "rotate")
    if reflect:
        if task.output_map is not None or not task.native:
            log("  reflect: skipped (pseudo-scalar/vector target or frame-dependent inputs)")
        else:
            run_view(transformed_task(task, te, Q=random_rotation(D, seed + 1, proper=False),
                                      shift=np.full(D, -0.21), seed=seed), "reflect")
    for s in gauge_noise or ():
        run_view(gauge_task(task, te, float(s), seed=seed), f"gauge={float(s):g}")
    for s in noise or ():
        run_view(noise_task(task, te, float(s), seed=seed), f"noise={float(s):g}")
    for r in rows:
        r["dR2"] = None if r["condition"].startswith(("extra:", "task:")) else r["R2"] - base["R2"]
    return rows
