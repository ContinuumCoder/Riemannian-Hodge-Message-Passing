"""Model registry: ``MODELS = {name: builder(task_data, target_params, **overrides)}``, applicability, budgets.

    from rhmp.baselines.registry import MODELS, SPECS, applicable, build_model, rhmp_param_count
    target = rhmp_param_count(task)                    # trainable parameters of the v2 model on this task
    ok, why = applicable("sccnn", task)
    model, info = build_model("sccnn", task, target)   # width matched by the v1 rule: params in [target, 1.2 target]

Families (see ``docs/BASELINES.md``):

* v2 and its physics-prior controls: ``rhmp`` (default), ``dec_fixed``, ``unit_star``, ``unit_fixed``
  (same architecture, not param-matched; they keep the task's orientation output map);
* v1 model: ``ours_v1`` (paper configuration; shared triangle meshes only);
* MeshGraphNet: ``mgn`` (15 processor steps, standard) and ``mgn_fast`` (8 steps), width param-matched;
* v1 graph baselines ``gcn gat schnet egnn`` and mesh/gauge baselines ``gauge_cnn gem_cnn``;
* v1 cell-complex baselines ``mpsn sccnn cw_net clifford_smpn``;
* operator baselines ``fno`` (regular grids: T1, T1q) and ``deeponet`` (node targets).

All non-v2 models predict the task target directly (the orientation output maps of native T1/T3/T6/T7 exist
because the v2 model is exactly orientation-equivariant and reflection-invariant; the baselines are not, so the
trainer drops ``task.output_map`` for them).  Benchmark CLI::

    python3 -m rhmp.baselines.registry --budgets T6 T7 HP_k100          # matched widths / params per model
    python3 -m rhmp.baselines.registry --bench T6 --models rhmp,mgn,mgn_fast --batch 64 --steps 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn

from .adapters import TaskIO, task_io
from .param_match import DEFAULT_CACHE, V1_TOL, count_params, match_params

__all__ = ["ModelSpec", "MODELS", "SPECS", "applicable", "build_model", "model_names", "rhmp_param_count",
           "default_mode", "from_checkpoint", "prepare_task", "resolve_head", "LEGACY_SCALAR_COLUMNS", "LEGACY_TASKS",
           "V1_EDGE_MODELS", "V1_EDGE_TASKS", "main"]

# legacy (v1) node inputs of T6 / T7 are [avg(c), avg*dx(c), avg*dy(c)]: the c invariant 'avg' columns
# (v1 gave EGNN only these; v1 formal_benchmark.py)
LEGACY_SCALAR_COLUMNS = {"T6": [0], "T6_100K": [0], "T7": [0, 1, 2]}
# paper tasks whose legacy (v1) inputs differ from the native ones: node-input baselines default to legacy there
LEGACY_TASKS = ("T1", "T3", "T5", "T6", "T7", "T6_100K")
# v1 edge protocol (v1 formal_benchmark.py + compute_all_metrics.py): MPSN / SCCNN / Clifford-SMPN were
# trained with a 1-channel edge readout on the edge-averaged node target 0.5 (y_src + y_dst) and scored in edge space
# (first target component).  Default for these models on the shared-mesh paper tasks v1 reported them on.
V1_EDGE_MODELS = ("mpsn", "sccnn", "clifford_smpn")
V1_EDGE_TASKS = ("T1", "T2", "T3", "T5", "T6", "T7")


@dataclass
class ModelSpec:
    """Registry metadata.

    Attributes:
        name: registry name.
        family: ``'rhmp' | 'v1_model' | 'mgn' | 'v1_graph' | 'v1_mesh' | 'v1_complex' | 'operator'``.
        description: one line.
        param_matched: width chosen by :func:`rhmp.baselines.param_match.match_params`.
        uses_output_map: keeps the task's orientation output map (v2 family only).
        data_star: reference star the task must be loaded with (``None``: task default).
        node_inputs: the model reads node features (``NodeInputEncoder``); such models default to the legacy
            inputs on :data:`LEGACY_TASKS`.
    """
    name: str
    family: str
    description: str
    param_matched: bool = True
    uses_output_map: bool = False
    data_star: str | None = None
    node_inputs: bool = True


SPECS: dict[str, ModelSpec] = {s.name: s for s in [
    ModelSpec("rhmp", "rhmp", "RHMP v2: learned bounded metric around the DEC star", False, True, None, False),
    ModelSpec("dec_fixed", "rhmp", "v2 stack with frozen H = DEC star (geometry prior, no metric learning)",
              False, True, None, False),
    ModelSpec("unit_star", "rhmp", "v2 stack, star = 1 (combinatorial prior) + learned metric", False, True,
              "unit", False),
    ModelSpec("unit_fixed", "rhmp", "v2 stack, star = 1 and H = 1 frozen (combinatorial Hodge Laplacians)",
              False, True, "unit", False),
    ModelSpec("ours_v1", "v1_model", "v1 GaugeHodgeNetwork (paper config, per-cell metric bases)", False),
    ModelSpec("mgn", "mgn", "MeshGraphNet, 15 processor steps (standard), width param-matched", True, False,
              None, False),
    ModelSpec("mgn_fast", "mgn", "MeshGraphNet, 8 processor steps, width param-matched", True, False, None,
              False),
    ModelSpec("gcn", "v1_graph", "GCN (Kipf & Welling 2017), topology only"),
    ModelSpec("gat", "v1_graph", "GAT (Velickovic et al. 2018), 4 heads, LayerNorm, residual"),
    ModelSpec("schnet", "v1_graph", "SchNet (Schuett et al. 2017), RBF distance filters, E(n)-invariant"),
    ModelSpec("egnn", "v1_graph", "EGNN (Satorras et al. 2021) without coordinate updates, invariant inputs"),
    ModelSpec("gauge_cnn", "v1_mesh", "gauge-equivariant mesh CNN (Cohen et al. 2019, simplified), edge-angle "
              "channel-pair transport"),
    ModelSpec("gem_cnn", "v1_mesh", "GEM-CNN (de Haan et al. 2021, simplified): Fourier-angle kernels + transport"),
    ModelSpec("mpsn", "v1_complex", "MPSN (Bodnar et al. 2021, simplified): per-degree adjacency message passing"),
    ModelSpec("sccnn", "v1_complex", "SCCNN (Yang et al. 2022): Hodge-Laplacian polynomial filters per degree"),
    ModelSpec("cw_net", "v1_complex", "CW Network (Bodnar et al. 2021): boundary/coboundary messages"),
    ModelSpec("clifford_smpn", "v1_complex", "Clifford simplicial MP (Liu et al. 2024, simplified), Cl(2,0) "
              "multivectors on nodes/edges"),
    ModelSpec("fno", "operator", "FNO (Li et al. 2021) on the 32x32 grid"),
    ModelSpec("deeponet", "operator", "DeepONet (Lu et al. 2021): branch on the mesh mean, trunk on positions"),
]}


def model_names() -> list[str]:
    """All registered names."""
    return list(SPECS)


def default_mode(name: str, task_name: str) -> str:
    """``'legacy'`` for node-input baselines on paper tasks with distinct legacy inputs, else ``'native'``."""
    spec = SPECS[name]
    return "legacy" if (spec.node_inputs and task_name in LEGACY_TASKS) else "native"


# ================================================================================================================
# applicability
# ================================================================================================================
def applicable(name: str, td: Any) -> tuple[bool, str]:
    """Whether model ``name`` can be trained on task ``td`` (``(True, '')`` or ``(False, reason)``)."""
    if name not in SPECS:
        return False, f"unknown model {name!r}; known: {model_names()}"
    io = task_io(td)
    t = io.target_degree
    if SPECS[name].family == "rhmp":
        return True, ""
    if name == "ours_v1":
        if io.variable_mesh:
            return False, "ours_v1 has per-cell parameters (n_k x 8 metric bases): shared-mesh tasks only"
        if io.cell_type != "triangle" or io.top_dim != 2:
            return False, f"ours_v1 (v1 CellComplex) needs a triangle surface mesh, task has {io.cell_type}"
        if t >= 2:
            return False, "ours_v1 has no face/tet readout (v1 tasks: scalar, edge_scalar, vector)"
        if io.target_kind == "node_vector" and io.out_dim != io.spatial_dim:
            return False, "ours_v1's vector readout predicts exactly one D-dimensional field"
        return True, ""
    if name == "fno":
        if io.variable_mesh or io.grid is None:
            return False, "fno needs a regular grid numbered i*ny+j (T1, T1q)"
        if t != 0:
            return False, "fno predicts node (grid) values only"
        return True, ""
    if name == "deeponet":
        if t != 0:
            return False, "deeponet (trunk evaluated at the nodes) predicts node targets only"
        return True, ""
    if name in ("gauge_cnn", "gem_cnn"):
        if io.top_dim != 2:
            return False, f"{name} is a 2-manifold (surface) method; the complex has dimension {io.top_dim}"
        return True, ""
    return True, ""


# ================================================================================================================
# builders
# ================================================================================================================
def _need_target(target_params: int | None, name: str) -> int:
    if target_params is None:
        raise ValueError(f"{name}: pass target_params (e.g. rhmp_param_count(task)) or an explicit hidden width")
    return int(target_params)


def _matched(name: str, io: TaskIO, make: Callable[..., nn.Module], target_params: int | None, *,
             hidden: int | None, sig: dict, tol: float, cache_path: str | None, start: int = 16, step: int = 4,
             refine_step: int | tuple[int, ...] = 1, options: list[dict] | None = None) -> tuple[int, dict, dict]:
    """Width (and option) for ``make``; returns ``(hidden, option, match_info)``."""
    if hidden is not None:
        return int(hidden), {}, {"hidden": int(hidden), "matched": False}
    target = _need_target(target_params, name)
    r = match_params(make, target, tol, start=start, step=step, options=options, key=(name, io.name),
                     signature=sig, cache_path=cache_path, refine_step=refine_step)
    return r.hidden, r.option, dict(r.to_dict(), matched=True)


def _io_sig(io: TaskIO) -> dict:
    return dict(in_dims=io.in_dims, even_dims=io.even_dims, D=io.spatial_dim, t=io.target_degree,
                kind=io.target_kind, out=io.out_dim, native=io.native)


def _finish(model: nn.Module, td: Any, name: str, match: dict, target_params: int | None) -> nn.Module:
    enc = getattr(model, "encoder", None)
    if enc is not None and hasattr(enc, "fit"):
        enc.fit(td)
    n = count_params(model)
    info = dict(getattr(model, "info", {}) or {})
    info.update(name=name, params=n, target_params=None if target_params is None else int(target_params),
                ratio=None if not target_params else n / int(target_params), match=match)
    model.info = info
    return model


# dec_fixed / unit_fixed freeze the metric heads at their zero initialisation (H = star exactly, tested);
# RHMPConfig(learn_metric=False) is the same operator with the metric heads removed instead of frozen.
def _build_rhmp_family(name: str):
    def build(td, target_params=None, *, args=None, amp=None, cache_path=None, tol=None, **cfg_overrides):
        from .dec_fixed import make_rhmp_variant
        # cache_path / tol are generic registry options (the v2 family is not param-matched); any other keyword
        # is an RHMPConfig field (unknown names raise in RHMPConfig.from_dict)
        if amp is not None:
            cfg_overrides["amp"] = bool(amp)
        model = make_rhmp_variant(name, td, args, **cfg_overrides)
        return _finish(model, td, name, {"matched": False, "note": "same architecture as rhmp"}, target_params)
    return build


def _build_v1(name: str):
    def build(td, target_params=None, *, hidden=None, n_layers=4, geometry="normalized", v1_exact=False,
              normalize=None, filter_init=None, head="auto", v1_components="first", invariant_inputs=None,
              amp=False, tol=V1_TOL, cache_path=DEFAULT_CACHE, **_):
        from .v1_wrappers import make_v1_baseline
        io = task_io(td)
        directional, cols = True, None
        if name == "egnn" and invariant_inputs is not False:          # v1 protocol: EGNN sees invariant inputs
            directional = False
            if not io.native and _base_name(td) in LEGACY_SCALAR_COLUMNS:
                cols = LEGACY_SCALAR_COLUMNS[_base_name(td)]
        head_mode = resolve_head(name, td, head) or "node"
        if head_mode == "v1_edge" and td.target_degree == 0:
            raise ValueError(f"{name}: the v1 edge protocol needs the edge-averaged target; call "
                             f"rhmp.baselines.registry.prepare_task('{name}', task, opts) first (rhmp.train does)")
        # SCCNN: v1's raw Laplacians under the v1 protocol; normalised Laplacians with identity-initialised filters
        # otherwise (raw ones explode ~1e2 per layer, normalised ones with v1's init collapse to a constant)
        if normalize is None:
            normalize = head_mode != "v1_edge"
        if filter_init is None:
            filter_init = "identity" if normalize else "v1"
        kw = dict(n_layers=n_layers, geometry=geometry, v1_exact=v1_exact, normalize=normalize,
                  filter_init=filter_init, head=head_mode, directional=directional, node_columns=cols, amp=amp)

        def make(h):
            return make_v1_baseline(name, io, h, **kw)
        sig = dict(_io_sig(io), **{k: v for k, v in kw.items() if k != "amp"})
        h, _, match = _matched(name, io, make, target_params, hidden=hidden, sig=sig, tol=tol, cache_path=cache_path)
        model = make(h)
        if head_mode == "v1_edge":
            model.build["v1_components"] = v1_components
            model.info["target_protocol"] = td.meta.get("v1_edge_protocol")
        return _finish(model, td, name, match, target_params)
    return build


def _build_mgn(name: str, default_layers: int):
    def build(td, target_params=None, *, hidden=None, n_layers=None, mlp_layers=2, act="relu",
              normalized_geometry=True, directional_edges=True, node_edge_encoding=True, amp=False, tol=V1_TOL,
              cache_path=DEFAULT_CACHE, **_):
        from .mgn import make_mgn
        io = task_io(td)
        L = int(n_layers or default_layers)
        kw = dict(mlp_layers=mlp_layers, act=act, normalized_geometry=normalized_geometry,
                  directional_edges=directional_edges, node_edge_encoding=node_edge_encoding)

        def make(h):
            return make_mgn(io, h, L, name=name, amp=amp, **kw)
        sig = dict(_io_sig(io), L=L, **kw)
        # widths in multiples of 8, refined in steps of 4 (widths not divisible by 4, e.g. 30, fall off the fast
        # vectorised gather/scatter/GEMM kernels: T6, L=8, H=30 is 1.8x slower than H=32), and in steps of 1 only
        # when no multiple of 4 meets the budget tolerance (budget parity first)
        h, _, match = _matched(name, io, make, target_params, hidden=hidden, sig=sig, tol=tol, cache_path=cache_path,
                               start=8, step=8, refine_step=(4, 1))
        return _finish(make(h), td, name, match, target_params)
    return build


def _build_ours_v1(td, target_params=None, *, C=None, eval_per_sample=True, amp=False, **_):
    from .v1_wrappers import make_ours_v1
    io = task_io(td)
    C = int(C or td.meta.get("v1_C", 128))
    model = make_ours_v1(io, C, eval_per_sample=eval_per_sample, amp=amp)
    return _finish(model, td, "ours_v1", {"matched": False, "note": "v1 paper configuration"}, target_params)


def _build_deeponet(td, target_params=None, *, hidden=None, n_basis=64, amp=False, tol=V1_TOL,
                    cache_path=DEFAULT_CACHE, **_):
    from .v1_wrappers import make_deeponet
    io = task_io(td)

    def make(h):
        return make_deeponet(io, h, n_basis=n_basis, amp=amp)
    h, _, match = _matched("deeponet", io, make, target_params, hidden=hidden, sig=dict(_io_sig(io), nb=n_basis),
                           tol=tol, cache_path=cache_path)
    return _finish(make(h), td, "deeponet", match, target_params)


def _build_fno(td, target_params=None, *, hidden=None, modes=None, n_layers=4, amp=False, tol=V1_TOL,
               cache_path=DEFAULT_CACHE, **_):
    from .v1_wrappers import make_fno
    io = task_io(td)

    def make(h, modes=12):
        return make_fno(io, h, modes=modes, n_layers=n_layers, amp=amp)
    # v1: modes=12; smaller budgets fall back to fewer modes (first option inside the tolerance)
    options = [{"modes": int(modes)}] if modes else [{"modes": m} for m in (12, 8, 6, 4, 3, 2)]
    if hidden is not None:
        m = int(modes or 12)
        return _finish(make(int(hidden), m), td, "fno", {"hidden": int(hidden), "matched": False}, target_params)
    target = _need_target(target_params, "fno")
    r = match_params(make, target, tol, options=options, key=("fno", io.name),
                     signature=dict(_io_sig(io), L=n_layers, modes=modes), cache_path=cache_path)
    return _finish(make(r.hidden, **r.option), td, "fno", dict(r.to_dict(), matched=True), target_params)


def _base_name(td: Any) -> str:
    """Task name before any protocol transform."""
    return td.meta.get("untransformed").name if td.meta.get("untransformed") is not None else td.name


def resolve_head(name: str, td: Any, head: str = "auto") -> str | None:
    """Head mode of MPSN / SCCNN / Clifford-SMPN on ``td`` (``None`` for other models).

    ``'auto'``: ``'v1_edge'`` on the shared-mesh paper tasks of :data:`V1_EDGE_TASKS` with node targets (v1's
    protocol), ``'node'`` elsewhere.  A task already transformed by :func:`prepare_task` resolves to ``'v1_edge'``.
    """
    if name not in V1_EDGE_MODELS:
        return None
    if td.meta.get("v1_edge_protocol"):
        return "v1_edge"
    if head == "auto":
        return "v1_edge" if (td.name in V1_EDGE_TASKS and td.target_degree == 0 and not td.variable_mesh) else "node"
    if head not in ("node", "v1_edge"):
        raise ValueError(f"head must be 'auto', 'node' or 'v1_edge', got {head!r}")
    return head


def prepare_task(name: str, td: Any, opts: dict | None = None) -> Any:
    """Task as model ``name`` must be trained and scored on (identity except for the v1 edge protocol).

    v1 edge protocol (MPSN / SCCNN / Clifford-SMPN, ``head`` resolving to ``'v1_edge'``): the node target ``y``
    becomes the edge-averaged target ``0.5 (y_src + y_dst)`` on the canonical edges (in normalised units, which equals
    v1's raw-space average normalised with the node statistics), the target cells become the edges (``even:1``), and
    - with ``v1_components='first'`` (default, as v1) - only the first target component is kept.  Metrics are then
    edge-space metrics, exactly as the v1 evaluation script ``compute_all_metrics.py`` scored these models.

    Args:
        name: registry name.
        td: :class:`rhmp.data.TaskData`.
        opts: builder options (``head``, ``v1_components``).
    Returns:
        ``td`` itself, or a transformed copy (``meta['v1_edge_protocol']`` describes it, ``meta['untransformed']``
        holds the original task).
    """
    import dataclasses

    from rhmp.data import Stats
    opts = dict(opts or {})
    if resolve_head(name, td, opts.get("head", "auto")) != "v1_edge" or td.target_degree != 0:
        return td
    if td.variable_mesh:
        raise ValueError("the v1 edge protocol is defined for shared-mesh tasks only")
    comps = opts.get("v1_components", "first")
    if comps not in ("first", "all"):
        raise ValueError(f"v1_components must be 'first' or 'all', got {comps!r}")
    e = td.K.cells[1].long().to(td.target.device)
    y = td.target
    ye = 0.5 * (y.index_select(1, e[:, 0]) + y.index_select(1, e[:, 1]))          # (N, n1, O)
    ys, r2s = td.y_stats, td.meta.get("r2_std")
    if comps == "first" and ye.shape[-1] > 1:
        ye = ye[..., :1]
        ys = Stats(ys.mean[:1], ys.std[:1], ys.kind)
        r2s = None if r2s is None else r2s[:1]
    desc = (f"v1 edge protocol: target 0.5 (y_src + y_dst) on the {td.K.n[1]} edges"
            + (", first component" if comps == "first" and td.out_dim > 1 else "") + "; edge-space metrics")
    meta = dict(td.meta, r2_std=r2s, v1_edge_protocol=desc, untransformed=td)
    return dataclasses.replace(td, target=ye.contiguous(), target_degree=1, target_kind="even",
                               out_dim=int(ye.shape[-1]), y_stats=ys, meta=meta, output_map=None)


MODELS: dict[str, Callable[..., nn.Module]] = {
    "rhmp": _build_rhmp_family("rhmp"),
    "dec_fixed": _build_rhmp_family("dec_fixed"),
    "unit_star": _build_rhmp_family("unit_star"),
    "unit_fixed": _build_rhmp_family("unit_fixed"),
    "ours_v1": _build_ours_v1,
    "mgn": _build_mgn("mgn", 15),
    "mgn_fast": _build_mgn("mgn_fast", 8),
    **{n: _build_v1(n) for n in ("gcn", "gat", "schnet", "egnn", "gauge_cnn", "gem_cnn", "mpsn", "sccnn", "cw_net",
                                 "clifford_smpn")},
    "fno": _build_fno,
    "deeponet": _build_deeponet,
}


def build_model(name: str, td: Any, target_params: int | None = None, **overrides) -> tuple[nn.Module, dict]:
    """Build model ``name`` for task ``td``.

    Args:
        name: registry name.
        td: :class:`rhmp.data.TaskData` (complexes on the target device).
        target_params: parameter budget for param-matched models (default: the v2 model's count on ``td``).
        **overrides: builder options (``hidden``, ``n_layers``, ``amp``, ``args`` for the v2 family, ...).
    Returns:
        ``(model, info)`` (model on the device of the task's complexes).
    Raises:
        ValueError: if the model is not applicable to the task.
    """
    ok, why = applicable(name, td)
    if not ok:
        raise ValueError(f"{name} is not applicable to {td.name}: {why}")
    if target_params is None and SPECS[name].param_matched and "hidden" not in overrides:
        target_params = rhmp_param_count(td, overrides.get("args"))
    if resolve_head(name, td, overrides.get("head", "auto")) == "v1_edge" and td.target_degree == 0:
        raise ValueError(f"{name} on {td.name} uses the v1 edge protocol: pass prepare_task('{name}', task, opts) "
                         "(the edge-averaged target) to build_model; rhmp.train does this")
    model = MODELS[name](td, target_params, **overrides)
    K0 = td.K[0] if td.variable_mesh else td.K
    model = model.to(K0.pos.device)
    return model, dict(model.info)


def rhmp_param_count(td: Any, args: Any | None = None) -> int:
    """Trainable parameters of the v2 model that ``rhmp.train`` builds for ``td`` (the default budget; for a task
    transformed by :func:`prepare_task`, the v2 model of the original task)."""
    from rhmp.model import RHMP

    from .dec_fixed import rhmp_config
    td = td.meta.get("untransformed") or td
    with torch.random.fork_rng(devices=[]):
        try:
            with torch.device("meta"):
                m = RHMP(rhmp_config(td, args), td.geo_dims)
        except Exception:  # noqa: BLE001
            m = RHMP(rhmp_config(td, args), td.geo_dims)
    return count_params(m)


def from_checkpoint(ck: dict, td: Any, map_location: Any = None) -> nn.Module:
    """Rebuild a model saved with ``to_checkpoint()`` (baselines: ``{'model', 'build', 'state_dict'}``)."""
    name = ck.get("model", "rhmp")
    if SPECS.get(name) and SPECS[name].family == "rhmp":
        from rhmp.model import RHMP, RHMPConfig
        from .dec_fixed import FixedMetricRHMP, VARIANTS
        cfg = RHMPConfig.from_dict(ck["cfg"])
        m = FixedMetricRHMP(cfg, ck["geo_dims"], name) if VARIANTS[name]["freeze_metric"] else RHMP(cfg, ck["geo_dims"])
    else:
        m = MODELS[name](td, None, **ck["build"])
    m.load_state_dict(ck["state_dict"])
    return m.to(map_location) if map_location is not None else m


# ================================================================================================================
# CLI: budgets and step benchmark
# ================================================================================================================
def _load(task: str, native: bool, device, star: str = "cotan", max_samples: int | None = None):
    from rhmp.tasks import load_task
    kw = {"fine": False} if task.startswith(("HP", "TET")) else {}
    if max_samples:
        kw["max_samples"] = max_samples
    return load_task(task, None, native=native, device=device, star=star, **kw)


def _budgets(tasks: list[str], models: list[str], device, cache_path: str) -> list[dict]:
    rows = []
    for t in tasks:
        loaded: dict = {}
        for name in models:
            mode = default_mode(name, t)
            star = SPECS[name].data_star or "cotan"
            key = (mode, star)
            if key not in loaded:
                try:
                    loaded[key] = _load(t, mode == "native", device, star, max_samples=30)
                except Exception as e:  # noqa: BLE001
                    loaded[key] = e
            td = loaded[key]
            if isinstance(td, Exception):
                rows.append(dict(task=t, model=name, mode=mode, error=str(td)))
                continue
            ok, why = applicable(name, td)
            if not ok:
                rows.append(dict(task=t, model=name, mode=mode, applicable=False, reason=why))
                continue
            target = rhmp_param_count(td)
            _, info = build_model(name, prepare_task(name, td), target, cache_path=cache_path)
            rows.append(dict(task=t, model=name, mode=mode, applicable=True, target=target, params=info["params"],
                             ratio=info["ratio"], hidden=info.get("hidden", info.get("C")),
                             n_layers=info.get("n_layers"), within_tol=info["match"].get("within_tol"),
                             option=info["match"].get("option")))
            print(json.dumps(rows[-1]), flush=True)
    return rows


def _other_gpu_pids() -> list[str]:
    """PIDs of other compute processes on the (first visible) GPU, from ``nvidia-smi``."""
    import subprocess
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    try:
        out = subprocess.run(["nvidia-smi", "-i", gpu, "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:  # noqa: BLE001
        return []
    return [q.strip() for q in out.splitlines() if q.strip() and q.strip() != str(os.getpid())]


MAX_WAIT_S = [1800.0]      # --max-wait (0: measure on a shared GPU without waiting)


def _wait_exclusive(max_wait_s: float | None = None, poll_s: float = 10.0) -> bool:
    """Block until no other process computes on our GPU (as ``bench/step_bench.py``); False after ``max_wait_s``."""
    max_wait_s = MAX_WAIT_S[0] if max_wait_s is None else max_wait_s
    t0 = time.time()
    while True:
        others = _other_gpu_pids()
        if not others:
            return True
        if time.time() - t0 >= max_wait_s:
            return False
        print(f"  waiting for an exclusive GPU (other pids {others})", flush=True)
        time.sleep(poll_s)


def _bench(task: str, models: list[str], device, batch: int, steps: int, amp: bool, compile_: bool) -> list[dict]:
    """Train-step / inference time and peak memory of each model on one real batch of ``task``.

    Protocol: 1 + 3 warm-up steps, then two passes of ``steps`` timed steps (Adam step included), best pass reported;
    each model waits for an exclusive GPU first (``exclusive`` flag in the row)."""
    import torch.nn.functional as F

    from rhmp.data import mesh_minibatch, shared_minibatch
    rows = []
    loaded: dict = {}
    for name in models:
        mode = default_mode(name, task)
        star = SPECS[name].data_star or "cotan"
        if (mode, star) not in loaded:
            loaded[(mode, star)] = _load(task, mode == "native", device, star,
                                         max_samples=None if not task.startswith(("HP", "TET", "T8")) else 3 * batch)
        td = loaded[(mode, star)]
        ok, why = applicable(name, td)
        if not ok:
            rows.append(dict(model=name, task=task, mode=mode, applicable=False, reason=why))
            continue
        torch.manual_seed(0)
        target = rhmp_param_count(td) if SPECS[name].param_matched else None
        td = prepare_task(name, td)
        model, info = build_model(name, td, target, amp=amp)
        model.record_diagnostics = False
        idx = td.split[0][:batch]
        if td.variable_mesh:
            K, xb, yb = mesh_minibatch(td.K, td.inputs, td.target, idx, device=device)
        else:
            K = td.K
            xb, yb = shared_minibatch(td.inputs, td.target, idx)
        omap = td.output_map if SPECS[name].uses_output_map else None
        net = torch.compile(model, dynamic=td.variable_mesh) if compile_ else model
        fwd = (lambda x, k: omap(net(x, k))) if omap is not None else net
        opt = torch.optim.Adam(model.parameters(), 1e-3)

        def step():
            opt.zero_grad(set_to_none=True)
            F.mse_loss(fwd(xb, K), yb).backward()
            opt.step()

        def sync():
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize()
        cuda = torch.device(device).type == "cuda"
        excl = _wait_exclusive() if cuda else True
        model.train()
        t0 = time.perf_counter()
        step()
        sync()
        first = time.perf_counter() - t0
        for _ in range(3):
            step()
        sync()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        passes = []
        for _ in range(2):
            t0 = time.perf_counter()
            for _ in range(steps):
                step()
            sync()
            passes.append((time.perf_counter() - t0) / steps)
        dt = min(passes)
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else 0.0
        model.eval()
        with torch.no_grad():
            for _ in range(3):
                fwd(xb, K)
            sync()
            ipasses = []
            for _ in range(2):
                t0 = time.perf_counter()
                for _ in range(steps):
                    fwd(xb, K)
                sync()
                ipasses.append((time.perf_counter() - t0) / steps)
            di = min(ipasses)
        excl = excl and (not cuda or not _other_gpu_pids())
        rows.append(dict(model=name, task=task, mode=mode, params=info["params"], hidden=info.get("hidden",
                         info.get("C")), n_layers=info.get("n_layers"), batch=int(idx.numel()),
                         train_ms=dt * 1e3, infer_ms=di * 1e3, peak_GB=peak, first_step_s=first, amp=amp,
                         compile=compile_, exclusive=excl))
        print(json.dumps(rows[-1]), flush=True)
        del model, opt, net, fwd
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    return rows


def _applicable_rows(task: str, models: list[str], device) -> list[dict]:
    """One row per model: default input mode for ``task`` and applicability (loads a few samples only)."""
    rows, loaded = [], {}
    for name in models:
        if name not in SPECS:
            rows.append(dict(model=name, task=task, mode="-", applicable=False, reason="unknown model"))
            continue
        mode = default_mode(name, task)
        star = SPECS[name].data_star or "cotan"
        if (mode, star) not in loaded:
            loaded[(mode, star)] = _load(task, mode == "native", device, star, max_samples=6)
        ok, why = applicable(name, loaded[(mode, star)])
        rows.append(dict(model=name, task=task, mode=mode, applicable=ok, reason=why))
    return rows


def _bench_v1_original(task: str, models: list[str], device, batch: int, steps: int) -> list[dict]:
    """Train-step time of the *unmodified* v1 modules (their own ``forward``/``forward_batch`` on the v1-compatible
    adapter: per-sample loops, dense adjacency / Laplacians rebuilt every call) at the matched widths, for comparison
    with the vectorised cores.  Shared-mesh tasks, legacy inputs."""
    import torch.nn.functional as F

    from .adapters import adapter_for
    from .v1_wrappers import COMPLEX_CORES, NODE_CORES, _v1_class
    td = _load(task, False, device)
    if td.variable_mesh:
        raise ValueError("--v1-original needs a shared-mesh task")
    rows = []
    A = adapter_for(td.K)
    idx = td.split[0][:batch].to(td.target.device)
    X = td.inputs[0].index_select(0, idx)                                   # (B, n0, F)
    Y = td.target.index_select(0, idx)
    target = rhmp_param_count(td)
    for name in models:
        if name not in NODE_CORES and name not in COMPLEX_CORES:
            continue
        m_v2, info = build_model(name, prepare_task(name, td), target)
        h = info["hidden"]
        cls = _v1_class(name)
        torch.manual_seed(0)
        f_in = X.shape[-1] if name != "egnn" else m_v2.encoder.out_dim
        Xb = X if name != "egnn" else X[..., m_v2.encoder.node_columns or slice(None)]
        if name in NODE_CORES:
            m = cls(f_in=f_in, hidden=h, n_layers=4, out_dim=Y.shape[-1], task="node").to(device)
            yref = Y
        else:
            m = cls(f_in=f_in, hidden=h, n_layers=4).to(device)
            yref = torch.zeros(X.shape[0], A.n1, 1, device=X.device)
        opt = torch.optim.Adam(m.parameters(), 1e-3)
        fb = m.forward_batch if hasattr(m, "forward_batch") else None

        def step():
            opt.zero_grad(set_to_none=True)
            F.mse_loss(fb(Xb, A), yref).backward()
            opt.step()
        cuda = torch.device(device).type == "cuda"
        excl = _wait_exclusive() if cuda else True
        for _ in range(2):
            step()
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        n = max(2, steps // 4)
        t0 = time.perf_counter()
        for _ in range(n):
            step()
        if cuda:
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else 0.0
        rows.append(dict(model=name + "[v1 original]", task=task, mode="legacy", params=count_params(m), hidden=h,
                         n_layers=4, batch=int(idx.numel()), train_ms=dt * 1e3, infer_ms=float("nan"),
                         peak_GB=peak, exclusive=excl and (not cuda or not _other_gpu_pids())))
        print(json.dumps(rows[-1]), flush=True)
        del m, opt, m_v2
        if cuda:
            torch.cuda.empty_cache()
    return rows


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--applicable", default=None, metavar="TASK",
                   help="print '<model> <mode> <yes|no> <reason>' per model for TASK (used by run_baselines.sh)")
    p.add_argument("--budgets", nargs="*", default=None, metavar="TASK", help="print matched budgets per model")
    p.add_argument("--bench", default=None, metavar="TASK", help="train-step benchmark on one batch of TASK")
    p.add_argument("--models", default=None, help="comma-separated model names (default: all)")
    p.add_argument("--batch", type=int, default=64, help="samples (shared mesh) or meshes (variable) per step")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--max-wait", type=float, default=1800.0,
                   help="seconds to wait for an exclusive GPU per model (0 = measure on a shared GPU)")
    p.add_argument("--v1-original", action="store_true",
                   help="with --bench: time the unmodified v1 modules instead of the vectorised cores")
    p.add_argument("--device", default=None)
    p.add_argument("--cache", default=DEFAULT_CACHE)
    p.add_argument("--json", default=None, help="write the rows to this file")
    a = p.parse_args(argv)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if torch.device(device).type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    models = a.models.split(",") if a.models else model_names()
    MAX_WAIT_S[0] = float(a.max_wait)
    rows: list[dict] = []
    if a.applicable:
        for r in _applicable_rows(a.applicable, models, device):
            print(f"{r['model']} {r['mode']} {'yes' if r['applicable'] else 'no'} {r['reason']}", flush=True)
        return
    if a.budgets is not None:
        rows += _budgets(a.budgets, models, device, a.cache)
    if a.bench and a.v1_original:
        rows += _bench_v1_original(a.bench, models, device, a.batch, a.steps)
    elif a.bench:
        rows += _bench(a.bench, models, device, a.batch, a.steps, a.amp, a.compile)
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
