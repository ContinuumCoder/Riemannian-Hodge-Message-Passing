"""Task registry: ``load_task(name, root, native=True, device='cuda') -> TaskData``.

Paper tasks (v1 pickles in ``datasets/*.pkl``; v1 protocol: sequential 70/15/15 split, train-part normalisation):

======  =========================================================================================================
name    content (legacy = v1 exact inputs/outputs;  native = cochain inputs/outputs where the physics lives)
======  =========================================================================================================
T1      CNS vorticity, 32x32 grid (triangulated). legacy: (rho,Vx,Vy,p) at nodes. native: (rho,p) at nodes +
        velocity 1-form on edges ``V_e = (V_s+V_d)/2 . (x_d-x_s)``.  target: vorticity at nodes.
T1q     T1 on the quad CW complex ``CochainComplex.from_grid(cell='quad')`` (same data, node order matched).
T2      torus advection-diffusion, scalar -> scalar at nodes (legacy == native).
T3      ellipsoid coexact flow psi -> v = n x grad psi.  node_vector readout in both modes; legacy keeps the v1
        per-component target normalisation (and v1-style ``direct`` vector readout), native uses isotropic scaling
        (E(n)-consistent) and the least-squares readout.
T5      Maxwell/Poisson rho -> E.  legacy: 2 node scalars (v1). native: node_vector (``direct`` readout = the
        generator's definition E_i = mean_e E_e t_e), isotropic target scaling.
T5g     T5 with the exact structure of its targets: readout ``grad`` (predict a potential phi, E_e = d0 phi on
        edges) + the fixed v1 edge->node average ``v_i = sum_e E_e t_e / deg_i`` (the generator's definition; the
        map applied to the generator's -d0 phi reproduces the stored targets to 3e-7).  Native only.
T6      U(1) Wilson loop. legacy: node encoding of theta. native: theta as an edge cochain in exact gauge
        connection mode (``connection_dims={1:1}``). target: plaquette flux averaged to nodes (v1 target).
T6f     native T6 inputs, target = plaquette flux ``d1 theta`` on faces (``cochain`` readout, degree 2).
T7      SU(2) Yang-Mills. legacy: node encoding of A (9 cols). native: A on edges (3 odd cols).  target F at nodes.
T7f     native T7 inputs, target = F = dA + [A,A] on faces (3 cols, ``cochain`` readout, degree 2).
T8      AirfRANS pressure, one mesh per sample (block-diagonal batches), v1 70/30 split with the last 10 % of the
        train part as validation set (v1 selected on the test part).  legacy == native (node scalars).
T8v     T8 + the inflow velocity as an odd edge 1-form ``V_inf . (x_d - x_s)`` (native only; node inputs, split,
        normalisation and block-diagonal batches as T8).  The scalar angle of attack alone cannot tell the pressure
        side from the suction side for a reflection-invariant model; the 1-form carries the flow direction.
T6_100K 100K-face Wilson loop (legacy only; used by the step benchmark).
======  =========================================================================================================

Orientation output maps (native T1/T1q/T3/T6/T7): vorticity, magnetic flux and ``n x grad psi`` are pseudo-scalars /
pseudo-vectors (defined w.r.t. the physical orientation), which the v2 model cannot output because it is exactly
equivariant under relabelling of cell orientations and reflection invariant.  The task then configures the model
with an orientation-free readout and applies a fixed ``rhmp.data.OutputMap`` that supplies the physical orientation:
T1/T1q/T6/T7 -> ``cochain:2`` + oriented face->node mean (sigma_f = +1 for CCW faces); T3 -> ``node_vector`` for the
true vector g, reported as v = n x g.  Targets and metrics stay exactly v1's.  ``load_task(..., output_map=False)``
disables it (ablation).  Legacy modes keep v1-like readouts (T1/T5 legacy inputs/targets are frame components, which
an isotropic equivariant model cannot relate to directions, so R2 stays near 0 there by construction).

Extension suite (``rhmp/tasks/suite.py``): SURF / SURF_heat (+ _geo / _topo transfer sets), DYN / DYNfix (+ _delta,
_cons, _cons_mass), HP_qual_graded / HP_qual_sliver (+ _ref) and the anisotropy tasks of ``rhmp/tasks/aniso.py``
(AHP, ASURF, ACURL, ADARCY) resolve through ``load_task`` as well (hook below; the paper tasks do not depend on those
modules).

Synthetic tasks (``datasets/v2/*.pt``, generators in ``datasets/generators/gen_*.py``), variable meshes:

* ``HP[<target>][_k<kappa>][_aniso[<R>]]`` (default kappa=100): 2-D hetero(-anisotropic) Poisson on variable meshes.
  Inputs: f (nodes); on edges (even) log sigma, or for anisotropic sets the log of the projected conductivity
  ``t^T Sigma t``; on faces (even) log sigma, or for the tensor sets ``_aniso<R>`` the invariants
  ``(log det Sigma, log anisotropy ratio)``.  ``_aniso10`` / ``_aniso100``: spatially varying SPD tensor field with
  a random principal direction and an eigenvalue ratio in [1, R]; ``_aniso``: legacy constant-ratio set.
  Targets (``<target>``): none -> u (nodes); ``flux`` -> ``-sigma_e d0 u`` (odd edge cochain, ``cochain:1``; its values
  scale with the edge length); ``fluxd`` -> flux density ``-sigma_e d0 u / |e|`` (resolution independent);
  ``fluxfem`` -> the conservative FEM flux ``-w_e d0 u`` (``w_e = -K_{src,dst}``; ``d0^T j = -M f`` exactly at
  interior nodes); ``grad`` -> ``E = -d0 u`` predicted through the exact readout ``grad`` (E = d0 phi, curl free).
  Extra test set ``fine``: first 500 test samples re-solved on meshes with 4x the nodes (zero-shot transfer).
* ``TET[<target>][_k<kappa>]``: 3-D tetrahedral Poisson; inputs f, log sigma on edges/faces/tets; same targets.
"""
from __future__ import annotations

import os
import re

from rhmp.data import Stats, TaskData

__all__ = ["TaskData", "Stats", "load_task", "list_tasks", "task_defaults", "TASK_DEFAULTS", "resolve_root"]

# v1 protocol defaults per task (v1 formal_benchmark.py TASKS: ours_C, batch 64, 4 layers)
TASK_DEFAULTS: dict[str, dict] = {
    "T1": dict(C=128, layers=4, batch=64),
    "T1q": dict(C=128, layers=4, batch=64),
    "T2": dict(C=64, layers=4, batch=64),
    "T3": dict(C=64, layers=4, batch=64),
    "T5": dict(C=160, layers=4, batch=64),
    "T5g": dict(C=160, layers=4, batch=64),
    "T6": dict(C=128, layers=4, batch=64),
    "T6f": dict(C=128, layers=4, batch=64),
    "T7": dict(C=160, layers=4, batch=64),
    "T7f": dict(C=160, layers=4, batch=64),
    "T8": dict(C=256, layers=4, batch=8),        # v1: per-sample (batch 1); v2: block-diagonal batches of 8 meshes
    "T8v": dict(C=256, layers=4, batch=8),
    "T6_100K": dict(C=16, layers=3, batch=2),
    "HP": dict(C=128, layers=4, batch=8),
    "TET": dict(C=128, layers=4, batch=8),
}

_HP_RE = re.compile(r"^HP(flux|fluxd|fluxfem|grad)?(?:_k(\d+))?(?:_aniso(\d*))?$")
_TET_RE = re.compile(r"^TET(flux|fluxd|fluxfem|grad)?(?:_k(\d+))?$")
_PAPER = ("T1", "T1q", "T2", "T3", "T5", "T5g", "T6", "T6f", "T7", "T7f", "T8", "T8v", "T6_100K")


def _accepts(fn, kwarg: str) -> bool:
    """True if ``fn`` (following ``functools.wraps``) takes the keyword ``kwarg`` or ``**kwargs``."""
    import inspect
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.kind is p.VAR_KEYWORD or p.name == kwarg for p in params)


def _suite():
    """The extension-suite registry ``rhmp.tasks.suite`` as ``(SUITE_TASKS, suite_task_defaults)``, or ``(None, None)``
    when it is missing or fails to import (the built-in tasks never depend on it)."""
    try:
        from rhmp.tasks.suite import SUITE_TASKS, suite_task_defaults
    except Exception:  # noqa: BLE001 - a broken optional module must not break the paper tasks
        return None, None
    return SUITE_TASKS, suite_task_defaults


def list_tasks() -> list[str]:
    """Canonical task names (HP/TET accept further ``_k<kappa>`` / ``_aniso`` suffixes; extension-suite names
    (SURF*, DYN*, HP_qual_*, anisotropy tasks) are appended when ``rhmp.tasks.suite`` is importable)."""
    suite, _ = _suite()
    extra = sorted(suite) if suite else []
    return list(_PAPER) + ["HP", "HPflux", "HPfluxd", "HPfluxfem", "HPgrad", "HP_aniso10", "HP_aniso100",
                           "HPflux_aniso100", "HP_k10", "HP_k1000", "TET", "TETflux", "TETfluxfem", "TETgrad"] + extra


def task_defaults(name: str) -> dict:
    """Training defaults (``C``, ``layers``, ``batch``) for a task name."""
    if name in TASK_DEFAULTS:
        return dict(TASK_DEFAULTS[name])
    if _HP_RE.match(name):
        return dict(TASK_DEFAULTS["HP"])
    if _TET_RE.match(name):
        return dict(TASK_DEFAULTS["TET"])
    suite, suite_defaults = _suite()
    if suite and name in suite:
        return suite_defaults(name)
    raise KeyError(f"unknown task {name!r}; known: {list_tasks()}")


def resolve_root(root: str | None) -> str:
    """Return the directory that contains the v1 ``*.pkl`` files (and ``v2/``).

    ``root`` may be the repository root (containing ``datasets/``) or the datasets directory itself.  ``None`` means
    the environment variable ``RHMP_DATA_ROOT`` if it is set (same two forms), else the repository this package
    lives in.
    """
    if root is None:
        root = os.environ.get("RHMP_DATA_ROOT") or \
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    root = os.path.expanduser(root)
    if os.path.isdir(os.path.join(root, "datasets")):
        return os.path.join(root, "datasets")
    return root


def load_task(name: str, root: str | None = None, *, native: bool = True, device="cuda", star: str = "cotan",
              **kw) -> TaskData:
    """Load a task as a GPU-resident, normalised :class:`TaskData`.

    Args:
        name: task name (see module docstring).
        root: repository root or datasets directory (``None``: ``$RHMP_DATA_ROOT`` if set, else this repository).
        native: native-cochain inputs/outputs (True) or v1 legacy inputs/outputs (False).
        device: target device of all tensors and complexes.
        star: reference Hodge star of the complexes (``cotan`` | ``barycentric`` | ``unit``; tetrahedral meshes
            use ``barycentric`` when ``cotan`` is requested).
        **kw: task-specific options (e.g. ``fine=False`` to skip the HP/TET resolution-transfer set,
            ``max_samples`` to truncate variable-mesh sets for smoke tests, ``cache=False``, ``whitney=False`` to
            drop the Whitney blocks that only tensor metrics use (41 % of a triangle complex), ``keep_on``,
            ``abs_scale=True`` to append the absolute mesh scale ``log(median edge length)`` as a constant even vertex
            column (the model is scale-free; use it when the physics has a fixed length scale), see
            :func:`rhmp.data.add_abs_scale`).
    Returns:
        TaskData.
    """
    abs_scale = kw.pop("abs_scale", False)
    td = _load(name, root, native=native, device=device, star=star, **kw)
    if abs_scale:
        from rhmp.data import add_abs_scale
        add_abs_scale(td)
    return td


def _load(name: str, root, *, native: bool, device, star: str, **kw) -> TaskData:
    droot = resolve_root(root)
    if name in _PAPER:
        from rhmp.tasks import paper
        return paper.load(name, droot, native=native, device=device, star=star, **kw)
    m = _HP_RE.match(name)
    if m:
        from rhmp.tasks import synthetic
        an = m.group(3)
        aniso = False if an is None else (True if an == "" else int(an))      # True: legacy constant-ratio set
        return synthetic.load_hp(name, droot, flux=m.group(1) or False, kappa=int(m.group(2) or 100),
                                 aniso=aniso, native=native, device=device, star=star, **kw)
    m = _TET_RE.match(name)
    if m:
        from rhmp.tasks import synthetic
        return synthetic.load_tet(name, droot, flux=m.group(1) or False, kappa=int(m.group(2) or 100),
                                  native=native, device=device, star=star, **kw)
    suite, _ = _suite()                       # extension suites (SURF*, DYN*, HP_qual_*, anisotropy tasks)
    if suite and name in suite:
        keep_whitney = kw.pop("whitney", True)
        loader = suite[name]
        if _accepts(loader, "whitney"):       # loaders that drop the blocks per complex while building (low peak)
            kw["whitney"] = keep_whitney
        td = loader(droot, native=native, device=device, star=star, **kw)
        if not keep_whitney:                  # no-op when the loader already dropped them
            from rhmp.data import strip_whitney
            strip_whitney(td)
        return td
    raise KeyError(f"unknown task {name!r}; known: {list_tasks()}")
