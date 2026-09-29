"""Evaluate a trained v2 run on the test sets of any task, or roll it out autoregressively (DYN tasks).

The script needs the run directory (``config.json``, ``best.pt``) and the data set of the evaluated task, and uses a
GPU when one is available.  It writes ``<out>/eval_<task>.json``, or ``<out>/rollout_<task>.json`` for rollouts
(``--out`` defaults to the run directory).

    # an HP_k100 model on a mesh-quality shift set (one result per quality level)
    python3 scripts/eval_on.py --run runs/hp_k100 --task HP_qual_graded_ref
    # autoregressive rollouts of a DYN / DYNfix model over the full 100-step test trajectories
    python3 scripts/eval_on.py --run runs/dyn --task DYN --rollout [--steps 100] [--project-mass]

Tasks are loaded with ``rhmp.tasks.load_task`` when the name is registered there, otherwise from
``rhmp.tasks.suite.SUITE_TASKS``.  This includes the transfer and quality-shift sets, e.g. ``SURF_geo`` /
``SURF_topo`` (the geometry and topology transfer of SURF, which the trainer also reports as ``geo_*`` / ``topo_*``)
and ``HP_qual_sliver`` / ``HP_qual_graded`` (with ``_ref``: targets from the 4x finer reference solution).  Inputs and
targets are re-expressed in the normalisation of the training run (``rhmp.train._renormalize``), and the complexes use
the run's reference star.  Node-scalar tasks also report the relative L2 error weighted with the lumped mass
(``relL2M_*``, ``rhmp.tasks.suite.mass_weighted_errors``), which does not depend on the mesh density and is the
quantity to compare across the levels of the graded meshes (the node-pooled R2 and NRMSE depend on where the nodes
are).

Runs of ``DYN_cons`` / ``DYNfix_cons`` trained with the first conservative convention (target: the mass change
``M du``, readout ``div:1``; no ``cons_convention`` tag in the run's data summary) are evaluated with that convention
(tasks ``DYN_cons_mass`` / ``DYNfix_cons_mass``).  ``--project-mass`` restores ``sum M u`` after every rollout step
(post-hoc exact conservation, reported as ``rollout_<task>_proj.json``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402


def load_any_task(name: str, root, *, native: bool, device, star: str, **kw):
    """``rhmp.tasks.load_task`` if it knows ``name``, else the suite registry."""
    from rhmp.tasks import load_task
    try:
        return load_task(name, root, native=native, device=device, star=star, **kw)
    except KeyError:
        from rhmp.tasks.suite import SUITE_TASKS
        if name not in SUITE_TASKS:
            raise
        from rhmp.tasks import resolve_root
        return SUITE_TASKS[name](resolve_root(root), native=native, device=device, star=star, **kw)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory with config.json and best.pt")
    ap.add_argument("--task", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--eval-batch", type=int, default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--rollout", action="store_true", help="autoregressive rollout (DYN / DYNfix)")
    ap.add_argument("--steps", type=int, default=None, help="rollout horizon (default: stored trajectory length)")
    ap.add_argument("--save-pred", action="store_true", help="also save the rollout predictions (.pt)")
    ap.add_argument("--project-mass", action="store_true",
                    help="rollout: shift each trajectory after every step so that sum M u is conserved exactly")
    a = ap.parse_args(argv)

    from rhmp import metrics as M
    from rhmp.model import RHMP
    from rhmp.tasks.suite import mass_weighted_errors
    from rhmp.train import _jsonable, _renormalize, predict
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    conf = json.load(open(os.path.join(a.run, "config.json")))
    rargs = conf["args"]
    kw = {} if a.max_samples is None else {"max_samples": a.max_samples}
    task_name = a.task
    dmeta = conf.get("data", {}).get("meta", {}) or {}
    if task_name in ("DYN_cons", "DYNfix_cons") and dmeta.get("target") == "cons" and "cons_convention" not in dmeta:
        task_name = task_name + "_mass"          # trained with the first convention (target M du, readout div:1)
        print(f"note: {a.run} was trained with the mass-change convention -> evaluating as {task_name}", flush=True)
    t0 = time.time()
    task = load_any_task(task_name, a.root or rargs.get("root"), native=rargs.get("native", True), device=device,
                         star=rargs.get("star", "cotan"), **kw)
    t_load = time.time() - t0
    task = _renormalize(task, conf["data"])
    ck = torch.load(os.path.join(a.run, "best.pt"), map_location=device, weights_only=False)
    model = RHMP.from_checkpoint(ck, map_location=device).to(device)
    model.eval()
    out_dir = a.out or a.run
    os.makedirs(out_dir, exist_ok=True)
    res = dict(run=a.run, train_task=rargs.get("task"), eval_task=a.task, loaded_task=task_name, load_seconds=t_load)
    if a.rollout:
        from rhmp.tasks.suite import rollout_eval
        r = rollout_eval(model, task, a.steps, device=device, return_pred=a.save_pred, project_mass=a.project_mass)
        suffix = "_proj" if a.project_mass else ""
        if a.save_pred:
            torch.save(r.pop("pred"), os.path.join(out_dir, f"rollout_{a.task}{suffix}_pred.pt"))
        res.update(r)
        path = os.path.join(out_dir, f"rollout_{a.task}{suffix}.json")
        msg = " ".join(f"{k}={v:.4f}" for k, v in r["summary"].items() if k.startswith(("R2@", "mass_drift@")))
    else:
        ebs = a.eval_batch or (rargs.get("bs_meshes") or task.meta.get("batch_size", 8) if task.variable_mesh
                               else rargs.get("batch") or 64)
        t1 = time.time()
        sets = ([("test", None, task.split[2])] if len(task.split[2]) else []) + \
            [(name, src, torch.arange(len(src["target"]))) for name, src in task.extra_tests.items()]
        for name, src, idx in sets:
            pred, tgt = predict(model, task, idx, ebs, device, source=src)
            res[name] = M.summarize(pred, tgt, task.y_stats.as_tuple(), r2_std=task.meta.get("r2_std"))
            if task.target_kind == "node_scalar":      # mesh-density independent error (graded meshes)
                res[name].update(mass_weighted_errors(task, pred, source=src, idx=idx))
        res["seconds"] = time.time() - t1
        for k in ("levels", "quality"):
            if k in task.meta:
                res[k] = task.meta[k]
        path = os.path.join(out_dir, f"eval_{a.task}.json")
        msg = " ".join(f"{k}: R2={v['R2']:.4f}" + (f" relL2M={v['relL2M_pooled']:.4f}" if "relL2M_pooled" in v else "")
                       for k, v in res.items() if isinstance(v, dict) and "R2" in v)
    with open(path, "w") as f:
        json.dump(_jsonable(res), f, indent=2)
    print(f"[{rargs.get('task')} -> {a.task}] {msg}  -> {path}", flush=True)
    return res


if __name__ == "__main__":
    main()
