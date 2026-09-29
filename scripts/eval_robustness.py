"""Symmetry and robustness table of a trained v2 run (a command-line interface to :mod:`rhmp.robustness`).

The script needs the run directory (``config.json``, ``best.pt``) and the data set of the evaluated task, and uses a
GPU when one is available.  It writes ``<out>/robustness_<task>.json`` and ``.md`` (``<out>`` defaults to the run
directory).  ``scripts/eval_on.py`` evaluates a run on transfer and quality-shift sets and computes DYN rollouts.

    python3 scripts/eval_robustness.py runs/new_tasks/T6f_s42 --all
    python3 scripts/eval_robustness.py RUN --all --max-test 300 --also-tasks HP_qual_graded,HP_qual_sliver

Rows (test split of ``--task``, by default the run's task, e.g. ``--task HP_k1000`` for the contrast transfer of an
HP_k100 run; data re-normalised with the run's statistics):
  base              best.pt as trained, plus every extra test set of the task (``fine`` = zero-shot 4x resolution,
                    quality-shift splits, ...)
  batch=<b>         evaluation batch size b (batch independence: identical metrics, max|dpred| at round-off level)
  relabel           random vertex permutation, complexes rebuilt, data of every degree remapped with orientation signs
  flip-orient       orientation of a random half of the faces reversed (odd face data change sign)
  rotate / reflect  random proper rotation (+ translation) / reflection of the positions (reflect: only tasks without
                    an orientation output map)
  gauge=<s>         theta -> theta + d0 lambda, lambda ~ N(0, s^2), on the connection inputs (T6/T6f/T7)
  noise=<s>         Gaussian noise of standard deviation s on the normalised inputs
  task:<name>       ``--also-tasks``: test split (and extra sets) of further tasks (e.g. HP_qual_graded, HP_k1000)
The R2 of orientation-odd cochain targets is uncentred (``rhmp.metrics``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# the transforms are defined in rhmp.robustness and re-exported here for code that imports them from this script
from rhmp.robustness import (gauge_task, noise_task, random_rotation, remap_complex,  # noqa: E402,F401
                             robustness_table, transformed_task)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--task", default=None, help="evaluation task (default: the run's task)")
    ap.add_argument("--root", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-sizes", default=None, help="comma-separated batch sizes (default: 1,<train batch>)")
    ap.add_argument("--gauge-noise", default=None, help="comma-separated lambda std values (connection tasks)")
    ap.add_argument("--noise", default=None, help="comma-separated input-noise std values (normalised units)")
    ap.add_argument("--relabel", action="store_true")
    ap.add_argument("--flip-orient", action="store_true")
    ap.add_argument("--rotate", action="store_true")
    ap.add_argument("--reflect", action="store_true")
    ap.add_argument("--all", action="store_true", help="relabel + flip-orient + rotate (+ reflect when valid) + "
                                                         "gauge 0.1,1,10 on connection tasks + noise 0.01,0.1")
    ap.add_argument("--max-test", type=int, default=None, help="evaluate only the first N test samples")
    ap.add_argument("--also-tasks", default=None, help="comma-separated further evaluation tasks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    from rhmp.model import RHMP
    from rhmp.robustness import robustness_table
    from rhmp.tasks import load_task
    from rhmp.train import _renormalize, evaluate

    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    conf = json.load(open(os.path.join(a.run_dir, "config.json")))
    rargs = conf["args"]
    if rargs.get("model", "rhmp") != "rhmp":
        raise SystemExit("the robustness table is defined for v2 (rhmp) runs")
    task_name = a.task or rargs["task"]
    kw = {} if not rargs.get("no_output_map") else {"output_map": False}
    if conf.get("model", {}).get("metric_type") != "tensor":
        kw["whitney"] = False                    # Whitney blocks are only used by tensor metrics
    if a.max_test:                               # variable-mesh loaders: build only 3 x max_test samples
        kw["max_samples"] = 3 * a.max_test
    if rargs.get("abs_scale"):
        kw["abs_scale"] = True
    root = a.root or rargs.get("root")
    load = dict(native=rargs.get("native", True), device=dev, star=rargs.get("star", "cotan"), **kw)
    t0 = time.time()
    task = _renormalize(load_task(task_name, root, **load), conf["data"])
    ck = torch.load(os.path.join(a.run_dir, "best.pt"), map_location=dev, weights_only=False)
    model = RHMP.from_checkpoint(ck, map_location=dev).eval()
    model.record_diagnostics = False
    te = task.split[2][: a.max_test] if a.max_test else task.split[2]
    bs_train = (rargs.get("bs_meshes") or 8) if task.variable_mesh else (rargs.get("batch") or 64)
    bsizes = [int(b) for b in (a.batch_sizes.split(",") if a.batch_sizes else [1, bs_train])]
    if a.all:
        a.relabel = a.flip_orient = a.rotate = True
        a.reflect = task.output_map is None and task.native
        if task.connection_dims.get(1, 0) and a.gauge_noise is None and not task.variable_mesh:
            a.gauge_noise = "0.1,1,10"
        if a.noise is None:
            a.noise = "0.01,0.1"
    print(f"[robustness] run={a.run_dir} train_task={rargs['task']} eval_task={task_name} N_test={len(te)} "
          f"load {time.time() - t0:.1f}s", flush=True)
    rows = robustness_table(
        model, task, te, eval_batch=bs_train, device=dev, relabel=a.relabel, flip_orient=a.flip_orient,
        rotate=a.rotate, reflect=a.reflect,
        gauge_noise=[float(x) for x in a.gauge_noise.split(",")] if a.gauge_noise else (),
        noise=[float(x) for x in a.noise.split(",")] if a.noise else (), batch_sizes=bsizes, seed=a.seed,
        log=lambda m: print(m, flush=True))
    keys = ("R2", "MSE", "NRMSE", "SSIM", "Pearson", "N")
    for tn in [x.strip() for x in (a.also_tasks or "").split(",") if x.strip()]:
        try:
            t2 = _renormalize(load_task(tn, root, **load), conf["data"])
        except Exception as e:  # noqa: BLE001 - report and continue with the other sets
            print(f"  task:{tn}: skipped ({e!r:.120})")
            continue
        if len(t2.split[2]):
            m = evaluate(model, t2, t2.split[2], bs_train, dev)
            rows.append({"condition": f"task:{tn}", **{k: m[k] for k in keys}, "dR2": None})
            print(f"  task:{tn:13s} R2={m['R2']:.6f}", flush=True)
        for name, src in t2.extra_tests.items():
            m = evaluate(model, t2, torch.arange(len(src["target"])), bs_train, dev, source=src)
            rows.append({"condition": f"task:{tn}:{name}", **{k: m[k] for k in keys}, "dR2": None})
            print(f"  task:{tn}:{name} R2={m['R2']:.6f}", flush=True)
        del t2
    out = a.out or a.run_dir
    os.makedirs(out, exist_ok=True)
    r2_def = "uncentred (odd cochain target)" if task.target_kind == "cochain" else "centred (v1)"
    res = {"run": a.run_dir, "train_task": rargs["task"], "eval_task": task_name, "native": rargs.get("native"),
           "r2_definition": r2_def, "rows": rows}
    jp = os.path.join(out, f"robustness_{task_name}.json")
    json.dump(res, open(jp, "w"), indent=2)
    lines = [f"# robustness: {rargs['task']} -> {task_name} ({a.run_dir}); R2 {r2_def}", "",
             "| condition | R2 | dR2 | NRMSE | SSIM | Pearson | N | max abs dpred |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        dr = "" if r.get("dR2") is None else f"{r['dR2']:+.2e}"
        md = f"{r['max_dpred']:.1e}" if "max_dpred" in r else ""
        lines.append(f"| {r['condition']} | {r['R2']:.6f} | {dr} | {r['NRMSE']:.6f} | {r['SSIM']:.4f} | "
                     f"{r['Pearson']:.4f} | {r['N']} | {md} |")
    open(jp.replace(".json", ".md"), "w").write("\n".join(lines) + "\n")
    print(f"saved {jp} (+ .md)")
    return res


if __name__ == "__main__":
    main()
