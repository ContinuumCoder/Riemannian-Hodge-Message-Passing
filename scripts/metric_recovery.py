"""Metric recovery: does the learned metric reproduce the true material?

The script analyses trained v2 runs on HP/TET-style tasks (variable meshes with an edge material input).  It needs the
run directory (``config.json``, ``best.pt``) and the data set of the task, and runs on the CPU by default
(``--device``).  It writes ``<run>/metric_recovery_<task>.json`` and, unless ``--no-plots``, a PNG figure (``--out``
selects another directory); ``--table`` writes a Markdown table over all analysed runs.

    python3 scripts/metric_recovery.py RUN_DIR [--n 32] [--device cpu]          # one run
    python3 scripts/metric_recovery.py --n 32 --device cpu --table runs/material_identification/METRIC_RECOVERY.md \
        --scan runs/material_identification runs/metric_variants_full_spd runs/metric_variants   # complete runs

The model is applied to the first ``--n`` test samples, and the metrics of every layer are read with the public
accessor ``RHMP.metric_fields(inputs, K)``.  All "true" material values come from the model's own inputs,
denormalised with the run's statistics (``config.json['data']``; material and metric-reference columns listed in
``config.json['raw_input_columns']`` are already raw log values):

* true edge log conductivity ``log sigma_e`` = raw even edge column 1:0 (isotropic sets: ``log sigma`` at the edge
  midpoint; anisotropic sets: ``log t_e^T Sigma t_e``);
* true face ``log sigma_f`` = raw even face column 2:0 (isotropic sets); tensor sets (``_aniso<R>``) store
  ``(log det Sigma, log ratio)``, so ``log sigma_f = col0 / 2`` (half log det) and the true ratio ``exp(col1)``.

Diagonal models (``metric_type='diag'``): per layer, the learned ``log(H_1/star_1)`` (``log_ratio``, including a
metric reference) and its learned part ``phi`` vs the true edge log conductivity.

Tensor models (``metric_type='tensor'``, per-cell ``sigma_f`` = ``metric_fields(...)[l]['sigma'][1]``, D x D SPD):
  (a) face: ``log det(sigma_f) / 2`` and ``log(mean eigenvalue)`` vs the true face ``log sigma_f``;
  (b) anisotropy: statistics of ``lambda_max / lambda_min`` (about 1 on isotropic data), and on tensor sets the error
      of the log ratio and of the principal direction against the stored true ``Sigma_f`` (``sigma_tensor_face``);
  (c) action: the edge conductance implied by the learned tensor, ``kappa_e = mean_{f ni e} t_e^T sigma_f t_e``
      (the coefficient with which the edge enters the Galerkin operator), ``log kappa_e`` vs the true edge
      conductivity (column 1:0).
Every comparison reports pooled Pearson and Spearman, the least-squares ``slope`` and ``intercept`` of
``learned = slope * true + intercept`` (slope 1 and intercept 0: the metric equals the material in physical units;
a nonzero intercept is a global scale), the mean per-sample Pearson correlation and the std of the per-sample
intercepts (per-sample constants, e.g. from the per-graph star normalisation).  Diagonal rows also give ``sat``, the
fraction of cells whose bounded correction is clamped (``|phi| > 0.95 log_range``).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rhmp.data import cell_alignment, iterate_indices, mesh_minibatch  # noqa: E402
from rhmp.train import _renormalize  # noqa: E402


# ----------------------------------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------------------------------
def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x - x.mean(), y - y.mean()
    d = math.sqrt(float((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / d) if d > 0 else float("nan")


def _spearman(x: np.ndarray, y: np.ndarray, max_n: int = 500_000) -> float:
    from scipy.stats import spearmanr
    if len(x) > max_n:
        sel = np.random.default_rng(0).choice(len(x), max_n, replace=False)
        x, y = x[sel], y[sel]
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def compare(learned: list[np.ndarray], truth: list[np.ndarray]) -> dict:
    """Pooled / per-sample agreement of ``learned`` with ``truth`` (lists of per-sample arrays)."""
    x, y = np.concatenate(learned), np.concatenate(truth)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    out = {"pearson": _pearson(x, y), "spearman": _spearman(x, y), "n": int(len(x)),
           "learned_std": float(x.std()), "truth_std": float(y.std())}
    if y.std() > 0:
        slope, intercept = np.polyfit(y, x, 1)
        out.update(slope=float(slope), intercept=float(intercept))
    else:
        out.update(slope=float("nan"), intercept=float(x.mean() - y.mean()))
    per, icpt = [], []
    for a, b in zip(learned, truth):
        m = np.isfinite(a) & np.isfinite(b)
        a, b = a[m], b[m]
        if a.std() > 0 and b.std() > 0:
            per.append(_pearson(a, b))
        icpt.append(float(a.mean() - (out["slope"] if np.isfinite(out["slope"]) else 1.0) * b.mean()))
    out["pearson_per_sample"] = float(np.mean(per)) if per else float("nan")
    out["intercept_per_sample_std"] = float(np.std(icpt)) if icpt else float("nan")
    return out


# ----------------------------------------------------------------------------------------------------------------
# learned quantities
# ----------------------------------------------------------------------------------------------------------------
def _edge_conductance(K, sigma: torch.Tensor) -> torch.Tensor:
    """``kappa_e = mean_{f ni e} t_e^T sigma_f t_e`` for a (batched) complex; ``sigma (n_top, D, D)`` -> ``(n1,)``."""
    A = K.d_abs[1].to_sparse_coo().coalesce() if K.dim == 2 else None
    if A is None:                                   # Limitation: triangle complexes only; tetrahedra are not supported
        raise NotImplementedError("edge conductance for tetrahedral complexes")
    f, e = A.indices()
    ev = K.edge_vectors().double()
    t = ev / ev.norm(dim=1, keepdim=True)
    q = torch.einsum("pi,pij,pj->p", t[e], sigma.double()[f], t[e])
    num = torch.zeros(K.n[1], dtype=torch.float64, device=q.device).index_add_(0, e, q)
    cnt = torch.zeros(K.n[1], dtype=torch.float64, device=q.device).index_add_(0, e, torch.ones_like(q))
    return num / cnt.clamp(min=1)


def _capture_legacy(model):
    """Fallback for models without ``metric_fields``: wrap the layers' metric functions."""
    store: dict = {}
    for i, layer in enumerate(model.layers):
        store[i] = {}
        orig = layer.metrics

        def wrapped(x, q, ctx, *args, _orig=orig, _i=i, **kw):
            out = _orig(x, q, ctx, *args, **kw)
            store[_i]["phi1"] = (out[0][1] - ctx.log_star[1].unsqueeze(1)).detach()[:, 0]
            store[_i]["lr1"] = store[_i]["phi1"]
            return out
        layer.metrics = wrapped
    return store


# ----------------------------------------------------------------------------------------------------------------
# one run
# ----------------------------------------------------------------------------------------------------------------
def recover(run_dir: str, *, task_name: str | None = None, n: int = 32, batch: int = 8, device: str = "cpu",
            root: str | None = None, plots: bool = True, out_dir: str | None = None, log=print,
            samples_out: dict | None = None) -> dict:
    """Metric-recovery analysis of one run (see module docstring); returns the JSON-able result.

    ``samples_out``: optional dict that receives the pooled per-cell arrays behind the statistics (for figures):
    ``t_edge`` / ``t_face`` (true log conductivities) and ``layers[l]`` with ``lr`` (diagonal ``log(H_1/star_1)``),
    ``hld`` (tensor ``log det(sigma_f)/2``) and ``kappa`` (tensor edge action), each a 1-D numpy array (or absent).
    """
    from rhmp.model import RHMP
    from rhmp.tasks import load_task

    dev = torch.device(device)
    conf = json.load(open(os.path.join(run_dir, "config.json")))
    rargs = conf["args"]
    if rargs.get("model", "rhmp") != "rhmp":
        raise ValueError("metric recovery needs a v2 (rhmp) run")
    task_name = task_name or rargs["task"]
    tensor_run = conf.get("model", {}).get("metric_type") == "tensor"
    t0 = time.time()
    task = load_task(task_name, root or rargs.get("root"), native=True, device=dev, star=rargs.get("star", "cotan"),
                     fine=False, max_samples=3 * n, whitney=tensor_run, abs_scale=bool(rargs.get("abs_scale")))
    if not task.variable_mesh or 1 not in task.x_stats:
        raise ValueError("metric recovery needs an HP/TET-style task (variable meshes with an edge material input)")
    task = _renormalize(task, conf["data"])            # the run's statistics (raw columns stay raw)
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location=dev, weights_only=False)
    model = RHMP.from_checkpoint(ck, map_location=dev).eval()
    model.record_diagnostics = False
    cfg = model.cfg
    metric_type = getattr(cfg, "metric_type", "diag")
    log_range = float(getattr(cfg, "log_range", 2.0))
    L = len(model.layers)
    tensor_set = task.in_dims.get(2, 0) >= 2 and bool(task.meta.get("aniso")) and task.meta.get("aniso") is not True
    use_fields = hasattr(model, "metric_fields")
    legacy_store = None if use_fields else _capture_legacy(model)
    te = task.split[2][:n]
    # ground-truth tensors of the tensor sets (evaluation-only field of the generator)
    blob = None
    if tensor_run and tensor_set:
        from rhmp.tasks.synthetic import _load_packed
        blob = _load_packed(task.meta["source"])
        if "sigma_tensor_face" not in blob:
            blob = None
    acc = {l: {"phi": [], "lr": [], "hld": [], "lme": [], "ratio": [], "kappa": [], "S": []} for l in range(L)}
    t_edge, t_face, t_ratio, true_T = [], [], [], []
    with torch.no_grad():
        for b in iterate_indices(te, batch, shuffle=False):
            Kb, xb, yb = mesh_minibatch(task.K, task.inputs, task.target, b, device=dev)
            ptr1 = Kb.meta["ptr"][1].tolist()
            ptr2 = Kb.meta["ptr"][2].tolist()
            e_raw = task.x_stats[1].denormalize(xb[1][:, 0, :]).double().cpu().numpy()
            f_raw = task.x_stats[2].denormalize(xb[2][:, 0, :]).double().cpu().numpy() if 2 in xb else None
            nb = len(b)
            for j in range(nb):
                t_edge.append(e_raw[ptr1[j]:ptr1[j + 1], 0])
                if f_raw is not None:
                    fr = f_raw[ptr2[j]:ptr2[j + 1]]
                    t_face.append(fr[:, 0] / (2.0 if tensor_set else 1.0))
                    t_ratio.append(np.exp(fr[:, 1]) if tensor_set and fr.shape[1] > 1 else np.ones(len(fr)))
            fields = model.metric_fields(xb, Kb) if use_fields else None
            if not use_fields:
                model(xb, Kb)
            for l in range(L):
                a_ = acc[l]
                if use_fields:
                    fl = fields[l]
                    if 1 in fl.get("phi", {}):
                        ph = fl["phi"][1][:, 0].double().cpu().numpy()
                        lr = fl["log_ratio"][1][:, 0].double().cpu().numpy()
                        for j in range(nb):
                            a_["phi"].append(ph[ptr1[j]:ptr1[j + 1]])
                            a_["lr"].append(lr[ptr1[j]:ptr1[j + 1]])
                    sig = fl.get("sigma", {}).get(1)
                else:
                    st = legacy_store[l]
                    if "phi1" in st:
                        ph = st["phi1"].double().cpu().numpy()
                        for j in range(nb):
                            a_["phi"].append(ph[ptr1[j]:ptr1[j + 1]])
                            a_["lr"].append(ph[ptr1[j]:ptr1[j + 1]])
                    sig = None
                if sig is not None:
                    S = sig[:, 0].double()                                          # (n2, D, D)
                    ev = torch.linalg.eigvalsh(S).clamp_min(1e-300)
                    hld = (0.5 * torch.log(ev).sum(-1)).cpu().numpy() if S.shape[-1] == 2 else \
                        (torch.log(ev).sum(-1) / S.shape[-1]).cpu().numpy()
                    lme = torch.log(ev.mean(-1)).cpu().numpy()
                    rat = (ev[:, -1] / ev[:, 0]).cpu().numpy()
                    kap = torch.log(_edge_conductance(Kb, S.to(Kb.pos.device))).cpu().numpy()
                    Sn = S.cpu().numpy()
                    for j in range(nb):
                        a_["hld"].append(hld[ptr2[j]:ptr2[j + 1]])
                        a_["lme"].append(lme[ptr2[j]:ptr2[j + 1]])
                        a_["ratio"].append(rat[ptr2[j]:ptr2[j + 1]])
                        a_["kappa"].append(kap[ptr1[j]:ptr1[j + 1]])
                        a_["S"].append(Sn[ptr2[j]:ptr2[j + 1]])
            if blob is not None:
                from rhmp.tasks.synthetic import _slices
                ids = task.meta.get("sample_ids")
                for i in b.tolist():
                    sid = int(ids[i]) if ids is not None else int(i)
                    St = _slices(blob, sid, "sigma_tensor_face", 2).double()
                    top = _slices(blob, sid, "faces", 2).long()
                    perm, _ = cell_alignment(task.K[i].cells[2].cpu(), top, task.K[i].n[0], oriented=False)
                    St = St[perm].numpy()
                    true_T.append(np.stack([np.stack([St[:, 0], St[:, 1]], -1),
                                            np.stack([St[:, 1], St[:, 2]], -1)], -2))
    res = {"run": run_dir, "task": task_name, "metric_type": metric_type, "log_range": log_range,
           "layer_types": list(getattr(cfg, "layers", None) or ["poly"] * L),
           "metric_reference": {str(k): v for k, v in dict(getattr(cfg, "metric_reference", {}) or {}).items()},
           "material_dims": {str(k): v for k, v in dict(getattr(cfg, "material_dims", {}) or {}).items()},
           "raw_input_columns": conf.get("raw_input_columns"), "solver_mode": bool(rargs.get("solver_mode")),
           "tensor_set": tensor_set, "n_samples": len(te), "params": int(model.num_parameters()),
           "accessor": "metric_fields" if use_fields else "wrapped layer metrics", "layers": []}
    rp = os.path.join(run_dir, "result.json")
    if os.path.exists(rp):
        r = json.load(open(rp))
        res["test_R2"] = r.get("test", {}).get("R2")
        res["fine_R2"] = r.get("fine", {}).get("R2")
    for l in range(L):
        a_ = acc[l]
        row: dict = {"layer": l, "type": res["layer_types"][l] if l < len(res["layer_types"]) else "?"}
        # per layer: layers that report a material tensor are analysed as tensors (their degree-1 up metric is the
        # Whitney tensor), the others (e.g. resolvent layers of tensor runs, diag runs) through log(H_1/star_1)
        if a_["lr"] and not a_["hld"]:
            row["diag_logratio_vs_edge"] = compare(a_["lr"], t_edge)
            row["diag_phi_vs_edge"] = compare(a_["phi"], t_edge)
            ph = np.concatenate(a_["phi"])
            row["sat"] = float((np.abs(ph) > 0.95 * log_range).mean())
        if a_["hld"]:
            if t_face:
                row["tensor_halflogdet_vs_face"] = compare(a_["hld"], t_face)
                row["tensor_logmeaneig_vs_face"] = compare(a_["lme"], t_face)
            rat = np.concatenate(a_["ratio"])
            row["tensor_anisotropy"] = {"median": float(np.median(rat)), "p90": float(np.quantile(rat, 0.9)),
                                        "max": float(rat.max()), "mean_log_ratio": float(np.log(rat).mean())}
            if tensor_set and t_ratio:
                row["tensor_logratio_vs_true"] = compare([np.log(r_) for r_ in a_["ratio"]],
                                                         [np.log(r_) for r_ in t_ratio])
            row["tensor_action_vs_edge"] = compare(a_["kappa"], t_edge)
            if true_T:
                Sl, St = np.concatenate(a_["S"]), np.concatenate(true_T)
                _, vl = np.linalg.eigh(Sl)
                et, vt = np.linalg.eigh(St)
                ang_l = np.arctan2(vl[:, 1, -1], vl[:, 0, -1])
                ang_t = np.arctan2(vt[:, 1, -1], vt[:, 0, -1])
                dang = np.abs((ang_l - ang_t + np.pi / 2) % np.pi - np.pi / 2)
                aniso = np.log(et[:, -1] / et[:, 0]) > np.log(1.5)
                row["tensor_angle_err_deg_median"] = float(np.degrees(np.median(dang[aniso]))) if aniso.any() else None
        res["layers"].append(row)
        log(f"  layer {l} ({row['type']}): " + _short(row))
    res["seconds"] = time.time() - t0
    if samples_out is not None:
        samples_out["t_edge"] = np.concatenate(t_edge) if t_edge else None
        samples_out["t_face"] = np.concatenate(t_face) if t_face else None
        samples_out["layers"] = {l: {k: np.concatenate(acc[l][k]) for k in ("lr", "hld", "kappa") if acc[l][k]}
                                 for l in range(L)}
    out = out_dir or run_dir
    os.makedirs(out, exist_ok=True)
    jp = os.path.join(out, f"metric_recovery_{task_name}.json")
    json.dump(res, open(jp, "w"), indent=2, default=lambda o: None)
    if plots:
        try:
            _plots(res, acc, t_edge, t_face, task, te, out, task_name)
        except Exception as e:  # noqa: BLE001 - figures are optional
            log(f"  (plots skipped: {e!r:.100})")
    log(f"saved {jp} ({res['seconds']:.0f}s)")
    return res


def _short(row: dict) -> str:
    parts = []
    for key, lab in (("diag_logratio_vs_edge", "edge r"), ("tensor_halflogdet_vs_face", "face r"),
                     ("tensor_action_vs_edge", "action r")):
        if key in row:
            c = row[key]
            parts.append(f"{lab}={c['pearson']:+.3f} slope={c.get('slope', float('nan')):.3f} "
                         f"icpt={c.get('intercept', float('nan')):+.3f}")
    if "tensor_anisotropy" in row:
        parts.append(f"ratio med={row['tensor_anisotropy']['median']:.3f}")
    if "sat" in row:
        parts.append(f"sat={row['sat']:.2f}")
    return "; ".join(parts)


def _plots(res, acc, t_edge, t_face, task, te, out, task_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = len(res["layers"])
    panels = []
    for l in range(L):
        if acc[l]["lr"] and not acc[l]["hld"]:
            panels.append((l, "edge", np.concatenate(t_edge), np.concatenate(acc[l]["lr"]), "log(H1/star1)"))
        if acc[l]["hld"] and t_face:
            panels.append((l, "face", np.concatenate(t_face), np.concatenate(acc[l]["hld"]), "log det(sigma_f)/2"))
            panels.append((l, "action", np.concatenate(t_edge), np.concatenate(acc[l]["kappa"]),
                           "log t^T sigma t (edge)"))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.0), squeeze=False)
    for ax, (l, kind, x, y, lab) in zip(axes[0], panels):
        ax.hexbin(x, y, gridsize=60, bins="log", cmap="viridis")
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
        ax.plot([lo, hi], [lo, hi], "w--", lw=0.8)
        ax.set_title(f"layer {l} {kind}: r = {_pearson(y, x):.3f}", fontsize=8)
        ax.set_xlabel("true log conductivity", fontsize=7)
        ax.set_ylabel(lab, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"metric_recovery_{task_name}.png"), dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------
# scan + table
# ----------------------------------------------------------------------------------------------------------------
def _fmt(c: dict | None, key: str = "pearson", fmt: str = "{:+.3f}") -> str:
    if not c or key not in c or c[key] is None or (isinstance(c[key], float) and not math.isfinite(c[key])):
        return "-"
    return fmt.format(c[key])


def table(results: list[dict]) -> str:
    """Markdown table over runs (per-layer correlations; slope/intercept of the best layer)."""
    lines = ["| run | task | metric | layers | ref / material | test R2 | edge: r per layer (diag log H1/star1; tensor "
             "action t^T sigma t) | best-layer slope / intercept (edge) | face: r per layer (tensor log det/2) | "
             "best-layer slope / intercept (face) | anisotropy ratio median (tensor) | sat (diag) |",
             "|---|---|---|---|---|---:|---|---|---|---|---|---|"]
    for r in results:
        rows = r["layers"]
        edge = [x.get("diag_logratio_vs_edge") or x.get("tensor_action_vs_edge") for x in rows]
        face = [x.get("tensor_halflogdet_vs_face") for x in rows]

        def best(cs):
            cs = [c for c in cs if c and c.get("pearson") is not None and math.isfinite(c["pearson"])]
            return max(cs, key=lambda c: abs(c["pearson"])) if cs else None
        be, bf = best(edge), best(face)
        refmat = ", ".join(f"ref {k}:{v}" for k, v in r["metric_reference"].items()) + \
            (", " if r["metric_reference"] and r["material_dims"] else "") + \
            ", ".join(f"mat {k}:{v}" for k, v in r["material_dims"].items())
        ratio = " / ".join(f"{x['tensor_anisotropy']['median']:.2f}" for x in rows if "tensor_anisotropy" in x)
        sat = " / ".join(f"{x['sat']:.2f}" for x in rows if "sat" in x)
        r2s = "-" if r.get("test_R2") is None else f"{r['test_R2']:.4f}"
        lines.append(
            f"| {os.path.basename(os.path.normpath(r['run']))} | {r['task']} | {r['metric_type']} | "
            f"{','.join(r['layer_types'])} | {refmat or '-'} | {r2s} | "
            f"{' / '.join(_fmt(c) for c in edge)} | {_fmt(be, 'slope', '{:.3f}')} / {_fmt(be, 'intercept', '{:+.3f}')} | "
            f"{' / '.join(_fmt(c) for c in face) if any(face) else '-'} | "
            f"{_fmt(bf, 'slope', '{:.3f}')} / {_fmt(bf, 'intercept', '{:+.3f}')} | {ratio or '-'} | {sat or '-'} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", default=None)
    ap.add_argument("--scan", nargs="*", default=None, help="directories whose complete runs are analysed")
    ap.add_argument("--task", default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--n", type=int, default=32, help="number of test samples")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--table", default=None, help="path of a Markdown table of all analysed runs")
    a = ap.parse_args(argv)
    runs = []
    if a.run_dir:
        runs.append(a.run_dir)
    for d in a.scan or []:
        for rd in sorted(glob.glob(os.path.join(d, "*"))):
            rp = os.path.join(rd, "result.json")
            if not (os.path.isdir(rd) and os.path.exists(os.path.join(rd, "best.pt")) and os.path.exists(rp)):
                continue
            try:
                r = json.load(open(rp))
            except ValueError:
                continue
            if r.get("complete"):
                runs.append(rd)
            else:
                print(f"skip {rd} (incomplete run)")
    results = []
    for rd in runs:
        print(f"[metric recovery] {rd}", flush=True)
        try:
            results.append(recover(rd, task_name=a.task, n=a.n, batch=a.batch, device=a.device, root=a.root,
                                   plots=not a.no_plots, out_dir=a.out))
        except Exception as e:  # noqa: BLE001 - keep scanning
            print(f"  failed: {e!r}", flush=True)
    if a.table and results:
        os.makedirs(os.path.dirname(a.table) or ".", exist_ok=True)
        hdr = ("# Metric recovery (learned metric vs true material)\n\n"
               f"Generated by `python3 scripts/metric_recovery.py --scan ... --n {a.n} --device {a.device}` from the "
               f"first {a.n} test samples of each run.  r: pooled Pearson correlation between the learned log metric "
               "and the true log conductivity, per layer (diagonal metrics: log(H_1/star_1) vs the edge log sigma; "
               "tensor metrics: the edge action log(mean_f t_e^T sigma_f t_e) vs the edge log sigma, and the face "
               "log det(sigma_f)/2 vs the face log sigma).  Slope and intercept: least-squares fit "
               "learned = slope * true + intercept for the layer with the largest |r|; slope 1 and intercept 0 mean "
               "that the metric equals the material in physical units.  The per-run files "
               "`<run>/metric_recovery_<task>.json` also contain the Spearman correlations, the per-sample "
               "correlations, the spread of the per-sample intercepts, the log mean eigenvalue and the anisotropy "
               "statistics.\n\n")
        open(a.table, "w").write(hdr + table(results))
        print(f"wrote {a.table}")
    return results


if __name__ == "__main__":
    main()
