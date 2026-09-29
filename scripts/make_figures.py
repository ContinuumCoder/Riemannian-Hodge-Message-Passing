"""Figures of REPORT.md and README.md (``docs/figures/fig*.png``).

Figures 1, 3, 5 and 6 are drawn from the result files in ``results/`` alone (matplotlib, no torch).  Figures 2 and 4
also need torch, the data sets and trained runs (``--recovery-tensor``, ``--recovery-general``, ``--ellipse-run``; the
general-stack run of figure 2 is not in the model zoo and is trained by ``scripts/run_metric_variants.sh``), and run
on the CPU by default (``--device``).  The figures are written to ``--out`` (default ``docs/figures/``); figure 4 also
writes its direction statistics to ``results/anisotropy/<run>/fig4_direction_stats.json`` when that directory exists.

    # figures 1, 3, 5 and 6
    python3 scripts/make_figures.py --results results --out docs/figures
    # all six figures
    python3 scripts/make_figures.py --results results --out docs/figures \\
        --recovery-tensor results/checkpoints/HP_k100_solver_tensor_learn \\
        --recovery-general runs/metric_variants/HP_k100_diag-resolvent_s42 \\
        --ellipse-run results/checkpoints/AHP_r100_tensor-solver_lr3e-4_s42

Figures:

1. ``fig1_hp_k100_transfer.png``: HP_k100 test R2 and zero-shot 4x-resolution R2 of the v2 variants and baselines.
2. ``fig2_metric_recovery.png``: learned vs true log conductivity (solver mode with a learned tensor metric; general
   stack with a diagonal metric and a resolvent layer).
3. ``fig3_aniso_diag_vs_tensor.png``: diagonal vs full-SPD tensor metric in solver mode on the anisotropy tasks.
4. ``fig4_tensor_ellipses.png``: true vs learned per-triangle material tensors on an AHP_r100 test mesh.
5. ``fig5_t6f_mesh_transfer.png``: T6f zero-shot transfer to new meshes and resolutions.
6. ``fig6_dyn_rollouts.png``: rollout R2 vs horizon (mass projection, persistence).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# colour palette (categorical colours in a fixed order) and the colours of the chart elements on a light background
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]   # one-hue blue ramp


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "grid.linestyle": "-", "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "semibold", "legend.frameon": False,
        "legend.fontsize": 8.5, "lines.linewidth": 2.0, "lines.solid_capstyle": "round"})
    return plt


def _load(res_root: str, rel: str) -> dict | None:
    try:
        with open(os.path.join(res_root, rel)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _r2(d: dict | None, key: str = "test") -> float | None:
    v = (d or {}).get(key)
    return v.get("R2") if isinstance(v, dict) else None


def _save(fig, out: str, name: str) -> str:
    path = os.path.join(out, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {path}")
    return path


def _hbar_pairs(ax, labels, a, b, la, lb, ca, cb, xmax=1.0):
    """Horizontal grouped bars (two series per row), values at the bar tips."""
    y = np.arange(len(labels))[::-1]
    h = 0.36
    for off, vals, lab, col in ((h / 2, a, la, ca), (-h / 2, b, lb, cb)):
        v = np.array([np.nan if x is None else x for x in vals], dtype=float)
        ax.barh(y + off, np.nan_to_num(v), height=h, color=col, edgecolor=SURFACE, linewidth=1.5, label=lab)
        for yi, vi in zip(y + off, v):
            if np.isfinite(vi):
                ax.text(max(vi, 0) + 0.01 * xmax, yi, f"{vi:.4f}", va="center", ha="left", fontsize=7.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK)
    ax.set_xlim(0, xmax * 1.12)
    ax.grid(axis="y", visible=False)


# ================================================================================================================
def fig1(res: str, out: str) -> None:
    plt = _plt()
    rows = [("Ours, physics-solver mode, learned tensor material",
             "material_identification/HP_k100_solver_tensor_learn"),
            ("Ours, physics-solver mode, learned scalar material", "material_identification/HP_k100_solver_diag_learn"),
            ("Ours, general network with an implicit (solve) layer", "metric_variants/HP_k100_diag-resolvent_s42"),
            ("Ours, general network, local layers only", "new_tasks/HP_k100_s42"),
            ("Ours, geometry only (no material learning)", "baselines/HP_k100_native/dec_fixed_s42"),
            ("MeshGraphNet, same size", "baselines/HP_k100_native/mgn_s42"),
            ("MeshGraphNet, 5x larger", "baselines_v1_budget/HP_k100_native/mgn_s42"),
            ("EGNN", "baselines/HP_k100_native/egnn_s42")]
    labels, a, b = [], [], []
    for lab, d in rows:
        r = _load(res, os.path.join(d, "result.json"))
        p = (r or {}).get("params")
        labels.append(f"{lab}  ({p / 1000:.1f}K params)" if p and p < 10_000 else
                      f"{lab}  ({p / 1000:.0f}K params)" if p else lab)
        a.append(_r2(r))
        b.append(_r2(r, "fine"))
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    _hbar_pairs(ax, labels, a, b, "meshes like the training ones", "4x finer meshes, never seen in training", BLUE, ORANGE)
    ax.set_xlabel("accuracy (R2) on heterogeneous-media Poisson problems, conductivity contrast 100")
    ax.set_title("Accuracy at the training resolution and on 4x finer meshes", loc="left", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.3)
    _save(fig, out, "fig1_hp_k100_transfer.png")
    plt.close(fig)


def fig3(res: str, out: str) -> None:
    plt = _plt()
    tasks = [("2-D Poisson\nanisotropy 10:1", "anisotropy/AHP_r10_diag-solver_s42", "anisotropy/AHP_r10_tensor-solver_s42"),
             ("2-D Poisson\nanisotropy 100:1", "anisotropy/AHP_r100_diag-solver_s42",
              "anisotropy/AHP_r100_tensor-solver_lr3e-4_s42"),
             ("curved surfaces\nfibre diffusion 100:1", "anisotropy_surfaces/ASURF_r100_diag-solver_s42",
              "anisotropy_surfaces/ASURF_r100_tensor-solver_s42"),
             ("3-D Darcy flow\nanisotropy 100:1", "anisotropy_3d/ADARCYp_r100_n1500_diag-solver_s42",
              "anisotropy_3d/ADARCYp_r100_n1500_tensor-solver_s42")]
    mgn = _load(res, "anisotropy_general/AHP_r100_mgn_s42/result.json")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), sharey=True)
    w = 0.26
    for ax, key, title in ((axes[0], "test", "meshes like the training ones"), (axes[1], "fine", "4x finer meshes, never seen in training")):
        x = np.arange(len(tasks))
        for j, (lab, col, off) in enumerate((("scalar (diagonal) material metric", ORANGE, -w / 2),
                                            ("full tensor material metric", BLUE, w / 2))):
            vals = [_r2(_load(res, os.path.join(t[1 + j], "result.json")), key) for t in tasks]
            v = np.array([np.nan if q is None else q for q in vals], dtype=float)
            ax.bar(x + off, np.nan_to_num(v), width=w, color=col, edgecolor=SURFACE, linewidth=1.5, label=lab)
            for xi, vi in zip(x + off, v):
                if np.isfinite(vi):
                    ax.text(xi, vi + 0.012, f"{vi:.3f}", ha="center", va="bottom", fontsize=7.2, color=INK2)
        mv = _r2(mgn, key)
        if mv is not None:
            ax.bar([1 + 1.5 * w], [mv], width=w, color=AQUA, edgecolor=SURFACE, linewidth=1.5,
                   label="MeshGraphNet")
            ax.text(1 + 1.5 * w, mv + 0.012, f"{mv:.3f}", ha="center", va="bottom", fontsize=7.2, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels([t[0] for t in tasks], color=INK)
        ax.set_ylim(0, 1.08)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("R2")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Anisotropic media: a scalar material metric is not enough, a tensor one is", x=0.01, ha="left",
                 fontsize=10.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    _save(fig, out, "fig3_aniso_diag_vs_tensor.png")
    plt.close(fig)


def fig5(res: str, out: str) -> None:
    plt = _plt()
    base = _load(res, "new_tasks/T6f_s42/result.json")
    pts = [("the mesh it was\ntrained on", _r2(base))]
    for tag, lab in (("mesh7_n1024", "a new random mesh\n(same size)"), ("mesh11_n1024", "another new mesh\n(same size)"),
                     ("mesh7_n4096", "a new mesh,\n4x finer"), ("mesh7_n256", "a new mesh,\n4x coarser")):
        pts.append((lab, _r2(_load(res, f"mesh_transfer/T6f_s42/transfer_T6f_{tag}.json"))))
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    x = np.arange(len(pts))
    v = np.array([np.nan if p[1] is None else p[1] for p in pts], dtype=float)
    ax.vlines(x, 0.99, v, color=GRID, linewidth=2)
    ax.plot(x, v, "o", color=BLUE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    for xi, vi in zip(x, v):
        if np.isfinite(vi):
            ax.text(xi, vi + 0.0004, f"{vi:.4f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in pts], color=INK, fontsize=8)
    ax.set_ylim(0.99, 1.0015)
    ax.set_ylabel("accuracy (R2)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Gauge-field task: one trained model, evaluated without retraining on meshes it has never seen",
                 loc="left", fontsize=9.5)
    _save(fig, out, "fig5_t6f_mesh_transfer.png")
    plt.close(fig)


def fig6(res: str, out: str) -> None:
    plt = _plt()
    specs = [("ours, with exact mass conservation enforced", "extension_suite/DYNfix_s42/rollout_DYNfix_proj.json",
              BLUE),
             ("ours, plain autoregressive", "extension_suite/DYNfix_s42/rollout_DYNfix.json", ORANGE),
]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    lo = -1.25
    pers = None
    for lab, rel, col in specs:
        r = _load(res, rel)
        if not r:
            print(f"  (fig6: {rel} missing, skipped)")
            continue
        h = np.array(r["horizon"], dtype=float)
        y = np.array([np.nan if (q is None or (isinstance(q, float) and not math.isfinite(q))) else q
                      for q in r["R2"]], dtype=float)
        ax.plot(h, np.clip(y, lo, None), color=col, label=lab)
        if y[-1] < lo:
            ax.annotate(f"{y[-1]:.1f} at step {int(h[-1])}", xy=(h[-1], lo), xytext=(h[-1] * 0.55, lo + 0.18),
                        fontsize=7.5, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        if pers is None and "DYNfix" in rel:
            pers = (h, np.array(r["persistence_R2"], dtype=float))
    if pers is not None:
        ax.plot(pers[0], np.clip(pers[1], lo, None), color=MUTED, linewidth=1.5, label="no-change baseline (repeat the last state)")
    ax.set_xscale("log")
    ax.set_xlim(1, 100)
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50, 100]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(lo, 1.05)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_xlabel("time steps predicted ahead (each step feeds the previous prediction)")
    ax.set_ylabel("accuracy (R2)")
    ax.set_title("Advection-diffusion, 100-step forecasts on a fixed mesh", loc="left")
    ax.legend(loc="lower left")
    _save(fig, out, "fig6_dyn_rollouts.png")
    plt.close(fig)


# ================================================================================================================
def _hex(ax, x, y, title, xlabel, ylabel):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("rhmp_blue", SEQ)
    ax.hexbin(x, y, gridsize=55, bins="log", cmap=cmap, mincnt=1, linewidths=0)
    lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1.0)
    xc, yc = x - x.mean(), y - y.mean()
    r = float((xc * yc).sum() / math.sqrt((xc * xc).sum() * (yc * yc).sum()))
    slope, icpt = np.polyfit(x, y, 1)
    ax.text(0.03, 0.97, f"r = {r:+.3f}\nslope {slope:.3f}, intercept {icpt:+.3f}", transform=ax.transAxes,
            va="top", ha="left", fontsize=8, color=INK)
    ax.set_title(title, loc="left", fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)


def fig2(res: str, out: str, tensor_run: str, general_run: str, device: str = "cpu", n: int = 32) -> None:
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    sys.path.insert(0, ROOT)
    from metric_recovery import recover  # noqa: E402
    plt = _plt()
    tmp = tempfile.mkdtemp(prefix="rhmp_fig2_")
    st, sg = {}, {}
    recover(tensor_run, n=n, device=device, plots=False, out_dir=tmp, samples_out=st)
    recover(general_run, n=n, device=device, plots=False, out_dir=tmp, samples_out=sg)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6))
    lt = st["layers"][0]
    _hex(axes[0], st["t_face"], lt["hld"], "physics-solver mode, per triangle", "true conductivity (log)",
         "learned conductivity (log)")
    _hex(axes[1], st["t_edge"], lt["kappa"], "physics-solver mode, per edge", "true conductivity (log)",
         "learned conductivity (log)")
    # general stack: the layer whose diagonal metric correlates most with the material (best case)
    best, best_r = None, -1.0
    for l, d in sg["layers"].items():
        if "lr" not in d:
            continue
        x, y = sg["t_edge"], d["lr"]
        ok = np.isfinite(x) & np.isfinite(y)
        rr = abs(np.corrcoef(x[ok], y[ok])[0, 1])
        if rr > best_r:
            best, best_r = l, rr
    _hex(axes[2], sg["t_edge"], sg["layers"][best]["lr"], "general network, its best layer",
         "true conductivity (log)", "learned metric (log)")
    fig.suptitle("Is the learned metric the true material?  Yes in physics-solver mode, no in a general network "
                 "(first 32 test samples, physical units)", x=0.01, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out, "fig2_metric_recovery.png")
    plt.close(fig)


def fig4(res: str, out: str, run: str, device: str = "cpu", sample: int = 0,
         window: tuple[float, float, float, float] = (0.3, 0.7, 0.3, 0.7)) -> None:
    import torch
    sys.path.insert(0, ROOT)
    from matplotlib.collections import EllipseCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    from rhmp.data import mesh_minibatch
    from rhmp.model import RHMP
    from rhmp.tasks import load_task
    from rhmp.tasks.aniso import _load_blob, cell_tensors
    from rhmp.train import _renormalize
    plt = _plt()
    dev = torch.device(device)
    conf = json.load(open(os.path.join(run, "config.json")))
    rargs = conf["args"]
    task = load_task(rargs["task"], rargs.get("root"), native=True, device=dev, star=rargs.get("star", "cotan"),
                     fine=False, max_samples=30, whitney=True)
    task = _renormalize(task, conf["data"])
    model = RHMP.from_checkpoint(os.path.join(run, "best.pt"), map_location=dev).eval()
    model.record_diagnostics = False
    i = int(task.split[2][sample])
    Kb, xb, _ = mesh_minibatch(task.K, task.inputs, task.target, torch.tensor([i]), device=dev)
    with torch.no_grad():
        S = model.metric_fields(xb, Kb)[0]["sigma"][1][:, 0].double().cpu().numpy()          # (n2, 2, 2)
    K = task.K[i]
    blob = _load_blob(task.meta["source"])
    T = cell_tensors(blob, int(task.meta["sample_ids"][i]), "sigma_face", K, "faces").numpy()
    pos = K.pos.double().cpu().numpy()[:, :2]
    faces = K.cells[2].long().cpu().numpy()
    cen = pos[faces].mean(1)
    e1, e2 = pos[faces[:, 1]] - pos[faces[:, 0]], pos[faces[:, 2]] - pos[faces[:, 0]]
    area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])

    def principal(M_):
        ev, V = np.linalg.eigh(M_)
        ev = np.clip(ev, 1e-300, None)
        return ev[:, 1] / ev[:, 0], np.arctan2(V[:, 1, 1], V[:, 0, 1])       # ratio, major-axis angle
    rt, at = principal(T)
    rl, al = principal(S)
    dang = np.degrees(np.abs((al - at + np.pi / 2) % np.pi - np.pi / 2))     # principal-direction error in [0, 90]
    x0, x1, y0, y1 = window
    sel = (cen[:, 0] > x0) & (cen[:, 0] < x1) & (cen[:, 1] > y0) & (cen[:, 1] < y1)
    size = 1.1 * np.sqrt(2.0 * area[sel])                                     # major axis ~ the cell diameter
    cmap = LinearSegmentedColormap.from_list("rhmp_blue", SEQ[2:])
    norm = Normalize(vmin=0.0, vmax=math.log(100.0))
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 4.7), gridspec_kw=dict(width_ratios=[1, 1, 0.045, 0.95],
                                                                        wspace=0.12))
    for ax, (ratio, ang), title in ((axes[0], (rt, at), "true Sigma_f (data generator)"),
                                    (axes[1], (rl, al), "learned sigma_f (solver mode, 2.5K parameters)")):
        r_, a_ = ratio[sel], ang[sel]
        ec = EllipseCollection(size, size / np.sqrt(r_), np.degrees(a_), units="xy", offsets=cen[sel],
                               offset_transform=ax.transData, cmap=cmap, norm=norm, edgecolors="none")
        ec.set_array(np.log(r_))
        ax.add_collection(ec)
        ax.triplot(pos[:, 0], pos[:, 1], faces, lw=0.3, color=GRID, zorder=0)
        ax.set(xlim=(x0, x1), ylim=(y0, y1), aspect="equal")
        ax.set_title(title, loc="left", fontsize=9)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    cb = fig.colorbar(ec, cax=axes[2])
    cb.set_label("anisotropy strength (log ratio)")
    cb.outline.set_visible(False)
    ax = axes[3]
    pos3 = ax.get_position()
    ax.set_position([pos3.x0 + 0.075, pos3.y0, pos3.width - 0.075, pos3.height])
    strong = rt > 10.0
    ax.hist(dang[strong], bins=np.linspace(0, 90, 31), color=BLUE, edgecolor=SURFACE, linewidth=1.0)
    med = float(np.median(dang[strong]))
    ax.axvline(med, color=INK2, linewidth=1.2)
    ax.text(med + 2, ax.get_ylim()[1] * 0.92, f"median {med:.1f} deg\n(random directions: 45)", fontsize=8,
            color=INK2, va="top")
    ax.set_xlim(0, 90)
    ax.set_xlabel("direction error (degrees)")
    ax.set_ylabel("number of triangles")
    ax.set_title("how well the learned fibre direction matches the true one (whole mesh)", loc="left", fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.suptitle("Anisotropic 2-D Poisson (ratio 100): the learned material tensor of each triangle, drawn as an ellipse, "
                 "next to the true one", x=0.01, ha="left", fontsize=10,
                 fontweight="semibold")
    _save(fig, out, "fig4_tensor_ellipses.png")
    plt.close(fig)
    stats = dict(run=run, task=rargs["task"], test_index=i, sample_id=int(task.meta["sample_ids"][i]),
                 n_vertices=int(K.n[0]), n_triangles=int(len(faces)), n_strong=int(strong.sum()),
                 median_direction_error_deg_true_ratio_gt_10=med,
                 mean_direction_error_deg_true_ratio_gt_10=float(np.mean(dang[strong])),
                 median_ratio_learned=float(np.median(rl)), median_ratio_true=float(np.median(rt)),
                 note="one AHP_r100 test mesh (first test sample); written by scripts/make_figures.py (figure 4)")
    sp = os.path.join(res, "anisotropy", os.path.basename(os.path.normpath(run)), "fig4_direction_stats.json")
    if os.path.isdir(os.path.dirname(sp)):
        with open(sp, "w") as fh:
            json.dump(stats, fh, indent=2)
        print(f"wrote {sp}")
    print(f"  fig4: median direction error {med:.1f} deg on {int(strong.sum())} triangles with true ratio > 10; "
          f"median learned / true ratio {np.median(rl):.1f} / {np.median(rt):.1f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "figures"))
    ap.add_argument("--recovery-tensor", default=None, help="run directory of a solver-mode tensor run")
    ap.add_argument("--recovery-general", default=None, help="run directory of a general-stack diagonal run")
    ap.add_argument("--ellipse-run", default=None, help="run directory of an AHP tensor-metric run")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", default=None, help="comma-separated figure numbers, e.g. 1,3")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    only = None if not a.only else {int(s) for s in a.only.split(",")}

    def want(k: int) -> bool:
        return only is None or k in only
    if want(1):
        fig1(a.results, a.out)
    if want(3):
        fig3(a.results, a.out)
    if want(5):
        fig5(a.results, a.out)
    if want(6):
        fig6(a.results, a.out)
    if want(2):
        if a.recovery_tensor and a.recovery_general:
            fig2(a.results, a.out, a.recovery_tensor, a.recovery_general, device=a.device)
        else:
            print("fig2 skipped (needs --recovery-tensor and --recovery-general)")
    if want(4):
        if a.ellipse_run:
            fig4(a.results, a.out, a.ellipse_run, device=a.device)
        else:
            print("fig4 skipped (needs --ellipse-run)")


if __name__ == "__main__":
    main()
