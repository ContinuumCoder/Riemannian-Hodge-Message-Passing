"""Figures of REPORT.md and README.md (``docs/figures/*.png``).

    # figures 1, 3, 5, 6 from the result files only (matplotlib, no torch)
    python3 scripts/make_figures.py --results results --out docs/figures
    # + figure 2 (metric-recovery scatter) and figure 4 (tensor ellipses): need torch, the checkpoints and the data
    python3 scripts/make_figures.py --results results --out docs/figures \\
        --recovery-tensor results/checkpoints/HP_k100_S3c_solver_tensor_learn \\
        --recovery-general RUNS/stage2/HP_k100_diag-resolvent_s42 \\
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

# reference palette (categorical slots in fixed order) and chart chrome, light surface
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
    rows = [("v2 solver mode, learned tensor metric", "cab75/stage3/HP_k100_S3c_solver_tensor_learn"),
            ("v2 solver mode, learned diagonal metric", "cab75/stage3/HP_k100_S3c_solver_diag_learn"),
            ("v2 general stack, resolvent layer", "cab75/stage2/HP_k100_diag-resolvent_s42"),
            ("v2 general stack, polynomial layers", "cab75/new/HP_k100_s42"),
            ("dec_fixed (v2 stack, frozen DEC star)", "cab16/baselines/HP_k100_native/dec_fixed_s42"),
            ("MeshGraphNet, v2 budget", "cab16/baselines/HP_k100_native/mgn_s42"),
            ("MeshGraphNet, v1 budget", "cab16/baselines_v1budget/HP_k100_native/mgn_s42"),
            ("EGNN, v2 budget", "cab16/baselines/HP_k100_native/egnn_s42")]
    labels, a, b = [], [], []
    for lab, d in rows:
        r = _load(res, os.path.join(d, "result.json"))
        p = (r or {}).get("params")
        labels.append(f"{lab}  ({p / 1000:.1f}K params)" if p and p < 10_000 else
                      f"{lab}  ({p / 1000:.0f}K params)" if p else lab)
        a.append(_r2(r))
        b.append(_r2(r, "fine"))
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    _hbar_pairs(ax, labels, a, b, "test split (training resolution)", "zero-shot 4x finer meshes", BLUE, ORANGE)
    ax.set_xlabel("R2 (HP_k100: heterogeneous Poisson, contrast 100, variable meshes)")
    ax.set_title("Resolution transfer needs the metric to be the physics", loc="left", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, borderaxespad=0.3)
    _save(fig, out, "fig1_hp_k100_transfer.png")
    plt.close(fig)


def fig3(res: str, out: str) -> None:
    plt = _plt()
    tasks = [("AHP_r10\n(2-D, ratio 10)", "cab75/aniso/AHP_r10_diag-solver_s42", "cab75/aniso/AHP_r10_tensor-solver_s42"),
             ("AHP_r100\n(2-D, ratio 100)", "cab75/aniso/AHP_r100_diag-solver_s42",
              "cab75/aniso/AHP_r100_tensor-solver_lr3e-4_s42"),
             ("ASURF_r100\n(surfaces)", "cab75/aniso_solver/ASURF_r100_diag-solver_s42",
              "cab75/aniso_solver/ASURF_r100_tensor-solver_s42"),
             ("ADARCYp_r100\n(3-D tetrahedra)", "cab16/aniso/ADARCYp_r100_n1500_diag-solver_s42",
              "cab16/aniso/ADARCYp_r100_n1500_tensor-solver_s42")]
    mgn = _load(res, "cab16/aniso/AHP_r100_mgn_s42/result.json")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), sharey=True)
    w = 0.26
    for ax, key, title in ((axes[0], "test", "test split"), (axes[1], "fine", "zero-shot 4x finer meshes")):
        x = np.arange(len(tasks))
        for j, (lab, col, off) in enumerate((("diagonal metric (solver mode)", ORANGE, -w / 2),
                                            ("full-SPD tensor metric (solver mode)", BLUE, w / 2))):
            vals = [_r2(_load(res, os.path.join(t[1 + j], "result.json")), key) for t in tasks]
            v = np.array([np.nan if q is None else q for q in vals], dtype=float)
            ax.bar(x + off, np.nan_to_num(v), width=w, color=col, edgecolor=SURFACE, linewidth=1.5, label=lab)
            for xi, vi in zip(x + off, v):
                if np.isfinite(vi):
                    ax.text(xi, vi + 0.012, f"{vi:.3f}", ha="center", va="bottom", fontsize=7.2, color=INK2)
        mv = _r2(mgn, key)
        if mv is not None:
            ax.bar([1 + 1.5 * w], [mv], width=w, color=AQUA, edgecolor=SURFACE, linewidth=1.5,
                   label="MeshGraphNet (92K params, general)")
            ax.text(1 + 1.5 * w, mv + 0.012, f"{mv:.3f}", ha="center", va="bottom", fontsize=7.2, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels([t[0] for t in tasks], color=INK)
        ax.set_ylim(0, 1.08)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("R2")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Anisotropic media: diagonal (M-matrix) metric vs full-SPD Whitney tensor metric", x=0.01, ha="left",
                 fontsize=10.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    _save(fig, out, "fig3_aniso_diag_vs_tensor.png")
    plt.close(fig)


def fig5(res: str, out: str) -> None:
    plt = _plt()
    base = _load(res, "cab75/new/T6f_s42/result.json")
    pts = [("training mesh\n(1024 vertices, test split)", _r2(base))]
    for tag, lab in (("mesh7_n1024", "new mesh, seed 7\n(1024)"), ("mesh11_n1024", "new mesh, seed 11\n(1024)"),
                     ("mesh7_n4096", "4x finer new mesh\n(4096)"), ("mesh7_n256", "4x coarser new mesh\n(256)")):
        pts.append((lab, _r2(_load(res, f"cab75/transfer/T6f_s42/transfer_T6f_{tag}.json"))))
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
    ax.set_ylabel("R2 (uncentred, face flux)")
    ax.grid(axis="x", visible=False)
    ax.set_title("T6f: one trained model evaluated zero-shot on unseen meshes and resolutions (v1: not applicable)",
                 loc="left", fontsize=9.5)
    _save(fig, out, "fig5_t6f_mesh_transfer.png")
    plt.close(fig)


def fig6(res: str, out: str) -> None:
    plt = _plt()
    specs = [("DYNfix, mass projection", "cab75/suite/DYNfix_s42/rollout_DYNfix_proj.json", BLUE),
             ("DYNfix, no projection", "cab75/suite/DYNfix_s42/rollout_DYNfix.json", ORANGE),
             ("DYN (variable meshes), no projection", "cab75/suite/DYN_s42/rollout_DYN.json", AQUA),
             ("DYN (variable meshes), mass projection", "cab75/suite/DYN_s42/rollout_DYN_proj.json", YELLOW)]
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
        if pers is None and "DYNfix" in lab:
            pers = (h, np.array(r["persistence_R2"], dtype=float))
    if pers is not None:
        ax.plot(pers[0], np.clip(pers[1], lo, None), color=MUTED, linewidth=1.5, label="persistence baseline (DYNfix)")
    ax.set_xscale("log")
    ax.set_xlim(1, 100)
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50, 100]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(lo, 1.05)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_xlabel("rollout step (autoregressive, 100-step test trajectories)")
    ax.set_ylabel("R2 (curves clipped at -1.25)")
    ax.set_title("Advection-diffusion rollouts: mass drift is the dominant long-horizon error", loc="left")
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
    _hex(axes[0], st["t_face"], lt["hld"], "solver mode, learned tensor: faces", "true log sigma (face)",
         "learned log det(sigma_f) / 2")
    _hex(axes[1], st["t_edge"], lt["kappa"], "solver mode, learned tensor: edge action", "true log sigma (edge)",
         "learned log mean_f t^T sigma_f t")
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
    _hex(axes[2], sg["t_edge"], sg["layers"][best]["lr"], f"general stack (diag + resolvent), best layer {best}",
         "true log sigma (edge)", "learned log(H_1 / star_1)")
    fig.suptitle("HP_k100: the learned metric is the material only when the architecture solves with it "
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
    cb.set_label("log eigenvalue ratio (anisotropy)")
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
    ax.set_xlabel("principal-direction error (deg)")
    ax.set_ylabel("triangles (true ratio > 10)")
    ax.set_title("direction agreement on the whole mesh", loc="left", fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.suptitle(f"AHP_r100 test mesh ({K.n[0]} vertices, window [{x0}, {x1}]^2): per-triangle tensors as ellipses "
                 "(major axis along the fibre, aspect sqrt(ratio))", x=0.01, ha="left", fontsize=10,
                 fontweight="semibold")
    _save(fig, out, "fig4_tensor_ellipses.png")
    plt.close(fig)
    stats = dict(run=run, task=rargs["task"], test_index=i, sample_id=int(task.meta["sample_ids"][i]),
                 n_vertices=int(K.n[0]), n_triangles=int(len(faces)), n_strong=int(strong.sum()),
                 median_direction_error_deg_true_ratio_gt_10=med,
                 mean_direction_error_deg_true_ratio_gt_10=float(np.mean(dang[strong])),
                 median_ratio_learned=float(np.median(rl)), median_ratio_true=float(np.median(rt)),
                 note="one AHP_r100 test mesh (first test sample); written by scripts/make_figures.py (figure 4)")
    sp = os.path.join(res, "cab75", "aniso", os.path.basename(os.path.normpath(run)), "fig4_direction_stats.json")
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
    ap.add_argument("--recovery-tensor", default=None, help="run dir (config.json + best.pt): solver-mode tensor run")
    ap.add_argument("--recovery-general", default=None, help="run dir: general-stack diagonal run")
    ap.add_argument("--ellipse-run", default=None, help="run dir: AHP tensor-metric run")
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
