"""Qualitative field figures of REPORT.md section 6.8 and README.md (``docs/figures/field*.png``).

Each figure draws the fields themselves on the mesh -- ground truth, our prediction(s), a baseline, error maps -- for
one test sample, so that a reader sees what the accuracy numbers of the tables mean.  Models are loaded and applied
exactly as ``rhmp.train.eval_ckpt`` does: the task is loaded with the run's star, re-expressed in the run's
normalisation (``rhmp.train._renormalize``: raw material / metric-reference columns, mean-free solver statistics),
predicted with ``rhmp.train.predict`` (block-diagonal batches) and denormalised with the run's target statistics.
The displayed sample is the first test sample of each task (no selection), except figure 2, which shows the median
case: among the first 20 test samples, the one whose scalar-metric R2 is closest to their median (``--ahp-sample``
overrides it).  The R2 in a panel title is the R2 of that one sample (``1 - SS_res / SS_tot`` centred on the sample's
own mean; uncentred for the odd face flux of figure 3), which is stricter than the pooled test R2 of the tables (the
pooled SS_tot also contains the between-sample variance); figures 1 and 2 print the medians over the first 20 test
samples in their captions.

    # all figures (GPU host with the data sets; about 1.5 minutes)
    python3 scripts/make_field_figures.py --out docs/figures
    python3 scripts/make_field_figures.py --only 1,3        # a subset

Figures:

1. ``field1_hetero_poisson.png``: heterogeneous Poisson (HP_k100), training resolution and the same problem on a 4x
   finer mesh (zero-shot): conductivity, source, true solution, physics-solver mode (learned tensor material),
   general network (local layers), MeshGraphNet, error maps; ``field1_hetero_poisson_compact.png``: the solution
   panels only (README).
2. ``field2_anisotropic.png``: anisotropic Poisson (AHP_r100, ratio 100): true fibre directions, true solution,
   solver mode with a scalar (diagonal) and a full tensor material metric, MeshGraphNet, error maps, and the edges on
   which the exact operator has a negative weight (a scalar metric cannot produce those).
3. ``field3_gauge_transfer.png``: U(1) gauge task (T6f): face flux on the training mesh, on a new random mesh and on
   a 4x finer new mesh (one physical field, regenerated with ``scripts/t6_mesh_transfer.generate``).
4. ``field4_surface.png``: screened Poisson on closed surfaces (SURF): a torus from the test split and a genus-2
   surface from the topology-transfer set, true vs predicted, error on the surface.
5. ``field5_rollout_snapshots.png``: advection-diffusion rollouts on a fixed mesh (DYNfix): truth, rollout with
   mass projection, plain rollout, steps 1 to 100.
6. ``field6_t3_vector.png``: tangent vector field on an ellipsoid (T3): true vs predicted arrows.

Checkpoints: all from the model zoo (``results/checkpoints``), including the three comparison runs
``HP_k100_general_s42`` (general stack, local layers; test R2 0.8255 / 4x 0.6992), ``HP_k100_mgn_s42`` (MeshGraphNet,
v2 budget; 0.9488 / 0.2581) and ``AHP_r100_mgn_s42`` (MeshGraphNet; 0.665 / 0.400); every run can be overridden
(``--hp-solver``, ``--hp-general``, ...).  The per-panel numbers are written to ``--stats`` (default
``results/cab75/fieldviz/field_figures_stats.json``).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from make_figures import INK, INK2, MUTED, SURFACE, _plt  # noqa: E402  (shared figure style)

ZOO = os.path.join(ROOT, "results", "checkpoints")
DPI = 170
FIELD_CMAP, ERR_CMAP, SIGNED_CMAP, INPUT_CMAP = "viridis", "Reds", "RdBu_r", "cividis"
EDGE_GREY = "#b9b8b1"
STATS: dict = {}
# planar comparison grid: input panel | its colour bar | spacer | 4 panels | colour bar
GRID_RATIOS = [1, 0.045, 0.21, 1, 1, 1, 1, 0.045]


# ================================================================================================================
# model loading and prediction (the rhmp.train.eval_ckpt path)
# ================================================================================================================
def _view(task):
    """Shallow copy of a TaskData whose data can be re-normalised without touching ``task`` (complexes shared)."""
    return dataclasses.replace(task, extra_tests={k: dict(v) for k, v in task.extra_tests.items()},
                               meta=dict(task.meta))


def load_run(run_dir: str, task, device):
    """``(model, task_view, config)`` for a trainer run directory, exactly as ``rhmp.train.eval_ckpt``: v2 runs through
    ``RHMP.from_checkpoint``, baselines through ``rhmp.baselines.registry.from_checkpoint`` (+ ``prepare_task``, output
    map dropped for non-v2 models); the view is re-expressed in the run's normalisation."""
    import torch

    from rhmp.model import RHMP
    from rhmp.train import _renormalize
    conf = json.load(open(os.path.join(run_dir, "config.json")))
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location=device, weights_only=False)
    view = _view(task)
    trained = conf["args"].get("model") or "rhmp"
    if trained == "rhmp":
        model = RHMP.from_checkpoint(ck, map_location=device)
    else:
        from rhmp.baselines.registry import SPECS, from_checkpoint, prepare_task
        view = prepare_task(trained, view, ck.get("build"))
        model = from_checkpoint(ck, view, map_location=device)
        if not SPECS[trained].uses_output_map and view.output_map is not None:
            view = dataclasses.replace(view, output_map=None)
    model.eval()
    if hasattr(model, "record_diagnostics"):
        model.record_diagnostics = False
    view = _renormalize(view, conf["data"])
    return model, view, conf


def predict_one(model, task, i: int, device, source: dict | None = None):
    """Prediction and target of sample ``i`` (of the task, or of an extra test set ``source``) in physical units,
    ``(n_t, O)`` float64 numpy arrays (``rhmp.train.predict`` + the run's target statistics)."""
    import torch

    from rhmp.train import predict
    p, t = predict(model, task, torch.tensor([int(i)]), 1, device, source=source)
    ys = task.y_stats
    return ys.denormalize(p[0]).double().cpu().numpy(), ys.denormalize(t[0]).double().cpu().numpy()


def r2(pred: np.ndarray, true: np.ndarray, center: bool = True) -> float:
    """R2 of one sample: ``1 - SS_res / SS_tot``, SS_tot centred on the sample mean (uncentred for odd cochains)."""
    p, t = np.asarray(pred, dtype=np.float64).ravel(), np.asarray(true, dtype=np.float64).ravel()
    den = ((t - t.mean()) ** 2).sum() if center else (t ** 2).sum()
    return float(1.0 - ((p - t) ** 2).sum() / max(den, 1e-300))


def raw_inputs(task, i: int, source: dict | None = None) -> dict:
    """Physical input columns ``{k: (n_k, F_k)}`` of sample ``i``."""
    src = source["inputs"] if source is not None else task.inputs
    x = src[i] if isinstance(src, (list, tuple)) else {k: v[i] for k, v in src.items()}    # variable / shared mesh
    return {k: task.x_stats[k].denormalize(v).double().cpu().numpy() for k, v in x.items()}


def mesh_arrays(K):
    """``(pos (n0, D), faces (n2, 3), edges (n1, 2), boundary edge mask (n1,) or None)`` as numpy arrays."""
    pos = K.pos.double().cpu().numpy()
    faces = K.cells[2].long().cpu().numpy()
    edges = K.cells[1].long().cpu().numpy()
    bnd = K.boundary[1].cpu().numpy().astype(bool) if K.boundary is not None and len(K.boundary) > 1 else None
    return pos, faces, edges, bnd


def face_areas(pos: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]
    if pos.shape[1] == 2:
        return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def weighted_quantile(v: np.ndarray, w: np.ndarray, q: float) -> float:
    """Quantile of ``v`` with weights ``w`` (e.g. triangle areas: the colour range covers ``q`` of the domain)."""
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return float(v[o][min(np.searchsorted(cw, q * cw[-1]), len(v) - 1)])


def farthest_points(x: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Indices of ``k`` well-spread points of ``x (n, D)`` (greedy farthest-point sampling)."""
    n = len(x)
    k = min(k, n)
    sel = [int(np.random.default_rng(seed).integers(n))]
    d = np.linalg.norm(x - x[sel[0]], axis=1)
    for _ in range(k - 1):
        j = int(np.argmax(d))
        sel.append(j)
        d = np.minimum(d, np.linalg.norm(x - x[j], axis=1))
    return np.asarray(sel)


# ================================================================================================================
# drawing helpers
# ================================================================================================================
def _clean(ax) -> None:
    ax.set_aspect("equal")
    ax.set_axis_off()


def _title(ax, main: str, sub: str | None = None, *, size: float = 9.0) -> None:
    """Two-line panel title: the quantity in plain words, the sample / accuracy in small print below it."""
    ax.set_title(main, fontsize=size, fontweight="semibold", color=INK, pad=13, loc="center")
    text = ax.text2D if hasattr(ax, "text2D") else ax.text          # 3-D axes place 2-D text with text2D
    text(0.5, 1.012, sub or "", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.2, color=INK2)


def _corner(ax, text: str) -> None:
    """Small annotation in the lower left corner of a panel."""
    put = ax.text2D if hasattr(ax, "text2D") else ax.text
    put(0.03, 0.03, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.8, color=INK2, zorder=10,
        bbox=dict(facecolor=SURFACE, alpha=0.85, edgecolor="none", pad=1.2))


def _outline(ax, pos, edges, bnd, color=MUTED, lw=0.6) -> None:
    """Domain boundary (edges with one incident face)."""
    from matplotlib.collections import LineCollection
    if bnd is None or not bnd.any():
        return
    ax.add_collection(LineCollection(pos[edges[bnd]][:, :, :2], colors=color, linewidths=lw, zorder=5))


def _limits(ax, pos) -> None:
    ax.set_xlim(pos[:, 0].min(), pos[:, 0].max())
    ax.set_ylim(pos[:, 1].min(), pos[:, 1].max())


def node_field(ax, pos, faces, vals, norm, cmap=FIELD_CMAP):
    """Vertex field, linear inside each triangle (Gouraud shading)."""
    import matplotlib.tri as mtri
    tri = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
    m = ax.tripcolor(tri, np.asarray(vals, dtype=np.float64).ravel(), shading="gouraud", cmap=cmap, norm=norm,
                     rasterized=True)
    _limits(ax, pos)
    return m


def face_field(ax, pos, faces, vals, norm, cmap=FIELD_CMAP):
    """Face field, constant on each triangle (flat shading; hairline edges in the face colour hide seams)."""
    import matplotlib.tri as mtri
    tri = mtri.Triangulation(pos[:, 0], pos[:, 1], faces)
    m = ax.tripcolor(tri, facecolors=np.asarray(vals, dtype=np.float64).ravel(), shading="flat", cmap=cmap,
                     norm=norm, edgecolors="face", linewidth=0.15, rasterized=True)
    _limits(ax, pos)
    return m


def mesh_lines(ax, pos, edges, color=EDGE_GREY, lw=0.25, zorder=1):
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(pos[edges][:, :, :2], colors=color, linewidths=lw, zorder=zorder))


def _cbar(fig, mappable, cax, label: str, *, extend: str = "neither", ticks=None):
    cb = fig.colorbar(mappable, cax=cax, extend=extend, ticks=ticks)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=2, colors=INK2)
    cb.set_label(label, fontsize=7.5, color=INK2, labelpad=3)
    return cb


def _suptitle(fig, text: str, y: float = 0.994) -> None:
    fig.text(0.012, y, text, ha="left", va="top", fontsize=11, fontweight="semibold", color=INK)


def _caption(fig, text: str, y: float = 0.008) -> None:
    fig.text(0.012, y, text, ha="left", va="bottom", fontsize=8, color=INK2, linespacing=1.35)


def _save(fig, out: str, name: str) -> str:
    path = os.path.join(out, name)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE, pil_kwargs={"optimize": True})
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.2f} MB)", flush=True)
    return path


def _sym_norm(vmax: float):
    from matplotlib.colors import Normalize
    return Normalize(vmin=-vmax, vmax=vmax)


def _lin_norm(lo: float, hi: float):
    from matplotlib.colors import Normalize
    return Normalize(vmin=lo, vmax=hi)


def _fmt_r2(v: float) -> str:
    if v >= 0.9995:
        return f"{v:.5f}"
    if v >= 0.99:
        return f"{v:.4f}"
    return f"{v:.3f}" if v > -10 else f"{v:.1f}"


def _peak(err: np.ndarray, true: np.ndarray) -> str:
    """Largest error as a percentage of the largest |true| value."""
    v = 100.0 * float(np.abs(err).max()) / float(np.abs(true).max())
    return f"largest error {v:.2g} % of max |u|" if v < 10 else f"largest error {v:.0f} % of max |u|"


# ================================================================================================================
# figure 1: heterogeneous Poisson, training resolution and 4x finer mesh
# ================================================================================================================
def fig1(out: str, device, runs: dict, sample: int = 0, n_median: int = 20) -> None:
    from rhmp.tasks import load_task
    t0 = time.time()
    base = load_task("HP_k100", None, native=True, device=device, max_samples=60)   # first 20 test samples
    te = base.split[2]
    n_med = min(n_median, len(te))
    if not 0 <= sample < n_med:
        raise ValueError(f"--hp-sample must be one of the first {n_med} test samples")
    fine = base.extra_tests["fine"]
    cids = list(fine.get("coarse_ids") or [])
    sids = [int(base.meta["sample_ids"][int(te[q])]) for q in range(n_med)]
    fidx = [cids.index(s_) if s_ in cids else None for s_ in sids]       # matching 4x sample of each test sample
    i, sid = int(te[sample]), sids[sample]
    paired = fidx[sample] is not None
    j = fidx[sample] if paired else 0
    labels = {"solver": "ours: physics-solver mode", "general": "ours: general network", "mgn": "MeshGraphNet"}
    preds = {}
    for key, rd in runs.items():
        model, view, _ = load_run(rd, base, device)
        pc, tc = predict_one(model, view, i, device)
        pf, tf = predict_one(model, view, j, device, source=view.extra_tests["fine"])
        # per-sample R2 of the first n_med test samples (and their 4x versions): the medians for the caption
        rc = [r2(*predict_one(model, view, int(te[q]), device)) for q in range(n_med)]
        rf = [r2(*predict_one(model, view, fidx[q], device, source=view.extra_tests["fine"]))
              for q in range(n_med) if fidx[q] is not None]
        preds[key] = dict(pc=pc[:, 0], tc=tc[:, 0], pf=pf[:, 0], tf=tf[:, 0], r2c=r2(pc, tc), r2f=r2(pf, tf),
                          all_c=rc, all_f=rf, med_c=float(np.median(rc)), med_f=float(np.median(rf)))
        print(f"  fig1 {key}: R2 {preds[key]['r2c']:.5f} (training resolution) / {preds[key]['r2f']:.5f} (4x); "
              f"medians over {n_med} test samples {preds[key]['med_c']:.4f} / {preds[key]['med_f']:.4f}", flush=True)
    xc, xf = raw_inputs(base, i), raw_inputs(base, j, fine)
    blocks = []
    for K, x, tk, pk, rk in ((base.K[i], xc, "tc", "pc", "r2c"), (fine["K"][j], xf, "tf", "pf", "r2f")):
        pos, faces, edges, bnd = mesh_arrays(K)
        blocks.append(dict(pos=pos, faces=faces, edges=edges, bnd=bnd, logsig=x[2][:, 0], f=x[0][:, 0],
                           true=preds["solver"][tk], pred={k: preds[k][pk] for k in runs},
                           r2={k: preds[k][rk] for k in runs}, n0=int(K.n[0])))
    # colour scales: u -> range of the true field (both meshes); errors -> one scale for all error panels
    unorm = _lin_norm(min(b["true"].min() for b in blocks), max(b["true"].max() for b in blocks))
    snorm = _lin_norm(math.log(0.1), math.log(10.0))
    fnorm = _sym_norm(max(np.abs(b["f"]).max() for b in blocks))
    enorm = _lin_norm(0.0, max(np.quantile(np.abs(b["pred"][k] - b["true"]), 0.995) for b in blocks for k in runs))
    order = ("solver", "general", "mgn")

    plt = _plt()
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(13.6, 12.6))
    gs = GridSpec(4, 8, figure=fig, width_ratios=GRID_RATIOS, wspace=0.06, hspace=0.30, left=0.012, right=0.955,
                  top=0.93, bottom=0.075)
    zoom = (0.35, 0.65)
    first_axes = []
    for bi, b in enumerate(blocks):
        r0 = 2 * bi
        pos, faces, edges, bnd = b["pos"], b["faces"], b["edges"], b["bnd"]
        # row A: conductivity | true u | three models
        ax = fig.add_subplot(gs[r0, 0])
        first_axes.append(ax)
        m_s = face_field(ax, pos, faces, b["logsig"], snorm, INPUT_CMAP)
        _outline(ax, pos, edges, bnd)
        _clean(ax)
        _title(ax, "conductivity σ (input)", "constant per triangle")
        cb = _cbar(fig, m_s, fig.add_subplot(gs[r0, 1]), "σ (log scale)",
                   ticks=[math.log(0.1), math.log(0.3), 0.0, math.log(3.0), math.log(10.0)])
        cb.set_ticklabels(["0.1", "0.3", "1", "3", "10"])
        ax = fig.add_subplot(gs[r0, 3])
        m_u = node_field(ax, pos, faces, b["true"], unorm)
        _outline(ax, pos, edges, bnd)
        _clean(ax)
        _title(ax, "true solution u", f"test sample {sample} ({'first of the test split, ' if sample == 0 else ''}"
                                      "not selected)")
        for c, k in enumerate(order):
            ax = fig.add_subplot(gs[r0, 4 + c])
            node_field(ax, pos, faces, b["pred"][k], unorm)
            _outline(ax, pos, edges, bnd)
            _clean(ax)
            _title(ax, labels[k], f"R2 {_fmt_r2(b['r2'][k])} (this sample)")
        _cbar(fig, m_u, fig.add_subplot(gs[r0, 7]), "u", extend="both")
        # row B: source | mesh zoom | error maps
        ax = fig.add_subplot(gs[r0 + 1, 0])
        m_f = node_field(ax, pos, faces, b["f"], fnorm, SIGNED_CMAP)
        _outline(ax, pos, edges, bnd)
        _clean(ax)
        _title(ax, "source f (input)", "at the vertices")
        _cbar(fig, m_f, fig.add_subplot(gs[r0 + 1, 1]), "f")
        ax = fig.add_subplot(gs[r0 + 1, 3])
        mesh_lines(ax, pos, edges, color=INK2, lw=0.35)
        ax.set_xlim(*zoom)
        ax.set_ylim(*zoom)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(MUTED)
            sp.set_linewidth(0.6)
        _title(ax, "the mesh", "zoom on the central 30 % x 30 %")
        for c, k in enumerate(order):
            ax = fig.add_subplot(gs[r0 + 1, 4 + c])
            err = np.abs(b["pred"][k] - b["true"])
            m_e = node_field(ax, pos, faces, err, enorm, ERR_CMAP)
            _outline(ax, pos, edges, bnd)
            _clean(ax)
            _title(ax, "error |pred − true|", labels[k].replace("ours: ", "ours, "))
            _corner(ax, _peak(err, b["true"]))
        _cbar(fig, m_e, fig.add_subplot(gs[r0 + 1, 7]), "|pred − true|", extend="max")
    hdr = dict(ha="left", va="bottom", fontsize=10, fontweight="semibold", color=INK)
    for bi, text in enumerate((f"Meshes like the training ones ({blocks[0]['n0']:,} vertices)",
                               f"The same problem on a 4x finer mesh, never seen in training ({blocks[1]['n0']:,} "
                               "vertices, zero-shot)")):
        fig.text(0.012, first_axes[bi].get_position().y1 + 0.029, text, **hdr)
    _suptitle(fig, "Heterogeneous-media Poisson problem (conductivity contrast 100): true and predicted solutions")
    short = {"solver": "solver", "general": "general network", "mgn": "MeshGraphNet"}
    meds = ", ".join(f"{short[k]} {_fmt_r2(preds[k]['med_c'])} / {_fmt_r2(preds[k]['med_f'])}" for k in order)
    by_res: dict = {}                     # where this sample is more than 0.1 away from a model's median R2
    for k in order:
        for a, m, res in (("r2c", "med_c", "training-resolution"), ("r2f", "med_f", "4x")):
            if abs(preds[k][a] - preds[k][m]) > 0.1:
                by_res.setdefault(res, []).append(
                    f"{'harder' if preds[k][a] < preds[k][m] else 'easier'} than the median for "
                    f"{'the ' if k == 'general' else ''}{short[k]} ({_fmt_r2(preds[k][a])} vs {_fmt_r2(preds[k][m])})")
    near = "This sample is close to these medians" + (", except that " + "; ".join(
        f"on the {res} mesh it is {' and '.join(ph)}" for res, ph in by_res.items()) if by_res else "") + "."
    _caption(fig, "-div(σ grad u) = f, u = 0 on the boundary; the 4x mesh carries the same conductivity and source."
                  + ("" if paired else " (4x: fine sample 0, no matching pair found.)") + "  One colour scale for all "
                  "u panels, one for all error panels; R2 of this one sample, centred on its own mean.\n"
                  f"Test sample {sample} is {'the first of the test split' if sample == 0 else 'a fixed test sample'} "
                  f"(not selected).  Medians of the per-sample R2 over the first {n_med} test samples (training "
                  f"resolution / 4x): {meds}.\n{near}\nPooled test R2 of the tables (all test samples, training / "
                  "4x): solver 0.9999 / 1.0000, general network 0.826 / 0.699, MeshGraphNet 0.949 / 0.258.")
    _save(fig, out, "field1_hetero_poisson.png")
    plt.close(fig)
    fig1_compact(out, blocks, unorm, labels, order, sample, meds, n_med)

    def peak_pct(b, k):
        return 100 * float(np.abs(b["pred"][k] - b["true"]).max() / np.abs(b["true"]).max())
    STATS["field1_hetero_poisson"] = dict(task="HP_k100", test_index=sample, sample_id=sid, fine_index=j,
                                          fine_paired=paired, n0=blocks[0]["n0"], n0_fine=blocks[1]["n0"], runs=runs,
                                          selection="first test sample (not selected)",
                                          R2={k: dict(train_res=preds[k]["r2c"], fine_4x=preds[k]["r2f"])
                                              for k in runs},
                                          median_R2_first_test_samples={k: dict(
                                              n=n_med, train_res=preds[k]["med_c"], fine_4x=preds[k]["med_f"])
                                              for k in runs},
                                          per_sample_R2_first_test_samples={k: dict(
                                              train_res=preds[k]["all_c"], fine_4x=preds[k]["all_f"]) for k in runs},
                                          largest_error_pct_of_max_u={k: dict(train_res=peak_pct(blocks[0], k),
                                                                              fine_4x=peak_pct(blocks[1], k))
                                                                      for k in runs},
                                          seconds=time.time() - t0)


def fig1_compact(out: str, blocks: list, unorm, labels: dict, order, sample: int, meds: str, n_med: int) -> None:
    """Solution panels of figure 1 only (README)."""
    plt = _plt()
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(10.4, 6.25))
    gs = GridSpec(2, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.045], wspace=0.06, hspace=0.26, left=0.05,
                  right=0.945, top=0.88, bottom=0.12)
    rows = ("training resolution", "4x finer, zero-shot")
    for r, b in enumerate(blocks):
        pos, faces, edges, bnd = b["pos"], b["faces"], b["edges"], b["bnd"]
        ax = fig.add_subplot(gs[r, 0])
        m = node_field(ax, pos, faces, b["true"], unorm)
        _outline(ax, pos, edges, bnd)
        _clean(ax)
        _title(ax, "true solution u", f"{b['n0']:,} vertices")
        ax.text(-0.05, 0.5, rows[r], transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=9.5,
                fontweight="semibold", color=INK)
        for c, k in enumerate(order):
            ax = fig.add_subplot(gs[r, 1 + c])
            node_field(ax, pos, faces, b["pred"][k], unorm)
            _outline(ax, pos, edges, bnd)
            _clean(ax)
            _title(ax, labels[k], f"R2 {_fmt_r2(b['r2'][k])} (this sample)")
        _cbar(fig, m, fig.add_subplot(gs[r, 4]), "u", extend="both")
    _suptitle(fig, "Heterogeneous-media Poisson (contrast 100): same test problem at the training resolution and on "
                   "a 4x finer mesh")
    _caption(fig, f"Test sample {sample} ({'the first of the test split, ' if sample == 0 else ''}not selected); one "
                  f"colour scale for all panels; R2 of this one sample (medians over the first {n_med} test samples,"
                  f"\ntraining / 4x: {meds}).  The physics-solver mode predicts through a finite-element solve whose"
                  "\nonly learned part is the material map, so it transfers across resolutions; the other two models "
                  "learn the solution map itself.")
    _save(fig, out, "field1_hetero_poisson_compact.png")
    plt.close(fig)


# ================================================================================================================
# figure 2: anisotropic Poisson, scalar vs tensor material metric
# ================================================================================================================
def fig2(out: str, device, runs: dict, sample="median", n_select: int = 20) -> None:
    """``sample='median'``: among the first ``n_select`` test samples, the one whose scalar-metric R2 is closest to
    their median (the median case); an integer shows that test sample instead."""
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    from rhmp.tasks import load_task
    from rhmp.tasks.aniso import _load_blob, _p1_stiffness, cell_tensors
    t0 = time.time()
    base = load_task("AHP_r100", None, native=True, device=device, max_samples=60, fine=False)   # 20 test samples
    te = base.split[2]
    n_sel = min(n_select, len(te))
    labels = {"diag": "ours: scalar material metric", "tensor": "ours: tensor material metric",
              "mgn": "MeshGraphNet"}
    per, fields = {}, {}
    for key, rd in runs.items():
        model, view, _ = load_run(rd, base, device)
        per[key], fields[key] = [], []
        for q in range(n_sel):
            p, t = predict_one(model, view, int(te[q]), device)
            per[key].append(r2(p, t))
            fields[key].append((p[:, 0], t[:, 0]))
    med = {k: float(np.median(v)) for k, v in per.items()}
    ties: list = []
    if str(sample) == "median":
        dist = np.abs(np.asarray(per["diag"]) - med["diag"])
        # with an even count the two middle samples are equally close to the median: the lower index is shown
        ties = [int(x) for x in np.flatnonzero(np.isclose(dist, dist.min(), rtol=0.0, atol=1e-12))]
        q = ties[0]
    else:
        q = int(sample)
        if not 0 <= q < n_sel:
            raise ValueError(f"--ahp-sample must be 'median' or one of the first {n_sel} test samples")
    i = int(te[q])
    sid = int(base.meta["sample_ids"][i])
    preds = {k: dict(p=fields[k][q][0], t=fields[k][q][1], r2=per[k][q]) for k in runs}
    for key in runs:
        print(f"  fig2 {key}: R2 {preds[key]['r2']:.4f} on test sample {q} (median over the first {n_sel} test "
              f"samples {med[key]:.4f})", flush=True)
    K = base.K[i].to("cpu")
    pos, faces, edges, bnd = mesh_arrays(K)
    x = raw_inputs(base, i)
    true = preds["tensor"]["t"]
    # ground truth (evaluation only): per-triangle tensors -> fibre direction and ratio; exact P1 edge weights
    blob = _load_blob(base.meta["source"])
    S = cell_tensors(blob, sid, "sigma_face", K, "faces")
    ev, V = np.linalg.eigh(S.numpy())
    ratio = ev[:, 1] / np.clip(ev[:, 0], 1e-300, None)
    major = V[:, :, 1]
    Kst = _p1_stiffness(K, S)
    off = np.asarray(Kst[edges[:, 0], edges[:, 1]]).ravel()           # stiffness entry K_ab of every edge
    interior = ~bnd if bnd is not None else np.ones(len(edges), bool)
    neg = (off > 1e-12 * np.abs(off).max()) & interior                  # positive off-diagonal = negative weight
    neg_frac = float(neg.sum() / interior.sum())
    unorm = _lin_norm(true.min(), true.max())
    enorm = _lin_norm(0.0, max(np.quantile(np.abs(preds[k]["p"] - true), 0.995) for k in runs))
    order = ("diag", "tensor", "mgn")

    plt = _plt()
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(13.6, 6.95))
    gs = GridSpec(2, 8, figure=fig, width_ratios=GRID_RATIOS, wspace=0.06, hspace=0.30, left=0.012, right=0.955,
                  top=0.892, bottom=0.135)
    # fibre directions: one stroke per triangle along the fast axis, coloured by the anisotropy ratio
    ax = fig.add_subplot(gs[0, 0])
    cen = pos[faces].mean(1)[:, :2]
    ln = 0.8 * np.sqrt(2.0 * face_areas(pos, faces))
    seg = np.stack([cen - 0.5 * ln[:, None] * major, cen + 0.5 * ln[:, None] * major], 1)
    lc = LineCollection(seg, cmap=LinearSegmentedColormap.from_list("fibre", ["#c3c2b7", "#6da7ec", "#256abf",
                                                                             "#0d366b"]),
                        norm=Normalize(vmin=0.0, vmax=2.0), linewidths=0.75, capstyle="round")
    lc.set_array(np.log10(ratio))
    ax.add_collection(lc)
    _outline(ax, pos, edges, bnd)
    _limits(ax, pos)
    _clean(ax)
    _title(ax, "true fibre direction", "fast axis of Σ, one stroke per triangle")
    cb = _cbar(fig, lc, fig.add_subplot(gs[0, 1]), "anisotropy ratio", ticks=[0, 1, 2])
    cb.set_ticklabels(["1", "10", "100"])
    ax = fig.add_subplot(gs[0, 3])
    m_u = node_field(ax, pos, faces, true, unorm)
    _outline(ax, pos, edges, bnd)
    _clean(ax)
    _title(ax, "true solution u", f"test sample {q} ({'the median case' if str(sample) == 'median' else 'fixed'}, "
                                  "see below)")
    for c, k in enumerate(order):
        ax = fig.add_subplot(gs[0, 4 + c])
        node_field(ax, pos, faces, preds[k]["p"], unorm)
        _outline(ax, pos, edges, bnd)
        _clean(ax)
        _title(ax, labels[k], f"R2 {_fmt_r2(preds[k]['r2'])} (this sample)")
    _cbar(fig, m_u, fig.add_subplot(gs[0, 7]), "u", extend="both")
    # row B: source | negative-weight edges | error maps
    ax = fig.add_subplot(gs[1, 0])
    fv = x[0][:, 0]
    m_f = node_field(ax, pos, faces, fv, _sym_norm(np.abs(fv).max()), SIGNED_CMAP)
    _outline(ax, pos, edges, bnd)
    _clean(ax)
    _title(ax, "source f (input)", "at the vertices")
    _cbar(fig, m_f, fig.add_subplot(gs[1, 1]), "f")
    ax = fig.add_subplot(gs[1, 3])
    mesh_lines(ax, pos, edges[~neg], color="#dddcd5", lw=0.3)
    mesh_lines(ax, pos, edges[neg], color="#c8331f", lw=0.8, zorder=2)
    _outline(ax, pos, edges, bnd)
    _limits(ax, pos)
    _clean(ax)
    _title(ax, "edges a scalar metric cannot reproduce",
           f"red: negative weight in the exact operator ({100 * neg_frac:.0f} % of edges)")
    for c, k in enumerate(order):
        ax = fig.add_subplot(gs[1, 4 + c])
        err = np.abs(preds[k]["p"] - true)
        m_e = node_field(ax, pos, faces, err, enorm, ERR_CMAP)
        _outline(ax, pos, edges, bnd)
        _clean(ax)
        _title(ax, "error |pred − true|", labels[k].replace("ours: ", "ours, "))
        _corner(ax, _peak(err, true))
    _cbar(fig, m_e, fig.add_subplot(gs[1, 7]), "|pred − true|", extend="max")
    _suptitle(fig, "Anisotropic Poisson problem, fibre anisotropy up to 100:1: a scalar material metric is not enough")
    if str(sample) == "median":
        tie = "".join(f"; equally close: sample {x} ({per['diag'][x]:.3f}), the lower index is shown" for x in ties[1:])
        pick = (f"Test sample {q} is the median case: among the first {n_sel} test samples its scalar-metric R2 "
                f"({per['diag'][q]:.3f}) is the closest to their median ({med['diag']:.3f}{tie}).\nOn it the "
                f"tensor metric reaches {per['tensor'][q]:.3f} and MeshGraphNet {per['mgn'][q]:.3f} (their medians: "
                f"{med['tensor']:.3f} / {med['mgn']:.3f}).")
    else:
        pick = (f"Test sample {q}, R2 {per['diag'][q]:.3f} / {per['tensor'][q]:.3f} / {per['mgn'][q]:.3f}; medians "
                f"over the first {n_sel} test samples {med['diag']:.3f} / {med['tensor']:.3f} / {med['mgn']:.3f}.")
    _caption(fig, "-div(Σ grad u) = f with a misaligned SPD tensor Σ per triangle; the models see only rotation-free "
                  "inputs (edge projections t·Σt, log det Σ, log ratio), never the direction itself.\nA scalar "
                  "(diagonal) metric can only give positive edge weights; the exact operator needs negative ones on "
                  "the red edges.  One colour scale for all u panels, one for all error panels; R2 of this one "
                  f"sample.\n{pick}")
    _save(fig, out, "field2_anisotropic.png")
    plt.close(fig)
    STATS["field2_anisotropic"] = dict(task="AHP_r100", test_index=q, sample_id=sid, n0=int(K.n[0]), runs=runs,
                                       selection=("median case: scalar-metric R2 closest to its median over the first "
                                                  f"{n_sel} test samples") if str(sample) == "median" else "fixed",
                                       R2={k: preds[k]["r2"] for k in runs},
                                       median_R2_first_test_samples={k: dict(n=n_sel, R2=med[k]) for k in runs},
                                       per_sample_R2_first_test_samples=per, tied_with=ties[1:],
                                       negative_weight_edge_fraction_interior=neg_frac,
                                       median_anisotropy_ratio=float(np.median(ratio)), seconds=time.time() - t0)


# ================================================================================================================
# figure 3: gauge task, face flux on the training mesh, a new mesh and a 4x finer new mesh
# ================================================================================================================
def fig3(out: str, device, run: str, n: int = 500, field_seed: int = 123) -> None:
    import torch

    from rhmp.complex import CochainComplex
    from rhmp.data import TaskData
    from t6_mesh_transfer import generate
    t0 = time.time()
    meshes = [("the training mesh", 42, 1024), ("a new random mesh", 7, 1024), ("a new mesh, 4x finer", 7, 4096)]
    cols = []
    for label, seed, npts in meshes:
        # as scripts/t6_mesh_transfer.py (mesh seed 42 = the training mesh, gen_common.delaunay_1024); the same n and
        # field_seed give the same physical fields on every mesh
        pts, faces, edges, theta, plaq = generate(seed, npts, n, field_seed, device)
        K = CochainComplex.from_triangles(torch.tensor(pts, dtype=torch.float32), torch.tensor(faces), star="cotan",
                                          device=device)
        task = TaskData.from_arrays(K, {1: theta.unsqueeze(-1)}, plaq.unsqueeze(-1), readout="cochain:2",
                                    connection_dims={1: 1}, split=(0.5, 0.1, 0.4), name=f"T6f_mesh{seed}_n{npts}")
        model, view, _ = load_run(run, task, device)
        i = int(view.split[2][0])                                         # first test sample
        p, t = predict_one(model, view, i, device)
        _, _, kedges, bnd = mesh_arrays(K)
        cols.append(dict(label=label, seed=seed, n0=npts, pts=pts, faces=faces, edges=kedges, bnd=bnd,
                         area=face_areas(pts, faces), p=p[:, 0], t=t[:, 0], r2=r2(p, t, center=False), index=i))
        print(f"  fig3 {label}: uncentred R2 {cols[-1]['r2']:.5f} (sample {i} of {n})", flush=True)
    # Whitney 2-form view of the face cochain: flux / area, constant per triangle (faces are counter-clockwise)
    dens = [c["t"] / c["area"] for c in cols]
    fnorm = _sym_norm(weighted_quantile(np.abs(dens[0]), cols[0]["area"], 0.995))
    errs = [np.abs(c["p"] - c["t"]) / c["area"] for c in cols]
    enorm = _lin_norm(0.0, max(weighted_quantile(e, c["area"], 0.995) for e, c in zip(errs, cols, strict=True)))

    plt = _plt()
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(11.6, 11.0))
    gs = GridSpec(3, 4, figure=fig, width_ratios=[1, 1, 1, 0.045], wspace=0.07, hspace=0.19, left=0.012,
                  right=0.93, top=0.93, bottom=0.06)
    for c, col in enumerate(cols):
        pts, faces = col["pts"], col["faces"]
        for r, (vals, norm, cmap, main, sub) in enumerate((
                (dens[c], fnorm, SIGNED_CMAP, f"true flux, {col['label']}", f"test sample 0, {col['n0']:,} vertices"),
                (col["p"] / col["area"], fnorm, SIGNED_CMAP, "ours (trained on the first mesh only)",
                 f"R2 {_fmt_r2(col['r2'])} (this sample)"),
                (errs[c], enorm, ERR_CMAP, "error |pred − true|", col["label"]))):
            ax = fig.add_subplot(gs[r, c])
            m = face_field(ax, pts, faces, vals, norm, cmap)
            _outline(ax, pts, col["edges"], col["bnd"])
            _clean(ax)
            _title(ax, main, sub)
            if c == 2 and r == 0:
                _cbar(fig, m, fig.add_subplot(gs[0:2, 3]), "flux per unit area (flux / triangle area)",
                      extend="both")
            if c == 2 and r == 2:
                _cbar(fig, m, fig.add_subplot(gs[2, 3]), "|pred − true| / triangle area", extend="max")
    _suptitle(fig, "U(1) gauge field: one trained model predicts the magnetic flux on meshes it has never seen")
    _caption(fig, "Input: the connection θ on the edges (a smooth pure-gauge part plus 1-3 vortices); target: the "
                  "flux through every triangle (a 2-cochain), drawn as flux / area.\nThe same physical field on all "
                  "three meshes.  Truth and prediction share one colour scale, the errors have another; uncentred R2 "
                  "(odd face target) of this sample.")
    _save(fig, out, "field3_gauge_transfer.png")
    plt.close(fig)
    STATS["field3_gauge_transfer"] = dict(task="T6f", run=run, n=n, field_seed=field_seed,
                                          meshes=[dict(label=c["label"], mesh_seed=c["seed"], n0=c["n0"], R2=c["r2"],
                                                       sample_index_in_generated_set=c["index"]) for c in cols],
                                          seconds=time.time() - t0)


# ================================================================================================================
# figure 4: closed surfaces (in distribution and genus 2)
# ================================================================================================================
def _surface(ax, pos, faces, fvals, norm, cmap, elev, azim, zoom=1.25, light=(0.3, -0.5, 0.8)):
    """Surface coloured per face, with mild Lambertian shading for the 3-D shape (colours keep >= 72 % of their
    value); returns a ScalarMappable for the colour bar."""
    import matplotlib as mpl
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    tri = pos[faces]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30
    L = np.asarray(light, dtype=np.float64)
    L /= np.linalg.norm(L)
    shade = 0.72 + 0.28 * np.abs(nrm @ L)
    cols = mpl.colormaps[cmap](norm(fvals))
    cols[:, :3] *= shade[:, None]
    ax.add_collection3d(Poly3DCollection(tri, facecolors=cols, edgecolors=cols, linewidths=0.05))
    lo, hi = pos.min(0), pos.max(0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo, zoom=zoom)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    return mpl.cm.ScalarMappable(norm=norm, cmap=cmap)


def fig4(out: str, device, run: str, sample: int = 0, topo_sample: int = 0) -> None:
    from rhmp.tasks import load_task
    t0 = time.time()
    base = load_task("SURF", None, native=True, device=device, max_samples=60)
    model, view, _ = load_run(run, base, device)
    fam_names = ("ellipsoid", "superquadric", "perturbed sphere", "torus", "genus-2 surface")
    i = int(view.split[2][sample])
    fam = int(view.meta["family_of_sample"][i])
    p, t = predict_one(model, view, i, device)
    rows = [dict(K=view.K[i], p=p[:, 0], t=t[:, 0], r2=r2(p, t), what=f"true u, {fam_names[fam]} (test split)",
                 sub=f"test sample {sample}", view=(34, -60), zoom=1.3)]
    src = view.extra_tests["topo"]
    p, t = predict_one(model, view, topo_sample, device, source=src)
    rows.append(dict(K=src["K"][topo_sample], p=p[:, 0], t=t[:, 0], r2=r2(p, t), what="true u, genus-2 surface",
                     sub=f"topology never seen in training; sample {topo_sample}", view=(52, -62), zoom=1.22))
    for rw in rows:
        print(f"  fig4 {rw['what']}: R2 {rw['r2']:.4f}", flush=True)
    enorm = _lin_norm(0.0, max(np.quantile(np.abs(rw["p"] - rw["t"]), 0.995) for rw in rows))
    plt = _plt()
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(12.6, 8.2))
    gs = GridSpec(2, 6, figure=fig, width_ratios=[1, 1, 0.035, 0.12, 1, 0.035], wspace=0.04, hspace=0.1,
                  left=0.01, right=0.95, top=0.9, bottom=0.075)
    for r, rw in enumerate(rows):
        pos, faces, _, _ = mesh_arrays(rw["K"])

        def fmean(v, faces=faces):                              # vertex field -> face colour (mean of the corners)
            return np.asarray(v)[faces].mean(1)
        unorm = _lin_norm(rw["t"].min(), rw["t"].max())
        elev, azim = rw["view"]
        for c, (vals, main, sub) in enumerate(((rw["t"], rw["what"], f"{rw['sub']}, {rw['K'].n[0]:,} vertices"),
                                               (rw["p"], "ours: general network",
                                                f"R2 {_fmt_r2(rw['r2'])} (this sample)"))):
            ax = fig.add_subplot(gs[r, c], projection="3d")
            sm = _surface(ax, pos, faces, fmean(vals), unorm, FIELD_CMAP, elev, azim, zoom=rw["zoom"])
            _title(ax, main, sub)
        _cbar(fig, sm, fig.add_subplot(gs[r, 2]), "u", extend="both")
        ax = fig.add_subplot(gs[r, 4], projection="3d")
        err = np.abs(rw["p"] - rw["t"])
        sm = _surface(ax, pos, faces, fmean(err), enorm, ERR_CMAP, elev, azim, zoom=rw["zoom"])
        _title(ax, "error |pred − true|", "same viewpoint")
        _corner(ax, _peak(err, rw["t"]))
        _cbar(fig, sm, fig.add_subplot(gs[r, 5]), "|pred − true|", extend="max")
    _suptitle(fig, "Screened Poisson problem on closed surfaces: a torus from the test split and a genus-2 surface")
    _caption(fig, "(M + ε L) u = M f on curved triangle meshes in 3-D; the model uses intrinsic geometry only.  "
                  "Training surfaces: ellipsoids, perturbed spheres and tori (genus 0 and 1).\nColour scale per row "
                  "= range of the true u; one error scale for both rows; faces coloured by the mean of their vertex "
                  "values; R2 of this one sample.")
    _save(fig, out, "field4_surface.png")
    plt.close(fig)
    STATS["field4_surface"] = dict(task="SURF", run=run, test_index=sample, family=fam_names[fam],
                                   topo_index=topo_sample, R2=dict(test=rows[0]["r2"], topo=rows[1]["r2"]),
                                   n0=[int(rw["K"].n[0]) for rw in rows], seconds=time.time() - t0)


# ================================================================================================================
# figure 5: rollout snapshots (DYNfix)
# ================================================================================================================
def fig5(out: str, device, run: str, traj: int = 0, steps=(1, 10, 25, 50, 100)) -> None:
    import torch

    from rhmp.model import RHMP
    from rhmp.tasks import load_task
    from rhmp.tasks.suite import rollout_eval
    from rhmp.train import _renormalize
    t0 = time.time()
    conf = json.load(open(os.path.join(run, "config.json")))
    task = load_task("DYNfix", None, native=True, device=device, star=conf["args"].get("star", "cotan"))
    task = _renormalize(task, conf["data"])                       # as scripts/eval_on.py --rollout (in place)
    model = RHMP.from_checkpoint(torch.load(os.path.join(run, "best.pt"), map_location=device, weights_only=False),
                                 map_location=device).eval()
    roll = {}
    for key, proj in (("proj", True), ("plain", False)):
        r = rollout_eval(model, task, max(steps), device=device, return_pred=True, project_mass=proj)
        roll[key] = dict(pred=r["pred"][traj].double().numpy(), pooled=r["summary"])
    true = task.rollout["traj"][traj].double().numpy()               # (T+1, n0)
    mass = task.rollout["mass"].double().numpy()
    pos, faces, edges, bnd = mesh_arrays(task.rollout["K"])
    unorm = _lin_norm(true[list(steps)].min(), true[list(steps)].max())
    rows = [("true state", true, None), ("ours, mass conserved exactly", roll["proj"]["pred"], "proj"),
            ("ours, plain rollout", roll["plain"]["pred"], "plain")]
    stats = {}
    plt = _plt()
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(13.4, 8.4))
    gs = GridSpec(3, 6, figure=fig, width_ratios=[1] * len(steps) + [0.045], wspace=0.06, hspace=0.26,
                  left=0.075, right=0.955, top=0.905, bottom=0.085)
    for r, (label, U, key) in enumerate(rows):
        for c, s in enumerate(steps):
            ax = fig.add_subplot(gs[r, c])
            m = node_field(ax, pos, faces, U[s], unorm)
            _outline(ax, pos, edges, bnd)
            _clean(ax)
            if key is None:
                _title(ax, f"step {s}", f"test trajectory {traj} (first)" if c == 0 else None)
            else:
                v = r2(U[s], true[s])
                drift = abs(float(mass @ U[s]) - float(mass @ true[0])) / float(mass @ np.abs(true[0]))
                stats.setdefault(key, {})[f"step{s}"] = dict(R2=v, mass_drift=drift)
                _title(ax, f"R2 {_fmt_r2(v)}", f"mass drift {100 * drift:.1f} %" if drift >= 5e-4 else "mass drift 0")
            if c == 0:
                ax.text(-0.06, 0.5, label, transform=ax.transAxes, rotation=90, ha="right", va="center",
                        fontsize=9.5, fontweight="semibold", color=INK)
        if r == 0:
            _cbar(fig, m, fig.add_subplot(gs[:, len(steps)]), "u (advected and diffused scalar)", extend="both")
    p100, q100 = roll["proj"]["pooled"].get("R2@100"), roll["plain"]["pooled"].get("R2@100")
    _suptitle(fig, "Advection-diffusion on a fixed mesh: 100-step forecasts, each step fed with the previous "
                   "prediction")
    _caption(fig, "One colour scale for all panels.  R2 of this trajectory at that step: late fields have little "
                  f"variance left, so it falls fast (pooled over all 75 test trajectories at step 100: {p100:.2f} "
                  f"with, {q100:.2f} without the mass correction).\nThe no-change baseline (repeat the initial state; "
                  "close to the step-1 truth) is not drawn: its pooled R2 is below 0 from step 25 on.")
    _save(fig, out, "field5_rollout_snapshots.png")
    plt.close(fig)
    STATS["field5_rollout_snapshots"] = dict(task="DYNfix", run=run, test_trajectory=traj, steps=list(steps),
                                             per_trajectory=stats,
                                             pooled={k: {str(h): roll[k]["pooled"].get(f"R2@{h}") for h in steps
                                                         if f"R2@{h}" in roll[k]["pooled"]} for k in roll},
                                             seconds=time.time() - t0)


# ================================================================================================================
# figure 6: tangent vector field on an ellipsoid (T3)
# ================================================================================================================
def fig6(out: str, device, run: str, sample: int = 0, n_arrows: int = 230) -> None:
    import matplotlib as mpl
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from rhmp.tasks import load_task
    t0 = time.time()
    conf = json.load(open(os.path.join(run, "config.json")))
    base = load_task("T3", None, native=True, device=device, star=conf["args"].get("star", "cotan"))
    model, view, _ = load_run(run, base, device)
    i = int(view.split[2][sample])
    p, t = predict_one(model, view, i, device)                        # (n0, 3): v = n x grad psi
    pos, faces, _, _ = mesh_arrays(view.K)
    psi = raw_inputs(base, i)[0][:, 0]
    val = r2(p, t)
    err = np.linalg.norm(p - t, axis=1)
    rel = float(np.linalg.norm(p - t) / np.linalg.norm(t))
    print(f"  fig6: R2 {val:.4f}, relative L2 error {rel:.4f}", flush=True)
    elev, azim, zoom = 24, -58, 1.3
    ce, se = math.cos(math.radians(elev)), math.sin(math.radians(elev))
    cam = np.array([ce * math.cos(math.radians(azim)), ce * math.sin(math.radians(azim)), se])
    tri = pos[faces]
    vn = np.zeros_like(pos)
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-30
    vn *= np.sign(((pos - pos.mean(0)) * vn).sum(1, keepdims=True))   # outward normals
    vis = np.where(vn @ cam > 0.2)[0]                                 # vertices facing the camera
    sel = vis[farthest_points(pos[vis], n_arrows)]                    # evenly spread arrows
    scale = 0.085 * np.ptp(pos, 0).max() / np.quantile(np.linalg.norm(t, axis=1), 0.95)
    pnorm = _sym_norm(np.abs(psi).max())
    enorm = _lin_norm(0.0, np.quantile(err, 0.995))
    lo, hi = pos.min(0), pos.max(0)
    plt = _plt()
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(13.0, 5.3))
    gs = GridSpec(1, 6, figure=fig, width_ratios=[1, 1, 0.035, 0.16, 1, 0.035], wspace=0.04, left=0.01,
                  right=0.95, top=0.86, bottom=0.12)
    panels = ((t, "true flow v = n x grad ψ", f"test sample {sample}, {len(pos):,} vertices"),
              (p, "ours: general network", f"R2 {_fmt_r2(val)} (this sample)"))
    for c, (vec, main, sub) in enumerate(panels):
        ax = fig.add_subplot(gs[0, c], projection="3d")
        cols = mpl.colormaps[SIGNED_CMAP](pnorm(psi[faces].mean(1)))
        cols[:, 3] = 0.6
        ax.add_collection3d(Poly3DCollection(tri, facecolors=cols, edgecolors="none", linewidths=0))
        q = pos[sel] + 0.004 * vn[sel]
        ax.quiver(q[:, 0], q[:, 1], q[:, 2], vec[sel, 0] * scale, vec[sel, 1] * scale, vec[sel, 2] * scale,
                  color=INK, linewidth=0.75, arrow_length_ratio=0.3, normalize=False)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo, zoom=zoom)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        _title(ax, main, sub)
    _cbar(fig, mpl.cm.ScalarMappable(norm=pnorm, cmap=SIGNED_CMAP), fig.add_subplot(gs[0, 2]),
          "stream function ψ (input)")
    ax = fig.add_subplot(gs[0, 4], projection="3d")
    sm = _surface(ax, pos, faces, err[faces].mean(1), enorm, ERR_CMAP, elev, azim, zoom=zoom)
    _title(ax, "error |pred − true| (vector length)", f"relative L2 error {100 * rel:.1f} %")
    _cbar(fig, sm, fig.add_subplot(gs[0, 5]), "|pred − true|", extend="max")
    _suptitle(fig, "Tangent flow on an ellipsoid: the model outputs a vector field on a curved surface")
    _caption(fig, "Input: the stream function ψ at the vertices (colour); target: the rotated gradient n x grad ψ "
                  "(arrows at evenly spread visible vertices, the same length scale in both panels).  R2 of this "
                  "sample.")
    _save(fig, out, "field6_t3_vector.png")
    plt.close(fig)
    STATS["field6_t3_vector"] = dict(task="T3", run=run, test_index=sample, R2=val, relative_L2=rel,
                                     seconds=time.time() - t0)


# ================================================================================================================
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "figures"))
    ap.add_argument("--stats", default=os.path.join(ROOT, "results", "cab75", "fieldviz", "field_figures_stats.json"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--only", default=None, help="comma-separated figure numbers, e.g. 1,3")
    ap.add_argument("--hp-sample", type=int, default=0, help="figure 1: test sample (default 0, the first)")
    ap.add_argument("--ahp-sample", default="median",
                    help="figure 2: 'median' (default: the median case of the scalar-metric R2 over the first 20 test "
                         "samples) or a test sample index")
    ap.add_argument("--hp-solver", default=os.path.join(ZOO, "HP_k100_S3c_solver_tensor_learn"))
    ap.add_argument("--hp-general", default=os.path.join(ZOO, "HP_k100_general_s42"))
    ap.add_argument("--hp-mgn", default=os.path.join(ZOO, "HP_k100_mgn_s42"))
    ap.add_argument("--ahp-diag", default=os.path.join(ZOO, "AHP_r100_diag-solver_s42"))
    ap.add_argument("--ahp-tensor", default=os.path.join(ZOO, "AHP_r100_tensor-solver_lr3e-4_s42"))
    ap.add_argument("--ahp-mgn", default=os.path.join(ZOO, "AHP_r100_mgn_s42"))
    ap.add_argument("--t6f", default=os.path.join(ZOO, "T6f_s42"))
    ap.add_argument("--surf", default=os.path.join(ZOO, "SURF_s42"))
    ap.add_argument("--dyn", default=os.path.join(ZOO, "DYNfix_s42"))
    ap.add_argument("--t3", default=os.path.join(ZOO, "T3_native_s42"))
    a = ap.parse_args(argv)
    import torch
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out, stats_path = os.path.abspath(a.out), os.path.abspath(a.stats)
    os.makedirs(out, exist_ok=True)
    only = None if not a.only else {int(s) for s in a.only.split(",")}
    rel = lambda p: os.path.relpath(os.path.abspath(p), ROOT)            # noqa: E731  (paths in the stats file)
    jobs = [
        (1, lambda: fig1(out, device, {"solver": rel(a.hp_solver), "general": rel(a.hp_general),
                                       "mgn": rel(a.hp_mgn)}, sample=a.hp_sample)),
        (2, lambda: fig2(out, device, {"diag": rel(a.ahp_diag), "tensor": rel(a.ahp_tensor),
                                       "mgn": rel(a.ahp_mgn)}, sample=a.ahp_sample)),
        (3, lambda: fig3(out, device, rel(a.t6f))),
        (4, lambda: fig4(out, device, rel(a.surf))),
        (5, lambda: fig5(out, device, rel(a.dyn))),
        (6, lambda: fig6(out, device, rel(a.t3))),
    ]
    prev = {}
    if os.path.exists(stats_path):
        try:
            prev = json.load(open(stats_path))
        except ValueError:
            prev = {}
    os.chdir(ROOT)                          # run directories are resolved relative to the repository
    for k, fn in jobs:
        if only is None or k in only:
            t0 = time.time()
            fn()
            print(f"figure {k}: {time.time() - t0:.1f} s", flush=True)
    prev.update(STATS)
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w") as fh:
        json.dump(prev, fh, indent=2)
    print(f"wrote {stats_path}")


if __name__ == "__main__":
    main()
