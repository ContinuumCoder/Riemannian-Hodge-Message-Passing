"""
Output comparison figure per task.

Loads all *_test.pkl files under ablation/outputs/{task}/ and builds a
grid: ground truth | ours_main | {ablation variants} | {selected baselines}.

Target output (for vector tasks, T3) is rendered as vector magnitude to keep
a single colormap.

Usage:
  python3 ablation/visualize_outputs.py --tasks T1 T6 T7 --sample 9500
"""
import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS)
sys.path.insert(0, THIS)
TASKS_MAP = {"T1": "T1_cns_vorticity", "T3": "T3_ellipsoid_surface_flow",
             "T6": "T6_wilson_loop", "T7": "T7_yang_mills_su2"}

ORDER = ["ours_main", "noH", "nocross", "relu", "learnD",
         "cw_net", "gauge_cnn", "fno", "egnn", "schnet", "mpsn", "clifford_smpn"]
LABELS = {
    "ours_main":  "Ours (full)",
    "noH":        r"A1: $H_k = I$",
    "nocross":    r"A3: no cross-dim",
    "relu":       r"A5: ReLU gate",
    "learnD":     r"A4: learnable $d_k$",
    "cw_net":     "CW Net",
    "gauge_cnn":  "GaugeEquivCNN",
    "fno":        "FNO",
    "egnn":       "EGNN",
    "schnet":     "SchNet",
    "mpsn":       "MPSN",
    "clifford_smpn": "Clifford-SMPN",
}
TITLE = {
    "T1": "T1  CNS Vorticity $\\omega$: prediction vs. ablations",
    "T3": "T3  Ellipsoid Surface Flow $\\|\\mathbf{v}_\\mathrm{tan}\\|$: prediction vs. ablations",
    "T6": "T6  Wilson Loop  $F$: prediction vs. ablations",
    "T7": "T7  Yang--Mills SU(2)  $\\|F^a\\|$: prediction vs. ablations",
}


def _r2(pred, tgt):
    ss_res = float(((pred - tgt) ** 2).sum())
    ss_tot = float(((tgt - tgt.mean()) ** 2).sum())
    return 1 - ss_res / max(ss_tot, 1e-12)


def _load_ground_truth(task: str):
    meta = pickle.load(open(os.path.join(ROOT, "ablation", "figures", task,
                                          "sample_meta.pkl"), "rb"))
    return meta  # {pts, faces, X, Y, sample_idx}


def visualize_task(task: str, sample_idx: int = 9500):
    out_dir = os.path.join(ROOT, "ablation", "outputs", task)
    if not os.path.isdir(out_dir):
        print(f"[{task}] no outputs directory, skipping"); return

    meta = _load_ground_truth(task)
    pts = meta["pts"]; faces = meta["faces"]
    pts2 = pts[:, :2]
    tri = Triangulation(pts2[:, 0], pts2[:, 1], triangles=faces)

    # Gather predictions. Test split = last 15% of N. Since N and test_idx live
    # in the full dataset frame, map: test_idx_in_pkl = sample_idx - (N - n_test).
    preds = {}
    for name in ORDER:
        p = os.path.join(out_dir, f"{name}_test.pkl")
        if not os.path.exists(p):
            continue
        d = pickle.load(open(p, "rb"))
        n_test = len(d["target"])
        # Reconstruct N from split ratio (15% test)
        N_est = round(n_test / 0.15)
        test_start_full = N_est - n_test
        test_idx = sample_idx - test_start_full
        if not (0 <= test_idx < n_test):
            test_idx = max(0, min(n_test - 1, test_idx))
        preds[name] = (d["pred"][test_idx], d["target"][test_idx])

    if not preds:
        print(f"[{task}] no predictions found, skipping"); return

    # Target once (from first available)
    gt = list(preds.values())[0][1]
    if gt.ndim == 1:
        gt = gt[:, None]

    # For vector tasks reduce to magnitude
    gt_vis = np.linalg.norm(gt, axis=-1) if gt.shape[-1] > 1 else gt[:, 0]

    names = [n for n in ORDER if n in preds]
    n_cols = len(names) + 1  # +1 for GT
    n_rows = 1
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.5 * n_cols, 3.0),
                             squeeze=False)
    fig.suptitle(TITLE.get(task, task), fontsize=12, y=1.02)

    vmin = float(gt_vis.min()); vmax = float(gt_vis.max())
    tc = axes[0, 0].tripcolor(tri, gt_vis, cmap="RdBu_r",
                              shading="gouraud", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("Ground truth", fontsize=10)
    axes[0, 0].set_aspect("equal"); axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])
    plt.colorbar(tc, ax=axes[0, 0], fraction=0.046, pad=0.02)

    # Cross-model full-test R² for annotation -- use the same formula as
    # run_ablation.py's compute_metrics (per-node mean as baseline).
    r2_full = {}
    import json as _json
    from variants import VARIANT_SPEC as _VS
    tid = TASKS_MAP[task]  # short -> dataset folder
    for name in names:
        r2 = None
        # 1) Prefer the cached result.json if this is an ablation variant
        if name in _VS:
            rp = os.path.join(ROOT, "ablation", "checkpoints", name, tid, "result.json")
            if os.path.exists(rp):
                r2 = _json.load(open(rp)).get("test_R2")
        # 2) ours_main -> reuse main benchmark all_metrics.json (100-ep)
        if r2 is None and name == "ours_main":
            am = os.path.join(ROOT, "all_metrics.json")
            if os.path.exists(am):
                r2 = _json.load(open(am)).get(task, {}).get("ours", {}).get("R2")
        # 3) baseline -> all_metrics.json
        if r2 is None:
            am = os.path.join(ROOT, "all_metrics.json")
            if os.path.exists(am):
                r2 = _json.load(open(am)).get(task, {}).get(name, {}).get("R2")
        # 4) fallback: compute ourselves with the right formula
        if r2 is None:
            pth = os.path.join(out_dir, f"{name}_test.pkl")
            d = pickle.load(open(pth, "rb"))
            pred = d["pred"]; tgt = d["target"]
            ss_res = float(((pred - tgt) ** 2).sum())
            ss_tot = float(((tgt - tgt.mean(0, keepdims=True)) ** 2).sum())
            r2 = 1 - ss_res / max(ss_tot, 1e-12)
        r2_full[name] = r2

    for j, name in enumerate(names, start=1):
        p, _ = preds[name]
        if p.ndim == 1:
            p = p[:, None]
        p_vis = np.linalg.norm(p, axis=-1) if p.shape[-1] > 1 else p[:, 0]
        tc = axes[0, j].tripcolor(tri, p_vis, cmap="RdBu_r",
                                  shading="gouraud", vmin=vmin, vmax=vmax)
        lbl = LABELS.get(name, name)
        r2 = r2_full.get(name, None)
        sub = f"\n$R^2$ = {r2:.3f}" if r2 is not None else ""
        axes[0, j].set_title(f"{lbl}{sub}", fontsize=10)
        axes[0, j].set_aspect("equal"); axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
        plt.colorbar(tc, ax=axes[0, j], fraction=0.046, pad=0.02)

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    save_dir = os.path.join(ROOT, "ablation", "figures", task)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "output_comparison.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}  ({len(names)+1} panels, {len(preds)} models)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=["T1", "T3", "T6"])
    ap.add_argument("--sample", type=int, default=9500)
    args = ap.parse_args()
    for t in args.tasks:
        visualize_task(t, sample_idx=args.sample)
