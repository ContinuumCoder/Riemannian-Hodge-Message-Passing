"""
One-sample input visualization per task (standalone figure).
Produces: ablation/figures/{task}/input.png

T1 : regular 32x32 grid, 4 channels (rho, Vx, Vy, p)  + mesh underlay
T6 : irregular Delaunay on 3D-embedded points, U(1) connection (3 encoded channels)
T7 : irregular Delaunay on 3D-embedded points, SU(2) connection (9 encoded channels)
"""
import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TASK_META = {
    "T1": {
        "pkl": "datasets/T1_cns_vorticity.pkl",
        "title": "T1  CNS Vorticity (regular 32$\\times$32 grid)",
        "channels": [r"$\rho$ (density)", r"$V_x$", r"$V_y$", r"$p$ (pressure)"],
        "output": r"$\omega$ (vorticity)",
    },
    "T6": {
        "pkl": "datasets/T6_wilson_loop.pkl",
        "title": "T6  Wilson Loop  U(1) gauge (irregular Delaunay)",
        "channels": [r"$\theta\!\cdot\!\bar 1$", r"$\theta\!\cdot\!d_x$", r"$\theta\!\cdot\!d_y$"],
        "output": r"$F$ (U(1) curvature)",
    },
    "T7": {
        "pkl": "datasets/T7_yang_mills_su2.pkl",
        "title": "T7  Yang--Mills SU(2) (irregular Delaunay, 3 Lie-algebra channels)",
        "channels": [f"$A^{{{a}}}\\!\\cdot\\!{g}$"
                     for a in (1, 2, 3) for g in (r"\bar 1", r"d_x", r"d_y")],
        "output": r"$F^a = dA^a + \frac{1}{2}[A,A]^a$",
    },
    "T3": {
        "pkl": "datasets/T3_ellipsoid_coexact.pkl",
        "title": "T3  Ellipsoid Surface Flow (triangulated closed surface, tangent vector)",
        "channels": [r"$\psi$ (stream function)"],
        "output": r"$\mathbf{v}_\mathrm{tan}$ (tangent velocity)",
    },
}


def _plot_grid_channels(ax, pts, tri, values, title, cmap="RdBu_r"):
    tc = ax.tripcolor(tri, values, cmap=cmap, shading="gouraud")
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    plt.colorbar(tc, ax=ax, fraction=0.046, pad=0.02)


def visualize(task: str, sample_idx: int = 0, save_dir: str | None = None):
    meta = TASK_META[task]
    d = pickle.load(open(os.path.join(ROOT, meta["pkl"]), "rb"))
    pts = np.asarray(d["points"], dtype=np.float32)
    faces = np.asarray(d["faces"], dtype=np.int64)
    X = np.asarray(d["X_data"])[sample_idx]
    Y = np.asarray(d["Y_data"])[sample_idx]
    if Y.ndim == 1: Y = Y[:, None]
    if X.ndim == 1: X = X[:, None]

    # 2D embedding for triangulation
    pts2 = pts[:, :2]
    tri = Triangulation(pts2[:, 0], pts2[:, 1], triangles=faces)

    ch = meta["channels"]
    n_ch = len(ch); n_out = Y.shape[-1]

    # Layout: inputs in a (ceil(n_ch/3) x min(n_ch,3)) grid, then a separate
    # "target" row of width = max(n_out, 3). Keeps the figure compact for T7 (9 in).
    in_cols = 3 if n_ch > 4 else n_ch
    in_rows = (n_ch + in_cols - 1) // in_cols
    bottom_cols = max(n_out, in_cols)

    height = 2.3 * in_rows + 2.6
    width = 2.3 * max(in_cols, bottom_cols)
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(in_rows + 1, max(in_cols, bottom_cols),
                          hspace=0.45, wspace=0.35)
    fig.suptitle(meta["title"], fontsize=12, y=0.98)

    for idx, channel in enumerate(ch):
        r, c = divmod(idx, in_cols)
        ax = fig.add_subplot(gs[r, c])
        _plot_grid_channels(ax, pts2, tri, X[:, idx], f"Input: {channel}")
    for j in range(n_out):
        ax = fig.add_subplot(gs[in_rows, j])
        label = meta["output"] if n_out == 1 else f"{meta['output']} ({'xyz'[j]})"
        _plot_grid_channels(ax, pts2, tri, Y[:, j], f"Target: {label}", cmap="viridis")

    plt.tight_layout(rect=(0, 0, 1, 0.96))

    if save_dir is None:
        save_dir = os.path.join(ROOT, "ablation", "figures", task)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "input.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    # Also dump mesh + sample idx metadata
    with open(os.path.join(save_dir, "sample_meta.pkl"), "wb") as f:
        pickle.dump({"task": task, "sample_idx": sample_idx,
                     "pts": pts, "faces": faces, "X": X, "Y": Y}, f)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=["T1", "T3", "T6"])
    ap.add_argument("--sample", type=int, default=9500,
                    help="sample index (picked from test split)")
    args = ap.parse_args()
    for t in args.tasks:
        visualize(t, sample_idx=args.sample)
