"""
Re-run inference on held-out split from each ablation variant's best_model.pt.
Overwrites ablation/outputs/{task}/{variant}_test.pkl in place.

Used before regenerating qualitative figures so the pkls match the checkpoints
on disk (rather than whatever was dumped at training end).
"""
import os, sys, pickle
import numpy as np
import torch

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from gauge_hodge_mp.cell_complex import CellComplex
from gauge_hodge_mp.network import GaugeHodgeNetwork
from formal_benchmark import TASKS
from variants import VARIANT_SPEC

TASK_LIST = ["T1", "T3", "T6"]
VARIANTS = ["noH", "nocross", "relu", "learnD"]


def reinfer(task_id: str, variant: str, device: str = "cuda"):
    cfg = TASKS[task_id]
    flags = {k: v for k, v in VARIANT_SPEC[variant].items() if k != "desc"}
    with open(os.path.join(ROOT, cfg["dataset"]), "rb") as f:
        d = pickle.load(f)
    pts = torch.tensor(np.array(d["points"], dtype=np.float32))
    faces = torch.tensor(np.array(d["faces"], dtype=np.int64))
    K = CellComplex.from_triangulation(pts, faces).to(device)

    X = torch.tensor(np.array(d["X_data"], dtype=np.float32))
    Y = torch.tensor(np.array(d["Y_data"], dtype=np.float32))
    if X.ndim == 2: X = X.unsqueeze(-1)
    if Y.ndim == 2: Y = Y.unsqueeze(-1)
    f_in, out_dim = X.shape[-1], Y.shape[-1]

    n = len(X); nt = int(0.7 * n); nv = int(0.15 * n)
    xm, xs = X[:nt].mean((0, 1)), X[:nt].std((0, 1)).clamp(1e-6)
    ym, ys = Y[:nt].mean((0, 1)), Y[:nt].std((0, 1)).clamp(1e-6)
    X_test = ((X[nt + nv:] - xm) / xs).to(device)
    Y_test = Y[nt + nv:]

    m = GaugeHodgeNetwork(
        f_in=f_in, C=cfg["ours_C"], n_layers=4,
        n0=K.n0, n1=K.n1, n2=K.n2,
        task=cfg["task_type"], spatial_dim=cfg["spatial_dim"],
        out_dim=out_dim, mp_hidden=16,
        metric_type=cfg["metric_type"], metric_rank=cfg["metric_rank"],
        **flags,
    ).to(device)
    with torch.no_grad():
        _ = m.forward_batch(X_test[:1], K)

    ckpt = os.path.join(ROOT, "ablation", "checkpoints", variant,
                        f"{task_id}_{cfg['name']}", "best_model.pt")
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    m.eval()

    preds = []
    bs = cfg["batch_size"]
    with torch.no_grad():
        for i in range(0, len(X_test), bs):
            preds.append(m.forward_batch(X_test[i:i + bs], K).cpu())
    pred_n = torch.cat(preds)
    pred_r = (pred_n * ys + ym).numpy()
    tgt_r = Y_test.numpy()

    out_dir = os.path.join(ROOT, "ablation", "outputs", task_id)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{variant}_test.pkl"), "wb") as f:
        pickle.dump({"pred": pred_r, "target": tgt_r,
                     "variant": variant, "task": task_id}, f)

    ss_res = float(((pred_r - tgt_r) ** 2).sum())
    ss_tot = float(((tgt_r - tgt_r.mean(0, keepdims=True)) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    print(f"[{task_id}/{variant}] reinferred; R2={r2:.4f}")


def main():
    for t in TASK_LIST:
        for v in VARIANTS:
            reinfer(t, v)


if __name__ == "__main__":
    main()
