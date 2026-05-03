"""
Structure-preservation diagnostics on existing trained checkpoints.

Computes two measurements (no retraining):

  1. Conservation residual
        C(task, variant) = ||d_0 (Y_pred - Y_target)||_F / ||d_0 Y_target||_F
     averaged over the held-out test split. d_0 is the fixed CW-complex
     coboundary of the dataset's mesh. A model that preserves the discrete
     differential structure has its prediction error annihilated (or at least
     not amplified) by d_0. The learnable-d_k ablation, which silently learns
     d_0 / d_1 entries, gives up the d^2 = 0 constraint and the prediction
     error is no longer compatible with the fixed coboundary.

  2. Gauge consistency on T6 Wilson loop
        G(variant) = || f(X + (d_0 phi)_encoded) - f(X) ||_F / ||f(X)||_F
     averaged over 5 random node-potential samples phi ~ N(0, 0.01 I_{n0}).
     The model's input is a node-encoded representation of the U(1) connection;
     adding a node-level potential is the discrete analog of the U(1) gauge
     transformation A -> A + d phi. A model whose H_k is built from O(C)-
     invariants (full GSHMP) is closer to gauge-consistent than one without
     learnable metric (noH variant) because the metric responds to gauge-
     invariant statistics rather than coordinate values.

Outputs ablation/results/conservation.json
"""
import os
import sys
import json
import pickle

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TASKS_TO_MEASURE = ["T1", "T3", "T6", "T7"]


def _load_split(task_id):
    cfg = TASKS[task_id]
    with open(os.path.join(ROOT, cfg["dataset"]), "rb") as f:
        d = pickle.load(f)
    pts = torch.tensor(np.array(d["points"], dtype=np.float32))
    faces = torch.tensor(np.array(d["faces"], dtype=np.int64))
    K = CellComplex.from_triangulation(pts, faces).to(DEVICE)

    X = torch.tensor(np.array(d["X_data"], dtype=np.float32))
    Y = torch.tensor(np.array(d["Y_data"], dtype=np.float32))
    if X.ndim == 2:
        X = X.unsqueeze(-1)
    if Y.ndim == 2:
        Y = Y.unsqueeze(-1)
    n = len(X)
    nt = int(0.7 * n)
    nv = int(0.15 * n)
    xm, xs = X[:nt].mean((0, 1)), X[:nt].std((0, 1)).clamp(1e-6)
    ym, ys = Y[:nt].mean((0, 1)), Y[:nt].std((0, 1)).clamp(1e-6)
    X_test = ((X - xm) / xs)[nt + nv:].to(DEVICE)
    Y_test_raw = Y[nt + nv:].to(DEVICE)
    return K, cfg, X_test, Y_test_raw, ym.to(DEVICE), ys.to(DEVICE)


def _build_model(cfg, variant, K, f_in, out_dim):
    flags = {k: v for k, v in VARIANT_SPEC[variant].items() if k != "desc"}
    return GaugeHodgeNetwork(
        f_in=f_in, C=cfg["ours_C"], n_layers=4,
        n0=K.n0, n1=K.n1, n2=K.n2,
        task=cfg["task_type"], spatial_dim=cfg["spatial_dim"],
        out_dim=out_dim, mp_hidden=16,
        metric_type=cfg["metric_type"], metric_rank=cfg["metric_rank"],
        **flags,
    ).to(DEVICE)


def _checkpoint_path(task_id, variant, cfg):
    if variant == "full":
        return os.path.join(ROOT, "checkpoints", f"{task_id}_{cfg['name']}",
                            "ours", "best_model.pt")
    return os.path.join(ROOT, "ablation", "checkpoints", variant,
                        f"{task_id}_{cfg['name']}", "best_model.pt")


def _load_model(task_id, variant, K, cfg, f_in, out_dim):
    m = _build_model(cfg, variant, K, f_in, out_dim)
    # Prime any lazy params (learnable d_k vals are created on first forward)
    with torch.no_grad():
        _ = m.forward_batch(torch.zeros(1, K.n0, f_in, device=DEVICE), K)
    sd = torch.load(_checkpoint_path(task_id, variant, cfg),
                    map_location=DEVICE, weights_only=False)
    m.load_state_dict(sd)
    m.eval()
    return m


def _inference(model, X_test, K, ym, ys, batch=64):
    preds = []
    with torch.no_grad():
        for i in range(0, X_test.size(0), batch):
            p = model.forward_batch(X_test[i:i + batch], K)
            preds.append(p)
    P = torch.cat(preds, 0)
    return P * ys + ym


def conservation_residual(task_id, variant):
    K, cfg, X_test, Y_target, ym, ys = _load_split(task_id)
    f_in = X_test.shape[-1]
    out_dim = Y_target.shape[-1]

    if Y_target.shape[1] != K.n0:
        return None  # not node-level output; skip

    m = _load_model(task_id, variant, K, cfg, f_in, out_dim)
    Y_pred = _inference(m, X_test, K, ym, ys)
    err = Y_pred - Y_target  # (B, n0, out_dim)

    d0 = K.d0.to_dense()  # (n1, n0)
    de = torch.einsum("ij,bjk->bik", d0, err)
    dt = torch.einsum("ij,bjk->bik", d0, Y_target)
    num = de.norm(dim=(1, 2))
    den = dt.norm(dim=(1, 2)).clamp_min(1e-12)
    val = float((num / den).mean().item())
    del m
    torch.cuda.empty_cache()
    return val


def gauge_consistency_T6(variant, n_trials=5, n_samples=64, scale=0.01, seed=0):
    """Approximation of U(1) gauge invariance: shift the (node-encoded) input
    by a random node potential and measure how much the prediction changes.

    The exact U(1) action is on edges (A -> A + d phi). T6's network sees a
    node-encoded summary of A; we proxy d phi by shifting node features by phi
    broadcast over channels. Magnitude scale=0.01 keeps the perturbation small
    and comparable across trials. Reported: relative L2 of (f(X+phi)-f(X)).
    """
    K, cfg, X_test, _Y, ym, ys = _load_split("T6")
    f_in = X_test.shape[-1]
    out_dim = _Y.shape[-1]
    m = _load_model("T6", variant, K, cfg, f_in, out_dim)

    X_take = X_test[:n_samples]
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    errs = []
    with torch.no_grad():
        y0 = m.forward_batch(X_take, K)
        for _ in range(n_trials):
            phi = scale * torch.randn(K.n0, 1, generator=g, device=DEVICE)
            X_shift = X_take + phi  # broadcast over batch and channels
            y1 = m.forward_batch(X_shift, K)
            rel = (y1 - y0).norm() / y0.norm().clamp_min(1e-12)
            errs.append(rel.item())
    del m
    torch.cuda.empty_cache()
    return float(np.mean(errs))


def main():
    out = {"conservation_residual": {}, "gauge_consistency_T6": {}}
    for t in TASKS_TO_MEASURE:
        out["conservation_residual"][t] = {}
        for v in ("full", "learnD"):
            try:
                c = conservation_residual(t, v)
                out["conservation_residual"][t][v] = c
                tag = f"{c:.4e}" if c is not None else "N/A"
                print(f"[{t}/{v}] conservation residual = {tag}")
            except Exception as e:
                out["conservation_residual"][t][v] = None
                print(f"[{t}/{v}] FAILED: {type(e).__name__}: {e}")

    for v in ("full", "noH"):
        try:
            g = gauge_consistency_T6(v)
            out["gauge_consistency_T6"][v] = g
            print(f"[T6/{v}] gauge consistency = {g:.4e}")
        except Exception as e:
            out["gauge_consistency_T6"][v] = None
            print(f"[T6/{v}] FAILED: {type(e).__name__}: {e}")

    op = os.path.join(THIS, "results", "conservation.json")
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {op}")


if __name__ == "__main__":
    main()
