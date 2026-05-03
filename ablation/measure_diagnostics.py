"""
Two diagnostic measurements to strengthen the ablation narrative:

  1. ||d_1 d_0||_F after training the A4 (learnable d_k) ablation.
     The full model has this == 0 exactly (Hodge axiom).
     If learnD drifts far from 0, conservation law is broken -- quantify.

  2. O(C) equivariance error:
     For a random orthogonal Q in R^{C x C}, compute
       err = || Q^T f(Q X) - f(X) ||_F / ||f(X)||_F
     A strictly O(C)-equivariant model has err ~ 1e-7 (numerical noise).
     ReLU gate should break this by several orders of magnitude.
     Compare at the 0-cochain representation (before readout).

Writes:
  ablation/results/diagnostics.json
"""
import os, sys, json, pickle
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

TASKS_TO_MEASURE = ["T1", "T3", "T6", "T7"]


def _load_complex(task_id, device="cuda"):
    cfg = TASKS[task_id]
    with open(os.path.join(ROOT, cfg["dataset"]), "rb") as f:
        d = pickle.load(f)
    pts = torch.tensor(np.array(d["points"], dtype=np.float32))
    faces = torch.tensor(np.array(d["faces"], dtype=np.int64))
    K = CellComplex.from_triangulation(pts, faces).to(device)
    X = torch.tensor(np.array(d["X_data"][:8], dtype=np.float32)).to(device)
    return K, X, cfg


def _build_model(cfg, variant, K, f_in, out_dim, device="cuda"):
    flags = {k: v for k, v in VARIANT_SPEC[variant].items() if k != "desc"}
    m = GaugeHodgeNetwork(
        f_in=f_in, C=cfg["ours_C"], n_layers=4,
        n0=K.n0, n1=K.n1, n2=K.n2,
        task=cfg["task_type"], spatial_dim=cfg["spatial_dim"],
        out_dim=out_dim, mp_hidden=16,
        metric_type=cfg["metric_type"], metric_rank=cfg["metric_rank"],
        **flags,
    ).to(device)
    return m


def measure_dd_norm(task_id):
    """||d_1 d_0||_F for learnD ablation."""
    device = "cuda"
    K, X, cfg = _load_complex(task_id, device)
    with open(os.path.join(ROOT, cfg["dataset"]), "rb") as f:
        d = pickle.load(f)
    Y = np.array(d["Y_data"])
    out_dim = 1 if Y.ndim == 2 else Y.shape[-1]
    # Init model + prime learned_d params by one forward
    m = _build_model(cfg, "learnD", K, X.shape[-1], out_dim, device)
    with torch.no_grad():
        _ = m.forward_batch(X, K)
    # Load trained state
    sd_path = os.path.join(ROOT, "ablation", "checkpoints", "learnD",
                            f"{task_id}_{cfg['name']}", "best_model.pt")
    state = torch.load(sd_path, map_location=device, weights_only=False)
    m.load_state_dict(state)
    # Build sparse d0, d1 from learned values (same routine as network)
    d0_fixed = K.d0.coalesce()
    d1_fixed = K.d1.coalesce()
    d0_dense_fixed = d0_fixed.to_dense()
    d1_dense_fixed = d1_fixed.to_dense()

    d0_learn = torch.sparse_coo_tensor(
        d0_fixed.indices(), m._learned_d0_vals.detach(), size=d0_fixed.shape
    ).coalesce().to_dense()
    d1_learn = torch.sparse_coo_tensor(
        d1_fixed.indices(), m._learned_d1_vals.detach(), size=d1_fixed.shape
    ).coalesce().to_dense()

    prod_fixed = (d1_dense_fixed @ d0_dense_fixed).norm().item()
    prod_learn = (d1_learn @ d0_learn).norm().item()

    # Also relative norm vs magnitude of d1 @ d0-like object (use ||d1||*||d0||)
    scale_fixed = (d1_dense_fixed.norm() * d0_dense_fixed.norm()).item()
    scale_learn = (d1_learn.norm() * d0_learn.norm()).item()

    return {
        "fixed_d1d0_frobenius": prod_fixed,
        "learned_d1d0_frobenius": prod_learn,
        "relative_drift": prod_learn / max(scale_learn, 1e-12),
    }


def measure_oc_equiv(task_id, variant, n_trials=5, seed=0):
    """Test O(C) equivariance of a SINGLE MP layer (bypassing LayerNorm).

    A strictly O(C)-equivariant update f(Qx) = Q f(x) should give error ~machine
    precision (1e-6 ~ 1e-7 in fp32). Norm-gated uses only ||x||^2 and
    <x_i, x_j> which are O(C)-invariant; ReLU is strictly elementwise, not
    equivariant, and should break by orders of magnitude.
    """
    device = "cuda"
    K, X, cfg = _load_complex(task_id, device)
    f_in = X.shape[-1]
    # Build with correct out_dim
    with open(os.path.join(ROOT, cfg["dataset"]), "rb") as f:
        d = pickle.load(f)
    Y = np.array(d["Y_data"])
    out_dim = 1 if Y.ndim == 2 else Y.shape[-1]
    m = _build_model(cfg, variant, K, f_in, out_dim, device)
    with torch.no_grad():
        _ = m.forward_batch(X, K)
    if variant == "full":
        sd_path = os.path.join(ROOT, "checkpoints", f"{task_id}_{cfg['name']}",
                                "ours", "best_model.pt")
    else:
        sd_path = os.path.join(ROOT, "ablation", "checkpoints", variant,
                                f"{task_id}_{cfg['name']}", "best_model.pt")
    m.load_state_dict(torch.load(sd_path, map_location=device, weights_only=False))
    m.eval()

    C = cfg["ours_C"]
    adj_00 = K.get_adjacency(0, 0)
    d0, d1 = m._maybe_learned_d(K)

    # Use the FIRST MP layer only, without LayerNorm, to isolate the effect of
    # the nonlinearity switch on O(C) equivariance.
    layer = m.layers[0]["mp0"]  # 0-cochain MP layer

    g = torch.Generator(device=device).manual_seed(seed)
    errors = []
    with torch.no_grad():
        for _ in range(n_trials):
            A = torch.randn(C, C, generator=g, device=device)
            Q, _ = torch.linalg.qr(A)
            # Lift once
            x0, x1, _ = m.lifting.forward_batch(X, K)
            # Baseline
            y_base = layer.forward_batch(
                x0, d_lower=None, d_upper=d0, adj=adj_00,
                x_lower=None, x_upper=x1)
            # Rotate input; push through same layer
            x0_r = x0 @ Q
            x1_r = x1 @ Q
            y_rot = layer.forward_batch(
                x0_r, d_lower=None, d_upper=d0, adj=adj_00,
                x_lower=None, x_upper=x1_r)
            # Un-rotate; compare with base
            y_unrot = y_rot @ Q.T
            num = (y_unrot - y_base).norm()
            den = y_base.norm().clamp(min=1e-12)
            errors.append((num / den).item())

    return {"mean": float(np.mean(errors)), "max": float(np.max(errors)),
            "trials": n_trials, "C": C}


def main():
    out = {"dd_norm": {}, "oc_equiv": {}}
    for t in TASKS_TO_MEASURE:
        print(f"\n=== {t} ===")
        # 1. ||d1 d0||_F for learnD
        try:
            d = measure_dd_norm(t)
            print(f"[{t}/learnD] fixed ||d1 d0||_F = {d['fixed_d1d0_frobenius']:.3e}")
            print(f"[{t}/learnD] learnd ||d1 d0||_F = {d['learned_d1d0_frobenius']:.3e}  "
                  f"(relative {d['relative_drift']:.3e})")
            out["dd_norm"][t] = d
        except Exception as e:
            print(f"[{t}] dd_norm FAILED: {e}")

        # 2. O(C) equivariance for full vs relu
        for variant in ("full", "relu"):
            try:
                d = measure_oc_equiv(t, variant)
                print(f"[{t}/{variant}] O(C) equiv error mean={d['mean']:.3e}, "
                      f"max={d['max']:.3e}")
                out["oc_equiv"].setdefault(t, {})[variant] = d
            except Exception as e:
                print(f"[{t}/{variant}] oc_equiv FAILED: {e}")

    os.makedirs(os.path.join(ROOT, "ablation", "results"), exist_ok=True)
    with open(os.path.join(ROOT, "ablation", "results", "diagnostics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote ablation/results/diagnostics.json")


if __name__ == "__main__":
    main()
