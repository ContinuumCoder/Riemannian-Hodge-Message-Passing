"""
Train a single (variant, task) ablation run.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 ablation/run_ablation.py T1 noH
  CUDA_VISIBLE_DEVICES=1 python3 ablation/run_ablation.py T6 learnD

Saves:
  ablation/checkpoints/{variant}/{task_id}_{name}/best_model.pt, history.json, result.json, checkpoint_ep*.pt
  ablation/outputs/{task_id}/{variant}_test.pkl   -- predictions + targets on held-out split
  ablation/logs/{task_id}_{variant}.log            -- stdout (caller redirects)
"""
import sys, os, argparse, json, time, pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from gauge_hodge_mp.cell_complex import CellComplex
from gauge_hodge_mp.network import GaugeHodgeNetwork
from ablation.variants import VARIANT_SPEC

# Reuse task configs from the main benchmark
sys.path.insert(0, os.path.join(ROOT, 'experiments'))
from formal_benchmark import TASKS, DS, collate_fn, compute_metrics, compute_ssim_pearson


def train_one(variant: str, task_id: str, n_epochs: int = 50, seed: int = 42):
    assert variant in VARIANT_SPEC, f"unknown variant {variant}"
    flags = {k: v for k, v in VARIANT_SPEC[variant].items() if k != "desc"}
    cfg = TASKS[task_id]
    device = "cuda"

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    save_dir = os.path.join(ROOT, "ablation", "checkpoints", variant,
                            f"{task_id}_{cfg['name']}")
    out_dir = os.path.join(ROOT, "ablation", "outputs", task_id)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{variant:8s} / {task_id}] {VARIANT_SPEC[variant]['desc']}")
    print(f"  save_dir = {save_dir}")

    # --- Load dataset ---
    with open(os.path.join(ROOT, cfg["dataset"]), "rb") as f:
        d = pickle.load(f)
    pts = torch.tensor(np.array(d["points"], dtype=np.float32))
    faces = torch.tensor(np.array(d["faces"], dtype=np.int64))
    K = CellComplex.from_triangulation(pts, faces)

    X = torch.tensor(np.array(d["X_data"], dtype=np.float32))
    Y = torch.tensor(np.array(d["Y_data"], dtype=np.float32))
    if X.ndim == 2: X = X.unsqueeze(-1)
    if Y.ndim == 2: Y = Y.unsqueeze(-1)
    f_in = X.shape[-1]; out_dim = Y.shape[-1]

    n = len(X); nt = int(0.7 * n); nv = int(0.15 * n)
    xm, xs = X[:nt].mean((0, 1)), X[:nt].std((0, 1)).clamp(1e-6)
    ym, ys = Y[:nt].mean((0, 1)), Y[:nt].std((0, 1)).clamp(1e-6)
    Xn = (X - xm) / xs
    Yn = (Y - ym) / ys

    train_loader = DataLoader(DS(Xn[:nt], Yn[:nt]), batch_size=cfg["batch_size"],
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(DS(Xn[nt:nt + nv], Yn[nt:nt + nv]), batch_size=cfg["batch_size"],
                            shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(DS(Xn[nt + nv:], Yn[nt + nv:]), batch_size=cfg["batch_size"],
                             shuffle=False, collate_fn=collate_fn)
    K_dev = K.to(device)

    # --- Build model ---
    model = GaugeHodgeNetwork(
        f_in=f_in, C=cfg["ours_C"], n_layers=4,
        n0=K.n0, n1=K.n1, n2=K.n2,
        task=cfg["task_type"], spatial_dim=cfg["spatial_dim"],
        out_dim=out_dim, mp_hidden=16,
        metric_type=cfg["metric_type"], metric_rank=cfg["metric_rank"],
        **flags,
    ).to(device)
    # learned_d parameters are registered lazily on first forward; do a dummy pass
    with torch.no_grad():
        _ = model.forward_batch(Xn[:1].to(device), K_dev)

    np_M = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  params = {np_M:.3f}M")

    # --- Optim ---
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)

    history = []
    best_r2 = -1e9; best_ep = 0; start_ep = 1

    # Resume
    ckpts = sorted([f for f in os.listdir(save_dir)
                    if f.startswith("checkpoint_ep") and f.endswith(".pt")])
    if ckpts:
        latest = ckpts[-1]
        resume_ep = int(latest.replace("checkpoint_ep", "").replace(".pt", ""))
        if resume_ep < n_epochs:
            model.load_state_dict(torch.load(os.path.join(save_dir, latest),
                                             map_location=device))
            start_ep = resume_ep + 1
            for _ in range(resume_ep):
                sched.step()
            hp = os.path.join(save_dir, "history.json")
            if os.path.exists(hp):
                history = json.load(open(hp))
                for h in history:
                    if h["R2"] > best_r2:
                        best_r2 = h["R2"]; best_ep = h["epoch"]
            print(f"  resumed from ep{resume_ep}, best R2={best_r2:.4f}")

    result_path = os.path.join(save_dir, "result.json")
    if os.path.exists(result_path):
        prev = json.load(open(result_path))
        if prev.get("best_epoch", 0) > 0 and prev["best_epoch"] >= start_ep - 1:
            # already done, skip training, but still regenerate test output pickle if missing
            if not os.path.exists(os.path.join(out_dir, f"{variant}_test.pkl")):
                model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt"),
                                                 map_location=device))
                _dump_test_outputs(model, test_loader, K_dev, ym, ys, out_dir, variant)
            print(f"  already completed: best_R2={prev['best_R2']:.4f}@ep{prev['best_epoch']}")
            return prev

    for ep in range(start_ep, n_epochs + 1):
        model.train(); t0 = time.time(); train_loss = 0; nb = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            pred = model.forward_batch(bx, K_dev)
            loss = F.mse_loss(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item(); nb += 1
        sched.step(); train_loss /= max(nb, 1); dt = time.time() - t0

        model.eval(); all_p, all_t = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                all_p.append(model.forward_batch(bx, K_dev).cpu())
                all_t.append(by.cpu())
        all_p = torch.cat(all_p); all_t = torch.cat(all_t)
        m = compute_metrics(all_p, all_t)
        m.update({"train_loss": train_loss, "epoch": ep, "time": dt})
        history.append(m)

        if m["R2"] > best_r2:
            best_r2 = m["R2"]; best_ep = ep
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

        if ep % 10 == 0 or ep == 1 or ep == n_epochs:
            torch.save(model.state_dict(),
                       os.path.join(save_dir, f"checkpoint_ep{ep}.pt"))
            with open(os.path.join(save_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=2)
            print(f"  ep{ep:3d}: R2={m['R2']:.4f} loss={train_loss:.5f} ({dt:.1f}s)",
                  flush=True)

    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Test with best ckpt
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt"),
                                     map_location=device, weights_only=False))
    model.eval()
    test_pred_n, test_tgt_n = [], []
    test_pred_r, test_tgt_r = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            p = model.forward_batch(bx, K_dev)
            test_pred_n.append(p.cpu()); test_tgt_n.append(by.cpu())
            test_pred_r.append((p.cpu() * ys + ym))
            test_tgt_r.append((by.cpu() * ys + ym))
    test_pred_n = torch.cat(test_pred_n); test_tgt_n = torch.cat(test_tgt_n)
    tm = compute_metrics(test_pred_n, test_tgt_n)
    test_pred_r = torch.cat(test_pred_r).numpy()
    test_tgt_r = torch.cat(test_tgt_r).numpy()
    ssim, pearson = compute_ssim_pearson(test_pred_r, test_tgt_r)

    # NRMSE = RMSE / range(target)
    rmse = float(np.sqrt(((test_pred_r - test_tgt_r) ** 2).mean()))
    data_range = float(test_tgt_r.max() - test_tgt_r.min())
    nrmse = rmse / max(data_range, 1e-8)

    result = {
        "variant": variant, "task": task_id, "desc": VARIANT_SPEC[variant]["desc"],
        "params_M": np_M,
        "best_R2": best_r2, "best_epoch": best_ep,
        "final_R2": history[-1]["R2"], "final_MSE": history[-1]["MSE"],
        "final_MAE": history[-1]["MAE"],
        "test_R2": tm["R2"], "test_MSE": tm["MSE"],
        "test_SSIM": ssim, "test_Pearson": pearson, "test_NRMSE": nrmse,
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    # Save test outputs for visualization
    with open(os.path.join(out_dir, f"{variant}_test.pkl"), "wb") as f:
        pickle.dump({"pred": test_pred_r, "target": test_tgt_r,
                     "variant": variant, "task": task_id}, f)

    print(f"[{variant}/{task_id}] DONE  R2={best_r2:.4f}@ep{best_ep}  "
          f"test_R2={tm['R2']:.4f}  SSIM={ssim:.4f}  NRMSE={nrmse:.4f}")
    return result


def _dump_test_outputs(model, test_loader, K_dev, ym, ys, out_dir, variant):
    """Regenerate test_pred/target pickle from an already-trained best_model."""
    device = "cuda"
    model.eval()
    ps, ts = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            p = model.forward_batch(bx, K_dev)
            ps.append((p.cpu() * ys + ym))
            ts.append((by.cpu() * ys + ym))
    import pickle as _pkl
    with open(os.path.join(out_dir, f"{variant}_test.pkl"), "wb") as f:
        _pkl.dump({"pred": torch.cat(ps).numpy(), "target": torch.cat(ts).numpy(),
                   "variant": variant}, f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task_id")
    ap.add_argument("variant")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train_one(args.variant, args.task_id, n_epochs=args.epochs, seed=args.seed)
