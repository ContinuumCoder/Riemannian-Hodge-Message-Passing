"""
PDEBench training script.

Supports:
  - Gauge-Hodge MP (ours) and 7 baselines
  - MSE / relative L2 / R² evaluation
  - Vector field divergence error (d1 @ predicted_1cochain)
  - Data efficiency experiments (--data_fraction)
  - Scalability experiments (--grid_size)
  - Checkpoint saving + JSON result logging
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _EXPERIMENT_DIR not in sys.path:
    sys.path.insert(0, _EXPERIMENT_DIR)

from gauge_hodge_mp.cell_complex import CellComplex
from gauge_hodge_mp.network import GaugeHodgeNetwork
from experiments.baselines_graph import GCNBaseline, GATBaseline, SchNetBaseline, EGNNBaseline
from experiments.baselines_topo import MPSNBaseline, SCCNNBaseline, HodgeAwareBaseline
from dataset_pdebench import PDEBenchDataset, split_dataset, pdebench_collate_fn


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(
    name: str,
    f_in: int,
    hidden: int,
    n_layers: int,
    out_dim: int,
    cell_complex: CellComplex,
    task: str = "scalar",
) -> nn.Module:
    """
    Build a model.

    Args:
        name: model name
        f_in: input feature dimension
        hidden: hidden dimension
        n_layers: number of layers
        out_dim: output dimension
        cell_complex: CellComplex (used to get n0/n1/n2)
        task: "scalar" / "vector" / "edge_scalar"

    Returns:
        nn.Module
    """
    n0, n1, n2 = cell_complex.n0, cell_complex.n1, cell_complex.n2

    if name == "ours":
        model = GaugeHodgeNetwork(
            f_in=f_in, C=hidden, n_layers=n_layers,
            n0=n0, n1=n1, n2=n2,
            task=task, spatial_dim=2, out_dim=out_dim,
            mp_hidden=hidden,
        )
    elif name == "gcn":
        bl_task = "node" if task == "scalar" else "edge"
        model = GCNBaseline(f_in=f_in, hidden=hidden, n_layers=n_layers, out_dim=out_dim, task=bl_task)
    elif name == "gat":
        bl_task = "node" if task == "scalar" else "edge"
        model = GATBaseline(f_in=f_in, hidden=hidden, n_layers=n_layers, out_dim=out_dim, task=bl_task)
    elif name == "schnet":
        bl_task = "node" if task == "scalar" else "edge"
        model = SchNetBaseline(f_in=f_in, hidden=hidden, n_layers=n_layers, out_dim=out_dim, task=bl_task)
    elif name == "egnn":
        bl_task = "node" if task == "scalar" else "edge"
        model = EGNNBaseline(f_in=f_in, hidden=hidden, n_layers=n_layers, out_dim=out_dim, task=bl_task)
    elif name == "mpsn":
        model = MPSNBaseline(f_in=f_in, hidden=hidden, n_layers=n_layers)
    elif name == "sccnn":
        model = SCCNNBaseline(f_in=f_in, hidden=hidden, n_layers=n_layers)
    elif name == "hodge_aware":
        model = HodgeAwareBaseline(f_in=f_in, hidden=hidden, n_layers=n_layers)
    else:
        raise ValueError(
            f"Unknown model: {name}. "
            f"Choose from: ours, gcn, gat, schnet, egnn, mpsn, sccnn, hodge_aware"
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model [{name}]: {n_params:,} parameters")
    return model


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict:
    """
    Compute regression evaluation metrics.

    Args:
        pred: (B, N, D) or (N, D)
        target: same shape as pred

    Returns:
        dict: mse, rel_l2, r2
    """
    # Flatten batch dim if present
    pred_flat = pred.reshape(-1, pred.size(-1))
    tgt_flat = target.reshape(-1, target.size(-1))

    # MSE
    mse = F.mse_loss(pred_flat, tgt_flat).item()

    # Relative L2 error: ||pred - target||₂ / ||target||₂
    diff_norm = torch.norm(pred_flat - tgt_flat).item()
    tgt_norm = torch.norm(tgt_flat).item()
    rel_l2 = diff_norm / max(tgt_norm, 1e-8)

    # R² = 1 - SS_res / SS_tot
    ss_res = ((pred_flat - tgt_flat) ** 2).sum().item()
    ss_tot = ((tgt_flat - tgt_flat.mean(dim=0, keepdim=True)) ** 2).sum().item()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)

    return {"mse": mse, "rel_l2": rel_l2, "r2": r2}


def compute_divergence_error(
    pred_edge: torch.Tensor,
    d1: torch.Tensor,
) -> float:
    """
    Compute divergence error of predicted 1-cochain: ||d1 @ pred||_2 / ||pred||_2

    For physically divergence-free fields, this value should be close to 0.
    d1 @ 1-cochain gives the "circulation" per face (discrete curl/divergence).

    Args:
        pred_edge: (n1, 1) or (n1,) predicted edge values (1-cochain)
        d1: (n2, n1) sparse coboundary operator

    Returns:
        Relative divergence error (scalar)
    """
    if pred_edge.dim() == 2:
        pred_edge = pred_edge.squeeze(-1)
    # d1 @ pred -> (n2,) per-face divergence
    div = torch.sparse.mm(d1, pred_edge.unsqueeze(-1)).squeeze(-1)  # (n2,)
    div_norm = torch.norm(div).item()
    pred_norm = torch.norm(pred_edge).item()
    return div_norm / max(pred_norm, 1e-8)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    cell_complex: CellComplex,
    is_edge_model: bool = False,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    K = cell_complex.to(device)

    for batch_feat, _, batch_tgt in loader:
        B = batch_feat.size(0)
        batch_feat = batch_feat.to(device)  # (B, n_nodes, f_in)
        batch_tgt = batch_tgt.to(device)    # (B, n_nodes, out_dim)

        optimizer.zero_grad()

        # Batched forward: exploit linearity of spmm with shared CellComplex
        has_batch = hasattr(model, 'forward_batch')
        if has_batch:
            pred_stack = model.forward_batch(batch_feat, K)
        else:
            preds = [model(batch_feat[i], K) for i in range(B)]
            pred_stack = torch.stack(preds, dim=0)

        if is_edge_model:
            src, dst = K.edges[:, 0], K.edges[:, 1]
            edge_tgt = 0.5 * (batch_tgt[:, src, :] + batch_tgt[:, dst, :])
            if edge_tgt.size(-1) > 1:
                edge_tgt = edge_tgt[..., :1]
            loss = F.mse_loss(pred_stack, edge_tgt)
        else:
            loss = F.mse_loss(pred_stack, batch_tgt)

        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * B
        n_samples += B

    return {"train_loss": total_loss / max(n_samples, 1)}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    cell_complex: CellComplex,
    is_edge_model: bool = False,
    compute_div: bool = False,
) -> dict:
    """Evaluate the model."""
    model.eval()
    K = cell_complex.to(device)

    all_preds = []
    all_targets = []
    div_errors = []

    for batch_feat, _, batch_tgt in loader:
        B = batch_feat.size(0)
        batch_feat = batch_feat.to(device)
        batch_tgt = batch_tgt.to(device)

        has_batch = hasattr(model, 'forward_batch')
        if has_batch:
            pred_batch = model.forward_batch(batch_feat, K)
            if is_edge_model:
                src, dst = K.edges[:, 0], K.edges[:, 1]
                edge_tgt_batch = 0.5 * (batch_tgt[:, src, :] + batch_tgt[:, dst, :])
                if edge_tgt_batch.size(-1) > 1:
                    edge_tgt_batch = edge_tgt_batch[..., :1]
                for i in range(B):
                    all_preds.append(pred_batch[i].cpu())
                    all_targets.append(edge_tgt_batch[i].cpu())
            else:
                for i in range(B):
                    all_preds.append(pred_batch[i].cpu())
                    all_targets.append(batch_tgt[i].cpu())
        else:
            for i in range(B):
                pred = model(batch_feat[i], K)
                if is_edge_model:
                    src, dst = K.edges[:, 0], K.edges[:, 1]
                    edge_tgt = 0.5 * (batch_tgt[i, src, :] + batch_tgt[i, dst, :])
                    if edge_tgt.size(-1) > 1:
                        edge_tgt = edge_tgt[..., :1]
                    all_preds.append(pred.cpu())
                    all_targets.append(edge_tgt.cpu())
                    if compute_div:
                        div_err = compute_divergence_error(pred, K.d1)
                        div_errors.append(div_err)
                else:
                    all_preds.append(pred.cpu())
                    all_targets.append(batch_tgt[i].cpu())

    pred_cat = torch.stack(all_preds, dim=0)    # (N, nodes_or_edges, dim)
    tgt_cat = torch.stack(all_targets, dim=0)

    metrics = compute_metrics(pred_cat, tgt_cat)

    if compute_div and div_errors:
        metrics["div_error_mean"] = float(np.mean(div_errors))
        metrics["div_error_std"] = float(np.std(div_errors))

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train PDEBench experiments for Gauge-Hodge MP"
    )
    # Data
    parser.add_argument("--dataset", type=str, default="swe",
                        choices=["swe", "cns_rand", "cns_turb"])
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(_PROJECT_ROOT, "data", "pdebench"))
    parser.add_argument("--grid_size", type=int, default=32,
                        help="Grid resolution for scalability experiments")
    parser.add_argument("--data_fraction", type=float, default=1.0,
                        help="Fraction of training data to use (data efficiency)")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Max number of trajectories to load")

    # Model
    parser.add_argument("--model", type=str, default="ours",
                        choices=["ours", "gcn", "gat", "schnet", "egnn",
                                 "mpsn", "sccnn", "hodge_aware"])
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (0 = disabled)")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "step", "none"])

    # System
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--save_dir", type=str,
                        default=os.path.join(_PROJECT_ROOT, "results", "pdebench"))
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pt file to resume training")
    parser.add_argument("--ckpt_every", type=int, default=50,
                        help="Save periodic checkpoint every N epochs (0=off)")

    args = parser.parse_args()

    # --- Device ---
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # --- Random seed ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- Experiment identifier ---
    exp_name = (
        f"{args.dataset}_{args.model}_h{args.hidden}_l{args.layers}"
        f"_g{args.grid_size}_df{args.data_fraction}_s{args.seed}"
    )
    save_dir = os.path.join(args.save_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Experiment: {exp_name}")
    print(f"Save dir:   {save_dir}")

    # --- Data ---
    print("\n--- Loading Dataset ---")
    dataset = PDEBenchDataset(
        data_dir=args.data_dir,
        dataset=args.dataset,
        grid_size=args.grid_size,
        n_samples=args.n_samples,
        device="cpu",
    )

    train_set, val_set, test_set = split_dataset(
        dataset,
        data_fraction=args.data_fraction,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=pdebench_collate_fn,
        pin_memory=(device != "cpu"),
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=pdebench_collate_fn,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=pdebench_collate_fn,
    )

    cell_complex = dataset.get_cell_complex()
    f_in = dataset.f_in
    out_dim = dataset.out_dim

    # --- Determine model type ---
    # edge-level baselines: mpsn, sccnn, hodge_aware output (n1, 1)
    # node-level models: ours, gcn, gat, schnet, egnn output (n0, out_dim)
    edge_models = {"mpsn", "sccnn", "hodge_aware"}
    is_edge_model = args.model in edge_models

    # For ours, task is determined by the dataset
    task = "scalar"

    # --- Model ---
    print("\n--- Building Model ---")
    model = build_model(
        name=args.model,
        f_in=f_in,
        hidden=args.hidden,
        n_layers=args.layers,
        out_dim=out_dim,
        cell_complex=cell_complex,
        task=task,
    )
    model = model.to(device)

    # --- Optimizer ---
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, args.epochs // 3), gamma=0.5
        )
    else:
        scheduler = None

    # --- Resume from checkpoint ---
    start_epoch = 1
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_metrics", {}).get("mse", float("inf"))
        best_epoch = ckpt.get("epoch", 0)
        print(f"  Resumed at epoch {start_epoch}, best_val_mse={best_val_loss:.6f}")
    elif args.resume:
        print(f"WARNING: --resume path not found: {args.resume}")

    # --- Training ---
    print("\n--- Training ---")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            cell_complex, is_edge_model=is_edge_model,
        )

        # Validate
        val_metrics = evaluate(
            model, val_loader, device, cell_complex,
            is_edge_model=is_edge_model, compute_div=is_edge_model,
        )

        if scheduler is not None:
            scheduler.step()

        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        # Logging
        log_str = (
            f"Epoch {epoch:3d}/{args.epochs} "
            f"| train_loss={train_metrics['train_loss']:.6f} "
            f"| val_mse={val_metrics['mse']:.6f} "
            f"| val_rel_l2={val_metrics['rel_l2']:.4f} "
            f"| val_r2={val_metrics['r2']:.4f} "
            f"| lr={lr_now:.2e} "
            f"| {dt:.1f}s"
        )
        if "div_error_mean" in val_metrics:
            log_str += f" | div_err={val_metrics['div_error_mean']:.4f}"
        print(log_str)

        record = {
            "epoch": epoch,
            "lr": lr_now,
            "time": dt,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)

        # Best model checkpoint
        val_loss = val_metrics["mse"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            ckpt_path = os.path.join(save_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "args": vars(args),
            }, ckpt_path)
        else:
            patience_counter += 1

        # Periodic checkpoint
        if args.ckpt_every > 0 and epoch % args.ckpt_every == 0:
            periodic_path = os.path.join(save_dir, f"checkpoint_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "args": vars(args),
            }, periodic_path)
            print(f"  [Checkpoint] {periodic_path}")

        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(best epoch: {best_epoch}, best val_mse: {best_val_loss:.6f})")
            break

    # --- Testing ---
    print("\n--- Testing ---")

    # Load best model
    ckpt_path = os.path.join(save_dir, "best_model.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded best model from epoch {ckpt['epoch']}")

    test_metrics = evaluate(
        model, test_loader, device, cell_complex,
        is_edge_model=is_edge_model, compute_div=True,
    )

    print(f"\n  Test Results:")
    print(f"    MSE:       {test_metrics['mse']:.6f}")
    print(f"    Rel L2:    {test_metrics['rel_l2']:.4f}")
    print(f"    R²:        {test_metrics['r2']:.4f}")
    if "div_error_mean" in test_metrics:
        print(f"    Div Error: {test_metrics['div_error_mean']:.4f} "
              f"(+/- {test_metrics['div_error_std']:.4f})")

    # --- Save results ---
    results = {
        "experiment": exp_name,
        "args": vars(args),
        "best_epoch": best_epoch,
        "best_val_mse": best_val_loss,
        "test_metrics": test_metrics,
        "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "cell_complex_info": {
            "n0": cell_complex.n0,
            "n1": cell_complex.n1,
            "n2": cell_complex.n2,
        },
        "history": history,
    }

    results_path = os.path.join(save_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")
    print(f"  Checkpoint saved to: {ckpt_path}")

    # Save final checkpoint
    final_ckpt_path = os.path.join(save_dir, "final_model.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "test_metrics": test_metrics,
        "args": vars(args),
    }, final_ckpt_path)

    return results


if __name__ == "__main__":
    main()
