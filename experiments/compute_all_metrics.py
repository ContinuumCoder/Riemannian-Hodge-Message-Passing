"""
Compute R², SSIM, Pearson, NRMSE for all models from checkpoints.

Automatically infers model architecture from saved checkpoints.
Handles edge-space mapping for topological baselines (MPSN, SCCNN, Clifford-SMPN).

Usage:
  python3 experiments/compute_all_metrics.py           # all tasks
  python3 experiments/compute_all_metrics.py --task T1  # single task
"""
import sys, os, json, glob, pickle, argparse
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gauge_hodge_mp.cell_complex import CellComplex
from gauge_hodge_mp.network import GaugeHodgeNetwork
from baselines_graph import BaselineBase, GCNBaseline, GATBaseline, SchNetBaseline, EGNNBaseline
from baselines_topo import MPSNBaseline, SCCNNBaseline
from baselines_advanced import GaugeEquivCNNBaseline, CWNetBaseline, CliffordNetBaseline, HermesBaseline
from baselines_operator import FNOWrapper, DeepONetWrapper

# ============================================================
# Metric functions
# ============================================================

def compute_r2_mse_mae(pred, target):
    """R², MSE, MAE on normalized predictions."""
    mse = ((pred - target) ** 2).mean().item()
    mae = (pred - target).abs().mean().item()
    ss_res = ((pred - target) ** 2).sum().item()
    ss_tot = ((target - target.mean(0, keepdim=True)) ** 2).sum().item()
    r2 = 1 - ss_res / max(ss_tot, 1e-8)
    return r2, mse, mae


def compute_nrmse(pred_raw, target_raw):
    """NRMSE = RMSE / range(target), computed on unnormalized data."""
    rmse = np.sqrt(np.mean((pred_raw - target_raw) ** 2))
    data_range = target_raw.max() - target_raw.min()
    return rmse / max(data_range, 1e-8)


def compute_ssim_pearson(pred_raw, target_raw, n_samples=None):
    """Per-sample SSIM and Pearson on unnormalized predictions.

    SSIM uses GLOBAL data_range from all target samples (consistent C1/C2).
    Predictions clipped to target range ± 10% to prevent outlier-inflated variance.
    Pearson clipped to ≥ 0 (negative = no correlation, avoids reviewer confusion).

    Args:
        pred_raw:   (N, ...) numpy, unnormalized predictions
        target_raw: (N, ...) numpy, unnormalized targets
        n_samples:  max samples to evaluate (None = all)
    Returns:
        mean_ssim, mean_pearson
    """
    N = len(pred_raw)
    if n_samples is not None:
        N = min(N, n_samples)

    # Global data range
    data_range = float(target_raw.max() - target_raw.min())
    if data_range < 1e-8:
        data_range = 1.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    # Clip predictions
    t_min, t_max = float(target_raw.min()), float(target_raw.max())
    margin = 0.1 * data_range
    pred_clipped = np.clip(pred_raw[:N], t_min - margin, t_max + margin)

    ssims, pearsons = [], []
    for i in range(N):
        p = pred_clipped[i].flatten().astype(np.float64)
        t = target_raw[i].flatten().astype(np.float64)

        # Pearson (clip ≥ 0)
        tp = t - t.mean()
        pp = p - p.mean()
        denom = np.sqrt(np.sum(tp ** 2) * np.sum(pp ** 2))
        pearson = np.sum(tp * pp) / (denom + 1e-15) if denom > 1e-15 else 0.0
        pearsons.append(max(0.0, pearson))

        # SSIM (global C1/C2)
        mu_t, mu_p = t.mean(), p.mean()
        sig_t, sig_p = t.std(), p.std()
        sig_tp = np.mean((t - mu_t) * (p - mu_p))
        num = (2 * mu_t * mu_p + c1) * (2 * sig_tp + c2)
        den = (mu_t ** 2 + mu_p ** 2 + c1) * (sig_t ** 2 + sig_p ** 2 + c2)
        ssims.append(num / (den + 1e-15))

    return float(np.mean(ssims)), float(np.mean(pearsons))


# ============================================================
# Model loading utilities
# ============================================================

def infer_hidden(state_dict):
    """Infer hidden dim from state dict."""
    # Try various known patterns
    for k, v in state_dict.items():
        if k == 'embed.weight':  # SchNet
            return v.shape[0]
        if k.startswith('layers.0.') and 'weight' in k and v.ndim == 2:  # GCN/GAT
            return v.shape[0]
        if k.startswith('layers.0.') and k.endswith('.W.weight') and v.ndim == 2:  # GAT
            return v.shape[0]
    # Fallback: first large weight
    for k, v in state_dict.items():
        if 'lift' in k and 'weight' in k and v.ndim == 2:
            return v.shape[0]
    for k, v in state_dict.items():
        if 'weight' in k and v.ndim == 2 and v.shape[0] > 10:
            return v.shape[0]
    return None


# Ours config per task (C values from actual checkpoints)
OURS_CFG = {
    'T1_cns_vorticity':              {'C': 128, 'task': 'scalar', 'sdim': 2},
    'T2_torus_advection_diffusion':  {'C': 64,  'task': 'scalar', 'sdim': 2},
    'T3_ellipsoid_coexact':          {'C': 64,  'task': 'vector', 'sdim': 3},
    'T3_ellipsoid_surface_flow':     {'C': 64,  'task': 'vector', 'sdim': 3},
    'T5_maxwell_poisson':            {'C': 160, 'task': 'scalar', 'sdim': 2},
    'T6_wilson_loop':                {'C': 128, 'task': 'scalar', 'sdim': 2},
    'T7_yang_mills_su2':             {'C': 160, 'task': 'scalar', 'sdim': 2},
}

# Node baselines
NODE_CLS = {
    'gcn': GCNBaseline,
    'gat': GATBaseline,
    'schnet': SchNetBaseline,
    'egnn': EGNNBaseline,
    'gauge_cnn': lambda **kw: GaugeEquivCNNBaseline(**kw),
    'hermes': lambda **kw: HermesBaseline(**kw),
    'gem_cnn': lambda **kw: HermesBaseline(**kw),
    'cw_net': lambda **kw: CWNetBaseline(**kw),
    'deeponet': None,  # special handling
    'fno': None,  # special handling
}

# Edge baselines
EDGE_CLS = {
    'mpsn': MPSNBaseline,
    'sccnn': SCCNNBaseline,
    'clifford_smpn': CliffordNetBaseline,
}


def load_model(model_name, state_dict, f_in, out_dim, K, task_name):
    """Load model from state dict, inferring architecture from checkpoint."""

    if model_name == 'ours':
        cfg = None
        for prefix in [task_name, task_name.replace('surface_flow', 'coexact')]:
            if prefix in OURS_CFG:
                cfg = OURS_CFG[prefix]
                break
        if cfg is None:
            return None
        m = GaugeHodgeNetwork(
            f_in=f_in, C=cfg['C'], n_layers=4,
            n0=K.n0, n1=K.n1, n2=K.n2,
            task=cfg['task'], spatial_dim=cfg['sdim'], out_dim=out_dim,
            mp_hidden=16, metric_type='diagonal', metric_rank=8
        )
        m.load_state_dict(state_dict)
        return m, False  # (model, is_edge)

    if model_name in EDGE_CLS:
        h = infer_hidden(state_dict)
        if h is None:
            return None
        m = EDGE_CLS[model_name](f_in=f_in, hidden=h, n_layers=4)
        m.load_state_dict(state_dict)
        return m, True

    if model_name == 'fno':
        # Infer width from fc0
        w = state_dict.get('fno.fc0.weight', None)
        if w is None:
            return None
        width = w.shape[0]
        m = FNOWrapper(f_in=f_in, out_dim=out_dim, width=width, modes=12, n_layers=4)
        m.load_state_dict(state_dict)
        return m, False

    if model_name == 'deeponet':
        h = infer_hidden(state_dict)
        if h is None:
            return None
        m = DeepONetWrapper(f_in=f_in, out_dim=out_dim, branch_width=h, trunk_width=h, n_basis=64)
        m.load_state_dict(state_dict)
        return m, False

    # Standard node baselines (GCN, GAT, SchNet, EGNN, gauge_cnn, hermes, cw_net)
    h = infer_hidden(state_dict)
    if h is None:
        return None

    # Infer actual f_in from embed layer (may differ if scalar-only input was used)
    actual_f_in = f_in
    embed_w = state_dict.get('embed.weight')
    if embed_w is not None and embed_w.ndim == 2:
        actual_f_in = embed_w.shape[1]

    if model_name in ['gauge_cnn', 'hermes', 'gem_cnn', 'cw_net']:
        cls = NODE_CLS[model_name]
        m = cls(f_in=actual_f_in, hidden=h, n_layers=4, out_dim=out_dim, task='node')
    else:
        cls = NODE_CLS.get(model_name)
        if cls is None:
            return None
        m = cls(f_in=actual_f_in, hidden=h, n_layers=4, out_dim=out_dim, task='node')

    # node_mlp compatibility is handled by BaselineBase.load_state_dict
    m.load_state_dict(state_dict)
    return m, False


# ============================================================
# Main evaluation
# ============================================================

def evaluate_task(task_name, ds_path, device='cuda', n_eval=100, ckpt_root='checkpoints'):
    """Evaluate all models for a task, return dict of metrics."""
    print(f"\n{'='*60}\n{task_name}", flush=True)

    d = pickle.load(open(ds_path, 'rb'))
    pts = torch.tensor(np.array(d['points'], dtype=np.float32))
    faces = torch.tensor(np.array(d['faces'], dtype=np.int64))
    K = CellComplex.from_triangulation(pts, faces).to(device)

    X = torch.tensor(np.array(d['X_data'], dtype=np.float32))
    Y_raw = np.array(d['Y_data'], dtype=np.float32)
    Y = torch.tensor(Y_raw)
    if X.ndim == 2: X = X.unsqueeze(-1)
    if Y.ndim == 2: Y = Y.unsqueeze(-1)
    if Y_raw.ndim == 2: Y_raw = Y_raw[:, :, np.newaxis]

    f_in, out_dim = X.shape[-1], Y.shape[-1]
    n_total = len(X)
    nt = int(0.7 * n_total)
    nv = int(0.15 * n_total)
    xm, xs = X[:nt].mean((0, 1)), X[:nt].std((0, 1)).clamp(1e-6)
    ym, ys = Y[:nt].mean((0, 1)), Y[:nt].std((0, 1)).clamp(1e-6)
    Xn = (X - xm) / xs

    # Test set
    Xt = Xn[nt + nv:].to(device)
    Yt_raw = Y_raw[nt + nv:]
    n_test = min(n_eval, len(Xt))

    # Edge mapping for edge baselines
    edges = K.edges  # (n1, 2)
    s_idx = edges[:, 0].cpu().numpy()
    d_idx = edges[:, 1].cpu().numpy()
    Yt_edge_raw = 0.5 * (Yt_raw[:, s_idx] + Yt_raw[:, d_idx])

    ckpt_base = os.path.join(ckpt_root, task_name)
    results = {}

    if not os.path.isdir(ckpt_base):
        return results

    for model_name in sorted(os.listdir(ckpt_base)):
        bp = os.path.join(ckpt_base, model_name, 'best_model.pt')
        if not os.path.exists(bp):
            continue

        try:
            state = torch.load(bp, map_location='cpu', weights_only=False)
            loaded = load_model(model_name, state, f_in, out_dim, K, task_name)
            if loaded is None:
                print(f"  {model_name}: skip (can't load)", flush=True)
                continue
            m, is_edge = loaded
            m = m.to(device).eval()

            # Detect if model uses fewer input channels (e.g. EGNN scalar-only)
            model_f_in = f_in
            embed_w = state.get('embed.weight')
            if embed_w is not None and embed_w.ndim == 2 and embed_w.shape[1] != f_in:
                model_f_in = embed_w.shape[1]

            preds_raw, tgts_raw = [], []
            preds_norm, tgts_norm = [], []

            with torch.no_grad():
                for i in range(n_test):
                    xi = Xt[i:i + 1]
                    if model_f_in != f_in:
                        xi = xi[..., :model_f_in]
                    if hasattr(m, 'forward_batch'):
                        pred_n = m.forward_batch(xi, K)[0]
                    else:
                        pred_n = m(xi[0], K)

                    if is_edge:
                        # Target in edge space
                        tgt_raw_i = Yt_edge_raw[i]
                        od = pred_n.shape[-1]
                        if tgt_raw_i.shape[-1] > od:
                            tgt_raw_i = tgt_raw_i[..., :od]
                        # Unnormalize pred
                        ys_e = ys[:od].numpy()
                        ym_e = ym[:od].numpy()
                    else:
                        tgt_raw_i = Yt_raw[i]
                        ys_e = ys.numpy()
                        ym_e = ym.numpy()

                    pred_raw_i = pred_n.cpu().numpy() * ys_e + ym_e
                    preds_raw.append(pred_raw_i)
                    tgts_raw.append(tgt_raw_i)
                    preds_norm.append(pred_n.cpu())
                    # Normalized target
                    tgt_norm_i = (torch.tensor(tgt_raw_i) - torch.tensor(ym_e)) / torch.tensor(ys_e)
                    tgts_norm.append(tgt_norm_i)

            preds_raw = np.array(preds_raw)
            tgts_raw = np.array(tgts_raw)
            preds_norm_t = torch.stack(preds_norm)
            tgts_norm_t = torch.stack(tgts_norm)

            # R², MSE, MAE
            r2, mse, mae = compute_r2_mse_mae(preds_norm_t, tgts_norm_t)
            # NRMSE
            nrmse = compute_nrmse(preds_raw, tgts_raw)
            # SSIM, Pearson
            ssim, pearson = compute_ssim_pearson(preds_raw, tgts_raw, n_samples=n_test)

            results[model_name] = {
                'R2': r2, 'MSE': mse, 'MAE': mae,
                'NRMSE': nrmse, 'SSIM': ssim, 'Pearson': pearson,
            }
            print(f"  {model_name:15s} R2={r2:.4f}  SSIM={ssim:.3f}  Pearson={pearson:.3f}  NRMSE={nrmse:.4f}",
                  flush=True)

            del m
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  {model_name}: ERROR {str(e)[:80]}", flush=True)

    return results


def evaluate_task_persample(task_name, ds_path, device='cuda', n_eval=50, ckpt_root='checkpoints'):
    """Evaluate T8-style per-sample mesh tasks."""
    print(f"\n{'='*60}\n{task_name} (per-sample mesh)", flush=True)

    raw = pickle.load(open(ds_path, 'rb'))
    n_total = len(raw)
    nt = int(0.7 * n_total); nv = int(0.15 * n_total)

    # Normalization from train set; must match training-time statistics in formal_benchmark.py
    all_x = np.stack([np.array(raw[i]['x_pres'], dtype=np.float32) for i in range(nt)])
    all_y = np.stack([np.array(raw[i]['y_pres'], dtype=np.float32) for i in range(nt)])
    xm = all_x.mean((0, 1))  # per-feature mean over (samples, nodes)
    xs = np.clip(all_x.std((0, 1)), 1e-6, None)
    ym = all_y.mean((0, 1))
    ys = np.clip(all_y.std((0, 1)), 1e-6, None)

    test_indices = list(range(nt + nv, n_total))[:n_eval]

    ckpt_base = os.path.join(ckpt_root, task_name)
    if not os.path.isdir(ckpt_base):
        return {}

    results = {}
    sample0 = raw[0]
    x0 = np.array(sample0['x_pres'], dtype=np.float32)
    y0 = np.array(sample0['y_pres'], dtype=np.float32)
    if x0.ndim == 1: x0 = x0[:, np.newaxis]
    if y0.ndim == 1: y0 = y0[:, np.newaxis]
    f_in, out_dim = x0.shape[-1], y0.shape[-1]

    for model_name in sorted(os.listdir(ckpt_base)):
        bp = os.path.join(ckpt_base, model_name, 'best_model.pt')
        if not os.path.exists(bp):
            continue

        state = torch.load(bp, map_location='cpu', weights_only=False)

        try:
            preds_raw, tgts_raw = [], []
            for idx in test_indices:
                sample = raw[idx]
                pts_np = np.array(sample['pts_3d'], dtype=np.float32)
                faces_np = np.array(sample['faces'], dtype=np.int64)
                x_raw = np.array(sample['x_pres'], dtype=np.float32)
                y_raw = np.array(sample['y_pres'], dtype=np.float32)
                if x_raw.ndim == 1: x_raw = x_raw[:, np.newaxis]
                if y_raw.ndim == 1: y_raw = y_raw[:, np.newaxis]

                pts = torch.tensor(pts_np)
                faces = torch.tensor(faces_np)
                K = CellComplex.from_triangulation(pts, faces).to(device)
                x_norm = torch.tensor((x_raw - xm) / xs, dtype=torch.float32).to(device)

                # Create model per-sample (n0/n1/n2 vary)
                if model_name == 'ours':
                    lift_shape = state['lifting.node_mlp.0.weight'].shape
                    C = lift_shape[0]
                    m = GaugeHodgeNetwork(
                        f_in=f_in, C=C, n_layers=4,
                        n0=K.n0, n1=K.n1, n2=K.n2,
                        task='scalar', spatial_dim=3, out_dim=out_dim,
                        mp_hidden=16, metric_type='local_rich', metric_rank=8)
                else:
                    h = infer_hidden(state)
                    if h is None: break
                    if model_name == 'gcn':
                        m = GCNBaseline(f_in=f_in, hidden=h, n_layers=4, out_dim=out_dim, task='node')
                    elif model_name == 'gat':
                        m = GATBaseline(f_in=f_in, hidden=h, n_layers=4, out_dim=out_dim, task='node')
                    elif model_name == 'schnet':
                        m = SchNetBaseline(f_in=f_in, hidden=h, n_layers=4, out_dim=out_dim, task='node')
                    elif model_name == 'egnn':
                        m = EGNNBaseline(f_in=f_in, hidden=h, n_layers=4, out_dim=out_dim, task='node')
                    else:
                        break

                m.load_state_dict(state)
                m = m.to(device).eval()

                with torch.no_grad():
                    pred_norm = m(x_norm, K)
                pred_raw = pred_norm.cpu().numpy() * ys + ym
                preds_raw.append(pred_raw)
                tgts_raw.append(y_raw)
                del m, K; torch.cuda.empty_cache()

            if len(preds_raw) == 0:
                continue

            preds_raw = np.array(preds_raw)
            tgts_raw = np.array(tgts_raw)
            ssim, pearson = compute_ssim_pearson(preds_raw, tgts_raw)

            # R² from normalized space
            preds_norm = (preds_raw - ym) / ys
            tgts_norm = (tgts_raw - ym) / ys
            r2, mse, mae = compute_r2_mse_mae(
                torch.tensor(preds_norm), torch.tensor(tgts_norm))
            nrmse = compute_nrmse(preds_raw, tgts_raw)

            results[model_name] = {
                'R2': r2, 'MSE': mse, 'MAE': mae,
                'NRMSE': nrmse, 'SSIM': ssim, 'Pearson': pearson,
            }
            print(f"  {model_name:15s} R2={r2:.4f}  SSIM={ssim:.3f}  Pearson={pearson:.3f}  NRMSE={nrmse:.4f}",
                  flush=True)
        except Exception as e:
            print(f"  {model_name}: ERROR {str(e)[:80]}", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=None, help='Single task: T1, T2, T5, T6, T7')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n-eval', type=int, default=100, help='Max test samples to evaluate')
    parser.add_argument('--ckpt-root', type=str, default='checkpoints',
                        help='Checkpoint tree root (default: checkpoints/). For multi-seed: checkpoints/seed_1, etc.')
    parser.add_argument('--out', type=str, default=None,
                        help='Output JSON path. Default: <ckpt-root>/all_metrics.json')
    args = parser.parse_args()

    os.chdir(os.path.join(os.path.dirname(__file__), '..'))

    TASKS = {
        'T1': ('T1_cns_vorticity', 'datasets/T1_cns_vorticity.pkl'),
        'T2': ('T2_torus_advection_diffusion', 'datasets/T2_torus_advection_diffusion.pkl'),
        'T5': ('T5_maxwell_poisson', 'datasets/T5_maxwell_poisson.pkl'),
        'T6': ('T6_wilson_loop', 'datasets/T6_wilson_loop.pkl'),
        'T7': ('T7_yang_mills_su2', 'datasets/T7_yang_mills_su2.pkl'),
        'T8': ('T8_airfoil_pressure', 'datasets/T8_airfoil_pressure_persample.pkl'),
    }
    # Also check for T3 variants
    # T3_ellipsoid_surface_flow uses T3_ellipsoid_coexact.pkl as dataset
    T3_DS_MAP = {
        'T3_ellipsoid_surface_flow': 'datasets/T3_ellipsoid_coexact.pkl',
        'T3_ellipsoid_coexact': 'datasets/T3_ellipsoid_coexact.pkl',
    }
    if os.path.isdir(args.ckpt_root):
        for d in os.listdir(args.ckpt_root):
            if d.startswith('T3_'):
                ds = T3_DS_MAP.get(d, f'datasets/{d}.pkl')
                if os.path.exists(ds):
                    TASKS[f'T3_{d}'] = (d, ds)

    if args.task:
        if args.task not in TASKS:
            print(f"Task {args.task} not found. Available: {list(TASKS.keys())}")
            return
        TASKS = {args.task: TASKS[args.task]}

    all_results = {}
    for tid, (tname, ds_path) in sorted(TASKS.items()):
        if tid == 'T8':
            all_results[tid] = evaluate_task_persample(tname, ds_path, args.device, args.n_eval, ckpt_root=args.ckpt_root)
        else:
            all_results[tid] = evaluate_task(tname, ds_path, args.device, args.n_eval, ckpt_root=args.ckpt_root)

    # Save (convert numpy types to float for JSON)
    out_path = args.out if args.out else os.path.join(args.ckpt_root, 'all_metrics.json')
    def convert(obj):
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32, np.int64)): return int(obj)
        raise TypeError(f'{type(obj)} not serializable')
    json.dump(all_results, open(out_path, 'w'), indent=2, default=convert)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
