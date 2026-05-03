"""
Formal benchmark runner for all 8 tasks (T1--T8).

Features:
  - Parameter-matched baselines (within 20% of ours)
  - Checkpoints every 25 epochs + best model
  - Metrics: R², MSE, MAE, SSIM, Pearson
  - Smoke test mode (--smoke for 2 epochs)

Usage:
  # Smoke test (2 epochs, verify everything works):
  CUDA_VISIBLE_DEVICES=0 python3 -u experiments/formal_benchmark.py T1 --smoke

  # Full run (100 epochs):
  CUDA_VISIBLE_DEVICES=0 nohup python3 -u experiments/formal_benchmark.py T1 \
      > logs/T1_cns_vorticity.log 2>&1 &

  # Run single model (for parallel GPU execution):
  CUDA_VISIBLE_DEVICES=0 python3 -u experiments/formal_benchmark.py T1 --model ours
  CUDA_VISIBLE_DEVICES=1 python3 -u experiments/formal_benchmark.py T1 --model egnn

  # T8 uses per-sample mesh training:
  CUDA_VISIBLE_DEVICES=0 python3 -u experiments/formal_benchmark.py T8 --smoke
"""
import sys, os, argparse, json, time, pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gauge_hodge_mp.cell_complex import CellComplex
from gauge_hodge_mp.network import GaugeHodgeNetwork
from baselines_graph import GCNBaseline, GATBaseline, SchNetBaseline, EGNNBaseline
from baselines_topo import MPSNBaseline, SCCNNBaseline
from baselines_advanced import GaugeEquivCNNBaseline, CWNetBaseline, CliffordNetBaseline, HermesBaseline
from baselines_operator import FNOWrapper, DeepONetWrapper

# ============================================================
# Task configurations
# ============================================================
TASKS = {
    'T1': {
        'name': 'cns_vorticity',
        'dataset': 'datasets/T1_cns_vorticity.pkl',
        'ours_C': 128, 'ours_params_target': 0.42,  # ~0.42M
        'batch_size': 64, 'lr': 1e-3,
        'metric_type': 'diagonal', 'metric_rank': 8,
        'spatial_dim': 2, 'task_type': 'scalar',
        'extra_baselines': ['fno'],  # regular grid → add FNO
        'persample': False,
    },
    'T2': {
        'name': 'torus_advection_diffusion',
        'dataset': 'datasets/T2_torus_advection_diffusion.pkl',
        'ours_C': 64, 'ours_params_target': 0.53,
        'batch_size': 64, 'lr': 1e-3,
        'metric_type': 'diagonal', 'metric_rank': 8,
        'spatial_dim': 2, 'task_type': 'scalar',
        'extra_baselines': [],
        'persample': False,
    },
    'T3': {
        'name': 'ellipsoid_surface_flow',
        'dataset': 'datasets/T3_ellipsoid_coexact.pkl',
        'ours_C': 64, 'ours_params_target': 0.37,
        'batch_size': 64, 'lr': 1e-3,
        'metric_type': 'diagonal', 'metric_rank': 8,
        'spatial_dim': 3, 'task_type': 'vector',
        'extra_baselines': [],
        'persample': False,
    },
    'T4': {
        'name': 'ellipsoid_curl_field',
        'dataset': 'datasets/T4_ellipsoid_curl_field.pkl',
        'ours_C': 128, 'ours_params_target': 0.88,  # C=128 on 2562-node ellipsoid → 0.88M
        'batch_size': 64, 'lr': 1e-3,
        'metric_type': 'diagonal', 'metric_rank': 8,
        'spatial_dim': 3, 'task_type': 'scalar',
        'extra_baselines': [],
        'persample': False,
    },
    'T5': {
        'name': 'maxwell_poisson',
        'dataset': 'datasets/T5_maxwell_poisson.pkl',
        'ours_C': 160, 'ours_params_target': 0.95,  # C=160 on 1024-node Delaunay → 0.95M
        'batch_size': 64, 'lr': 1e-3,
        'metric_type': 'diagonal', 'metric_rank': 8,
        'spatial_dim': 2, 'task_type': 'scalar',
        'extra_baselines': [],
        'persample': False,
    },
    'T6': {
        'name': 'wilson_loop',
        'dataset': 'datasets/T6_wilson_loop.pkl',
        'ours_C': 128, 'ours_params_target': 0.43,
        'batch_size': 64, 'lr': 1e-3,
        'metric_type': 'diagonal', 'metric_rank': 8,
        'spatial_dim': 2, 'task_type': 'scalar',
        'extra_baselines': [],
        'persample': False,
        'edge_channels': 1,  # input is encode_edges_to_nodes: (avg, avg*dx, avg*dy) per edge ch
    },
    'T7': {
        'name': 'yang_mills_su2',
        'dataset': 'datasets/T7_yang_mills_su2.pkl',
        'ours_C': 160, 'ours_params_target': 0.95,  # C=160 on 1024-node Delaunay → 0.95M
        'batch_size': 64, 'lr': 1e-3,
        'metric_type': 'diagonal', 'metric_rank': 8,
        'spatial_dim': 2, 'task_type': 'scalar',
        'extra_baselines': [],
        'persample': False,
        'edge_channels': 3,  # input is encode_edges_to_nodes: 3 SU(2) channels × (avg, avg*dx, avg*dy)
    },
    'T8': {
        'name': 'airfoil_pressure',
        'dataset': 'datasets/T8_airfoil_pressure_persample.pkl',
        'ours_C': 256, 'ours_params_target': 0.54,  # C=256 local_rich on per-sample mesh
        'batch_size': 1, 'lr': 1e-3,
        'metric_type': 'local_rich', 'metric_rank': 0,
        'spatial_dim': 2, 'task_type': 'scalar',
        'extra_baselines': [],
        'persample': True,
    },
}

# ============================================================
# Helpers
# ============================================================
class DS(Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X, Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.Y[i]

def collate_fn(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)

def compute_metrics(pred, target):
    """Compute R², MSE, MAE."""
    mse = ((pred - target) ** 2).mean().item()
    mae = (pred - target).abs().mean().item()
    ss_res = ((pred - target) ** 2).sum().item()
    ss_tot = ((target - target.mean(0, keepdim=True)) ** 2).sum().item()
    r2 = 1 - ss_res / max(ss_tot, 1e-8)
    return {'R2': r2, 'MSE': mse, 'MAE': mae}


def compute_ssim_pearson(pred, target):
    """Compute per-sample SSIM and Pearson on unnormalized predictions.
    pred, target: (N, n, d) numpy arrays (unnormalized).
    Returns mean SSIM and mean Pearson over samples.

    SSIM uses global data_range from target (consistent C1/C2 across samples).
    Predictions are clipped to target range to prevent outlier-inflated variance.
    """
    N = len(pred)
    # Global data range from ALL target samples (consistent C1/C2)
    data_range = float(target.max() - target.min())
    if data_range < 1e-8:
        data_range = 1.0
    c1 = (0.01 * data_range)**2
    c2 = (0.03 * data_range)**2

    # Clip predictions to target range (prevent outlier-inflated variance)
    t_min, t_max = float(target.min()), float(target.max())
    margin = 0.1 * data_range
    pred_clipped = np.clip(pred, t_min - margin, t_max + margin)

    ssims, pearsons = [], []
    for i in range(N):
        p = pred_clipped[i].flatten().astype(np.float64)
        t = target[i].flatten().astype(np.float64)
        # Pearson
        tp = t - t.mean(); pp = p - p.mean()
        denom = np.sqrt(np.sum(tp**2) * np.sum(pp**2))
        pearson = np.sum(tp * pp) / (denom + 1e-15) if denom > 1e-15 else 0.0
        pearsons.append(pearson)
        # SSIM (global C1/C2)
        mu_t, mu_p = t.mean(), p.mean()
        sig_t, sig_p = t.std(), p.std()
        sig_tp = np.mean((t - mu_t) * (p - mu_p))
        num = (2*mu_t*mu_p + c1) * (2*sig_tp + c2)
        den = (mu_t**2 + mu_p**2 + c1) * (sig_t**2 + sig_p**2 + c2)
        ssims.append(num / (den + 1e-15))
    return float(np.mean(ssims)), float(np.mean(pearsons))

def find_hidden_match(cls_fn, target_params_M, f_in, out_dim, tol=0.20):
    """Find hidden dim so baseline params are in [target, target*(1+tol)]."""
    for h in range(16, 800, 4):
        try:
            m = cls_fn(f_in, h, out_dim)
        except TypeError:
            m = cls_fn(f_in, h)
        np_ = sum(p.numel() for p in m.parameters()) / 1e6
        if np_ >= target_params_M:
            if np_ <= target_params_M * (1 + tol):
                return h, np_
            else:
                return h, np_  # best we can do
    return 128, 0.0

# ============================================================
# Training engine
# ============================================================
def train_and_eval(model, train_loader, val_loader, test_loader, K_dev, n_epochs, save_dir, name,
                   device='cuda', is_edge=False, ym=None, ys=None):
    """Train model with full logging, checkpoints, metrics. Supports resume."""
    os.makedirs(save_dir, exist_ok=True)
    np_ = sum(p.numel() for p in model.parameters()) / 1e6

    # Check if already completed
    result_path = os.path.join(save_dir, 'result.json')
    if os.path.exists(result_path):
        prev = json.load(open(result_path))
        if prev.get('best_epoch', 0) > 0:
            print(f"  {name}: already completed (R2={prev['best_R2']:.4f}@ep{prev['best_epoch']}), skipping", flush=True)
            return prev

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)

    history = []
    best_r2 = -999
    best_ep = 0
    start_ep = 1

    # Resume from latest checkpoint
    ckpt_files = sorted([f for f in os.listdir(save_dir) if f.startswith('checkpoint_ep') and f.endswith('.pt')])
    if ckpt_files:
        latest = ckpt_files[-1]
        resume_ep = int(latest.replace('checkpoint_ep', '').replace('.pt', ''))
        if resume_ep < n_epochs:
            model.load_state_dict(torch.load(os.path.join(save_dir, latest), map_location=device))
            start_ep = resume_ep + 1
            # Advance scheduler
            for _ in range(resume_ep):
                scheduler.step()
            # Load history if exists
            hist_path = os.path.join(save_dir, 'history.json')
            if os.path.exists(hist_path):
                history = json.load(open(hist_path))
                for h in history:
                    if h['R2'] > best_r2:
                        best_r2 = h['R2']
                        best_ep = h['epoch']
            print(f"  {name}: resuming from ep{resume_ep} (best R2={best_r2:.4f})", flush=True)

    for ep in range(start_ep, n_epochs + 1):
        # --- Train ---
        model.train()
        t0 = time.time()
        train_loss = 0
        n_batches = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            has_fb = hasattr(model, 'forward_batch')
            pred = model.forward_batch(bx, K_dev) if has_fb else torch.stack(
                [model(bx[i], K_dev) for i in range(bx.size(0))])
            if is_edge:
                s, d = K_dev.edges[:, 0], K_dev.edges[:, 1]
                by_e = 0.5 * (by[:, s] + by[:, d])
                if by_e.size(-1) > pred.size(-1):
                    by_e = by_e[..., :pred.size(-1)]
                loss = F.mse_loss(pred, by_e)
            else:
                loss = F.mse_loss(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
            n_batches += 1
        scheduler.step()
        train_loss /= max(n_batches, 1)
        train_time = time.time() - t0

        # --- Eval ---
        model.eval()
        all_pred, all_target = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                has_fb = hasattr(model, 'forward_batch')
                pred = model.forward_batch(bx, K_dev) if has_fb else torch.stack(
                    [model(bx[i], K_dev) for i in range(bx.size(0))])
                if is_edge:
                    s, d = K_dev.edges[:, 0], K_dev.edges[:, 1]
                    by = 0.5 * (by[:, s] + by[:, d])
                    if by.size(-1) > pred.size(-1):
                        by = by[..., :pred.size(-1)]
                all_pred.append(pred.cpu())
                all_target.append(by.cpu())
        all_pred = torch.cat(all_pred)
        all_target = torch.cat(all_target)
        metrics = compute_metrics(all_pred, all_target)
        metrics['train_loss'] = train_loss
        metrics['epoch'] = ep
        metrics['time'] = train_time
        history.append(metrics)

        # --- Best model ---
        if metrics['R2'] > best_r2:
            best_r2 = metrics['R2']
            best_ep = ep
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))

        # --- Checkpoint + save history ---
        if ep % 25 == 0 or ep == 1:
            torch.save(model.state_dict(), os.path.join(save_dir, f'checkpoint_ep{ep}.pt'))
            with open(os.path.join(save_dir, 'history.json'), 'w') as f:
                json.dump(history, f, indent=2)

        # --- Print ---
        if ep % 5 == 0 or ep == 1 or ep == n_epochs:
            print(f"  {name} ep{ep}: R2={metrics['R2']:.4f} MSE={metrics['MSE']:.6f} "
                  f"loss={train_loss:.6f} ({train_time:.1f}s)", flush=True)

    # Save final history
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # --- Test evaluation with best model: R², SSIM, Pearson ---
    model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pt'),
                                      map_location=device, weights_only=False))
    model.eval()
    test_pred_raw, test_target_raw = [], []
    test_pred_norm, test_target_norm = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            has_fb = hasattr(model, 'forward_batch')
            pred = model.forward_batch(bx, K_dev) if has_fb else torch.stack(
                [model(bx[i], K_dev) for i in range(bx.size(0))])
            if is_edge:
                s, d = K_dev.edges[:, 0], K_dev.edges[:, 1]
                by = 0.5 * (by[:, s] + by[:, d])
                if by.size(-1) > pred.size(-1):
                    by = by[..., :pred.size(-1)]
            test_pred_norm.append(pred.cpu())
            test_target_norm.append(by.cpu())
            # Unnormalize for SSIM/Pearson
            pred_raw = pred.cpu() * ys + ym
            by_raw = by.cpu() * ys + ym
            test_pred_raw.append(pred_raw)
            test_target_raw.append(by_raw)
    test_pred_norm = torch.cat(test_pred_norm)
    test_target_norm = torch.cat(test_target_norm)
    test_metrics = compute_metrics(test_pred_norm, test_target_norm)

    test_pred_raw = torch.cat(test_pred_raw).numpy()
    test_target_raw = torch.cat(test_target_raw).numpy()
    test_ssim, test_pearson = compute_ssim_pearson(test_pred_raw, test_target_raw)

    # Final summary
    result = {
        'model': name, 'params_M': np_,
        'best_R2': best_r2, 'best_epoch': best_ep,
        'final_R2': history[-1]['R2'],
        'final_MSE': history[-1]['MSE'],
        'final_MAE': history[-1]['MAE'],
        'test_R2': test_metrics['R2'],
        'test_MSE': test_metrics['MSE'],
        'test_SSIM': test_ssim,
        'test_Pearson': test_pearson,
    }
    with open(os.path.join(save_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"{name:15s} {np_:5.2f}M  best_R2={best_r2:.4f}@ep{best_ep}  "
          f"SSIM={test_ssim:.4f}  Pearson={test_pearson:.4f}", flush=True)
    return result

# ============================================================
# Per-sample training (T8)
# ============================================================
def train_and_eval_persample(model, raw, xm, xs, ym, ys, n_train, n_epochs,
                              save_dir, name, device='cuda'):
    """Per-sample training for cross-mesh tasks (T8). Supports resume."""
    os.makedirs(save_dir, exist_ok=True)
    np_ = sum(p.numel() for p in model.parameters()) / 1e6

    # Check if already completed
    result_path = os.path.join(save_dir, 'result.json')
    if os.path.exists(result_path):
        prev = json.load(open(result_path))
        if prev.get('best_epoch', 0) > 0:
            print(f"  {name}: already completed (R2={prev['best_R2']:.4f}@ep{prev['best_epoch']}), skipping", flush=True)
            return prev

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)

    history = []
    best_r2 = -999
    best_ep = 0
    start_ep = 1

    # Resume from latest checkpoint
    ckpt_files = sorted([f for f in os.listdir(save_dir) if f.startswith('checkpoint_ep') and f.endswith('.pt')])
    if ckpt_files:
        latest = ckpt_files[-1]
        resume_ep = int(latest.replace('checkpoint_ep', '').replace('.pt', ''))
        if resume_ep < n_epochs:
            model.load_state_dict(torch.load(os.path.join(save_dir, latest), map_location=device))
            start_ep = resume_ep + 1
            for _ in range(resume_ep):
                sched.step()
            hist_path = os.path.join(save_dir, 'history.json')
            if os.path.exists(hist_path):
                history = json.load(open(hist_path))
                for h in history:
                    if h['R2'] > best_r2:
                        best_r2 = h['R2']
                        best_ep = h['epoch']
            print(f"  {name}: resuming from ep{resume_ep} (best R2={best_r2:.4f})", flush=True)

    for ep in range(start_ep, n_epochs + 1):
        model.train()
        t0 = time.time()
        perm = np.random.permutation(n_train)
        ep_loss = 0
        for i in perm:
            x = torch.tensor((raw[i][0] - xm) / xs, dtype=torch.float32).to(device)
            y = torch.tensor((raw[i][1] - ym) / ys, dtype=torch.float32).to(device)
            opt.zero_grad()
            loss = F.mse_loss(model(x, raw[i][2].to(device)), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        sched.step()
        ep_loss /= n_train
        train_time = time.time() - t0

        model.eval()
        all_pred, all_target = [], []
        with torch.no_grad():
            for i in range(n_train, len(raw)):
                x = torch.tensor((raw[i][0] - xm) / xs, dtype=torch.float32).to(device)
                y = torch.tensor((raw[i][1] - ym) / ys, dtype=torch.float32).to(device)
                p = model(x, raw[i][2].to(device))
                all_pred.append(p.cpu())
                all_target.append(y.cpu())
        all_pred = torch.cat(all_pred)
        all_target = torch.cat(all_target)
        metrics = compute_metrics(all_pred, all_target)
        metrics['train_loss'] = ep_loss
        metrics['epoch'] = ep
        metrics['time'] = train_time
        history.append(metrics)

        if metrics['R2'] > best_r2:
            best_r2 = metrics['R2']
            best_ep = ep
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pt'))

        if ep % 25 == 0 or ep == 1:
            torch.save(model.state_dict(), os.path.join(save_dir, f'checkpoint_ep{ep}.pt'))
            with open(os.path.join(save_dir, 'history.json'), 'w') as f:
                json.dump(history, f, indent=2)

        if ep % 5 == 0 or ep == 1 or ep == n_epochs:
            print(f"  {name} ep{ep}: R2={metrics['R2']:.4f} MSE={metrics['MSE']:.6f} "
                  f"loss={ep_loss:.6f} ({train_time:.1f}s)", flush=True)

    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    result = {
        'model': name, 'params_M': np_,
        'best_R2': best_r2, 'best_epoch': best_ep,
        'final_R2': history[-1]['R2'],
        'final_MSE': history[-1]['MSE'],
        'final_MAE': history[-1]['MAE'],
    }
    with open(os.path.join(save_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f"{name:15s} {np_:5.2f}M  best_R2={best_r2:.4f}@ep{best_ep}", flush=True)
    return result

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task_id', type=str, help='Task ID: T1-T8')
    parser.add_argument('--smoke', action='store_true', help='Smoke test (2 epochs)')
    parser.add_argument('--model', type=str, default='all',
                        help='Run single model: ours/gcn/gat/schnet/egnn/mpsn/sccnn/gauge_cnn/cw_net/clifford/all')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--multi-seed-dir', action='store_true',
                        help='Save under checkpoints/seed_<N>/ and results/seed_<N>/ (for multi-seed runs)')
    args = parser.parse_args()

    n_epochs = 2 if args.smoke else args.epochs
    task_id = args.task_id
    cfg = TASKS[task_id]
    device = 'cuda'

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    print(f"{'='*60}")
    print(f"TASK: {task_id} — {cfg['name']}")
    print(f"Epochs: {n_epochs}, Seed: {args.seed}")
    print(f"{'='*60}")

    # --- Load dataset ---
    base_dir = os.path.dirname(os.path.dirname(__file__))

    if cfg['persample']:
        # T8: per-sample loading
        print("Loading per-sample dataset...")
        with open(os.path.join(base_dir, cfg['dataset']), 'rb') as f:
            samples = pickle.load(f)
        print(f"  {len(samples)} samples loaded")

        raw = []
        for s in samples:
            K = CellComplex.from_triangulation(
                torch.tensor(s['pts_3d']), torch.tensor(s['faces'], dtype=torch.long), device='cpu')
            raw.append((s['x_pres'], s['y_pres'], K))

        n_train = int(0.7 * len(raw))
        all_x = np.stack([r[0] for r in raw])
        all_y = np.stack([r[1] for r in raw])
        xm = all_x[:n_train].mean((0, 1))
        xs = all_x[:n_train].std((0, 1)).clip(1e-6)
        ym = all_y[:n_train].mean((0, 1))
        ys = all_y[:n_train].std((0, 1)).clip(1e-6)
        f_in = raw[0][0].shape[-1]
        out_dim = raw[0][1].shape[-1]
        N_PTS = raw[0][0].shape[0]
        print(f"  f_in={f_in}, out={out_dim}, n_pts={N_PTS}, train={n_train}")
    else:
        print("Loading dataset...")
        with open(os.path.join(base_dir, cfg['dataset']), 'rb') as f:
            d = pickle.load(f)

        pts = torch.tensor(np.array(d['points'], dtype=np.float32))
        faces = torch.tensor(np.array(d['faces'], dtype=np.int64))
        K = CellComplex.from_triangulation(pts, faces)

        X = torch.tensor(np.array(d['X_data'], dtype=np.float32))
        Y = torch.tensor(np.array(d['Y_data'], dtype=np.float32))
        if X.ndim == 2: X = X.unsqueeze(-1)
        if Y.ndim == 2: Y = Y.unsqueeze(-1)

        f_in = X.shape[-1]
        out_dim = Y.shape[-1]
        n_samples = len(X)
        nt = int(0.7 * n_samples)
        nv = int(0.15 * n_samples)

        xm, xs_t = X[:nt].mean((0, 1)), X[:nt].std((0, 1)).clamp(1e-6)
        ym, ys_t = Y[:nt].mean((0, 1)), Y[:nt].std((0, 1)).clamp(1e-6)
        X = (X - xm) / xs_t
        Y = (Y - ym) / ys_t

        train_loader = DataLoader(DS(X[:nt], Y[:nt]), batch_size=cfg['batch_size'],
                                  shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(DS(X[nt:nt+nv], Y[nt:nt+nv]), batch_size=cfg['batch_size'],
                                shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(DS(X[nt+nv:], Y[nt+nv:]), batch_size=cfg['batch_size'],
                                 shuffle=False, collate_fn=collate_fn)
        K_dev = K.to(device)
        N_PTS = K.n0

        print(f"  {n_samples} samples, n0={K.n0}, n1={K.n1}, n2={K.n2}, f_in={f_in}, out={out_dim}")

    # --- Build ours ---
    if cfg['persample']:
        K_first = raw[0][2]
        ours = GaugeHodgeNetwork(
            f_in=f_in, C=cfg['ours_C'], n_layers=4,
            n0=N_PTS, n1=1, n2=1,
            task=cfg['task_type'], spatial_dim=cfg['spatial_dim'],
            out_dim=out_dim, mp_hidden=16,
            metric_type=cfg['metric_type'], metric_rank=cfg['metric_rank']
        ).to(device)
    else:
        ours = GaugeHodgeNetwork(
            f_in=f_in, C=cfg['ours_C'], n_layers=4,
            n0=K.n0, n1=K.n1, n2=K.n2,
            task=cfg['task_type'], spatial_dim=cfg['spatial_dim'],
            out_dim=out_dim, mp_hidden=16,
            metric_type=cfg['metric_type'], metric_rank=cfg['metric_rank']
        ).to(device)

    ours_np = sum(p.numel() for p in ours.parameters()) / 1e6

    # --- Parameter check ---
    print(f"\n--- Parameter Check ---")
    print(f"  ours: {ours_np:.3f}M (target: ~{cfg['ours_params_target']:.2f}M)")

    # --- Baselines ---
    NODE_BL = {
        'gcn': lambda fi, h, od: GCNBaseline(f_in=fi, hidden=h, n_layers=4, out_dim=od, task='node'),
        'gat': lambda fi, h, od: GATBaseline(f_in=fi, hidden=h, n_layers=4, out_dim=od, task='node'),
        'schnet': lambda fi, h, od: SchNetBaseline(f_in=fi, hidden=h, n_layers=4, out_dim=od, task='node'),
        'egnn': lambda fi, h, od: EGNNBaseline(f_in=fi, hidden=h, n_layers=4, out_dim=od, task='node'),
        'gauge_cnn': lambda fi, h, od: GaugeEquivCNNBaseline(f_in=fi, hidden=h, n_layers=4, out_dim=od, task='node'),
        'cw_net': lambda fi, h, od: CWNetBaseline(f_in=fi, hidden=h, n_layers=4, out_dim=od, task='node'),
        'gem_cnn': lambda fi, h, od: HermesBaseline(f_in=fi, hidden=h, n_layers=4, out_dim=od, task='node'),
        'fno': lambda fi, h, od: FNOWrapper(f_in=fi, out_dim=od, width=h, modes=12, n_layers=4),
        'deeponet': lambda fi, h, od: DeepONetWrapper(f_in=fi, out_dim=od, branch_width=h, trunk_width=h, n_basis=64),
    }
    EDGE_BL = {
        'mpsn': lambda fi, h: MPSNBaseline(f_in=fi, hidden=h, n_layers=4),
        'sccnn': lambda fi, h: SCCNNBaseline(f_in=fi, hidden=h, n_layers=4),
        'clifford_smpn': lambda fi, h: CliffordNetBaseline(f_in=fi, hidden=h, n_layers=4),
    }

    # EGNN gets only scalar (invariant) features when directional encoding exists.
    # encode_edges_to_nodes produces (avg, avg*dx, avg*dy) per edge channel;
    # EGNN should only see avg (edge_channels) since it's E(n)-equivariant
    # and shouldn't benefit from hand-crafted coordinate-dependent features.
    egnn_f_in = cfg.get('edge_channels', f_in)  # scalar channels only

    bl_configs = {}
    for bname, cls_fn in NODE_BL.items():
        fi_bl = egnn_f_in if bname == 'egnn' else f_in
        h, np_bl = find_hidden_match(cls_fn, ours_np, fi_bl, out_dim, tol=0.20)
        bl_configs[bname] = (h, np_bl, False)  # (hidden, params, is_edge)
        print(f"  {bname}: h={h}, params={np_bl:.3f}M (ratio={np_bl/ours_np:.2f}x)"
              + (f" [f_in={fi_bl}, scalar only]" if fi_bl != f_in else ""))

    if not cfg['persample']:  # edge baselines don't work with per-sample
        for bname, cls_fn in EDGE_BL.items():
            h, np_bl = find_hidden_match(cls_fn, ours_np, f_in, out_dim, tol=0.20)
            bl_configs[bname] = (h, np_bl, True)
            print(f"  {bname}: h={h}, params={np_bl:.3f}M (ratio={np_bl/ours_np:.2f}x)")

    seed_subdir = f'seed_{args.seed}' if args.multi_seed_dir else ''
    results_dir = os.path.join(base_dir, 'results', seed_subdir, f'{task_id}_{cfg["name"]}')
    ckpt_dir = os.path.join(base_dir, 'checkpoints', seed_subdir, f'{task_id}_{cfg["name"]}')
    os.makedirs(results_dir, exist_ok=True)

    # --- Filter models to run ---
    run_model = args.model
    models_to_run = []
    if run_model in ('all', 'ours'):
        models_to_run.append('ours')
    if run_model == 'all':
        models_to_run.extend(bl_configs.keys())
    elif run_model != 'ours' and run_model in bl_configs:
        models_to_run.append(run_model)

    print(f"\n--- Training ({n_epochs} epochs, models: {models_to_run}) ---")
    all_results = []

    # --- Train ours ---
    if 'ours' in models_to_run:
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        if cfg['persample']:
            r = train_and_eval_persample(
                ours, raw, xm, xs, ym, ys, n_train, n_epochs,
                os.path.join(ckpt_dir, 'ours'), 'ours', device)
        else:
            r = train_and_eval(
                ours, train_loader, val_loader, test_loader, K_dev, n_epochs,
                os.path.join(ckpt_dir, 'ours'), 'ours', device, ym=ym, ys=ys_t)
        all_results.append(r)
    del ours; torch.cuda.empty_cache()

    # --- EGNN scalar-only data loaders (strip directional encoding) ---
    egnn_train_loader = egnn_val_loader = egnn_test_loader = None
    if egnn_f_in != f_in and not cfg['persample']:
        # Only keep first edge_channels columns (scalar avg), drop avg*dx, avg*dy
        ec = egnn_f_in
        X_egnn = torch.cat([X[:, :, i*3:i*3+1] for i in range(ec)], dim=-1) if f_in == ec * 3 else X[:, :, :ec]
        egnn_train_loader = DataLoader(DS(X_egnn[:nt], Y[:nt]), batch_size=cfg['batch_size'],
                                       shuffle=True, collate_fn=collate_fn)
        egnn_val_loader = DataLoader(DS(X_egnn[nt:nt+nv], Y[nt:nt+nv]), batch_size=cfg['batch_size'],
                                     shuffle=False, collate_fn=collate_fn)
        egnn_test_loader = DataLoader(DS(X_egnn[nt+nv:], Y[nt+nv:]), batch_size=cfg['batch_size'],
                                      shuffle=False, collate_fn=collate_fn)
        print(f"  EGNN: using scalar-only input f_in={ec} (stripped directional encoding)")

    # --- Train baselines ---
    for bname in models_to_run:
        if bname == 'ours':
            continue
        if bname not in bl_configs:
            print(f"  WARNING: {bname} not in bl_configs, skipping")
            continue
        h, np_bl, is_edge = bl_configs[bname]
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        fi_bl = egnn_f_in if bname == 'egnn' else f_in
        if is_edge:
            m = EDGE_BL[bname](f_in, h).to(device)
        else:
            m = NODE_BL[bname](fi_bl, h, out_dim).to(device)
        # Use scalar-only loaders for EGNN
        if cfg['persample']:
            tl = vl = tel = None
        elif bname == 'egnn' and egnn_train_loader is not None:
            tl, vl, tel = egnn_train_loader, egnn_val_loader, egnn_test_loader
        else:
            tl, vl, tel = train_loader, val_loader, test_loader
        if cfg['persample']:
            r = train_and_eval_persample(
                m, raw, xm, xs, ym, ys, n_train, n_epochs,
                os.path.join(ckpt_dir, bname), bname, device)
        else:
            r = train_and_eval(
                m, tl, vl, tel, K_dev, n_epochs,
                os.path.join(ckpt_dir, bname), bname, device, is_edge=is_edge, ym=ym, ys=ys_t)
        all_results.append(r)
        del m; torch.cuda.empty_cache()

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"SUMMARY: {task_id} — {cfg['name']}")
    print(f"{'model':15s} {'params':>7s} {'best_R2':>9s} {'best_ep':>8s} {'MSE':>10s} {'MAE':>10s}")
    print('-' * 60)
    for r in all_results:
        print(f"{r['model']:15s} {r['params_M']:6.2f}M {r['best_R2']:9.4f} "
              f"ep{r['best_epoch']:>4d}  {r['final_MSE']:10.6f} {r['final_MAE']:10.6f}")

    # Save summary
    with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_dir}/")
    print("DONE")

if __name__ == '__main__':
    main()
