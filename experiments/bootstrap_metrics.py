"""
Bootstrap std over test set for all (task, model) pairs.

Strategy:
1. For each (task, model): load best_model.pt, run inference once, cache per-sample
   (pred_raw, target_raw, pred_norm, target_norm).
2. Bootstrap: resample test indices with replacement N_BOOT times, recompute all 6
   metrics on each resample.
3. Report mean, std (and optionally 95% CI) per (task, model, metric).

Legit: this is standard bootstrap-over-test-set — represents uncertainty from
finite test sample, not seed-to-seed training variance.

Usage:
  python3 experiments/bootstrap_metrics.py             # all tasks
  python3 experiments/bootstrap_metrics.py --task T1   # single task
  python3 experiments/bootstrap_metrics.py --n-boot 1000 --seed 0
"""
import sys, os, json, pickle, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Reuse existing eval utilities by importing compute_all_metrics as a module
THIS_DIR = os.path.dirname(__file__)
sys.path.insert(0, THIS_DIR)
import compute_all_metrics as cam  # noqa

from gauge_hodge_mp.cell_complex import CellComplex


# ------------------------------------------------------------
# Metrics on arrays (vectorized across a boot sample)
# ------------------------------------------------------------

def _r2_mse_mae(pred_n, tgt_n):
    """pred_n, tgt_n: np arrays shape (B, ...)  -- normalized."""
    diff = pred_n - tgt_n
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    ss_res = float(np.sum(diff ** 2))
    mu = tgt_n.mean(axis=0, keepdims=True)
    ss_tot = float(np.sum((tgt_n - mu) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
    return r2, mse, mae


def _nrmse(pred_raw, tgt_raw, data_range):
    """NRMSE with fixed global data_range (property of the data, not resample)."""
    rmse = float(np.sqrt(np.mean((pred_raw - tgt_raw) ** 2)))
    return rmse / max(data_range, 1e-8)


def _ssim_pearson_precomputed(ssim_per, pearson_per, idx):
    """Both metrics are already per-sample means — just average over resample idx."""
    return float(np.mean(ssim_per[idx])), float(np.mean(pearson_per[idx]))


def _precompute_per_sample_ssim_pearson(preds_raw, tgts_raw):
    """Compute per-sample SSIM and Pearson (matching cam.compute_ssim_pearson logic)."""
    N = len(preds_raw)
    data_range = float(tgts_raw.max() - tgts_raw.min())
    if data_range < 1e-8:
        data_range = 1.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    t_min, t_max = float(tgts_raw.min()), float(tgts_raw.max())
    margin = 0.1 * data_range
    pred_clipped = np.clip(preds_raw, t_min - margin, t_max + margin)

    ssims = np.empty(N, dtype=np.float64)
    pearsons = np.empty(N, dtype=np.float64)
    for i in range(N):
        p = pred_clipped[i].flatten().astype(np.float64)
        t = tgts_raw[i].flatten().astype(np.float64)
        tp = t - t.mean()
        pp = p - p.mean()
        denom = np.sqrt((tp ** 2).sum() * (pp ** 2).sum())
        pearson = (tp * pp).sum() / (denom + 1e-15) if denom > 1e-15 else 0.0
        pearsons[i] = max(0.0, float(pearson))
        mu_t, mu_p = t.mean(), p.mean()
        sig_t, sig_p = t.std(), p.std()
        sig_tp = ((t - mu_t) * (p - mu_p)).mean()
        num = (2 * mu_t * mu_p + c1) * (2 * sig_tp + c2)
        den = (mu_t ** 2 + mu_p ** 2 + c1) * (sig_t ** 2 + sig_p ** 2 + c2)
        ssims[i] = num / (den + 1e-15)
    return ssims, pearsons, data_range


# ------------------------------------------------------------
# Inference stage: collect per-sample preds for every (task, model)
# ------------------------------------------------------------

def _run_inference_task(task_name, ds_path, device, n_eval):
    """Return dict: model_name -> (preds_raw, tgts_raw, preds_norm, tgts_norm)."""
    print(f"\n{'='*60}\n{task_name} -- inference", flush=True)
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
    nt = int(0.7 * n_total); nv = int(0.15 * n_total)
    xm, xs = X[:nt].mean((0, 1)), X[:nt].std((0, 1)).clamp(1e-6)
    ym, ys = Y[:nt].mean((0, 1)), Y[:nt].std((0, 1)).clamp(1e-6)
    Xn = (X - xm) / xs
    Xt = Xn[nt + nv:].to(device)
    Yt_raw = Y_raw[nt + nv:]
    n_test = min(n_eval, len(Xt))

    edges = K.edges
    s_idx = edges[:, 0].cpu().numpy()
    d_idx = edges[:, 1].cpu().numpy()
    Yt_edge_raw = 0.5 * (Yt_raw[:, s_idx] + Yt_raw[:, d_idx])

    ckpt_base = f'checkpoints/{task_name}'
    out = {}
    if not os.path.isdir(ckpt_base):
        return out

    for model_name in sorted(os.listdir(ckpt_base)):
        bp = os.path.join(ckpt_base, model_name, 'best_model.pt')
        if not os.path.exists(bp):
            continue
        try:
            state = torch.load(bp, map_location='cpu', weights_only=False)
            loaded = cam.load_model(model_name, state, f_in, out_dim, K, task_name)
            if loaded is None:
                continue
            m, is_edge = loaded
            m = m.to(device).eval()
            model_f_in = f_in
            embed_w = state.get('embed.weight')
            if embed_w is not None and embed_w.ndim == 2 and embed_w.shape[1] != f_in:
                model_f_in = embed_w.shape[1]

            preds_raw, tgts_raw, preds_norm, tgts_norm = [], [], [], []
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
                        tgt_raw_i = Yt_edge_raw[i]
                        od = pred_n.shape[-1]
                        if tgt_raw_i.shape[-1] > od:
                            tgt_raw_i = tgt_raw_i[..., :od]
                        ys_e = ys[:od].numpy()
                        ym_e = ym[:od].numpy()
                    else:
                        tgt_raw_i = Yt_raw[i]
                        ys_e = ys.numpy()
                        ym_e = ym.numpy()
                    pred_raw_i = pred_n.cpu().numpy() * ys_e + ym_e
                    preds_raw.append(pred_raw_i)
                    tgts_raw.append(tgt_raw_i)
                    preds_norm.append(pred_n.cpu().numpy())
                    tgts_norm.append((tgt_raw_i - ym_e) / ys_e)

            out[model_name] = (
                np.array(preds_raw), np.array(tgts_raw),
                np.array(preds_norm), np.array(tgts_norm),
            )
            print(f"  {model_name:15s} inferred {n_test} samples", flush=True)
            del m
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {model_name}: ERROR {str(e)[:120]}", flush=True)
    return out


def _run_inference_task_persample(task_name, ds_path, device, n_eval):
    """T8-style per-sample meshes."""
    print(f"\n{'='*60}\n{task_name} (per-sample) -- inference", flush=True)
    raw = pickle.load(open(ds_path, 'rb'))
    n_total = len(raw)
    nt = int(0.7 * n_total); nv = int(0.15 * n_total)

    all_x = np.stack([np.array(raw[i]['x_pres'], dtype=np.float32) for i in range(nt)])
    all_y = np.stack([np.array(raw[i]['y_pres'], dtype=np.float32) for i in range(nt)])
    xm = all_x.mean((0, 1))
    xs = np.clip(all_x.std((0, 1)), 1e-6, None)
    ym = all_y.mean((0, 1))
    ys = np.clip(all_y.std((0, 1)), 1e-6, None)

    test_indices = list(range(nt + nv, n_total))[:n_eval]
    ckpt_base = f'checkpoints/{task_name}'
    if not os.path.isdir(ckpt_base):
        return {}
    sample0 = raw[0]
    x0 = np.array(sample0['x_pres'], dtype=np.float32)
    y0 = np.array(sample0['y_pres'], dtype=np.float32)
    if x0.ndim == 1: x0 = x0[:, np.newaxis]
    if y0.ndim == 1: y0 = y0[:, np.newaxis]
    f_in, out_dim = x0.shape[-1], y0.shape[-1]

    out = {}
    for model_name in sorted(os.listdir(ckpt_base)):
        bp = os.path.join(ckpt_base, model_name, 'best_model.pt')
        if not os.path.exists(bp):
            continue
        state = torch.load(bp, map_location='cpu', weights_only=False)
        try:
            preds_raw, tgts_raw = [], []
            from baselines_graph import GCNBaseline, GATBaseline, SchNetBaseline, EGNNBaseline
            from gauge_hodge_mp.network import GaugeHodgeNetwork
            for idx in test_indices:
                sample = raw[idx]
                pts_np = np.array(sample['pts_3d'], dtype=np.float32)
                faces_np = np.array(sample['faces'], dtype=np.int64)
                x_raw = np.array(sample['x_pres'], dtype=np.float32)
                y_raw = np.array(sample['y_pres'], dtype=np.float32)
                if x_raw.ndim == 1: x_raw = x_raw[:, np.newaxis]
                if y_raw.ndim == 1: y_raw = y_raw[:, np.newaxis]
                pts = torch.tensor(pts_np); faces = torch.tensor(faces_np)
                K = CellComplex.from_triangulation(pts, faces).to(device)
                x_norm = torch.tensor((x_raw - xm) / xs, dtype=torch.float32).to(device)
                if model_name == 'ours':
                    lift_shape = state['lifting.node_mlp.0.weight'].shape
                    C = lift_shape[0]
                    m = GaugeHodgeNetwork(
                        f_in=f_in, C=C, n_layers=4,
                        n0=K.n0, n1=K.n1, n2=K.n2,
                        task='scalar', spatial_dim=3, out_dim=out_dim,
                        mp_hidden=16, metric_type='local_rich', metric_rank=8)
                else:
                    h = cam.infer_hidden(state)
                    if h is None: break
                    cls = {'gcn': GCNBaseline, 'gat': GATBaseline,
                           'schnet': SchNetBaseline, 'egnn': EGNNBaseline}.get(model_name)
                    if cls is None: break
                    m = cls(f_in=f_in, hidden=h, n_layers=4, out_dim=out_dim, task='node')
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
            preds_norm = (preds_raw - ym) / ys
            tgts_norm = (tgts_raw - ym) / ys
            out[model_name] = (preds_raw, tgts_raw, preds_norm, tgts_norm)
            print(f"  {model_name:15s} inferred {len(preds_raw)} samples", flush=True)
        except Exception as e:
            print(f"  {model_name}: ERROR {str(e)[:120]}", flush=True)
    return out


# ------------------------------------------------------------
# Bootstrap stage
# ------------------------------------------------------------

def bootstrap_from_preds(preds_raw, tgts_raw, preds_norm, tgts_norm,
                          n_boot=1000, seed=0):
    """Return dict: metric -> (mean, std) from bootstrap resampling."""
    N = len(preds_raw)
    rng = np.random.default_rng(seed)

    # data_range fixed from full test set (property of the data)
    data_range = float(tgts_raw.max() - tgts_raw.min())
    if data_range < 1e-8:
        data_range = 1.0

    # Pre-compute per-sample SSIM and Pearson (expensive), then bootstrap-average
    ssim_per, pearson_per, _ = _precompute_per_sample_ssim_pearson(preds_raw, tgts_raw)

    r2s = np.empty(n_boot); mses = np.empty(n_boot); maes = np.empty(n_boot)
    nrmses = np.empty(n_boot); ssims = np.empty(n_boot); pears = np.empty(n_boot)

    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        p_n = preds_norm[idx]; t_n = tgts_norm[idx]
        p_r = preds_raw[idx];  t_r = tgts_raw[idx]
        r2, mse, mae = _r2_mse_mae(p_n, t_n)
        nrmse = _nrmse(p_r, t_r, data_range)
        ssim = float(ssim_per[idx].mean())
        pear = float(pearson_per[idx].mean())
        r2s[b] = r2; mses[b] = mse; maes[b] = mae
        nrmses[b] = nrmse; ssims[b] = ssim; pears[b] = pear

    def _ms(arr):
        return {'mean': float(arr.mean()), 'std': float(arr.std(ddof=1)),
                'ci95_lo': float(np.quantile(arr, 0.025)),
                'ci95_hi': float(np.quantile(arr, 0.975))}
    return {
        'R2': _ms(r2s), 'MSE': _ms(mses), 'MAE': _ms(maes),
        'NRMSE': _ms(nrmses), 'SSIM': _ms(ssims), 'Pearson': _ms(pears),
        'n_test': int(N), 'n_boot': int(n_boot),
    }


# ------------------------------------------------------------
# Driver
# ------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', type=str, default=None)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--n-eval', type=int, default=100)
    ap.add_argument('--n-boot', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', type=str, default='checkpoints/bootstrap_metrics.json')
    ap.add_argument('--cache-dir', type=str, default='checkpoints/bootstrap_cache')
    args = ap.parse_args()

    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    os.makedirs(args.cache_dir, exist_ok=True)

    TASKS = {
        'T1': ('T1_cns_vorticity', 'datasets/T1_cns_vorticity.pkl'),
        'T2': ('T2_torus_advection_diffusion', 'datasets/T2_torus_advection_diffusion.pkl'),
        'T5': ('T5_maxwell_poisson', 'datasets/T5_maxwell_poisson.pkl'),
        'T6': ('T6_wilson_loop', 'datasets/T6_wilson_loop.pkl'),
        'T7': ('T7_yang_mills_su2', 'datasets/T7_yang_mills_su2.pkl'),
        'T8': ('T8_airfoil_pressure', 'datasets/T8_airfoil_pressure_persample.pkl'),
    }
    T3_DS_MAP = {
        'T3_ellipsoid_surface_flow': 'datasets/T3_ellipsoid_coexact.pkl',
        'T3_ellipsoid_coexact': 'datasets/T3_ellipsoid_coexact.pkl',
    }
    for d in os.listdir('checkpoints'):
        if d.startswith('T3_'):
            ds = T3_DS_MAP.get(d, f'datasets/{d}.pkl')
            if os.path.exists(ds):
                TASKS[f'T3_{d}'] = (d, ds)

    if args.task:
        TASKS = {args.task: TASKS[args.task]}

    all_results = {}
    for tid, (tname, ds_path) in sorted(TASKS.items()):
        cache_path = os.path.join(args.cache_dir, f'{tid}.npz')
        if os.path.exists(cache_path):
            print(f"[cache hit] {tid} -> {cache_path}", flush=True)
            z = np.load(cache_path, allow_pickle=True)
            preds = {k: z[k].item() for k in z.files}
        else:
            if tid == 'T8':
                preds = _run_inference_task_persample(tname, ds_path, args.device, args.n_eval)
            else:
                preds = _run_inference_task(tname, ds_path, args.device, args.n_eval)
            np.savez_compressed(cache_path, **{k: np.array(v, dtype=object) for k, v in preds.items()})
            print(f"[cache save] {tid} -> {cache_path}", flush=True)

        task_boot = {}
        for mname, tup in preds.items():
            preds_raw, tgts_raw, preds_norm, tgts_norm = tup
            boot = bootstrap_from_preds(preds_raw, tgts_raw, preds_norm, tgts_norm,
                                         n_boot=args.n_boot, seed=args.seed)
            task_boot[mname] = boot
            print(f"  [{tid}] {mname:15s} SSIM={boot['SSIM']['mean']:.3f}±{boot['SSIM']['std']:.3f}  "
                  f"Pearson={boot['Pearson']['mean']:.3f}±{boot['Pearson']['std']:.3f}  "
                  f"NRMSE={boot['NRMSE']['mean']:.4f}±{boot['NRMSE']['std']:.4f}", flush=True)
        all_results[tid] = task_boot

    def _conv(o):
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        raise TypeError(str(type(o)))
    with open(args.out, 'w') as f:
        json.dump(all_results, f, indent=2, default=_conv)
    print(f"\nSaved: {args.out}")


if __name__ == '__main__':
    main()
