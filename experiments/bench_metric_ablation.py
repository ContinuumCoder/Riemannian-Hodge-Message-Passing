"""
Metric-type ablation on small Yang-Mills SU(2) (256 nodes / 749 edges / 494 faces).
Compares: scalar / diagonal (rank=0) / diagonal (rank=8) / diagonal (rank=16) / full Cholesky.
No external baselines — only ours-variants.
"""
import sys, os, time, pickle, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, torch, torch.nn.functional as F

from gauge_hodge_mp.cell_complex import CellComplex
from gauge_hodge_mp.network import GaugeHodgeNetwork

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from compute_all_metrics import compute_ssim_pearson

device = 'cuda'
torch.manual_seed(42); np.random.seed(42)

print("Loading T7_small...")
with open(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'T7_small_yang_mills.pkl'), 'rb') as f:
    d = pickle.load(f)
pts = torch.tensor(d['points'], dtype=torch.float32)
faces = torch.tensor(d['faces'], dtype=torch.long)
X = torch.tensor(d['X_data'], dtype=torch.float32)
Y = torch.tensor(d['Y_data'], dtype=torch.float32)
N, n_pts, f_in = X.shape; out_dim = Y.shape[-1]
print(f"  N={N}, n_pts={n_pts}, f_in={f_in}, out_dim={out_dim}")

xm = X.mean(dim=(0, 1), keepdim=True); xs = X.std(dim=(0, 1), keepdim=True) + 1e-6
ym = Y.mean(dim=(0, 1), keepdim=True); ys = Y.std(dim=(0, 1), keepdim=True) + 1e-6
Xn = (X - xm) / xs; Yn = (Y - ym) / ys

perm = torch.randperm(N, generator=torch.Generator().manual_seed(42))
n_tr, n_va, n_te = 1000, 200, 500
tr = perm[:n_tr]; va = perm[n_tr:n_tr+n_va]; te = perm[n_tr+n_va:n_tr+n_va+n_te]
Xtr, Ytr = Xn[tr].to(device), Yn[tr].to(device)
Xva, Yva = Xn[va].to(device), Yn[va].to(device)
Xte, Yte = Xn[te].to(device), Yn[te].to(device)
Yte_raw = Y[te].to(device)

K = CellComplex.from_triangulation(pts, faces, device='cpu').to(device)
print(f"K: n0={K.n0}, n1={K.n1}, n2={K.n2}")

VARIANTS = [
    ('scalar',         dict(metric_type='scalar',   metric_rank=0)),
    ('diagonal_rank0', dict(metric_type='diagonal', metric_rank=0)),
    ('diagonal_rank8', dict(metric_type='diagonal', metric_rank=8)),
    ('diagonal_rank16',dict(metric_type='diagonal', metric_rank=16)),
    ('full_cholesky',  dict(metric_type='full',     metric_rank=0)),
]

C = 32; n_layers = 3; mp_hidden = 16

def n_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def run(name, kw, n_epochs=40, bsz=16, lr=1e-3):
    torch.manual_seed(42)
    model = GaugeHodgeNetwork(
        f_in=f_in, C=C, n_layers=n_layers,
        n0=K.n0, n1=K.n1, n2=K.n2,
        task='scalar', spatial_dim=2, out_dim=out_dim,
        mp_hidden=mp_hidden, **kw,
    ).to(device)
    np_ = n_params(model)
    print(f"\n--- {name}  ({np_/1e6:.3f}M params) ---")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr*0.01)

    def fwd(x):
        if hasattr(model, 'forward_batch'):
            return model.forward_batch(x, K)
        return torch.stack([model(x[i], K) for i in range(x.shape[0])], dim=0)

    best_va = -1e9; best_state = None
    t0 = time.time()
    for ep in range(n_epochs):
        model.train()
        idx = torch.randperm(n_tr)
        ep_loss = 0; ep_n = 0
        for i in range(0, n_tr, bsz):
            sel = idx[i:i+bsz]
            x = Xtr[sel]; y = Ytr[sel]
            opt.zero_grad()
            pred = fwd(x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(sel); ep_n += len(sel)
        sched.step()
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, n_va, bsz):
                preds.append(fwd(Xva[i:i+bsz]))
            pv = torch.cat(preds, 0)
            ssr = ((pv - Yva) ** 2).sum().item()
            sst = ((Yva - Yva.mean()) ** 2).sum().item()
            r2_va = 1 - ssr / sst
        if r2_va > best_va:
            best_va = r2_va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (ep+1) % 5 == 0 or ep == 0:
            print(f"  ep {ep+1:3d}: train={ep_loss/ep_n:.4f} val_R²={r2_va:.4f} best={best_va:.4f} ({time.time()-t0:.0f}s)")

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, n_te, bsz):
            preds.append(fwd(Xte[i:i+bsz]))
        pt = torch.cat(preds, 0)
        pt_raw = pt * ys.to(device) + ym.to(device)
        ssr = ((pt_raw - Yte_raw) ** 2).sum().item()
        sst = ((Yte_raw - Yte_raw.mean()) ** 2).sum().item()
        r2_te = 1 - ssr / sst
        rmse = ((pt_raw - Yte_raw) ** 2).mean().sqrt().item()
        nrmse = rmse / (Yte_raw.max() - Yte_raw.min()).item()
        ssim, pearson = compute_ssim_pearson(pt_raw.cpu().numpy(), Yte_raw.cpu().numpy())
    return {'name': name, 'params_M': np_/1e6,
            'val_R2': best_va, 'test_R2': r2_te,
            'test_RMSE': rmse, 'test_NRMSE': nrmse,
            'test_SSIM': ssim, 'test_Pearson': pearson,
            'wall_s': time.time() - t0}

results = {}
for name, kw in VARIANTS:
    try:
        results[name] = run(name, kw)
    except RuntimeError as e:
        print(f"  {name} FAILED: {str(e)[:200]}")
        results[name] = {'name': name, 'error': str(e)[:200]}
    torch.cuda.empty_cache()

out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'metric_ablation.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 104)
print(f"{'Variant':<18}{'Params(M)':>11}{'Val R²':>10}{'Test R²':>10}{'NRMSE':>10}{'SSIM':>10}{'Pearson':>10}{'Wall(s)':>10}")
print("-" * 104)
for r in results.values():
    if 'error' in r:
        print(f"{r['name']:<18}{'FAIL':>11}{'  ' + r.get('error','')[:50]}")
    else:
        print(f"{r['name']:<18}{r['params_M']:>11.3f}{r['val_R2']:>10.4f}{r['test_R2']:>10.4f}"
              f"{r['test_NRMSE']:>10.4f}{r['test_SSIM']:>10.4f}{r['test_Pearson']:>10.4f}{r['wall_s']:>10.0f}")
print("=" * 104)
