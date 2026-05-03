"""
Quick 100K-cell accuracy comparison on Wilson loop:
  ours (C=16) vs GAT vs EGNN
500 samples (200 train / 100 val / 200 test), 30 epochs.
"""
import sys, os, time, pickle, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, torch, torch.nn.functional as F

from gauge_hodge_mp.cell_complex import CellComplex
from gauge_hodge_mp.network import GaugeHodgeNetwork
from baselines_advanced import CWNetBaseline

device = 'cuda'
torch.manual_seed(42); np.random.seed(42)

# === Load ===
print("Loading T6_100K...")
with open(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'T6_100K_wilson_loop.pkl'), 'rb') as f:
    d = pickle.load(f)
pts = torch.tensor(d['points'], dtype=torch.float32, device=device)
faces = torch.tensor(d['faces'], dtype=torch.long, device=device)
X = torch.tensor(d['X_data'], dtype=torch.float32)   # (N, n_pts, 3)
Y = torch.tensor(d['Y_data'], dtype=torch.float32)   # (N, n_pts, 1)
N, n_pts, f_in = X.shape
out_dim = Y.shape[-1]
print(f"  N={N}, n_pts={n_pts}, f_in={f_in}, out_dim={out_dim}")

# Normalize per-channel
xm = X.mean(dim=(0, 1), keepdim=True); xs = X.std(dim=(0, 1), keepdim=True) + 1e-6
ym = Y.mean(dim=(0, 1), keepdim=True); ys = Y.std(dim=(0, 1), keepdim=True) + 1e-6
Xn = (X - xm) / xs
Yn = (Y - ym) / ys

# Split
perm = torch.randperm(N, generator=torch.Generator().manual_seed(42))
tr = perm[:200]; va = perm[200:300]; te = perm[300:500]
Xtr, Ytr = Xn[tr].to(device), Yn[tr].to(device)
Xva, Yva = Xn[va].to(device), Yn[va].to(device)
Xte, Yte = Xn[te].to(device), Yn[te].to(device)
Yte_raw = Y[te].to(device)  # for un-normalized R²/NRMSE

# Cell complex (shared, on device)
print("Building CellComplex...")
t0 = time.time()
K = CellComplex.from_triangulation(pts.cpu(), faces.cpu(), device='cpu')
K = K.to(device)
print(f"  done in {time.time()-t0:.1f}s. n0={K.n0}, n1={K.n1}, n2={K.n2}")

# === Models ===
# CWNet (Bodnar et al., ICLR 2022): cell-complex MP with face-level structure,
# the architecturally-closest non-ours baseline for plaquette-curl tasks.
# hidden=512, n_layers=6 → ~11.6M params, matching ours (10.8M).
def build_models():
    return {
        'ours':   GaugeHodgeNetwork(
            f_in=f_in, C=16, n_layers=3,
            n0=K.n0, n1=K.n1, n2=K.n2,
            task='scalar', spatial_dim=2, out_dim=out_dim,
            mp_hidden=16, metric_type='diagonal', metric_rank=8,
        ).to(device),
        'cw_net': CWNetBaseline(f_in=f_in, hidden=512, n_layers=6, out_dim=out_dim, task='node').to(device),
    }

def n_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

# === Train / eval ===
def run(name, model, n_epochs=30, bsz=2, lr=1e-3):
    print(f"\n--- {name} ({n_params(model)/1e6:.2f}M params) ---")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr*0.01)

    Xtr_, Xva_, Xte_ = Xtr, Xva, Xte

    def fwd_batch(model, x_batch):
        if hasattr(model, 'forward_batch'):
            return model.forward_batch(x_batch, K)
        return torch.stack([model(x_batch[i], K) for i in range(x_batch.shape[0])], dim=0)

    best_va = -1e9; best_state = None
    n_train = Xtr_.shape[0]
    t_start = time.time()
    for ep in range(n_epochs):
        model.train()
        idx = torch.randperm(n_train)
        ep_loss = 0; ep_n = 0
        for i in range(0, n_train, bsz):
            sel = idx[i:i+bsz]
            x = Xtr_[sel]; y = Ytr[sel]
            opt.zero_grad()
            pred = fwd_batch(model, x)
            loss = F.mse_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * len(sel); ep_n += len(sel)
        sched.step()

        # val (R² in normalized space)
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, Xva_.shape[0], bsz):
                preds.append(fwd_batch(model, Xva_[i:i+bsz]))
            pv = torch.cat(preds, 0)
            ssr = ((pv - Yva) ** 2).sum().item()
            sst = ((Yva - Yva.mean()) ** 2).sum().item()
            r2_va = 1 - ssr / sst
        if r2_va > best_va:
            best_va = r2_va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (ep+1) % 5 == 0 or ep == 0:
            print(f"  ep {ep+1:3d}: train_mse={ep_loss/ep_n:.4f} val_R²={r2_va:.4f} best={best_va:.4f} ({time.time()-t_start:.0f}s)")

    # Load best, eval test in original units
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, Xte_.shape[0], bsz):
            preds.append(fwd_batch(model, Xte_[i:i+bsz]))
        pt = torch.cat(preds, 0)
        # de-normalize
        pt_raw = pt * ys.to(device) + ym.to(device)
        ssr = ((pt_raw - Yte_raw) ** 2).sum().item()
        sst = ((Yte_raw - Yte_raw.mean()) ** 2).sum().item()
        r2_te = 1 - ssr / sst
        rmse = ((pt_raw - Yte_raw) ** 2).mean().sqrt().item()
        nrmse = rmse / (Yte_raw.max() - Yte_raw.min()).item()
    return {'name': name, 'params_M': n_params(model)/1e6,
            'val_R2': best_va, 'test_R2': r2_te,
            'test_RMSE': rmse, 'test_NRMSE': nrmse,
            'wall_time_s': time.time() - t_start}

results = {}
for name, model in build_models().items():
    try:
        results[name] = run(name, model, n_epochs=30, bsz=2)
    except RuntimeError as e:
        print(f"  {name} FAILED: {e}")
        results[name] = {'name': name, 'error': str(e)[:200]}
    torch.cuda.empty_cache()

# Save
out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'T6_100K_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print(f"{'Model':<10}{'Params(M)':>12}{'Test R²':>12}{'Test NRMSE':>14}{'Wall(s)':>10}")
print("-" * 70)
for r in results.values():
    if 'error' in r:
        print(f"{r['name']:<10}{'FAILED':>12}{r.get('error', '')[:40]}")
    else:
        print(f"{r['name']:<10}{r['params_M']:>12.2f}{r['test_R2']:>12.4f}{r['test_NRMSE']:>14.4f}{r['wall_time_s']:>10.0f}")
print("=" * 70)
print(f"\nSaved: {os.path.join(out_dir, 'T6_100K_results.json')}")
