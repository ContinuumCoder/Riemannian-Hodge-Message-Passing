"""Debug T6_100K NaN: run single fwd/bwd, locate NaN source."""
import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import torch, torch.nn.functional as F
from gauge_hodge_mp.cell_complex import CellComplex
from gauge_hodge_mp.network import GaugeHodgeNetwork

device = 'cuda'
torch.manual_seed(42)

with open(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'T6_100K_wilson_loop.pkl'), 'rb') as f:
    d = pickle.load(f)
pts = torch.tensor(d['points'], dtype=torch.float32)
faces = torch.tensor(d['faces'], dtype=torch.long)
X = torch.tensor(d['X_data'], dtype=torch.float32)
Y = torch.tensor(d['Y_data'], dtype=torch.float32)

print(f"X stats: mean={X.mean():.3f} std={X.std():.3f} min={X.min():.3f} max={X.max():.3f} any_nan={torch.isnan(X).any().item()} any_inf={torch.isinf(X).any().item()}")
print(f"Y stats: mean={Y.mean():.3f} std={Y.std():.3f} min={Y.min():.3f} max={Y.max():.3f} any_nan={torch.isnan(Y).any().item()} any_inf={torch.isinf(Y).any().item()}")

xm = X.mean(dim=(0, 1), keepdim=True); xs = X.std(dim=(0, 1), keepdim=True) + 1e-6
ym = Y.mean(dim=(0, 1), keepdim=True); ys = Y.std(dim=(0, 1), keepdim=True) + 1e-6
Xn = (X - xm) / xs; Yn = (Y - ym) / ys

K = CellComplex.from_triangulation(pts, faces, device='cpu').to(device)
print(f"K: n0={K.n0}, n1={K.n1}, n2={K.n2}")

model = GaugeHodgeNetwork(
    f_in=3, C=16, n_layers=3, n0=K.n0, n1=K.n1, n2=K.n2,
    task='scalar', spatial_dim=2, out_dim=1, mp_hidden=16,
    metric_type='diagonal', metric_rank=8,
).to(device)

# Hook every named module: log if NaN/Inf in output
hits = []
def hook(name):
    def fn(mod, inp, out):
        if isinstance(out, torch.Tensor):
            t = out
        elif isinstance(out, (tuple, list)):
            t = next((x for x in out if isinstance(x, torch.Tensor)), None)
        else:
            return
        if t is None: return
        nan = torch.isnan(t).any().item()
        inf = torch.isinf(t).any().item()
        mx = float(t.detach().abs().max())
        if nan or inf or mx > 1e4:
            hits.append((name, nan, inf, mx, tuple(t.shape)))
    return fn

for name, m in model.named_modules():
    m.register_forward_hook(hook(name))

x = Xn[0].to(device); y = Yn[0].to(device)
print(f"\nInput x: mean={x.mean():.3f} std={x.std():.3f} any_nan={torch.isnan(x).any().item()}")

torch.autograd.set_detect_anomaly(True)
out = model(x, K)
print(f"\nOutput: shape={tuple(out.shape)} mean={float(out.detach().mean()):.3f} std={float(out.detach().std()):.3f} any_nan={torch.isnan(out).any().item()}")

print("\n--- Hooks: NaN/Inf or |out|>1e4 ---")
for h in hits[:30]:
    print(f"  {h[0]:<60} nan={h[1]} inf={h[2]} max={h[3]:.2e} shape={h[4]}")

# Backward
loss = F.mse_loss(out, y)
print(f"\nLoss: {float(loss):.4f} (nan={torch.isnan(loss).item()})")
loss.backward()

# Check grads
gnan = 0; ginf = 0; gmax = 0
for n, p in model.named_parameters():
    if p.grad is None: continue
    if torch.isnan(p.grad).any(): gnan += 1
    if torch.isinf(p.grad).any(): ginf += 1
    gmax = max(gmax, float(p.grad.abs().max()))
print(f"\nGrads: NaN-params={gnan} Inf-params={ginf} max_abs={gmax:.2e}")
