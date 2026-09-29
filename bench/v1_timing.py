"""Train-step / inference timing and batch-coupling check of the original v1 implementation (vendored in
``rhmp.baselines.v1``) on the T6, T7 and T6_100K meshes.

    python3 bench/v1_timing.py        # CUDA GPU; needs the v1 pickles in datasets/
"""
import sys, os, time, pickle
import numpy as np, torch, torch.nn.functional as F
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rhmp.baselines.v1.gauge_hodge_mp.cell_complex import CellComplex
from rhmp.baselines.v1.gauge_hodge_mp.network import GaugeHodgeNetwork
dev = 'cuda'

def load(name, n=256):
    d = pickle.load(open(os.path.join(ROOT, 'datasets', name), 'rb'))
    pts = torch.tensor(np.array(d['points'], dtype=np.float32)); faces = torch.tensor(np.array(d['faces'], dtype=np.int64))
    t0 = time.time(); K = CellComplex.from_triangulation(pts, faces).to(dev); tk = time.time() - t0
    X = torch.tensor(np.array(d['X_data'][:n], dtype=np.float32)); Y = torch.tensor(np.array(d['Y_data'][:n], dtype=np.float32))
    if X.ndim == 2: X = X.unsqueeze(-1)
    if Y.ndim == 2: Y = Y.unsqueeze(-1)
    print(f"[{name}] complex build {tk:.1f}s")
    return K, X.to(dev), Y.to(dev)

def bench(K, X, Y, C, L, B, steps=20, tag=''):
    m = GaugeHodgeNetwork(f_in=X.shape[-1], C=C, n_layers=L, n0=K.n0, n1=K.n1, n2=K.n2, task='scalar',
                          spatial_dim=2, out_dim=Y.shape[-1], mp_hidden=16, metric_type='diagonal', metric_rank=8).to(dev)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    nparam = sum(p.numel() for p in m.parameters())
    torch.cuda.reset_peak_memory_stats()
    def step():
        opt.zero_grad(); loss = F.mse_loss(m.forward_batch(X[:B], K), Y[:B]); loss.backward(); opt.step()
    for _ in range(3): step()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(steps): step()
    torch.cuda.synchronize(); dt = (time.time() - t0) / steps
    mem = torch.cuda.max_memory_allocated() / 2**30
    m.eval()
    with torch.no_grad():
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(steps): m.forward_batch(X[:B], K)
        torch.cuda.synchronize(); dti = (time.time() - t0) / steps
        rel = lambda a, b: ((a - b).norm() / b.norm()).item()
        p_batch = m.forward_batch(X[:B], K)[0]
        p_single = m.forward_batch(X[:1], K)[0]
        coupling = rel(p_single, p_batch)
        if 2 * B - 1 <= X.shape[0]:
            p_other = m.forward_batch(torch.cat([X[:1], X[B:2 * B - 1]]), K)[0]
            coupling2 = rel(p_other, p_batch)
        else:
            coupling2 = float('nan')
    print(f"{tag} n0={K.n0} n1={K.n1} n2={K.n2} C={C} L={L} B={B} params={nparam/1e6:.2f}M | "
          f"train step {dt*1000:.1f} ms ({dt*1000/B:.2f} ms/sample) | infer {dti*1000:.1f} ms | peak {mem:.2f} GB | "
          f"batch-coupling rel.diff: single-vs-batch {coupling:.3e}, other-batch {coupling2:.3e}", flush=True)

if __name__ == '__main__':
    K, X, Y = load('T6_wilson_loop.pkl')
    bench(K, X, Y, 128, 4, 64, tag='T6')
    bench(K, X, Y, 128, 4, 1, tag='T6')
    K, X, Y = load('T7_yang_mills_su2.pkl', n=128)
    bench(K, X, Y, 160, 4, 64, tag='T7')
    K, X, Y = load('T6_100K_wilson_loop.pkl', n=8)
    bench(K, X, Y, 16, 3, 2, steps=5, tag='T6_100K')
