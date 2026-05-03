"""
Generate T1: CNS Vorticity dataset from PDEBench.
Task: (ρ, Vx, Vy, p)_t → curl(V)_{t+1} (vorticity prediction)
Source: PDEBench 2D Compressible Navier-Stokes M=0.1

Usage:
  python3 datasets/gen_T1_cns_vorticity.py
Output:
  datasets/T1_cns_vorticity.pkl
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, '/home/dzheng/GaugeStructuredHodgeMP_NIPS')
import numpy as np, h5py, pickle, time, torch
from src.pdebench.dataset_pdebench import build_regular_grid

print("Loading CNS M=0.1 from PDEBench...")
fpath = '/home/dzheng/GaugeStructuredHodgeMP_NIPS/data/pdebench/2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5'
with h5py.File(fpath, 'r') as f:
    Vx = f['Vx'][:500]
    Vy = f['Vy'][:500]
    rho = f['density'][:500]
    pres = f['pressure'][:500]

# Subsample 128→32
gs = 32; step = 128 // gs; ix = np.arange(0, 128, step)[:gs]
Vx = Vx[:, :, ix][:, :, :, ix]
Vy = Vy[:, :, ix][:, :, :, ix]
rho = rho[:, :, ix][:, :, :, ix]
pres = pres[:, :, ix][:, :, :, ix]

n = gs * gs
pos_np, faces_np = build_regular_grid(gs, gs)
dx = 1.0 / gs

# Build samples: (ρ,Vx,Vy,p)_t → vorticity_{t+1}
print("Building samples...")
t0 = time.time()
ins, tgs = [], []
for s in range(500):
    for t in range(min(20, Vx.shape[1] - 1)):
        inp = np.stack([rho[s, t], Vx[s, t], Vy[s, t], pres[s, t]], -1).reshape(n, 4)
        # Vorticity at t+1 (periodic finite difference, matching CNS periodic BC)
        vx_next = Vx[s, t + 1]
        vy_next = Vy[s, t + 1]
        dvy_dx = (np.roll(vy_next, -1, axis=0) - np.roll(vy_next, 1, axis=0)) / (2 * dx)
        dvx_dy = (np.roll(vx_next, -1, axis=1) - np.roll(vx_next, 1, axis=1)) / (2 * dx)
        omega = (dvy_dx - dvx_dy).reshape(n, 1)
        ins.append(torch.tensor(inp, dtype=torch.float32))
        tgs.append(torch.tensor(omega, dtype=torch.float32))

X = torch.stack(ins)
Y = torch.stack(tgs)
print(f"  X: {X.shape}, Y: {Y.shape} ({time.time()-t0:.1f}s)")
print(f"  X std={X.std():.4f}, Y std={Y.std():.4f}")

dataset = {
    'points': pos_np, 'faces': faces_np,
    'X_data': X.numpy(), 'Y_data': Y.numpy(),
    'n_nodes': n, 'n_samples': len(X), 'grid_size': gs,
    'task': 'cns_vorticity_prediction',
    'topology': 'regular_grid_32x32',
    'physics': 'CNS: (ρ,Vx,Vy,p)_t → ω_{t+1} = curl(V)_{t+1}, compressible Navier-Stokes M=0.1',
}
outpath = os.path.join(os.path.dirname(__file__), 'T1_cns_vorticity.pkl')
with open(outpath, 'wb') as f:
    pickle.dump(dataset, f)
print(f"Saved: {outpath} ({os.path.getsize(outpath)/1e6:.1f} MB)")
