"""
Pre-process AirfRANS into per-sample pickle: each sample has (x, y_pressure, y_vorticity, K).
Save as a list so no re-loading from VTP needed.

Usage:
  python3 experiments/gen_airfrans_persample.py
Output:
  data/airfrans_processed/airfrans_1000_pressure.pkl
  data/airfrans_processed/airfrans_1000_vorticity.pkl
"""
import sys; sys.path.insert(0, '/home/dzheng/GaugeStructuredHodgeMP_NIPS')
import torch, warnings, os, numpy as np, time, pickle
warnings.filterwarnings('ignore')
from gauge_hodge_mp.cell_complex import CellComplex
from scipy.spatial import Delaunay
import airfrans as af

N_PTS = 2000
root = 'data/airfrans/Dataset'
names = sorted(os.listdir(root))[:1000]
rng = np.random.RandomState(42)

print(f"Processing {len(names)} AirfRANS samples...")
t0 = time.time()

samples = []
for ni, name in enumerate(names):
    data = af.Simulation(root=root, name=name)
    pos = np.array(data.position, dtype=np.float32)
    vel = np.array(data.velocity, dtype=np.float32)
    sdf = np.array(data.sdf, dtype=np.float32)
    pres = np.array(data.pressure, dtype=np.float32)
    idx = rng.choice(len(pos), N_PTS, replace=False)

    tri = Delaunay(pos[idx])
    pts_3d = np.column_stack([pos[idx], np.zeros((N_PTS, 1))]).astype(np.float32)
    faces = tri.simplices.astype(np.int64)

    # Input features for pressure task: (sdf, inlet_vel, AoA)
    x_pres = np.column_stack([
        sdf[idx],
        np.full((N_PTS, 1), float(data.inlet_velocity), dtype=np.float32),
        np.full((N_PTS, 1), float(data.angle_of_attack), dtype=np.float32)
    ])

    # Input features for vorticity task: (Vx, Vy, sdf, p, inlet_vel, AoA)
    x_vort = np.column_stack([
        vel[idx], sdf[idx], pres[idx],
        np.full((N_PTS, 1), float(data.inlet_velocity), dtype=np.float32),
        np.full((N_PTS, 1), float(data.angle_of_attack), dtype=np.float32)
    ])

    # Compute vorticity on mesh
    vx_n, vy_n = vel[idx, 0], vel[idx, 1]
    pts_2d = pos[idx]
    omega = np.zeros(N_PTS, dtype=np.float32)
    count = np.zeros(N_PTS)
    for face in faces:
        i, j, k = face
        e1 = pts_2d[j] - pts_2d[i]; e2 = pts_2d[k] - pts_2d[i]
        a2 = e1[0]*e2[1] - e1[1]*e2[0]
        if abs(a2) < 1e-12: continue
        dvy = np.array([vy_n[j]-vy_n[i], vy_n[k]-vy_n[i]])
        dvx = np.array([vx_n[j]-vx_n[i], vx_n[k]-vx_n[i]])
        w = (dvy[0]*e2[1]-dvy[1]*e1[1])/a2 - (-dvx[0]*e2[0]+dvx[1]*e1[0])/a2
        for v in [i,j,k]: omega[v] += w; count[v] += 1
    count[count==0] = 1; omega /= count

    samples.append({
        'pts_3d': pts_3d,
        'faces': faces,
        'x_pres': x_pres,
        'x_vort': x_vort,
        'y_pres': pres[idx].reshape(N_PTS, 1),
        'y_vort': omega.reshape(N_PTS, 1),
    })

    if (ni+1) % 100 == 0:
        print(f"  {ni+1}/{len(names)} ({time.time()-t0:.1f}s)")

print(f"Done: {len(samples)} samples ({time.time()-t0:.1f}s)")

outdir = 'data/airfrans_processed'
os.makedirs(outdir, exist_ok=True)
path = os.path.join(outdir, 'airfrans_1000_persample.pkl')
with open(path, 'wb') as f:
    pickle.dump(samples, f)
print(f"Saved: {path} ({os.path.getsize(path)/1e6:.1f} MB)")
