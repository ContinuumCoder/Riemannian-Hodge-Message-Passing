"""
Generate ellipsoid curl field dataset: ω → v_coexact (scalar → vector)
Physics: v = n × ∇(L⁻¹ω) = pure coexact component of velocity
This is EXACTLY what Hodge decomposition isolates — our inductive bias.

On genus-0 ellipsoid there's no harmonic, so:
  any tangent vector field = exact (∇φ) + coexact (n×∇ψ)
  Given ω, the coexact part is uniquely determined: v_coex = n×∇(L⁻¹ω)
  Baselines have no notion of this decomposition.

Usage:
  python3 experiments/gen_ellipsoid_curl_field.py
Output:
  data/ellipsoid_curl/ellipsoid_curl_dataset.pkl
"""
import sys; sys.path.insert(0, '/home/dzheng/GaugeStructuredHodgeMP_NIPS')
import pickle, numpy as np, os, time
from scipy.sparse import csr_matrix, eye as speye
from scipy.sparse.linalg import factorized


def build_L(pts, fcs):
    n = len(pts)
    rows, cols, vals = [], [], []
    areas = np.zeros(n)
    for face in fcs:
        i, j, k = face
        vi, vj, vk = pts[i], pts[j], pts[k]
        eij, ejk, eki = vj - vi, vk - vj, vi - vk
        area = np.linalg.norm(np.cross(eij, -eki)) / 2
        if area < 1e-12: continue
        def cot(a, b):
            c = np.clip(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12), -0.999, 0.999)
            return c / (np.sqrt(1-c**2)+1e-12)
        for a, b, cv in [(j,k,cot(eij,-eki)), (k,i,cot(ejk,-eij)), (i,j,cot(eki,-ejk))]:
            w = cv/2
            rows.extend([a,b,a,b]); cols.extend([b,a,a,b]); vals.extend([w,w,-w,-w])
        areas[[i,j,k]] += area/3
    return csr_matrix((vals, (rows, cols)), shape=(n,n)), areas


print("Loading ellipsoid mesh...")
with open('/home/dzheng/icml_rebuttal_unified/data_ellipsoid_helmholtz/ellipsoid_helmholtz_dataset.pkl', 'rb') as f:
    d = pickle.load(f)
pts = np.array(d['points'], dtype=np.float64)
fcs = np.array(d['faces'], dtype=np.int64)
n = len(pts)
print(f"  {n} nodes, {len(fcs)} faces")

# Normals
normals = np.zeros((n, 3))
for face in fcs:
    i, j, k = face
    nm = np.cross(pts[j] - pts[i], pts[k] - pts[i])
    normals[i] += nm; normals[j] += nm; normals[k] += nm
normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)

# Laplacian + Poisson solver (pure, no kappa — cleanest coexact)
L, areas = build_L(pts, fcs)
total_area = areas.sum()
solve = factorized((L + 1e-6 * speye(n)).tocsc())

# Gradient operator setup
fi, fj, fk = fcs[:, 0], fcs[:, 1], fcs[:, 2]
e1 = pts[fj] - pts[fi]; e2 = pts[fk] - pts[fi]
fn = np.cross(e1, e2)
fa2 = np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
fn /= fa2
cr_kj = np.cross(fn, pts[fk] - pts[fj])
cr_ik = np.cross(fn, pts[fi] - pts[fk])
cr_ji = np.cross(fn, pts[fj] - pts[fi])
nc = np.zeros(n)
np.add.at(nc, fi, 1); np.add.at(nc, fj, 1); np.add.at(nc, fk, 1)
nc[nc == 0] = 1

edges = set()
for face in fcs:
    for i in range(3):
        edges.add(tuple(sorted([face[i], face[(i+1)%3]])))
edge_index = np.array(list(edges), dtype=np.int64)

# Generate: simple monopole sources → clean coexact fields
n_samples = 10000
rng = np.random.RandomState(42)
X_data = np.zeros((n_samples, n), dtype=np.float32)
Y_data = np.zeros((n_samples, n, 3), dtype=np.float32)

print(f"Generating {n_samples} samples (ω → v_coexact = n×∇ψ)...")
t0 = time.time()
for s in range(n_samples):
    omega = np.zeros(n)

    # Simple monopole sources (no pairs → clean, strong signal)
    n_sources = rng.randint(2, 5)
    for _ in range(n_sources):
        center = pts[rng.randint(0, n)]
        sigma = rng.uniform(0.08, 0.25)
        strength = rng.uniform(1.0, 4.0) * rng.choice([-1, 1])
        dd = np.linalg.norm(pts - center, axis=1)
        omega += strength * np.exp(-dd**2 / (2*sigma**2))

    omega -= np.sum(omega * areas) / total_area
    X_data[s] = omega

    # Solve Poisson: Lψ = Mω
    psi = solve(areas * omega)
    psi -= np.sum(psi * areas) / total_area

    # v_coexact = n × ∇ψ (pure coexact component)
    pi, pj, pk = psi[fi], psi[fj], psi[fk]
    gt = (pi[:, None] * cr_kj + pj[:, None] * cr_ik + pk[:, None] * cr_ji) / fa2
    grad = np.zeros((n, 3))
    for dd in range(3):
        np.add.at(grad[:, dd], fi, gt[:, dd])
        np.add.at(grad[:, dd], fj, gt[:, dd])
        np.add.at(grad[:, dd], fk, gt[:, dd])
    grad /= nc[:, None]
    v = np.cross(normals, grad)
    v -= np.sum(v * normals, axis=1, keepdims=True) * normals

    # DON'T normalize per-sample — keep physical scale
    Y_data[s] = v.astype(np.float32)

    if (s+1) % 2000 == 0:
        print(f"  {s+1}/{n_samples} ({time.time()-t0:.1f}s)")

print(f"Done: {time.time()-t0:.1f}s")
print(f"X (ω): mean={X_data.mean():.4f} std={X_data.std():.4f}")
print(f"Y (v_coexact): mean={Y_data.mean():.4f} std={Y_data.std():.4f}")

dataset = {
    'points': pts, 'faces': fcs, 'edge_index': edge_index,
    'X_data': X_data, 'Y_data': Y_data,
    'n_nodes': n, 'n_samples': n_samples,
    'task': 'coexact_velocity_from_vorticity',
    'topology': 'ellipsoid_genus0',
    'physics': 'Poisson: Lpsi=omega, v=n×grad(psi) = pure coexact field',
}

outdir = 'data/ellipsoid_curl'
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, 'ellipsoid_curl_dataset.pkl')
with open(outpath, 'wb') as f:
    pickle.dump(dataset, f)
print(f"Saved: {outpath} ({os.path.getsize(outpath)/1e6:.1f} MB)")
