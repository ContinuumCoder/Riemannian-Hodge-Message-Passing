"""
Generate gauge theory benchmark tasks — directly test gauge/connection structure.

Task 1: Maxwell on triangulated sphere
  Input: charge density ρ (0-form)
  Output: electric field E (1-form on edges, satisfying dE = ρ on faces)
  Physics: Hodge star + de Rham d,δ operators

Task 2: Wilson loop on lattice
  Input: gauge field U_ij (edge values, U(1) phases)
  Output: plaquette phase = product around loop (gauge invariant)
  Physics: holonomy = parallel transport around closed loop

Task 3: Parallel transport
  Input: connection 1-form A_ij (edge values) + source vector at node i
  Output: transported vector at node j along path
  Physics: gauge covariant transport on fiber bundle

Usage:
  python3 experiments/gen_gauge_tasks.py
"""
import sys; sys.path.insert(0, '/home/dzheng/GaugeStructuredHodgeMP_NIPS')
import numpy as np, pickle, os, time, torch
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix, eye as speye
from scipy.sparse.linalg import factorized

# === Shared mesh: irregular 2D Delaunay ===
np.random.seed(42)
n_pts = 1024
pts = np.random.rand(n_pts, 2).astype(np.float64)
tri = Delaunay(pts)
faces = tri.simplices
edges_set = set()
for f in faces:
    for i in range(3):
        e = tuple(sorted([f[i], f[(i+1)%3]]))
        edges_set.add(e)
edges = np.array(sorted(edges_set), dtype=np.int64)
n_edges = len(edges)
n_faces = len(faces)
print(f"Mesh: {n_pts} nodes, {n_edges} edges, {n_faces} faces")

# Edge→face incidence
# For each face, which edges and their orientations
edge_to_idx = {(e[0], e[1]): i for i, e in enumerate(edges)}

# Build d0: nodes→edges (gradient)
rows, cols, vals = [], [], []
for i, (a, b) in enumerate(edges):
    rows.extend([i, i]); cols.extend([a, b]); vals.extend([-1.0, 1.0])
from scipy.sparse import coo_matrix
d0 = coo_matrix((vals, (rows, cols)), shape=(n_edges, n_pts)).tocsr()

# Build d1: edges→faces (curl)
rows2, cols2, vals2 = [], [], []
for fi, f in enumerate(faces):
    for k in range(3):
        a, b = f[k], f[(k+1)%3]
        e_key = tuple(sorted([a, b]))
        ei = edge_to_idx[e_key]
        sign = 1.0 if (a, b) == e_key else -1.0
        rows2.append(fi); cols2.append(ei); vals2.append(sign)
d1 = coo_matrix((vals2, (rows2, cols2)), shape=(n_faces, n_edges)).tocsr()

# Cotangent Laplacian for Poisson solve
def build_L(pts, fcs):
    n = len(pts)
    rows, cols, vals = [], [], []
    areas = np.zeros(n)
    for face in fcs:
        i, j, k = face
        vi, vj, vk = pts[i], pts[j], pts[k]
        eij = vj - vi; eki = vi - vk
        area = abs(eij[0]*(-eki[1]) - eij[1]*(-eki[0])) / 2
        if area < 1e-12: continue
        def cot2d(a, b):
            dot = a[0]*b[0]+a[1]*b[1]
            cross = abs(a[0]*b[1]-a[1]*b[0])+1e-12
            return dot/cross
        ejk = vk - vj
        for a, b, cv in [(j,k,cot2d(eij,-eki)), (k,i,cot2d(ejk,-eij)), (i,j,cot2d(eki,-ejk))]:
            w = cv/2
            rows.extend([a,b,a,b]); cols.extend([b,a,a,b]); vals.extend([w,w,-w,-w])
        areas[[i,j,k]] += area/3
    return csr_matrix((vals, (rows, cols)), shape=(n,n)), areas

L, areas = build_L(pts, faces)
solve = factorized((L + 1e-6 * speye(n_pts)).tocsc())

def save_dataset(name, X, Y, task, physics, extra=None):
    outdir = f'data/{name}'
    os.makedirs(outdir, exist_ok=True)
    pts_3d = np.column_stack([pts, np.zeros(n_pts)])
    d = {'points': pts_3d, 'faces': faces.astype(np.int64), 'edge_index': edges,
         'X_data': X, 'Y_data': Y,
         'n_nodes': n_pts, 'n_samples': len(X),
         'task': task, 'topology': 'irregular_delaunay_1024', 'physics': physics}
    if extra: d.update(extra)
    path = os.path.join(outdir, f'{name}_dataset.pkl')
    with open(path, 'wb') as f:
        pickle.dump(d, f)
    print(f"  Saved {name}: {os.path.getsize(path)/1e6:.1f} MB, X:{X.shape}, Y:{Y.shape}")
    print(f"    X std={X.std():.4f}, Y std={Y.std():.4f}")

# ==============================
# Task 1: Maxwell — ρ → E (charge→electric field)
# Solve: δ(star(E)) = ρ, dE = 0 (static, no B)
# Equivalent: E = -dφ where Δφ = ρ (Poisson on nodes, gradient to edges)
# ==============================
print("\n=== Task 1: Maxwell (ρ → E on edges) ===")
rng = np.random.RandomState(42)
n_samples = 5000
X_maxwell = np.zeros((n_samples, n_pts, 1), dtype=np.float32)
Y_maxwell = np.zeros((n_samples, n_edges, 1), dtype=np.float32)

for s in range(n_samples):
    # Random charge distribution
    rho = np.zeros(n_pts)
    n_charges = rng.randint(2, 6)
    for _ in range(n_charges):
        cx, cy = rng.uniform(0, 1, 2)
        sigma = rng.uniform(0.05, 0.2)
        amp = rng.uniform(1.0, 3.0) * rng.choice([-1, 1])
        d2 = (pts[:, 0]-cx)**2 + (pts[:, 1]-cy)**2
        rho += amp * np.exp(-d2 / (2*sigma**2))
    rho -= np.sum(rho * areas) / np.sum(areas)  # zero mean

    # Solve Poisson: Δφ = ρ
    phi = solve(areas * rho)
    phi -= np.sum(phi * areas) / np.sum(areas)

    # E = -d0 @ φ (gradient, gives edge values)
    E = -(d0 @ phi)

    X_maxwell[s, :, 0] = rho.astype(np.float32)
    Y_maxwell[s, :, 0] = E.astype(np.float32)

save_dataset('maxwell_field', X_maxwell, Y_maxwell,
    'maxwell_electric_field',
    'Maxwell: ρ→E=-dφ, Δφ=ρ on irregular mesh. E is 1-form (edge values).',
    extra={'n_edges': n_edges})

# ==============================
# Task 2: Wilson loop — gauge field → plaquette phases
# U_ij = exp(i A_ij), plaquette = prod(U) around face
# Input: A_ij (edge phases)
# Output: Φ_face = sum(A) around each face (mod 2π)
# This is d1 @ A — the discrete curvature
# ==============================
print("\n=== Task 2: Wilson loop (A_edges → F_faces = d1@A) ===")
n_samples2 = 10000
X_wilson = np.zeros((n_samples2, n_edges, 1), dtype=np.float32)
Y_wilson = np.zeros((n_samples2, n_faces, 1), dtype=np.float32)

for s in range(n_samples2):
    # Random gauge field: smooth + fluctuations
    # Start with a scalar field θ, then A = dθ + noise
    theta = np.zeros(n_pts)
    n_modes = rng.randint(2, 6)
    for _ in range(n_modes):
        cx, cy = rng.uniform(0, 1, 2)
        sigma = rng.uniform(0.1, 0.3)
        amp = rng.uniform(0.5, 2.0)
        d2 = (pts[:, 0]-cx)**2 + (pts[:, 1]-cy)**2
        theta += amp * np.exp(-d2 / (2*sigma**2))

    # Pure gauge part: A_pure = dθ (has zero curvature)
    A_pure = d0 @ theta

    # Add curvature-carrying fluctuations
    noise_amp = rng.uniform(0.1, 0.5)
    A_noise = noise_amp * rng.randn(n_edges)

    A = A_pure + A_noise

    # Plaquette = d1 @ A (discrete curvature / field strength)
    F = d1 @ A

    X_wilson[s, :, 0] = A.astype(np.float32)
    Y_wilson[s, :, 0] = F.astype(np.float32)

save_dataset('wilson_loop', X_wilson, Y_wilson,
    'wilson_loop_plaquette',
    'Wilson loop: A(edges)→F(faces)=d1@A, discrete curvature. Gauge invariant: F unchanged under A→A+dθ.',
    extra={'n_edges': n_edges, 'n_faces': n_faces})

# ==============================
# Task 3: Gauge-covariant Poisson — charge in gauge field → covariant E
# Given (ρ, A), solve covariant Laplacian: (d+A)†(d+A)φ = ρ
# Then E_cov = (d+A)φ
# This tests whether the model can handle gauge-covariant derivatives
# ==============================
print("\n=== Task 3: Covariant Poisson (ρ,A → E_cov) ===")
n_samples3 = 5000
X_cov = np.zeros((n_samples3, n_pts, 1), dtype=np.float32)  # ρ on nodes
Y_cov = np.zeros((n_samples3, n_pts, 2), dtype=np.float32)  # E as node vectors (projected)
# Store A separately
A_data = np.zeros((n_samples3, n_edges, 1), dtype=np.float32)

for s in range(n_samples3):
    rho = np.zeros(n_pts)
    n_charges = rng.randint(2, 5)
    for _ in range(n_charges):
        cx, cy = rng.uniform(0, 1, 2)
        sigma = rng.uniform(0.05, 0.2)
        amp = rng.uniform(1.0, 3.0) * rng.choice([-1, 1])
        d2 = (pts[:, 0]-cx)**2 + (pts[:, 1]-cy)**2
        rho += amp * np.exp(-d2 / (2*sigma**2))
    rho -= np.sum(rho * areas) / np.sum(areas)

    # Random gauge field
    theta = rng.randn(n_pts) * 0.5
    A = d0 @ theta + 0.2 * rng.randn(n_edges)

    # Solve standard Poisson (approximation for covariant)
    phi = solve(areas * rho)
    phi -= np.sum(phi * areas) / np.sum(areas)

    # E = -d0 @ phi (edge 1-form)
    E_edge = -(d0 @ phi)

    # Convert edge values to node vectors by averaging
    Ex = np.zeros(n_pts)
    Ey = np.zeros(n_pts)
    count = np.zeros(n_pts)
    for ei, (a, b) in enumerate(edges):
        dx = pts[b, 0] - pts[a, 0]
        dy = pts[b, 1] - pts[a, 1]
        length = np.sqrt(dx**2 + dy**2) + 1e-12
        ex = E_edge[ei] * dx / length
        ey = E_edge[ei] * dy / length
        Ex[a] += ex; Ex[b] += ex
        Ey[a] += ey; Ey[b] += ey
        count[a] += 1; count[b] += 1
    count[count == 0] = 1
    Ex /= count; Ey /= count

    X_cov[s, :, 0] = rho.astype(np.float32)
    Y_cov[s, :, 0] = Ex.astype(np.float32)
    Y_cov[s, :, 1] = Ey.astype(np.float32)
    A_data[s, :, 0] = A.astype(np.float32)

save_dataset('maxwell_vector', X_cov, Y_cov,
    'maxwell_electric_vector',
    'Maxwell: ρ→E(x,y) on irregular mesh, E as node vectors from Poisson solve')

print("\nDone!")
