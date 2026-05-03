"""
Generate T6_100K: Wilson loop on a ~100K-cell Delaunay mesh.
Same physics as T6 (smooth pure-gauge + magnetic vortex defects), larger mesh
for scalability accuracy comparison.

Output: datasets/T6_100K_wilson_loop.pkl
"""
import sys, os, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np, torch
from scipy.spatial import Delaunay

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Mesh: target ~100K faces ===
np.random.seed(42)
n_pts = 50010  # Delaunay 2D gives ~2× faces vs points
pts = np.random.rand(n_pts, 2).astype(np.float64)
print(f"Triangulating {n_pts} points...")
t0 = time.time()
tri = Delaunay(pts)
faces = tri.simplices.astype(np.int64)
print(f"  Delaunay done in {time.time()-t0:.1f}s, {len(faces)} faces")

t0 = time.time()
# Vectorized edge extraction
edge_pairs = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
edge_pairs = np.sort(edge_pairs, axis=1)
edges, inv = np.unique(edge_pairs, axis=0, return_inverse=True)
n_edges = len(edges)
n_faces = len(faces)
print(f"  edges built in {time.time()-t0:.1f}s ({n_edges} edges)")

# face_edge_idx[fi, k] = edge index for k-th side of face fi (sorted-key edge)
face_edge_idx = inv.reshape(3, n_faces).T.astype(np.int64)  # (n_faces, 3)

# Sign: +1 if face's directed edge (a,b) matches sorted key, else -1
face_edge_sign = np.zeros((n_faces, 3), dtype=np.float32)
for k in range(3):
    a = faces[:, k]
    b = faces[:, (k + 1) % 3]
    face_edge_sign[:, k] = np.where(a < b, 1.0, -1.0)

# Edge direction (for node encoding)
edge_dir = pts[edges[:, 1]] - pts[edges[:, 0]]
edge_len = np.linalg.norm(edge_dir, axis=1, keepdims=True) + 1e-12
edge_dir_norm = (edge_dir / edge_len).astype(np.float32)

pts_t = torch.tensor(pts, dtype=torch.float32, device=device)
edges_t = torch.tensor(edges, device=device)
face_edge_idx_t = torch.tensor(face_edge_idx, device=device)
face_edge_sign_t = torch.tensor(face_edge_sign, device=device)
edge_dir_t = torch.tensor(edge_dir_norm, device=device)
edge_mid_t = ((pts_t[edges_t[:, 0]] + pts_t[edges_t[:, 1]]) / 2)  # (n_edges, 2)

print(f"Mesh: nodes={n_pts}, edges={n_edges}, faces={n_faces}")

# ==========================================
# Generate samples in chunks (memory-friendly)
# ==========================================
N = 500
torch.manual_seed(42)
print(f"\nGenerating {N} samples on 100K mesh...")
t0 = time.time()

CHUNK = 25  # samples per chunk
theta_all_cpu = []
plaq_all_cpu = []

for ci in range(0, N, CHUNK):
    n = min(CHUNK, N - ci)
    theta_edges = torch.zeros(n, n_edges, device=device)

    # Smooth pure-gauge part: dθ from 5 random Gaussians
    for _ in range(5):
        cx = torch.rand(n, 1, device=device)
        cy = torch.rand(n, 1, device=device)
        sigma = torch.rand(n, 1, device=device) * 0.3 + 0.1
        amp = torch.randn(n, 1, device=device) * 1.0
        d2 = (pts_t[None, :, 0] - cx) ** 2 + (pts_t[None, :, 1] - cy) ** 2
        theta_node = amp * torch.exp(-d2 / (2 * sigma ** 2))
        theta_edges += theta_node[:, edges_t[:, 1]] - theta_node[:, edges_t[:, 0]]
        del theta_node, d2

    # Vortex defects (vectorized over edges)
    n_vortices = torch.randint(1, 4, (n,), device=device)
    for v in range(3):
        mask = (n_vortices > v).float()
        cx = torch.rand(n, 1, device=device)
        cy = torch.rand(n, 1, device=device)
        strength = (torch.randn(n, 1, device=device) * 2.0) * mask.unsqueeze(-1)
        rx = edge_mid_t[None, :, 0] - cx           # (n, n_edges)
        ry = edge_mid_t[None, :, 1] - cy
        r2 = rx * rx + ry * ry + 0.01
        contrib = strength * (-ry * edge_dir_t[None, :, 0] + rx * edge_dir_t[None, :, 1]) / r2
        theta_edges += contrib
        del rx, ry, r2, contrib

    plaq = (theta_edges[:, face_edge_idx_t[:, 0]] * face_edge_sign_t[None, :, 0] +
            theta_edges[:, face_edge_idx_t[:, 1]] * face_edge_sign_t[None, :, 1] +
            theta_edges[:, face_edge_idx_t[:, 2]] * face_edge_sign_t[None, :, 2])

    theta_all_cpu.append(theta_edges.cpu().numpy().astype(np.float32))
    plaq_all_cpu.append(plaq.cpu().numpy().astype(np.float32))
    del theta_edges, plaq
    torch.cuda.empty_cache()
    print(f"  chunk [{ci}:{ci+n}] done")

theta_edges_np = np.concatenate(theta_all_cpu, axis=0)
plaq_np = np.concatenate(plaq_all_cpu, axis=0)
print(f"  generated in {time.time()-t0:.1f}s")
print(f"  theta_edges: {theta_edges_np.shape} std={theta_edges_np.std():.3f}")
print(f"  plaq:        {plaq_np.shape} std={plaq_np.std():.3f}")

# ==========================================
# Encode edge values → node features (avg, avg*dx, avg*dy) using scatter
# Encode face values → node features (avg) using scatter
# ==========================================
print("Encoding to node features (vectorized)...")
t0 = time.time()

theta_t = torch.tensor(theta_edges_np, device=device)        # (N, n_edges)
plaq_t = torch.tensor(plaq_np, device=device)                # (N, n_faces)
e0 = edges_t[:, 0]; e1 = edges_t[:, 1]
dx = edge_dir_t[:, 0]; dy = edge_dir_t[:, 1]

deg = torch.zeros(n_pts, device=device)
deg.scatter_add_(0, e0, torch.ones_like(e0, dtype=torch.float32))
deg.scatter_add_(0, e1, torch.ones_like(e1, dtype=torch.float32))
deg.clamp_(min=1.0)

# Face degree (per-node count of incident faces) — computed ONCE outside the chunk loop
fdeg = torch.zeros(n_pts, device=device)
faces_d = torch.tensor(faces, device=device, dtype=torch.long)
for k in range(3):
    fdeg.scatter_add_(0, faces_d[:, k], torch.ones(n_faces, device=device))
fdeg.clamp_(min=1.0)

X_chunks = []
Y_chunks = []
for ci in range(0, N, CHUNK):
    n = min(CHUNK, N - ci)
    th = theta_t[ci:ci+n]                                    # (n, n_edges)
    avg = torch.zeros(n, n_pts, device=device)
    adx = torch.zeros(n, n_pts, device=device)
    ady = torch.zeros(n, n_pts, device=device)
    avg.scatter_add_(1, e0.unsqueeze(0).expand(n, -1), th)
    avg.scatter_add_(1, e1.unsqueeze(0).expand(n, -1), th)
    adx.scatter_add_(1, e0.unsqueeze(0).expand(n, -1), th * dx)
    adx.scatter_add_(1, e1.unsqueeze(0).expand(n, -1), th * dx)
    ady.scatter_add_(1, e0.unsqueeze(0).expand(n, -1), th * dy)
    ady.scatter_add_(1, e1.unsqueeze(0).expand(n, -1), th * dy)
    avg /= deg[None, :]; adx /= deg[None, :]; ady /= deg[None, :]
    X_chunks.append(torch.stack([avg, adx, ady], dim=-1).cpu().numpy().astype(np.float32))

    # Faces → nodes
    pl = plaq_t[ci:ci+n]
    yavg = torch.zeros(n, n_pts, device=device)
    for k in range(3):
        fk = faces_d[:, k]
        yavg.scatter_add_(1, fk.unsqueeze(0).expand(n, -1), pl)
    yavg /= fdeg[None, :]
    Y_chunks.append(yavg.unsqueeze(-1).cpu().numpy().astype(np.float32))
    del th, avg, adx, ady, pl, yavg
    torch.cuda.empty_cache()

X = np.concatenate(X_chunks, axis=0)  # (N, n_pts, 3)
Y = np.concatenate(Y_chunks, axis=0)  # (N, n_pts, 1)
print(f"  encode done in {time.time()-t0:.1f}s, X{X.shape} Y{Y.shape}")

# ==========================================
# Save
# ==========================================
pts_3d = np.column_stack([pts.astype(np.float32), np.zeros(n_pts, dtype=np.float32)])
out = {
    'points': pts_3d,
    'faces': faces.astype(np.int64),
    'edge_index': edges.astype(np.int64),
    'X_data': X,
    'Y_data': Y,
    'n_nodes': int(n_pts),
    'n_samples': int(N),
    'task': 'wilson_loop_vortex_100K',
    'topology': f'irregular_delaunay_{n_pts}',
    'physics': 'Wilson loop on 100K-face Delaunay mesh; θ(edges)→plaquette flux; magnetic vortex defects.',
}
out_path = os.path.join(os.path.dirname(__file__), 'T6_100K_wilson_loop.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(out, f, protocol=4)
print(f"\nSaved {out_path}: {os.path.getsize(out_path)/1e6:.1f} MB")
