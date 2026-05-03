"""
Generate small-mesh Yang-Mills SU(2) (256-node Delaunay) for metric-type ablation.
Same physics as T7: F = dA + [A,A], with strong coupling.
Output: datasets/T7_small_yang_mills.pkl
"""
import os, sys, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np, torch
from scipy.spatial import Delaunay

device = 'cuda' if torch.cuda.is_available() else 'cpu'
np.random.seed(42); torch.manual_seed(42)

n_pts = 256
pts = np.random.rand(n_pts, 2).astype(np.float64)
tri = Delaunay(pts)
faces = tri.simplices.astype(np.int64)

edge_pairs = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
edge_pairs = np.sort(edge_pairs, axis=1)
edges, inv = np.unique(edge_pairs, axis=0, return_inverse=True)
n_edges = len(edges); n_faces = len(faces)
face_edge_idx = inv.reshape(3, n_faces).T.astype(np.int64)
face_edge_sign = np.zeros((n_faces, 3), dtype=np.float32)
for k in range(3):
    face_edge_sign[:, k] = np.where(faces[:, k] < faces[:, (k+1) % 3], 1.0, -1.0)

edge_dir = pts[edges[:, 1]] - pts[edges[:, 0]]
edge_len = np.linalg.norm(edge_dir, axis=1, keepdims=True) + 1e-12
edge_dir_norm = (edge_dir / edge_len).astype(np.float32)

print(f"Mesh: nodes={n_pts}, edges={n_edges}, faces={n_faces}")

pts_t = torch.tensor(pts, dtype=torch.float32, device=device)
edges_t = torch.tensor(edges, device=device)
fei_t = torch.tensor(face_edge_idx, device=device)
fes_t = torch.tensor(face_edge_sign, device=device)

N = 5000
A_edges = torch.randn(N, n_edges, 3, device=device) * 0.5
for comp in range(3):
    for _ in range(4):
        cx = torch.rand(N, 1, device=device); cy = torch.rand(N, 1, device=device)
        sigma = torch.rand(N, 1, device=device) * 0.25 + 0.1
        amp = torch.randn(N, 1, device=device) * 2.0
        d2 = (pts_t[None, :, 0] - cx) ** 2 + (pts_t[None, :, 1] - cy) ** 2
        theta = amp * torch.exp(-d2 / (2 * sigma ** 2))
        A_edges[:, :, comp] += theta[:, edges_t[:, 1]] - theta[:, edges_t[:, 0]]

A_f = []
for k in range(3):
    A_f.append(A_edges[:, fei_t[:, k]] * fes_t[None, :, k:k+1])
dA = A_f[0] + A_f[1] + A_f[2]
comm = (torch.cross(A_f[0], A_f[1], dim=-1) +
        torch.cross(A_f[1], A_f[2], dim=-1) +
        torch.cross(A_f[2], A_f[0], dim=-1))
F_faces = dA + comm

# Encode (vectorized scatter)
e0 = edges_t[:, 0]; e1 = edges_t[:, 1]
dx = torch.tensor(edge_dir_norm[:, 0], device=device)
dy = torch.tensor(edge_dir_norm[:, 1], device=device)
deg = torch.zeros(n_pts, device=device)
deg.scatter_add_(0, e0, torch.ones_like(e0, dtype=torch.float32))
deg.scatter_add_(0, e1, torch.ones_like(e1, dtype=torch.float32))
deg.clamp_(min=1.0)

CHUNK = 250
X_chunks, Y_chunks = [], []
fdeg = torch.zeros(n_pts, device=device)
for k in range(3):
    fk = torch.tensor(faces[:, k], device=device, dtype=torch.long)
    fdeg.scatter_add_(0, fk, torch.ones_like(fk, dtype=torch.float32))
fdeg.clamp_(min=1.0)
faces_d = torch.tensor(faces, device=device, dtype=torch.long)

for ci in range(0, N, CHUNK):
    n = min(CHUNK, N - ci)
    A = A_edges[ci:ci+n]            # (n, n_edges, 3)
    F = F_faces[ci:ci+n]            # (n, n_faces, 3)
    # edges -> nodes (avg, avg*dx, avg*dy) per channel
    feat = torch.zeros(n, n_pts, 9, device=device)
    for ch in range(3):
        a = A[..., ch]
        for s, w in [(torch.ones_like(a), 0), (dx[None].expand_as(a) * a, 3), (dy[None].expand_as(a) * a, 6)]:
            pass
    # simpler: scatter
    avg = torch.zeros(n, n_pts, 3, device=device)
    adx = torch.zeros(n, n_pts, 3, device=device)
    ady = torch.zeros(n, n_pts, 3, device=device)
    for ch in range(3):
        a = A[..., ch]
        avg[..., ch].scatter_add_(1, e0.unsqueeze(0).expand(n, -1), a)
        avg[..., ch].scatter_add_(1, e1.unsqueeze(0).expand(n, -1), a)
        adx[..., ch].scatter_add_(1, e0.unsqueeze(0).expand(n, -1), a * dx)
        adx[..., ch].scatter_add_(1, e1.unsqueeze(0).expand(n, -1), a * dx)
        ady[..., ch].scatter_add_(1, e0.unsqueeze(0).expand(n, -1), a * dy)
        ady[..., ch].scatter_add_(1, e1.unsqueeze(0).expand(n, -1), a * dy)
    avg /= deg[None, :, None]; adx /= deg[None, :, None]; ady /= deg[None, :, None]
    X_chunks.append(torch.cat([avg, adx, ady], dim=-1).cpu().numpy().astype(np.float32))

    # faces -> nodes
    yavg = torch.zeros(n, n_pts, 3, device=device)
    for k in range(3):
        fk = faces_d[:, k]
        yavg.scatter_add_(1, fk.view(1, -1, 1).expand(n, -1, 3), F)
    yavg /= fdeg[None, :, None]
    Y_chunks.append(yavg.cpu().numpy().astype(np.float32))

X = np.concatenate(X_chunks, 0); Y = np.concatenate(Y_chunks, 0)
print(f"X{X.shape} std={X.std():.3f}, Y{Y.shape} std={Y.std():.3f}")

pts_3d = np.column_stack([pts.astype(np.float32), np.zeros(n_pts, dtype=np.float32)])
out = {
    'points': pts_3d, 'faces': faces, 'edge_index': edges,
    'X_data': X, 'Y_data': Y,
    'n_nodes': int(n_pts), 'n_samples': int(N),
    'task': 'su2_field_strength_small',
    'topology': f'irregular_delaunay_{n_pts}',
    'physics': 'Yang-Mills SU(2): A(nodes,9ch)→F(nodes,3ch), 256-node mesh for metric ablation.',
}
out_path = os.path.join(os.path.dirname(__file__), 'T7_small_yang_mills.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(out, f, protocol=4)
print(f"Saved {out_path}: {os.path.getsize(out_path)/1e6:.1f} MB")
