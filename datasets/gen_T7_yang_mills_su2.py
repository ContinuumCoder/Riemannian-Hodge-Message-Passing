"""
Yang-Mills v3: weaker A → dA dominates over [A,A] → more like Wilson loop → bigger gap.
Also try F² (gauge invariant scalar) as alternative output.

CUDA_VISIBLE_DEVICES=0 python3 -u experiments/gen_yang_mills_v3.py
"""
import sys; sys.path.insert(0, '/home/dzheng/GaugeStructuredHodgeMP_NIPS')
import torch, numpy as np, pickle, os, time
from scipy.spatial import Delaunay

device = 'cuda' if torch.cuda.is_available() else 'cpu'

np.random.seed(42)
n_pts = 1024
pts = np.random.rand(n_pts, 2).astype(np.float64)
tri = Delaunay(pts)
faces = tri.simplices.astype(np.int64)
edges_set = set()
for f in faces:
    for i in range(3):
        edges_set.add(tuple(sorted([f[i], f[(i+1)%3]])))
edges = np.array(sorted(edges_set), dtype=np.int64)
n_edges = len(edges)
n_faces = len(faces)

edge_to_idx = {}
for ei, (a, b) in enumerate(edges):
    edge_to_idx[(a, b)] = ei
    edge_to_idx[(b, a)] = ei

face_edge_idx = np.zeros((n_faces, 3), dtype=np.int64)
face_edge_sign = np.zeros((n_faces, 3), dtype=np.float32)
for fi, f in enumerate(faces):
    for k in range(3):
        a, b = f[k], f[(k+1)%3]
        key = tuple(sorted([a, b]))
        face_edge_idx[fi, k] = edge_to_idx[key]
        face_edge_sign[fi, k] = 1.0 if (a, b) == key else -1.0

edge_dir = pts[edges[:, 1]] - pts[edges[:, 0]]
edge_len = np.linalg.norm(edge_dir, axis=1, keepdims=True) + 1e-12
edge_dir_norm = edge_dir / edge_len

pts_t = torch.tensor(pts, dtype=torch.float32, device=device)
edges_t = torch.tensor(edges, device=device)
face_edge_idx_t = torch.tensor(face_edge_idx, device=device)
face_edge_sign_t = torch.tensor(face_edge_sign, device=device)

print(f"Mesh: {n_pts} nodes, {n_edges} edges, {n_faces} faces")

def encode_edges_to_nodes(A_edges, n_ch=1):
    N = A_edges.shape[0]
    avg = torch.zeros(N, n_pts, n_ch, device=device)
    avg_dx = torch.zeros(N, n_pts, n_ch, device=device)
    avg_dy = torch.zeros(N, n_pts, n_ch, device=device)
    count = torch.zeros(n_pts, device=device)
    dx = torch.tensor(edge_dir_norm[:, 0], dtype=torch.float32, device=device)
    dy = torch.tensor(edge_dir_norm[:, 1], dtype=torch.float32, device=device)
    for ei in range(n_edges):
        a, b = edges[ei]
        val = A_edges[:, ei]
        if val.ndim == 1: val = val.unsqueeze(-1)
        avg[:, a] += val; avg[:, b] += val
        avg_dx[:, a] += val * dx[ei]; avg_dx[:, b] += val * dx[ei]
        avg_dy[:, a] += val * dy[ei]; avg_dy[:, b] += val * dy[ei]
        count[a] += 1; count[b] += 1
    count[count == 0] = 1
    avg /= count[None, :, None]
    avg_dx /= count[None, :, None]
    avg_dy /= count[None, :, None]
    return torch.cat([avg, avg_dx, avg_dy], dim=-1)

def encode_faces_to_nodes(F_faces, n_ch=1):
    N = F_faces.shape[0]
    avg = torch.zeros(N, n_pts, n_ch, device=device)
    count = torch.zeros(n_pts, device=device)
    for fi in range(n_faces):
        val = F_faces[:, fi]
        if val.ndim == 1: val = val.unsqueeze(-1)
        for v in faces[fi]:
            avg[:, v] += val; count[v] += 1
    count[count == 0] = 1
    return avg / count[None, :, None]

def save_dataset(name, X, Y, task, physics):
    outdir = f'data/{name}'
    os.makedirs(outdir, exist_ok=True)
    pts_3d = np.column_stack([pts, np.zeros(n_pts)])
    d = {'points': pts_3d, 'faces': faces, 'edge_index': edges,
         'X_data': X.cpu().numpy(), 'Y_data': Y.cpu().numpy(),
         'n_nodes': n_pts, 'n_samples': len(X),
         'task': task, 'topology': 'irregular_delaunay_1024', 'physics': physics}
    path = os.path.join(outdir, f'{name}_dataset.pkl')
    with open(path, 'wb') as f:
        pickle.dump(d, f)
    print(f"  Saved {name}: {os.path.getsize(path)/1e6:.1f}MB")
    print(f"    X: {X.shape} std={X.std():.4f}, Y: {Y.shape} std={Y.std():.4f}")

# ==========================================
# Yang-Mills v3: WEAK A → dA ≫ [A,A]
# ==========================================
print("\n=== Yang-Mills v3 (weak coupling, dA dominates) ===")
t0 = time.time()
N = 10000
torch.manual_seed(42)

# SMALL A → [A,A] ~ A² ≪ dA ~ A
A_edges = torch.randn(N, n_edges, 3, device=device) * 0.1  # small noise

# Smooth structure with MODERATE amplitude
for comp in range(3):
    for _ in range(4):
        cx = torch.rand(N, 1, device=device)
        cy = torch.rand(N, 1, device=device)
        sigma = torch.rand(N, 1, device=device) * 0.25 + 0.1
        amp = torch.randn(N, 1, device=device) * 0.5  # moderate (v2 was 2.0)
        d2 = (pts_t[None, :, 0] - cx)**2 + (pts_t[None, :, 1] - cy)**2
        theta = amp * torch.exp(-d2 / (2 * sigma**2))
        A_edges[:, :, comp] += theta[:, edges_t[:, 1]] - theta[:, edges_t[:, 0]]

# Add vortex-like structures (non-trivial curvature)
for comp in range(3):
    for _ in range(2):
        cx = torch.rand(N, 1, device=device)
        cy = torch.rand(N, 1, device=device)
        strength = torch.randn(N, 1, device=device) * 0.3
        edge_mid_x = (pts_t[edges_t[:, 0], 0] + pts_t[edges_t[:, 1], 0]) / 2
        edge_mid_y = (pts_t[edges_t[:, 0], 1] + pts_t[edges_t[:, 1], 1]) / 2
        rx = edge_mid_x[None, :] - cx
        ry = edge_mid_y[None, :] - cy
        r2 = rx**2 + ry**2 + 0.01
        dx_e = torch.tensor(edge_dir[:, 0], dtype=torch.float32, device=device)
        dy_e = torch.tensor(edge_dir[:, 1], dtype=torch.float32, device=device)
        A_edges[:, :, comp] += strength * (-ry * dx_e[None, :] + rx * dy_e[None, :]) / r2

# F = dA + [A,A]
A_f = []
for k in range(3):
    A_f.append(A_edges[:, face_edge_idx_t[:, k]] * face_edge_sign_t[None, :, k:k+1])
dA = A_f[0] + A_f[1] + A_f[2]
comm = (torch.cross(A_f[0], A_f[1], dim=-1) +
        torch.cross(A_f[1], A_f[2], dim=-1) +
        torch.cross(A_f[2], A_f[0], dim=-1))
F_faces = dA + comm

# Check ratio
dA_norm = dA.norm(dim=-1).mean().item()
comm_norm = comm.norm(dim=-1).mean().item()
print(f"  |dA| = {dA_norm:.4f}, |[A,A]| = {comm_norm:.4f}, ratio = {dA_norm/max(comm_norm,1e-8):.1f}x")

X_ym = encode_edges_to_nodes(A_edges.reshape(N, n_edges, 3), n_ch=3)
Y_ym = encode_faces_to_nodes(F_faces, n_ch=3)
print(f"  Generated in {time.time()-t0:.1f}s")
save_dataset('yang_mills_v3', X_ym.float(), Y_ym.float(),
    'su2_field_strength_weak',
    'Yang-Mills v3: weak A → dA≫[A,A], curl-dominated, irregular mesh')

# ==========================================
# Yang-Mills F²: gauge-invariant scalar output
# ==========================================
print("\n=== Yang-Mills F² (gauge invariant scalar) ===")
F2_faces = (F_faces ** 2).sum(dim=-1)  # (N, n_faces) = Tr(F²)
Y_f2 = encode_faces_to_nodes(F2_faces, n_ch=1)
save_dataset('yang_mills_f2', X_ym.float(), Y_f2.float(),
    'su2_action_density',
    'Yang-Mills F²: A(nodes)→|F|²(nodes), gauge invariant action density')

print("\nDone!")
