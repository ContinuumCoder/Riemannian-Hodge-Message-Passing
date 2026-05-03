"""
Gauge theory tasks v2: smoother connection, directional node encoding, stronger signal.

Task 1: Wilson loop v2 — magnetic vortex defects
  Connection is mostly pure gauge (smooth), with localized magnetic flux insertions.
  Node features encode directional edge info (not just average).

Task 2: Yang-Mills v2 — stronger non-abelian coupling
  Larger A amplitude makes [A,A] commutator term significant.
  Better node encoding.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 -u experiments/gen_gauge_v2.py
"""
import sys; sys.path.insert(0, '/home/dzheng/GaugeStructuredHodgeMP_NIPS')
import torch, numpy as np, pickle, os, time
from scipy.spatial import Delaunay

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Mesh ===
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
        ei = edge_to_idx[key]
        sign = 1.0 if (a, b) == key else -1.0
        face_edge_idx[fi, k] = ei
        face_edge_sign[fi, k] = sign

# Edge direction vectors
edge_dir = pts[edges[:, 1]] - pts[edges[:, 0]]  # (n_edges, 2)
edge_len = np.linalg.norm(edge_dir, axis=1, keepdims=True) + 1e-12
edge_dir_norm = edge_dir / edge_len  # unit direction

pts_t = torch.tensor(pts, dtype=torch.float32, device=device)
edges_t = torch.tensor(edges, device=device)
face_edge_idx_t = torch.tensor(face_edge_idx, device=device)
face_edge_sign_t = torch.tensor(face_edge_sign, device=device)

print(f"Mesh: {n_pts} nodes, {n_edges} edges, {n_faces} faces")

def encode_edges_to_nodes(A_edges, n_ch=1):
    """
    Encode edge values to node features with directional information.
    For each node: (avg_A, avg_A*dx, avg_A*dy, max_A, min_A)
    Returns (N, n_pts, 5*n_ch) for n_ch edge channels.
    """
    N = A_edges.shape[0]
    # Average
    avg = torch.zeros(N, n_pts, n_ch, device=device)
    avg_dx = torch.zeros(N, n_pts, n_ch, device=device)
    avg_dy = torch.zeros(N, n_pts, n_ch, device=device)
    count = torch.zeros(n_pts, device=device)

    dx = torch.tensor(edge_dir_norm[:, 0], dtype=torch.float32, device=device)
    dy = torch.tensor(edge_dir_norm[:, 1], dtype=torch.float32, device=device)

    for ei in range(n_edges):
        a, b = edges[ei]
        val = A_edges[:, ei]  # (N, n_ch) or (N,)
        if val.ndim == 1:
            val = val.unsqueeze(-1)
        avg[:, a] += val; avg[:, b] += val
        avg_dx[:, a] += val * dx[ei]; avg_dx[:, b] += val * dx[ei]
        avg_dy[:, a] += val * dy[ei]; avg_dy[:, b] += val * dy[ei]
        count[a] += 1; count[b] += 1

    count[count == 0] = 1
    avg /= count[None, :, None]
    avg_dx /= count[None, :, None]
    avg_dy /= count[None, :, None]

    return torch.cat([avg, avg_dx, avg_dy], dim=-1)  # (N, n_pts, 3*n_ch)

def encode_faces_to_nodes(F_faces, n_ch=1):
    """Encode face values to node features."""
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
    print(f"  Saved {name}: {os.path.getsize(path)/1e6:.1f}MB, X:{X.shape} Y:{Y.shape}")
    print(f"    X std={X.std():.4f}, Y std={Y.std():.4f}")


# ==========================================
# Task 1: Wilson loop v2 — magnetic vortex defects
# ==========================================
print("\n=== Wilson Loop v2 (vortex defects) ===")
t0 = time.time()
N = 10000
torch.manual_seed(42)

# Connection = smooth pure gauge + localized flux defects
theta_edges = torch.zeros(N, n_edges, device=device)

# Smooth part: dθ from Gaussian scalar fields
for _ in range(5):
    cx = torch.rand(N, 1, device=device)
    cy = torch.rand(N, 1, device=device)
    sigma = torch.rand(N, 1, device=device) * 0.3 + 0.1
    amp = torch.randn(N, 1, device=device) * 1.0
    d2 = (pts_t[None, :, 0] - cx)**2 + (pts_t[None, :, 1] - cy)**2
    theta_node = amp * torch.exp(-d2 / (2 * sigma**2))
    theta_edges += theta_node[:, edges_t[:, 1]] - theta_node[:, edges_t[:, 0]]

# Add magnetic vortex defects: localized curvature at random points
# These create non-trivial plaquettes that pure gauge can't explain
n_vortices = torch.randint(1, 4, (N,), device=device)
for v in range(3):
    mask = (n_vortices > v).float()  # (N,)
    cx = torch.rand(N, 1, device=device)
    cy = torch.rand(N, 1, device=device)
    strength = (torch.randn(N, 1, device=device) * 2.0) * mask.unsqueeze(-1)

    # Vortex: A = strength * (-y, x) / r² → circulation around center
    for ei in range(n_edges):
        a, b = edges[ei]
        mid_x = (pts[a, 0] + pts[b, 0]) / 2
        mid_y = (pts[a, 1] + pts[b, 1]) / 2
        dx_e = edge_dir[ei, 0]
        dy_e = edge_dir[ei, 1]
        rx = mid_x - cx.squeeze(-1)
        ry = mid_y - cy.squeeze(-1)
        r2 = rx**2 + ry**2 + 0.01
        # A · dl ≈ strength * (-ry*dx + rx*dy) / r²
        theta_edges[:, ei] += strength.squeeze(-1) * (-ry * dx_e + rx * dy_e) / r2

# Plaquette flux = d1 @ theta
plaq = (theta_edges[:, face_edge_idx_t[:, 0]] * face_edge_sign_t[None, :, 0] +
        theta_edges[:, face_edge_idx_t[:, 1]] * face_edge_sign_t[None, :, 1] +
        theta_edges[:, face_edge_idx_t[:, 2]] * face_edge_sign_t[None, :, 2])  # (N, n_faces)

# Encode to nodes with directional info
X_wl = encode_edges_to_nodes(theta_edges, n_ch=1)  # (N, n_pts, 3)
Y_wl = encode_faces_to_nodes(plaq, n_ch=1)  # (N, n_pts, 1)

print(f"  Generated in {time.time()-t0:.1f}s")
save_dataset('wilson_v2', X_wl.float(), Y_wl.float(),
    'wilson_loop_vortex',
    'Wilson loop v2: θ(edges)→plaquette, magnetic vortex defects, directional node encoding')

# ==========================================
# Task 2: Yang-Mills v2 — strong non-abelian
# ==========================================
print("\n=== Yang-Mills v2 (strong coupling) ===")
t0 = time.time()
N2 = 10000
torch.manual_seed(42)

# SU(2) connection with LARGE amplitude → [A,A] matters
A_edges = torch.randn(N2, n_edges, 3, device=device) * 0.5

# Smooth large-scale structure
for comp in range(3):
    for _ in range(4):
        cx = torch.rand(N2, 1, device=device)
        cy = torch.rand(N2, 1, device=device)
        sigma = torch.rand(N2, 1, device=device) * 0.25 + 0.1
        amp = torch.randn(N2, 1, device=device) * 2.0  # LARGER amplitude
        d2 = (pts_t[None, :, 0] - cx)**2 + (pts_t[None, :, 1] - cy)**2
        theta = amp * torch.exp(-d2 / (2 * sigma**2))
        A_edges[:, :, comp] += theta[:, edges_t[:, 1]] - theta[:, edges_t[:, 0]]

# Field strength: F = dA + [A,A]
A_f = []
for k in range(3):
    Ak = (A_edges[:, face_edge_idx_t[:, k]] *
          face_edge_sign_t[None, :, k:k+1])  # (N2, n_faces, 3)
    A_f.append(Ak)

dA = A_f[0] + A_f[1] + A_f[2]
comm = (torch.cross(A_f[0], A_f[1], dim=-1) +
        torch.cross(A_f[1], A_f[2], dim=-1) +
        torch.cross(A_f[2], A_f[0], dim=-1))
F_faces = dA + comm  # (N2, n_faces, 3)

# Encode to nodes
X_ym = encode_edges_to_nodes(A_edges.reshape(N2, n_edges, 3), n_ch=3)  # (N2, n_pts, 9)
Y_ym = encode_faces_to_nodes(F_faces, n_ch=3)  # (N2, n_pts, 3)

print(f"  Generated in {time.time()-t0:.1f}s")
save_dataset('yang_mills_v2', X_ym.float(), Y_ym.float(),
    'su2_field_strength_v2',
    'Yang-Mills v2: A(nodes,9ch)→F(nodes,3ch), strong coupling, directional encoding')

print("\nDone!")
