"""
Generate T3: Ellipsoid Surface Flow
  Input:  ψ (scalar stream function on ellipsoid nodes)
  Output: v (tangential velocity field on ellipsoid surface, 3D vector)

Physics: Standard stream-function-to-velocity mapping on a curved surface.
  Given ψ, the tangential velocity is v = n × ∇ψ (perpendicular gradient).
  Requires understanding local mesh geometry (cotangent-weighted gradient)
  and surface normals for tangential projection.

Mesh: ellipsoid (a=1.4, b=1.0, c=0.7), ~1157 nodes (genus-0)
Samples: 10000
Source: sum of 3-6 broad Gaussians (σ=0.15-0.45, amp=1-5) centered on mesh nodes

Usage: python3 datasets/gen_T3_ellipsoid_coexact.py
"""
import pickle, numpy as np, os, time
import pyvista as pv


def create_ellipsoid_mesh(a=1.4, b=1.0, c=0.7, n_target=1200):
    """Create ellipsoid mesh with ~n_target nodes."""
    res = int(np.sqrt(n_target)) + 1
    sphere = pv.Sphere(radius=1.0, theta_resolution=res, phi_resolution=res)
    sphere = sphere.triangulate().clean()
    pts = sphere.points.copy().astype(np.float64)
    pts[:, 0] *= a; pts[:, 1] *= b; pts[:, 2] *= c
    scale = np.max(np.abs(pts))
    pts /= scale
    faces = sphere.faces.reshape((-1, 4))[:, 1:].astype(np.int64)
    return pts, faces


def generate(output_path, n_samples=10000, seed=42):
    print("Creating ellipsoid mesh...")
    pts, fcs = create_ellipsoid_mesh()
    n = len(pts)
    print(f"  {n} nodes, {len(fcs)} faces")

    # Vertex normals
    normals = np.zeros((n, 3))
    for face in fcs:
        i, j, k = face
        nm = np.cross(pts[j] - pts[i], pts[k] - pts[i])
        normals[i] += nm; normals[j] += nm; normals[k] += nm
    normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)

    # Gradient operator (per-face → averaged to nodes)
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

    # Edge index
    edges = set()
    for face in fcs:
        for i in range(3):
            edges.add(tuple(sorted([face[i], face[(i + 1) % 3]])))
    edge_index = np.array(list(edges), dtype=np.int64)

    def compute_coexact(psi):
        """ψ → v = n × ∇ψ (coexact tangent velocity)."""
        pi, pj, pk = psi[fi], psi[fj], psi[fk]
        gt = (pi[:, None] * cr_kj + pj[:, None] * cr_ik + pk[:, None] * cr_ji) / fa2
        grad = np.zeros((n, 3))
        for d in range(3):
            np.add.at(grad[:, d], fi, gt[:, d])
            np.add.at(grad[:, d], fj, gt[:, d])
            np.add.at(grad[:, d], fk, gt[:, d])
        grad /= nc[:, None]
        v = np.cross(normals, grad)
        v -= np.sum(v * normals, axis=1, keepdims=True) * normals
        return v

    # Generate samples
    rng = np.random.RandomState(seed)
    X_data = np.zeros((n_samples, n, 1), dtype=np.float32)
    Y_data = np.zeros((n_samples, n, 3), dtype=np.float32)

    print(f"Generating {n_samples} samples (ψ → n×∇ψ)...")
    t0 = time.time()
    for s in range(n_samples):
        psi = np.zeros(n)
        nb = rng.randint(3, 7)
        for _ in range(nb):
            c = pts[rng.randint(0, n)]
            sig = rng.uniform(0.15, 0.45)
            amp = rng.uniform(1.0, 5.0) * rng.choice([-1, 1])
            psi += amp * np.exp(-np.linalg.norm(pts - c, axis=1) ** 2 / (2 * sig ** 2))

        X_data[s, :, 0] = psi.astype(np.float32)
        Y_data[s] = compute_coexact(psi).astype(np.float32)

        if (s + 1) % 2000 == 0:
            print(f"  {s+1}/{n_samples} ({time.time()-t0:.1f}s)")

    print(f"Done: {time.time()-t0:.1f}s")
    print(f"X (ψ): std={X_data.std():.4f}")
    print(f"Y (n×∇ψ): std={Y_data.std():.4f}")

    dataset = {
        'points': pts.astype(np.float32),
        'faces': fcs,
        'edge_index': edge_index,
        'X_data': X_data,
        'Y_data': Y_data,
        'n_nodes': n,
        'n_samples': n_samples,
        'task': 'ellipsoid_coexact_velocity',
        'topology': 'ellipsoid_genus0',
        'physics': 'v = n × grad(psi), coexact tangent velocity from stream function',
    }

    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Saved: {output_path} ({os.path.getsize(output_path)/1e6:.1f} MB)")


if __name__ == '__main__':
    generate(os.path.join(os.path.dirname(__file__), 'T3_ellipsoid_coexact.pkl'))
