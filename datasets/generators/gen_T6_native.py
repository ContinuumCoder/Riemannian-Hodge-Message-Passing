"""Regenerate T6 (U(1) Wilson loop with vortex defects) together with its raw edge cochain.

Reproduces the v1 generator ``datasets/gen_T6_wilson_loop.py`` of the original repository (same mesh seed, same
torch RNG call sequence on CUDA; the per-edge Python loops are vectorised with identical per-element arithmetic) and
dumps what v1 did not save:

    theta_edges (N, n1) float32   U(1) connection per edge, canonical orientation src < dst, edges sorted
                                  lexicographically (= generator order); ``edges`` is stored alongside
    plaq        (N, n2) float32   plaquette flux  (d1 theta)_f  in the orientation of the stored ``faces``

The mesh (``points``, ``faces``) is taken from ``datasets/T6_wilson_loop.pkl`` (checked against the seeded
regeneration).  Consistency check: the v1 node encodings recomputed from ``theta_edges`` / ``plaq`` are compared
with the stored ``X_data`` / ``Y_data`` (max abs and relative error printed and saved in the output ``meta``).

Usage (CUDA GPU):  python3 -u datasets/generators/gen_T6_native.py            -> datasets/v2/T6_native.pt
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_common import delaunay_1024, edge_face_incidence, encode_edges_to_nodes, encode_faces_to_nodes  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(out_path: str = os.path.join(ROOT, "datasets", "v2", "T6_native.pt")):
    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: the v1 dataset was generated with the CUDA RNG; CPU draws will not reproduce it")
    ref = pickle.load(open(os.path.join(ROOT, "datasets", "T6_wilson_loop.pkl"), "rb"))
    pts_ref = np.asarray(ref["points"], dtype=np.float64)
    faces = np.asarray(ref["faces"], dtype=np.int64)
    pts_regen, faces_regen = delaunay_1024()
    assert np.allclose(pts_ref[:, :2], pts_regen, atol=0, rtol=0), "mesh seed does not reproduce the stored points"
    assert np.abs(pts_ref[:, 2]).max() == 0.0
    same_faces = faces_regen.shape == faces.shape and bool((faces_regen == faces).all())
    print(f"points reproduced exactly; stored faces identical to scipy regeneration: {same_faces}")
    pts = pts_ref[:, :2].copy()
    n_pts = pts.shape[0]
    edges, face_edge_idx, face_edge_sign = edge_face_incidence(faces, n_pts)
    n_edges, n_faces = len(edges), len(faces)
    assert np.array_equal(edges, np.asarray(ref["edge_index"], dtype=np.int64)), "edge list differs from stored"
    edge_dir = pts[edges[:, 1]] - pts[edges[:, 0]]
    print(f"Mesh: {n_pts} nodes, {n_edges} edges, {n_faces} faces")

    pts_t = torch.tensor(pts, dtype=torch.float32, device=device)
    edges_t = torch.tensor(edges, device=device)
    fei = torch.tensor(face_edge_idx, device=device)
    fes = torch.tensor(face_edge_sign, device=device)

    # ---------------- identical RNG call sequence to the v1 generator gen_T6_wilson_loop.py ----------------
    N = 10000
    torch.manual_seed(42)
    theta_edges = torch.zeros(N, n_edges, device=device)
    for _ in range(5):
        cx = torch.rand(N, 1, device=device)
        cy = torch.rand(N, 1, device=device)
        sigma = torch.rand(N, 1, device=device) * 0.3 + 0.1
        amp = torch.randn(N, 1, device=device) * 1.0
        d2 = (pts_t[None, :, 0] - cx) ** 2 + (pts_t[None, :, 1] - cy) ** 2
        theta_node = amp * torch.exp(-d2 / (2 * sigma ** 2))
        theta_edges += theta_node[:, edges_t[:, 1]] - theta_node[:, edges_t[:, 0]]

    n_vortices = torch.randint(1, 4, (N,), device=device)
    # v1 evaluates, per edge ei, with numpy float64 scalars (cast to float32 inside the float32 kernels):
    #   rx = mid_x - cx ; ry = mid_y - cy ; r2 = rx**2 + ry**2 + 0.01
    #   theta[:, ei] += strength * (-ry * dx_e + rx * dy_e) / r2
    mid_x = torch.tensor((pts[edges[:, 0], 0] + pts[edges[:, 1], 0]) / 2, dtype=torch.float32, device=device)
    mid_y = torch.tensor((pts[edges[:, 0], 1] + pts[edges[:, 1], 1]) / 2, dtype=torch.float32, device=device)
    dx_e = torch.tensor(edge_dir[:, 0], dtype=torch.float32, device=device)
    dy_e = torch.tensor(edge_dir[:, 1], dtype=torch.float32, device=device)
    for v in range(3):
        mask = (n_vortices > v).float()
        cx = torch.rand(N, 1, device=device)
        cy = torch.rand(N, 1, device=device)
        strength = (torch.randn(N, 1, device=device) * 2.0) * mask.unsqueeze(-1)
        rx = mid_x[None, :] - cx
        ry = mid_y[None, :] - cy
        r2 = rx ** 2 + ry ** 2 + 0.01
        theta_edges += strength * (-ry * dx_e[None, :] + rx * dy_e[None, :]) / r2
        del rx, ry, r2

    plaq = (theta_edges[:, fei[:, 0]] * fes[None, :, 0] +
            theta_edges[:, fei[:, 1]] * fes[None, :, 1] +
            theta_edges[:, fei[:, 2]] * fes[None, :, 2])            # (N, n_faces)
    t_gen = time.time() - t_start

    # ---------------- consistency with the stored v1 node encodings ----------------
    X_ref = torch.tensor(np.asarray(ref["X_data"], dtype=np.float32), device=device)
    Y_ref = torch.tensor(np.asarray(ref["Y_data"], dtype=np.float32), device=device)
    X_re = encode_edges_to_nodes(theta_edges, edges, pts, n_pts)
    Y_re = encode_faces_to_nodes(plaq, faces, n_pts)
    ex = (X_re - X_ref).abs().max().item()
    ey = (Y_re - Y_ref).abs().max().item()
    rx_ = ex / X_ref.abs().max().item()
    ry_ = ey / Y_ref.abs().max().item()
    print(f"consistency: max|X_re - X_data| = {ex:.3e} (rel {rx_:.3e}),  max|Y_re - Y_data| = {ey:.3e} (rel {ry_:.3e})")
    ok = rx_ < 1e-4 and ry_ < 1e-4

    meta = {
        "source": "datasets/gen_T6_wilson_loop.py (vectorised, same RNG sequence)",
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch": torch.__version__,
        "N": N, "n0": n_pts, "n1": n_edges, "n2": n_faces,
        "edge_orientation": "canonical src<dst, lexicographic order (== v1 edge_index)",
        "face_orientation": "as stored in datasets/T6_wilson_loop.pkl['faces']",
        "consistency_X_maxabs": ex, "consistency_X_rel": rx_,
        "consistency_Y_maxabs": ey, "consistency_Y_rel": ry_,
        "consistent": bool(ok),
        "gen_seconds": t_gen,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({
        "points": torch.tensor(pts), "faces": torch.tensor(faces), "edges": torch.tensor(edges),
        "theta_edges": theta_edges.cpu(), "plaq": plaq.cpu(), "meta": meta,
    }, out_path)
    meta["file_MB"] = os.path.getsize(out_path) / 1e6
    meta["total_seconds"] = time.time() - t_start
    json.dump(meta, open(out_path.replace(".pt", ".json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))
    if not ok:
        print("WARNING: regenerated fields do NOT reproduce the stored v1 inputs; the native variant is still "
              "self-consistent (targets recomputed from the regenerated edges) but not the same samples as v1")


if __name__ == "__main__":
    main(*sys.argv[1:])
