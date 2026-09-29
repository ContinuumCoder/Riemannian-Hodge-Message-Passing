"""Regenerate T7 (SU(2) Yang-Mills, weak-coupling configuration of the v1 generator) with its raw edge cochain.

Reproduces the v1 generator ``datasets/gen_T7_yang_mills_su2.py`` of the original repository (same mesh seed and
torch RNG call sequence on CUDA) and dumps

    A_edges (N, n1, 3) float32   su(2) connection per edge (3 Lie-algebra components), canonical orientation
                                 src < dst, edges sorted lexicographically (= generator order == v1 edge_index)
    F_faces (N, n2, 3) float32   field strength  F = dA + [A, A]  per face (orientation of the stored ``faces``)

The mesh is taken from ``datasets/T7_yang_mills_su2.pkl`` (checked against the seeded regeneration).  Consistency:
the v1 node encodings recomputed from ``A_edges`` / ``F_faces`` are compared with the stored ``X_data`` / ``Y_data``.

Usage (CUDA GPU):  python3 -u datasets/generators/gen_T7_native.py            -> datasets/v2/T7_native.pt
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


def main(out_path: str = os.path.join(ROOT, "datasets", "v2", "T7_native.pt")):
    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: the v1 dataset was generated with the CUDA RNG; CPU draws will not reproduce it")
    ref = pickle.load(open(os.path.join(ROOT, "datasets", "T7_yang_mills_su2.pkl"), "rb"))
    pts_ref = np.asarray(ref["points"], dtype=np.float64)
    faces = np.asarray(ref["faces"], dtype=np.int64)
    pts_regen, faces_regen = delaunay_1024()
    assert np.array_equal(pts_ref[:, :2], pts_regen), "mesh seed does not reproduce the stored points"
    same_faces = faces_regen.shape == faces.shape and bool((faces_regen == faces).all())
    print(f"points reproduced exactly; stored faces identical to scipy regeneration: {same_faces}")
    pts = pts_ref[:, :2].copy()
    n_pts = pts.shape[0]
    edges, face_edge_idx, face_edge_sign = edge_face_incidence(faces, n_pts)
    assert np.array_equal(edges, np.asarray(ref["edge_index"], dtype=np.int64)), "edge list differs from stored"
    n_edges, n_faces = len(edges), len(faces)
    edge_dir = pts[edges[:, 1]] - pts[edges[:, 0]]
    print(f"Mesh: {n_pts} nodes, {n_edges} edges, {n_faces} faces")

    pts_t = torch.tensor(pts, dtype=torch.float32, device=device)
    edges_t = torch.tensor(edges, device=device)
    fei = torch.tensor(face_edge_idx, device=device)
    fes = torch.tensor(face_edge_sign, device=device)

    # ---------------- identical RNG call sequence to the v1 generator gen_T7_yang_mills_su2.py ----------------
    N = 10000
    torch.manual_seed(42)
    A_edges = torch.randn(N, n_edges, 3, device=device) * 0.1
    for comp in range(3):
        for _ in range(4):
            cx = torch.rand(N, 1, device=device)
            cy = torch.rand(N, 1, device=device)
            sigma = torch.rand(N, 1, device=device) * 0.25 + 0.1
            amp = torch.randn(N, 1, device=device) * 0.5
            d2 = (pts_t[None, :, 0] - cx) ** 2 + (pts_t[None, :, 1] - cy) ** 2
            theta = amp * torch.exp(-d2 / (2 * sigma ** 2))
            A_edges[:, :, comp] += theta[:, edges_t[:, 1]] - theta[:, edges_t[:, 0]]
    for comp in range(3):
        for _ in range(2):
            cx = torch.rand(N, 1, device=device)
            cy = torch.rand(N, 1, device=device)
            strength = torch.randn(N, 1, device=device) * 0.3
            edge_mid_x = (pts_t[edges_t[:, 0], 0] + pts_t[edges_t[:, 1], 0]) / 2
            edge_mid_y = (pts_t[edges_t[:, 0], 1] + pts_t[edges_t[:, 1], 1]) / 2
            rx = edge_mid_x[None, :] - cx
            ry = edge_mid_y[None, :] - cy
            r2 = rx ** 2 + ry ** 2 + 0.01
            dx_e = torch.tensor(edge_dir[:, 0], dtype=torch.float32, device=device)
            dy_e = torch.tensor(edge_dir[:, 1], dtype=torch.float32, device=device)
            A_edges[:, :, comp] += strength * (-ry * dx_e[None, :] + rx * dy_e[None, :]) / r2

    A_f = [A_edges[:, fei[:, k]] * fes[None, :, k:k + 1] for k in range(3)]
    dA = A_f[0] + A_f[1] + A_f[2]
    comm = (torch.cross(A_f[0], A_f[1], dim=-1) +
            torch.cross(A_f[1], A_f[2], dim=-1) +
            torch.cross(A_f[2], A_f[0], dim=-1))
    F_faces = dA + comm
    ratio = dA.norm(dim=-1).mean().item() / max(comm.norm(dim=-1).mean().item(), 1e-8)
    del A_f, dA, comm
    t_gen = time.time() - t_start

    # ---------------- consistency with the stored v1 node encodings ----------------
    X_ref = torch.tensor(np.asarray(ref["X_data"], dtype=np.float32), device=device)
    X_re = encode_edges_to_nodes(A_edges, edges, pts, n_pts)
    ex = (X_re - X_ref).abs().max().item()
    rx_ = ex / X_ref.abs().max().item()
    del X_ref, X_re
    Y_ref = torch.tensor(np.asarray(ref["Y_data"], dtype=np.float32), device=device)
    Y_re = encode_faces_to_nodes(F_faces, faces, n_pts)
    ey = (Y_re - Y_ref).abs().max().item()
    ry_ = ey / Y_ref.abs().max().item()
    print(f"|dA|/|[A,A]| = {ratio:.1f}")
    print(f"consistency: max|X_re - X_data| = {ex:.3e} (rel {rx_:.3e}),  max|Y_re - Y_data| = {ey:.3e} (rel {ry_:.3e})")
    ok = rx_ < 1e-4 and ry_ < 1e-4

    meta = {
        "source": "datasets/gen_T7_yang_mills_su2.py (same RNG sequence)",
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch": torch.__version__,
        "N": N, "n0": n_pts, "n1": n_edges, "n2": n_faces, "dA_over_comm": ratio,
        "edge_orientation": "canonical src<dst, lexicographic order (== v1 edge_index)",
        "face_orientation": "as stored in datasets/T7_yang_mills_su2.pkl['faces']",
        "consistency_X_maxabs": ex, "consistency_X_rel": rx_,
        "consistency_Y_maxabs": ey, "consistency_Y_rel": ry_,
        "consistent": bool(ok),
        "gen_seconds": t_gen,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({
        "points": torch.tensor(pts), "faces": torch.tensor(faces), "edges": torch.tensor(edges),
        "A_edges": A_edges.cpu(), "F_faces": F_faces.cpu(), "meta": meta,
    }, out_path)
    meta["file_MB"] = os.path.getsize(out_path) / 1e6
    meta["total_seconds"] = time.time() - t_start
    json.dump(meta, open(out_path.replace(".pt", ".json"), "w"), indent=2)
    print(json.dumps(meta, indent=2))
    if not ok:
        print("WARNING: regenerated fields do NOT reproduce the stored v1 inputs")


if __name__ == "__main__":
    main(*sys.argv[1:])
