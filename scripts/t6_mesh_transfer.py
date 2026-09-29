"""Zero-shot mesh / resolution transfer for a trained T6f model (edge U(1) connection -> face flux).

The Wilson-loop physics of ``datasets/generators/gen_T6_native.py`` (smooth pure-gauge bumps + 1..3 vortices) is regenerated
on NEW random Delaunay meshes (different mesh seed; optionally more nodes) and a finished run is evaluated with the
trainer's ``eval_ckpt`` machinery (the training run's normalisation is used).  v1 cannot be evaluated on a new mesh
at all: its metric has per-cell parameters tied to the training mesh.

usage: python3 scripts/t6_mesh_transfer.py --run runs/new/T6f_s42 --mesh-seed 7 --n-pts 1024 --n 500
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "datasets", "generators"))
from gen_common import delaunay_1024, edge_face_incidence  # noqa: E402

from rhmp import train as T  # noqa: E402
from rhmp.complex import CochainComplex  # noqa: E402
from rhmp.data import TaskData  # noqa: E402


def generate(mesh_seed: int, n_pts: int, N: int, field_seed: int, device):
    pts, faces = delaunay_1024(seed=mesh_seed, n_pts=n_pts)
    v0, v1, v2 = pts[faces[:, 0]], pts[faces[:, 1]], pts[faces[:, 2]]
    area = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
    faces = np.where(area[:, None] < 0, faces[:, [0, 2, 1]], faces)          # consistent CCW orientation
    edges, fei, fes = edge_face_incidence(faces, n_pts)
    edge_dir = pts[edges[:, 1]] - pts[edges[:, 0]]
    pts_t = torch.tensor(pts, dtype=torch.float32, device=device)
    edges_t = torch.tensor(edges, device=device)
    fei_t = torch.tensor(fei, device=device)
    fes_t = torch.tensor(fes, device=device)
    torch.manual_seed(field_seed)
    theta = torch.zeros(N, len(edges), device=device)
    for _ in range(5):                                                       # smooth pure-gauge part
        cx = torch.rand(N, 1, device=device)
        cy = torch.rand(N, 1, device=device)
        sigma = torch.rand(N, 1, device=device) * 0.3 + 0.1
        amp = torch.randn(N, 1, device=device)
        d2 = (pts_t[None, :, 0] - cx) ** 2 + (pts_t[None, :, 1] - cy) ** 2
        node = amp * torch.exp(-d2 / (2 * sigma ** 2))
        theta += node[:, edges_t[:, 1]] - node[:, edges_t[:, 0]]
    n_vort = torch.randint(1, 4, (N,), device=device)
    mid_x = torch.tensor((pts[edges[:, 0], 0] + pts[edges[:, 1], 0]) / 2, dtype=torch.float32, device=device)
    mid_y = torch.tensor((pts[edges[:, 0], 1] + pts[edges[:, 1], 1]) / 2, dtype=torch.float32, device=device)
    dx_e = torch.tensor(edge_dir[:, 0], dtype=torch.float32, device=device)
    dy_e = torch.tensor(edge_dir[:, 1], dtype=torch.float32, device=device)
    for v in range(3):                                                       # magnetic vortex defects
        mask = (n_vort > v).float()
        cx = torch.rand(N, 1, device=device)
        cy = torch.rand(N, 1, device=device)
        strength = (torch.randn(N, 1, device=device) * 2.0) * mask.unsqueeze(-1)
        rx = mid_x[None, :] - cx
        ry = mid_y[None, :] - cy
        theta += strength * (-ry * dx_e[None, :] + rx * dy_e[None, :]) / (rx ** 2 + ry ** 2 + 0.01)
    plaq = (theta[:, fei_t[:, 0]] * fes_t[None, :, 0] + theta[:, fei_t[:, 1]] * fes_t[None, :, 1]
            + theta[:, fei_t[:, 2]] * fes_t[None, :, 2])
    return pts, faces, edges, theta, plaq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--mesh-seed", type=int, default=7)
    ap.add_argument("--n-pts", type=int, default=1024)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--field-seed", type=int, default=123)
    ap.add_argument("--out", default=None)
    ap.add_argument("--star", default="cotan")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    pts, faces, edges, theta, plaq = generate(a.mesh_seed, a.n_pts, a.n, a.field_seed, device)
    K = CochainComplex.from_triangles(torch.tensor(pts, dtype=torch.float32), torch.tensor(faces), star=a.star,
                                      device=device)
    assert K.n[1] == len(edges) and torch.equal(K.cells[1].cpu(), torch.tensor(edges)), "edge convention mismatch"
    name = f"T6f_mesh{a.mesh_seed}_n{a.n_pts}"
    task = TaskData.from_arrays(K, {1: theta.unsqueeze(-1)}, plaq.unsqueeze(-1), readout="cochain:2",
                                connection_dims={1: 1}, split=(0.5, 0.1, 0.4), name=name)
    out = a.out or os.path.join(ROOT, "runs", "transfer", os.path.basename(a.run.rstrip("/")))
    os.makedirs(out, exist_ok=True)
    args = T.parse_args(["--task", name, "--native", "--eval-ckpt", a.run, "--out", out])
    res = T.eval_ckpt(args, task, device, out)
    res.update(mesh_seed=a.mesh_seed, n_pts=a.n_pts, n1=K.n[1], n2=K.n[2], n_samples=a.n,
               gen_seconds=time.time() - t0)
    json.dump(res, open(os.path.join(out, f"transfer_{name}.json"), "w"), indent=2)
    print(f"[{name}] n0={a.n_pts} n1={K.n[1]} n2={K.n[2]}  test R2={res['test']['R2']:.5f}  "
          f"NRMSE={res['test']['NRMSE']:.5f}  (uncentred R2 for odd cochains)")


if __name__ == "__main__":
    main()
