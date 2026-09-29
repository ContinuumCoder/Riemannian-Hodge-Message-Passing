"""Load a zoo checkpoint and use it without any data set (CPU, a few seconds).

    python3 results/checkpoints/load_example.py

The HP_k100 solver-mode model with a learned tensor metric (2,355 parameters) was trained on heterogeneous Poisson
problems ``-div(sigma grad u) = f`` (u = 0 on the boundary) with the log conductivity given as material columns on
edges and faces.  Here it is applied to a new random mesh and a synthetic conductivity field it has never seen, and
its learned per-face tensors are compared with that field: in solver mode the metric head has learned the material
map itself, so ``log det(sigma_f) / 2`` should follow the input ``log sigma`` with slope close to 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import Delaunay

ROOT = Path(__file__).resolve().parents[2]
try:
    import rhmp  # noqa: F401
except ImportError:  # running from a source checkout without `pip install -e .`
    sys.path.insert(0, str(ROOT))

from rhmp import RHMP, CochainComplex


def main() -> None:
    ckpt = ROOT / "results" / "checkpoints" / "HP_k100_S3c_solver_tensor_learn" / "best.pt"
    model = RHMP.from_checkpoint(ckpt, map_location="cpu").eval()
    cfg = model.cfg
    print(f"loaded {ckpt.parent.name}: layers {cfg.layer_types}, metric {cfg.metric_type} ({cfg.tensor_param}), "
          f"material columns {cfg.material_dims}, {model.num_parameters()} parameters")

    # a new mesh of the unit square: 30 points per side on the boundary, 1200 random interior points
    rng = np.random.default_rng(0)
    s = np.linspace(0.0, 1.0, 31)[:-1]
    boundary = np.concatenate([np.c_[s, 0 * s], np.c_[1 + 0 * s, s], np.c_[1 - s, 1 + 0 * s], np.c_[0 * s, 1 - s]])
    pts = np.concatenate([boundary, rng.uniform(0.02, 0.98, (1200, 2))])
    K = CochainComplex.from_triangles(pts, Delaunay(pts).simplices)

    # a synthetic smooth log-conductivity field with contrast about 100, sampled at edge midpoints and face centroids
    def log_sigma(x: np.ndarray) -> np.ndarray:
        return 2.3 * np.tanh(2.0 * np.sin(3.0 * x[:, 0] + 0.5) * np.cos(2.0 * x[:, 1] - 0.3))
    e, f = K.cells[1].numpy(), K.cells[2].numpy()
    ls_e, ls_f = log_sigma(pts[e].mean(1)), log_sigma(pts[f].mean(1))
    # inputs {degree: (n_k, B, F_k)}: a source on the vertices (normalised units) and the RAW log conductivity on
    # edges and faces (the material columns of this run are raw physical values, see config.json raw_input_columns)
    inputs = {0: torch.randn(K.n[0], 1, 1, generator=torch.Generator().manual_seed(0)),
              1: torch.tensor(ls_e, dtype=torch.float32)[:, None, None],
              2: torch.tensor(ls_f, dtype=torch.float32)[:, None, None]}
    with torch.no_grad():
        u = model(inputs, K)                                            # (n0, 1, 1): the solution, normalised units
        sigma = model.metric_fields(inputs, K)[0]["sigma"][1][:, 0]     # (n2, 2, 2): learned per-face tensors
    half_logdet = 0.5 * torch.logdet(sigma.double()).numpy()
    ev = torch.linalg.eigvalsh(sigma.double())
    r = float(np.corrcoef(half_logdet, ls_f)[0, 1])
    slope, intercept = np.polyfit(ls_f, half_logdet, 1)
    print(f"new mesh: {K.n[0]} vertices, {K.n[2]} triangles; output u {tuple(u.shape)}, "
          f"rms {u.pow(2).mean().sqrt().item():.3f} (normalised units)")
    print(f"learned log det(sigma_f)/2 vs input log sigma_f: r = {r:.4f}, slope {slope:.3f}, intercept {intercept:+.3f}; "
          f"median anisotropy ratio {float((ev[:, 1] / ev[:, 0]).median()):.2f}")


if __name__ == "__main__":
    main()
