"""Example 05: variable meshes, one complex per sample, trained with block-diagonal batches.

Shows how ``CochainComplex.batch`` concatenates several meshes into one block-diagonal complex (``B = 1``, cells of
all meshes stacked, ``K.batch[k]`` = graph id of every k-cell, ``K.meta['ptr'][k]`` = offsets), that the model's
output for each mesh is independent of the other meshes in the batch, how to split outputs back per mesh, how the
DEC solver ``rhmp.dec.cg_solve`` runs one independent CG per mesh on such a batch, and how to wrap per-sample data
in a variable-mesh ``TaskData`` for ``rhmp.train.run``.

Task: harmonic extension.  Boundary values ``g`` (a smooth random function on the boundary vertices, 0 inside) are
the node input; the target is the discrete harmonic function ``u`` with ``u = g`` on the boundary and
``(d_0^T star_1 d_0 u)_i = 0`` at interior vertices (cotan Laplacian).  The problem has no length scale, which suits
a model that is exactly invariant to the length unit (see docs/TUTORIAL.md, "Length scales").

Run (CPU, about 15-30 s):  python examples/05_variable_meshes.py [--out DIR]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import Delaunay

try:
    import rhmp  # noqa: F401
except ImportError:  # running from a source checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rhmp import RHMP, CochainComplex, RHMPConfig, dec
from rhmp.data import TaskData, feature_stats, mesh_minibatch, scale_stats, sequential_split
from rhmp.train import parse_args, run


def square_mesh(rng: np.random.Generator, m: int):
    """Delaunay mesh of the unit square (``m`` boundary points per side, jittered interior grid)."""
    s = np.linspace(0.0, 1.0, m + 1)[:-1]
    boundary = np.concatenate([np.c_[s, 0 * s], np.c_[1 + 0 * s, s], np.c_[1 - s, 1 + 0 * s], np.c_[0 * s, 1 - s]])
    g = np.arange(1, m) / m
    ii, jj = np.meshgrid(g, g, indexing="ij")
    interior = np.c_[ii.ravel(), jj.ravel()] + rng.uniform(-0.35, 0.35, (g.size ** 2, 2)) / m
    pos = np.concatenate([boundary, interior])
    return pos, Delaunay(pos).simplices


def boundary_values(rng: np.random.Generator, pos: np.ndarray, on_boundary: np.ndarray) -> np.ndarray:
    """Smooth random function of the position, restricted to the boundary vertices (0 elsewhere)."""
    w = rng.normal(0.0, 1.0, (4, 2))
    phase = rng.uniform(0.0, 2 * np.pi, 4)
    amp = rng.normal(0.0, 1.0, 4)
    return (amp * np.cos(2 * np.pi * pos @ w.T + phase)).sum(-1) * on_boundary


def harmonic_extension(Kb: CochainComplex, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve ``L_II u_I = -L_IB g_B``, ``u_B = g_B`` for every mesh of a block-diagonal batch at once.

    The masked system ``A = P_I L P_I + P_B`` is SPD; ``cg_solve(..., batch=Kb.batch[0], num_graphs=G)`` runs one
    independent CG per mesh (per-graph inner products), so meshes never influence each other.
    """
    L = dec.hodge_laplacian(Kb, 0)                                  # d0^T star1 d0 (cotan Laplacian), per mesh
    bnd = Kb.boundary[0].to(g.dtype)[:, None]                        # (n0, 1)
    inner = 1.0 - bnd

    def A(v):
        return inner * L((inner * v).contiguous()) + bnd * v

    rhs = -inner * L((bnd * g).contiguous()) + bnd * g
    batch = None if Kb.batch is None else Kb.batch[0]              # K.batch is None for a single complex
    return dec.cg_solve(A, rhs, iters=2000, tol=1e-10, batch=batch, num_graphs=Kb.num_graphs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meshes", type=int, default=240)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="run directory (default: a temporary directory)")
    a = ap.parse_args()
    out = Path(a.out) if a.out else Path(tempfile.mkdtemp(prefix="rhmp_ex05_"))
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    # ------------------------------------------------------------------ 1. one complex per sample
    t0 = time.time()
    Ks, gs = [], []
    for _ in range(a.meshes):
        pos, faces = square_mesh(rng, int(rng.integers(7, 13)))
        K = CochainComplex.from_triangles(pos, faces)
        Ks.append(K)
        gs.append(torch.tensor(boundary_values(rng, pos, K.boundary[0].numpy()), dtype=torch.float32)[:, None])
    print(f"{a.meshes} complexes built in {time.time() - t0:.2f} s; vertices per mesh "
          f"{min(K.n[0] for K in Ks)}..{max(K.n[0] for K in Ks)}")

    # ------------------------------------------------------------------ 2. what a block-diagonal batch looks like
    Kb = CochainComplex.batch(Ks[:4])
    print(f"\nbatch of 4 meshes: {Kb}")
    print(f"  meta['sizes'] (cells per mesh and degree) = {Kb.meta['sizes']}")
    print(f"  meta['ptr'][0] (vertex offsets) = {Kb.meta['ptr'][0].tolist()}, "
          f"K.batch[0] counts = {torch.bincount(Kb.batch[0]).tolist()}, |d d| = {Kb.check_d2()}")

    # ------------------------------------------------------------------ 3. targets: one CG per mesh on a batch
    t0 = time.time()
    targets = []
    for s in range(0, a.meshes, 60):                              # solve 60 meshes per batch
        Kc = CochainComplex.batch(Ks[s:s + 60]).to(dtype=torch.float64)
        u, res = harmonic_extension(Kc, torch.cat(gs[s:s + 60]).double())
        ptr = Kc.meta["ptr"][0].tolist()
        targets += [u[ptr[i]:ptr[i + 1]].float() for i in range(len(ptr) - 1)]
    print(f"\nharmonic extensions of all meshes by per-graph CG in {time.time() - t0:.2f} s "
          f"(last batch: {res.shape[0] - 1} iterations, max relative residual {float(res[-1].max()):.1e})")
    K0 = Ks[0].to(dtype=torch.float64)                             # check one mesh against a separate solve
    u0, _ = harmonic_extension(K0, gs[0].double())
    print(f"  mesh 0 solved alone vs inside the batch: max difference {float((u0.float() - targets[0]).abs().max()):.1e}")

    # ------------------------------------------------------------------ 4. batch independence of the model
    cfg = RHMPConfig(in_dims={0: 1}, C=32, n_layers=4, readout="node_scalar")
    model = RHMP(cfg, Ks[0].geo_dims)
    model.eval()
    with torch.no_grad():
        y_batch = model({0: torch.cat(gs[:4])[:, None]}, Kb)       # inputs (sum n0, 1, F): B = 1
        ptr = Kb.meta["ptr"][model.output_degree].tolist()
        diffs = [float((y_batch[ptr[i]:ptr[i + 1]] - model({0: gs[i][:, None]}, Ks[i])).abs().max())
                 for i in range(4)]
    print(f"\nmodel output per mesh, batched vs alone: max |difference| = {max(diffs):.1e}")
    with torch.no_grad():
        t0 = time.time()
        for i in range(32):
            model({0: gs[i][:, None]}, Ks[i])
        t_loop = time.time() - t0
        Kb32 = CochainComplex.batch(Ks[:32])
        t0 = time.time()
        model({0: torch.cat(gs[:32])[:, None]}, Kb32)
        t_batch = time.time() - t0
    print(f"  inference on 32 meshes: loop {t_loop * 1e3:.0f} ms, one block-diagonal batch {t_batch * 1e3:.0f} ms")

    # ------------------------------------------------------------------ 5. a variable-mesh TaskData + the trainer
    split = sequential_split(a.meshes)
    x_stats = {0: scale_stats([gs[i] for i in split[0].tolist()])}      # scale only: interior zeros stay 0
    y_stats = feature_stats([targets[i] for i in split[0].tolist()])
    td = TaskData(
        name="harmonic", K=Ks,                                     # list of complexes = variable meshes
        inputs=[{0: x_stats[0].normalize(g)} for g in gs],       # list of {k: (n_k_i, F_k)}
        target=[y_stats.normalize(u) for u in targets],          # list of (n_t_i, out_dim)
        target_degree=0, target_kind="node_scalar",
        in_dims={0: 1}, even_dims={}, connection_dims={},
        split=split, x_stats=x_stats, y_stats=y_stats, spatial_dim=2, out_dim=1,
        meta={"batch_size": 8},                                    # meshes per block-diagonal batch
    )
    Kmb, xmb, ymb = mesh_minibatch(td.K, td.inputs, td.target, [0, 1, 2])   # what the trainer feeds the model
    print(f"\nmesh_minibatch of 3 samples: {Kmb.num_graphs} graphs, inputs {tuple(xmb[0].shape)}, "
          f"target {tuple(ymb.shape)}")
    args = parse_args(["--task", td.name, "--device", "cpu", "--epochs", str(a.epochs), "--C", "32", "--layers", "4",
                       "--lr", "3e-3", "--bs-meshes", "8", "--seed", str(a.seed), "--out", str(out), "--force",
                       "--no-resume"])
    result = run(args, task=td)
    r2 = result["test"]["R2"]
    print(f"trainer: test R2 {r2:.4f} on {result['test']['N']} held-out meshes")

    assert max(diffs) < 1e-4 and float(res[-1].max()) < 1e-6
    print(f"EXAMPLE 05 OK  test_R2={r2:.4f}")


if __name__ == "__main__":
    main()
