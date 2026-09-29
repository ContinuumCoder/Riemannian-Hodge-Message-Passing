"""Example 04: wrap your own data in a ``TaskData`` and train it with the library trainer (``rhmp.train.run``).

The custom task lives on one shared mesh (``N`` samples, one complex) and uses inputs on two degrees:

* edges (degree 1): a U(1) gauge connection ``theta_e`` (odd, orientation ``src -> dst``) declared as a
  *connection* column, so it enters the network only through ``d_1 theta``;
* faces (degree 2): a positive coupling ``beta_f`` given as the *even* column ``log beta_f``;
* target on faces (degree 2): ``y_f = beta_f * sin((d_1 theta)_f)``, the coupling times the imaginary part of the
  plaquette variable ``exp(i (d_1 theta)_f)``.  It is an odd face cochain (it flips sign with the face orientation)
  and gauge invariant (``theta -> theta + d_0 lambda`` leaves it unchanged).

The trainer runs the v1 protocol (Adam, cosine schedule, gradient clipping, model selection on the validation split),
writes ``config.json``, ``history.json``, ``best.pt``, ``last.pt`` and ``result.json`` to ``--out``, and reports
R2 / NRMSE / SSIM / Pearson.  Afterwards the script reloads ``best.pt`` and checks the exact gauge invariance of the
trained model.

Run (CPU, about 10-30 s):  python examples/04_custom_task.py [--epochs 8] [--out DIR]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import Delaunay

try:
    import rhmp  # noqa: F401
except ImportError:  # running from a source checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rhmp import RHMP, CochainComplex
from rhmp.data import TaskData, feature_stats, scale_stats, sequential_split, shared_minibatch
from rhmp.train import parse_args, run


def square_mesh(rng: np.random.Generator, m: int = 12):
    """Delaunay mesh of the unit square (``m`` boundary points per side, jittered interior grid)."""
    s = np.linspace(0.0, 1.0, m + 1)[:-1]
    boundary = np.concatenate([np.c_[s, 0 * s], np.c_[1 + 0 * s, s], np.c_[1 - s, 1 + 0 * s], np.c_[0 * s, 1 - s]])
    g = np.arange(1, m) / m
    ii, jj = np.meshgrid(g, g, indexing="ij")
    interior = np.c_[ii.ravel(), jj.ravel()] + rng.uniform(-0.35, 0.35, (g.size ** 2, 2)) / m
    pos = np.concatenate([boundary, interior])
    return pos, Delaunay(pos).simplices


def build_task(n_samples: int, seed: int) -> TaskData:
    """Generate the data and return a normalised, CPU-resident ``TaskData``."""
    rng = np.random.default_rng(seed)
    K = CochainComplex.from_triangles(*square_mesh(rng))
    theta = torch.tensor(rng.normal(0.0, 0.6, (K.n[1], n_samples)), dtype=torch.float32)   # (n1, N)
    log_beta = torch.tensor(rng.normal(0.0, 0.5, (K.n[2], n_samples)), dtype=torch.float32)  # (n2, N)
    flux = K.apply_d(1, theta)                                                   # (n2, N) = d1 theta
    y = torch.exp(log_beta) * torch.sin(flux)

    # TaskData layout for a shared mesh: inputs[k] (N, n_k, F_k), target (N, n_t, out_dim)
    raw = {1: theta.T[..., None], 2: log_beta.T[..., None]}
    target = y.T[..., None]
    split = sequential_split(n_samples)                                          # first 70 % train, 15 % val, 15 % test
    tr = split[0]
    x_stats = {1: scale_stats(raw[1][tr]),      # odd / connection column: scale only (a shift would break the gauge
               2: feature_stats(raw[2][tr])}    # symmetry and the orientation parity); even column: mean/std
    y_stats = scale_stats(target[tr])           # odd target: scale only
    return TaskData(
        name="plaquette", K=K,
        inputs={k: x_stats[k].normalize(v) for k, v in raw.items()},
        target=y_stats.normalize(target),
        target_degree=2, target_kind="cochain",  # -> readout 'cochain:2' (bias-free, odd)
        in_dims={1: 1, 2: 1}, even_dims={2: 1}, connection_dims={1: 1},
        split=split, x_stats=x_stats, y_stats=y_stats, spatial_dim=2, out_dim=1,
        meta={"batch_size": 16},                 # samples per step (``--batch`` overrides it)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="run directory (default: a temporary directory)")
    a = ap.parse_args()
    out = Path(a.out) if a.out else Path(tempfile.mkdtemp(prefix="rhmp_ex04_"))

    td = build_task(a.samples, a.seed)
    s = td.summary()
    print(f"task {s['name']}: N={s['N']}, cells {s['cells']}, inputs {s['inputs']}, target {s['target']}, "
          f"readout {s['readout']}, split {s['split']}")

    # the CLI parser gives the trainer its defaults; ``--task`` only names the run for a custom TaskData
    args = parse_args(["--task", td.name, "--device", "cpu", "--epochs", str(a.epochs), "--C", "32",
                       "--layers", "3", "--lr", "3e-3", "--seed", str(a.seed), "--out", str(out), "--force",
                       "--no-resume"])
    result = run(args, task=td)
    print(f"trainer result: best epoch {result['best_epoch']}, test R2 {result['test']['R2']:.4f} "
          f"(uncentred R2 for an odd target), NRMSE {result['test']['NRMSE']:.4f}")
    print(f"run directory {out}: {sorted(p.name for p in out.iterdir())}")

    # reload the best model and check the exact gauge invariance on the test split
    model = RHMP.from_checkpoint(str(out / "best.pt"))
    model.eval()
    x, y = shared_minibatch(td.inputs, td.target, td.split[2])                   # {k: (n_k, B, F_k)}, (n2, B, 1)
    K = td.K
    lam = 3.0 * torch.randn(K.n[0], x[1].shape[1], 1, generator=torch.Generator().manual_seed(1))
    x_gauge = {**x, 1: x[1] + K.apply_d(0, lam)}                                 # theta -> theta + d0 lambda
    x_noise = {**x, 1: x[1] + 0.1 * torch.randn(x[1].shape, generator=torch.Generator().manual_seed(2))}
    with torch.no_grad():
        p, p_gauge, p_noise = model(x, K), model(x_gauge, K), model(x_noise, K)
    rel_gauge = float((p_gauge - p).norm() / p.norm())
    rel_noise = float((p_noise - p).norm() / p.norm())
    print(f"relative change of the prediction: gauge transformation {rel_gauge:.1e} (exact invariance), "
          f"non-exact perturbation of theta {rel_noise:.1e}")
    y_phys = td.y_stats.denormalize(p)                                           # back to physical units
    print(f"physical-unit predictions for the test split: {tuple(y_phys.shape)} = (n2, samples, 1)")
    assert rel_gauge < 1e-4 < rel_noise
    print(f"EXAMPLE 04 OK  test_R2={result['test']['R2']:.4f}")


if __name__ == "__main__":
    main()
