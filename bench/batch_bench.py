"""Variable-mesh throughput: per-sample loop vs block-diagonal batches (T8 AirfRANS meshes by default).

Modes (training = forward + backward + Adam step, samples/s over ``--samples`` samples after warm-up):
  loop1         one mesh per step (the v1 T8 protocol: batch size 1)
  loop_acc<b>   one mesh per forward/backward, gradients accumulated over b meshes, one step per b meshes
                (same optimisation semantics as a block batch of b, sequential execution)
  block<b>      block-diagonal batch of b meshes (``CochainComplex.batch``), one step per batch
  v1_loop1      v1 GaugeHodgeNetwork (local_rich metric, per-sample, as formal_benchmark.py) for reference
Inference: per-sample loop vs block batches.  The batch-building time (CochainComplex.batch + input concatenation)
is included in the block timings and also reported separately.  A consistency check verifies that the block-batch
loss equals the size-weighted per-sample loss.

    python3 bench/batch_bench.py                          # T8, C=256, L=4, b in 4/8/16
    python3 bench/batch_bench.py --task HP_k100 --C 128   # hetero-Poisson meshes (1-2K nodes, variable sizes)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bench"))

from rhmp.data import mesh_minibatch  # noqa: E402

DEV = "cuda"


def _sync():
    torch.cuda.synchronize()


def build_model(td, C, L, amp=False):
    from rhmp.model import RHMP, RHMPConfig
    cfg = RHMPConfig(in_dims=td.in_dims, even_dims=td.even_dims, connection_dims=td.connection_dims, C=C,
                     n_layers=L, readout=td.readout, out_dim=td.out_dim, amp=amp)
    torch.manual_seed(0)
    m = RHMP(cfg, td.geo_dims).to(DEV)
    m.record_diagnostics = False
    return m


def _wait():
    from step_bench import wait_exclusive
    return wait_exclusive()


def train_throughput(td, model, ids, mode: str, warm: int = 2) -> tuple[float, float]:
    """Samples per second and peak GB for a training pass over ``ids`` in the given mode."""
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    if mode == "loop1":
        groups, acc = [[i] for i in ids], 1
    elif mode.startswith("loop_acc"):
        b = int(mode[len("loop_acc"):])
        groups, acc = [ids[s:s + b] for s in range(0, len(ids), b)], b
    elif mode.startswith("block"):
        b = int(mode[len("block"):])
        groups, acc = [ids[s:s + b] for s in range(0, len(ids), b)], 0
    else:
        raise ValueError(mode)

    def run_group(g):
        opt.zero_grad(set_to_none=True)
        if acc == 0:
            K, xb, yb = mesh_minibatch(td.K, td.inputs, td.target, g, device=DEV)
            F.mse_loss(model(xb, K), yb).backward()
        else:
            tot = sum(td.target[i].numel() for i in g)
            for i in g:
                K, xb, yb = mesh_minibatch(td.K, td.inputs, td.target, [i], device=DEV)
                (F.mse_loss(model(xb, K), yb) * (yb.numel() / tot)).backward()
        opt.step()

    for g in groups[:warm]:
        run_group(g)
    _sync()
    _wait()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    n = 0
    for g in groups[warm:]:
        run_group(g)
        n += len(g)
    _sync()
    return n / (time.perf_counter() - t0), torch.cuda.max_memory_allocated() / 2 ** 30


@torch.no_grad()
def infer_throughput(td, model, ids, b: int) -> float:
    model.eval()
    groups = [ids[s:s + b] for s in range(0, len(ids), b)]
    for g in groups[:2]:
        K, xb, _ = mesh_minibatch(td.K, td.inputs, td.target, g, device=DEV)
        model(xb, K)
    _sync()
    _wait()
    t0 = time.perf_counter()
    n = 0
    for g in groups[2:]:
        K, xb, _ = mesh_minibatch(td.K, td.inputs, td.target, g, device=DEV)
        model(xb, K)
        n += len(g)
    _sync()
    model.train()
    return n / (time.perf_counter() - t0)


def batch_build_ms(td, ids, b: int, reps: int = 20) -> float:
    groups = [ids[s:s + b] for s in range(0, len(ids), b)][:reps]
    _sync()
    t0 = time.perf_counter()
    for g in groups:
        mesh_minibatch(td.K, td.inputs, td.target, g, device=DEV)
    _sync()
    return (time.perf_counter() - t0) / len(groups) * 1e3


def v1_loop(task: str, ids, td, C: int, L: int) -> tuple[float, float]:
    """v1 per-sample training throughput (T8 recipe: local_rich metric)."""
    import pickle
    from rhmp.baselines.v1.gauge_hodge_mp.cell_complex import CellComplex
    from rhmp.baselines.v1.gauge_hodge_mp.network import GaugeHodgeNetwork
    raw = pickle.load(open(os.path.join(ROOT, "datasets", "T8_airfoil_pressure_persample.pkl"), "rb"))
    Ks = {i: CellComplex.from_triangulation(torch.tensor(np.asarray(raw[i]["pts_3d"], dtype=np.float32)),
                                            torch.tensor(np.asarray(raw[i]["faces"], dtype=np.int64))).to(DEV)
          for i in ids}
    m = GaugeHodgeNetwork(f_in=td.in_dims[0], C=C, n_layers=L, n0=2000, n1=1, n2=1, task="scalar", spatial_dim=2,
                          out_dim=1, mp_hidden=16, metric_type="local_rich", metric_rank=0).to(DEV)
    opt = torch.optim.Adam(m.parameters(), 1e-3)

    def step(i):
        opt.zero_grad(set_to_none=True)
        F.mse_loss(m(td.inputs[i][0], Ks[i]), td.target[i]).backward()
        opt.step()

    for i in ids[:2]:
        step(i)
    _sync()
    _wait()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for i in ids[2:]:
        step(i)
    _sync()
    return (len(ids) - 2) / (time.perf_counter() - t0), torch.cuda.max_memory_allocated() / 2 ** 30


def consistency(td, model, ids) -> float:
    """|block-batch loss - size-weighted per-sample loss| / loss for one batch."""
    with torch.no_grad():
        K, xb, yb = mesh_minibatch(td.K, td.inputs, td.target, ids, device=DEV)
        lb = F.mse_loss(model(xb, K), yb).item()
        tot = sum(td.target[i].numel() for i in ids)
        ll = 0.0
        for i in ids:
            K, xb, yb = mesh_minibatch(td.K, td.inputs, td.target, [i], device=DEV)
            ll += F.mse_loss(model(xb, K), yb).item() * yb.numel() / tot
    return abs(lb - ll) / max(abs(ll), 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="T8")
    ap.add_argument("--C", type=int, default=None)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--samples", type=int, default=96)
    ap.add_argument("--batches", default="4,8,16")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--no-v1", action="store_true")
    ap.add_argument("--data-on", default="auto", choices=["auto", "device", "cpu"])
    ap.add_argument("--json", default=os.path.join(ROOT, "runs", "bench", "batch_bench.json"))
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    from rhmp.tasks import load_task, task_defaults
    kw = {} if a.data_on == "auto" else {"keep_on": DEV if a.data_on == "device" else "cpu"}
    td = load_task(a.task, ROOT, native=True, device=DEV, fine=False,
                   max_samples=(3 * (a.samples + 16) if a.task != "T8" else None), **kw)
    C = a.C or task_defaults(a.task)["C"]
    ids = td.split[0][: a.samples + 16].tolist()
    sizes = [td.K[i].n[0] for i in ids]
    print(f"GPU {torch.cuda.get_device_name(0)} | task {a.task} C={C} L={a.layers} amp={a.amp} | "
          f"{len(ids)} meshes, n0 {min(sizes)}..{max(sizes)}")
    model = build_model(td, C, a.layers, a.amp)
    rows = []
    rel = consistency(td, model, ids[:8])
    print(f"consistency: |loss(block8) - weighted per-sample loss| / loss = {rel:.2e}")
    bs = [int(b) for b in a.batches.split(",")]
    modes = ["loop1"] + [f"loop_acc{b}" for b in bs[:1]] + [f"block{b}" for b in bs]
    print("| mode | train samples/s | speed-up vs loop1 | peak GB | infer samples/s | batch build ms |\n|---|---|---|---|---|---|")
    base = None
    for mode in modes:
        model = build_model(td, C, a.layers, a.amp)
        sps, mem = train_throughput(td, model, ids[: a.samples + 16], mode)
        b = 1 if mode.startswith("loop") else int(mode[5:])
        isps = infer_throughput(td, model, ids, b)
        bms = batch_build_ms(td, ids, b) if b > 1 else batch_build_ms(td, ids, 1)
        base = base or sps
        rows.append(dict(mode=mode, train_sps=sps, speedup=sps / base, peak_GB=mem, infer_sps=isps, build_ms=bms))
        print(f"| {mode} | {sps:.1f} | {sps / base:.2f}x | {mem:.2f} | {isps:.1f} | {bms:.2f} |", flush=True)
        del model
        torch.cuda.empty_cache()
    if a.task == "T8" and not a.no_v1:
        sps, mem = v1_loop(a.task, ids[:34], td, C, a.layers)
        rows.append(dict(mode="v1_loop1", train_sps=sps, speedup=sps / base, peak_GB=mem))
        print(f"| v1_loop1 (GaugeHodgeNetwork local_rich) | {sps:.1f} | {sps / base:.2f}x | {mem:.2f} | - | - |")
    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    json.dump({"task": a.task, "C": C, "L": a.layers, "amp": a.amp, "gpu": torch.cuda.get_device_name(0),
               "consistency_rel": rel, "rows": rows}, open(a.json, "w"), indent=2)
    print(f"saved {a.json}")


if __name__ == "__main__":
    main()
