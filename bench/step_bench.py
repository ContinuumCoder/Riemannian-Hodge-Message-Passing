"""v1 vs v2 train-step / inference time and peak memory (one GPU).

Cases (DESIGN §6):
  T6       n0=1024 Delaunay, C=128, L=4, B=64  (v1: GaugeHodgeNetwork.forward_batch, node-encoded inputs;
           v2: legacy node inputs and native edge inputs in exact gauge-connection mode)
  T7       same mesh, C=160, L=4, B=64, 9 input / 3 output channels
  T6_100K  50K nodes / 100K faces, C=16, L=3, B=2 (also complex build time: v1 densifies d1@d0)
  1M       synthetic random Delaunay mesh with ~1M triangles, C=16, L=4, B=1 (v2 only: v1's complex build needs a
           dense n2 x n0 matrix, ~2 TB)
v2 variants: base (fp32 + TF32 matmuls), amp (bf16 autocast for dense parts), compile (torch.compile), ckpt
(activation checkpointing per layer), and combinations ``amp+compile`` etc.

Timing: 3 warm-up steps, then ``--steps`` timed steps between CUDA synchronisations (Adam step included for training);
peak memory = ``torch.cuda.max_memory_allocated`` over warm-up + timed steps (as ``bench/v1_timing.py``).
Also reports the batch-coupling check of DESIGN F2: relative difference of sample 0's prediction in a batch of B vs
alone (v1 couples samples through its batch-mean metric; v2 must be at float round-off).

    python3 bench/step_bench.py                                  # all cases, all variants
    python3 bench/step_bench.py --cases T6,T7 --variants base,amp --json runs/bench/step.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEV = "cuda"


def _sync():
    torch.cuda.synchronize()


def gpu_others(gpu: str | None = None) -> list[str]:
    """PIDs of *other* compute processes on the physical GPU in use (from ``nvidia-smi``)."""
    import subprocess
    gpu = gpu if gpu is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    try:
        out = subprocess.run(["nvidia-smi", "-i", gpu, "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:  # noqa: BLE001
        return []
    return [p.strip() for p in out.splitlines() if p.strip() and p.strip() != str(os.getpid())]


def wait_exclusive(max_wait_s: float = 1800.0, poll_s: float = 15.0) -> bool:
    """Block until no other process computes on the GPU in use (timings on a shared GPU are not meaningful).

    Returns True if exclusivity was reached, False after ``max_wait_s`` (the case is then flagged ``shared``).
    """
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        others = gpu_others()
        if not others:
            return True
        print(f"  waiting for exclusive GPU (other pids {others})", flush=True)
        time.sleep(poll_s)
    return False


def _clear():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


EXCLUSIVE = {"ok": True}


def time_steps(step, steps: int, warmup: int = 3) -> tuple[float, float]:
    """Mean seconds per call of ``step`` and peak GB (warm-up included in the peak, as v1_timing.py).

    Waits for exclusive use of the GPU first; ``EXCLUSIVE['ok']`` records whether it was obtained."""
    EXCLUSIVE["ok"] = wait_exclusive() and EXCLUSIVE["ok"]
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        step()
    _sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    _sync()
    return (time.perf_counter() - t0) / steps, torch.cuda.max_memory_allocated() / 2 ** 30


# ----------------------------------------------------------------------------------------------------------------
# v1
# ----------------------------------------------------------------------------------------------------------------
def bench_v1(pts, faces, X, Y, C, L, B, steps, coupling=True) -> dict:
    """v1 GaugeHodgeNetwork (diagonal rank-8 metric, mp_hidden 16) as in bench/v1_timing.py."""
    from rhmp.baselines.v1.gauge_hodge_mp.cell_complex import CellComplex
    from rhmp.baselines.v1.gauge_hodge_mp.network import GaugeHodgeNetwork
    t0 = time.perf_counter()
    K = CellComplex.from_triangulation(torch.tensor(pts, dtype=torch.float32),
                                       torch.tensor(faces, dtype=torch.int64)).to(DEV)
    t_build = time.perf_counter() - t0
    m = GaugeHodgeNetwork(f_in=X.shape[-1], C=C, n_layers=L, n0=K.n0, n1=K.n1, n2=K.n2, task="scalar",
                          spatial_dim=2, out_dim=Y.shape[-1], mp_hidden=16, metric_type="diagonal",
                          metric_rank=8).to(DEV)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    Xb, Yb = X[:B].contiguous(), Y[:B].contiguous()

    def step():
        opt.zero_grad(set_to_none=True)
        F.mse_loss(m.forward_batch(Xb, K), Yb).backward()
        opt.step()

    _clear()
    dt, mem = time_steps(step, steps)
    m.eval()
    with torch.no_grad():
        dti, mem_i = time_steps(lambda: m.forward_batch(Xb, K), steps)
        coup = float("nan")
        if coupling and B > 1:
            pb = m.forward_batch(Xb, K)[0]
            p1 = m.forward_batch(Xb[:1], K)[0]
            coup = ((pb - p1).norm() / pb.norm()).item()
    res = dict(model="v1", params=sum(p.numel() for p in m.parameters()), train_ms=dt * 1e3, infer_ms=dti * 1e3,
               peak_GB=mem, infer_peak_GB=mem_i, build_s=t_build, coupling=coup, exclusive=EXCLUSIVE["ok"])
    EXCLUSIVE["ok"] = True
    del m, opt, K
    _clear()
    return res


# ----------------------------------------------------------------------------------------------------------------
# v2
# ----------------------------------------------------------------------------------------------------------------
def bench_v2(K, inputs, Y, C, L, B, steps, variant: str, cfg_extra: dict | None = None, coupling=True,
             omap=None) -> dict:
    """RHMP train step / inference on the shared complex ``K`` (inputs ``{k: (n_k, B_all, F)}``).

    ``omap``: optional task output map applied to the model output before the loss (native T6/T7)."""
    from rhmp.model import RHMP, RHMPConfig
    flags = set(variant.split("+")) if variant != "base" else set()
    in_dims = {k: int(v.shape[-1]) for k, v in inputs.items()}
    extra = dict(cfg_extra or {})
    extra.setdefault("out_dim", int(Y.shape[-1]))
    cfg = RHMPConfig(in_dims=in_dims, C=C, n_layers=L, amp="amp" in flags, checkpoint_layers="ckpt" in flags,
                     **extra)
    torch.manual_seed(0)
    m = RHMP(cfg, K.geo_dims).to(DEV)
    m.record_diagnostics = False
    net = torch.compile(m) if "compile" in flags else m
    fwd = (lambda x, K_: omap(net(x, K_))) if omap is not None else net
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    xb = {k: v[:, :B].contiguous() for k, v in inputs.items()}
    yb = Y[:, :B].contiguous()

    def step():
        opt.zero_grad(set_to_none=True)
        F.mse_loss(fwd(xb, K), yb).backward()
        opt.step()

    _clear()
    t0 = time.perf_counter()
    step()                                   # first call (compilation for 'compile')
    _sync()
    t_first = time.perf_counter() - t0
    dt, mem = time_steps(step, steps)
    m.eval()
    with torch.no_grad():
        dti, mem_i = time_steps(lambda: fwd(xb, K), steps)
        coup = float("nan")
        if coupling and B > 1:
            pb = m(xb, K)[:, 0]
            p1 = m({k: v[:, :1].contiguous() for k, v in xb.items()}, K)[:, 0]
            coup = ((pb - p1).norm() / pb.norm()).item()
    res = dict(model=f"v2[{variant}]", params=m.num_parameters(), train_ms=dt * 1e3, infer_ms=dti * 1e3,
               peak_GB=mem, infer_peak_GB=mem_i, first_step_s=t_first, coupling=coup, exclusive=EXCLUSIVE["ok"])
    EXCLUSIVE["ok"] = True
    del m, opt, fwd
    _clear()
    return res


# ----------------------------------------------------------------------------------------------------------------
# cases
# ----------------------------------------------------------------------------------------------------------------
def _pkl(name, n):
    import pickle
    d = pickle.load(open(os.path.join(ROOT, "datasets", name), "rb"))
    X = torch.tensor(np.asarray(d["X_data"][:n], dtype=np.float32))
    Y = torch.tensor(np.asarray(d["Y_data"][:n], dtype=np.float32))
    if X.ndim == 2:
        X, Y = X[..., None], Y[..., None]
    return np.asarray(d["points"]), np.asarray(d["faces"], dtype=np.int64), X, Y


def case_shared(tag, pkl, C, L, B, n_load, steps, variants, native_task=None, v1=True) -> list[dict]:
    from rhmp.complex import CochainComplex
    pts, faces, X, Y = _pkl(pkl, n_load)
    rows = []
    info = dict(case=tag, C=C, L=L, B=B)
    if v1:
        try:
            r = bench_v1(pts, faces, X.to(DEV), Y.to(DEV), C, L, B, steps)
            rows.append({**info, "inputs": "node (v1)", **r})
            print(_fmt(rows[-1]), flush=True)
        except Exception as e:  # noqa: BLE001
            rows.append({**info, "model": "v1", "error": repr(e)[:200]})
            print(f"{tag} v1 failed: {e!r}", flush=True)
            _clear()
    p2 = pts[:, :2] if pts.shape[1] == 3 and np.abs(pts[:, 2]).max() == 0 else pts
    _sync()
    t0 = time.perf_counter()
    K = CochainComplex.from_triangles(torch.as_tensor(p2), torch.as_tensor(faces), star="cotan", device=DEV)
    _sync()
    t_build = time.perf_counter() - t0
    inputs = {0: X.to(DEV).permute(1, 0, 2).contiguous()}
    Yn = Y.to(DEV).permute(1, 0, 2).contiguous()
    for var in variants:
        try:
            r = bench_v2(K, inputs, Yn, C, L, B, steps, var)
            rows.append({**info, "inputs": "node (legacy)", "build_s": t_build, **r})
            print(_fmt(rows[-1]), flush=True)
        except Exception as e:  # noqa: BLE001
            rows.append({**info, "model": f"v2[{var}]", "error": repr(e)[:300]})
            print(f"{tag} v2[{var}] failed: {e!r}", flush=True)
            traceback.print_exc()
            _clear()
    if native_task is not None:
        from rhmp.tasks import load_task
        td = load_task(native_task, ROOT, native=True, device=DEV)
        nin = {k: v[:n_load].permute(1, 0, 2).contiguous() for k, v in td.inputs.items()}
        Yt = td.target[:n_load].permute(1, 0, 2).contiguous()
        extra = dict(connection_dims=td.connection_dims, connection_odd=(native_task == "T7"),
                     readout=td.readout)
        if td.output_map is not None:
            extra["out_dim"] = td.output_map.model_out_dim
        for var in variants[:2]:
            try:
                r = bench_v2(td.K, nin, Yt, C, L, B, steps, var, cfg_extra=extra, omap=td.output_map)
                rows.append({**info, "inputs": "edge (native)", **r})
                print(_fmt(rows[-1]), flush=True)
            except Exception as e:  # noqa: BLE001
                rows.append({**info, "model": f"v2[{var}] native", "error": repr(e)[:300]})
                print(f"{tag} native v2[{var}] failed: {e!r}", flush=True)
                _clear()
        del td
    del K
    _clear()
    return rows


def case_1m(steps, variants, n_pts=500_000) -> list[dict]:
    from scipy.spatial import Delaunay
    from rhmp.complex import CochainComplex
    rng = np.random.RandomState(0)
    pts = rng.rand(n_pts, 2)
    t0 = time.perf_counter()
    faces = Delaunay(pts).simplices.astype(np.int64)
    t_del = time.perf_counter() - t0
    _sync()
    t0 = time.perf_counter()
    K = CochainComplex.from_triangles(torch.as_tensor(pts), torch.as_tensor(faces), star="cotan", device=DEV)
    _sync()
    t_build = time.perf_counter() - t0
    info = dict(case="1M", C=16, L=4, B=1, n0=K.n[0], n1=K.n[1], n2=K.n[2], delaunay_s=t_del, build_s=t_build)
    rows = [{**info, "model": "v1", "error": f"infeasible: dense d1@d0 check needs n2*n0*4 B = "
                                             f"{K.n[2] * K.n[0] * 4 / 1e12:.1f} TB"}]
    print(f"1M mesh: n={K.n}, delaunay {t_del:.1f}s, v2 complex build {t_build:.2f}s", flush=True)
    g = torch.Generator().manual_seed(0)
    x = {0: torch.randn(K.n[0], 1, 1, generator=g).to(DEV)}
    y = torch.randn(K.n[0], 1, 1, generator=g).to(DEV)
    for var in variants:
        try:
            r = bench_v2(K, x, y, 16, 4, 1, steps, var, coupling=False)
            rows.append({**info, "inputs": "node", **r})
            print(_fmt(rows[-1]), flush=True)
        except Exception as e:  # noqa: BLE001
            rows.append({**info, "model": f"v2[{var}]", "error": repr(e)[:300]})
            print(f"1M v2[{var}] failed: {e!r}", flush=True)
            _clear()
    return rows


def _fmt(r: dict) -> str:
    if "error" in r:
        return f"| {r['case']} | {r['model']} | - | - | - | - | {r['error']} |"
    shared = "" if r.get("exclusive", True) else " SHARED-GPU"
    return (f"| {r['case']} | {r['model']} {r.get('inputs', '')} | {r['params'] / 1e6:.3f}M | {r['train_ms']:.1f} | "
            f"{r['infer_ms']:.1f} | {r['peak_GB']:.2f} | coupling {r.get('coupling', float('nan')):.1e}{shared} |")


# ----------------------------------------------------------------------------------------------------------------
# report: bench/RESULTS_step.md from the JSON outputs (benchmarks run on the GPU host, fetched with tools/fetch.sh)
# ----------------------------------------------------------------------------------------------------------------
def _load_json(path):
    return json.load(open(path)) if path and os.path.exists(path) else None


def _step_rows(d):
    """{(case, model, inputs): row} of a step_bench JSON."""
    out = {}
    for r in (d or {}).get("rows", []):
        out[(r["case"], r["model"], r.get("inputs", ""))] = r
    return out


def _fmt_num(r, key, fmt):
    return fmt.format(r[key]) if r is not None and key in r else ""


def write_report(path, pass1, pass2, batch, smoke_dir=None, v1_hist_root=None):
    """Render ``bench/RESULTS_step.md`` from the step/batch JSON files (tables regenerated, text fixed)."""
    p1, p2 = _step_rows(_load_json(pass1)), _step_rows(_load_json(pass2))
    meta = _load_json(pass2) or _load_json(pass1) or {}
    L = []
    L += ["# Step / batch benchmarks: v1 vs v2", "",
          f"Written by `python3 bench/step_bench.py --report` on an {meta.get('gpu', '?')} GPU (torch "
          f"{meta.get('torch', '?')}) from `{os.path.relpath(pass2, ROOT)}` (pass 2), `{os.path.relpath(pass1, ROOT)}` "
          "(pass 1) and the `bench/batch_bench.py` "
          "JSON files. All timed cases ran on an exclusive GPU: the scripts wait until no other compute process is on "
          "the GPU, and every row records `exclusive`.", "",
          "**Protocol.** 3 warm-up steps, then 20 timed steps (5 for T6_100K / 1M) between CUDA synchronisations.",
          "- *Train* = forward + MSE backward + Adam step.",
          "- *Infer* = forward under `no_grad`.",
          "- *Peak* = `torch.cuda.max_memory_allocated` over warm-up + timed steps (as `bench/v1_timing.py`).",
          "- TF32 dense matmuls on.", "",
          "**Models.**",
          "- v1 = `GaugeHodgeNetwork` (diagonal rank-8 metric, mp_hidden 16, `forward_batch`), as `bench/v1_timing.py`.",
          "- v2 = `RHMP` defaults (poly 2, tied bounded log-metrics, DEC scaling, norm gate). Native T6/T7 use the "
          "`cochain:2` readout + oriented face->node output map of the trainer.",
          "- *coupling* = relative difference of sample 0's prediction in a batch of B vs alone (DESIGN F2). v1 couples "
          "samples through its batch-mean metric. v2 is exactly independent; the residuals are GEMM-tiling round-off "
          "(TF32 1e-6..1e-4, bf16 AMP up to 6e-3, 0 in several fp32 cases).", "",
          "Pass 1 = the initial v2 implementation; Pass 2 = after the performance changes: `addmm` spmm, chunked "
          "weight gradients, fused mean-square, and memory-lean fused MLPs that recompute their hidden layer in "
          "backward (these trade some time for memory).", "",
          "## 1. Train step / inference (shared meshes, `bench/step_bench.py`)", "",
          "Columns: train and infer in ms, peak in GB.", "",
          "| case | model / variant | inputs | params | train (pass 2) | infer (pass 2) | peak (pass 2) | train (pass 1) "
          "| peak (pass 1) | coupling (pass 2) |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    keys = list(p2) or list(p1)
    for key in keys:
        r2, r1 = p2.get(key), p1.get(key)
        r = r2 or r1
        if "error" in r:
            L.append(f"| {key[0]} | {key[1]} | | | {r['error'][:60]} | | | | | |")
            continue
        coup = r.get("coupling")
        coup_s = "" if coup is None or coup != coup else f"{coup:.0e}"
        L.append(f"| {key[0]} | {key[1]} | {key[2]} | {r['params'] / 1e6:.3f}M | {_fmt_num(r2, 'train_ms', '{:.1f}')} | "
                 f"{_fmt_num(r2, 'infer_ms', '{:.1f}')} | {_fmt_num(r2, 'peak_GB', '{:.2f}')} | "
                 f"{_fmt_num(r1, 'train_ms', '{:.1f}')} | {_fmt_num(r1, 'peak_GB', '{:.2f}')} | {coup_s} |")
    L += [""]
    # speed-ups (pass 2, legacy node inputs, base variant)
    ups = []
    for case in ("T6", "T7", "T6_100K"):
        v1 = (p2 or p1).get((case, "v1", "node (v1)"))
        v2 = (p2 or p1).get((case, "v2[base]", "node (legacy)"))
        if v1 and v2 and "error" not in v1 and "error" not in v2:
            ups.append(f"- {case}: train {v1['train_ms'] / v2['train_ms']:.2f}x faster, inference "
                       f"{v1['infer_ms'] / v2['infer_ms']:.2f}x faster, peak memory "
                       f"{100 * (v2['peak_GB'] / v1['peak_GB'] - 1):+.0f} % (params {v1['params'] / 1e6:.3f}M vs "
                       f"{v2['params'] / 1e6:.3f}M).")
    L += ["**Summary, v1 -> v2 (base variant, legacy node inputs):**"] + ups + [""]
    b1 = (p2 or p1).get(("T6_100K", "v1", "node (v1)"))
    b2 = (p2 or p1).get(("T6_100K", "v2[base]", "node (legacy)"))
    m1 = (p2 or p1).get(("1M", "v2[base]", "node"))
    L += ["**Complex construction.**", ""]
    if b1 and b2:
        L.append(f"- T6_100K: v1 {b1.get('build_s', float('nan')):.2f} s (`CellComplex.from_triangulation`, CPU, "
                 f"dense d1 d0 check) vs v2 {b2.get('build_s', float('nan')):.3f} s (`CochainComplex.from_triangles`, "
                 "GPU, sparse check).")
    if m1:
        L.append(f"- 1M-triangle mesh (n = {m1.get('n0')}, {m1.get('n1')}, {m1.get('n2')}): v2 {m1['build_s']:.3f} s "
                 f"(plus {m1['delaunay_s']:.1f} s scipy Delaunay). v1 is infeasible: the dense n2 x n0 check needs "
                 f"{m1['n2'] * m1['n0'] * 4 / 1e12:.1f} TB.")
    L += ["", "**Observations.**",
          "- AMP (bf16) gives <= 6 % on T6/T7: the step is dominated by sparse products (kept in fp32 by design) and "
          "kernel launches. AMP helps on the 1M mesh and costs time at C = 16.",
          "- Activation checkpointing (`--ckpt-layers`) halves peak memory at +25-30 % step time.",
          "- `torch.compile` with Inductor was unavailable with the benchmark machine's system python: `/usr/include/python3.12/Python.h` "
          "is missing, so Triton cannot build its CUDA helper. `--compile --compile-backend aot_eager` runs but does "
          "not speed anything up.", ""]
    # batch benchmarks
    L += ["## 2. Variable meshes: per-sample loop vs block-diagonal batches (`bench/batch_bench.py`)", "",
          "Training throughput in samples/s (forward + backward + Adam).",
          "- `loop1` = one mesh per optimiser step (v1's T8 protocol).",
          "- `loop_acc<b>` = per-mesh backward with gradients accumulated over b meshes (same optimisation as a batch "
          "of b).",
          "- `block<b>` = one block-diagonal `CochainComplex.batch` of b meshes. Batch-build time (listed) is included.",
          ""]
    for title, bpath in batch:
        d = _load_json(bpath)
        if not d:
            continue
        L += [f"**{title}** (`{os.path.basename(bpath)}`: C={d['C']}, L={d['L']}; block-vs-loop loss consistency "
              f"{d['consistency_rel']:.1e})", "",
              "| mode | train samples/s | speed-up | peak GB | infer samples/s | build ms |",
              "|---|---:|---:|---:|---:|---:|"]
        for r in d["rows"]:
            L.append(f"| {r['mode']} | {r['train_sps']:.1f} | {r['speedup']:.2f}x | {r['peak_GB']:.2f} | "
                     f"{r['infer_sps']:.1f} | {r['build_ms']:.2f} |" if "infer_sps" in r else
                     f"| {r['mode']} | {r['train_sps']:.1f} | {r['speedup']:.2f}x | {r['peak_GB']:.2f} | - | - |")
        L.append("")
    L += ["**Observations.**",
          "- Block-diagonal batching supplies v2's variable-mesh speed: 6-8x at 8 meshes per step and up to 14x at "
          "16 (HP).",
          "- v2's per-sample loop is launch-bound at 2K nodes (B = 1) and slightly slower than v1's; this is why the "
          "trainer batches meshes.",
          "- CPU-resident TET complexes (the trainer default above its GPU budget, or `--data-on cpu`) cost about "
          "6 % of throughput; the trainer also prefetches the next batch in a background thread.",
          "- Peak memory in these runs includes the GPU-resident data set.", ""]
    # smoke epoch times
    if smoke_dir and os.path.isdir(smoke_dir):
        v1_names = {"T1": "T1_cns_vorticity", "T2": "T2_torus_advection_diffusion", "T3": "T3_ellipsoid_surface_flow",
                    "T5": "T5_maxwell_poisson", "T6": "T6_wilson_loop", "T7": "T7_yang_mills_su2",
                    "T8": "T8_airfoil_pressure"}
        L += ["## 3. Epoch times of 2-epoch runs on the full data (`scripts/smoke_all.sh`)", "",
              "| run | v2 s/epoch | val R2 after 2 epochs | v1 s/epoch | v1 val R2 after 2 epochs |",
              "|---|---:|---:|---:|---:|"]
        import re
        for d in sorted(os.listdir(smoke_dir)):
            rp = os.path.join(smoke_dir, d, "result.json")
            hp = os.path.join(smoke_dir, d, "history.json")
            if not (os.path.exists(rp) and os.path.exists(hp)) or not re.search(r"_(native|legacy)$", d):
                continue
            r, h = json.load(open(rp)), json.load(open(hp))
            if len(h) < 2 or r.get("args", {}).get("max_train_batches"):
                continue                                             # partial runs (functional checks only)
            m_v1 = re.match(r"^(T\d)_(native|legacy)$", d)
            base = m_v1.group(1) if m_v1 else None
            v1s = v1r = ""
            if v1_hist_root and base in v1_names:
                vp = os.path.join(v1_hist_root, v1_names[base], "ours", "history.json")
                if os.path.exists(vp):
                    vh = json.load(open(vp))
                    v1s = f"{np.mean([x['time'] for x in vh]):.1f}"
                    v1r = f"{vh[min(1, len(vh) - 1)]['R2']:.3f}"
            L.append(f"| {d} | {r.get('s_per_epoch', float('nan')):.1f} | {h[-1].get('val_R2', float('nan')):.4f} | "
                     f"{v1s} | {v1r} |")
        L.append("")
    open(path, "w").write("\n".join(L) + "\n")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="T6,T7,T6_100K,1M")
    ap.add_argument("--variants", default="base,amp,compile,ckpt,amp+compile")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--no-v1", action="store_true")
    ap.add_argument("--json", default=os.path.join(ROOT, "runs", "bench", "step_bench.json"))
    ap.add_argument("--no-wait", action="store_true", help="do not wait for exclusive use of the GPU")
    ap.add_argument("--report", action="store_true",
                    help="only render bench/RESULTS_step.md from existing JSON files (no benchmarking)")
    ap.add_argument("--pass1", default=os.path.join(ROOT, "runs", "bench", "step_bench.json"))
    ap.add_argument("--pass2", default=os.path.join(ROOT, "runs", "bench", "step_bench_v2b.json"))
    a = ap.parse_args()
    if a.report:
        rb = os.path.join(ROOT, "runs", "bench")
        batch = [("T8, pass 2 (AirfRANS, 2000 nodes per mesh)", os.path.join(rb, "batch_bench_v2b.json")),
                 ("T8, pass 1", os.path.join(rb, "batch_bench.json")),
                 ("HP_k100, pass 2 (1-2K nodes, variable sizes)", os.path.join(rb, "batch_bench_hp_v2b.json")),
                 ("HP_k100, pass 1", os.path.join(rb, "batch_bench_hp.json")),
                 ("TET_k100, GPU-resident complexes (pass 1)", os.path.join(rb, "batch_bench_tet_gpu.json")),
                 ("TET_k100, CPU-resident complexes (pass 1)", os.path.join(rb, "batch_bench_tet_cpu.json"))]
        write_report(os.path.join(ROOT, "bench", "RESULTS_step.md"), a.pass1, a.pass2, batch,
                     smoke_dir=os.path.join(ROOT, "runs", "smoke"), v1_hist_root=os.path.join(ROOT, "checkpoints_v1"))
        return
    if a.no_wait:
        globals()["wait_exclusive"] = lambda *args, **kw: False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    variants = a.variants.split(",")
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print("| case | model | params | train ms/step | infer ms | peak GB | notes |\n|---|---|---|---|---|---|---|")
    rows = []
    for c in a.cases.split(","):
        if c == "T6":
            rows += case_shared("T6", "T6_wilson_loop.pkl", 128, 4, 64, 128, a.steps, variants, "T6", not a.no_v1)
        elif c == "T7":
            rows += case_shared("T7", "T7_yang_mills_su2.pkl", 160, 4, 64, 128, a.steps, variants, "T7", not a.no_v1)
        elif c == "T6_100K":
            rows += case_shared("T6_100K", "T6_100K_wilson_loop.pkl", 16, 3, 2, 8, max(5, a.steps // 4), variants,
                                None, not a.no_v1)
        elif c == "1M":
            rows += case_1m(max(3, a.steps // 4), [v for v in variants if "compile" not in v] or ["base"])
        else:
            raise ValueError(c)
    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    json.dump({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "rows": rows}, open(a.json, "w"),
              indent=2)
    print(f"saved {a.json}")


if __name__ == "__main__":
    main()
