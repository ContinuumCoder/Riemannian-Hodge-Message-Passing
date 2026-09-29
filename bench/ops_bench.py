"""Sparse operator and complex-construction benchmark (DESIGN §6).

Compares, for ``d0, d0^T, d1, d1^T`` of random 2-D Delaunay meshes with n0 in {1K, 10K, 100K} and
``B*C`` in {128, 8192}, forward and forward+backward:

* ``csr``       : ``ops.spmm(A, x, AT)`` - CSR product, custom autograd with the precomputed transpose (default)
* ``csr_auto``  : ``ops.spmm(A, x)`` - CSR product, torch's own backward (transposes at run time)
* ``csr_i32``   : as ``csr`` with int32 index arrays
* ``coo``       : coalesced COO product with a precomputed COO transpose
* ``gather``    : ``index_select`` for ``d`` (backward: ``index_add``), ``index_add`` for ``d^T``
* ``v1``        : v1's ``batch_spmm``: coalesced COO, ``.t()`` per call for ``d^T``, ``(B, n, C)`` input with
                  permute/reshape copies in and out, torch autograd

and times ``CochainComplex`` construction (100K and 1M faces on GPU, 100K on CPU, v1's T6_100K mesh).
``--whitney`` times the Whitney tensor metric (apply, row sums, tensor vs diagonal up-block, CG resolvent).

    python3 bench/ops_bench.py                # full run, writes bench/RESULTS_ops.md
    python3 bench/ops_bench.py --quick        # small smoke run, no file written
    python3 bench/ops_bench.py --whitney      # only the Whitney tensor-metric section of the results file
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import sys
import time

import numpy as np
import torch
from scipy.spatial import Delaunay

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rhmp.complex import CochainComplex  # noqa: E402
from rhmp.ops import _SpMM, gather_apply, scatter_apply, sparse_csr, spmm  # noqa: E402

METHODS = ("csr", "csr_auto", "csr_i32", "coo", "gather", "v1")


def wait_for_quiet_gpu(max_wait_s: float = 7200.0, poll_s: float = 30.0) -> None:
    """Block until no other compute process uses the GPU (call before CUDA is initialised: holds no context)."""
    import subprocess

    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    q = lambda *a: subprocess.run(["nvidia-smi", *a], capture_output=True, text=True, timeout=60).stdout  # noqa: E731
    try:
        uuid = q("-i", vis, "--query-gpu=uuid", "--format=csv,noheader").strip()
    except Exception:  # no nvidia-smi: nothing to wait for
        return
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        rows = [r.split(",") for r in q("--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader").splitlines()]
        others = [r for r in rows if len(r) == 2 and r[0].strip() == uuid and int(r[1]) != os.getpid()]
        if not others:
            return
        print(f"waiting for a quiet GPU ({len(others)} other compute process(es))", flush=True)
        time.sleep(poll_s)


def delaunay_mesh(n0: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    pts = rng.random((n0, 2))
    return pts, Delaunay(pts).simplices.astype(np.int64)


def cuda_time(fn, warmup: int = 3, reps: int = 10, inner: int = 1) -> float:
    """Median time per call of ``fn`` in ms (CUDA events around ``inner`` back-to-back calls)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(inner):
            fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) / inner)
    return float(np.median(ts))


def gpu_warmup(seconds: float = 3.0) -> None:
    """Bring the GPU to steady clocks before timing."""
    a = torch.randn(4096, 4096, device="cuda")
    t0 = time.time()
    while time.time() - t0 < seconds:
        a = (a @ a).tanh_()
    torch.cuda.synchronize()


def make_appliers(K: CochainComplex, k: int, transpose: bool):
    """Return {method: f(x) -> y} for d_k (or d_k^T)."""
    A, AT = (K.dT[k], K.d[k]) if transpose else (K.d[k], K.dT[k])
    r = 2 if k == 0 else 3
    col = K.d[k].col_indices().view(-1, r)
    val = K.d[k].values().view(-1, r)
    n_out = A.shape[0]

    def i32(M):
        return sparse_csr(M.crow_indices().int(), M.col_indices().int(), M.values(), tuple(M.shape))

    A32, AT32 = i32(A), i32(AT)
    Acoo, ATcoo = A.to_sparse_coo().coalesce(), AT.to_sparse_coo().coalesce()

    def coo(x):
        return _SpMM.apply(Acoo, ATcoo, x.view(x.shape[0], -1)).view((n_out,) + tuple(x.shape[1:]))

    gat = (lambda x: scatter_apply(col, val, x, n_out)) if transpose else (lambda x: gather_apply(col, val, x))
    d_coo = K.d[k].to_sparse_coo().coalesce()

    def v1(x_bnc):  # rhmp.baselines.v1.gauge_hodge_mp.utils.batch_spmm with d.t() per call, as in v1's hodge_mp.py
        M = d_coo.t() if transpose else d_coo
        Bb, n, Cc = x_bnc.shape
        out = torch.sparse.mm(M, x_bnc.permute(1, 0, 2).reshape(n, Bb * Cc))
        return out.reshape(M.shape[0], Bb, Cc).permute(1, 0, 2)

    return {
        "csr": lambda x: spmm(A, x, AT),
        "csr_auto": lambda x: spmm(A, x),
        "csr_i32": lambda x: spmm(A32, x, AT32),
        "coo": coo,
        "gather": gat,
        "v1": v1,
    }


def bench_ops(sizes, widths, reps, rounds: int = 3):
    rows = []
    gpu_warmup()
    for n0 in sizes:
        pts, tri = delaunay_mesh(n0)
        K = CochainComplex.from_triangles(pts, tri, device="cuda")
        for BC in widths:
            B, C = (8, BC // 8)
            inner = max(1, min(20, int(2e7 // (3 * n0 * BC))))
            for k, tr, name in ((0, False, "d0"), (0, True, "d0T"), (1, False, "d1"), (1, True, "d1T")):
                n_in = K.n[k + 1] if tr else K.n[k]
                n_out = K.n[k] if tr else K.n[k + 1]
                f = make_appliers(K, k, tr)
                x = torch.randn(n_in, B, C, device="cuda")
                g = torch.randn(n_out, B, C, device="cuda")

                def inputs(m):
                    """(x, grad-requiring alias of x, upstream grad) in the layout method m expects."""
                    if m == "v1":  # v1 works on contiguous (B, n, C) tensors
                        xb = x.transpose(0, 1).contiguous()
                        return xb, xb.detach().requires_grad_(True), g.transpose(0, 1).contiguous()
                    return x, x.detach().requires_grad_(True), g

                with torch.no_grad():
                    ref = f["csr"](x)
                _, xr, _ = inputs("csr")
                f["csr"](xr).backward(g)
                gref = xr.grad
                del xr
                errs = {}
                for m in METHODS:
                    try:
                        xi, xri, gi = inputs(m)
                        with torch.no_grad():
                            out = f[m](xi)
                            err = float(((out.transpose(0, 1) if m == "v1" else out) - ref).abs().max())
                            del out
                        f[m](xri).backward(gi)
                        gr = xri.grad.transpose(0, 1) if m == "v1" else xri.grad
                        errs[m] = max(err, float((gr - gref).abs().max()))
                        del xi, xri, gi, gr
                    except torch.OutOfMemoryError:  # shared GPU: record the failure and continue
                        errs[m] = float("nan")
                        torch.cuda.empty_cache()
                del ref, gref
                torch.cuda.empty_cache()
                times = {m: ([], []) for m in METHODS}
                for _ in range(rounds):  # interleave methods to average out clock/contention drift
                    for m in METHODS:
                        if errs[m] != errs[m]:  # OOM earlier
                            times[m][0].append(float("nan"))
                            times[m][1].append(float("nan"))
                            continue
                        try:
                            xi, xri, gi = inputs(m)
                            with torch.no_grad():
                                times[m][0].append(cuda_time(lambda: f[m](xi), reps=reps, inner=inner))

                            def fb():
                                xri.grad = None
                                f[m](xri).backward(gi)

                            times[m][1].append(cuda_time(fb, reps=reps, inner=inner))
                            del xi, xri, gi
                        except torch.OutOfMemoryError:
                            times[m][0].append(float("nan"))
                            times[m][1].append(float("nan"))
                        if m == "v1":
                            torch.cuda.empty_cache()
                res = {m: (float(np.nanmedian(times[m][0])) if not all(np.isnan(times[m][0])) else float("nan"),
                           float(np.nanmedian(times[m][1])) if not all(np.isnan(times[m][1])) else float("nan"),
                           errs[m]) for m in METHODS}
                rows.append(dict(op=name, n0=K.n[0], n1=K.n[1], n2=K.n[2], BC=BC, res=res))
                print(f"{name:4s} n0={K.n[0]:7d} BC={BC:5d} " + " ".join(
                    f"{m}={res[m][0]:.3f}/{res[m][1]:.3f}ms(e{res[m][2]:.0e})" for m in METHODS), flush=True)
                del x, g
                torch.cuda.empty_cache()
    return rows


def bench_build():
    out = []

    def timed(fn, device):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        K = fn()
        if device == "cuda":
            torch.cuda.synchronize()
        return K, time.perf_counter() - t0

    for n0, label in ((50_000, "random Delaunay, ~100K faces"), (500_000, "random Delaunay, ~1M faces")):
        pts, tri = delaunay_mesh(n0, seed=1)
        for dev in (("cuda", "cpu") if n0 <= 50_000 else ("cuda",)):
            ts = []
            for rep in range(3):
                K, t = timed(lambda: CochainComplex.from_triangles(pts, tri, device=dev), dev)
                ts.append(t)
            _, tc = timed(K.check_d2, dev)
            out.append(dict(mesh=label, n=K.n, device=dev, first=ts[0], warm=min(ts[1:]), check_d2=tc))
            print(out[-1], flush=True)
    # tetrahedra
    rng = np.random.default_rng(0)
    p3 = rng.random((20_000, 3))
    tets = Delaunay(p3).simplices.astype(np.int64)
    for dev in ("cuda", "cpu"):
        ts = []
        for rep in range(2):
            K, t = timed(lambda: CochainComplex.from_tetrahedra(p3, tets, device=dev), dev)
            ts.append(t)
        _, tc = timed(K.check_d2, dev)
        out.append(dict(mesh="random 3-D Delaunay tets", n=K.n, device=dev, first=ts[0], warm=ts[-1], check_d2=tc))
        print(out[-1], flush=True)
    path = os.path.join(ROOT, "datasets", "T6_100K_wilson_loop.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        pts, faces = np.asarray(d["points"], dtype=np.float64), np.asarray(d["faces"], dtype=np.int64)
        del d
        for dev in ("cuda", "cpu"):
            ts = []
            for rep in range(3):
                K, t = timed(lambda: CochainComplex.from_triangles(pts, faces, device=dev), dev)
                ts.append(t)
            _, tc = timed(K.check_d2, dev)
            out.append(dict(mesh="T6_100K mesh (v1: 6.5 s + 20 GB dense check)", n=K.n, device=dev, first=ts[0],
                            warm=min(ts[1:]), check_d2=tc))
            print(out[-1], flush=True)
    return out


WHITNEY_BEGIN, WHITNEY_END = "<!-- whitney:begin -->", "<!-- whitney:end -->"


def bench_whitney(reps: int = 20, rounds: int = 3) -> list[str]:
    """Timings of the Whitney (Galerkin) tensor metric (DESIGN §9.1) and of a CG resolvent; returns markdown lines."""
    from rhmp import dec

    gpu_warmup(1.0)
    cases = []
    for label, path, B, C in (("T6 mesh", "T6_wilson_loop.pkl", 64, 128), ("T6_100K mesh", "T6_100K_wilson_loop.pkl", 2, 16)):
        fp = os.path.join(ROOT, "datasets", path)
        if os.path.exists(fp):
            with open(fp, "rb") as f:
                d = pickle.load(f)
            K = CochainComplex.from_triangles(np.asarray(d["points"], np.float64), np.asarray(d["faces"], np.int64),
                                              device="cuda")
            del d
            cases.append((label, K, 1, B, C))
    rng = np.random.default_rng(0)
    p3 = rng.random((20_000, 3))
    Kt = CochainComplex.from_tetrahedra(p3, Delaunay(p3).simplices.astype(np.int64), device="cuda")
    cases += [("random tets (k=1, 6x6)", Kt, 1, 2, 16), ("random tets (k=2, 4x4)", Kt, 2, 2, 16)]
    L = [WHITNEY_BEGIN, "", "## Whitney tensor metric (DESIGN §9.1)", "",
         "`dec.apply_whitney_metric(K, k, b, a, x)` (gather, per-(cell, sample) m_k x m_k block product in a fixed "
         "order, scatter; bitwise per-sample) and `dec.whitney_rowsum_abs`; `up-block` = `d_{k-1}^T H_k d_{k-1} x` "
         "on (k-1)-cochains with the tensor metric vs the diagonal (lumped) metric; `resolvent` = "
         "`cg_solve((I + L_up) y = x, iters=16, early_exit=False)` with the tensor-metric up-block. CUDA-event medians "
         "(ms) over 3 rounds x 20 calls; f+b = forward + backward w.r.t. x, b, a.", "",
         "| case | n_k | n_top | B | C | apply fwd | apply f+b | rowsum fwd | up-block tensor f+b | up-block diag f+b "
         "| resolvent (16 it) fwd |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, K, k, B, C in cases:
        n_top = K.n[K.dim]
        m = K.whitney[k]["t"].shape[1]
        b = (torch.rand(n_top, B, device="cuda") + 0.5).requires_grad_(True)
        a = torch.rand(n_top, B, m, device="cuda").requires_grad_(True)
        x = torch.randn(K.n[k], B, C, device="cuda", requires_grad=True)
        x0 = torch.randn(K.n[k - 1], B, C, device="cuda", requires_grad=True)
        hd = K.star[k][:, None].expand(-1, B).contiguous()

        def up_tensor(v):
            return K.apply_dT(k - 1, dec.apply_whitney_metric(K, k, b, a, K.apply_d(k - 1, v)))

        def up_diag(v):
            return K.apply_dT(k - 1, (hd[..., None] * K.apply_d(k - 1, v)).contiguous())

        def fb(fn, *leaves):
            def run():
                for t in leaves:
                    t.grad = None
                fn().square().sum().backward()
            return run

        T = {key: [] for key in ("af", "afb", "rs", "ut", "ud", "cg")}
        for _ in range(rounds):
            with torch.no_grad():
                T["af"].append(cuda_time(lambda: dec.apply_whitney_metric(K, k, b, a, x), reps=reps))
                T["rs"].append(cuda_time(lambda: dec.whitney_rowsum_abs(K, k, b, a), reps=reps))
                sc = 1.0 / float(dec.whitney_rowsum_abs(K, k, b, a).max())  # crude normalisation for the solve
                T["cg"].append(cuda_time(lambda: dec.cg_solve(lambda v: v + sc * up_tensor(v), x0.detach(), iters=16,
                                                              early_exit=False), reps=max(3, reps // 4)))
            T["afb"].append(cuda_time(fb(lambda: dec.apply_whitney_metric(K, k, b, a, x), x, b, a), reps=reps))
            T["ut"].append(cuda_time(fb(lambda: up_tensor(x0), x0, b, a), reps=reps))
            T["ud"].append(cuda_time(fb(lambda: up_diag(x0), x0), reps=reps))
        med = {key: float(np.median(v)) for key, v in T.items()}
        L.append(f"| {label} | {K.n[k]} | {n_top} | {B} | {C} | {med['af']:.3f} | {med['afb']:.3f} | {med['rs']:.3f} "
                 f"| {med['ut']:.3f} | {med['ud']:.3f} | {med['cg']:.3f} |")
        print(L[-1], flush=True)
    L += ["", WHITNEY_END, ""]
    return L


def _replace_whitney_section(path: str, lines: list[str] | None) -> None:
    """Insert/replace the marked Whitney section of the results file (keeps it when the main table is rewritten)."""
    text = open(path).read() if os.path.exists(path) else ""
    if WHITNEY_BEGIN in text:
        head, rest = text.split(WHITNEY_BEGIN, 1)
        old = WHITNEY_BEGIN + rest.split(WHITNEY_END, 1)[0] + WHITNEY_END
        tail = rest.split(WHITNEY_END, 1)[1] if WHITNEY_END in rest else ""
        text = head.rstrip("\n") + "\n\n" + ("\n".join(lines).strip("\n") if lines else old) + "\n" + tail.lstrip("\n")
    elif lines:
        text = text.rstrip("\n") + "\n\n" + "\n".join(lines)
    with open(path, "w") as f:
        f.write(text)


def summarize(rows) -> list[str]:
    """Automatic conclusions: dispatch rule (> 1.3x over csr), csr vs v1 and vs torch's CSR backward."""
    wins, v1f, v1b, autob = [], [], [], []
    for r in rows:
        for p, name in ((0, "fwd"), (1, "f+b")):
            t = {m: r["res"][m][p] for m in METHODS if r["res"][m][p] == r["res"][m][p]}  # drop NaN (OOM)
            best = min((m for m in t if m != "v1"), key=t.get)
            if t["csr"] / t[best] > 1.3:
                wins.append(f"{best} {t['csr'] / t[best]:.2f}x on {r['op']} n0={r['n0']} B*C={r['BC']} {name}")
            if "v1" in t:
                (v1f if p == 0 else v1b).append(t["v1"] / t["csr"])
            if p == 1 and "csr_auto" in t:
                autob.append(t["csr_auto"] / t["csr"])
    big = [r for r in rows if r["n0"] >= 10_000 or r["BC"] >= 8192]
    alt = [np.nanmin([r["res"][m][p] for m in ("coo", "gather")]) / r["res"]["csr"][p] for r in big for p in (0, 1)
           if not all(np.isnan([r["res"][m][p] for m in ("coo", "gather")]))]
    out = [
        "* Dispatch rule (an alternative beating `csr` by > 1.3x in some regime): "
        + ("**none**; `spmm` / `apply_d` keep the CSR path, no dispatch." if not wins else "; ".join(wins)),
        f"* Best non-CSR method (COO or gather/scatter) on problems with n0 >= 10K or B*C = 8192 is "
        f"{min(alt):.1f}x-{max(alt):.1f}x slower than `csr` (extra memory passes; no fusion in eager mode).",
        f"* `csr` vs torch's built-in CSR backward (`csr_auto`, f+b): {min(autob):.2f}x-{max(autob):.2f}x faster; "
        "the gain is launch/setup overhead, largest on small problems. Always pass `AT` (or use `K.apply_d`).",
        f"* `csr` vs v1's `batch_spmm`: forward {min(v1f):.1f}x-{max(v1f):.1f}x, forward+backward "
        f"{min(v1b):.1f}x-{max(v1b):.1f}x faster.",
        "* int32 CSR indices bring no measurable gain; indices stay int64.",
        "* Caveat: at n0 = 100K, B*C = 8192 the working set of the COO / gather / v1 paths (several 6-10 GB "
        "temporaries) approaches the 96 GB device memory, so a few of their entries there are erratic (allocator "
        "pressure, e.g. `v1` f+b below its fwd on d1T); the `csr` numbers are stable across runs.",
    ]
    return out


def write_results(rows, builds, path):
    gpu = torch.cuda.get_device_name(0)
    L = [
        "# Ops benchmark",
        "",
        f"Written by `bench/ops_bench.py` on an {gpu} GPU (torch {torch.__version__}, "
        f"python {platform.python_version()}, {time.strftime('%Y-%m-%d')}). Times are CUDA-event medians in ms; "
        "`fwd` = forward only (no grad), `f+b` = forward + backward w.r.t. the dense input. Meshes: random 2-D "
        "Delaunay of the unit square. Layout `(n, B, C)` with `B = 8`; the sparse product sees the zero-copy "
        "`(n, B*C)` view (`v1` gets the same data as `(B, n, C)`). `err` = max abs deviation (output and "
        "gradient) from the `csr` reference. Each entry is the median over 3 interleaved rounds of 10 timings; "
        "small problems are timed over back-to-back calls (per-call time incl. launch overhead).",
        "",
        "Methods: `csr` = `ops.spmm(A, x, AT)` (CSR, custom autograd with the precomputed transpose; the default "
        "path of `K.apply_d/apply_dT`); `csr_auto` = CSR with torch's built-in backward (no `AT`); `csr_i32` = CSR "
        "with int32 indices; `coo` = coalesced COO + precomputed COO transpose; `gather` = `index_select` for `d` "
        "(backward via `index_add`), `index_add` for `d^T` (backward via `index_select`); `v1` = v1's "
        "`batch_spmm` (coalesced COO, `d.t()` per call, `(B,n,C)` permute copies in and out, torch autograd).",
        "",
        "## Operator application",
        "",
        "| op | n0 | B*C | pass | " + " | ".join(METHODS) + " | best | best vs csr | v1 / csr | max err |",
        "|---|---:|---:|---|" + "---:|" * len(METHODS) + "---|---:|---:|---:|",
    ]
    fmt = lambda v, spec: "OOM" if v != v else format(v, spec)  # noqa: E731
    for r in rows:
        for p, name in ((0, "fwd"), (1, "f+b")):
            t = {m: r["res"][m][p] for m in METHODS}
            ok = {m: v for m, v in t.items() if v == v}
            best = min(ok, key=ok.get)
            err = np.nanmax([r["res"][m][2] for m in METHODS])
            L.append(f"| {r['op']} | {r['n0']} | {r['BC']} | {name} | " + " | ".join(fmt(t[m], ".3f") for m in METHODS)
                     + f" | {best} | {t['csr'] / t[best]:.2f}x | {fmt(t['v1'] / t['csr'], '.1f')}x | {err:.1e} |")
    L += ["", "## Summary", ""] + summarize(rows)
    L += ["", "## Complex construction", "",
          "| mesh | n (cells per degree) | device | first call (s) | warm (s) | check_d2 (s) |",
          "|---|---|---|---:|---:|---:|"]
    for b in builds:
        L.append(f"| {b['mesh']} | {b['n']} | {b['device']} | {b['first']:.3f} | {b['warm']:.3f} | {b['check_d2']:.4f} |")
    L.append("")
    old = open(path).read() if os.path.exists(path) else ""
    keep = None
    if WHITNEY_BEGIN in old and WHITNEY_END in old:
        keep = [WHITNEY_BEGIN + old.split(WHITNEY_BEGIN, 1)[1].split(WHITNEY_END, 1)[0] + WHITNEY_END]
    with open(path, "w") as f:
        f.write("\n".join(L))
    if keep:
        _replace_whitney_section(path, keep)
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "bench", "RESULTS_ops.md"))
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--from-json", default=None, help="rewrite the markdown from a saved runs/ops_bench.json")
    ap.add_argument("--whitney", action="store_true", help="only (re)run the Whitney tensor-metric section")
    ap.add_argument("--no-wait", action="store_true", help="do not wait for other GPU processes to finish")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    if not (args.no_wait or args.from_json):
        wait_for_quiet_gpu()
    if args.whitney:
        _replace_whitney_section(args.out, bench_whitney())
        print("updated Whitney section of", args.out)
    elif args.from_json:
        with open(args.from_json) as f:
            saved = json.load(f)
        write_results(saved["rows"], saved["builds"], args.out)
    elif args.quick:
        rows = bench_ops([1000], [128], reps=3)
    else:
        rows = bench_ops([1_000, 10_000, 100_000], [128, 8192], reps=10)
        builds = [] if args.no_build else bench_build()
        os.makedirs(os.path.join(ROOT, "runs"), exist_ok=True)
        with open(os.path.join(ROOT, "runs", "ops_bench.json"), "w") as f:  # raw numbers (runs/ is not versioned)
            json.dump(dict(rows=rows, builds=builds), f)
        write_results(rows, builds, args.out)
