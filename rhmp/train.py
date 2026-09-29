"""Generic trainer / evaluator for RHMP v2 (v1 protocol by default).

    python3 -u -m rhmp.train --task T6 --native --epochs 100 --seed 42 --out runs/t6_native
    python3 -u -m rhmp.train --task T8 --epochs 2 --out runs/smoke_t8                  # block-diagonal batches of 8
    python3 -u -m rhmp.train --task T6 --legacy --eval-v1 checkpoints_v1/T6_wilson_loop/ours/best_model.pt \
                             --out runs/v1_t6                                             # v1 model, same split/metrics
    python3 -u -m rhmp.train --task HP_k100 --check-data                                  # load + print the data summary

Protocol (the v1 training script ``formal_benchmark.py``): Adam(lr 1e-3, weight decay 1e-5), cosine annealing per
epoch to 1e-5 over ``--epochs``, grad-norm clipping 1.0, batch 64 (shared meshes) / 8 meshes (variable meshes), MSE
on normalised targets, model selection by validation R2 (v1 formula), test metrics of the best model: R2/MSE/MAE
(normalised), NRMSE/SSIM/Pearson (unnormalised), on the full test split, on the first 100 test samples (the v1
paper tables, ``test100_*``) and on extra test sets (``fine_*``: zero-shot 4x resolution for HP/TET).

Data are GPU resident (no DataLoader).  Shared-mesh minibatches are gathered with ``index_select`` into the
``(n, B, C)`` layout; variable meshes are batched block-diagonally (``CochainComplex.batch``), with a background
thread preparing the next batch (complexes that do not fit the GPU budget stay on the CPU and are moved per batch).

Outputs in ``--out``: ``config.json``, ``history.json`` (per epoch: losses, val metrics, lr, train/eval seconds,
peak GPU memory, metric condition numbers), ``best.pt`` / ``last.pt`` (resume: re-run the same command),
``result.json`` (test metrics, wall-clock, parameters, diagnostics).  Seeding is deterministic (python, numpy,
torch, batch order from a CPU generator); CUDA atomics make runs close but not bitwise identical.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import random
import threading
import time
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from rhmp import metrics as M
from rhmp.data import TaskData, iterate_indices, mesh_minibatch, same_device, shared_minibatch

__all__ = ["parse_args", "run", "build_config", "model_output", "batch_loss", "pde_residual", "predict", "evaluate",
           "main"]


# ================================================================================================================
# CLI
# ================================================================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line interface (see module docstring)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, help="T1 T1q T2 T3 T5 T6 T6f T7 T7f T8 T6_100K HP* TET* (see rhmp.tasks)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--native", dest="native", action="store_true", default=True, help="native cochain I/O (default)")
    g.add_argument("--legacy", dest="native", action="store_false", help="v1 inputs/outputs")
    p.add_argument("--root", default=None, help="repository root or datasets dir (default: this repository)")
    p.add_argument("--out", default=None, help="output directory (default runs/<task>_<mode>_s<seed>)")
    p.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    # model
    p.add_argument("--model", default="rhmp",
                   help="model from rhmp.baselines.registry: rhmp (v2, default), dec_fixed, unit_star, unit_fixed, "
                        "ours_v1, mgn, mgn_fast, gcn, gat, schnet, egnn, gauge_cnn, gem_cnn, mpsn, sccnn, cw_net, "
                        "clifford_smpn, fno, deeponet (docs/BASELINES.md)")
    p.add_argument("--param-budget", type=float, default=None,
                   help="parameter budget (millions) of param-matched baselines (default: the v2 model's count for "
                        "the same task and --C/--layers)")
    p.add_argument("--model-opts", default=None,
                   help="JSON dict of builder options for --model, e.g. '{\"n_layers\": 8}' or '{\"hidden\": 64}'")
    p.add_argument("--C", type=int, default=None, help="hidden channels (default: v1 C of the task)")
    p.add_argument("--layers", type=str, default=None,
                   help="depth (e.g. 4; default: task default) or per-layer types, e.g. poly,poly,resolvent,poly")
    p.add_argument("--metric-type", default="diag", choices=["diag", "tensor"],
                   help="diagonal metrics (default) or the Whitney/Galerkin material-tensor metric (DESIGN 9.1)")
    p.add_argument("--resolvent-iters", type=int, default=None, help="CG iterations of resolvent layers")
    p.add_argument("--resolvent-grad", default=None, choices=["implicit", "unrolled"])
    p.add_argument("--latent", default=None, metavar="K:R[,K:R]",
                   help="r learnable per-cell input columns on degree k (RHMPConfig.latent_dims), e.g. 1:8; tied to one "
                        "mesh: shared-mesh tasks only")
    p.add_argument("--metric-ref", default=None, metavar="K:COL[,K:COL]",
                   help="metric reference input: the (even) input column COL of degree K is a fixed log offset of the "
                        "metric of degree K (H = star exp(ref) exp(a tanh(.))), e.g. 1:0 = log sigma_e on HP/TET edges")
    p.add_argument("--material", default=None, metavar="K:M[,K:M]",
                   help="the last M even input columns of degree K are material columns: they reach only the metric "
                        "heads and --metric-ref, never the lifting / gates / features (RHMPConfig.material_dims), e.g. "
                        "1:1,2:1 on HP/TET")
    p.add_argument("--solver-mode", action="store_true",
                   help="RHMPConfig.solver_preset: linear lifting, one 'solve' layer (y = (Delta_H + lam)^-1 x on the "
                        "free cells, physical units), no gates, no cross terms, linear readout (node_scalar -> "
                        "cochain:0), C = --C or 4: the metric is the only nonlinearity (--layers may list other types)")
    p.add_argument("--solve-iters", type=int, default=None, help="CG iterations of solve layers (default 64)")
    p.add_argument("--solve-precond", default=None, choices=["none", "twolevel"],
                   help="CG preconditioner of solve layers (default none = Jacobi; twolevel = Jacobi + coarse "
                        "correction on spatial aggregates, vertex solves)")
    p.add_argument("--solve-bc", default=None, choices=["dirichlet", "none", "neumann"],
                   help="solve layers: fixed boundary cells (K.boundary / K.meta['dirichlet']), none, or neumann (no "
                        "fixed cells, zero-mean vertex solution of the pure Neumann problem, e.g. T5/T5g)")
    p.add_argument("--tensor-param", default=None, choices=["full", "cone"],
                   help="tensor metrics: full SPD tensors b expm(sum s t t^T) (default) or the edge cone "
                        "b I + sum a t t^T with a >= 0 (the parameterisation of runs saved without this option)")
    p.add_argument("--aux-pde", type=float, default=0.0, metavar="W",
                   help="add W * model.operator_residual(u, f) (relative weak-form residual of the task's PDE with the "
                        "learned metric, true u and f) to the loss; tasks with TaskData.meta['pde'] only (HP/TET "
                        "potentials), ignored otherwise")
    p.add_argument("--no-learn-metric", action="store_true",
                   help="freeze the metric at the reference (H = star exp(ref); RHMPConfig.learn_metric=False): the pure "
                        "DEC/FEEC physics prior without metric heads")
    p.add_argument("--poly", type=int, default=2)
    p.add_argument("--log-range", type=float, default=2.0)
    p.add_argument("--metric-hidden", type=int, default=32)
    p.add_argument("--scaling", default="dec", choices=["dec", "jacobi", "none"])
    g = p.add_mutually_exclusive_group()
    g.add_argument("--tie", dest="tie", action="store_true", default=True)
    g.add_argument("--untie", dest="tie", action="store_false")
    p.add_argument("--no-cross", action="store_true")
    p.add_argument("--gate", default="norm", choices=["norm", "relu", "none"])
    p.add_argument("--identity-metric", action="store_true")
    p.add_argument("--star", default="cotan", choices=["cotan", "barycentric", "unit"])
    p.add_argument("--vector-mode", default=None, choices=["ls", "direct"], help="default: task specific")
    p.add_argument("--connection-odd", default=None, choices=["on", "off"],
                   help="connection columns also enter the odd path (default: on for SU(2) T7/T7f, off otherwise)")
    p.add_argument("--no-connection", action="store_true", help="treat native connection inputs as plain odd inputs")
    p.add_argument("--no-output-map", action="store_true",
                   help="disable the task's orientation output map (native T1/T3/T6/T7): the model then predicts the "
                        "pseudo-scalar/vector target directly (not representable by an equivariant model)")
    # optimisation
    p.add_argument("--batch", type=int, default=None, help="samples per step on shared meshes (default 64)")
    p.add_argument("--bs-meshes", type=int, default=None, help="meshes per block-diagonal batch (default 8)")
    p.add_argument("--eval-batch", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--eta-min", type=float, default=1e-5)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", help="bf16 autocast for the dense parts (sparse products fp32)")
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    p.add_argument("--compile-backend", default="inductor",
                   help="torch.compile backend (inductor needs the Python headers to build Triton's CUDA helpers; "
                        "'aot_eager' works everywhere but gives no speed-up)")
    p.add_argument("--ckpt-layers", action="store_true", help="activation checkpointing per layer")
    p.add_argument("--no-tf32", action="store_true", help="disable TF32 dense matmuls on CUDA")
    # run control
    p.add_argument("--max-samples", type=int, default=None, help="truncate the data set (smoke tests)")
    p.add_argument("--max-train-batches", type=int, default=None, help="limit steps per epoch (smoke tests)")
    p.add_argument("--no-fine", action="store_true", help="skip the resolution-transfer test set (HP/TET)")
    p.add_argument("--abs-scale", action="store_true",
                   help="append log(median edge length) as a constant even vertex input (absolute mesh scale; the "
                        "model is scale-free by design)")
    p.add_argument("--data-on", default="auto", choices=["auto", "device", "cpu"],
                   help="where variable-mesh complexes live (auto: GPU if the estimate is below "
                        "rhmp.tasks.synthetic.GPU_BUDGET_GB, env RHMP_GPU_BUDGET_GB)")
    p.add_argument("--no-resume", action="store_true", help="ignore an existing last.pt")
    p.add_argument("--force", action="store_true", help="re-run even if result.json exists")
    p.add_argument("--check-data", action="store_true", help="load the task, print its summary and exit")
    p.add_argument("--eval-v1", default=None, metavar="CKPT", help="evaluate a v1 checkpoint (legacy data)")
    p.add_argument("--eval-ckpt", default=None, metavar="RUN_DIR",
                   help="evaluate the best.pt of a finished v2 run on --task (transfer/robustness sweeps); inputs and "
                        "targets are re-normalised with the statistics of the training run")
    p.add_argument("--v1-eval-batch", type=int, default=1,
                   help="batch size for the v1 model (1 = per sample as compute_all_metrics.py; v1 couples batches)")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--stop-after", type=int, default=None, help=argparse.SUPPRESS)  # tests: simulate a crash
    return p.parse_args(argv)


# ================================================================================================================
# helpers
# ================================================================================================================
def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _peak_gb(device) -> float:
    if torch.device(device).type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2 ** 30
    return 0.0


def _reset_peak(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, Tensor):
        return o.tolist() if o.numel() <= 64 else f"<tensor {tuple(o.shape)}>"
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    if isinstance(o, (int, float, str, bool)) or o is None:
        return o
    return str(o)


def _write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_jsonable(obj), f, indent=2)
    os.replace(tmp, path)


def _log(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(msg, flush=True)


# ================================================================================================================
# model configuration
# ================================================================================================================
def build_config(task: TaskData, args: argparse.Namespace):
    """RHMPConfig for ``task`` from the CLI arguments and the task defaults."""
    from rhmp.model import RHMPConfig
    readout = task.readout                                  # canonical (TaskData.readout)
    out_dim = task.out_dim
    if task.output_map is not None:
        out_dim = task.output_map.model_out_dim
    elif readout == "node_vector":
        if task.out_dim % task.spatial_dim:
            raise ValueError(f"node_vector target width {task.out_dim} not a multiple of D={task.spatial_dim}")
        out_dim = task.out_dim // task.spatial_dim           # number of vector fields
    conn = {} if args.no_connection else dict(task.connection_dims)
    fields = set(getattr(RHMPConfig, "__dataclass_fields__", {}))
    extra: dict = {}
    n_layers = task.meta.get("layers", 4)
    if args.layers:
        if str(args.layers).isdigit():
            n_layers = int(args.layers)
        else:
            types = [t.strip() for t in str(args.layers).split(",") if t.strip()]
            if "layers" not in fields:
                raise ValueError("this RHMP version has no per-layer types (cfg.layers)")
            extra["layers"], n_layers = types, len(types)
    if getattr(args, "latent", None):
        if "latent_dims" not in fields:
            raise ValueError("this RHMP version has no latent_dims (--latent)")
        if task.variable_mesh:
            raise ValueError("--latent: learnable per-cell latent fields are tied to one mesh; task "
                             f"{task.name!r} has one mesh per sample (variable-mesh tasks are not supported)")
        lat = {}
        for item in str(args.latent).split(","):
            k, r = (int(v) for v in item.split(":"))
            if k not in task.geo_dims or r < 1:
                raise ValueError(f"--latent {item}: degree must be one of {sorted(task.geo_dims)} and R >= 1")
            lat[k] = r
        extra["latent_dims"] = lat
    if getattr(args, "material", None):
        if "material_dims" not in fields:
            raise ValueError("this RHMP version has no material_dims (--material)")
        mat = {}
        for item in str(args.material).split(","):
            k, M_ = (int(v) for v in item.split(":"))
            if not 1 <= M_ <= task.even_dims.get(k, 0):
                raise ValueError(f"--material {item}: M must be in 1..{task.even_dims.get(k, 0)} (the even input "
                                 f"columns of degree {k})")
            mat[k] = M_
        extra["material_dims"] = mat
    if getattr(args, "no_learn_metric", False):
        if "learn_metric" not in fields:
            raise ValueError("this RHMP version has no learn_metric (--no-learn-metric)")
        extra["learn_metric"] = False
    if getattr(args, "metric_ref", None):
        if "metric_reference" not in fields:
            raise ValueError("this RHMP version has no metric_reference (--metric-ref)")
        ref = {}
        for item in str(args.metric_ref).split(","):
            k, col = item.split(":")
            k, col = int(k), int(col)
            F, E = task.in_dims.get(k, 0), task.even_dims.get(k, 0)
            if not (F - E <= col < F):
                raise ValueError(f"--metric-ref {k}:{col}: column must be one of the even columns "
                                 f"{list(range(F - E, F))} of degree {k}")
            ref[k] = col
        extra["metric_reference"] = ref
    tparam = getattr(args, "tensor_param", None) or getattr(args, "_resume_tensor_param", None)
    for flag, key, val in (("metric_type", "metric_type", None), ("resolvent_iters", "resolvent_iters", None),
                           ("resolvent_grad", "resolvent_grad", None), ("solve_iters", "solve_iters", None),
                           ("solve_bc", "solve_bc", None), ("tensor_param", "tensor_param", tparam),
                           ("solve_precond", "solve_precond", None)):
        val = getattr(args, flag, None) if val is None else val
        if val is None or (key == "metric_type" and val == "diag" and key not in fields):
            continue
        if key not in fields:
            raise ValueError(f"this RHMP version does not support --{flag.replace('_', '-')}")
        extra[key] = val
    if args.connection_odd is None:
        conn_odd = bool(conn) and task.name in ("T7", "T7f")   # SU(2): connection also through the odd path
    else:
        conn_odd = args.connection_odd == "on"
    kw = dict(
        in_dims=dict(task.in_dims), even_dims=dict(task.even_dims), connection_dims=conn,
        C=args.C or task.meta.get("v1_C", 128), n_layers=n_layers,
        poly_order=args.poly, metric_hidden=args.metric_hidden, log_range=args.log_range, tie_metrics=args.tie,
        scaling=args.scaling, cross=not args.no_cross, gate=args.gate, identity_metric=args.identity_metric,
        readout=readout, out_dim=out_dim, vector_mode=args.vector_mode or task.meta.get("vector_mode", "ls"),
        checkpoint_layers=args.ckpt_layers, amp=args.amp, connection_odd=conn_odd, **extra,
    )
    if getattr(args, "solver_mode", False):
        if not hasattr(RHMPConfig, "solver_preset"):
            raise ValueError("this RHMP version has no solver preset (--solver-mode)")
        ro = "cochain:0" if readout == "node_scalar" else readout
        if ro.split(":")[0] not in ("cochain", "curl", "div", "mdiv", "grad"):
            raise ValueError(f"--solver-mode needs a linear readout (cochain:k, grad, curl, div[:k], mdiv[:k]); the "
                             f"task's readout {readout!r} is nonlinear")
        for key in ("gate", "cross", "n_layers"):                  # the preset's values (layers only if not given)
            kw.pop(key)
        kw.update(readout=ro, C=args.C or 4)
        return RHMPConfig.solver_preset(**kw)
    return RHMPConfig(**kw)


# ================================================================================================================
# batches, loss, prediction
# ================================================================================================================
class _Prefetch:
    """Run a batch builder one step ahead in a background thread (variable-mesh tasks)."""

    def __init__(self, fn, items: Iterable, depth: int = 2):
        self.q: queue.Queue = queue.Queue(maxsize=depth)
        self.err: list[BaseException] = []

        def work():
            try:
                for it in items:
                    self.q.put((it, fn(it)))
            except BaseException as e:  # noqa: BLE001 - re-raised in the consumer
                self.err.append(e)
            finally:
                self.q.put(None)

        self.t = threading.Thread(target=work, daemon=True)
        self.t.start()

    def __iter__(self):
        while True:
            item = self.q.get()
            if item is None:
                if self.err:
                    raise self.err[0]
                return
            yield item


def make_batch(task: TaskData, idx: Tensor, device, cache: dict | None = None):
    """``(K, inputs {k: (n_k, B, F_k)}, target (n_t, B, O))`` for sample indices ``idx`` (both task kinds)."""
    if task.variable_mesh:
        return mesh_minibatch(task.K, task.inputs, task.target, idx, device=device, cache=cache)
    xb, yb = shared_minibatch(task.inputs, task.target, idx)
    return task.K, xb, yb


def model_output(task: TaskData, fwd, xb: dict, K) -> Tensor:
    """Model prediction in the task's target space (applies ``task.output_map`` when present)."""
    y = fwd(xb, K)
    return task.output_map(y) if task.output_map is not None else y


def batch_loss(model, task: TaskData, idx: Tensor, device=None, fwd=None) -> Tensor:
    """MSE of one (block-diagonal) minibatch on normalised targets (pooled over all cells/samples/channels)."""
    device = device or next(model.parameters()).device
    K, xb, yb = make_batch(task, idx, device)
    return F.mse_loss(model_output(task, fwd or model, xb, K), yb)


def pde_residual(model, task: TaskData, xb: dict, yb: Tensor, K) -> Tensor | None:
    """Relative weak-form PDE residual of the true solution with the model's metric, ``(num_graphs|1, B)``.

    Uses ``task.meta['pde'] = {'degree': k, 'source': 'inputs[j][...,c]', 'bc': 'dirichlet'|'none'}``: the target
    (denormalised) is ``u`` on the degree-k cells and the (denormalised) input column ``c`` of degree ``j`` is ``f``;
    returns ``model.operator_residual(xb, K, u, f, k, bc=bc)`` or ``None`` when the task / model has no such data.
    """
    import re
    pde = task.meta.get("pde") if isinstance(task.meta, dict) else None
    if not pde or not hasattr(model, "operator_residual") or task.output_map is not None:
        return None
    k = int(pde.get("degree", 0))
    m = re.fullmatch(r"inputs\[(\d+)\]\[\.\.\.,(\d+)\]", str(pde["source"]).replace(" ", ""))
    if m is None:
        raise ValueError(f"meta['pde']['source'] = {pde['source']!r}: expected 'inputs[<degree>][...,<column>]'")
    if task.target_degree != k or yb.shape[-1] != 1:
        return None
    j, c = int(m.group(1)), int(m.group(2))
    f = task.x_stats[j].denormalize(xb[j])[..., c]
    u = task.y_stats.denormalize(yb)[..., 0]
    return model.operator_residual(xb, K, u, f, k=k, bc=pde.get("bc"))


@torch.no_grad()
def predict(model, task: TaskData, idx: Tensor, batch_size: int, device, fwd=None, cache: dict | None = None,
            source: dict | None = None):
    """Normalised predictions and targets for samples ``idx``.

    Args:
        source: optional extra test set ``{'K', 'inputs', 'target'}`` (lists) instead of the task's own data.
    Returns:
        ``(pred, target)``: ``(N, n_t, O)`` tensors for shared meshes or equal-size variable meshes (stacked, so
        that R2 uses the v1 per-position mean), else lists of ``(n_t_i, O)`` tensors.
    """
    fwd = fwd or model
    preds, tgts = [], []
    if source is not None:
        view = TaskData(name=task.name, K=source["K"], inputs=source["inputs"], target=source["target"],
                        target_degree=task.target_degree, target_kind=task.target_kind, in_dims=task.in_dims,
                        even_dims=task.even_dims, connection_dims=task.connection_dims, split=task.split,
                        x_stats=task.x_stats, y_stats=task.y_stats, spatial_dim=task.spatial_dim,
                        out_dim=task.out_dim, output_map=task.output_map, readout=task.readout)
    else:
        view = task
    batches = list(iterate_indices(idx, batch_size, shuffle=False))
    if view.variable_mesh:
        it = _Prefetch(lambda b: make_batch(view, b, device, cache), batches)
        for b, (K, xb, yb) in it:
            p = model_output(view, fwd, xb, K)
            sizes = [int(view.target[int(i)].shape[0]) for i in b.tolist()]
            preds.extend(p[:, 0].split(sizes, 0))
            tgts.extend(yb[:, 0].split(sizes, 0))
        if len({t.shape for t in tgts}) == 1:
            return torch.stack(preds), torch.stack(tgts)
        return preds, tgts
    for b in batches:
        K, xb, yb = make_batch(view, b, device)
        p = model_output(view, fwd, xb, K)
        preds.append(p.permute(1, 0, 2))
        tgts.append(yb.permute(1, 0, 2))
    return torch.cat(preds), torch.cat(tgts)


def evaluate(model, task: TaskData, idx: Tensor, batch_size: int, device, *, fwd=None, cache=None, source=None,
             n_samples: int | None = None, prefix: str = "") -> dict:
    """``rhmp.metrics.summarize`` of the predictions on ``idx`` (normalised pred/target + ``task.y_stats``)."""
    was = model.training
    model.eval()
    pred, tgt = predict(model, task, idx, batch_size, device, fwd=fwd, cache=cache, source=source)
    model.train(was)
    return M.summarize(pred, tgt, task.y_stats.as_tuple(), r2_std=task.meta.get("r2_std"), n_samples=n_samples,
                       prefix=prefix, center=task.target_kind != "cochain")


def _cond_summary(diag: dict, suffix: str | None = None) -> dict:
    """Max over layers of a per-metric diagnostic keyed by metric name, e.g. ``{'H1.cond_total': 12.3}``.

    ``suffix`` selects the statistic: ``.cond_total`` (max H / min H incl. star and metric reference; the default,
    ``.cond`` in the diagnostics of earlier versions), ``.cond_learned`` (learned part only), ``.clamp_fraction``,
    ``.sat``."""
    if suffix is None:
        suffix = ".cond_total" if any(k.endswith(".cond_total") for k in diag) else ".cond"
    out: dict[str, float] = {}
    for k, v in diag.items():
        if k.endswith(suffix):
            name = k.split(".", 1)[1]
            out[name] = max(out.get(name, 0.0), float(v))
    return out


def _sat_summary(diag: dict) -> dict:
    """Per-layer clamp saturation of the bounded metric corrections: ``{'layer{l}.H{m}': frac}``."""
    return {k[: -len(".sat")]: float(v) for k, v in diag.items() if k.endswith(".sat")}


# ================================================================================================================
# checkpoints
# ================================================================================================================
def _rng_state(gen: torch.Generator) -> dict:
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
          "batch_gen": gen.get_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st: dict, gen: torch.Generator) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    gen.set_state(st["batch_gen"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def _run_signature(args: argparse.Namespace) -> dict:
    keys = ("task", "native", "seed", "C", "layers", "poly", "log_range", "metric_hidden", "scaling", "tie",
            "no_cross", "gate", "identity_metric", "star", "vector_mode", "connection_odd", "no_connection", "batch",
            "bs_meshes", "lr", "wd", "eta_min", "clip", "epochs", "max_samples", "max_train_batches", "no_output_map",
            "metric_type", "resolvent_iters", "resolvent_grad", "metric_ref", "abs_scale", "latent")
    sig = {k: getattr(args, k) for k in keys}
    for k, default in (("material", None), ("solver_mode", False), ("solve_iters", None), ("solve_bc", None),
                       ("tensor_param", None), ("aux_pde", 0.0), ("solve_precond", None)):   # only when set, so that
        v = getattr(args, k, default)                                                      # runs without them resume
        if v != default:
            sig[k] = v
    if getattr(args, "metric_ref", None) or getattr(args, "material", None):
        sig["metric_reference_raw"] = True      # runs with normalised reference columns must not resume
    if getattr(args, "model", "rhmp") != "rhmp":
        sig.update(model=args.model, param_budget=args.param_budget, model_opts=args.model_opts)
    return sig


def _build_baseline(args: argparse.Namespace, task: TaskData, device):
    """Model ``--model`` from ``rhmp.baselines.registry`` for ``task``.

    Param-matched baselines get the v2 model's trainable-parameter count (same task, ``--C``, ``--layers``) unless
    ``--param-budget`` is given.  Non-v2 models predict the task target directly, so the task's orientation output
    map is dropped for them (returned task copy).

    Returns:
        ``(model, info, task)``.
    """
    import dataclasses

    from rhmp.baselines.registry import SPECS, build_model, prepare_task, rhmp_param_count
    if args.model not in SPECS:
        raise ValueError(f"unknown --model {args.model!r}; known: {sorted(SPECS)}")
    spec = SPECS[args.model]
    opts = json.loads(args.model_opts) if args.model_opts else {}
    if not isinstance(opts, dict):
        raise ValueError("--model-opts must be a JSON object")
    budget = int(round(args.param_budget * 1e6)) if args.param_budget else None
    if spec.param_matched and budget is None and "hidden" not in opts:
        budget = rhmp_param_count(task, args)
    if spec.family == "rhmp":
        opts.setdefault("args", args)
    opts.setdefault("amp", args.amp)
    task = prepare_task(args.model, task, opts)       # v1 edge protocol of MPSN/SCCNN/Clifford on the paper tasks
    model, info = build_model(args.model, task, budget, **opts)
    if task.meta.get("v1_edge_protocol"):
        info["target_protocol"] = task.meta["v1_edge_protocol"]
    if not spec.uses_output_map and task.output_map is not None:
        info["dropped_output_map"] = task.output_map.description
        task = dataclasses.replace(task, output_map=None)
    return model.to(device), info, task


# ================================================================================================================
# training
# ================================================================================================================
def run(args: argparse.Namespace, task: TaskData | None = None) -> dict:
    """Train + evaluate (or ``--eval-v1`` / ``--check-data``).  Returns the ``result.json`` content."""
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32
        torch.backends.cudnn.allow_tf32 = not args.no_tf32
    mode = "native" if args.native else "legacy"
    model_name = getattr(args, "model", "rhmp")
    tag = "" if model_name == "rhmp" else f"_{model_name}"
    out = args.out or os.path.join("runs", f"{args.task}_{mode}{tag}_s{args.seed}")
    if model_name != "rhmp" and task is None:
        from rhmp.baselines.registry import SPECS
        need = SPECS[model_name].data_star if model_name in SPECS else None
        if need and args.star != need:
            _log(f"--model {model_name}: loading the task with star={need!r} (was {args.star!r})", args.quiet)
            args.star = need
    os.makedirs(out, exist_ok=True)
    q = args.quiet
    res_path = os.path.join(out, "result.json")
    if args.eval_v1 is None and args.eval_ckpt is None and not args.check_data and os.path.exists(res_path) \
            and not args.force:
        prev = json.load(open(res_path))
        if prev.get("complete"):
            # always printed (also with --quiet): the caller gets the previous result, nothing is trained
            print(f"[{args.task}] skipping: {out}/result.json is complete (test R2 "
                  f"{prev.get('test', {}).get('R2')}, {prev.get('epochs_run')} epochs); use --force to re-run",
                  flush=True)
            return prev

    seed_everything(args.seed)
    t_load = time.time()
    if task is None:
        from rhmp.tasks import load_task
        kw = {}
        if args.max_samples is not None:
            kw["max_samples"] = args.max_samples
        if args.no_fine:
            kw["fine"] = False
        if args.no_output_map:
            kw["output_map"] = False
        if args.data_on != "auto":
            kw["keep_on"] = str(device) if args.data_on == "device" else "cpu"
        need_whitney = getattr(args, "model", "rhmp") == "rhmp" and getattr(args, "metric_type", "diag") == "tensor"
        if args.eval_ckpt is not None:          # the evaluated run decides (its saved model config)
            try:
                rconf = json.load(open(os.path.join(args.eval_ckpt, "config.json")))
                need_whitney = rconf.get("model", {}).get("metric_type") == "tensor"
            except (OSError, ValueError):
                need_whitney = True
        if not need_whitney:
            kw["whitney"] = False               # Whitney blocks: tensor metrics only (41 % of a complex's memory)
        abs_scale = getattr(args, "abs_scale", False)
        if args.eval_ckpt is not None:          # evaluation inputs must match the run's
            try:
                abs_scale = json.load(open(os.path.join(args.eval_ckpt, "config.json")))["args"].get("abs_scale")
            except (OSError, ValueError, KeyError):
                pass
        if abs_scale:
            kw["abs_scale"] = True
        task = load_task(args.task, args.root, native=args.native, device=device, star=args.star, **kw)
    t_load = time.time() - t_load
    if args.check_data:
        s = task.summary()
        s["load_seconds"] = t_load
        _log(json.dumps(_jsonable(s), indent=2), q)
        _write_json(os.path.join(out, "data_summary.json"), s)
        return s
    if args.eval_v1 is not None:
        return eval_v1(args, task, device, out)
    if args.eval_ckpt is not None:
        return eval_ckpt(args, task, device, out)

    from rhmp.model import RHMP, RHMPConfig
    last0 = os.path.join(out, "last.pt")
    if model_name == "rhmp" and getattr(args, "metric_type", "diag") == "tensor" and \
            getattr(args, "tensor_param", None) is None and os.path.exists(last0) and not args.no_resume:
        try:                                    # a saved config without tensor_param resumes as 'cone'
            args._resume_tensor_param = RHMPConfig.from_dict(
                torch.load(last0, map_location="cpu", weights_only=False)["model"]["cfg"]).tensor_param
        except Exception:  # noqa: BLE001 - unreadable last.pt: the resume below reports it
            pass
    cfg = build_config(task, args)
    # material / metric-reference columns arrive raw (physical log values) in every mode
    if getattr(args, "solver_mode", False) and model_name == "rhmp":
        raw_cols = _solver_normalization(task, cfg)         # + mean-free source / target (exact linear class)
    else:
        raw_cols = _raw_reference_normalization(task, cfg)
    seed_everything(args.seed)
    model_info = None
    if model_name == "rhmp":
        model = RHMP(cfg, task.geo_dims).to(device)
        if getattr(cfg, "latent_dims", None):
            model.init_latents(task.K)          # latent fields must exist before the optimizer is built
    else:
        model, model_info, task = _build_baseline(args, task, device)
    n_params = model.num_parameters()
    if model_name == "rhmp" and getattr(args, "solver_mode", False) and \
            not (os.path.exists(os.path.join(out, "last.pt")) and not args.no_resume):
        # the solver class is exact up to the gain of lifting x readout: least-squares readout from the first training
        # batch instead of learning a gain of O(100) from a random start
        bs0 = (args.bs_meshes or task.meta.get("batch_size", 8)) if task.variable_mesh else \
            (args.batch or task.meta.get("batch_size", 64))
        idx0 = task.split[0][:bs0]
        K0, xb0, yb0 = make_batch(task, idx0 if task.variable_mesh else idx0.to(device), device)
        rel0 = model.fit_linear_readout_(xb0, K0, yb0, output_map=task.output_map)
        _log(f"  solver mode: least-squares readout from the first training batch (relative residual {rel0:.2e})", args.quiet)
    fwd = torch.compile(model, dynamic=task.variable_mesh, backend=args.compile_backend) if args.compile else model
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.eta_min)
    gen = torch.Generator().manual_seed(args.seed)
    bs = (args.bs_meshes or task.meta.get("batch_size", 8)) if task.variable_mesh else \
        (args.batch or task.meta.get("batch_size", 64))
    ebs = args.eval_batch or bs
    tr_idx, va_idx, te_idx = task.split
    sig = _run_signature(args)
    model_desc = cfg.to_dict() if model_info is None else {"name": model_name, **model_info,
                                                            "rhmp_reference_config": cfg.to_dict()}
    _write_json(os.path.join(out, "config.json"), {"args": vars(args), "model": model_desc, "params": n_params,
                                                   "data": task.summary(), "load_seconds": t_load,
                                                   "metric_reference_raw": True, "raw_input_columns": raw_cols})
    arch = (f"C={cfg.C} L={cfg.n_layers} readout={cfg.readout}" if model_info is None else
            f"model={model_name} hidden={model_info.get('hidden', model_info.get('C'))} "
            f"layers={model_info.get('n_layers')} budget={model_info.get('target_params')}")
    _log(f"[{args.task}/{mode}] N={task.num_samples} split={[len(s) for s in task.split]} params={n_params/1e6:.3f}M "
         f"{arch} batch={bs} device={device} load={t_load:.1f}s", q)

    history: list[dict] = []
    best_r2, best_ep, start_ep = -float("inf"), 0, 1
    last_path, best_path = os.path.join(out, "last.pt"), os.path.join(out, "best.pt")
    if os.path.exists(last_path) and not args.no_resume:
        ck = torch.load(last_path, map_location=device, weights_only=False)
        if ck.get("signature") != sig:
            raise RuntimeError(f"{last_path} was written by a different configuration; use --no-resume or a new --out")
        model.load_state_dict(ck["model"]["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        _set_rng_state(ck["rng"], gen)
        history, best_r2, best_ep, start_ep = ck["history"], ck["best_r2"], ck["best_epoch"], ck["epoch"] + 1
        _log(f"  resumed from epoch {ck['epoch']} (best val R2 {best_r2:.4f} @ {best_ep})", q)

    # the validation batches of variable meshes are fixed, but their block-diagonal complexes are not memoised: that
    # would duplicate ~15 % of a GPU-resident variable-mesh data set, while re-batching on the GPU costs ~2.5 ms per
    # batch
    val_cache = None
    use_pde = (getattr(args, "aux_pde", 0.0) or 0.0) > 0 and model_name == "rhmp"
    if use_pde and not (isinstance(task.meta, dict) and task.meta.get("pde")):
        _log(f"  --aux-pde: task {args.task} has no meta['pde'] (PDE source / boundary data); ignored", q)
        use_pde = False
    t_fit = time.time()
    for ep in range(start_ep, args.epochs + 1):
        model.train()
        model.record_diagnostics = False
        _reset_peak(device)
        _sync(device)
        t0 = time.time()
        src_idx = tr_idx if task.variable_mesh else tr_idx.to(device)   # shared meshes: gather on the GPU
        batches = list(iterate_indices(src_idx, bs, shuffle=True, generator=gen))
        if args.max_train_batches:
            batches = batches[:args.max_train_batches]
        loss_sum, n_steps = torch.zeros((), device=device), 0
        aux_sum = torch.zeros((), device=device)
        if task.variable_mesh:
            stream = _Prefetch(lambda b: make_batch(task, b, device), batches)
        else:
            stream = ((b, make_batch(task, b, device)) for b in batches)
        for _, (K, xb, yb) in stream:
            opt.zero_grad(set_to_none=True)
            mse = F.mse_loss(model_output(task, fwd, xb, K), yb)
            loss = mse
            if use_pde:
                res = pde_residual(model, task, xb, yb, K)
                if res is not None:
                    aux = res.mean()
                    loss = mse + args.aux_pde * aux
                    aux_sum += aux.detach()
            loss.backward()
            if args.clip and args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            loss_sum += mse.detach()
            n_steps += 1
        sched.step()
        _sync(device)
        t_train = time.time() - t0
        train_loss = float(loss_sum.item()) / max(n_steps, 1)
        if not math.isfinite(train_loss):
            raise FloatingPointError(f"non-finite training loss at epoch {ep}")
        t1 = time.time()
        model.record_diagnostics = True
        val = evaluate(model, task, va_idx, ebs, device, fwd=model, cache=val_cache)
        diag_ep = model.diagnostics if hasattr(model, "diagnostics") else {}
        conds, sat = _cond_summary(diag_ep), _sat_summary(diag_ep)
        conds_learned = _cond_summary(diag_ep, ".cond_learned")
        _sync(device)
        t_eval = time.time() - t1
        rec = {"epoch": ep, "lr": opt.param_groups[0]["lr"], "train_loss": train_loss, "steps": n_steps,
               **{f"val_{k}": v for k, v in val.items()}, "R2": val["R2"],
               "time": t_train, "eval_time": t_eval, "peak_mem_GB": _peak_gb(device), "cond": conds,
               "cond_learned": conds_learned, "sat": sat}
        if use_pde:
            rec["train_aux_pde"] = float(aux_sum.item()) / max(n_steps, 1)
        solve = {k: v for k, v in diag_ep.items() if ".solve_" in k}
        if solve:                               # last validation batch: CG iterations / final residual per solve layer
            rec["solve"] = solve
        history.append(rec)
        if val["R2"] > best_r2:
            best_r2, best_ep = val["R2"], ep
            torch.save({**model.to_checkpoint(), "epoch": ep, "val": val, "task": args.task, "native": args.native},
                       best_path)
        torch.save({"model": model.to_checkpoint(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "rng": _rng_state(gen), "history": history, "best_r2": best_r2, "best_epoch": best_ep,
                    "epoch": ep, "signature": sig}, last_path)
        _write_json(os.path.join(out, "history.json"), history)
        if args.stop_after is not None and ep >= args.stop_after and ep < args.epochs:
            _log(f"  stopping after epoch {ep} (--stop-after)", q)
            return {"stopped_after": ep, "history": history, "complete": False}
        if ep % 5 == 0 or ep <= 2 or ep == args.epochs:
            _log(f"  ep{ep:4d} loss={train_loss:.5f} val R2={val['R2']:.4f} MSE={val['MSE']:.5f} "
                 f"({t_train:.1f}s train, {t_eval:.1f}s eval, {rec['peak_mem_GB']:.2f} GB) "
                 f"cond={max(conds.values()) if conds else float('nan'):.2f}", q)
    t_fit = time.time() - t_fit + sum(h["time"] + h["eval_time"] for h in history[:start_ep - 1])

    # ---------------- test with the best model ----------------
    ck = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    model.record_diagnostics = True
    t2 = time.time()
    test = evaluate(model, task, te_idx, ebs, device)
    diag = model.diagnostics if hasattr(model, "diagnostics") else {}
    r2_def = ("uncentred: 1 - SS_res / sum t^2 (orientation-odd cochain target)" if task.target_kind == "cochain"
              else ("1 - SS_res / SS_tot, SS_tot centred on the pooled mean (ragged meshes)"
                    if task.variable_mesh and not isinstance(test.get("N"), int) else
                    "v1: 1 - SS_res / SS_tot, SS_tot centred on the per-position mean over samples"
                    " (pooled mean for meshes of different sizes)"))
    result = {"task": args.task, "native": args.native, "mode": mode, "seed": args.seed, "model": model_name,
              "model_info": model_info, "params": n_params, "r2_definition": r2_def,
              "metric_reference_raw": True,
              "params_M": n_params / 1e6, "best_epoch": best_ep, "best_val_R2": best_r2,
              "test": test, "test_R2": test["R2"], "test_SSIM": test["SSIM"], "test_Pearson": test["Pearson"],
              "test_NRMSE": test["NRMSE"]}
    pe = task.meta.get("paper_eval")
    if pe is not None and len(pe):
        result["test100"] = evaluate(model, task, pe, ebs, device)
    for name, src in task.extra_tests.items():
        idx = torch.arange(len(src["target"]))
        result[name] = evaluate(model, task, idx, ebs, device, source=src)
    result.update({
        "test_seconds": time.time() - t2, "wall_clock_s": t_fit, "load_seconds": t_load,
        "epochs_run": len(history), "s_per_epoch": float(np.mean([h["time"] for h in history])) if history else None,
        "peak_mem_GB": max((h["peak_mem_GB"] for h in history), default=0.0),
        "diagnostics": {"cond_max": _cond_summary(diag), "cond_learned_max": _cond_summary(diag, ".cond_learned"),
                        "sat": _sat_summary(diag),
                        "clamp_fraction_max": _cond_summary(diag, ".clamp_fraction"), "all": diag},
        "config": cfg.to_dict(), "args": vars(args), "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "complete": len(history) >= args.epochs,
    })
    _write_json(res_path, result)
    extra = "".join(f" {k}_R2={result[k]['R2']:.4f}" for k in ("test100", *task.extra_tests) if k in result)
    _log(f"[{args.task}/{mode}] best val R2={best_r2:.4f}@{best_ep} test R2={test['R2']:.4f} "
         f"SSIM={test['SSIM']:.4f} Pearson={test['Pearson']:.4f} NRMSE={test['NRMSE']:.4f}{extra} "
         f"({t_fit:.0f}s, {result['s_per_epoch']:.1f} s/epoch, {n_params/1e6:.3f}M params)", q)
    return result


# ================================================================================================================
# transfer evaluation of a trained v2 run
# ================================================================================================================
def _raw_columns(cfg) -> dict[int, list[int]]:
    """Material columns (last ``material_dims[k]`` even columns) and metric-reference columns per degree."""
    out: dict[int, set] = {}
    for k, M_ in dict(getattr(cfg, "material_dims", {}) or {}).items():
        if M_:
            F = int(cfg.in_dims.get(k, 0))
            out.setdefault(int(k), set()).update(range(F - int(M_), F))
    for k, col in dict(getattr(cfg, "metric_reference", {}) or {}).items():
        out.setdefault(int(k), set()).add(int(col))
    return {k: sorted(v) for k, v in sorted(out.items())}


def _set_stats(task: TaskData, new_x: dict, new_y) -> None:
    """Re-express the task's normalised data (training, validation, test and ``extra_tests``) in new statistics
    (in place): ``x -> new.normalize(old.denormalize(x))`` per degree, likewise for the target."""
    old_x, old_y = task.x_stats, task.y_stats
    keys = list(old_x) if isinstance(old_x, dict) else list(range(len(old_x)))

    def cx(k, x):
        return new_x[k].normalize(old_x[k].denormalize(x)) if k in new_x else x

    def cy(y):
        return new_y.normalize(old_y.denormalize(y)) if new_y is not old_y else y
    if task.variable_mesh:
        task.inputs = [{k: cx(k, v) for k, v in d.items()} for d in task.inputs]
        task.target = [cy(y) for y in task.target]
    else:
        task.inputs = {k: cx(k, v).contiguous() for k, v in task.inputs.items()}
        task.target = cy(task.target).contiguous()
    for src in task.extra_tests.values():
        if "inputs" in src and "target" in src:
            src["inputs"] = [{k: cx(k, v) for k, v in d.items()} for d in src["inputs"]]
            src["target"] = [cy(y) for y in src["target"]]
    merged = {k: new_x.get(k, old_x[k]) for k in keys}
    task.x_stats = merged if isinstance(old_x, dict) else [merged[k] for k in keys]
    task.y_stats = new_y


def _raw_reference_normalization(task: TaskData, cfg) -> dict[int, list[int]]:
    """Material and metric-reference columns arrive raw (identity statistics: physical log values), in every mode.

    ``metric_reference`` is an absolute log material (``H = star exp(ref)``): with the loaders' mean/std statistics
    the metric would be ``star sigma^(1/s) exp(-mu/s)``.  The metric heads still see these columns as descriptors
    (log fields of order one).  Returns the raw columns ``{k: [col, ...]}`` (in place; no-op if there are none).
    """
    from rhmp.data import Stats
    raw = _raw_columns(cfg)
    new_x = {}
    for k, cols in raw.items():
        st = task.x_stats[k]
        idx = torch.as_tensor(cols, dtype=torch.long, device=st.mean.device)
        mean, std = st.mean.clone(), st.std.clone()
        mean[idx], std[idx] = 0.0, 1.0
        new_x[k] = Stats(mean, std, st.kind)
    if new_x:
        _set_stats(task, new_x, task.y_stats)
    return raw


def _solver_normalization(task: TaskData, cfg) -> dict[int, list[int]]:
    """``--solver-mode``: statistics under which the solver class stays exact (in place, before training).

    A model that is linear in the field inputs cannot absorb the offsets of affine (mean/std) normalisations (a shifted
    source and a shifted target break ``u = S_H f`` and the zero boundary values), and a metric reference must be the
    absolute log material (``H = star exp(ref)``, physical units).  So: the lifted input columns and the target get
    mean-free statistics (``std = sqrt(std^2 + mean^2)``, the RMS of the training split), the material and
    metric-reference columns identity statistics (raw values).  R2 is unaffected (affine invariant); MSE/MAE are in
    the new normalised units.  Returns the raw columns.
    """
    from rhmp.data import Stats

    def rms_only(st: Stats, raw_cols: list[int]) -> Stats:
        std = torch.sqrt(st.std.square() + st.mean.square()).clamp_min(1e-6)
        if raw_cols:
            std[torch.as_tensor(raw_cols, dtype=torch.long, device=std.device)] = 1.0
        return Stats(torch.zeros_like(st.mean), std, "solver")

    raw = _raw_columns(cfg)
    keys = list(task.x_stats) if isinstance(task.x_stats, dict) else list(range(len(task.x_stats)))
    _set_stats(task, {k: rms_only(task.x_stats[k], raw.get(k, [])) for k in keys}, rms_only(task.y_stats, []))
    return raw


def _renormalize(task: TaskData, train_summary: dict) -> TaskData:
    """Re-express ``task``'s normalised inputs/targets in the normalisation of a training run (in place)."""
    from rhmp.data import Stats

    def st(d, like):
        return Stats(torch.tensor(d["mean"], dtype=like.dtype, device=like.device),
                     torch.tensor(d["std"], dtype=like.dtype, device=like.device), d.get("kind", "v1"))

    xs_tr = {int(k): v for k, v in train_summary["x_stats"].items()}
    if set(xs_tr) != set(task.x_stats):
        raise ValueError(f"input degrees differ: run {sorted(xs_tr)} vs task {sorted(task.x_stats)}")
    like = task.y_stats.mean
    ys_tr = st(train_summary["y_stats"], like)
    conv_x = {k: (task.x_stats[k], st(xs_tr[k], like)) for k in xs_tr}
    conv_y = (task.y_stats, ys_tr)

    def cx(k, x):
        a, b = conv_x[k]
        return b.normalize(a.denormalize(x))

    def cy(y):
        return conv_y[1].normalize(conv_y[0].denormalize(y))
    if task.variable_mesh:
        task.inputs = [{k: cx(k, v) for k, v in d.items()} for d in task.inputs]
        task.target = [cy(y) for y in task.target]
    else:
        task.inputs = {k: cx(k, v) for k, v in task.inputs.items()}
        task.target = cy(task.target)
    for src in task.extra_tests.values():
        src["inputs"] = [{k: cx(k, v) for k, v in d.items()} for d in src["inputs"]]
        src["target"] = [cy(y) for y in src["target"]]
    task.x_stats = {k: conv_x[k][1] for k in conv_x}
    task.y_stats = ys_tr
    return task


def eval_ckpt(args: argparse.Namespace, task: TaskData, device, out: str) -> dict:
    """Evaluate ``<RUN_DIR>/best.pt`` on the test split (and extra test sets) of ``task``; writes
    ``result_eval_<task>.json`` into ``--out``.  Typical use: train on ``HP_k100``, evaluate on ``HP_k1000`` /
    ``HP_k10000`` (contrast transfer) or on ``--star unit`` complexes, etc."""
    from rhmp.model import RHMP
    run_dir = args.eval_ckpt
    conf = json.load(open(os.path.join(run_dir, "config.json")))
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location=device, weights_only=False)
    trained = conf["args"].get("model") or "rhmp"
    if trained == "rhmp":
        model = RHMP.from_checkpoint(ck, map_location=device)
    else:
        import dataclasses

        from rhmp.baselines.registry import SPECS, from_checkpoint, prepare_task
        task = prepare_task(trained, task, ck.get("build"))
        model = from_checkpoint(ck, task, map_location=device)
        if not SPECS[trained].uses_output_map and task.output_map is not None:
            task = dataclasses.replace(task, output_map=None)
    model.eval()
    task = _renormalize(task, conf["data"])
    ebs = args.eval_batch or conf["args"].get("eval_batch") or \
        (conf["args"].get("bs_meshes") or 8 if task.variable_mesh else conf["args"].get("batch") or 64)
    t0 = time.time()
    res = {"run": run_dir, "train_task": conf["args"]["task"], "eval_task": args.task, "native": args.native,
           "test": evaluate(model, task, task.split[2], ebs, device)}
    for name, src in task.extra_tests.items():
        res[name] = evaluate(model, task, torch.arange(len(src["target"])), ebs, device, source=src)
    res["seconds"] = time.time() - t0
    _write_json(os.path.join(out, f"result_eval_{args.task}.json"), res)
    extra = "".join(f" {k}_R2={res[k]['R2']:.4f}" for k in task.extra_tests if k in res)
    _log(f"[eval {conf['args']['task']} -> {args.task}] test R2={res['test']['R2']:.4f} "
         f"SSIM={res['test']['SSIM']:.4f}{extra}", args.quiet)
    return res


# ================================================================================================================
# v1 reference evaluation
# ================================================================================================================
V1_CFG = {  # v1 compute_all_metrics.py OURS_CFG (+ T8 from evaluate_task_persample)
    "T1": dict(C=128, task="scalar", sdim=2), "T2": dict(C=64, task="scalar", sdim=2),
    "T3": dict(C=64, task="vector", sdim=3), "T5": dict(C=160, task="scalar", sdim=2),
    "T6": dict(C=128, task="scalar", sdim=2), "T7": dict(C=160, task="scalar", sdim=2),
    "T8": dict(C=None, task="scalar", sdim=3),
}


def eval_v1(args: argparse.Namespace, task: TaskData, device, out: str) -> dict:
    """Evaluate a v1 ``GaugeHodgeNetwork`` checkpoint on the legacy data of ``task`` with the v2 metrics.

    The v1 model (vendored in ``rhmp.baselines.v1``) is rebuilt exactly as the v1 evaluation script
    ``compute_all_metrics.py`` does; predictions are made per sample (``--v1-eval-batch 1``, as for the v1 paper
    tables) because v1's ``forward_batch`` couples the samples of a batch through the metric.  Writes
    ``result_v1.json``.
    """
    from rhmp.baselines.v1.gauge_hodge_mp.cell_complex import CellComplex
    from rhmp.baselines.v1.gauge_hodge_mp.network import GaugeHodgeNetwork

    if task.native and task.name not in ("T2", "T8"):
        raise ValueError("--eval-v1 needs the legacy inputs: add --legacy")
    base = task.name
    if base not in V1_CFG:
        raise ValueError(f"no v1 model for task {base}")
    vc = V1_CFG[base]
    state = torch.load(args.eval_v1, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    f_in = task.in_dims[0]
    out_dim = task.out_dim
    tr_idx, va_idx, te_idx = task.split
    t0 = time.time()
    import pickle
    from rhmp.tasks import resolve_root
    from rhmp.tasks.paper import V1_FILES
    droot = resolve_root(args.root)
    raw = pickle.load(open(os.path.join(droot, V1_FILES[base][0]), "rb"))

    def v1_complex(pts, faces):
        return CellComplex.from_triangulation(torch.tensor(np.asarray(pts, dtype=np.float32)),
                                              torch.tensor(np.asarray(faces, dtype=np.int64))).to(device)

    preds = {}
    with torch.no_grad():
        if not task.variable_mesh:
            Kv1 = v1_complex(raw["points"], raw["faces"])
            model = GaugeHodgeNetwork(f_in=f_in, C=vc["C"], n_layers=4, n0=Kv1.n0, n1=Kv1.n1, n2=Kv1.n2,
                                      task=vc["task"], spatial_dim=vc["sdim"], out_dim=out_dim, mp_hidden=16,
                                      metric_type="diagonal", metric_rank=8)
            model.load_state_dict(state)
            model = model.to(device).eval()
            X = task.inputs[0]
            bsz = max(1, args.v1_eval_batch)

            def run_idx(idx):
                outs = []
                for s in range(0, len(idx), bsz):
                    b = idx[s:s + bsz].to(device)
                    outs.append(model.forward_batch(X.index_select(0, b), Kv1))
                return torch.cat(outs)
            for name, idx in (("val", va_idx), ("test", te_idx)):
                preds[name] = (run_idx(idx), task.target.index_select(0, idx.to(device)))
        else:
            C = vc["C"] or state["lifting.node_mlp.0.weight"].shape[0]
            model = None
            for name, idx in (("val", va_idx), ("test", te_idx)):
                ps, ts = [], []
                for i in idx.tolist():
                    s = raw[i]
                    Kv1 = v1_complex(s["pts_3d"], s["faces"])
                    if model is None:
                        model = GaugeHodgeNetwork(f_in=f_in, C=C, n_layers=4, n0=Kv1.n0, n1=Kv1.n1, n2=Kv1.n2,
                                                  task="scalar", spatial_dim=3, out_dim=out_dim, mp_hidden=16,
                                                  metric_type="local_rich", metric_rank=8)
                        model.load_state_dict(state)
                        model = model.to(device).eval()
                    ps.append(model(task.inputs[i][0].to(device), Kv1))
                    ts.append(task.target[i].to(device))
                preds[name] = (torch.stack(ps), torch.stack(ts))
    t_pred = time.time() - t0
    stats = task.y_stats.as_tuple()
    r2s = task.meta.get("r2_std")
    result = {"task": base, "model": "v1 GaugeHodgeNetwork", "checkpoint": args.eval_v1,
              "params": sum(p.numel() for p in model.parameters()),
              "val": M.summarize(*preds["val"], stats, r2_std=r2s),
              "test": M.summarize(*preds["test"], stats, r2_std=r2s), "predict_seconds": t_pred,
              "v1_eval_batch": args.v1_eval_batch}
    pe = task.meta.get("paper_eval")
    if pe is not None and len(pe):
        pos = {int(i): j for j, i in enumerate(te_idx.tolist())}
        sel = torch.tensor([pos[int(i)] for i in pe.tolist() if int(i) in pos], device=device)
        p, t = preds["test"]
        result["test100"] = M.summarize(p.index_select(0, sel), t.index_select(0, sel), stats, r2_std=r2s)
    result["test_R2"] = result["test"]["R2"]
    _write_json(os.path.join(out, "result_v1.json"), result)
    _log(f"[v1 {base}] val R2={result['val']['R2']:.4f} test R2={result['test']['R2']:.4f} "
         f"test100 R2={result.get('test100', {}).get('R2', float('nan')):.4f} SSIM={result['test']['SSIM']:.4f} "
         f"Pearson={result['test']['Pearson']:.4f} ({t_pred:.1f}s)", args.quiet)
    return result


def main(argv: list[str] | None = None) -> None:
    """CLI entry point (``python3 -m rhmp.train``)."""
    run(parse_args(argv))


if __name__ == "__main__":
    main()
