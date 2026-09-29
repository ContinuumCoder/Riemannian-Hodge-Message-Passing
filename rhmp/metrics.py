"""Evaluation metrics in torch (vectorised, any device), numerically matching the v1 numpy/torch code.

Reference implementations: the v1 evaluation script ``compute_all_metrics.py``, functions ``compute_r2_mse_mae``,
``compute_nrmse``, ``compute_ssim_pearson`` (vendored in ``rhmp.baselines.v1.metrics_v1``).  ``tests/test_metrics.py`` checks
agreement to 1e-6 on random data.

Conventions (identical to v1):

* ``R2``, ``MSE``, ``MAE`` are computed in the *normalised* target space (``(y - mean) / std``, per feature,
  statistics from the training split).  ``R2 = 1 - SS_res / max(SS_tot, 1e-8)`` with
  ``SS_tot = sum (t - t.mean(0))**2``: the mean is taken over samples, per (cell, feature) position.
* ``NRMSE``, ``SSIM`` and ``Pearson`` are computed on *unnormalised* values.  NRMSE = RMSE / range(target).
  SSIM / Pearson are per-sample (all cells and all output features of a sample flattened together) and then
  averaged over samples; SSIM uses a *global* data range (from all target samples) for C1/C2 and predictions are
  clipped to ``[t_min - 0.1 range, t_max + 0.1 range]``; Pearson is clipped to ``>= 0``.

Orientation-odd cochain targets (``cochain:k`` readouts: fluxes, curvatures) are scored with an *uncentred* R2,
``SS_tot = sum t**2`` (``center=False``): the mean of an odd quantity depends on the orientation convention of the
cells, so centring would make R2 depend on vertex labels.  The trainer selects this automatically.

Ragged data (variable meshes, e.g. T8/HP/TET block-diagonal batches) is supported by passing lists of per-sample
tensors ``[(n_i, d), ...]``.  For ragged data the R2 total sum of squares uses the pooled per-feature mean
(positions of different meshes do not correspond); for equal-size samples pass a stacked ``(N, n, d)`` tensor to get
exactly the v1 definition.

All reductions are done in float64.
"""
from __future__ import annotations

from typing import Sequence, Union

import torch

Tensor = torch.Tensor
TensorOrList = Union[Tensor, Sequence[Tensor]]

R2_EPS = 1e-8       # v1: 1 - ss_res / max(ss_tot, 1e-8)
RANGE_EPS = 1e-8    # v1: data-range floor (NRMSE denominator; SSIM falls back to range 1.0 below it)
DEN_EPS = 1e-15     # v1: SSIM / Pearson denominator guard
CLIP_MARGIN = 0.1   # v1: predictions clipped to target range +- 10 % for SSIM / Pearson

__all__ = [
    "r2_score", "mse", "mae", "nrmse", "ssim_pearson", "summarize", "unnormalize", "flatten_ragged",
]


# ----------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------
def flatten_ragged(x: TensorOrList) -> tuple[Tensor, Tensor, int]:
    """Concatenate per-sample tensors.

    Args:
        x: ``(N, n, ...)`` tensor, or a list of ``N`` tensors ``(n_i, ...)`` with equal trailing shape.
    Returns:
        ``(flat, seg, N)``: ``flat`` is ``(M, d)`` (trailing dims flattened into ``d``), ``seg`` is ``(M,)`` int64
        sample ids, ``N`` the number of samples.
    """
    if isinstance(x, Tensor):
        N = x.shape[0]
        n = x.shape[1] if x.dim() > 1 else 1
        flat = x.reshape(N * n, -1)
        seg = torch.arange(N, device=x.device).repeat_interleave(n)
        return flat, seg, N
    xs = [xi.reshape(xi.shape[0], -1) if xi.dim() > 1 else xi.reshape(-1, 1) for xi in x]
    if len(xs) == 0:
        raise ValueError("flatten_ragged: empty list")
    flat = torch.cat(xs, 0)
    sizes = torch.tensor([xi.shape[0] for xi in xs], device=flat.device)
    seg = torch.arange(len(xs), device=flat.device).repeat_interleave(sizes)
    return flat, seg, len(xs)


def _as_pair(pred: TensorOrList, target: TensorOrList) -> tuple[Tensor, Tensor, Tensor, int, bool]:
    """Return ``(p_flat, t_flat, seg, N, dense)`` with matching shapes."""
    dense = isinstance(pred, Tensor) and isinstance(target, Tensor)
    if dense:
        if pred.shape != target.shape:
            raise ValueError(f"pred {tuple(pred.shape)} and target {tuple(target.shape)} differ")
    p, sp, N = flatten_ragged(pred)
    t, st, Nt = flatten_ragged(target)
    if p.shape != t.shape or N != Nt:
        raise ValueError(f"ragged pred/target mismatch: {tuple(p.shape)} vs {tuple(t.shape)}")
    return p, t, st, N, dense


def unnormalize(x: TensorOrList, mean: Tensor, std: Tensor) -> TensorOrList:
    """``x * std + mean`` (broadcast over the last dim); works on tensors or lists of tensors."""
    if isinstance(x, Tensor):
        return x * std.to(x) + mean.to(x)
    return [xi * std.to(xi) + mean.to(xi) for xi in x]


# ----------------------------------------------------------------------------------------------------------------
# normalised-space metrics
# ----------------------------------------------------------------------------------------------------------------
def r2_score(pred: TensorOrList, target: TensorOrList, center: bool = True) -> float:
    """Coefficient of determination, v1 definition.

    Args:
        pred, target: ``(N, n, d)`` (dense; SS_tot uses the per-position mean over samples, as v1) or lists of
            ``(n_i, d)`` (ragged; SS_tot uses the pooled per-feature mean).
        center: subtract the mean in SS_tot (v1).  ``False``: ``SS_tot = sum t**2`` (orientation-odd targets).
    Returns:
        ``1 - SS_res / max(SS_tot, 1e-8)`` as a Python float.
    """
    if isinstance(pred, Tensor) and isinstance(target, Tensor):
        if pred.shape != target.shape:
            raise ValueError(f"pred {tuple(pred.shape)} and target {tuple(target.shape)} differ")
        p = pred.double()
        t = target.double()
    else:
        p, t, _, _, _ = _as_pair(pred, target)
        p = p.double()
        t = t.double()
    ss_res = ((p - t) ** 2).sum()
    ss_tot = ((t - t.mean(0, keepdim=True)) ** 2).sum() if center else (t ** 2).sum()
    return float(1.0 - ss_res.item() / max(ss_tot.item(), R2_EPS))


def mse(pred: TensorOrList, target: TensorOrList) -> float:
    """Mean squared error over all elements (float64 accumulation)."""
    p, t, _, _, _ = _as_pair(pred, target)
    return float(((p.double() - t.double()) ** 2).mean().item())


def mae(pred: TensorOrList, target: TensorOrList) -> float:
    """Mean absolute error over all elements (float64 accumulation)."""
    p, t, _, _, _ = _as_pair(pred, target)
    return float((p.double() - t.double()).abs().mean().item())


# ----------------------------------------------------------------------------------------------------------------
# unnormalised-space metrics
# ----------------------------------------------------------------------------------------------------------------
def nrmse(pred_raw: TensorOrList, target_raw: TensorOrList) -> float:
    """``RMSE / max(range(target), 1e-8)`` over all elements (unnormalised values)."""
    p, t, _, _, _ = _as_pair(pred_raw, target_raw)
    rmse = torch.sqrt(((p.double() - t.double()) ** 2).mean()).item()
    rng = (t.max() - t.min()).item()          # original dtype, as numpy does
    return float(rmse / max(rng, RANGE_EPS))


def ssim_pearson(pred_raw: TensorOrList, target_raw: TensorOrList,
                 n_samples: int | None = None) -> tuple[float, float]:
    """Per-sample global-range SSIM and clipped Pearson correlation, averaged over samples (v1 definition).

    Args:
        pred_raw, target_raw: unnormalised ``(N, ...)`` tensors, or lists of per-sample ``(n_i, ...)`` tensors.
        n_samples: evaluate only the first ``n_samples`` samples (the global data range and clipping bounds are
            still computed from *all* targets, exactly as v1).
    Returns:
        ``(mean_ssim, mean_pearson)``.
    """
    p, t, seg, N, _ = _as_pair(pred_raw, target_raw)
    t_max = t.max()
    t_min = t.min()
    data_range = float((t_max - t_min).item())   # original dtype subtraction, as numpy
    if data_range < RANGE_EPS:
        data_range = 1.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    margin = CLIP_MARGIN * data_range
    lo, hi = float(t_min.item()) - margin, float(t_max.item()) + margin

    if n_samples is not None:
        N = min(N, int(n_samples))
        keep = seg < N
        p, t, seg = p[keep], t[keep], seg[keep]
    # clip in the original dtype (np.clip keeps float32), then promote
    p = torch.clamp(p, lo, hi).double().reshape(-1)
    d = t.shape[1] if t.dim() > 1 else 1
    t = t.double().reshape(-1)
    seg = seg.repeat_interleave(d)

    dev = p.device
    cnt = torch.zeros(N, dtype=torch.float64, device=dev).index_add_(0, seg, torch.ones_like(p))
    mu_t = torch.zeros(N, dtype=torch.float64, device=dev).index_add_(0, seg, t) / cnt
    mu_p = torch.zeros(N, dtype=torch.float64, device=dev).index_add_(0, seg, p) / cnt
    tp = t - mu_t[seg]
    pp = p - mu_p[seg]
    s_tt = torch.zeros(N, dtype=torch.float64, device=dev).index_add_(0, seg, tp * tp)
    s_pp = torch.zeros(N, dtype=torch.float64, device=dev).index_add_(0, seg, pp * pp)
    s_tp = torch.zeros(N, dtype=torch.float64, device=dev).index_add_(0, seg, tp * pp)

    # Pearson (clipped >= 0)
    denom = torch.sqrt(s_tt * s_pp)
    pearson = torch.where(denom > DEN_EPS, s_tp / (denom + DEN_EPS), torch.zeros_like(denom))
    pearson = pearson.clamp(min=0.0)

    # SSIM with global C1/C2 (population statistics, ddof = 0)
    var_t = s_tt / cnt
    var_p = s_pp / cnt
    cov = s_tp / cnt
    num = (2 * mu_t * mu_p + c1) * (2 * cov + c2)
    den = (mu_t ** 2 + mu_p ** 2 + c1) * (var_t + var_p + c2)
    ssim = num / (den + DEN_EPS)
    return float(ssim.mean().item()), float(pearson.mean().item())


# ----------------------------------------------------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------------------------------------------------
def _stats_tensors(y_stats, like: Tensor) -> tuple[Tensor, Tensor]:
    if isinstance(y_stats, dict):
        mean, std = y_stats["mean"], y_stats["std"]
    else:
        mean, std = y_stats
    return torch.as_tensor(mean, device=like.device), torch.as_tensor(std, device=like.device)


def summarize(pred: TensorOrList, target: TensorOrList, y_stats, *, r2_std: Tensor | None = None,
              n_samples: int | None = None, prefix: str = "", center: bool = True) -> dict[str, float]:
    """All metrics for normalised predictions/targets.

    Args:
        pred, target: normalised model outputs and targets, ``(N, n, d)`` or lists of ``(n_i, d)``.
        y_stats: ``(mean, std)`` or ``{'mean','std'}`` with shape ``(d,)``: raw = normalised * std + mean.
        r2_std: optional per-feature std ``(d,)`` defining the v1 normalisation. When the training normalisation
            differs from v1's (e.g. isotropic scaling of vector targets), ``R2_v1`` recomputes R2 in the v1
            normalised space (only the std matters: the mean cancels in R2).
        n_samples: restrict every metric (including the SSIM data range) to the first ``n_samples`` samples, as
            ``compute_all_metrics.evaluate_task`` did for the paper tables (100 test samples).
        prefix: prepended to every key.
        center: centred (v1) or uncentred (orientation-odd targets) R2, see :func:`r2_score`.
    Returns:
        dict with ``R2, MSE, MAE`` (normalised space), ``NRMSE, SSIM, Pearson`` (raw space), optional ``R2_v1``,
        and ``N`` (number of samples evaluated).
    """
    if n_samples is not None:
        pred, target = pred[:n_samples], target[:n_samples]
    like = pred if isinstance(pred, Tensor) else pred[0]
    mean, std = _stats_tensors(y_stats, like)
    out: dict[str, float] = {}
    out["R2"] = r2_score(pred, target, center=center)
    out["MSE"] = mse(pred, target)
    out["MAE"] = mae(pred, target)
    p_raw = unnormalize(pred, mean, std)
    t_raw = unnormalize(target, mean, std)
    out["NRMSE"] = nrmse(p_raw, t_raw)
    ssim, pear = ssim_pearson(p_raw, t_raw)
    out["SSIM"] = ssim
    out["Pearson"] = pear
    if r2_std is not None:
        s = torch.as_tensor(r2_std, device=like.device)
        if isinstance(p_raw, Tensor):
            out["R2_v1"] = r2_score(p_raw / s, t_raw / s, center=center)
        else:
            out["R2_v1"] = r2_score([a / s for a in p_raw], [b / s for b in t_raw], center=center)
    out["N"] = int(len(pred))
    if prefix:
        out = {prefix + k: v for k, v in out.items()}
    return out
