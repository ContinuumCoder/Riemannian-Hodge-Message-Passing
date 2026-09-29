# Vendored v1 code.  Original: the metric functions of the v1 evaluation script `compute_all_metrics.py` of the
# RHMP v1 code base, https://github.com/ContinuumCoder/Riemannian-Hodge-Message-Passing, which accompanies
# "Learning Discrete Riemannian Metrics for Physical Fields with Cochain-Frame Equivariance"
# (Zheng & Allen-Blanchette, arXiv:2608.14556).  The three functions are copied verbatim; the rest of that script
# (checkpoint loading and the paper tables) is not needed here.
"""v1 metric functions (reference for ``rhmp.metrics``; ``tests/test_metrics.py`` checks agreement to 1e-6).

``compute_r2_mse_mae`` works on normalised torch tensors; ``compute_nrmse`` and ``compute_ssim_pearson`` on
unnormalised numpy arrays, exactly as in the v1 paper tables (first 100 test samples for SSIM / Pearson).
"""
import numpy as np
import torch  # noqa: F401  (the functions receive torch tensors / numpy arrays)

__all__ = ["compute_r2_mse_mae", "compute_nrmse", "compute_ssim_pearson"]


def compute_r2_mse_mae(pred, target):
    """R², MSE, MAE on normalized predictions."""
    mse = ((pred - target) ** 2).mean().item()
    mae = (pred - target).abs().mean().item()
    ss_res = ((pred - target) ** 2).sum().item()
    ss_tot = ((target - target.mean(0, keepdim=True)) ** 2).sum().item()
    r2 = 1 - ss_res / max(ss_tot, 1e-8)
    return r2, mse, mae


def compute_nrmse(pred_raw, target_raw):
    """NRMSE = RMSE / range(target), computed on unnormalized data."""
    rmse = np.sqrt(np.mean((pred_raw - target_raw) ** 2))
    data_range = target_raw.max() - target_raw.min()
    return rmse / max(data_range, 1e-8)


def compute_ssim_pearson(pred_raw, target_raw, n_samples=None):
    """Per-sample SSIM and Pearson on unnormalized predictions.

    SSIM uses GLOBAL data_range from all target samples (consistent C1/C2).
    Predictions clipped to target range ± 10% to prevent outlier-inflated variance.
    Pearson clipped to ≥ 0 (negative = no correlation, avoids reviewer confusion).

    Args:
        pred_raw:   (N, ...) numpy, unnormalized predictions
        target_raw: (N, ...) numpy, unnormalized targets
        n_samples:  max samples to evaluate (None = all)
    Returns:
        mean_ssim, mean_pearson
    """
    N = len(pred_raw)
    if n_samples is not None:
        N = min(N, n_samples)

    # Global data range
    data_range = float(target_raw.max() - target_raw.min())
    if data_range < 1e-8:
        data_range = 1.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    # Clip predictions
    t_min, t_max = float(target_raw.min()), float(target_raw.max())
    margin = 0.1 * data_range
    pred_clipped = np.clip(pred_raw[:N], t_min - margin, t_max + margin)

    ssims, pearsons = [], []
    for i in range(N):
        p = pred_clipped[i].flatten().astype(np.float64)
        t = target_raw[i].flatten().astype(np.float64)

        # Pearson (clip ≥ 0)
        tp = t - t.mean()
        pp = p - p.mean()
        denom = np.sqrt(np.sum(tp ** 2) * np.sum(pp ** 2))
        pearson = np.sum(tp * pp) / (denom + 1e-15) if denom > 1e-15 else 0.0
        pearsons.append(max(0.0, pearson))

        # SSIM (global C1/C2)
        mu_t, mu_p = t.mean(), p.mean()
        sig_t, sig_p = t.std(), p.std()
        sig_tp = np.mean((t - mu_t) * (p - mu_p))
        num = (2 * mu_t * mu_p + c1) * (2 * sig_tp + c2)
        den = (mu_t ** 2 + mu_p ** 2 + c1) * (sig_t ** 2 + sig_p ** 2 + c2)
        ssims.append(num / (den + 1e-15))

    return float(np.mean(ssims)), float(np.mean(pearsons))

