"""rhmp.metrics must reproduce the v1 metric functions (v1 compute_all_metrics.py, vendored as
rhmp.baselines.v1.metrics_v1) to 1e-6."""
import numpy as np
import pytest
import torch

from rhmp import metrics as M
from rhmp.baselines.v1 import metrics_v1 as v1

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
TOL = 1e-6


def _random_case(seed, N=24, n=57, d=3, dtype=np.float64):
    rng = np.random.RandomState(seed)
    t = rng.randn(N, n, d) * rng.uniform(0.1, 5.0) + rng.uniform(-3, 3)
    p = t + rng.randn(N, n, d) * rng.uniform(0.05, 2.0)
    # a few pathological samples: anti-correlated (Pearson clip), constant prediction (zero denominator),
    # outliers beyond the clipping range
    p[1] = -t[1]
    p[2] = 0.3
    p[3, :5] = t.max() + 50.0
    return p.astype(dtype), t.astype(dtype)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_r2_mse_mae_match_v1(seed, dtype, device):
    p, t = _random_case(seed, dtype=dtype)
    r2_ref, mse_ref, mae_ref = v1.compute_r2_mse_mae(torch.tensor(p, dtype=torch.float64),
                                                     torch.tensor(t, dtype=torch.float64))
    pt, tt = torch.tensor(p, device=device), torch.tensor(t, device=device)
    assert abs(M.r2_score(pt, tt) - r2_ref) < TOL
    assert abs(M.mse(pt, tt) - mse_ref) < TOL * max(1.0, abs(mse_ref))
    assert abs(M.mae(pt, tt) - mae_ref) < TOL * max(1.0, abs(mae_ref))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nrmse_ssim_pearson_match_v1(seed, dtype, device):
    p, t = _random_case(seed, dtype=dtype)
    pt, tt = torch.tensor(p, device=device), torch.tensor(t, device=device)
    assert abs(M.nrmse(pt, tt) - float(v1.compute_nrmse(p, t))) < TOL
    for n_samples in (None, 7):
        s_ref, p_ref = v1.compute_ssim_pearson(p, t, n_samples=n_samples)
        s, pr = M.ssim_pearson(pt, tt, n_samples=n_samples)
        assert abs(s - s_ref) < TOL, (s, s_ref)
        assert abs(pr - p_ref) < TOL, (pr, p_ref)


@pytest.mark.parametrize("device", DEVICES)
def test_degenerate_range(device):
    """Constant targets: v1 falls back to data_range = 1 for SSIM and 1e-8 for NRMSE."""
    t = np.full((5, 10, 1), 2.0)
    p = t + np.random.RandomState(0).randn(5, 10, 1) * 1e-3
    pt, tt = torch.tensor(p, device=device), torch.tensor(t, device=device)
    s_ref, p_ref = v1.compute_ssim_pearson(p, t)
    s, pr = M.ssim_pearson(pt, tt)
    assert abs(s - s_ref) < TOL and abs(pr - p_ref) < TOL
    assert abs(M.nrmse(pt, tt) - float(v1.compute_nrmse(p, t))) / float(v1.compute_nrmse(p, t)) < 1e-6


@pytest.mark.parametrize("device", DEVICES)
def test_ragged_equals_dense(device):
    """Lists of per-sample tensors give the same per-sample metrics as the stacked tensor; ragged R2 = pooled R2."""
    p, t = _random_case(3)
    pt, tt = torch.tensor(p, device=device), torch.tensor(t, device=device)
    pl, tl = list(pt.unbind(0)), list(tt.unbind(0))
    assert abs(M.nrmse(pl, tl) - M.nrmse(pt, tt)) < 1e-12
    s1, p1 = M.ssim_pearson(pl, tl)
    s2, p2 = M.ssim_pearson(pt, tt)
    assert abs(s1 - s2) < 1e-12 and abs(p1 - p2) < 1e-12
    # genuinely ragged sizes: compare with a per-sample numpy loop (v1 formula applied per sample)
    rng = np.random.RandomState(5)
    sizes = [13, 40, 7, 25]
    tl = [rng.randn(s, 2) * 2 + 1 for s in sizes]
    pl = [x + rng.randn(*x.shape) * 0.4 for x in tl]
    t_all = np.concatenate(tl)
    rng_ = t_all.max() - t_all.min()
    c1, c2 = (0.01 * rng_) ** 2, (0.03 * rng_) ** 2
    ss, pp = [], []
    for a, b in zip(pl, tl):
        a = np.clip(a, t_all.min() - 0.1 * rng_, t_all.max() + 0.1 * rng_).ravel()
        b = b.ravel()
        num = (2 * b.mean() * a.mean() + c1) * (2 * np.mean((b - b.mean()) * (a - a.mean())) + c2)
        den = (b.mean() ** 2 + a.mean() ** 2 + c1) * (b.std() ** 2 + a.std() ** 2 + c2)
        ss.append(num / (den + 1e-15))
        pp.append(max(0.0, np.corrcoef(a, b)[0, 1]))
    s, pr = M.ssim_pearson([torch.tensor(x, device=device) for x in pl], [torch.tensor(x, device=device) for x in tl])
    assert abs(s - np.mean(ss)) < TOL and abs(pr - np.mean(pp)) < TOL
    p_all = np.concatenate(pl)
    r2_pooled = 1 - ((p_all - t_all) ** 2).sum() / ((t_all - t_all.mean(0)) ** 2).sum()
    r2 = M.r2_score([torch.tensor(x, device=device) for x in pl], [torch.tensor(x, device=device) for x in tl])
    assert abs(r2 - r2_pooled) < TOL


@pytest.mark.parametrize("device", DEVICES)
def test_summarize_matches_v1_pipeline(device):
    """summarize(normalised pred/target, y_stats) == v1 evaluate_task arithmetic (first n samples)."""
    rng = np.random.RandomState(7)
    t_raw = (rng.randn(30, 40, 2) * np.array([3.0, 0.5]) + np.array([1.0, -2.0])).astype(np.float64)
    p_raw = t_raw + rng.randn(*t_raw.shape) * 0.3
    ym, ys = np.array([0.7, -1.9]), np.array([2.9, 0.45])
    pn, tn = (p_raw - ym) / ys, (t_raw - ym) / ys
    out = M.summarize(torch.tensor(pn, device=device), torch.tensor(tn, device=device),
                      (torch.tensor(ym), torch.tensor(ys)), n_samples=20, r2_std=torch.tensor(ys))
    r2, mse, mae = v1.compute_r2_mse_mae(torch.tensor(pn[:20]), torch.tensor(tn[:20]))
    s_ref, p_ref = v1.compute_ssim_pearson(p_raw[:20], t_raw[:20])
    assert abs(out["R2"] - r2) < TOL and abs(out["MSE"] - mse) < TOL and abs(out["MAE"] - mae) < TOL
    assert abs(out["R2_v1"] - r2) < TOL
    assert abs(out["NRMSE"] - float(v1.compute_nrmse(p_raw[:20], t_raw[:20]))) < TOL
    assert abs(out["SSIM"] - s_ref) < TOL and abs(out["Pearson"] - p_ref) < TOL
    assert out["N"] == 20


def test_formal_benchmark_r2_is_the_same_formula():
    """formal_benchmark.compute_metrics (used for model selection in v1) is the same R2 formula."""
    p, t = _random_case(4)
    ss_res = ((p - t) ** 2).sum()
    ss_tot = ((t - t.mean(0, keepdims=True)) ** 2).sum()
    assert abs(M.r2_score(torch.tensor(p), torch.tensor(t)) - (1 - ss_res / ss_tot)) < 1e-12


def test_uncentred_r2_for_odd_targets():
    """center=False: SS_tot = sum t^2, invariant to flipping the sign of any subset of cells (odd cochains)."""
    rng = np.random.RandomState(11)
    t = rng.randn(6, 30, 1) + 0.5
    p = t + 0.3 * rng.randn(*t.shape)
    flip = np.where(rng.rand(1, 30, 1) < 0.5, -1.0, 1.0)
    r_a = M.r2_score(torch.tensor(p), torch.tensor(t), center=False)
    r_b = M.r2_score(torch.tensor(p * flip), torch.tensor(t * flip), center=False)
    ref = 1 - ((p - t) ** 2).sum() / (t ** 2).sum()
    assert abs(r_a - ref) < 1e-12 and abs(r_a - r_b) < 1e-12
    # ragged lists behave the same
    rl = M.r2_score(list(torch.tensor(p * flip).unbind(0)), list(torch.tensor(t * flip).unbind(0)), center=False)
    assert abs(rl - r_a) < 1e-12
