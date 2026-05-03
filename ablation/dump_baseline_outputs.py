"""
Dump per-task test predictions for the 'main' ours + a curated set of baselines
(uses checkpoints saved by experiments/formal_benchmark.py).

Saves to ablation/outputs/{task}/{name}_test.pkl
 -> same format as run_ablation.py (dict with 'pred' and 'target', unnormalized).
"""
import os, sys, pickle
import numpy as np
import torch

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'experiments'))

from gauge_hodge_mp.cell_complex import CellComplex
from gauge_hodge_mp.network import GaugeHodgeNetwork
from baselines_graph import GCNBaseline, GATBaseline, SchNetBaseline, EGNNBaseline
from baselines_topo import MPSNBaseline, SCCNNBaseline
from baselines_advanced import GaugeEquivCNNBaseline, CWNetBaseline, CliffordNetBaseline, HermesBaseline
from baselines_operator import FNOWrapper
from formal_benchmark import TASKS, find_hidden_match


# Representative baselines to include in the output comparison figures.
# Keep small to avoid figure overcrowding.
BASELINE_POOL = {
    "T1": ["cw_net", "gauge_cnn", "fno", "egnn"],
    "T3": ["cw_net", "gauge_cnn", "gem_cnn", "clifford_smpn"],
    "T6": ["cw_net", "gauge_cnn", "egnn", "clifford_smpn"],
    "T7": ["cw_net", "gauge_cnn", "schnet", "mpsn"],
}


def _rebuild(cls_fn, fi, h, od):
    try:
        return cls_fn(fi, h, od)
    except TypeError:
        return cls_fn(fi, h)


def build_baseline(name: str, f_in: int, hidden: int, out_dim: int, device="cuda"):
    """Return baseline module + is_edge flag."""
    if name == "gcn":
        return GCNBaseline(f_in=f_in, hidden=hidden, n_layers=4, out_dim=out_dim, task='node').to(device), False
    if name == "gat":
        return GATBaseline(f_in=f_in, hidden=hidden, n_layers=4, out_dim=out_dim, task='node').to(device), False
    if name == "schnet":
        return SchNetBaseline(f_in=f_in, hidden=hidden, n_layers=4, out_dim=out_dim, task='node').to(device), False
    if name == "egnn":
        return EGNNBaseline(f_in=f_in, hidden=hidden, n_layers=4, out_dim=out_dim, task='node').to(device), False
    if name == "cw_net":
        return CWNetBaseline(f_in=f_in, hidden=hidden, n_layers=4, out_dim=out_dim, task='node').to(device), False
    if name == "gauge_cnn":
        return GaugeEquivCNNBaseline(f_in=f_in, hidden=hidden, n_layers=4, out_dim=out_dim, task='node').to(device), False
    if name == "gem_cnn":
        return HermesBaseline(f_in=f_in, hidden=hidden, n_layers=4, out_dim=out_dim, task='node').to(device), False
    if name == "fno":
        return FNOWrapper(f_in=f_in, out_dim=out_dim, width=hidden, modes=12, n_layers=4).to(device), False
    if name == "mpsn":
        return MPSNBaseline(f_in=f_in, hidden=hidden, n_layers=4).to(device), True
    if name == "sccnn":
        return SCCNNBaseline(f_in=f_in, hidden=hidden, n_layers=4).to(device), True
    if name == "clifford_smpn":
        return CliffordNetBaseline(f_in=f_in, hidden=hidden, n_layers=4).to(device), True
    raise ValueError(name)


def _get_bl_hidden(name, ours_np_M, f_in, out_dim):
    """Recover the hidden size used during training (param-matched within 20% of ours)."""
    cls_fn_map = {
        "gcn": lambda fi, h, od: GCNBaseline(f_in=fi, hidden=h, n_layers=4, out_dim=od, task='node'),
        "gat": lambda fi, h, od: GATBaseline(f_in=fi, hidden=h, n_layers=4, out_dim=od, task='node'),
        "schnet": lambda fi, h, od: SchNetBaseline(f_in=fi, hidden=h, n_layers=4, out_dim=od, task='node'),
        "egnn": lambda fi, h, od: EGNNBaseline(f_in=fi, hidden=h, n_layers=4, out_dim=od, task='node'),
        "cw_net": lambda fi, h, od: CWNetBaseline(f_in=fi, hidden=h, n_layers=4, out_dim=od, task='node'),
        "gauge_cnn": lambda fi, h, od: GaugeEquivCNNBaseline(f_in=fi, hidden=h, n_layers=4, out_dim=od, task='node'),
        "gem_cnn": lambda fi, h, od: HermesBaseline(f_in=fi, hidden=h, n_layers=4, out_dim=od, task='node'),
        "fno": lambda fi, h, od: FNOWrapper(f_in=fi, out_dim=od, width=h, modes=12, n_layers=4),
        "mpsn": lambda fi, h, od: MPSNBaseline(f_in=fi, hidden=h, n_layers=4),
        "sccnn": lambda fi, h, od: SCCNNBaseline(f_in=fi, hidden=h, n_layers=4),
        "clifford_smpn": lambda fi, h, od: CliffordNetBaseline(f_in=fi, hidden=h, n_layers=4),
    }
    cls_fn = cls_fn_map[name]
    h, _ = find_hidden_match(cls_fn, ours_np_M, f_in, out_dim, tol=0.20)
    return h


def dump_task(task_id: str, device: str = "cuda"):
    cfg = TASKS[task_id]
    out_dir = os.path.join(ROOT, "ablation", "outputs", task_id)
    os.makedirs(out_dir, exist_ok=True)
    ck_root = os.path.join(ROOT, "checkpoints", f"{task_id}_{cfg['name']}")

    # --- Load split ---
    with open(os.path.join(ROOT, cfg['dataset']), "rb") as f:
        d = pickle.load(f)
    pts = torch.tensor(np.array(d['points'], dtype=np.float32))
    faces = torch.tensor(np.array(d['faces'], dtype=np.int64))
    K = CellComplex.from_triangulation(pts, faces).to(device)

    X = torch.tensor(np.array(d['X_data'], dtype=np.float32))
    Y = torch.tensor(np.array(d['Y_data'], dtype=np.float32))
    if X.ndim == 2: X = X.unsqueeze(-1)
    if Y.ndim == 2: Y = Y.unsqueeze(-1)
    n = len(X); nt = int(0.7 * n); nv = int(0.15 * n)
    xm, xs = X[:nt].mean((0, 1)), X[:nt].std((0, 1)).clamp(1e-6)
    ym, ys = Y[:nt].mean((0, 1)), Y[:nt].std((0, 1)).clamp(1e-6)
    Xn = (X - xm) / xs
    Yn = (Y - ym) / ys
    X_test = Xn[nt + nv:].to(device); Y_test_raw = Y[nt + nv:].numpy()

    f_in = X.shape[-1]; out_dim = Y.shape[-1]

    # --- Ours (main / 100ep) ---
    ours = GaugeHodgeNetwork(
        f_in=f_in, C=cfg['ours_C'], n_layers=4,
        n0=K.n0, n1=K.n1, n2=K.n2,
        task=cfg['task_type'], spatial_dim=cfg['spatial_dim'],
        out_dim=out_dim, mp_hidden=16,
        metric_type=cfg['metric_type'], metric_rank=cfg['metric_rank'],
    ).to(device)
    sd = torch.load(os.path.join(ck_root, "ours", "best_model.pt"),
                    map_location=device, weights_only=False)
    ours.load_state_dict(sd)
    ours.eval()
    with torch.no_grad():
        p = ours.forward_batch(X_test, K).cpu().numpy() * ys.numpy() + ym.numpy()
    pickle.dump({"pred": p, "target": Y_test_raw, "variant": "ours_main"},
                open(os.path.join(out_dir, "ours_main_test.pkl"), "wb"))
    ours_np_M = sum(pp.numel() for pp in ours.parameters()) / 1e6
    print(f"[{task_id}] ours_main  saved  (params={ours_np_M:.3f}M)")
    del ours; torch.cuda.empty_cache()

    # --- Baselines ---
    for bname in BASELINE_POOL[task_id]:
        bl_ckpt = os.path.join(ck_root, bname, "best_model.pt")
        if not os.path.exists(bl_ckpt):
            print(f"[{task_id}] {bname}: skip (no checkpoint)")
            continue
        # EGNN uses scalar-only input
        if bname == "egnn":
            ec = cfg.get("edge_channels", f_in)
            fi_bl = ec
            X_bl = torch.cat([X_test[:, :, i*3:i*3+1] for i in range(ec)], dim=-1) \
                   if f_in == ec * 3 else X_test[:, :, :ec]
        else:
            fi_bl = f_in
            X_bl = X_test
        h = _get_bl_hidden(bname, ours_np_M, fi_bl, out_dim)
        m, is_edge = build_baseline(bname, fi_bl, h, out_dim, device=device)
        try:
            m.load_state_dict(torch.load(bl_ckpt, map_location=device, weights_only=False))
        except RuntimeError as e:
            print(f"[{task_id}] {bname}: load failed ({str(e)[:80]}) -- skip")
            del m; torch.cuda.empty_cache(); continue
        m.eval()
        try:
            with torch.no_grad():
                if hasattr(m, 'forward_batch'):
                    p = m.forward_batch(X_bl, K)
                else:
                    p = torch.stack([m(X_bl[i], K) for i in range(X_bl.size(0))])
            # Edge baselines predict on edges; map back to node values via endpoint avg
            if is_edge:
                # We mapped Y to edge by averaging endpoints during training; inverse is
                # ill-posed in general. For viz, report node-space reconstruction by
                # scattering edge prediction back to endpoints.
                src, dst = K.edges[:, 0], K.edges[:, 1]
                nodes = torch.zeros(X_bl.size(0), K.n0, p.size(-1), device=device)
                cnt = torch.zeros(K.n0, device=device)
                nodes.index_add_(1, src, p); nodes.index_add_(1, dst, p)
                cnt.index_add_(0, src, torch.ones_like(src, dtype=torch.float))
                cnt.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
                p = nodes / cnt.clamp(min=1)[None, :, None]
                if p.size(-1) < out_dim:
                    p = p.expand(-1, -1, out_dim)
                elif p.size(-1) > out_dim:
                    p = p[..., :out_dim]
            p = p.cpu().numpy() * ys.numpy() + ym.numpy()
            pickle.dump({"pred": p, "target": Y_test_raw, "variant": bname},
                        open(os.path.join(out_dir, f"{bname}_test.pkl"), "wb"))
            print(f"[{task_id}] {bname}  saved")
        except Exception as e:
            print(f"[{task_id}] {bname}: FAILED ({str(e)[:100]})")
        del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=["T1", "T3", "T6"])
    args = ap.parse_args()
    for t in args.tasks:
        dump_task(t)
