"""Trainer tests (CPU): smoke runs, block-diagonal batches == per-sample loop, resume, metric consistency."""
import json
import os

import numpy as np
import pytest
import torch
from scipy.spatial import Delaunay

from rhmp import metrics as M
from rhmp.complex import CochainComplex
from rhmp.data import TaskData, feature_stats, scale_stats, sequential_split
from rhmp.train import batch_loss, build_config, evaluate, parse_args, predict, run


def _mesh(n, seed):
    rng = np.random.RandomState(seed)
    pts = rng.rand(n, 2)
    return pts, Delaunay(pts).simplices


def _smooth(K, x):
    """Target: one explicit graph-Laplacian smoothing step of the node input (linear, local)."""
    L = (K.d[0].to_dense().T @ K.d[0].to_dense()).to(x)
    return x - 0.05 * torch.einsum("ij,...jf->...if", L, x)


def tiny_shared_task(N=32, n=30, seed=0):
    pts, faces = _mesh(n, seed)
    K = CochainComplex.from_triangles(pts, faces)
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(N, n, 1, generator=g)
    Y = _smooth(K, X)
    E = torch.randn(N, K.n[1], 1, generator=g)          # an odd edge input on top
    split = sequential_split(N)
    xs0, xs1, ys = feature_stats(X[split[0]]), scale_stats(E[split[0]]), feature_stats(Y[split[0]])
    return TaskData(name="tiny", K=K, inputs={0: xs0.normalize(X), 1: xs1.normalize(E)}, target=ys.normalize(Y),
                    target_degree=0, target_kind="node_scalar", in_dims={0: 1, 1: 1}, even_dims={},
                    connection_dims={}, split=split, x_stats={0: xs0, 1: xs1}, y_stats=ys, spatial_dim=2,
                    out_dim=1, meta={"v1_C": 8, "layers": 2, "batch_size": 8, "r2_std": ys.std})


def tiny_variable_task(N=15, seed=0, target="node"):
    Ks, xs, ys = [], [], []
    g = torch.Generator().manual_seed(seed)
    for i in range(N):
        pts, faces = _mesh(18 + 3 * (i % 5), seed * 100 + i)
        K = CochainComplex.from_triangles(pts, faces)
        x = torch.randn(K.n[0], 1, generator=g)
        s = torch.randn(K.n[1], 1, generator=g)             # even edge input
        y = _smooth(K, x) if target == "node" else K.apply_d(0, x.unsqueeze(1).contiguous())[:, 0]
        Ks.append(K)
        xs.append({0: x, 1: s})
        ys.append(y)
    split = sequential_split(N)
    tr = split[0].tolist()
    st = {k: feature_stats([xs[i][k] for i in tr]) for k in (0, 1)}
    yst = feature_stats([ys[i] for i in tr]) if target == "node" else scale_stats([ys[i] for i in tr])
    return TaskData(name="tinyvar", K=Ks, inputs=[{k: st[k].normalize(x[k]) for k in (0, 1)} for x in xs],
                    target=[yst.normalize(y) for y in ys], target_degree=0 if target == "node" else 1,
                    target_kind="node_scalar" if target == "node" else "cochain", in_dims={0: 1, 1: 1},
                    even_dims={1: 1}, connection_dims={}, split=split, x_stats=st, y_stats=yst, spatial_dim=2,
                    out_dim=1, meta={"v1_C": 8, "layers": 2, "batch_size": 4, "r2_std": yst.std})


def _args(tmp_path, *extra):
    return parse_args(["--task", "tiny", "--epochs", "2", "--out", str(tmp_path), "--device", "cpu", "--C", "8",
                       "--layers", "2", "--seed", "0", "--quiet", *extra])


def test_trainer_smoke_shared(tmp_path):
    torch.set_num_threads(2)
    task = tiny_shared_task()
    res = run(_args(tmp_path, "--batch", "8"), task=task)
    assert res["complete"] and res["epochs_run"] == 2
    for f in ("config.json", "history.json", "result.json", "best.pt", "last.pt"):
        assert os.path.exists(tmp_path / f), f
    hist = json.load(open(tmp_path / "history.json"))
    assert len(hist) == 2 and all(np.isfinite(h["train_loss"]) and "peak_mem_GB" in h for h in hist)
    for k in ("R2", "MSE", "MAE", "NRMSE", "SSIM", "Pearson", "R2_v1"):
        assert np.isfinite(res["test"][k]), k
    assert res["params"] > 0 and res["wall_clock_s"] > 0
    assert hist[-1]["train_loss"] < hist[0]["train_loss"] * 1.5
    # a second call with the same output directory is a no-op (already complete)
    again = run(_args(tmp_path, "--batch", "8"), task=task)
    assert again["best_epoch"] == res["best_epoch"]


def test_trainer_smoke_variable_meshes(tmp_path):
    torch.set_num_threads(2)
    for target in ("node", "cochain"):
        task = tiny_variable_task(target=target)
        out = tmp_path / target
        res = run(_args(out, "--bs-meshes", "4"), task=task)
        assert res["complete"] and np.isfinite(res["test"]["R2"])
        assert res["test"]["N"] == len(task.split[2])


def test_block_diagonal_equals_per_sample_loop():
    """Loss and gradients of a block-diagonal batch == size-weighted per-sample losses (same parameters)."""
    torch.manual_seed(0)
    task = tiny_variable_task(N=6)
    from rhmp.model import RHMP
    args = parse_args(["--task", "tinyvar", "--C", "8", "--layers", "2", "--device", "cpu"])
    model = RHMP(build_config(task, args), task.geo_dims).double()
    task.K = [K.to(dtype=torch.float64) for K in task.K]
    task.inputs = [{k: v.double() for k, v in x.items()} for x in task.inputs]
    task.target = [y.double() for y in task.target]
    idx = torch.arange(6)
    loss_b = batch_loss(model, task, idx, device="cpu")
    g_b = torch.autograd.grad(loss_b, list(model.parameters()), allow_unused=True)
    sizes = torch.tensor([float(task.target[i].numel()) for i in idx.tolist()])
    losses = [batch_loss(model, task, idx[i:i + 1], device="cpu") for i in range(6)]
    loss_l = sum(l * s for l, s in zip(losses, sizes)) / sizes.sum()
    g_l = torch.autograd.grad(loss_l, list(model.parameters()), allow_unused=True)
    assert abs(loss_b.item() - loss_l.item()) <= 1e-5 * max(1.0, abs(loss_l.item()))
    for a, b in zip(g_b, g_l):
        if a is None or b is None:
            assert a is None and b is None
            continue
        assert torch.allclose(a, b, atol=1e-8, rtol=1e-5)
    # predictions of one sample do not depend on the other meshes in the batch
    model.eval()
    with torch.no_grad():
        p_all, _ = predict(model, task, idx, 6, "cpu")
        p_one, _ = predict(model, task, idx[2:3], 1, "cpu")
    assert torch.allclose(p_all[2], p_one[0], atol=1e-10)


def test_resume_matches_uninterrupted(tmp_path):
    torch.set_num_threads(1)
    task = tiny_shared_task()
    ref = run(_args(tmp_path / "ref", "--batch", "8", "--epochs", "3"), task=task)
    part = run(_args(tmp_path / "res", "--batch", "8", "--epochs", "3", "--stop-after", "1"), task=task)
    assert part["stopped_after"] == 1 and not os.path.exists(tmp_path / "res" / "result.json")
    res = run(_args(tmp_path / "res", "--batch", "8", "--epochs", "3"), task=task)
    h_ref = json.load(open(tmp_path / "ref" / "history.json"))
    h_res = json.load(open(tmp_path / "res" / "history.json"))
    assert [h["epoch"] for h in h_res] == [1, 2, 3]
    for a, b in zip(h_ref, h_res):
        assert abs(a["train_loss"] - b["train_loss"]) <= 1e-6 * max(1.0, abs(a["train_loss"]))
        assert abs(a["val_R2"] - b["val_R2"]) <= 1e-6
    assert abs(ref["test"]["R2"] - res["test"]["R2"]) <= 1e-6
    # a resume with a different configuration is refused
    with pytest.raises(RuntimeError):
        run(_args(tmp_path / "res", "--batch", "4", "--epochs", "4", "--force"), task=task)


def test_evaluate_matches_metrics_module(tmp_path):
    """evaluate() == summarize() of predict() outputs, and R2 is invariant to the evaluation batch size."""
    task = tiny_shared_task()
    from rhmp.model import RHMP
    args = parse_args(["--task", "tiny", "--C", "8", "--layers", "2", "--device", "cpu"])
    torch.manual_seed(0)
    model = RHMP(build_config(task, args), task.geo_dims)
    te = task.split[2]
    m1 = evaluate(model, task, te, 3, "cpu")
    m2 = evaluate(model, task, te, 64, "cpu")
    p, t = predict(model, task, te, 5, "cpu")
    m3 = M.summarize(p, t, task.y_stats.as_tuple(), r2_std=task.meta["r2_std"])
    for k in ("R2", "MSE", "NRMSE", "SSIM", "Pearson"):
        assert abs(m1[k] - m2[k]) < 1e-5 and abs(m1[k] - m3[k]) < 1e-5, k


def test_output_map_pipeline(tmp_path):
    """Oriented face->node output map: exact on a pure d1 target, and the trainer runs through it."""
    from rhmp.tasks.paper import oriented_face_to_node
    torch.set_num_threads(2)
    task = tiny_shared_task()
    K = task.K
    omap = oriented_face_to_node(K, 1)
    # target: oriented node average of the face circulation of the edge input (a pseudo-scalar node field)
    E = task.x_stats[1].denormalize(task.inputs[1])                          # (N, n1, 1)
    flux = K.apply_d(1, E.permute(1, 0, 2).contiguous())                    # (n2, N, 1)
    y = omap(flux).permute(1, 0, 2)                                         # (N, n0, 1)
    # the map equals the explicit oriented incidence average
    P = K.pos.double()
    F_ = K.cells[2]
    a, b = P[F_[:, 1]] - P[F_[:, 0]], P[F_[:, 2]] - P[F_[:, 0]]
    sig = torch.sign(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    ref = torch.zeros(K.n[0], flux.shape[1], dtype=torch.float64)
    cnt = torch.zeros(K.n[0], dtype=torch.float64)
    for j in range(3):
        ref.index_add_(0, F_[:, j], sig[:, None] * flux[:, :, 0].double())
        cnt.index_add_(0, F_[:, j], torch.ones(K.n[2], dtype=torch.float64))
    assert torch.allclose(y[..., 0].t().double(), ref / cnt[:, None], atol=1e-5)
    ys = feature_stats(y[task.split[0]])
    task.target, task.y_stats, task.output_map = ys.normalize(y), ys, omap
    task.readout = omap.model_readout          # TaskData.readout is canonical (resolved at construction otherwise)
    task.meta["r2_std"] = ys.std
    args = _args(tmp_path, "--batch", "8")
    assert build_config(task, args).readout == "cochain:2"
    res = run(args, task=task)
    assert res["complete"] and np.isfinite(res["test"]["R2"])


def test_eval_ckpt_transfer(tmp_path):
    """--eval-ckpt re-normalises another task's data to the run's statistics: evaluating the run on its own data
    with a *different* normalisation reproduces its test metrics."""
    torch.set_num_threads(2)
    task = tiny_shared_task()
    res = run(_args(tmp_path / "run", "--batch", "8"), task=task)
    other = tiny_shared_task()
    # same raw data, different (arbitrary) normalisation
    from rhmp.data import Stats
    for k in other.inputs:
        s_old = other.x_stats[k]
        s_new = Stats(s_old.mean + 0.7, s_old.std * 1.9, s_old.kind)
        other.inputs[k] = s_new.normalize(s_old.denormalize(other.inputs[k]))
        other.x_stats[k] = s_new
    y_new = Stats(other.y_stats.mean - 0.3, other.y_stats.std * 0.5)
    other.target = y_new.normalize(other.y_stats.denormalize(other.target))
    other.y_stats = y_new
    ev = run(_args(tmp_path / "ev", "--eval-ckpt", str(tmp_path / "run")), task=other)
    for k in ("R2", "MSE", "NRMSE", "SSIM", "Pearson"):
        assert abs(ev["test"][k] - res["test"][k]) < 1e-5, k


def _tiny_face_task(N=12, n=28, seed=3):
    """Odd edge input E, odd face target d1 E (a T6f-like task on one planar mesh)."""
    pts, faces = _mesh(n, seed)
    K = CochainComplex.from_triangles(pts, faces)
    g = torch.Generator().manual_seed(seed)
    E = torch.randn(N, K.n[1], 1, generator=g)
    Y = K.apply_d(1, E.permute(1, 0, 2).contiguous()).permute(1, 0, 2).contiguous()
    split = sequential_split(N)
    xs, ys = scale_stats(E[split[0]]), scale_stats(Y[split[0]])
    return TaskData(name="tinyface", K=K, inputs={1: xs.normalize(E)}, target=ys.normalize(Y), target_degree=2,
                    target_kind="cochain", in_dims={1: 1}, even_dims={}, connection_dims={}, split=split,
                    x_stats={1: xs}, y_stats=ys, spatial_dim=2, out_dim=1, meta={"r2_std": ys.std})


def test_robustness_transforms_are_exact():
    """rhmp.robustness transforms (vertex relabelling, face-orientation flips, rotations/reflections) remap every
    degree correctly: an exactly equivariant model gives the same metrics on the transformed data."""
    import rhmp.robustness as E
    from rhmp.model import RHMP
    torch.manual_seed(0)
    cases = [tiny_shared_task(), _tiny_face_task(), tiny_variable_task(N=9), tiny_variable_task(N=9, target="cochain")]
    for task in cases:
        args = parse_args(["--task", task.name, "--C", "8", "--layers", "2", "--device", "cpu"])
        model = RHMP(build_config(task, args), task.geo_dims).double().eval()
        with torch.no_grad():                            # move away from the (degenerate) initialisation
            for p in model.parameters():
                p.add_(0.05 * torch.randn_like(p))
        if task.variable_mesh:
            task.K = [K.to(dtype=torch.float64) for K in task.K]
            task.inputs = [{k: v.double() for k, v in x.items()} for x in task.inputs]
            task.target = [y.double() for y in task.target]
        else:
            task.K = task.K.to(dtype=torch.float64)
            task.inputs = {k: v.double() for k, v in task.inputs.items()}
            task.target = task.target.double()
        te = torch.arange(task.num_samples)
        base = evaluate(model, task, te, 4, "cpu")
        views = {
            "relabel": E.transformed_task(task, te, perm_seed=1, seed=1),
            "flip": E.transformed_task(task, te, flip_frac=0.5, seed=2),
            "rotate": E.transformed_task(task, te, Q=E.random_rotation(2, 3), shift=np.array([0.3, -0.2]), seed=3),
            "reflect": E.transformed_task(task, te, Q=E.random_rotation(2, 4, proper=False), shift=np.zeros(2)),
        }
        for name, view in views.items():
            if view.variable_mesh:
                view.K = [K.to(dtype=torch.float64) for K in view.K]
            else:
                view.K = view.K.to(dtype=torch.float64)
            m = evaluate(model, view, view.split[2], 4, "cpu")
            # geometry is cached in float32 by the complex builder, so a rebuilt (relabelled / rotated) mesh differs
            # at ~1e-7; a wrong permutation or orientation sign would change the metrics at O(1)
            for k in ("R2", "MSE"):
                assert abs(m[k] - base[k]) < 1e-6 * max(1.0, abs(base[k])), (task.name, name, k, m[k], base[k])


def test_from_arrays(tmp_path):
    """TaskData.from_arrays: per-column odd/even statistics, readout -> target kind, trainer runs on it."""
    torch.set_num_threads(2)
    pts, faces = _mesh(30, 0)
    K = CochainComplex.from_triangles(pts, faces)
    g = torch.Generator().manual_seed(0)
    N = 20
    X0 = torch.randn(N, K.n[0], 1, generator=g) * 3 + 2
    E = torch.randn(N, K.n[1], 2, generator=g) + 1.0                   # column 0 odd, column 1 even
    Y = K.apply_d(0, X0.permute(1, 0, 2).contiguous()).permute(1, 0, 2).contiguous()   # odd edge target d0 x
    td = TaskData.from_arrays(K, {0: X0, 1: E}, Y, readout="cochain:1", even_dims={1: 1})
    tr = td.split[0]
    assert [len(s) for s in td.split] == [14, 3, 3]
    assert td.readout == "cochain:1" and td.target_kind == "cochain" and td.target_degree == 1
    assert float(td.x_stats[1].mean[0]) == 0.0                                    # odd column: scale only
    assert abs(float(td.x_stats[1].mean[1]) - float(E[tr][..., 1].mean())) < 1e-5  # even column: mean/std
    assert float(td.y_stats.mean.abs().max()) == 0.0 and td.meta["batch_size"] == 64
    assert abs(float(td.inputs[0][tr].mean())) < 1e-5
    res = run(_args(tmp_path / "a", "--batch", "8"), task=td)
    assert res["complete"] and res["r2_definition"].startswith("uncentred")
    td2 = TaskData.from_arrays(K, {0: X0}, -Y, readout="grad")                   # exact curl-free readout
    assert td2.readout == "grad" and td2.target_kind == "cochain" and td2.target_degree == 1
    assert build_config(td2, parse_args(["--task", "x", "--C", "8", "--layers", "1"])).readout == "grad"
    # variable meshes + node target
    var = tiny_variable_task(N=9)
    raw_x = [{k: var.x_stats[k].denormalize(v) for k, v in d.items()} for d in var.inputs]
    raw_y = [var.y_stats.denormalize(y) for y in var.target]
    td3 = TaskData.from_arrays(var.K, raw_x, raw_y, readout="node_scalar", even_dims={1: 1}, split=var.split)
    assert td3.variable_mesh and td3.readout == "node_scalar" and td3.meta["batch_size"] == 8
    for a, b in zip(td3.target, var.target):
        assert torch.allclose(a, b, atol=1e-5)
    res3 = run(_args(tmp_path / "b", "--bs-meshes", "4"), task=td3)
    assert res3["complete"]


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_pack_to_preserves_complexes_and_nesting(device):
    """rhmp.data.pack_to moves nested structures (complexes with CSR operators, dicts, lists) with packed copies."""
    from rhmp.data import pack_to
    Ks = [CochainComplex.from_triangles(*_mesh(15 + 5 * i, i)) for i in range(3)]
    xs = [{0: torch.randn(K.n[0], 2), 1: torch.randn(K.n[1], 1)} for K in Ks]
    obj = (Ks, xs, [torch.arange(4)], {"a": 1.5, "b": [torch.ones(2, dtype=torch.bool)]})
    Kd, xd, other, extra = pack_to(obj, device)
    for K, K2, x, x2 in zip(Ks, Kd, xs, xd):
        assert K2.pos.device.type == device and K2.d[0].device.type == device and K2.n == K.n
        v = torch.randn(K.n[0], 3, 2)
        assert torch.allclose(K2.apply_d(0, v.to(device)).cpu(), K.apply_d(0, v))
        assert float(K2.apply_d(1, K2.apply_d(0, v.to(device).contiguous())).abs().max()) < 1e-5    # d1 d0 = 0
        for k in range(K.dim + 1):
            assert torch.equal(K2.geo[k].cpu(), K.geo[k]) and torch.equal(K2.boundary[k].cpu(), K.boundary[k])
        assert all(torch.equal(x2[k].cpu(), x[k]) for k in x)
    assert torch.equal(other[0].cpu(), torch.arange(4)) and extra["a"] == 1.5 and extra["b"][0].dtype == torch.bool
    Kb = CochainComplex.batch(Kd)                      # packed complexes batch like ordinary ones
    assert Kb.n[0] == sum(K.n[0] for K in Ks)


def test_add_abs_scale(tmp_path):
    """abs_scale: constant even vertex column log(median edge length), standardised on the train split."""
    from rhmp.data import add_abs_scale
    torch.set_num_threads(2)
    task = tiny_variable_task(N=9)
    # rescale two meshes so the absolute scale differs between samples
    for i in (0, 4):
        K = task.K[i]
        task.K[i] = CochainComplex.from_triangles(K.pos.double() * 3.0, K.cells[2])
    add_abs_scale(task)
    assert task.in_dims[0] == 2 and task.even_dims[0] == 1 and task.x_stats[0].mean.numel() == 2
    col = [d[0][:, -1] for d in task.inputs]
    assert all(float(c.max() - c.min()) < 1e-6 for c in col)              # constant per mesh
    assert float(col[0][0]) > float(col[1][0]) + 0.5                       # the 3x larger mesh stands out
    tr = task.split[0].tolist()
    assert abs(float(torch.stack([col[i][0] for i in tr]).mean())) < 1e-5
    assert run(_args(tmp_path, "--bs-meshes", "4"), task=task)["complete"]
    shared = add_abs_scale(tiny_shared_task())                              # one mesh: an all-zero column
    assert shared.inputs[0].shape[-1] == 2 and float(shared.inputs[0][..., -1].abs().max()) == 0.0


def test_pack_to_large_buffer_without_pinning(monkeypatch):
    """Buffers above PIN_MAX_BYTES are copied without page-locking (pinning multi-GB buffers fails on CUDA)."""
    import rhmp.data as D
    monkeypatch.setattr(D, "PIN_MAX_BYTES", 1024)            # force the unpinned path for a 4 MB buffer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    big = torch.arange(1_000_000, dtype=torch.float32)
    small = torch.arange(10, dtype=torch.int64)
    out = D.pack_to({"big": big, "small": [small]}, dev)
    assert out["big"].device.type == dev and torch.equal(out["big"].cpu(), big)
    assert torch.equal(out["small"][0].cpu(), small)


def test_latent_flag_train_resume_eval(tmp_path):
    """--latent K:R -> RHMPConfig.latent_dims; latents created before the optimizer, resumed and re-loaded."""
    torch.set_num_threads(2)
    task = tiny_shared_task()
    part = run(_args(tmp_path / "a", "--batch", "8", "--latent", "1:3", "--epochs", "3", "--stop-after", "1"),
               task=task)
    assert part["stopped_after"] == 1
    res = run(_args(tmp_path / "a", "--batch", "8", "--latent", "1:3", "--epochs", "3"), task=task)
    assert res["complete"] and {int(k): v for k, v in res["config"]["latent_dims"].items()} == {1: 3}
    ck = torch.load(tmp_path / "a" / "best.pt", map_location="cpu", weights_only=False)
    lat = [k for k in ck["state_dict"] if k.startswith("latent.")]
    assert lat and ck["state_dict"][lat[0]].shape == (task.K.n[1], 3)
    ev = run(_args(tmp_path / "ev", "--eval-ckpt", str(tmp_path / "a")), task=tiny_shared_task())
    assert abs(ev["test"]["R2"] - res["test"]["R2"]) < 1e-5
    with pytest.raises(ValueError, match="variable-mesh"):
        run(_args(tmp_path / "v", "--bs-meshes", "4", "--latent", "1:2"), task=tiny_variable_task(N=6))


def test_t8_inflow_form_is_uniform_field_line_integral():
    """T8v edge input: theta_e = V_inf . (x_dst - x_src), V_inf = U (cos a, sin a); exact for the midpoint rule."""
    from rhmp.tasks import task_defaults
    from rhmp.tasks.paper import _t8_inflow_forms
    pts, faces = _mesh(25, 2)
    K = CochainComplex.from_triangles(pts, faces)
    U, a = 41.5, 0.13
    x = torch.stack([torch.rand(K.n[0]), torch.full((K.n[0],), U), torch.full((K.n[0],), a)], 1)
    (theta,) = _t8_inflow_forms([K], [x])
    e = K.cells[1].long()
    V = np.array([U * np.cos(a), U * np.sin(a)])
    P = np.asarray(pts)
    ref = ((P[e[:, 1]] - P[e[:, 0]]) @ V)[:, None]
    assert theta.shape == (K.n[1], 1) and np.allclose(theta.numpy(), ref, atol=1e-4 * U)
    # a closed loop (face boundary) integrates to zero: d1 theta = 0 for a uniform field
    assert float(K.apply_d(1, theta.unsqueeze(1).contiguous()).abs().max()) < 1e-3
    assert task_defaults("T8v")["batch"] == 8


def test_direct_vector_map_and_t5g_structure(tmp_path):
    """T5g output map: v_i = sum_e w_e t_e / deg_i (generator loop), grad readout + map; exact under relabelling,
    face flips, rotations and reflections; trains."""
    from rhmp.tasks.paper import direct_vector_map
    import rhmp.robustness as R
    torch.set_num_threads(2)
    pts, faces = _mesh(28, 5)
    K = CochainComplex.from_triangles(pts, faces)
    omap = direct_vector_map(K)
    g = torch.Generator().manual_seed(0)
    w = torch.randn(K.n[1], 3, 1, generator=g)
    v = omap(w)                                                            # (n0, 3, 2)
    P, e = np.asarray(pts), K.cells[1].long().numpy()
    ref = np.zeros((K.n[0], 3, 2))
    cnt = np.zeros(K.n[0])
    for ei, (a, b) in enumerate(e):                                        # the T5 generator's loop
        t = (P[b] - P[a]) / np.linalg.norm(P[b] - P[a])
        for vtx in (a, b):
            ref[vtx] += w[ei, :, 0].numpy()[:, None] * t[None]
            cnt[vtx] += 1
    ref /= np.maximum(cnt, 1)[:, None, None]
    assert np.allclose(v.numpy(), ref, atol=1e-5)
    # a T5g-like task: target = map(-d0 phi) of a smoothed node input
    N = 16
    X = torch.randn(N, K.n[0], 1, generator=g)
    phi = _smooth(K, X)
    Y = omap(-K.apply_d(0, phi.permute(1, 0, 2).contiguous())).permute(1, 0, 2).contiguous()
    task = TaskData.from_arrays(K, {0: X}, Y, readout="node_vector", output_map=omap)
    task.readout = "grad"
    assert build_config(task, parse_args(["--task", "x", "--C", "8", "--layers", "2"])).readout == "grad"
    from rhmp.model import RHMP
    model = RHMP(build_config(task, parse_args(["--task", "x", "--C", "8", "--layers", "2"])), task.geo_dims)
    model = model.double().eval()
    with torch.no_grad():
        for p_ in model.parameters():
            p_.add_(0.05 * torch.randn_like(p_))
    task.K = task.K.to(dtype=torch.float64)
    task.inputs = {k: v_.double() for k, v_ in task.inputs.items()}
    task.target = task.target.double()
    task.output_map = direct_vector_map(task.K)
    te = torch.arange(N)
    base = evaluate(model, task, te, 4, "cpu")
    for name, view in {"relabel": R.transformed_task(task, te, perm_seed=1, seed=1),
                       "flip": R.transformed_task(task, te, flip_frac=0.5, seed=2),
                       "rotate": R.transformed_task(task, te, Q=R.random_rotation(2, 3), shift=np.array([0.2, 0.1])),
                       "reflect": R.transformed_task(task, te, Q=R.random_rotation(2, 4, proper=False),
                                                     shift=np.zeros(2))}.items():
        view.K = view.K.to(dtype=torch.float64)
        view.output_map = direct_vector_map(view.K)
        m = evaluate(model, view, view.split[2], 4, "cpu")
        assert abs(m["R2"] - base["R2"]) < 1e-6 * max(1.0, abs(base["R2"])), (name, m["R2"], base["R2"])
    fresh = TaskData.from_arrays(K, {0: X}, Y, readout="node_vector", output_map=direct_vector_map(K))
    fresh.readout = "grad"
    assert run(_args(tmp_path, "--batch", "8"), task=fresh)["complete"]
