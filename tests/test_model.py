"""Model-level tests: config/checkpoint round trips, ablation flags, fused vs reference kernels, mesh types,
input validation, diagnostics, AMP and ``torch.compile`` smoke tests.

Also hosts the small helpers shared by ``test_equivariance.py`` and ``test_numerics.py``.
"""
from __future__ import annotations

import dataclasses
import json
import math

import pytest
import torch

from conftest import DEVICES, MESH_NAMES, make_complex  # noqa: F401  (DEVICES used via the device fixture)
from rhmp import RHMP, RHMPConfig

F64 = torch.float64


# ----------------------------------------------------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------------------------------------------------
def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative Frobenius error ``|a - b| / |b|``."""
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def randomize_(model: torch.nn.Module, scale: float = 0.5, seed: int = 0) -> torch.nn.Module:
    """Replace every all-zero parameter (zero-initialised metric/gate heads, higher polynomial coefficients) by
    random values so that metrics, lifting gates and polynomial filters are non-trivial in symmetry tests."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.numel() and bool((p == 0).all()):
                p.copy_((torch.randn(p.shape, generator=g, dtype=F64) * scale).to(p))
    return model


def make_inputs(cfg: RHMPConfig, K, B: int = 2, seed: int = 0, dtype: torch.dtype = torch.float32,
                even_range: tuple[float, float] | None = None) -> dict[int, torch.Tensor]:
    """Random inputs ``{k: (n_k, B, F_k)}`` matching ``cfg`` on the device of ``K``.

    Even columns are positive (log-uniform in ``10**even_range`` if given, else ``|N(0,1)| + 0.1``).
    """
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, F in sorted(cfg.in_dims.items()):
        if F == 0:
            continue
        x = torch.randn(K.n[k], B, F, generator=g, dtype=F64)
        E = cfg.even_dims.get(k, 0)
        if E:
            if even_range is None:
                x[..., F - E:] = x[..., F - E:].abs() + 0.1
            else:
                lo, hi = even_range
                x[..., F - E:] = 10.0 ** (lo + (hi - lo) * torch.rand(K.n[k], B, E, generator=g, dtype=F64))
        out[k] = x.to(device=K.pos.device, dtype=dtype)
    return out


def small_cfg(dim: int = 2, **kw) -> RHMPConfig:
    """Small test configuration with node, edge and face (or tet) inputs, including even columns."""
    base = dict(in_dims={0: 3, 1: 2, dim: 2}, even_dims={0: 1, dim: 1}, C=8, n_layers=2, poly_order=2)
    base.update(kw)
    return RHMPConfig(**base)


def build_model(cfg: RHMPConfig, K, seed: int = 0, randomize: bool = True, dtype: torch.dtype | None = None) -> RHMP:
    """Seeded model on the device (and dtype) of ``K``."""
    torch.manual_seed(seed)
    m = RHMP(cfg, K.geo_dims)
    if randomize:
        randomize_(m, seed=seed)
    return m.to(device=K.pos.device, dtype=dtype or K.star[0].dtype)


def loss_of(y: torch.Tensor, seed: int = 1) -> torch.Tensor:
    """A generic scalar loss with random weights (exercises every output entry)."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(y.shape, generator=g, dtype=F64).to(y)
    return (y * w).sum()


def all_finite(model: torch.nn.Module, *tensors: torch.Tensor) -> bool:
    ok = all(bool(torch.isfinite(t).all()) for t in tensors)
    return ok and all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)


# ----------------------------------------------------------------------------------------------------------------
# config / checkpoint
# ----------------------------------------------------------------------------------------------------------------
def test_config_roundtrip():
    cfg = RHMPConfig(in_dims={0: 3, 1: 2}, even_dims={1: 1}, connection_dims={1: 1}, readout="cochain:1",
                     out_dim=2, C=16, n_layers=3, poly_order=3, scaling="jacobi", tie_metrics=False, amp=True)
    d = cfg.to_dict()
    assert RHMPConfig.from_dict(d) == cfg
    assert RHMPConfig.from_dict(json.loads(json.dumps(d))) == cfg      # string keys after JSON
    with pytest.raises(ValueError):
        RHMPConfig.from_dict({**d, "bogus": 1})
    for bad in (dict(scaling="foo"), dict(gate="tanh"), dict(readout="cochain:x"), dict(vector_mode="x"),
                dict(even_dims={1: 2}, connection_dims={1: 1}), dict(poly_order=0), dict(log_range=0.0)):
        with pytest.raises(ValueError):
            RHMPConfig(**{**dict(in_dims={0: 3, 1: 2}), **bad})


@pytest.mark.parametrize("readout", ["node_vector", "grad", "curl", "div", "div:1", "cochain:1", "resolvent",
                                     "tensor"])
def test_checkpoint_roundtrip(tmp_path, readout, device):
    K = make_complex("grid", device)
    if readout == "resolvent":
        cfg = small_cfg(layers=["poly", "resolvent"], resolvent_iters=12, out_dim=2)
    elif readout == "tensor":
        cfg = small_cfg(metric_type="tensor", out_dim=2)
    else:
        cfg = small_cfg(readout=readout, out_dim=2)
    m = build_model(cfg, K)
    inp = make_inputs(cfg, K)
    y = m(inp, K)
    path = tmp_path / "rhmp.pt"
    torch.save(m.to_checkpoint(), path)
    ck = torch.load(path, map_location=device, weights_only=True)
    assert set(ck) == {"cfg", "geo_dims", "state_dict"}
    assert ck["geo_dims"] == K.geo_dims
    m2 = RHMP.from_checkpoint(path, map_location=device)
    assert m2.cfg == cfg
    assert rel(m2(inp, K), y) < 1e-6
    m3 = RHMP.from_checkpoint(m.to_checkpoint())
    assert rel(m3.to(device)(inp, K), y) < 1e-6


# ----------------------------------------------------------------------------------------------------------------
# ablations, meshes, parameter counts
# ----------------------------------------------------------------------------------------------------------------
ABLATIONS = {
    "default": {}, "relu_gate": dict(gate="relu"), "no_cross": dict(cross=False),
    "identity_metric": dict(identity_metric=True), "scaling_none": dict(scaling="none"),
    "scaling_jacobi": dict(scaling="jacobi"), "poly1": dict(poly_order=1), "poly3": dict(poly_order=3),
    "untied": dict(tie_metrics=False), "checkpoint": dict(checkpoint_layers=True), "reference": dict(fused=False),
    "vector_ls": dict(readout="node_vector", out_dim=2), "vector_direct": dict(readout="node_vector",
                                                                               vector_mode="direct"),
    "cochain1": dict(readout="cochain:1", out_dim=3), "cochain2": dict(readout="cochain:2"),
    "even0": dict(readout="even:0"), "even1": dict(readout="even:1"), "even2": dict(readout="even:2"),
    "zero_layers": dict(n_layers=0), "connection": dict(in_dims={0: 3, 1: 3, 2: 2}, connection_dims={1: 1}),
    "connection_odd": dict(in_dims={0: 3, 1: 3, 2: 2}, connection_dims={1: 1}, connection_odd=True),
    "no_inputs_on_0": dict(in_dims={1: 2}, even_dims={}), "amp": dict(amp=True),
    "grad": dict(readout="grad"), "curl": dict(readout="curl", out_dim=2), "div": dict(readout="div"),
    "div1": dict(readout="div:1"), "resolvent": dict(layers=["poly", "resolvent"]),
    "resolvent_unrolled": dict(layers=["resolvent"], resolvent_grad="unrolled"),
    "tensor": dict(metric_type="tensor"), "tensor_none": dict(metric_type="tensor", scaling="none"),
    "tensor_untied_p3": dict(metric_type="tensor", tie_metrics=False, poly_order=3),
    "tensor_resolvent": dict(metric_type="tensor", layers=["poly", "resolvent", "poly"]),
}


@pytest.mark.parametrize("name", list(ABLATIONS))
def test_ablation_flags_run(name, device):
    if name == "amp" and device != "cuda":
        pytest.skip("autocast is only enabled on CUDA")
    K = make_complex("delaunay", device)
    cfg = small_cfg(**ABLATIONS[name])
    m = build_model(cfg, K)
    y = m(make_inputs(cfg, K, B=3), K)
    loss_of(y).backward()
    assert y.dtype == torch.float32 and y.shape[1] == 3
    assert all_finite(m, y)


def test_unit_star_complex_runs(device):
    K = make_complex("delaunay", device, star="unit")
    cfg = small_cfg()
    m = build_model(cfg, K)
    y = m(make_inputs(cfg, K), K)
    loss_of(y).backward()
    assert all_finite(m, y)


@pytest.mark.parametrize("name", MESH_NAMES)
@pytest.mark.parametrize("readout", ["node_scalar", "node_vector", "cochain:top", "even:1", "grad", "curl", "div"])
def test_all_mesh_types(name, readout, device):
    K = make_complex(name, device)
    top = K.dim
    readout = readout.replace("top", str(top))
    cfg = small_cfg(dim=top, readout=readout, out_dim=2)
    m = build_model(cfg, K)
    y = m(make_inputs(cfg, K, B=2), K)
    loss_of(y).backward()
    width = 2 * K.pos.shape[1] if readout == "node_vector" else 2
    assert y.shape == (K.n[m.output_degree], 2, width)
    assert all_finite(m, y)


def _constraint_residuals(readout: str, y: torch.Tensor, K) -> list[float]:
    """Relative residuals of the exact constraints of the grad / curl / div / mdiv readouts."""
    if readout == "grad":
        return [float(K.apply_d(1, y.contiguous()).norm()) / float(y.norm())] if K.dim >= 2 else []
    if readout == "curl":
        return [float(K.apply_d(2, y.contiguous()).norm()) / float(y.norm())] if K.dim >= 3 else []
    k = K.dim if readout in ("div", "mdiv") else int(readout.split(":")[1])
    if readout.startswith("mdiv"):              # mdiv:1 -> sum_i star0_i y_i = 0; mdiv:k -> S y = d^T b co-closed
        from rhmp.layers import make_context
        S = torch.exp(make_context(K, scaling="dec", B=y.shape[1], dtype=y.dtype).log_star[k - 1]).view(-1, 1, 1)
        y = y * (K.star[0].to(y.dtype).view(-1, 1, 1) if k == 1 else S)
    if k >= 2:
        return [float(K.apply_dT(k - 2, y.contiguous()).norm()) / float(y.norm())]
    if K.batch is None:                                                     # div:1 -> vertex sums vanish
        return [float(y.sum(0).abs().max()) / float(y.abs().sum(0).max())]
    from rhmp import ops
    sums = ops.segment_sum(y, K.batch[0], K.num_graphs)
    return [float(sums.abs().max()) / float(ops.segment_sum(y.abs(), K.batch[0], K.num_graphs).max())]


@pytest.mark.parametrize("name", ["grid", "sphere", "tets", "mixed", "batch"])
@pytest.mark.parametrize("readout", ["grad", "curl", "div", "div:1", "div:2", "mdiv:1", "mdiv:2"])
def test_constraint_readouts(name, readout, device):
    """grad: d_1 E = 0;  curl: d_2 F = 0 (volumes);  div:k: d_{k-2}^T y = 0, and zero vertex sums for k = 1."""
    from conftest import make_batch
    K = make_batch(device)[0] if name == "batch" else make_complex(name, device)
    cfg = small_cfg(dim=K.dim, readout=readout, out_dim=3)
    m = build_model(cfg, K)
    y = m(make_inputs(cfg, K, B=1 if name == "batch" else 3), K)
    assert y.shape[0] == K.n[m.output_degree]
    for r in _constraint_residuals(readout, y.detach(), K):
        assert r < 1e-6, (readout, r)


def test_parameter_count_independent_of_mesh_size():
    Ks = [make_complex("delaunay", n=n) for n in (30, 400)] + [make_complex("sphere", subdiv=s) for s in (1, 3)]
    cfg = small_cfg(readout="node_vector")
    models = [RHMP(cfg, K.geo_dims) for K in Ks]
    shapes = [{k: tuple(v.shape) for k, v in m.state_dict().items()} for m in models]
    assert all(s == shapes[0] for s in shapes)
    m = models[0]
    for K in Ks:  # one model runs on every mesh
        y = m(make_inputs(cfg, K), K)
        assert y.shape == (K.n[0], 2, K.pos.shape[1])


# ----------------------------------------------------------------------------------------------------------------
# kernels and execution modes
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("poly_order", [1, 2, 3])
@pytest.mark.parametrize("scaling", ["dec", "none"])
@pytest.mark.parametrize("cross", [True, False])
def test_fused_matches_reference(poly_order, scaling, cross, device):
    K = make_complex("sphere", device, subdiv=1).to(dtype=F64)
    cfg = small_cfg(poly_order=poly_order, scaling=scaling, cross=cross, readout="cochain:1")
    m1 = build_model(cfg, K)
    m2 = build_model(dataclasses.replace(cfg, fused=False), K, randomize=False)
    m2.load_state_dict(m1.state_dict())
    inp = make_inputs(cfg, K, dtype=F64)
    y1, y2 = m1(inp, K), m2(inp, K)
    assert rel(y1, y2) < 1e-12
    loss_of(y1).backward()
    loss_of(y2).backward()
    for (n, p1), p2 in zip(m1.named_parameters(), m2.parameters()):
        assert (p1.grad is None) == (p2.grad is None), n
        if p1.grad is not None:
            assert rel(p1.grad, p2.grad) < 1e-10, n


def test_checkpoint_layers_same_gradients(device):
    K = make_complex("delaunay", device).to(dtype=F64)
    cfg = small_cfg(n_layers=3)
    m1 = build_model(cfg, K)
    m2 = build_model(dataclasses.replace(cfg, checkpoint_layers=True), K, randomize=False)
    m2.load_state_dict(m1.state_dict())
    inp = make_inputs(cfg, K, dtype=F64)
    y1, y2 = m1(inp, K), m2(inp, K)
    assert rel(y1, y2) < 1e-13
    loss_of(y1).backward()
    loss_of(y2).backward()
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        if p1.grad is not None:
            assert rel(p1.grad, p2.grad) < 1e-12


def test_forward_equals_readout_of_hidden(device):
    """The last layer of ``forward`` only updates the readout degree; the result must equal readout(hidden)."""
    K = make_complex("sphere", device)
    for readout in ("node_scalar", "node_vector", "cochain:2", "even:1", "grad", "curl", "div", "div:1"):
        cfg = small_cfg(readout=readout, n_layers=3)
        m = build_model(cfg, K)
        inp = make_inputs(cfg, K)
        y = m(inp, K)
        h = m.hidden(inp, K)
        _, ctx = m._prepare(inp, K)
        y2 = m.readout([h[k] for k in range(K.dim + 1)], ctx)
        assert rel(y, y2) < 1e-6, readout


def test_input_validation(device):
    K = make_complex("grid", device)
    cfg = small_cfg()
    m = build_model(cfg, K)
    inp = make_inputs(cfg, K, B=2)
    bad = [
        {**inp, 0: inp[0][:-1]},                                  # wrong n_0
        {**inp, 1: inp[1][..., :1]},                              # wrong F_1
        {k: v for k, v in inp.items() if k != 2},                 # missing degree
        {**inp, 2: inp[2][:, :1]},                                # inconsistent B
        {**inp, 0: inp[0][:, :, None]},                           # wrong rank
    ]
    for b in bad:
        with pytest.raises(ValueError):
            m(b, K)
    with pytest.raises(ValueError):
        m(inp, K.to(dtype=F64))                                   # dtype mismatch with the model
    with pytest.raises(ValueError):
        m(inp, make_complex("tets", device))                      # wrong top degree
    m0 = build_model(RHMPConfig(in_dims={1: 2}, C=8, n_layers=1), K)
    with pytest.raises(ValueError):
        m0({0: inp[0], 1: inp[1]}, K)                             # input on a degree the config does not use


def test_diagnostics(device):
    """One key scheme: layer{l}.H{m}.<stat> (untied: up_/down_ prefixes), layer{l}.beta_{up,down}{k}; the learned
    condition number is 1 for a fresh model while the total one includes the (graded) star."""
    K = make_complex("delaunay", device)
    stats = ("mean", "std", "min", "max", "cond_learned", "cond_total", "clamp_fraction", "sat")
    for tie in (True, False):
        cfg = small_cfg(tie_metrics=tie, n_layers=2)
        m = build_model(cfg, K)
        m(make_inputs(cfg, K), K)
        d = m.diagnostics
        assert d and all(math.isfinite(v) for v in d.values())
        prefixes = ["H0.", "H1.", "H2."] if tie else ["H1.up_", "H2.up_", "H0.down_", "H1.down_"]
        for pre in prefixes:
            for st in stats:
                assert f"layer0.{pre}{st}" in d, (pre, st)
            assert d[f"layer0.{pre}cond_learned"] >= 1.0 and abs(d[f"layer0.{pre}max"]) <= cfg.log_range + 1e-6
        assert all(d[f"layer0.beta_{b}"] > 0 for b in ("up0", "up1", "down1", "down2"))
        assert not any(".cond" == k[-5:] or "beta_dn" in k or ".Hu" in k or ".Hd" in k for k in d)
    fresh = build_model(small_cfg(), K, randomize=False)
    fresh(make_inputs(small_cfg(), K), K)
    d = fresh.diagnostics
    assert d["layer0.H1.cond_learned"] == pytest.approx(1.0) and d["layer0.H1.cond_total"] > 10.0


@pytest.mark.parametrize("tensor_param", ["full", "cone"])
def test_diagnostics_tensor_and_resolvent_keys(tensor_param, device):
    K = make_complex("delaunay", device)
    cfg = small_cfg(metric_type="tensor", tensor_param=tensor_param,
                    layers=["poly", "resolvent", "poly"])                    # a non-final resolvent layer
    m = build_model(cfg, K)
    m(make_inputs(cfg, K), K)
    d = m.diagnostics
    own = ("s_absmax", "clamp_s") if tensor_param == "full" else ("a_mean", "a_max", "clamp_a")
    for st in ("logb_mean", "aniso_mean", "aniso_max", "clamp_b") + own:
        assert f"layer0.H1.tensor_{st}" in d
    assert d["layer0.H1.tensor_aniso_max"] >= 1.0
    assert all(f"layer1.{k}" in d for k in ("tau_up0", "tau_up1", "tau_down1", "tau_down2", "cg_res0", "cg_res2"))
    assert all(k.split(".")[1].startswith(("H", "beta_", "tau_", "cg_res")) for k in d)


def test_version_and_lazy_exports():
    import rhmp
    assert rhmp.__version__ == "2.0.0"
    from rhmp import CochainComplex as CC, cg_solve, flat, hodge_decompose, load_task, sharp  # noqa: F401
    from rhmp import dec
    assert cg_solve is dec.cg_solve and sharp is dec.sharp and flat is dec.flat and hodge_decompose is dec.hodge_decompose
    assert callable(load_task) and CC.__name__ == "CochainComplex"
    assert set(rhmp.__all__) >= {"RHMP", "RHMPConfig", "CochainComplex", "cg_solve", "load_task", "__version__"}


def test_from_checkpoint_trainer_formats(tmp_path, device):
    """RHMP.from_checkpoint reads to_checkpoint() files, the trainer's best.pt (extra metadata) and last.pt (model
    under 'model' next to optimizer / scheduler / numpy RNG state, which needs full unpickling)."""
    import numpy as np
    K = make_complex("grid", device)
    cfg = small_cfg()
    m = build_model(cfg, K)
    inp = make_inputs(cfg, K)
    y = m(inp, K)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10)
    best = {**m.to_checkpoint(), "epoch": 3, "val": {"R2": 0.5}, "task": "X", "native": True}
    last = {"model": m.to_checkpoint(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "rng": {"numpy": np.random.get_state(), "python": (1, 2)}, "history": [], "epoch": 3}
    torch.save(best, tmp_path / "best.pt")
    torch.save(last, tmp_path / "last.pt")
    for path in ("best.pt", "last.pt"):
        m2 = RHMP.from_checkpoint(tmp_path / path, map_location=device)
        assert isinstance(m2, RHMP) and m2.cfg == cfg and rel(m2(inp, K), y) < 1e-6
    assert rel(RHMP.from_checkpoint(last).to(device)(inp, K), y) < 1e-6                  # dict with nested model
    with pytest.raises(ValueError):
        RHMP.from_checkpoint({"optimizer": {}})


def test_complex_graph_ids_and_from_list(device):
    from conftest import make_batch
    from rhmp import CochainComplex
    K = make_complex("grid", device)
    assert all(K.graph_ids(k) is None for k in range(K.dim + 1))
    with pytest.raises(ValueError):
        K.graph_ids(K.dim + 1)
    Kb, parts = make_batch(device)
    Kb, parts = Kb.to(dtype=F64), [P.to(dtype=F64) for P in parts]
    Kl = CochainComplex.from_list(parts)
    assert Kl.n == Kb.n and Kl.num_graphs == Kb.num_graphs == len(parts)
    for k in range(Kl.dim + 1):
        assert torch.equal(Kl.graph_ids(k), Kb.batch[k])
    cfg = small_cfg(readout="cochain:1")
    m = build_model(cfg, parts[0], dtype=F64)
    inp = {k: torch.cat([make_inputs(cfg, P, B=1, seed=i, dtype=F64)[k] for i, P in enumerate(parts)], 0)
           for k in cfg.in_dims}
    # identical complexes; CUDA scatter/segment reductions are not bitwise deterministic (repeated runs on the same
    # complex already differ at the rounding level), hence the device-dependent tolerance
    assert rel(m(inp, Kl), m(inp, Kb)) < (1e-12 if str(device) == "cpu" else 1e-10)


def test_amp_matches_fp32():
    if not torch.cuda.is_available():
        pytest.skip("CUDA only")
    K = make_complex("delaunay", "cuda")
    cfg = small_cfg(C=32)
    m = build_model(cfg, K)
    inp = make_inputs(cfg, K, B=4)
    y32 = m(inp, K)
    m.cfg.amp = True
    y16 = m(inp, K)
    loss_of(y16).backward()
    assert y16.dtype == torch.float32
    assert rel(y16, y32) < 5e-2
    assert all_finite(m, y16)


def _python_headers_available() -> bool:
    import os
    import sysconfig
    return os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h"))


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_torch_compile_smoke(backend):
    """``torch.compile`` must not break the model (graph breaks around the sparse products are allowed)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA only")
    if backend == "inductor" and not _python_headers_available():
        pytest.xfail("Triton cannot build its CUDA launcher on this host: the Python development headers "
                     "(Python.h) are missing; the model itself traces fine (see the aot_eager case)")
    K = make_complex("delaunay", "cuda")
    cfg = small_cfg()
    m = build_model(cfg, K)
    inp = make_inputs(cfg, K)
    y = m(inp, K)
    torch._dynamo.reset()
    mc = torch.compile(m, backend=backend)
    yc = mc(inp, K)
    loss_of(yc).backward()
    assert rel(yc, y) < 1e-4
    assert all_finite(m, yc)


def test_tensor_config_errors(device):
    K = make_complex("mixed", device)
    with pytest.raises(ValueError):
        build_model(small_cfg(metric_type="tensor", scaling="jacobi"), make_complex("grid", device))
    m = build_model(small_cfg(metric_type="tensor"), K)
    with pytest.raises(ValueError):                                           # polygons have no Whitney blocks
        m(make_inputs(small_cfg(metric_type="tensor"), K), K)


def test_config_backward_compatible():
    """Stage-2 config keys have defaults: dicts saved before they existed still load (diag metric, all poly)."""
    d = RHMPConfig(in_dims={0: 3}, C=16).to_dict()
    for key in ("layers", "resolvent_iters", "resolvent_grad", "metric_type"):
        d.pop(key)
    cfg = RHMPConfig.from_dict(d)
    assert cfg.metric_type == "diag" and cfg.layer_types == ["poly"] * cfg.n_layers
    assert RHMPConfig(in_dims={0: 3}, layers=["poly", "resolvent"]).n_layers == 2
    with pytest.raises(ValueError):
        RHMPConfig(in_dims={0: 3}, layers=["poly", "fourier"])


@pytest.mark.parametrize("metric_type,tie", [("diag", True), ("diag", False), ("tensor", True)])
def test_metric_reference_offset(metric_type, tie, device):
    """metric_reference = log of a positive field => H/star equals that field at initialisation (the learned bounded
    part is zero); tensor metrics scale b_f by the geometric mean of the field over the edges of f (a = 0)."""
    K = make_complex("delaunay", device).to(dtype=F64)
    cfg = small_cfg(in_dims={0: 3, 1: 2, 2: 2}, even_dims={0: 1, 1: 1, 2: 1}, metric_reference={1: 1, 2: 1},
                    metric_type=metric_type, tie_metrics=tie)
    assert RHMPConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
    m = build_model(cfg, K, randomize=False)
    inp = make_inputs(cfg, K, dtype=F64)
    g = torch.Generator().manual_seed(5)
    field = torch.exp(torch.randn(K.n[1], inp[1].shape[1], generator=g, dtype=F64)).to(device)
    inp[1] = inp[1].clone()
    inp[1][..., 1] = torch.log(field)
    f = m.metric_fields(inp, K)
    assert len(f) == cfg.n_layers
    for layer in f:
        assert rel(layer["log_ratio"][1], torch.log(field)) < 1e-12
        assert float(layer["phi"][1].abs().max()) < 1e-12
        if metric_type == "tensor":                  # sigma_f = exp(mean edge reference + face reference) I
            cells = K.whitney[1]["cells"]
            expect = torch.exp(torch.log(field)[cells].mean(1) + inp[2][..., 1])    # (n2, B)
            eye = torch.eye(2, dtype=F64, device=device)
            assert rel(layer["sigma"][1], expect[..., None, None] * eye) < 1e-12
            b, a = layer["tensor"][1]                  # min-norm {I, dyad} coordinates (reporting)
            t = K.whitney[1]["t"].to(F64)
            sig = b[..., None, None] * eye + torch.einsum("fbj,fjd,fje->fbde", a, t, t)
            assert rel(sig, layer["sigma"][1]) < 1e-6
    d = m.diagnostics
    assert all(v == 0.0 for k, v in d.items() if k.endswith(".clamp_fraction") or k.endswith(".sat"))
    with pytest.raises(ValueError):                                         # must be an even column
        small_cfg(in_dims={0: 3, 1: 2, 2: 2}, even_dims={0: 1, 1: 1, 2: 1}, metric_reference={1: 0})


def test_clamp_fraction_detects_saturation(device):
    K = make_complex("delaunay", device)
    cfg = small_cfg()
    m = build_model(cfg, K)
    with torch.no_grad():
        for layer in m.layers:
            for head in layer.heads.values():
                head.fc2.weight.mul_(1000.0)
                head.fc2.bias.fill_(50.0)
    m(make_inputs(cfg, K), K)
    d = m.diagnostics
    assert max(v for k, v in d.items() if k.endswith(".clamp_fraction")) > 0.5
    assert max(v for k, v in d.items() if k.endswith(".sat")) > 0.5


def test_resolvent_warm_start(device):
    """Warm-starting a resolvent layer from the previous one's solution gives the same converged result and the same
    (implicit) gradients; from the exact solution the correction is ~0 after a single iteration."""
    from rhmp.layers import block_operands, make_context, resolvent_apply
    K = make_complex("delaunay", device).to(dtype=F64)
    cfg = small_cfg(layers=["resolvent", "poly", "resolvent"], resolvent_iters=40, readout="cochain:1")
    m1 = build_model(cfg, K)
    m2 = build_model(dataclasses.replace(cfg, resolvent_warm_start=False), K, randomize=False)
    m2.load_state_dict(m1.state_dict())
    inp = make_inputs(cfg, K, dtype=F64)
    y1, y2 = m1(inp, K), m2(inp, K)
    assert rel(y1, y2) < 1e-8
    loss_of(y1).backward()
    loss_of(y2).backward()
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        if p1.grad is not None:
            assert rel(p1.grad, p2.grad) < 1e-6
    ctx = make_context(K, scaling="dec", B=2, dtype=F64)
    lh = torch.zeros(K.n[2], 2, dtype=F64, device=device)
    op = block_operands(ctx, 1, True, lh)
    t = torch.tensor(2.0, dtype=F64, device=device)
    x = torch.randn(K.n[1], 2, 3, dtype=F64, generator=torch.Generator().manual_seed(0)).to(device)
    y, _ = resolvent_apply(ctx, 1, x, op, None, t, None, iters=200)
    y_warm, res = resolvent_apply(ctx, 1, x, op, None, t, None, iters=1, x0=y)
    assert float(res.max()) < 1e-10 and rel(y_warm, y) < 1e-10


@pytest.mark.parametrize("variant", [{}, dict(tie_metrics=False), dict(metric_reference={1: 1, 2: 1}),
                                     dict(metric_type="tensor"), dict(metric_type="tensor", metric_reference={1: 1}),
                                     dict(layers=["poly", "resolvent"])])
def test_learn_metric_false_is_the_fixed_dec_prior(variant, tmp_path, device):
    """learn_metric=False: no metric parameters, H_m / star_m == exp(ref_m) exactly in every layer (tensor metric:
    b = exp(mean ref), a = 0, i.e. the Galerkin star), and checkpoints round-trip."""
    K = make_complex("delaunay", device).to(dtype=F64)
    base = dict(in_dims={0: 3, 1: 2, 2: 2}, even_dims={0: 1, 1: 1, 2: 1}, learn_metric=False, readout="cochain:1")
    cfg = small_cfg(**{**base, **variant})
    m = build_model(cfg, K)
    assert not any(".heads." in k or ".theads." in k for k in m.state_dict())
    assert m.num_parameters() < build_model(dataclasses.replace(cfg, learn_metric=True), K).num_parameters()
    inp = make_inputs(cfg, K, dtype=F64)
    for layer in m.metric_fields(inp, K):
        for deg, lr in layer["log_ratio"].items():
            col = cfg.metric_reference.get(deg)
            expect = inp[deg][..., col] if col is not None else torch.zeros_like(lr)
            assert float((lr - expect).abs().max()) < 1e-12 and float(layer["phi"][deg].abs().max()) < 1e-12
        for km, (b, a) in layer["tensor"].items():
            col = cfg.metric_reference.get(km)
            ref = inp[km][..., col] if col is not None else torch.zeros(K.n[km], 2, dtype=F64, device=device)
            cells = K.whitney[km]["cells"]
            assert rel(b, torch.exp(ref[cells].mean(1))) < 1e-12 and float(a.abs().max()) == 0.0
    y = m(inp, K)
    loss_of(y).backward()
    assert all_finite(m, y)
    d = m.diagnostics
    assert all(v == pytest.approx(1.0) for k, v in d.items() if k.endswith("cond_learned"))
    torch.save(m.to_checkpoint(), tmp_path / "fixed.pt")
    m2 = RHMP.from_checkpoint(tmp_path / "fixed.pt", map_location=device).to(F64)
    assert m2.cfg.learn_metric is False and rel(m2(inp, K), y) < 1e-12


def _flat_tensors(obj, prefix=""):
    """(name, tensor) pairs of every tensor (CSR components included) in nested containers of a complex field."""
    if isinstance(obj, torch.Tensor):
        if obj.layout == torch.sparse_csr:
            return [(prefix + ".crow", obj.crow_indices()), (prefix + ".col", obj.col_indices()),
                    (prefix + ".val", obj.values())]
        return [(prefix, obj)]
    if isinstance(obj, (list, tuple)):
        return [p for i, v in enumerate(obj) for p in _flat_tensors(v, f"{prefix}[{i}]")]
    if isinstance(obj, dict):
        return [p for k, v in obj.items() for p in _flat_tensors(v, f"{prefix}.{k}")]
    return []


def test_complex_to_cuda_pinned_nonblocking():
    """CPU -> CUDA moves use pinned, asynchronous copies: results are bitwise identical to blocking copies, the source
    complex keeps its values (now page-locked), repeated moves work, and CUDA -> CPU is unchanged."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA only")
    from rhmp.complex import _TENSOR_FIELDS
    K = make_complex("tets", "cpu")
    ref = {n: t.clone() for f in _TENSOR_FIELDS for n, t in _flat_tensors(getattr(K, f), f)}
    Kb = K.to("cuda", non_blocking=False)                                     # plain blocking copies
    Kp = K.to("cuda")                                                          # pinned + non_blocking
    Kp2 = K.to("cuda", dtype=F64)
    torch.cuda.synchronize()
    for f in _TENSOR_FIELDS:
        a, b = dict(_flat_tensors(getattr(Kp, f), f)), dict(_flat_tensors(getattr(Kb, f), f))
        assert a.keys() == b.keys()
        for n in a:
            assert a[n].device.type == "cuda" and torch.equal(a[n], b[n]), n
    for f in _TENSOR_FIELDS:                                                   # source: same values, now pinned
        for n, t in _flat_tensors(getattr(K, f), f):
            assert torch.equal(t, ref[n]) and t.is_pinned(), n
    assert Kp2.star[0].dtype == F64 and torch.equal(Kp2.star[0].float().cpu(), K.star[0])
    back = Kp.to("cpu")
    assert back.pos.device.type == "cpu" and torch.equal(back.pos, K.pos)
    cfg = small_cfg(dim=3)
    m = build_model(cfg, Kb)
    inp = make_inputs(cfg, Kb)
    assert rel(m(inp, Kp), m(inp, Kb)) == 0.0


def test_latent_fields_shapes_and_gradients(device):
    """latent_dims: learnable per-cell columns Z_k (n_k, r) inserted before the even block; they receive gradients,
    also on degrees without data inputs; init is N(0, 0.1^2)."""
    import warnings as _w
    K = make_complex("delaunay", device)
    cfg = small_cfg(latent_dims={0: 2, 1: 3}, readout="cochain:1")
    m = build_model(cfg, K, randomize=False)
    assert len(m.latent) == 0
    m.init_latents(K)
    assert m.latent["0"].shape == (K.n[0], 2) and m.latent["1"].shape == (K.n[1], 3)
    assert 0.05 < float(m.latent["1"].detach().std()) < 0.2
    with _w.catch_warnings():
        _w.simplefilter("error")                                             # no lazy-creation warning after init
        y = m(make_inputs(cfg, K), K)
    loss_of(y).backward()
    assert m.latent["1"].grad is not None and float(m.latent["1"].grad.abs().sum()) > 0 and all_finite(m, y)
    m2 = build_model(RHMPConfig(in_dims={0: 3}, latent_dims={1: 4}, C=8, n_layers=2, readout="cochain:1"), K)
    with pytest.warns(RuntimeWarning, match="lazily"):
        y2 = m2({0: torch.randn(K.n[0], 2, 3, device=device)}, K)           # latent on a degree without inputs
    assert y2.shape == (K.n[1], 2, 1) and m2.latent["1"].shape == (K.n[1], 4)


def test_latent_fields_errors_on_other_meshes(device):
    from conftest import make_batch
    K1, K2 = make_complex("delaunay", device, n=60), make_complex("delaunay", device, n=80)
    cfg = small_cfg(latent_dims={1: 2})
    m = build_model(cfg, K1).init_latents(K1)
    m(make_inputs(cfg, K1), K1)
    with pytest.raises(ValueError, match="tied to one mesh"):
        m(make_inputs(cfg, K2), K2)
    Kb, parts = make_batch(device)
    fresh = build_model(cfg, parts[0])
    with pytest.raises(ValueError, match="block-diagonal"):
        fresh.init_latents(Kb)


def test_latent_fields_checkpoint_roundtrip(tmp_path, device):
    K = make_complex("grid", device)
    cfg = small_cfg(latent_dims={0: 1, 1: 2, 2: 1})
    m = build_model(cfg, K).init_latents(K)
    inp = make_inputs(cfg, K)
    y = m(inp, K)
    torch.save(m.to_checkpoint(), tmp_path / "latent.pt")
    m2 = RHMP.from_checkpoint(tmp_path / "latent.pt", map_location=device)
    assert set(m2.latent) == {"0", "1", "2"} and rel(m2(inp, K), y) < 1e-6
    m3 = build_model(cfg, K, randomize=False)                                # trainer resume: load_state_dict
    m3.load_state_dict(m.state_dict())
    assert rel(m3(inp, K), y) < 1e-6
    assert RHMPConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
