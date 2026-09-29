"""Baselines (rhmp.baselines): adapters, v1 fidelity, forward/backward on every fixture, param matching, MGN
batch consistency and orientation equivariance, fixed-metric controls, applicability, trainer integration."""
from __future__ import annotations

import copy
import os
import pickle
import sys

import numpy as np
import pytest
import torch

from conftest import ROOT, make_batch, make_complex, mesh_delaunay_2d, relabel_mesh
from rhmp.baselines import registry
from rhmp.baselines.adapters import NodeInputEncoder, V1ComplexAdapter, adapter_for, task_io
from rhmp.baselines.v1_wrappers import (COMPLEX_CORES, NODE_CORES, import_v1)
from rhmp.complex import CochainComplex
from rhmp.data import Stats, TaskData, edge_alignment, mesh_minibatch, sequential_split

ALL_MODELS = registry.model_names()


# ================================================================================================================
# tiny tasks
# ================================================================================================================
def _inputs_for(K, g, top_even: bool = True) -> dict:
    """node (1) | edge (odd 1, even 1) | top degree (even 1) inputs for one sample, ``(n_k, F)``."""
    x = {0: torch.randn(K.n[0], 1, generator=g), 1: torch.randn(K.n[1], 2, generator=g)}
    if top_even:
        x[K.dim] = torch.randn(K.n[K.dim], 1, generator=g)
    return x


def _target_for(K, target: str, g) -> tuple[torch.Tensor, int, str, int]:
    if target == "node":
        return torch.randn(K.n[0], 1, generator=g), 0, "node_scalar", 1
    if target == "vector":
        D = K.pos.shape[1]
        return torch.randn(K.n[0], D, generator=g), 0, "node_vector", D
    if target == "edge":
        return torch.randn(K.n[1], 1, generator=g), 1, "cochain", 1
    if target == "edge_even":
        return torch.randn(K.n[1], 1, generator=g), 1, "even", 1
    if target == "face":
        return torch.randn(K.n[2], 1, generator=g), 2, "cochain", 1
    raise KeyError(target)


def _stats(F: int) -> Stats:
    return Stats(torch.zeros(F), torch.ones(F))


def tiny_task(Ks, target: str = "node", N: int = 8, seed: int = 0, name: str = "tiny") -> TaskData:
    """Shared-mesh task (``Ks`` a complex) or variable-mesh task (``Ks`` a list) with random data."""
    g = torch.Generator().manual_seed(seed)
    shared = not isinstance(Ks, (list, tuple))
    K0 = Ks if shared else Ks[0]
    dev = K0.pos.device
    if shared:
        xs = [_inputs_for(K0, g) for _ in range(N)]
        ys = [_target_for(K0, target, g) for _ in range(N)]
        inputs = {k: torch.stack([x[k] for x in xs]).to(dev) for k in xs[0]}
        tgt = torch.stack([y[0] for y in ys]).to(dev)
    else:
        N = len(Ks)
        xs = [_inputs_for(K, g) for K in Ks]
        ys = [_target_for(K, target, g) for K in Ks]
        inputs = [{k: v.to(dev) for k, v in x.items()} for x in xs]
        tgt = [y[0].to(dev) for y in ys]
    _, t, kind, out = ys[0]
    in_dims = {k: int(v.shape[-1]) for k, v in xs[0].items()}
    even = {1: 1, K0.dim: 1}
    return TaskData(name=name, K=Ks, inputs=inputs, target=tgt, target_degree=t, target_kind=kind,
                    in_dims=in_dims, even_dims=even, connection_dims={}, split=sequential_split(N),
                    x_stats={k: _stats(v) for k, v in in_dims.items()}, y_stats=_stats(out),
                    spatial_dim=int(K0.pos.shape[1]), out_dim=out, native=True,
                    meta={"v1_C": 8, "layers": 2, "batch_size": 4})


def batch_of(td: TaskData, B: int = 3):
    """``(K, inputs {k: (n_k, B, F)}, target (n_t, B, O))`` of the first ``B`` samples."""
    idx = torch.arange(B)
    if td.variable_mesh:
        return mesh_minibatch(td.K, td.inputs, td.target, idx)
    xb = {k: v[:B].transpose(0, 1).contiguous() for k, v in td.inputs.items()}
    return td.K, xb, td.target[:B].transpose(0, 1).contiguous()


def _small_overrides(name: str) -> dict:
    """Tiny widths so that every model builds and runs fast in tests."""
    fam = registry.SPECS[name].family
    if fam == "rhmp":
        from rhmp.train import parse_args
        return {"args": parse_args(["--task", "tiny", "--C", "8", "--layers", "2"])}
    if name == "ours_v1":
        return {"C": 8}
    if name in ("mgn", "mgn_fast"):
        return {"hidden": 8, "n_layers": 2}
    if name == "fno":
        return {"hidden": 8, "modes": 3}
    return {"hidden": 8}


FIXTURES = {
    "tri": lambda dev, star: make_complex("delaunay", dev, star=star, n=40, seed=1),
    "sphere": lambda dev, star: make_complex("sphere", dev, star=star, subdiv=1),
    "quad": lambda dev, star: make_complex("quads", dev, star=star),
    "tet": lambda dev, star: make_complex("tets", dev, star=star, n=30),
    "grid": lambda dev, star: CochainComplex.from_grid((8, 8), 1 / 7, cell="tri", diagonal="main",
                                                       star=star or "cotan", device=dev),
    "batch": lambda dev, star: [make_complex("delaunay", dev, star=star, n=n, seed=s)
                                for s, n in enumerate((20, 30, 40, 25, 35, 30))],
}
CASES = [(f, t) for f in ("tri", "batch") for t in ("node", "edge", "face")] + \
        [("sphere", "vector"), ("quad", "node"), ("tet", "node"), ("grid", "node"), ("tri", "edge_even")]


# ================================================================================================================
# adapter
# ================================================================================================================
def test_import_v1_returns_vendored_modules():
    """The v1 code is vendored in rhmp.baselines.v1 (no sys.path manipulation, no external checkout)."""
    v1 = import_v1()
    assert set(v1) == {"graph", "topo", "adv", "op", "network"}
    for mod in v1.values():
        assert mod.__name__.startswith("rhmp.baselines.v1."), mod.__name__


def test_adapter_matches_v1_cellcomplex():
    from rhmp.baselines.v1.gauge_hodge_mp.cell_complex import CellComplex
    pos, faces = mesh_delaunay_2d(60, seed=3)
    K = CochainComplex.from_triangles(pos, faces)
    A = V1ComplexAdapter(K)
    Kv = CellComplex.from_triangulation(torch.tensor(pos, dtype=torch.float32), torch.tensor(faces))
    assert torch.equal(A.edges, Kv.edges)
    assert torch.equal(A.faces, Kv.faces)
    assert (A.n0, A.n1, A.n2) == (Kv.n0, Kv.n1, Kv.n2)
    assert torch.equal(A.d0.to_dense(), Kv.d0.to_dense())
    assert torch.equal(A.d1.to_dense(), Kv.d1.to_dense())
    assert torch.equal(A.get_adjacency(0, 0), Kv.get_adjacency(0, 0))
    assert torch.equal(A.get_adjacency(1, 1), Kv.get_adjacency(1, 1))
    # non-triangle complexes refuse the v1 'faces' attribute with the list of inapplicable models
    Aq = V1ComplexAdapter(make_complex("quads"))
    with pytest.raises(NotImplementedError, match="ours_v1"):
        _ = Aq.faces
    with pytest.raises(NotImplementedError):
        _ = V1ComplexAdapter(make_complex("tets")).faces


@pytest.mark.parametrize("op", ["gcn_adj", "A00", "A11lo", "A11up", "A22", "L0_n", "L1_n", "L2_n", "mean_1to0",
                                "mean_2to0", "mean_0to2", "mean_1to2", "mean_2to1"])
def test_adapter_block_diagonal(op, device):
    """Every cached operator of a block-diagonal batch acts as the per-mesh operators (per-mesh normalisations)."""
    Kb, parts = make_batch(device)
    src_deg = {"gcn_adj": 0, "A00": 0, "A11lo": 1, "A11up": 1, "A22": 2, "L0_n": 0, "L1_n": 1, "L2_n": 2,
               "mean_1to0": 1, "mean_2to0": 2, "mean_0to2": 0, "mean_1to2": 1, "mean_2to1": 2}[op]
    g = torch.Generator().manual_seed(0)
    xs = [torch.randn(K.n[src_deg], 2, 3, generator=g).to(device) for K in parts]
    yb = adapter_for(Kb).apply(op, torch.cat(xs))
    yl = torch.cat([V1ComplexAdapter(K).apply(op, x) for K, x in zip(parts, xs)])
    assert torch.allclose(yb, yl, atol=1e-5)


def test_adapter_geometry_and_orientation():
    K = make_complex("delaunay", n=50, seed=2)
    A = V1ComplexAdapter(K)
    assert torch.allclose(A.edge_length_normalized().mean(), torch.tensor(1.0), atol=1e-5)
    sig = A.face_orientation()
    F_ = K.cells[2]
    P = K.pos.double()
    a, b = P[F_[:, 1]] - P[F_[:, 0]], P[F_[:, 2]] - P[F_[:, 0]]
    ref = torch.sign(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).float()
    assert torch.equal(sig, ref)
    assert V1ComplexAdapter(make_complex("sphere")).face_orientation() is None
    assert A.grid_shape() is None
    Kg = CochainComplex.from_grid((5, 7), 0.25, cell="tri", diagonal="main")
    assert V1ComplexAdapter(Kg).grid_shape() == (5, 7)


# ================================================================================================================
# input encoder == v1 encoding
# ================================================================================================================
def _v1_loop_encoding(theta: np.ndarray, edges: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Literal port of ``encode_edges_to_nodes`` (v1 generator ``gen_T6_wilson_loop.py``; Python loop over edges)."""
    N, n1, c = theta.shape
    n = pts.shape[0]
    ed = pts[edges[:, 1]] - pts[edges[:, 0]]
    ed = ed / (np.linalg.norm(ed, axis=1, keepdims=True) + 1e-12)
    avg, adx, ady = np.zeros((N, n, c)), np.zeros((N, n, c)), np.zeros((N, n, c))
    cnt = np.zeros(n)
    for ei in range(n1):
        a, b = edges[ei]
        v = theta[:, ei]
        for node in (a, b):
            avg[:, node] += v
            adx[:, node] += v * ed[ei, 0]
            ady[:, node] += v * ed[ei, 1]
            cnt[node] += 1
    cnt[cnt == 0] = 1
    return np.concatenate([avg, adx, ady], -1) / cnt[None, :, None]


@pytest.mark.parametrize("c", [1, 3])
def test_node_encoder_matches_v1_loop(c):
    pos, faces = mesh_delaunay_2d(80, seed=5)
    K = CochainComplex.from_triangles(pos, faces)
    theta = np.random.default_rng(0).standard_normal((4, K.n[1], c)).astype(np.float32)
    ref = _v1_loop_encoding(theta.astype(np.float64), K.cells[1].numpy(), pos)
    enc = NodeInputEncoder({1: c}, {}, 2)
    out = enc({1: torch.from_numpy(theta).transpose(0, 1).contiguous()}, adapter_for(K))
    assert out.shape == (K.n[0], 4, 3 * c)
    assert np.abs(out.transpose(0, 1).numpy() - ref).max() < 1e-5
    # EGNN mode: avg columns only
    inv = NodeInputEncoder({1: c}, {}, 2, directional=False)
    o2 = inv({1: torch.from_numpy(theta).transpose(0, 1).contiguous()}, adapter_for(K))
    assert np.abs(o2.transpose(0, 1).numpy() - ref[..., :c]).max() < 1e-5


def test_node_encoder_reproduces_T6_legacy_inputs():
    """Encoding the native T6 edge connection reproduces the stored v1 node inputs X_data (needs the T6 datasets)."""
    nat = os.path.join(ROOT, "datasets", "v2", "T6_native.pt")
    pkl = os.path.join(ROOT, "datasets", "T6_wilson_loop.pkl")
    if not (os.path.exists(nat) and os.path.exists(pkl)):
        pytest.skip("T6 datasets not present (see datasets/README.md)")
    d = pickle.load(open(pkl, "rb"))
    blob = torch.load(nat, map_location="cpu", weights_only=False)
    pts = np.asarray(d["points"], dtype=np.float64)[:, :2]
    K = CochainComplex.from_triangles(pts, np.asarray(d["faces"], dtype=np.int64))
    perm, sign = edge_alignment(K.cells[1], blob["edges"], K.n[0])
    theta = (blob["theta_edges"][:32][:, perm] * sign[None]).unsqueeze(-1)           # (32, n1, 1)
    X = torch.as_tensor(np.asarray(d["X_data"][:32], dtype=np.float32))
    out = NodeInputEncoder({1: 1}, {}, 2)({1: theta.transpose(0, 1).contiguous()}, adapter_for(K))
    err = (out.transpose(0, 1) - X).abs().max().item() / X.abs().max().item()
    assert err < 1e-5, err


# ================================================================================================================
# v1 fidelity of the vectorised cores
# ================================================================================================================
def _v1_model(name: str, f_in: int, h: int, out: int):
    reg = registry.MODELS  # noqa: F841  (ensures the registry imports)
    from rhmp.baselines.v1_wrappers import _v1_class
    cls = _v1_class(name)
    if name in NODE_CORES:
        return cls(f_in=f_in, hidden=h, n_layers=3, out_dim=out, task="node")
    return cls(f_in=f_in, hidden=h, n_layers=3)


@pytest.mark.parametrize("name", list(NODE_CORES) + list(COMPLEX_CORES))
def test_core_fidelity(name):
    """core (v1_exact, raw geometry, unnormalised SCCNN) + v1 head == v1 forward, per sample and batched."""
    torch.manual_seed(0)
    pos, faces = mesh_delaunay_2d(45, seed=7)
    K = CochainComplex.from_triangles(pos, faces)
    A = V1ComplexAdapter(K)
    B, f_in, h = 3, 4, 12
    m = _v1_model(name, f_in, h, 2)
    with torch.no_grad():
        for p in m.parameters():                     # non-trivial values for zero-initialised parameters
            p.add_(0.05 * torch.randn_like(p))
    f0 = torch.randn(B, K.n[0], f_in)
    kw = {}
    if name in ("schnet", "egnn"):
        kw["geometry"] = "raw"
    if name in ("gauge_cnn", "gem_cnn"):
        kw["v1_exact"] = True
    if name == "sccnn":
        kw["normalize"] = False
    core = (NODE_CORES.get(name) or COMPLEX_CORES[name])(copy.deepcopy(m), **kw)
    with torch.no_grad():
        feats = core(f0.transpose(0, 1).contiguous(), A)
        if name in ("gcn", "gat", "schnet", "egnn"):
            ref = torch.stack([m.forward(f0[i], A) for i in range(B)])            # v1 per-sample path
            got = m.node_mlp(feats[0])
        elif name in ("gauge_cnn", "gem_cnn"):
            ref = m.forward_batch(f0, A)
            got = m.decoder(feats[0])
        elif name == "cw_net":
            ref = m.forward_batch(f0, A)
            got = m.decoder(feats[0])
        else:                                        # mpsn / sccnn / clifford: v1 edge readout on x1
            ref = m.forward_batch(f0, A)
            got = m.readout(feats[1])
    got = got.transpose(0, 1)
    scale = ref.abs().max().clamp_min(1e-3)
    assert torch.allclose(got, ref, atol=2e-5 * scale, rtol=1e-4), (got - ref).abs().max().item()


def test_ours_v1_wrapper_matches_v1():
    """ours_v1 on the adapter == v1 GaugeHodgeNetwork.forward_batch on v1's CellComplex (same weights)."""
    from rhmp.baselines.v1.gauge_hodge_mp.cell_complex import CellComplex
    torch.manual_seed(0)
    pos, faces = mesh_delaunay_2d(40, seed=4)
    K = CochainComplex.from_triangles(pos, faces)
    td = tiny_task(K, "node")
    model, _ = registry.build_model("ours_v1", td, None, C=8)
    Kv = CellComplex.from_triangulation(torch.tensor(pos, dtype=torch.float32), torch.tensor(faces))
    _, xb, _ = batch_of(td, 3)
    model.train()
    with torch.no_grad():
        y = model(xb, K)
        f0 = model.encoder(xb, adapter_for(K)).transpose(0, 1).contiguous()
        ref = model.core.v1.forward_batch(f0, Kv).transpose(0, 1)
    assert torch.allclose(y, ref, atol=1e-5)


# ================================================================================================================
# every model: forward / backward on every applicable fixture
# ================================================================================================================
def _build_case(name, fixture, target, device):
    star = registry.SPECS[name].data_star
    Ks = FIXTURES[fixture](device, star)
    td = tiny_task(Ks, target)
    ok, why = registry.applicable(name, td)
    if not ok:
        pytest.skip(f"{name} on {fixture}/{target}: {why}")
    torch.manual_seed(0)
    model, info = registry.build_model(name, td, None, cache_path=None, **_small_overrides(name))
    return td, model, info


@pytest.mark.parametrize("fixture,target", CASES)
@pytest.mark.parametrize("name", ALL_MODELS)
def test_forward_backward(name, fixture, target, device):
    td, model, info = _build_case(name, fixture, target, device)
    K, xb, yb = batch_of(td, 3)
    model.train()
    y = model(xb, K)
    assert y.shape == yb.shape, (y.shape, yb.shape)
    loss = torch.nn.functional.mse_loss(y, yb)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads if g is not None)
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
    assert info["params"] == sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.eval()
    with torch.no_grad():
        y2 = model(xb, K)
    assert torch.isfinite(y2).all()


@pytest.fixture
def no_tf32():
    """Full fp32 on CUDA (cuDNN picks batch-size dependent TF32 algorithms, e.g. FNO's 1x1 convolutions)."""
    old = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


@pytest.mark.parametrize("name", [n for n in ALL_MODELS if registry.SPECS[n].family != "rhmp" and n != "ours_v1"])
def test_predictions_independent_of_batch(name, device, no_tf32):
    """Sample i's prediction does not depend on the other samples (shared mesh: B=3 vs B=1; variable meshes:
    block-diagonal batch vs single mesh)."""
    for fixture in ("tri", "batch", "grid"):
        td = tiny_task(FIXTURES[fixture](device, None), "node")
        ok, _ = registry.applicable(name, td)
        if not ok:
            continue
        torch.manual_seed(0)
        model, _ = registry.build_model(name, td, None, cache_path=None, **_small_overrides(name))
        model.eval()
        with torch.no_grad():
            K, xb, _ = batch_of(td, 3)
            y3 = model(xb, K)
            if td.variable_mesh:
                K1, x1, _ = mesh_minibatch(td.K, td.inputs, td.target, torch.arange(1))
                y1 = model(x1, K1)
                assert torch.allclose(y3[:y1.shape[0]], y1, atol=1e-5)
            else:
                y1 = model({k: v[:, :1].contiguous() for k, v in xb.items()}, K)
                assert torch.allclose(y3[:, :1], y1, atol=1e-5)


def _fit_identity(name: str, steps: int = 40, **ov) -> tuple[float, float]:
    """Train ``name`` for ``steps`` Adam steps on a learnable tiny task (target = the node input); returns
    (initial loss, best of the last 5 losses)."""
    fixture = "grid" if name == "fno" else "tri"
    td = tiny_task(FIXTURES[fixture]("cpu", None), "node", N=16)
    td.target = td.inputs[0].clone()                              # y = node input: learnable by every local model
    torch.manual_seed(0)
    base = {"C": 8} if name == "ours_v1" else {"hidden": 16}
    model, _ = registry.build_model(name, td, None, cache_path=None, **{**base, **ov})
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], 3e-3)
    losses = []
    for _ in range(steps):
        K, xb, yb = batch_of(td, 8)
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(xb, K), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    return losses[0], min(losses[-5:])


@pytest.mark.parametrize("name", [n for n in ALL_MODELS if registry.SPECS[n].family != "rhmp" and n != "deeponet"])
def test_trainable_on_learnable_tiny_task(name):
    """Every baseline reduces the loss of a learnable tiny task within 40 Adam steps (guards against signal
    collapse: SCCNN with normalised Laplacians and v1's filter initialisation predicts a constant and does not learn).
    DeepONet is excluded: its branch sees only the mesh mean of the inputs (no local map).  GCN only has to make
    progress: four normalised-adjacency averages cannot reproduce a white-noise node field (oversmoothing)."""
    first, last = _fit_identity(name)
    assert last < (0.95 if name == "gcn" else 0.7) * first, (first, last)


def test_sccnn_normalised_needs_identity_init():
    """SCCNN's normalised-Laplacian variant does not train with v1's filter initialisation (collapse to a constant)
    and trains with ``filter_init='identity'`` (the default whenever the Laplacians are normalised)."""
    f_v1, l_v1 = _fit_identity("sccnn", normalize=True, filter_init="v1")
    f_id, l_id = _fit_identity("sccnn", normalize=True, filter_init="identity")
    assert l_v1 > 0.9 * f_v1 and l_id < 0.7 * f_id, (f_v1, l_v1, f_id, l_id)


def test_v1_edge_protocol():
    """MPSN / SCCNN / Clifford on a v1 paper task: edge-averaged target (first component), v1 edge readout."""
    from rhmp.baselines.v1_wrappers import V1EdgeReadout
    K = FIXTURES["tri"]("cpu", None)
    td = tiny_task(K, "vector", name="T6")                        # node target with 2 components
    y = td.target
    assert registry.resolve_head("sccnn", td) == "v1_edge" and registry.resolve_head("cw_net", td) is None
    with pytest.raises(ValueError, match="prepare_task"):
        registry.build_model("sccnn", td, None, cache_path=None, hidden=8)
    td2 = registry.prepare_task("sccnn", td)
    e = K.cells[1]
    assert td2.target_degree == 1 and td2.target_kind == "even" and td2.out_dim == 1
    assert torch.allclose(td2.target, 0.5 * (y[:, e[:, 0], :1] + y[:, e[:, 1], :1]))
    assert td2.y_stats.mean.numel() == 1 and td2.meta["untransformed"] is td
    assert registry.rhmp_param_count(td2) == registry.rhmp_param_count(td)
    for name in registry.V1_EDGE_MODELS:
        m, info = registry.build_model(name, registry.prepare_task(name, td), None, cache_path=None, hidden=8)
        assert isinstance(m.head, V1EdgeReadout) and m.build["head"] == "v1_edge"
        Kb, xb, yb = batch_of(td2, 3)
        out = m(xb, Kb)
        assert out.shape == yb.shape == (K.n[1], 3, 1)
    # the symmetric node head stays available; non-paper tasks and cw_net are untouched
    m, _ = registry.build_model("sccnn", td, None, cache_path=None, hidden=8, head="node")
    assert m(batch_of(td, 2)[1], K).shape == (K.n[0], 2, 2)
    t = tiny_task(K, "node")
    assert registry.prepare_task("sccnn", t) is t
    assert registry.prepare_task("cw_net", td) is td
    assert registry.prepare_task("sccnn", td, {"head": "node"}) is td
    assert registry.prepare_task("sccnn", td, {"v1_components": "all"}).out_dim == 2


# ================================================================================================================
# MeshGraphNet
# ================================================================================================================
@pytest.mark.parametrize("target", ["node", "edge", "face"])
def test_mgn_batch_consistency(target, device):
    """MGN on a block-diagonal batch == per-mesh forward (1e-5); shared mesh B=4 == per-sample."""
    Ks = FIXTURES["batch"](device, None)
    td = tiny_task(Ks, target)
    torch.manual_seed(0)
    model, _ = registry.build_model("mgn", td, None, hidden=16, n_layers=3, cache_path=None)
    model.eval()
    with torch.no_grad():
        idx = torch.arange(4)
        Kb, xb, _ = mesh_minibatch(td.K, td.inputs, td.target, idx)
        yb = model(xb, Kb)
        ys = [model({k: v.unsqueeze(1) for k, v in td.inputs[i].items()}, td.K[i]) for i in range(4)]
    assert torch.allclose(yb, torch.cat(ys), atol=1e-5)
    td2 = tiny_task(Ks[2], target)
    with torch.no_grad():
        K, x4, _ = batch_of(td2, 4)
        y4 = model(x4, K)
        y1 = torch.cat([model({k: v[:, i:i + 1].contiguous() for k, v in x4.items()}, K) for i in range(4)], 1)
    assert torch.allclose(y4, y1, atol=1e-5)


def test_mgn_relabel_orientation_equivariance():
    """MGN is permutation-equivariant and exactly odd under edge re-orientation: relabelling the vertices (which
    re-orients edges, since edges are stored src < dst) permutes / sign-flips inputs and edge outputs consistently."""
    pos, faces = mesh_delaunay_2d(40, seed=11)
    K1 = CochainComplex.from_triangles(pos, faces)
    pos2, faces2, perm = relabel_mesh(pos, faces, seed=3)
    K2 = CochainComplex.from_triangles(pos2, faces2)
    td = tiny_task(K1, "edge")
    torch.manual_seed(0)
    model, _ = registry.build_model("mgn", td, None, hidden=16, n_layers=3, cache_path=None)
    model.eval()
    perm_t = torch.as_tensor(perm)
    e2_orig = perm_t[K2.cells[1]]                                 # K2 edges in original vertex labels
    ip, sg = edge_alignment(e2_orig, K1.cells[1], K1.n[0])
    K, xb, _ = batch_of(td, 2)
    x2 = {0: xb[0][perm_t], 1: torch.cat([xb[1][ip, :, :1] * sg.view(-1, 1, 1), xb[1][ip, :, 1:]], -1),
          2: xb[2][torch.as_tensor(_face_map(K1, K2, perm))]}
    with torch.no_grad():
        y1 = model(xb, K1)
        y2 = model(x2, K2)
    assert torch.allclose(y2, y1[ip] * sg.view(-1, 1, 1), atol=1e-5)


def _face_map(K1, K2, perm) -> np.ndarray:
    """Index of every K2 face in K1 (faces as vertex sets in original labels)."""
    f1 = {tuple(sorted(f)): i for i, f in enumerate(K1.cells[2].tolist())}
    return np.array([f1[tuple(sorted(perm[f].tolist()))] for f in K2.cells[2].numpy()])


def test_mgn_torch_compile_capture():
    """MGN runs under torch.compile (aot_eager: graph capture + autograd, no Triton needed) on shared and
    block-diagonal batches, and matches eager; the processor has no graph break."""
    import torch._dynamo as dynamo
    Ks = FIXTURES["batch"]("cpu", None)
    td = tiny_task(Ks, "edge")
    torch.manual_seed(0)
    model, _ = registry.build_model("mgn", td, None, hidden=16, n_layers=3, cache_path=None)
    dynamo.reset()
    cm = torch.compile(model, backend="aot_eager", dynamic=True)
    for idx in (torch.arange(3), torch.arange(2, 5)):
        Kb, xb, yb = mesh_minibatch(td.K, td.inputs, td.target, idx)
        y_c = cm(xb, Kb)
        y_e = model(xb, Kb)
        assert torch.allclose(y_c, y_e, atol=1e-5)
        torch.nn.functional.mse_loss(y_c, yb).backward()
    A_args = model._prepare(xb, Kb)
    expl = dynamo.explain(model._process)(*A_args[1:])
    assert expl.graph_break_count == 0, expl.break_reasons
    dynamo.reset()


def test_mgn_param_count_formula():
    from rhmp.baselines.mgn import mgn_param_count
    td = tiny_task(FIXTURES["tri"]("cpu", None), "node")
    model, info = registry.build_model("mgn", td, None, hidden=24, n_layers=5, cache_path=None)
    io = task_io(td)
    assert info["params"] == mgn_param_count(info["node_in"], info["edge_in"], io.out_dim, 24, 5)


# ================================================================================================================
# parameter matching
# ================================================================================================================
@pytest.mark.parametrize("name", ["gcn", "gat", "egnn", "sccnn", "cw_net", "mgn", "mgn_fast", "deeponet", "fno"])
def test_param_matching_tolerance(name, tmp_path):
    fixture = "grid" if name == "fno" else "tri"
    td = tiny_task(FIXTURES[fixture]("cpu", None), "node")
    target = 60_000
    cache = str(tmp_path / "pm.json")
    model, info = registry.build_model(name, td, target, cache_path=cache)
    m = info["match"]
    assert m["matched"] and m["within_tol"], m
    assert target <= info["params"] <= 1.2 * target
    assert info["params"] == sum(p.numel() for p in model.parameters() if p.requires_grad)
    # cached, and the cache does not change the initialisation (matching never consumes the RNG)
    torch.manual_seed(1)
    model2, info2 = registry.build_model(name, td, target, cache_path=cache)
    torch.manual_seed(1)
    model3, _ = registry.build_model(name, td, None, cache_path=None, hidden=info["hidden"],
                                     **({"modes": m["option"]["modes"]} if name == "fno" else {}))
    assert info2["match"]["cached"] and info2["hidden"] == info["hidden"]
    for (k2, p2), (k3, p3) in zip(model2.state_dict().items(), model3.state_dict().items()):
        assert k2 == k3 and torch.equal(p2, p3), k2


def test_rhmp_param_count_matches_trainer_model():
    from rhmp.model import RHMP
    from rhmp.baselines.dec_fixed import rhmp_config
    from rhmp.train import parse_args
    td = tiny_task(FIXTURES["tri"]("cpu", None), "node")
    args = parse_args(["--task", "tiny"])
    n = registry.rhmp_param_count(td, args)
    assert n == RHMP(rhmp_config(td, args), td.geo_dims).num_parameters()


# ================================================================================================================
# fixed-metric controls
# ================================================================================================================
def test_dec_fixed_keeps_H_equal_star():
    from rhmp.baselines.dec_fixed import metric_is_star
    td = tiny_task(FIXTURES["tri"]("cpu", None), "node")
    ov = _small_overrides("dec_fixed")
    torch.manual_seed(0)
    fixed, _ = registry.build_model("dec_fixed", td, None, **ov)
    torch.manual_seed(0)
    learned, _ = registry.build_model("rhmp", td, None, **ov)
    assert fixed.num_parameters() < learned.num_parameters()
    K, xb, yb = batch_of(td, 3)
    opt = torch.optim.Adam([p for p in fixed.parameters() if p.requires_grad], 1e-2)
    for _ in range(3):
        opt.zero_grad()
        torch.nn.functional.mse_loss(fixed(xb, K), yb).backward()
        opt.step()
    assert metric_is_star(fixed, xb, K, atol=0.0)
    # the learned model moves away from the star after a few steps
    opt = torch.optim.Adam(learned.parameters(), 1e-2)
    for _ in range(3):
        opt.zero_grad()
        torch.nn.functional.mse_loss(learned(xb, K), yb).backward()
        opt.step()
    assert not metric_is_star(learned, xb, K, atol=1e-6)


def test_unit_variants_require_unit_star():
    td = tiny_task(FIXTURES["tri"]("cpu", None), "node")
    with pytest.raises(ValueError, match="star='unit'"):
        registry.build_model("unit_star", td, None, **_small_overrides("unit_star"))
    tdu = tiny_task(FIXTURES["tri"]("cpu", "unit"), "node")
    with pytest.raises(ValueError, match="unit"):
        registry.build_model("dec_fixed", tdu, None, **_small_overrides("dec_fixed"))
    m, info = registry.build_model("unit_fixed", tdu, None, **_small_overrides("unit_fixed"))
    assert info["metric"].startswith("frozen")


# ================================================================================================================
# applicability
# ================================================================================================================
def test_applicability_table():
    tri = tiny_task(FIXTURES["tri"]("cpu", None), "node")
    var = tiny_task(FIXTURES["batch"]("cpu", None), "node")
    tet = tiny_task(FIXTURES["tet"]("cpu", None), "node")
    quad = tiny_task(FIXTURES["quad"]("cpu", None), "node")
    face = tiny_task(FIXTURES["tri"]("cpu", None), "face")
    grid = tiny_task(FIXTURES["grid"]("cpu", None), "node")
    edge = tiny_task(FIXTURES["tri"]("cpu", None), "edge")
    expect_false = [("ours_v1", var), ("ours_v1", tet), ("ours_v1", quad), ("ours_v1", face), ("fno", tri),
                    ("fno", var), ("gauge_cnn", tet), ("gem_cnn", tet), ("deeponet", edge), ("deeponet", face)]
    for name, td in expect_false:
        ok, why = registry.applicable(name, td)
        assert not ok and why, (name, td.name)
    for name in ALL_MODELS:
        if name != "fno":
            assert registry.applicable(name, tri)[0], name
    assert registry.applicable("fno", grid)[0]
    with pytest.raises(ValueError, match="not applicable"):
        registry.build_model("ours_v1", var, None, C=8)


# ================================================================================================================
# trainer integration (rhmp.train --model)
# ================================================================================================================
def _trainer_has_model_flag() -> bool:
    from rhmp.train import parse_args
    try:
        return getattr(parse_args(["--task", "tiny", "--model", "mgn"]), "model", None) == "mgn"
    except SystemExit:
        return False


@pytest.mark.parametrize("name,target,tname", [("mgn", "node", "tiny"), ("gcn", "edge", "tiny"),
                                               ("dec_fixed", "node", "tiny"), ("cw_net", "face", "tiny"),
                                               ("ours_v1", "node", "tiny"), ("sccnn", "node", "T6")])
def test_trainer_runs_baseline(name, target, tname, tmp_path):
    if not _trainer_has_model_flag():
        pytest.skip("rhmp.train has no --model option")
    from rhmp.train import parse_args, run
    torch.set_num_threads(2)
    td = tiny_task(FIXTURES["tri"]("cpu", None), target, N=12, name=tname)
    opts = {"mgn": '{"hidden": 8, "n_layers": 2}', "gcn": '{"hidden": 8}', "cw_net": '{"hidden": 8}',
            "ours_v1": '{"C": 8}', "dec_fixed": "{}", "sccnn": '{"hidden": 8}'}[name]
    args = parse_args(["--task", tname, "--model", name, "--model-opts", opts, "--epochs", "2", "--out",
                       str(tmp_path), "--device", "cpu", "--C", "8", "--layers", "2", "--batch", "4", "--quiet"])
    res = run(args, task=td)
    assert res["complete"] and np.isfinite(res["test"]["R2"])
    assert res.get("model") == name
    if tname == "T6":                                             # v1 edge protocol: metrics over the edges
        assert "v1 edge protocol" in res["model_info"]["target_protocol"]
        assert res["test"]["N"] == len(td.split[2])
    for f in ("config.json", "history.json", "result.json", "best.pt", "last.pt"):
        assert os.path.exists(tmp_path / f), f
    # --eval-ckpt rebuilds the baseline from its checkpoint and reproduces the test metrics
    ev = run(parse_args(["--task", tname, "--eval-ckpt", str(tmp_path), "--out", str(tmp_path / "ev"), "--device",
                         "cpu", "--quiet", "--batch", "4"]),
             task=tiny_task(FIXTURES["tri"]("cpu", None), target, N=12, name=tname))
    for k in ("R2", "MSE"):
        assert abs(ev["test"][k] - res["test"][k]) < 1e-5, (k, ev["test"][k], res["test"][k])
