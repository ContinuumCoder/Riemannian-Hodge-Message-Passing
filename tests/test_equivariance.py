"""Symmetry tests: O(C) cochain-frame equivariance, E(n) invariance/equivariance, vertex relabelling (canonical
edge-orientation flips), face orientation flips, batch independence and exact gauge invariance.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from conftest import _MESHES, make_batch, make_complex, random_orthogonal, relabel_mesh
from rhmp import CochainComplex, RHMPConfig
from rhmp.readout import parse_readout, readout_degrees
from test_model import F64, build_model, make_inputs, rel, small_cfg


def _build(name: str, pos, cells, device) -> CochainComplex:
    """Complex of a conftest mesh family from explicit (possibly transformed / relabelled) arrays."""
    if name == "tets":
        return CochainComplex.from_tetrahedra(pos, cells, device=device)
    if name in ("quads", "mixed"):
        return CochainComplex.from_polygons(pos, cells, device=device)
    return CochainComplex.from_triangles(pos, cells, device=device)


def _readouts(top: int) -> list[str]:
    return ["node_scalar", "node_vector", "cochain:1", f"cochain:{top}", "even:1", f"even:{top}", "grad", "curl",
            "div", "div:1", "mdiv:1", "mdiv"]


def _out_info(readout: str, top: int) -> tuple[int, bool]:
    """(output degree, whether the output is an odd cochain)."""
    kind, _ = parse_readout(readout)
    _, odeg = readout_degrees(readout, top)
    return odeg, (odeg >= 1 and kind in ("cochain", "grad", "curl", "div", "mdiv"))


# ----------------------------------------------------------------------------------------------------------------
# O(C)
# ----------------------------------------------------------------------------------------------------------------
VARIANTS = {
    "default": {}, "untied": dict(tie_metrics=False), "jacobi": dict(scaling="jacobi"), "none": dict(scaling="none"),
    "poly3": dict(poly_order=3), "no_cross": dict(cross=False), "identity": dict(identity_metric=True),
    "reference": dict(fused=False), "resolvent": dict(layers=["resolvent", "poly"]),
    "resolvent_unrolled": dict(layers=["poly", "resolvent"], resolvent_grad="unrolled"),
    "resolvent_jacobi": dict(layers=["resolvent", "resolvent"], scaling="jacobi"),
    "tensor": dict(metric_type="tensor"), "tensor_none_p3": dict(metric_type="tensor", scaling="none", poly_order=3),
    # CG budgets large enough to converge on the sliver-heavy Delaunay fixture (exactness needs convergence)
    "solve": dict(layers=["poly", "solve"], solve_iters=4000, solve_tol=1e-12),
    "solve_tensor": dict(layers=["solve", "poly"], metric_type="tensor", solve_iters=4000, solve_tol=1e-12),
    "resolvent_phys": dict(layers=["resolvent", "poly"], resolvent_normalize=False, resolvent_iters=600),
    "solve_twolevel": dict(layers=["poly", "solve"], solve_precond="twolevel", solve_iters=600, solve_tol=1e-12),
}


def _check_OC(K, cfg, dtype, tol, seed=0):
    m = build_model(cfg, K, seed=seed, dtype=dtype)
    inp = make_inputs(cfg, K, B=2, seed=seed, dtype=dtype)
    x = m.lift(inp, K)
    for reflect in (False, True):
        Q = torch.as_tensor(random_orthogonal(cfg.C, seed=seed + 1, reflect=reflect), dtype=dtype, device=K.pos.device)
        a = m.propagate({k: v @ Q for k, v in x.items()}, K, inp)
        b = m.propagate(x, K, inp)
        for k in b:
            assert rel(a[k], b[k] @ Q) <= tol, (k, reflect, rel(a[k], b[k] @ Q))
            assert rel(b[k], x[k]) > 1e-3                                  # the layers did something


@pytest.mark.parametrize("name", ["grid", "sphere", "tets", "mixed"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["fp32", "fp64"])
def test_OC_equivariance(name, dtype, device):
    K = make_complex(name, device).to(dtype=dtype)
    cfg = small_cfg(dim=K.dim, n_layers=3)
    _check_OC(K, cfg, dtype, 1e-5 if dtype == torch.float32 else 1e-10)


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_OC_equivariance_variants(variant, device):
    K = make_complex("delaunay", device).to(dtype=F64)
    _check_OC(K, small_cfg(n_layers=2, **VARIANTS[variant]), F64, 1e-10)


def test_hidden_is_lift_then_propagate(device):
    K = make_complex("sphere", device)
    cfg = small_cfg()
    m = build_model(cfg, K)
    inp = make_inputs(cfg, K)
    h = m.hidden(inp, K)
    h2 = m.propagate(m.lift(inp, K), K, inp)
    assert all(rel(h[k], h2[k]) < 1e-6 for k in h)


# ----------------------------------------------------------------------------------------------------------------
# E(n)
# ----------------------------------------------------------------------------------------------------------------
def _apply_En(y: torch.Tensor, readout: str, R: torch.Tensor, D: int) -> torch.Tensor:
    """Expected transformation of an output: node vectors rotate (v -> R v), everything else is invariant."""
    if readout != "node_vector":
        return y
    return (y.view(*y.shape[:2], -1, D) @ R.T).reshape(y.shape)


@pytest.mark.parametrize("name", ["delaunay", "grid", "sphere", "torus", "tets", "mixed"])
@pytest.mark.parametrize("reflect", [False, True])
def test_En_invariance_and_equivariance(name, reflect, device):
    """End to end (fp32, complex rebuilt from transformed positions): rotations (+reflections) + translations
    leave scalar/cochain readouts invariant and rotate node vectors.  The vector tolerance reflects the
    conditioning of per-vertex least squares at hull slivers (fp32 rounding of the rebuilt positions)."""
    pos, cells = _MESHES[name]()
    D = pos.shape[1]
    R = random_orthogonal(D, seed=4, reflect=reflect)
    t = np.random.default_rng(5).normal(size=D) * 3.0
    K = _build(name, pos, cells, device)
    K2 = _build(name, pos @ R.T + t, cells, device)
    Rt = torch.as_tensor(R, dtype=torch.float32, device=device)
    for readout in _readouts(K.dim):
        cfg = small_cfg(dim=K.dim, readout=readout, out_dim=2)
        m = build_model(cfg, K)
        inp = make_inputs(cfg, K)
        y, y2 = _apply_En(m(inp, K), readout, Rt, D), m(inp, K2)
        tol = 1e-4 if readout == "node_vector" else 1e-5
        assert rel(y2, y) < tol, (readout, rel(y2, y))


@pytest.mark.parametrize("name", ["delaunay", "sphere", "tets"])
@pytest.mark.parametrize("reflect", [False, True])
@pytest.mark.parametrize("metric_type", ["diag", "tensor"])
def test_En_equivariance_exact_fp64(name, reflect, metric_type, device):
    """The model itself is exactly E(n)-equivariant: same complex, positions transformed exactly in fp64.

    Tensor metrics recompute the Galerkin element geometry (barycentric gradients) from the positions, so the fp64
    rounding of the rotated positions is amplified by the element conditioning (cond(Gram) ~ 2e7 on the slivers of
    the Delaunay fixture): tolerance 1e-9 there (diagonal metrics never touch the positions)."""
    K = make_complex(name, device).to(dtype=F64)
    D = K.pos.shape[1]
    R = torch.as_tensor(random_orthogonal(D, seed=6, reflect=reflect), dtype=F64, device=device)
    t = torch.randn(D, dtype=F64, generator=torch.Generator().manual_seed(8)).to(device) * 3.0
    K2 = dataclasses.replace(K, pos=K.pos @ R.T + t)
    if metric_type == "tensor":                                    # direction vectors (diagnostics) rotate too
        K2.whitney = {km: {**W, "t": W["t"] @ R.T} for km, W in K.whitney.items()}
    for readout in _readouts(K.dim):
        cfg = small_cfg(dim=K.dim, readout=readout, out_dim=2, metric_type=metric_type)
        m = build_model(cfg, K)
        inp = make_inputs(cfg, K, dtype=F64)
        y, y2 = _apply_En(m(inp, K), readout, R, D), m(inp, K2)
        assert rel(y2, y) < (1e-9 if metric_type == "tensor" else 1e-10), (readout, rel(y2, y))


# ----------------------------------------------------------------------------------------------------------------
# relabelling and orientation
# ----------------------------------------------------------------------------------------------------------------
def _orientation(r: tuple, c2: tuple, simplex: bool) -> int:
    """Sign relating the relabelled old cell ``r`` to the stored new cell ``c2`` (same vertex set)."""
    if simplex:
        p = [r.index(v) for v in c2]
        inv = sum(1 for i in range(len(p)) for j in range(i + 1, len(p)) if p[i] > p[j])
        return -1 if inv % 2 else 1
    n, j = len(r), c2.index(r[0])
    if all(c2[(j + i) % n] == r[i] for i in range(n)):
        return 1
    if all(c2[(j - i) % n] == r[i] for i in range(n)):
        return -1
    raise AssertionError("polygon vertex orders are not cyclically related")


def _cell_maps(K_old, K_new, inv: np.ndarray) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """For k >= 1: index in ``K_new`` of every old k-cell and the relative orientation (+-1)."""
    maps = {}
    for k in range(1, K_old.dim + 1):
        lookup = {}
        for i, c in enumerate(K_new.cells[k].cpu().numpy()):
            c = tuple(int(v) for v in c if v >= 0)
            lookup[tuple(sorted(c))] = (i, c)
        idx, sign = [], []
        for c in K_old.cells[k].cpu().numpy():
            r = tuple(int(inv[v]) for v in c if v >= 0)
            i, c2 = lookup[tuple(sorted(r))]
            idx.append(i)
            sign.append(_orientation(r, c2, simplex=(K_old.dim == 3 or len(r) <= 3)))
        dev = K_old.pos.device
        maps[k] = (torch.tensor(idx, device=dev), torch.tensor(sign, dtype=torch.float32, device=dev))
    return maps


def _map_cells(t: torch.Tensor, idx: torch.Tensor, sign: torch.Tensor, n_odd: int) -> torch.Tensor:
    """Move old per-cell values ``(n, B, F)`` to the new indexing; the first ``n_odd`` columns flip with the sign."""
    col = torch.ones(t.shape[-1], device=t.device, dtype=t.dtype)
    col[n_odd:] = 0
    s = 1.0 + (sign.view(-1, 1, 1).to(t.dtype) - 1.0) * col
    out = torch.empty_like(t)
    out[idx] = t * s
    return out


@pytest.mark.parametrize("name,metric_type", [("delaunay", "diag"), ("sphere", "diag"), ("tets", "diag"),
                                              ("mixed", "diag"), ("nonmanifold", "diag"), ("delaunay", "tensor"),
                                              ("sphere", "tensor"), ("tets", "tensor")])
def test_vertex_relabelling(name, metric_type, device):
    """Random vertex permutation, cells re-indexed, complex rebuilt (canonical edge orientations change):
    outputs are permuted, odd outputs pick up the orientation signs."""
    pos, cells = _MESHES[name]()
    pos2, cells2, perm = relabel_mesh(pos, cells, seed=1)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size)
    K, K2 = _build(name, pos, cells, device), _build(name, pos2, cells2, device)
    maps = _cell_maps(K, K2, inv)
    assert any(bool((s < 0).any()) for _, s in maps.values())            # some orientations really flip
    permt = torch.as_tensor(perm, device=device)
    for readout in _readouts(K.dim):
        cfg = small_cfg(dim=K.dim, readout=readout, out_dim=2, metric_type=metric_type)
        m = build_model(cfg, K)
        inp = make_inputs(cfg, K)
        inp2 = {0: inp[0][permt]}
        for k, v in inp.items():
            if k > 0:
                inp2[k] = _map_cells(v, *maps[k], n_odd=cfg.in_dims[k] - cfg.even_dims.get(k, 0))
        y, y2 = m(inp, K), m(inp2, K2)
        odeg, odd = _out_info(readout, K.dim)
        expect = y[permt] if odeg == 0 else _map_cells(y, *maps[odeg], n_odd=(y.shape[-1] if odd else 0))
        assert rel(y2, expect) < 1e-5, (readout, rel(y2, expect))


@pytest.mark.parametrize("name,metric_type", [("grid", "diag"), ("sphere", "diag"), ("mixed", "diag"),
                                              ("grid", "tensor"), ("sphere", "tensor")])
def test_face_orientation_flip(name, metric_type, device):
    """Reversing the orientation of some faces flips the sign of odd face inputs/outputs and nothing else."""
    pos, faces = _MESHES[name]()
    flip = np.random.default_rng(7).random(len(faces)) < 0.4
    faces2 = faces.copy()
    for f in np.nonzero(flip)[0]:
        valid = faces[f][faces[f] >= 0]
        faces2[f, :len(valid)] = valid[::-1]
    K, K2 = _build(name, pos, faces, device), _build(name, pos, faces2, device)
    sign = torch.as_tensor(np.where(flip, -1.0, 1.0), dtype=torch.float32, device=device)
    idx = torch.arange(len(faces), device=device)
    for readout in _readouts(2):
        cfg = small_cfg(readout=readout, out_dim=2, metric_type=metric_type)
        m = build_model(cfg, K)
        inp = make_inputs(cfg, K)
        inp2 = {**inp, 2: _map_cells(inp[2], idx, sign, n_odd=cfg.in_dims[2] - cfg.even_dims.get(2, 0))}
        y, y2 = m(inp, K), m(inp2, K2)
        odeg, odd = _out_info(readout, 2)
        expect = _map_cells(y, idx, sign, n_odd=y.shape[-1]) if (odeg == 2 and odd) else y
        assert rel(y2, expect) < 1e-5, (readout, rel(y2, expect))


# ----------------------------------------------------------------------------------------------------------------
# batch independence
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("readout", ["node_scalar", "node_vector", "cochain:1", "even:2", "grad", "div"])
def test_batch_independence_shared_mesh(readout, device):
    K = make_complex("delaunay", device)
    cfg = small_cfg(readout=readout, n_layers=3)
    m = build_model(cfg, K)
    inp = make_inputs(cfg, K, B=64, seed=0)
    other = make_inputs(cfg, K, B=64, seed=1)
    alone = m({k: v[:, :1].contiguous() for k, v in inp.items()}, K)
    y64 = m(inp, K)
    mixed = m({k: torch.cat([inp[k][:, :1], other[k][:, 1:]], 1) for k in inp}, K)
    assert rel(y64[:, :1], alone) < 1e-6
    assert rel(mixed[:, :1], alone) < 1e-6
    assert rel(y64[:, 1:2], alone) > 1e-3                                  # different samples differ


@pytest.mark.parametrize("readout", ["node_scalar", "node_vector", "cochain:1", "cochain:2", "grad", "curl",
                                     "div:1"])
def test_batch_independence_block_diagonal(readout, device):
    Kb, parts = make_batch(device)
    cfg = small_cfg(readout=readout, n_layers=3)
    m = build_model(cfg, parts[0])
    ins = [make_inputs(cfg, P, B=1, seed=i) for i, P in enumerate(parts)]
    inb = {k: torch.cat([x[k] for x in ins], 0) for k in ins[0]}
    yb = m(inb, Kb)
    ptr = Kb.meta["ptr"][m.output_degree].tolist()
    for i, (P, x) in enumerate(zip(parts, ins)):
        yi = m(x, P)
        assert rel(yb[ptr[i]:ptr[i + 1]], yi) < 1e-6, (i, rel(yb[ptr[i]:ptr[i + 1]], yi))


# ----------------------------------------------------------------------------------------------------------------
# gauge invariance
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("readout", ["node_scalar", "cochain:1", "cochain:2", "node_vector", "curl", "grad", "div"])
def test_gauge_invariance_edge_connection(readout, device):
    """A -> A + d_0 lambda leaves every output unchanged (A enters only through d_1 A)."""
    K = make_complex("sphere", device)
    cfg = RHMPConfig(in_dims={0: 1, 1: 2}, even_dims={0: 1}, connection_dims={1: 1}, C=8, n_layers=3, readout=readout)
    m = build_model(cfg, K)
    inp = make_inputs(cfg, K, B=2)
    lam = 3.0 * torch.randn(K.n[0], 2, 1, generator=torch.Generator().manual_seed(9)).to(device)
    inp2 = {**inp, 1: inp[1].clone()}
    inp2[1][..., :1] += K.apply_d(0, lam)
    y, y2 = m(inp, K), m(inp2, K)
    assert rel(y2, y) < 1e-5
    m_odd = build_model(dataclasses.replace(cfg, connection_odd=True), K)   # non-Abelian path: not invariant
    assert rel(m_odd(inp2, K), m_odd(inp, K)) > 1e-3
    inp3 = {**inp, 1: inp[1].clone()}
    inp3[1][..., :1] += torch.randn_like(inp[1][..., :1])                  # a non-exact change is seen
    assert rel(m(inp3, K), y) > 1e-3


def test_gauge_invariance_face_connection_tets(device):
    """On tetrahedra a face connection B -> B + d_1 mu is invisible (only d_2 B enters)."""
    K = make_complex("tets", device)
    cfg = RHMPConfig(in_dims={0: 1, 2: 2}, connection_dims={2: 1}, C=8, n_layers=2, readout="cochain:3")
    m = build_model(cfg, K)
    inp = make_inputs(cfg, K, B=2)
    mu = torch.randn(K.n[1], 2, 1, generator=torch.Generator().manual_seed(3)).to(device)
    inp2 = {**inp, 2: inp[2].clone()}
    inp2[2][..., :1] += K.apply_d(1, mu)
    assert rel(m(inp2, K), m(inp, K)) < 1e-5


@pytest.mark.parametrize("layers", [["resolvent", "poly"], ["poly", "resolvent", "resolvent"]])
def test_batch_independence_resolvent(layers, device):
    """Resolvent layers use per-sample (per-graph) CG scalars: shared-mesh batches and block-diagonal batches give
    the same per-sample result."""
    K = make_complex("delaunay", device)
    cfg = small_cfg(layers=layers, readout="cochain:1")
    m = build_model(cfg, K)
    inp = make_inputs(cfg, K, B=16, seed=0)
    alone = m({k: v[:, :1].contiguous() for k, v in inp.items()}, K)
    assert rel(m(inp, K)[:, :1], alone) < 1e-6
    Kb, parts = make_batch(device)
    ins = [make_inputs(cfg, P, B=1, seed=i) for i, P in enumerate(parts)]
    yb = m({k: torch.cat([x[k] for x in ins], 0) for k in ins[0]}, Kb)
    ptr = Kb.meta["ptr"][m.output_degree].tolist()
    for i, (P, x) in enumerate(zip(parts, ins)):
        assert rel(yb[ptr[i]:ptr[i + 1]], m(x, P)) < 1e-6


@pytest.mark.parametrize("name", ["delaunay", "tets"])
def test_batch_independence_tensor(name, device):
    """Tensor metrics are per sample and per graph: B=1 vs B=16 vs block-diagonal batch."""
    K = make_complex(name, device)
    cfg = small_cfg(dim=K.dim, metric_type="tensor", readout="cochain:1")
    m = build_model(cfg, K)
    inp = make_inputs(cfg, K, B=16, seed=0)
    alone = m({k: v[:, :1].contiguous() for k, v in inp.items()}, K)
    assert rel(m(inp, K)[:, :1], alone) < 1e-6
    if name == "delaunay":
        Kb, parts = make_batch(device)
        ins = [make_inputs(cfg, P, B=1, seed=i) for i, P in enumerate(parts)]
        yb = m({k: torch.cat([x[k] for x in ins], 0) for k in ins[0]}, Kb)
        ptr = Kb.meta["ptr"][m.output_degree].tolist()
        for i, (P, x) in enumerate(zip(parts, ins)):
            assert rel(yb[ptr[i]:ptr[i + 1]], m(x, P)) < 1e-6


def test_OC_equivariance_tensor_tets(device):
    K = make_complex("tets", device).to(dtype=F64)
    _check_OC(K, small_cfg(dim=3, n_layers=2, metric_type="tensor"), F64, 1e-10)


def test_latent_field_is_an_odd_cochain_input(device):
    """A latent edge field is applied like an odd edge input: relabelling the mesh together with the latent
    (permuted, orientation signs applied) permutes / sign-maps the outputs consistently."""
    pos, cells = _MESHES["delaunay"]()
    pos2, cells2, perm = relabel_mesh(pos, cells, seed=1)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size)
    K, K2 = _build("delaunay", pos, cells, device), _build("delaunay", pos2, cells2, device)
    maps = _cell_maps(K, K2, inv)
    permt = torch.as_tensor(perm, device=device)
    for readout in ("node_scalar", "cochain:1", "cochain:2"):
        cfg = small_cfg(latent_dims={0: 1, 1: 3, 2: 2}, readout=readout)
        m = build_model(cfg, K).init_latents(K)
        m2 = build_model(cfg, K2, randomize=False)
        m2.load_state_dict(m.state_dict())                                  # same weights ...
        with torch.no_grad():                                               # ... and the latent moved to K2's labels
            m2.latent["0"].copy_(m.latent["0"][permt])
            for k in (1, 2):
                m2.latent[str(k)].copy_(_map_cells(m.latent[str(k)].unsqueeze(1), *maps[k], n_odd=99).squeeze(1))
        inp = make_inputs(cfg, K)
        inp2 = {0: inp[0][permt]}
        for k, v in inp.items():
            if k > 0:
                inp2[k] = _map_cells(v, *maps[k], n_odd=cfg.in_dims[k] - cfg.even_dims.get(k, 0))
        y, y2 = m(inp, K), m2(inp2, K2)
        odeg, odd = _out_info(readout, K.dim)
        expect = y[permt] if odeg == 0 else _map_cells(y, *maps[odeg], n_odd=(y.shape[-1] if odd else 0))
        assert rel(y2, expect) < 1e-5, (readout, rel(y2, expect))
