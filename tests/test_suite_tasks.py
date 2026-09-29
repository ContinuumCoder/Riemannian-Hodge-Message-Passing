"""Tests of the extension task suite (SURF / DYN / QUAL generators, loaders and rollout evaluation).

Tiny generations on the CPU:
* surfaces: every family is a closed, consistently oriented surface of the right genus, with no degenerate faces,
  ``d^2 = 0`` and no dropped faces in ``CochainComplex``; the DEC operators of the generators agree with the complex
  (``star0`` = lumped mass, unclamped ``star1`` = cotan weights, ``L = d0^T diag(w) d0``);
* solvers: relative residuals < 1e-8, constants preserved, heat flow mass-conserving; DYN fluxes exactly
  divergence-free, advection skew and conservative, per-step mass residual at round-off; the QUAL solver equals
  ``gen_HP.solve_sample`` on the HP mesh;
* loaders: split sizes, shapes, degrees, extra test sets, test-only views; ``rollout_eval`` bookkeeping checked with
  an oracle model (R2 = 1 at every horizon) and with a small RHMP; one trainer epoch on tiny SURF / DYN data.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

pytest.importorskip("scipy.sparse.linalg")
pytest.importorskip("scipy.spatial")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, "datasets", "generators")
for p in (ROOT, GEN):
    if p not in sys.path:
        sys.path.insert(0, p)

import gen_dyn  # noqa: E402
import gen_HP  # noqa: E402
import gen_qual  # noqa: E402
import gen_surf  # noqa: E402
import surfaces as S  # noqa: E402
from rhmp.complex import CochainComplex  # noqa: E402
from rhmp.data import edge_alignment  # noqa: E402

H_TEST = 0.2   # coarse target edge length: meshes of a few hundred vertices


def _cfg(**kw):
    return gen_surf.default_config(h=H_TEST, **kw)


# ======================================================================================================================
# surfaces
# ======================================================================================================================
@pytest.mark.parametrize("family", gen_surf.FAMILIES)
def test_surface_family_valid(family):
    rng = np.random.default_rng([1, gen_surf.FAMILIES.index(family)])
    for _ in range(2):
        pts, faces, params = gen_surf.make_surface(family, rng, _cfg())
        rep = S.mesh_report(pts, faces)
        S.assert_valid_closed(rep, gen_surf.GENUS[family], family)
        K = CochainComplex.from_triangles(pts, faces, star="cotan", device="cpu")
        assert K.check_d2() == 0.0
        assert K.n[2] == len(faces) and K.n[0] == len(pts)
        assert K.n[0] - K.n[1] + K.n[2] == 2 - 2 * gen_surf.GENUS[family]
        assert not bool(K.boundary[1].any())                       # closed surface
        assert abs(rep["edge_median"] / H_TEST - 1) < 0.5


def test_cube_sphere_counts_and_orientation():
    for n in (3, 5):
        pts, faces = S.cube_sphere(n, np.random.default_rng(n))
        assert len(pts) == 6 * n * n + 2 and len(faces) == 12 * n * n
        np.testing.assert_allclose(np.linalg.norm(pts, axis=1), 1.0, atol=1e-12)
        rep = S.mesh_report(pts, faces)
        S.assert_valid_closed(rep, 0)


def test_double_torus_genus_and_neck():
    rng = np.random.default_rng(3)
    pts, faces, info = S.double_torus_mesh(0.8, 0.35, 0.3, 0.08, 3, rng)
    rep = S.mesh_report(pts, faces)
    S.assert_valid_closed(rep, 2)
    assert info["loop_len"] == 24 and info["neck_rings"] >= 1
    assert rep["min_angle"] > 10.0


@pytest.mark.parametrize("family", ["ellipsoid", "torus"])
def test_cotan_operators_match_complex(family):
    rng = np.random.default_rng(5)
    pts, faces, _ = gen_surf.make_surface(family, rng, _cfg())
    ops = S.cotan_operators(pts, faces)
    K = CochainComplex.from_triangles(torch.as_tensor(pts), torch.as_tensor(faces), star="cotan", device="cpu")
    # vertex order is kept: star0 = barycentric lumped mass
    np.testing.assert_allclose(K.star[0].double().numpy(), ops["mass"], rtol=1e-5)
    perm, sign = edge_alignment(K.cells[1], torch.as_tensor(ops["edges"]), len(pts))
    assert bool((sign == 1).all())
    w = ops["w"][perm.numpy()]
    s1 = K.star[1].double().numpy()
    unclamped = s1 > 1.01 * np.median(s1) * 1e-2
    np.testing.assert_allclose(s1[unclamped], w[unclamped], rtol=1e-4, atol=1e-6)
    # L = d0^T diag(w) d0 with the complex's d0 (edge order of the complex)
    d0 = K.d[0].to_dense().double().numpy()
    L = d0.T @ np.diag(w) @ d0
    np.testing.assert_allclose(L, ops["L"].toarray(), atol=1e-9)
    np.testing.assert_allclose(ops["L"] @ np.ones(len(pts)), 0.0, atol=1e-10)


@pytest.mark.parametrize("family", gen_surf.FAMILIES)
def test_surface_sample_solvers(family):
    s = gen_surf.make_sample((0, family, 11, _cfg()))
    st = s["stats"]
    assert st["res_u"] < 1e-8 and st["res_heat"] < 1e-8
    assert st["heat_mass_drift"] < 1e-10
    for k in ("f", "u", "heat"):
        assert s[k].shape == (st["n0"],) and np.isfinite(s[k]).all()
    assert s["pos"].dtype == np.float32 and s["faces"].dtype == np.int32
    # constants are preserved by both operators (L 1 = 0)
    pts, faces = s["pos"].astype(np.float64), s["faces"].astype(np.int64)
    sol = gen_surf.solve_surface(pts, faces, np.ones(len(pts)), 0.05, 0.05, 10)
    np.testing.assert_allclose(sol["u"], 1.0, atol=1e-10)
    np.testing.assert_allclose(sol["heat"], 1.0, atol=1e-10)
    # the stored (float32) mesh reproduces the solve (positions were rounded before solving)
    sol2 = gen_surf.solve_surface(pts, faces, s["f"].astype(np.float64), 0.05, 0.05, 10)
    np.testing.assert_allclose(sol2["u"], s["u"], rtol=1e-5, atol=1e-5)


# ======================================================================================================================
# DYN
# ======================================================================================================================
def _dyn_mesh(n=300, seed=0):
    return gen_HP.draw_mesh_2d(seed, 0, n)


def test_dyn_fluxes_divergence_free_and_skew():
    pts, faces = _dyn_mesh()
    cfg = gen_dyn.default_config()
    p = gen_dyn.draw_stream(np.random.default_rng(0), cfg)
    ops = S.cotan_operators(pts, faces)
    edges, n = ops["edges"], len(pts)
    Phi = gen_dyn.dual_fluxes(p, pts, faces, edges)
    div = np.zeros(n)
    np.add.at(div, edges[:, 0], Phi)
    np.add.at(div, edges[:, 1], -Phi)
    assert np.abs(div).max() < 1e-12 * np.abs(Phi).max()
    A = gen_dyn.advection_matrix(Phi, edges, n).toarray()
    np.testing.assert_allclose(A.sum(0), 0.0, atol=1e-14)            # 1^T A = 0: conservation
    np.testing.assert_allclose(A.sum(1), 0.0, atol=1e-12)            # A 1 = div Phi = 0: constants are steady
    np.testing.assert_allclose(A + A.T, np.diag(np.diag(A + A.T)), atol=1e-14)   # skew off the diagonal
    Au = gen_dyn.advection_matrix(Phi, edges, n, upwind=1.0).toarray()
    np.testing.assert_allclose(Au.sum(0), 0.0, atol=1e-14)
    assert np.linalg.eigvalsh(0.5 * (Au + Au.T)).min() > -1e-12       # dissipative
    # the edge 1-form is the trapezoidal line integral of v
    v = gen_dyn.velocity(p, pts)
    th = gen_dyn.edge_one_form(v, pts, edges)
    i, j = edges[0]
    assert abs(th[0] - 0.5 * (v[i] + v[j]) @ (pts[j] - pts[i])) < 1e-14


def test_dyn_simulation_conserves_mass_and_constants():
    pts, faces = _dyn_mesh()
    cfg = gen_dyn.default_config(steps=4, substeps=3)
    p = gen_dyn.draw_stream(np.random.default_rng(1), cfg)
    u0 = gen_dyn.blobs(np.random.default_rng(2), pts, cfg)
    sim = gen_dyn.simulate(pts, faces, p, u0, cfg)
    assert sim["traj"].shape == (5, len(pts))
    assert sim["mass_res"] < 1e-12 and sim["mass_drift"] < 1e-12 and sim["div_rel"] < 1e-12
    const = gen_dyn.simulate(pts, faces, p, np.full(len(pts), 2.0), cfg)
    np.testing.assert_allclose(const["traj"], 2.0, atol=1e-11)
    t = gen_dyn.make_trajectory((0, 300, 3, cfg, None))
    assert t["traj"].shape == (300, 5) and t["theta"].shape == (t["stats"]["n1"],)
    assert t["stats"]["mass_res"] < 1e-12


# ======================================================================================================================
# QUAL
# ======================================================================================================================
def test_qual_solver_matches_gen_HP():
    idx, n_tot = 3, 300
    ref = gen_HP.solve_sample((idx, n_tot, 2026, 100.0, False, None))
    F = gen_HP.draw_fields(2026, idx, 2, False)
    pts, faces = gen_HP.draw_mesh_2d(2026, idx, n_tot)
    mine = gen_qual.solve_on_mesh(F, pts, faces, 100.0)
    for k in ("f", "u", "sigma_edge", "logsigma_face", "flux", "flux_fem"):
        np.testing.assert_array_equal(ref[k], mine[k].astype(np.float32))
    np.testing.assert_array_equal(ref["bnd"], mine["bnd"])
    assert mine["residual"] < 1e-8


def test_qual_levels_valid():
    levels = ("base", "corner_r16", "circle_a30", "a4")
    out = gen_qual.make_base((7, 300, 2026, 100.0, levels, 4, 5))
    assert [s["stats"]["level"] for s in out] == list(levels)
    base_pos = out[0]["pos"]
    for s in out:
        pts, faces = s["pos"].astype(np.float64), s["faces"].astype(np.int64)
        P = pts[faces]
        sa = 0.5 * ((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) - (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0]))
        assert (sa > 0).all() and abs(sa.sum() - 1.0) < 1e-6          # CCW, covers the unit square
        assert len(np.unique(faces)) == len(pts)
        K = CochainComplex.from_triangles(pts, faces, device="cpu")
        assert K.n[2] == len(faces) and K.check_d2() == 0.0
        st = s["stats"]
        assert st["residual"] < 1e-8 and st["conservation"] < 1e-8
        assert np.isfinite(s["u_ref"]).all() and (s["u_ref"][s["bnd"]] == 0).all()
    assert np.array_equal(out[3]["pos"], base_pos)                   # slivers: same nodes, new connectivity
    assert out[3]["stats"]["aspect_p99"] > out[0]["stats"]["aspect_p99"]
    assert out[0]["stats"]["rel_err_vs_ref"] < 0.1


# ======================================================================================================================
# end to end: tiny data sets -> loaders -> rollouts / trainer
# ======================================================================================================================
@pytest.fixture(scope="module")
def tiny_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("suite")
    out = str(root / "datasets" / "v2")
    gen_surf.main(["--n", "9", "--n-geo", "3", "--n-topo", "3", "--h", str(H_TEST), "--workers", "1",
                   "--out-dir", out])
    gen_dyn.main(["--n", "10", "--n-fix", "10", "--steps", "6", "--substeps", "2", "--nmin", "200", "--nmax", "260",
                  "--n-fix-nodes", "230", "--workers", "1", "--out-dir", out])
    gen_qual.main(["--n-base", "2", "--hp-n", "20", "--hp-nmin", "200", "--hp-nmax", "260", "--workers", "1",
                   "--out-dir", out])
    return str(root)


def _suite():
    from rhmp.tasks.suite import SUITE_TASKS
    return SUITE_TASKS


def test_generated_files(tiny_root):
    v2 = os.path.join(tiny_root, "datasets", "v2")
    for name in ("SURF", "SURF_geo", "SURF_topo", "DYN", "DYNfix", "HP_qual_graded", "HP_qual_sliver"):
        assert os.path.exists(os.path.join(v2, name + ".pt")) and os.path.exists(os.path.join(v2, name + ".json"))
    blob = torch.load(os.path.join(v2, "SURF.pt"), weights_only=False)
    assert blob["family"].tolist() == [0, 2, 3] * 3 and blob["genus"].tolist() == [0, 0, 1] * 3
    assert max(s["res_u"] for s in blob["sample_stats"]) < 1e-8
    topo = torch.load(os.path.join(v2, "SURF_topo.pt"), weights_only=False)
    assert (topo["genus"] == 2).all()
    d = torch.load(os.path.join(v2, "DYN.pt"), weights_only=False)
    assert d["traj"].shape[1] == 7 and max(s["mass_res"] for s in d["sample_stats"]) < 1e-12


def test_surf_loader(tiny_root):
    T = _suite()
    td = T["SURF"](tiny_root, device="cpu")
    assert td.variable_mesh and td.num_samples == 9 and [len(s) for s in td.split] == [6, 1, 2]
    assert td.in_dims == {0: 1} and td.even_dims == {} and td.readout == "node_scalar" and td.spatial_dim == 3
    for K, x, y in zip(td.K, td.inputs, td.target):
        assert x[0].shape == (K.n[0], 1) and y.shape == (K.n[0], 1) and K.check_d2() == 0.0
    fams = {gen_surf.MAIN_FAMILIES[i % 3] for i in td.split[2].tolist()}      # test ids 7, 8 of 9
    assert set(td.extra_tests) == {"geo", "topo"} | {f"test_{f}" for f in fams}
    assert len(td.extra_tests["topo"]["K"]) == 3
    tr = torch.cat([td.target[i] for i in td.split[0].tolist()])
    assert abs(float(tr.mean())) < 1e-5 and abs(float(tr.std()) - 1) < 1e-4
    th = T["SURF_heat"](tiny_root, device="cpu")
    assert not torch.allclose(th.target[0], td.target[0])
    tt = T["SURF_topo"](tiny_root, device="cpu")
    assert [len(s) for s in tt.split] == [0, 0, 3] and tt.meta["test_only"]
    small = T["SURF"](tiny_root, device="cpu", max_samples=3, fine=False)
    assert [len(s) for s in small.split] == [1, 1, 1] and "geo" not in small.extra_tests


class _Oracle(torch.nn.Module):
    """Returns the normalised true next state of the rollout trajectories (checks rollout_eval's bookkeeping)."""

    def __init__(self, task):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.task, self.t = task, 0

    def forward(self, inputs, K):
        roll, ys = self.task.rollout, self.task.y_stats
        self.t += 1
        mode = roll.get("target", "state")
        if roll["shared"]:
            nxt = roll["traj"][:, self.t].T                              # (n0, B)
            prv = roll["traj"][:, self.t - 1].T
            mass = roll["mass"][:, None]
        else:
            nxt = torch.cat([tr[self.t] for tr in roll["traj"]])[:, None]
            prv = torch.cat([tr[self.t - 1] for tr in roll["traj"]])[:, None]
            mass = torch.cat(roll["mass"])[:, None]
        y = {"state": nxt, "delta": nxt - prv, "cons": nxt - prv, "cons_mass": mass * (nxt - prv)}[mode]
        y = ys.normalize(y[..., None])
        if self.task.output_map is not None:                           # DYNfix_cons: map(z) = (mean M / M) z
            y = y * (mass / mass.mean())[..., None]
        return y


@pytest.mark.parametrize("name", ["DYN", "DYNfix", "DYN_delta", "DYNfix_delta", "DYNfix_cons", "DYN_cons_mass",
                                  "DYNfix_cons_mass"])
def test_dyn_loaders_and_oracle_rollout(tiny_root, name):
    from rhmp.tasks.suite import rollout_eval
    kw = {"cons_param": "div"} if name == "DYNfix_cons" else {}       # the oracle inverts the diagonal map
    td = _suite()[name](tiny_root, device="cpu", windows=3, **kw)
    assert td.in_dims == {0: 1, 1: 1} and td.even_dims == {} and td.connection_dims == {}
    assert [len(s) for s in td.split] == [7 * 3, 1 * 3, 2 * 3]
    if td.variable_mesh:
        K, x = td.K[0], td.inputs[0]
        assert x[0].shape == (K.n[0], 1) and x[1].shape == (K.n[1], 1) and td.target[0].shape == (K.n[0], 1)
        assert len(td.rollout["traj"]) == 2
    else:
        assert td.inputs[0].shape == (21 + 3 + 6, td.K.n[0], 1) and td.inputs[1].shape == (30, td.K.n[1], 1)
        assert td.rollout["traj"].shape[:2] == (2, 7)
    res = rollout_eval(_Oracle(td), td, device="cpu")
    assert res["steps"] == 6 and len(res["R2"]) == 6
    assert min(res["R2"]) > 1 - 1e-5, res["R2"]
    assert max(res["mass_drift_max"]) < 1e-5 and res["true_mass_drift_max"] < 1e-5
    assert res["first_nonfinite_step"] is None


def test_dyn_legacy_and_rhmp_rollout(tiny_root):
    from rhmp.model import RHMP, RHMPConfig
    from rhmp.tasks.suite import rollout_eval
    for name in ("DYN", "DYNfix"):
        td = _suite()[name](tiny_root, device="cpu", windows=3)
        cfg = RHMPConfig(in_dims=td.in_dims, even_dims=td.even_dims, C=8, n_layers=1, readout="node_scalar")
        model = RHMP(cfg, td.geo_dims)
        res = rollout_eval(model, td, steps=4, device="cpu", batch_size=1)
        assert len(res["R2"]) == 4 and np.isfinite(res["persistence_R2"]).all()
        leg = _suite()[name](tiny_root, device="cpu", windows=3, native=False)
        assert leg.in_dims == {0: 3}
        cfg = RHMPConfig(in_dims=leg.in_dims, C=8, n_layers=1, readout="node_scalar")
        res = rollout_eval(RHMP(cfg, leg.geo_dims), leg, steps=2, device="cpu")
        assert len(res["NRMSE"]) == 2


@pytest.mark.parametrize("name", ["HP_qual_graded", "HP_qual_sliver_ref"])
def test_qual_loader(tiny_root, name):
    td = _suite()[name](tiny_root, device="cpu")
    levels = td.meta["levels"]
    assert [len(s) for s in td.split] == [0, 0, 2 * len(levels)]
    assert td.in_dims == {0: 1, 1: 1, 2: 1} and td.even_dims == {1: 1, 2: 1}
    assert set(td.extra_tests) == set(levels) and all(len(v["K"]) == 2 for v in td.extra_tests.values())
    for K, x in zip(td.K, td.inputs):
        assert x[0].shape == (K.n[0], 1) and x[1].shape == (K.n[1], 1) and x[2].shape == (K.n[2], 1)


def test_trainer_one_epoch_on_suite_tasks(tiny_root, tmp_path):
    from rhmp.train import parse_args, run
    for name in ("SURF", "DYN", "DYNfix_cons"):
        td = _suite()[name](tiny_root, device="cpu", **({"windows": 3} if name.startswith("DYN") else {}))
        args = parse_args(["--task", name, "--epochs", "1", "--device", "cpu", "--C", "8", "--layers", "1",
                           "--out", str(tmp_path / name), "--quiet"])
        res = run(args, task=td)
        assert res["complete"] and np.isfinite(res["test"]["R2"])
        if name == "SURF":
            assert "topo" in res and "geo" in res


def _same(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        if a.layout == torch.sparse_csr:
            return all(torch.equal(x.cpu(), y.cpu()) for x, y in ((a.crow_indices(), b.crow_indices()),
                                                                  (a.col_indices(), b.col_indices()),
                                                                  (a.values(), b.values()))) and a.shape == b.shape
        return a.device == b.device and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b if isinstance(a, (int, float, str, bool, type(None))) else True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_fast_complex_transfer_matches_to():
    import dataclasses
    from rhmp.tasks.suite import _complexes_to, _to_views
    pts, faces = S.cube_sphere(4, np.random.default_rng(0))
    K = CochainComplex.from_triangles(pts, faces, device="cpu")
    ref = K.to("cuda")
    fast = _complexes_to([K, K], "cuda")
    assert fast[0] is fast[1]
    for f in dataclasses.fields(K):
        assert _same(getattr(ref, f.name), getattr(fast[0], f.name)), f.name
    xs = [torch.randn(n, 1) for n in (3, 5, 2)]
    ys = _to_views(xs, "cuda")
    assert all(torch.equal(x, y.cpu()) and y.is_cuda for x, y in zip(xs, ys))


def test_surf_integral_error(tiny_root):
    from rhmp.model import RHMP, RHMPConfig
    from rhmp.tasks.suite import surf_integral_error
    td = _suite()["SURF"](tiny_root, device="cpu")

    class Exact(torch.nn.Module):          # returns the stored target: integral error = target error (round-off)
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))
            self.calls = []

        def forward(self, inputs, K):
            return self.out.pop(0)

    m = Exact()
    te = td.split[2].tolist()
    m.out = [torch.cat([td.target[i] for i in te])[:, None]]
    r = surf_integral_error(m, td, batch_size=len(te), device="cpu")
    assert r["N"] == len(te) and r["max"] < 1e-5 and r["target_max"] < 1e-5
    model = RHMP(RHMPConfig(in_dims={0: 1}, C=8, n_layers=1), td.geo_dims)
    r = surf_integral_error(model, td, source=td.extra_tests["topo"], device="cpu")
    assert r["N"] == 3 and np.isfinite(r["mean"])


def test_mass_weighted_errors(tiny_root):
    from rhmp.tasks.suite import mass_weighted_errors
    td = _suite()["HP_qual_graded"](tiny_root, device="cpu")
    idx = td.split[2]
    exact = [td.target[i] for i in idx.tolist()]
    r = mass_weighted_errors(td, exact)
    assert r["relL2M_pooled"] < 1e-6 and r["relL2M_mean"] < 1e-6
    noisy = [t + 0.1 for t in exact]
    r2 = mass_weighted_errors(td, noisy)
    assert r2["relL2M_pooled"] > 1e-3


@pytest.mark.parametrize("name", ["DYNfix_cons", "DYN_cons_mass", "DYNfix_cons_mass"])
def test_dyn_cons_rollouts_conserve_mass_exactly(tiny_root, name):
    """div:1 readout (z = d0^T b, + the fixed 1/M output map for DYNfix_cons): any model, trained or not, conserves
    sum M u in rollouts (float32 round-off)."""
    from rhmp.model import RHMP, RHMPConfig
    from rhmp.tasks.suite import rollout_eval
    kw = {"cons_param": "div"} if name == "DYNfix_cons" else {}
    td = _suite()[name](tiny_root, device="cpu", windows=3, **kw)
    assert td.model_readout == "div:1" and float(td.y_stats.mean.abs().max()) == 0.0
    torch.manual_seed(0)
    model = RHMP(RHMPConfig(in_dims=td.in_dims, even_dims=td.even_dims, C=8, n_layers=1, readout=td.model_readout),
                 td.geo_dims)
    for p in model.parameters():                     # make the (zero-initialised) readout non-trivial
        torch.nn.init.normal_(p, std=0.3)
    res = rollout_eval(model, td, device="cpu", return_pred=True)
    assert max(res["mass_drift_max"]) < 1e-5, res["mass_drift_max"]
    assert res["first_nonfinite_step"] is None
    moved = max(float((P[-1] - P[0]).abs().max()) for P in res["pred"])
    assert moved > 1e-3                               # the rollout is not trivially constant


def test_dynfix_cons_is_a_u_space_target(tiny_root):
    """DYNfix_cons (div form): the loss sees the density increment du (not M du); the map is diag(mean M / M)."""
    td = _suite()["DYNfix_cons"](tiny_root, device="cpu", windows=3, cons_param="div")
    roll, mass = td.rollout, td.rollout["mass"]
    assert td.output_map is not None and td.readout == "div:1" and td.meta["cons_convention"] == "u-space"
    diag = td.output_map.M.to_dense().diagonal()
    torch.testing.assert_close(diag, (mass.double().mean() / mass.double()).float())
    j, t = td.meta["traj_split"][0], td.meta["window_times"][0]          # first val window: trajectory j, time t
    blob = torch.load(os.path.join(tiny_root, "datasets", "v2", "DYNfix.pt"), weights_only=False)
    du = blob["traj"][j, t + 1] - blob["traj"][j, t]
    torch.testing.assert_close(td.y_stats.denormalize(td.target[td.split[1][0]])[:, 0], du.float(), rtol=1e-5,
                               atol=1e-7)


def test_dyn_cons_requires_mass_div_readout(tiny_root):
    from rhmp.tasks.suite import mass_div_readout_available
    if mass_div_readout_available():
        td = _suite()["DYN_cons"](tiny_root, device="cpu", windows=3)
        assert td.readout == "mdiv:1" and float(td.y_stats.mean.abs().max()) == 0.0
    else:
        with pytest.raises(NotImplementedError, match="mdiv"):
            _suite()["DYN_cons"](tiny_root, device="cpu", windows=3)


@pytest.mark.parametrize("name", ["DYN", "DYNfix_delta"])
def test_rollout_mass_projection(tiny_root, name):
    """project_mass=True makes any model's rollout conserve sum M u exactly; without it a random model drifts."""
    from rhmp.model import RHMP, RHMPConfig
    from rhmp.tasks.suite import rollout_eval
    td = _suite()[name](tiny_root, device="cpu", windows=3)
    torch.manual_seed(1)
    model = RHMP(RHMPConfig(in_dims=td.in_dims, C=8, n_layers=1, readout="node_scalar"), td.geo_dims)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.3)
    free = rollout_eval(model, td, device="cpu")
    proj = rollout_eval(model, td, device="cpu", project_mass=True, batch_size=1)
    assert max(free["mass_drift_max"]) > 1e-3
    assert max(proj["mass_drift_max"]) < 1e-5 and proj["project_mass"]


def test_dynfix_cons_flux_parametrisation(tiny_root):
    """cons_param='flux': du = (mean M / M) d0^T (l* . b) from an odd edge cochain; every column of M * map sums to 0
    (exact conservation for any b) and rollouts of a random model keep sum M u."""
    from rhmp.model import RHMP, RHMPConfig
    from rhmp.tasks.suite import dual_edge_lengths, rollout_eval
    td = _suite()["DYNfix_cons"](tiny_root, device="cpu", windows=3, cons_param="flux")
    assert td.readout == "cochain:1" and td.meta["cons_param"] == "flux"
    A = td.output_map.M.to_dense().double()                          # (n0, n1)
    mass = td.rollout["mass"].double()
    assert (mass @ A).abs().max() < 1e-6 * A.abs().max()
    K = td.K
    ls = dual_edge_lengths(K)
    e = 0
    a, b = K.cells[1][e].tolist()
    faces = [f for f in K.cells[2].tolist() if a in f and b in f]
    P = K.pos.double()
    ref = sum(float((P[f].mean(0) - P[[a, b]].mean(0)).norm()) for f in faces)
    assert abs(float(ls[e]) - ref) < 1e-9
    torch.manual_seed(2)
    model = RHMP(RHMPConfig(in_dims=td.in_dims, C=8, n_layers=1, readout=td.readout), td.geo_dims)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.3)
    res = rollout_eval(model, td, device="cpu", return_pred=True)
    assert max(res["mass_drift_max"]) < 1e-5
    assert max(float((P_[-1] - P_[0]).abs().max()) for P_ in res["pred"]) > 1e-3
