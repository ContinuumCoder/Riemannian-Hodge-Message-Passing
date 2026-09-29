"""Tests of the anisotropy task suite (``datasets/generators/gen_aniso.py``, ``aniso_fields.py``, ``rhmp/tasks/aniso.py``).

* FEEC discretisations: manufactured-solution convergence (P1, mixed RT0, Nedelec), exact ``d^2 = 0`` of the
  generator's incidence matrices and agreement with the complex's ``d``; the Whitney 2-form basis has unit flux
  through its own face;
* operator equivalence: the generator's scipy assemblies (anisotropic P1 stiffness, Whitney 1-/2-form Galerkin
  masses, curl-curl operator, mixed-Darcy flux mass) equal ``rhmp.dec.assemble_whitney_metric`` with the signed dyad
  coefficients of the stored tensors on the dataset samples (relative Frobenius error <= 1e-6);
* solvers: residuals <= 1e-8, ``d2 B = 0``, per-tet conservation ``d2 J = F`` and ``d1^T M2(K^-1) J = 0``;
* representability diagnostics: cone membership, the full (expm) parameterisation round trip, cone projection, and
  the 2-D counter-example (misaligned anisotropy gives negative P1 edge weights, which no positive diagonal star
  produces);
* loaders: tiny data sets for all tasks and both ratios -> shapes, degrees, readouts, split, extra tests, the
  ``load_task`` hook, structure metrics (oracle and random models), and one trainer epoch with the tensor metric.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import scipy.sparse as sp
import torch

pytest.importorskip("scipy.sparse.linalg")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, "datasets", "generators")
for _p in (ROOT, GEN):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import aniso_fields as AF  # noqa: E402
import gen_aniso as G  # noqa: E402
from rhmp.complex import CochainComplex  # noqa: E402
from rhmp.data import cell_alignment, edge_alignment  # noqa: E402
from rhmp.dec import assemble_whitney_metric  # noqa: E402

RATIOS = (10.0, 100.0)
TINY = {"AHP": dict(nmin=200, nmax=260), "ASURF": dict(h=0.2), "ACURL": dict(nmin=260, nmax=320),
        "ADARCY": dict(nmin=260, nmax=320)}
N_TINY = 9          # split 6 / 1 / 2


def _cfg(task: str) -> dict:
    return dict(G.CONFIG[task], N=N_TINY, n_fine=1, **TINY[task])


def _scipy(csr: torch.Tensor) -> sp.csr_matrix:
    return sp.csr_matrix((csr.values().double().numpy(), csr.col_indices().numpy(), csr.crow_indices().numpy()),
                         shape=tuple(csr.shape))


def _relF(A, B) -> float:
    D = (A - B).tocsr() if sp.issparse(A) else A - B
    num = sp.linalg.norm(D) if sp.issparse(D) else np.linalg.norm(D)
    den = sp.linalg.norm(B) if sp.issparse(B) else np.linalg.norm(B)
    return float(num / den)


def _perm_matrix(perm: torch.Tensor, sign: torch.Tensor | None, n: int) -> sp.csr_matrix:
    """``P`` with ``(P v)_i = sign_i v_perm(i)`` (generator order -> complex order)."""
    s = np.ones(len(perm)) if sign is None else sign.numpy()
    return sp.csr_matrix((s, (np.arange(len(perm)), perm.numpy())), shape=(len(perm), n))


def _dyad_signed(K, k: int, T: np.ndarray) -> torch.Tensor:
    """Signed dyad coefficients ``T = sum_j a_j t_j t_j^T`` in the Whitney frame of ``K.whitney[k]`` (planar /
    volume cells; surfaces use the tangent frame of the first edge)."""
    t = K.whitney[k]["t"].double().numpy()
    D = t.shape[-1]
    if D == 3 and t.shape[1] == 3:                              # surface triangles: tangent frame
        P = K.pos.double().numpy()[K.cells[2].long().numpy()]
        nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        return torch.as_tensor(AF.dyad_coeffs(T, t, AF.local_frame(t, nrm)))
    return torch.as_tensor(AF.dyad_coeffs(T, t))


# ======================================================================================================================
# discretisations
# ======================================================================================================================
def test_selftest_convergence_small():
    out = G.selftest(sizes_2d=(300, 1200), sizes_3d=(600, 2400))
    assert set(out) == {"AHP", "ADARCY", "ACURL"}


def test_tet_topology_matches_complex():
    pts, tets, _ = G.tet_mesh(5, 0, 300)
    n = len(pts)
    topo = AF.tet_topology(tets, n)
    d0, d1, d2 = topo["d0"], topo["d1"], topo["d2"]
    assert abs(d1 @ d0).max() == 0 and abs(d2 @ d1).max() == 0
    K = CochainComplex.from_tetrahedra(pts, tets, device="cpu")
    assert K.check_d2() == 0.0 and K.n[3] == len(tets)
    pe, se = edge_alignment(K.cells[1], torch.as_tensor(topo["edges"]), n)
    pf, sf = cell_alignment(K.cells[2], torch.as_tensor(topo["faces"]), n)
    pt, st = cell_alignment(K.cells[3], torch.as_tensor(tets), n)
    Pe, Pf, Pt = (_perm_matrix(pe, se, len(pe)), _perm_matrix(pf, sf, len(pf)), _perm_matrix(pt, st, len(pt)))
    for k, (Pa, Pb, dg) in enumerate(((Pe, None, d0), (Pf, Pe, d1), (Pt, Pf, d2))):
        dK = _scipy(K.d[k])
        mine = Pa @ dg if Pb is None else Pa @ dg @ Pb.T
        assert abs(mine - dK).max() == 0, f"d{k} differs from the complex"
    assert np.array_equal(K.boundary[1].numpy(), topo["boundary_edges"][pe.numpy()])
    assert np.array_equal(K.boundary[0].numpy(), topo["boundary_nodes"])


def test_whitney2_unit_flux():
    """w_f (canonical sorted face orientation) has flux 1 through face f and 0 through the other faces of the tet."""
    pts, tets, _ = G.tet_mesh(6, 0, 200)
    topo = AF.tet_topology(tets, len(pts))
    V = pts[tets]
    g = AF.bary_grads(V)
    tri_lam = np.array([[1 / 3, 1 / 3, 1 / 3], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])   # degree 3
    wts = np.array([-27 / 48, 25 / 48, 25 / 48, 25 / 48])
    n3 = len(tets)
    for i in range(4):                                             # quadrature on face i (omits local vertex i)
        lam = np.zeros((4, 4))
        lam[:, list(AF.TET_FACES[i])] = tri_lam
        W = AF.whitney2_values_tet(lam, g, topo["face_local"])     # (n3, Q, 4 faces, 3)
        loc = topo["face_local"][:, i]                             # local vertices of face i, increasing global id
        Pf = V[np.arange(n3)[:, None], loc]                        # (n3, 3, 3)
        nA = 0.5 * np.cross(Pf[:, 1] - Pf[:, 0], Pf[:, 2] - Pf[:, 0])
        flux = np.einsum("q,tqjd,td->tj", wts, W, nA)
        expect = np.zeros((n3, 4))
        expect[:, i] = 1.0
        np.testing.assert_allclose(flux, expect, atol=1e-10)


# ======================================================================================================================
# operator equivalence with rhmp's Whitney blocks on dataset samples
# ======================================================================================================================
def _ahp_one(r=100.0):
    cfg = _cfg("AHP")
    (idx, n_tot, seed, _, _), = G.job_args("AHP", [4], cfg, cfg["seed"], (r,), False)
    F = G.ahp_fields(seed, idx, cfg)
    pts, faces = G.ahp_mesh(seed, idx, n_tot)
    Sig, _, _, _ = G.ahp_tensors(F, pts, faces, r, cfg)
    return pts, faces, Sig


@pytest.mark.parametrize("task", ["AHP", "ASURF"])
def test_p1_equals_whitney_galerkin_triangles(task):
    if task == "AHP":
        pts, faces, Sig = _ahp_one()
    else:
        cfg = _cfg("ASURF")
        pts, faces, _ = G.asurf_surface(2, "torus", cfg["seed"], cfg["h"])
        F = G.asurf_fields(cfg["seed"], 2, "torus", pts)
        Sig, _, _, _, _ = G.asurf_tensors(F, pts, faces, 100.0, cfg)
    n = len(pts)
    Kp, _, _ = AF.p1_stiffness(pts, faces, Sig)
    K = CochainComplex.from_triangles(pts, faces, device="cpu")
    fperm, _ = cell_alignment(K.cells[2], torch.as_tensor(faces), n, oriented=False)
    T = Sig[fperm.numpy()]
    a = _dyad_signed(K, 1, T)
    H1 = _scipy(assemble_whitney_metric(K, 1, torch.zeros(K.n[2], dtype=torch.float64), a))
    d0 = _scipy(K.d[0])
    assert _relF(d0.T @ H1 @ d0, Kp) < 1e-6                  # rhmp.dec (fp32 blocks, signed dyad coefficients)
    H1m = _model_galerkin(K, 1, T)
    assert _relF(d0.T @ H1m @ d0, Kp) < 1e-9                 # the model's float64 full-tensor assembly
    # the generator's own Whitney 1-form blocks equal rhmp's (edge order of the complex)
    topo = AF.tri_topology(faces, n)
    M1 = AF.assemble(AF.tri_whitney1_mass(pts, faces, topo, Sig), topo["face_edges"], len(topo["edges"]))
    pe, se = edge_alignment(K.cells[1], torch.as_tensor(topo["edges"]), n)
    Pe = _perm_matrix(pe, se, len(pe))
    assert _relF(Pe @ M1 @ Pe.T, H1) < 1e-6
    assert _relF(Pe @ M1 @ Pe.T, H1m) < 1e-9


def _model_galerkin(K, k: int, T: np.ndarray) -> sp.csr_matrix:
    """The model's own full-tensor Galerkin assembly (``rhmp.layers.galerkin_blocks``, float64)."""
    from rhmp.tasks.aniso import _galerkin
    return _galerkin(K, k, torch.as_tensor(T))


def test_curlcurl_and_mixed_equal_whitney_galerkin_tets():
    cfg = _cfg("ACURL")
    (idx, n_tot, seed, _, _), = G.job_args("ACURL", [3], cfg, cfg["seed"], (100.0,), False)
    F = G.acurl_fields(seed, idx, cfg)
    pts, tets, _ = G.tet_mesh(seed, idx, n_tot)
    n = len(pts)
    topo = AF.tet_topology(tets, n)
    nu, _, _, _ = G.acurl_tensors(F, pts, tets, 100.0, cfg)
    S, M1I = G.acurl_operator(pts, tets, topo, nu, cfg["eps"])
    K = CochainComplex.from_tetrahedra(pts, tets, device="cpu")
    pt, _ = cell_alignment(K.cells[3], torch.as_tensor(tets), n, oriented=False)
    pe, se = edge_alignment(K.cells[1], torch.as_tensor(topo["edges"]), n)
    pf, sf = cell_alignment(K.cells[2], torch.as_tensor(topo["faces"]), n)
    Pe, Pf = _perm_matrix(pe, se, len(pe)), _perm_matrix(pf, sf, len(pf))
    d0, d1 = _scipy(K.d[0]), _scipy(K.d[1])
    ones = torch.ones(K.n[3], dtype=torch.float64)
    # the fixed Whitney star of rhmp.dec (precomputed fp32 blocks): identity 1-form mass
    H1 = _scipy(assemble_whitney_metric(K, 1, ones, None))
    assert _relF(Pe @ M1I @ Pe.T, H1) < 1e-6
    # full tensors: the model's float64 Galerkin blocks
    H2 = _model_galerkin(K, 2, nu[pt.numpy()])
    assert _relF(Pe @ S @ Pe.T, d1.T @ H2 @ d1 + cfg["eps"] * H1) < 1e-6
    assert _relF(Pe @ S @ Pe.T, d1.T @ H2 @ d1 + cfg["eps"] * _model_galerkin(K, 1, np.broadcast_to(
        np.eye(3), (K.n[3], 3, 3)).copy())) < 1e-9
    # mixed Darcy: Whitney 2-form mass of K^-1 and the scalar P1 stiffness of K
    Kt, Kinv, _, _, _ = G.adarcy_tensors(F, pts, tets, 100.0, G.CONFIG["ADARCY"])
    M2 = AF.assemble(AF.tet_whitney2_mass(pts, tets, topo, Kinv), topo["tet_faces"], len(topo["faces"]))
    assert _relF(Pf @ M2 @ Pf.T, _model_galerkin(K, 2, Kinv[pt.numpy()])) < 1e-9
    Kp, _, _ = AF.p1_stiffness(pts, tets, Kt)
    assert _relF(d0.T @ _model_galerkin(K, 1, Kt[pt.numpy()]) @ d0, Kp) < 1e-9
    # rhmp.dec with signed dyad coefficients agrees up to its fp32 block storage (well-conditioned: ratio 10)
    Kt10, _, _, _, _ = G.adarcy_tensors(F, pts, tets, 10.0, G.CONFIG["ADARCY"])
    H1k = _scipy(assemble_whitney_metric(K, 1, torch.zeros(K.n[3], dtype=torch.float64),
                                         _dyad_signed(K, 1, Kt10[pt.numpy()])))
    assert _relF(d0.T @ H1k @ d0, AF.p1_stiffness(pts, tets, Kt10)[0]) < 1e-5


# ======================================================================================================================
# solvers and constraints
# ======================================================================================================================
def test_sample_solvers_and_constraints():
    for task in G.TASKS:
        cfg = _cfg(task)
        (args,) = G.job_args(task, [7], cfg, cfg["seed"], RATIOS, False)
        s = G.SAMPLE_FN[task](args)
        for r in RATIOS:
            st = s[G.rtag(r)]["stats"]
            for key, v in st.items():
                if key.startswith("res") or key == "conservation":
                    assert v < 1e-8, (task, r, key, v)
            assert 0.0 <= st["cone_frac"] <= 1.0 and st["smax_median"] > 0
        if task == "ACURL":
            pts, tets = s["pos"].astype(np.float64), s["tets"].astype(np.int64)
            topo = AF.tet_topology(tets, len(pts))
            A = s["r100"]["A"].astype(np.float64)
            assert np.abs(A[topo["boundary_edges"]]).max() == 0.0               # PEC
            assert s["r100"]["stats"]["divB_max"] < 1e-10
            assert s["stats"]["div_j_rel"] < 1e-10 and s["r100"]["stats"]["gauge"] < 1e-10   # discrete Coulomb gauge
            assert 0.0 < s["stats"]["grad_removed"] < 0.5
        if task == "ADARCY":
            pts, tets = s["pos"].astype(np.float64), s["tets"].astype(np.int64)
            topo = AF.tet_topology(tets, len(pts))
            F = G.adarcy_fields(cfg["seed"], 7, cfg)
            _, Kinv, _, _, _ = G.adarcy_tensors(F, pts, tets, 100.0, cfg)
            M2 = AF.assemble(AF.tet_whitney2_mass(pts, tets, topo, Kinv), topo["tet_faces"], len(topo["faces"]))
            J = s["r100"]["J"].astype(np.float64)
            F_T = s["F_tet"].astype(np.float64)
            assert np.abs(topo["d2"] @ J - F_T).max() < 1e-5 * np.abs(F_T).max()    # float32 storage
            MJ = M2 @ J
            assert np.linalg.norm(topo["d1"].T @ MJ) < 1e-5 * np.linalg.norm(MJ)
            assert np.abs(s["r100"]["p"][s["bnd"]]).max() == 0.0


# ======================================================================================================================
# representability diagnostics
# ======================================================================================================================
def test_cone_membership_and_full_parameterisation():
    rng = np.random.default_rng(0)
    pts, tets, _ = G.tet_mesh(7, 0, 200)
    t6 = G.cell_dirs(pts, tets, AF.TET_EDGES)
    n = len(tets)
    b = np.exp(rng.normal(size=n))
    a = rng.uniform(0.0, 2.0, (n, 6))
    T_in = b[:, None, None] * np.eye(3) + np.einsum("nj,nja,njb->nab", a, t6, t6)
    rep = AF.representability(T_in, t6)
    assert rep["in_cone"].all()
    np.testing.assert_allclose(AF.cone_project(T_in[:20], t6[:20]), T_in[:20], atol=1e-8)
    # strongly prolate tensors along a fixed non-edge direction: mostly outside the cone
    tau = np.array([1.0, 2.0, 2.0]) / 3.0
    T_out = np.broadcast_to(np.eye(3) * 0.01 + np.outer(tau, tau), (n, 3, 3)).copy()
    assert AF.representability(T_out, t6)["in_cone"].mean() < 0.5
    # full parameterisation: exact round trip, and the minimax shift never increases max |s|
    for T in (T_in, T_out):
        s, logb, smax = AF.full_param_coeffs(T, t6)
        np.testing.assert_allclose(AF.full_param_tensor(s, logb, t6), T, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(np.abs(s).max(1), smax, rtol=1e-10)
    # surfaces: tangent frame
    cfg = _cfg("ASURF")
    P, Fc, _ = G.asurf_surface(1, "sphere_pert", cfg["seed"], cfg["h"])
    Ff = G.asurf_fields(cfg["seed"], 1, "sphere_pert", P)
    Sig, _, _, _, nrm = G.asurf_tensors(Ff, P, Fc, 10.0, cfg)
    ts = G.cell_dirs(P, Fc, AF.TRI_SLOTS)
    s, logb, smax = AF.full_param_coeffs(Sig, ts, AF.local_frame(ts, nrm))
    np.testing.assert_allclose(AF.full_param_tensor(s, logb, ts, AF.local_frame(ts, nrm)), Sig, rtol=1e-7,
                               atol=1e-12)


def test_edge_reconstruction_exact_for_constant_tensors():
    """The edge inputs determine the cell tensors: for a constant field the edge-averaged projections equal every
    cell's own projections and the Gram reconstruction is exact (tets, planar triangles, surface triangles)."""
    tau = np.array([1.0, 2.0, 2.0]) / 3.0
    T3 = (np.eye(3) + 99 * np.outer(tau, tau)) / 100 ** (1 / 3)
    pts, tets, _ = G.tet_mesh(8, 0, 250)
    topo = AF.tet_topology(tets, len(pts))
    t6 = G.cell_dirs(pts, tets, AF.TET_EDGES)
    T = np.broadcast_to(T3, (len(tets), 3, 3)).copy()
    proj = AF.cell_edge_average(np.einsum("tjd,tde,tje->tj", t6, T, t6), topo["tet_edges"], len(topo["edges"]))
    np.testing.assert_allclose(G.edge_reconstruction(proj, topo["tet_edges"], t6), T, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(G.edge_reconstruction(proj, topo["tet_edges"], t6, inverse=True),
                               np.linalg.inv(T), rtol=1e-8, atol=1e-10)
    pts2, faces2, S2 = _ahp_one()
    S2c = np.broadcast_to(S2[0], S2.shape).copy()
    topo2 = AF.tri_topology(faces2, len(pts2))
    ts = G.cell_dirs(pts2, faces2, AF.TRI_SLOTS)
    proj2 = AF.cell_edge_average(np.einsum("fjd,fde,fje->fj", ts, S2c, ts), topo2["face_edges"], len(topo2["edges"]))
    np.testing.assert_allclose(G.edge_reconstruction(proj2, topo2["face_edges"], ts), S2c, rtol=1e-8, atol=1e-10)


def test_diagonal_counterexample():
    """Equilateral triangle, anisotropy r across one edge: the P1 weight of that edge is negative for r > 3, so no
    positive diagonal edge star (and no tensor of the cone b I + sum a_j t_j t_j^T, a >= 0) reproduces it."""
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    faces = np.array([[0, 1, 2]])
    edges = AF.canonical_edges(faces, 3)
    tau = np.array([0.0, 1.0])                                   # perpendicular to edge (0, 1)
    for r, neg in ((2.0, False), (10.0, True), (100.0, True)):
        A = (np.eye(2) + (r - 1) * np.outer(tau, tau))[None] / np.sqrt(r)
        K, _, _ = AF.p1_stiffness(pts, faces, A)
        w = AF.edge_weights(K, edges)
        assert (w[0] < 0) == neg, (r, w)                         # edge (0, 1) is the first canonical edge
        t = G.cell_dirs(pts, faces, AF.TRI_SLOTS)
        assert bool(AF.representability(A, t)["in_cone"][0]) == (not neg)
    # a positive diagonal star always gives non-negative weights
    d0 = AF.tri_topology(faces, 3)["d0"]
    Kd = (d0.T @ sp.diags([0.3, 1.0, 2.0]) @ d0).tocsr()
    assert (AF.edge_weights(Kd, edges) > 0).all()


# ======================================================================================================================
# tiny data sets -> loaders / hook / structure metrics / trainer
# ======================================================================================================================
@pytest.fixture(scope="module")
def tiny_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("aniso")
    out = str(root / "datasets" / "v2")
    for task in G.TASKS:
        G.write_task(task, RATIOS, 1, out, n=N_TINY, n_fine=1, overrides=TINY[task])
    return str(root)


def test_generated_files(tiny_root):
    v2 = os.path.join(tiny_root, "datasets", "v2")
    for task in G.TASKS:
        for r in ("r10", "r100"):
            for suf in ("", "_fine"):
                path = os.path.join(v2, f"{task}_{r}{suf}.pt")
                assert os.path.exists(path) and os.path.exists(path.replace(".pt", ".json")), path
            blob = torch.load(os.path.join(v2, f"{task}_{r}.pt"), weights_only=False)
            assert len(blob["ptr0"]) == N_TINY + 1 and blob["meta"]["ratio"] == float(r[1:])
            top = "faces" if task in ("AHP", "ASURF") else "tets"
            deg = 2 if top == "faces" else 3
            assert blob[f"ptr{deg}"][-1] == blob[top].shape[0]
            fine = torch.load(os.path.join(v2, f"{task}_{r}_fine.pt"), weights_only=False)
            assert fine["meta"]["coarse_ids"] == [7]
            assert fine["ptr0"][-1] > 2.5 * (blob["ptr0"][8] - blob["ptr0"][7])      # ~4x the nodes


KINDS = ("AHP", "ASURF", "ASURF_heat", "ACURL", "ACURLb", "ADARCY", "ADARCYp")


@pytest.mark.parametrize("kind", KINDS)
def test_loaders(tiny_root, kind):
    from rhmp.tasks.aniso import ANISO_TASKS, SPECS
    for r in (10, 100):
        name = f"{kind}_r{r}"
        td = ANISO_TASKS[name](tiny_root, device="cpu")
        spec = SPECS[kind]
        assert td.variable_mesh and td.num_samples == N_TINY and [len(s) for s in td.split] == [6, 1, 2]
        assert td.readout == spec["readout"] and td.meta["aniso_kind"] == kind and td.meta["ratio"] == r
        tri = spec["top"] == "faces"
        assert td.spatial_dim == (2 if kind == "AHP" else 3)
        if tri:
            assert td.in_dims == {0: 1, 1: 1, 2: 2} and td.even_dims == {1: 1, 2: 2}
        elif kind.startswith("ACURL"):
            assert td.in_dims == {1: 2, 2: 1, 3: 2} and td.even_dims == {1: 1, 2: 1, 3: 2}
        else:
            assert td.in_dims == {0: 1, 1: 1, 2: 1, 3: 3} and td.even_dims == {1: 1, 2: 1, 3: 2}
        deg = {"node_scalar": 0, "cochain:1": 1, "cochain:2": 2, "curl": 2}[td.readout]
        assert td.target_degree == deg
        for K, x, y in zip(td.K, td.inputs, td.target):
            assert K.check_d2() == 0.0
            for k, v in x.items():
                assert v.shape == (K.n[k], td.in_dims[k]) and torch.isfinite(v).all()
            assert y.shape == (K.n[deg], 1) and torch.isfinite(y).all()
        assert "fine" in td.extra_tests and len(td.extra_tests["fine"]["K"]) == 1
        Kf = td.extra_tests["fine"]["K"][0]
        assert Kf.n[0] > 2.5 * td.K[7].n[0]
        if kind.startswith("ASURF"):
            assert {"test_sphere_pert", "test_torus"} <= set(td.extra_tests)
        assert set(td.meta["recovery"]["tensor"]) >= {spec["tensor_km"]} and td.meta["metric_ref"] == spec["metric_ref"]
        assert 0.0 <= td.meta["representability"]["cone_frac"] <= 1.0
        tr = torch.cat([td.target[i] for i in td.split[0].tolist()])
        if td.target_kind == "cochain":
            assert abs(float((tr ** 2).mean()) - 1.0) < 1e-4                 # scale-only (odd target)
        else:
            assert abs(float(tr.mean())) < 1e-5 and abs(float(tr.std()) - 1) < 1e-3
        if kind == "ACURLb":                                               # B = d1 A exactly (physical units)
            ta = ANISO_TASKS[f"ACURL_r{r}"](tiny_root, device="cpu", fine=False)
            for i in range(N_TINY):
                A = ta.y_stats.denormalize(ta.target[i]).double()
                B = td.y_stats.denormalize(td.target[i]).double()
                dA = td.K[i].apply_d(1, A[:, None, :].contiguous())[:, 0]
                assert float((dA - B).abs().max()) < 1e-5 * float(B.abs().max())


def test_load_task_hook(tiny_root):
    from rhmp.tasks import list_tasks, load_task, task_defaults
    names = list_tasks()
    assert "ACURLb_r100" in names and "AHP_r10" in names and "SURF" in names
    assert task_defaults("ADARCY_r10") == {"C": 128, "layers": 4, "batch": 8}
    td = load_task("AHP_r10", tiny_root, device="cpu", max_samples=3, fine=False)
    assert [len(s) for s in td.split] == [1, 1, 1] and "fine" not in td.extra_tests
    td = load_task("ADARCY_r100", tiny_root, device="cpu", whitney=False, abs_scale=True)
    assert all(not K.whitney for K in td.K) and td.in_dims[0] == 2 and td.even_dims[0] == 1
    from rhmp.tasks.aniso import load_aniso
    sub = load_aniso("ACURL_r10", tiny_root, device="cpu", n_samples=6, fine=False)
    assert [len(s) for s in sub.split] == [4, 0, 2] and sub.meta["n_used"] == 6 and sub.meta["n_file"] == N_TINY
    lean = load_aniso("ACURLb_r10", tiny_root, device="cpu", whitney=False)      # blocks dropped while building
    full = load_aniso("ACURLb_r10", tiny_root, device="cpu")
    assert all(not K.whitney for K in lean.K + lean.extra_tests["fine"]["K"]) and all(K.whitney for K in full.K)
    assert all(torch.equal(a, b) for a, b in zip(lean.target, full.target))
    assert "ACURLb_r100_n1500" in names


class _Oracle(torch.nn.Module):
    """Returns the stored normalised targets of the evaluated samples (in batch order)."""

    def __init__(self, targets):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.targets = list(targets)

    def forward(self, inputs, K):
        n = K.num_graphs
        out, self.targets = self.targets[:n], self.targets[n:]
        return torch.cat(out, 0)[:, None, :]


@pytest.mark.parametrize("kind", KINDS)
def test_structure_metrics(tiny_root, kind):
    from rhmp.model import RHMP, RHMPConfig
    from rhmp.tasks.aniso import ANISO_TASKS, structure_metrics
    td = ANISO_TASKS[f"{kind}_r100"](tiny_root, device="cpu")
    te = td.split[2].tolist()
    res = structure_metrics(_Oracle([td.target[i] for i in te]), td, batch_size=1, device="cpu")
    # float32 storage of the targets sets the floor (pde_res ~ cond x 6e-8); a wrong operator gives O(1)
    tol = {"pde_res": 1e-3, "bc": 1e-6, "integral": 1e-5, "div": 1e-5, "cons": 1e-5, "irrot": 1e-4, "gauge": 1e-4}
    for k, v in res.items():
        if isinstance(v, dict) and not k.startswith("target_"):
            assert v["max"] < tol[k], (kind, k, v)
            assert res["target_" + k]["max"] < tol[k]
    cfg = RHMPConfig(in_dims=td.in_dims, even_dims=td.even_dims, C=8, n_layers=1, readout=td.readout)
    torch.manual_seed(0)
    model = RHMP(cfg, td.geo_dims)
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.2)
    res = structure_metrics(model, td, batch_size=2, device="cpu", targets=False)
    assert all(np.isfinite(v["mean"]) for k, v in res.items() if isinstance(v, dict))
    if kind == "ACURLb":                                  # curl readout: d2 B = 0 for any weights
        assert res["div"]["max"] < 1e-5
    res_f = structure_metrics(model, td, source=td.extra_tests["fine"], device="cpu", targets=False)
    assert res_f["N"] == 1


@pytest.mark.parametrize("kind", ["AHP", "ASURF", "ACURLb", "ADARCY"])
def test_trainer_epoch_tensor_metric(tiny_root, tmp_path, kind):
    from rhmp.tasks.aniso import ANISO_TASKS
    from rhmp.train import parse_args, run
    td = ANISO_TASKS[f"{kind}_r100"](tiny_root, device="cpu")
    ref = td.meta["metric_ref"]
    args = parse_args(["--task", f"{kind}_r100", "--epochs", "1", "--device", "cpu", "--C", "8", "--layers", "1",
                       "--metric-type", "tensor", "--metric-ref", ref, "--out", str(tmp_path / kind), "--quiet"])
    res = run(args, task=td)
    assert res["complete"] and np.isfinite(res["test"]["R2"]) and "fine" in res
