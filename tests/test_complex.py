"""Tests for rhmp.complex and rhmp.geometry (DESIGN §5)."""
from __future__ import annotations

import math
import os
import pickle
import time

import numpy as np
import pytest
import torch
from conftest import (
    DEVICES,
    MESH_NAMES,
    ROOT,
    make_batch,
    make_complex,
    mesh_delaunay_2d,
    mesh_mixed_polygons,
    mesh_nonmanifold,
    mesh_plane_grid,
    mesh_quad_grid,
    mesh_sphere,
    mesh_tet_cube,
    mesh_torus,
    random_orthogonal,
    relabel_mesh,
)

from rhmp.complex import CochainComplex, batch_complexes, validate_triangles
from rhmp.geometry import COT_CLAMP, GEO_FEATURE_NAMES, KAPPA

TRI_MESHES = ("grid", "sphere", "torus", "delaunay", "sliver", "nonmanifold")
POLY_MESHES = ("quads", "mixed")
MESH_2D = ("grid", "delaunay", "sliver", "quads", "mixed")


def dense(A: torch.Tensor) -> torch.Tensor:
    return A.to_dense().cpu().double()


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def raw_mesh(name):
    from conftest import _MESHES

    return _MESHES[name]()


def build(name, pos, cells, device, **kw):
    if name == "tets":
        return CochainComplex.from_tetrahedra(pos, cells, device=device, **kw)
    if name in POLY_MESHES:
        return CochainComplex.from_polygons(pos, cells, device=device, **kw)
    return CochainComplex.from_triangles(pos, cells, device=device, **kw)


def assert_same_geometry(K1, K2, rtol=1e-5, atol=1e-5):
    for k in range(K1.dim + 1):
        torch.testing.assert_close(K1.geo[k].cpu(), K2.geo[k].cpu(), rtol=rtol, atol=atol, msg=f"geo[{k}]")
        torch.testing.assert_close(K1.star[k].cpu(), K2.star[k].cpu(), rtol=rtol, atol=atol, msg=f"star[{k}]")
        assert torch.equal(K1.boundary[k].cpu(), K2.boundary[k].cpu())


# ----------------------------------------------------------------------------------------------
# exactness of the topology
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", MESH_NAMES)
def test_d_squared_is_exactly_zero(name, device):
    K = make_complex(name, device)
    assert K.check_d2() == 0.0
    for k in range(K.dim - 1):
        assert torch.count_nonzero(dense(K.d[k + 1]) @ dense(K.d[k])) == 0


@pytest.mark.parametrize("diagonal", ["v1", "main", "anti", "alternate"])
def test_d_squared_grids(diagonal, device):
    Kt = CochainComplex.from_grid((6, 5), 0.3, cell="tri", diagonal=diagonal, device=device)
    Kq = CochainComplex.from_grid((6, 5), (0.3, 0.2), cell="quad", device=device)
    assert Kt.check_d2() == 0.0 and Kq.check_d2() == 0.0
    assert Kt.n == [30, 5 * 5 + 6 * 4 + 20, 40] and Kq.n == [30, 49, 20]


def test_d_squared_batches(device):
    Kb, parts = make_batch(device)
    assert Kb.check_d2() == 0.0
    Kt = CochainComplex.batch([make_complex("tets", device), make_complex("tets", device, n=40, seed=3)])
    assert Kt.check_d2() == 0.0
    Kp = CochainComplex.batch([make_complex("quads", device), make_complex("mixed", device)])
    assert Kp.check_d2() == 0.0


@pytest.mark.parametrize("name", MESH_NAMES)
def test_operator_structure(name, device):
    K = make_complex(name, device)
    E = K.cells[1]
    assert bool((E[:, 0] < E[:, 1]).all())
    assert torch.unique(E, dim=0).shape[0] == K.n[1]
    assert K.cells[0].shape == (K.n[0], 1)
    for k in range(K.dim):
        for A, shape in ((K.d[k], (K.n[k + 1], K.n[k])), (K.dT[k], (K.n[k], K.n[k + 1])),
                         (K.d_abs[k], (K.n[k + 1], K.n[k])), (K.dT_abs[k], (K.n[k], K.n[k + 1]))):
            assert A.layout == torch.sparse_csr and A.dtype == torch.float32
            assert tuple(A.shape) == shape and A.device.type == torch.device(device).type
            with torch.sparse.check_sparse_tensor_invariants():  # sorted, unique columns per row
                torch.sparse_csr_tensor(A.crow_indices(), A.col_indices(), A.values(), A.shape)
        Dk = dense(K.d[k])
        assert torch.equal(dense(K.dT[k]), Dk.T)
        assert torch.equal(dense(K.d_abs[k]), Dk.abs())
        assert torch.equal(dense(K.dT_abs[k]), Dk.abs().T)
        assert set(torch.unique(Dk).tolist()) <= {-1.0, 0.0, 1.0}
    D0 = dense(K.d[0])
    r = torch.arange(K.n[1])
    assert bool((D0[r, E[:, 0].cpu()] == -1).all()) and bool((D0[r, E[:, 1].cpu()] == 1).all())
    assert bool((D0.abs().sum(1) == 2).all())


@pytest.mark.parametrize("name,chi", [("grid", 1), ("sphere", 2), ("torus", 0), ("delaunay", 1), ("sliver", 1),
                                      ("nonmanifold", 1), ("tets", 1), ("quads", 1), ("mixed", 1)])
def test_euler_characteristic(name, chi):
    K = make_complex(name)
    assert sum((-1) ** k * nk for k, nk in enumerate(K.n)) == chi


@pytest.mark.parametrize("name", TRI_MESHES + POLY_MESHES)
def test_face_orientation_and_d1_signs(name):
    pos, cells = raw_mesh(name)
    K = build(name, pos, cells, "cpu")
    F = K.cells[2].numpy()
    assert np.array_equal(F, cells if name in POLY_MESHES else cells.astype(np.int64))
    eid = {tuple(e): i for i, e in enumerate(K.cells[1].tolist())}
    D1 = dense(K.d[1]).numpy()
    FE, FS = K.meta["face_edges"].numpy(), K.meta["face_edge_signs"].numpy()
    for f, face in enumerate(F.tolist()):
        face = [v for v in face if v >= 0]
        expect = np.zeros(K.n[1])
        for j, (a, b) in enumerate(zip(face, face[1:] + face[:1])):  # cyclic boundary with signs
            expect[eid[(min(a, b), max(a, b))]] += 1.0 if a < b else -1.0
            assert FE[f, j] == eid[(min(a, b), max(a, b))] and FS[f, j] == (1.0 if a < b else -1.0)
        assert np.array_equal(D1[f], expect)
        assert bool((FE[f, len(face):] == -1).all()) and bool((FS[f, len(face):] == 0).all())


def test_tet_orientation_signs():
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]])
    K = CochainComplex.from_tetrahedra(pos, np.array([[0, 1, 2, 3]]))
    assert K.cells[2].tolist() == [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    assert dense(K.d[2]).tolist() == [[-1.0, 1.0, -1.0, 1.0]]
    K2 = CochainComplex.from_tetrahedra(pos, np.array([[1, 0, 2, 3]]))
    assert dense(K2.d[2]).tolist() == [[1.0, -1.0, 1.0, -1.0]]
    assert K.check_d2() == 0.0 and K2.check_d2() == 0.0
    # consistently (positively) oriented tets: interior faces cancel, boundary faces appear once
    pts, tets = mesh_tet_cube()
    X = pts[tets]
    vol = np.einsum("ij,ij->i", X[:, 1] - X[:, 0], np.cross(X[:, 2] - X[:, 0], X[:, 3] - X[:, 0]))
    tets = np.where((vol < 0)[:, None], tets[:, [1, 0, 2, 3]], tets)
    K = CochainComplex.from_tetrahedra(pts, tets)
    TF, TS, FE = K.meta["tet_faces"], K.meta["tet_face_signs"], K.meta["face_edges"]
    assert torch.equal(dense(K.d[2])[torch.arange(K.n[3])[:, None], TF], TS.double())
    assert torch.equal(dense(K.d[1])[torch.arange(K.n[2])[:, None], FE], K.meta["face_edge_signs"].double())
    F, E = K.cells[2], K.cells[1]
    for j in range(3):  # edge j of a face joins its vertices j and j+1
        a, b = F[:, j], F[:, (j + 1) % 3]
        assert torch.equal(E[FE[:, j]], torch.stack([torch.minimum(a, b), torch.maximum(a, b)], 1))
    for i in range(4):  # face i of a tet omits local vertex i
        others = [c for c in range(4) if c != i]
        assert torch.equal(F[TF[:, i]], K.cells[3][:, others].sort(dim=1).values)
    colsum = dense(K.d[2]).sum(0)
    assert torch.equal(colsum.abs() == 1, K.boundary[2]) and bool((colsum[~K.boundary[2]] == 0).all())
    assert bool((K.cells[2][:, 0] < K.cells[2][:, 1]).all() and (K.cells[2][:, 1] < K.cells[2][:, 2]).all())


# ----------------------------------------------------------------------------------------------
# validation and input handling
# ----------------------------------------------------------------------------------------------
def test_validation_drops_degenerate_and_duplicates(device):
    pos, faces = mesh_plane_grid(5, 4)
    n = len(faces)
    extra = np.array([faces[0][[0, 2, 1]], faces[3], [5, 5, 6], [0, 1, 2]])  # dup(flipped), dup, repeated, collinear
    with pytest.warns(RuntimeWarning, match="dropped 2 degenerate and 2 duplicate"):
        K = CochainComplex.from_triangles(pos, np.concatenate([faces, extra]), device=device)
    assert K.n[2] == n and torch.equal(K.meta["face_index"].cpu(), torch.arange(n))
    v = K.meta["validation"]
    assert (v["n_input"], v["n_degenerate_index"], v["n_degenerate_measure"], v["n_duplicate"], v["n_kept"]) == \
        (n + 4, 1, 1, 2, n)
    Fc, stats = validate_triangles(pos, np.concatenate([faces, extra]))
    assert Fc.shape == (n, 3) and stats["n_duplicate"] == 2
    # first occurrence is kept, whatever its orientation
    with pytest.warns(RuntimeWarning):
        K2 = CochainComplex.from_triangles(pos, np.concatenate([faces[:1, [0, 2, 1]], faces]), device=device)
    assert K2.cells[2][0].tolist() == faces[0, [0, 2, 1]].tolist() and K2.meta["face_index"][1].item() == 2
    assert K.check_d2() == 0.0 and K2.check_d2() == 0.0


def test_validation_errors_and_unvalidated_duplicates():
    pos, faces = mesh_plane_grid(4, 4)
    with pytest.raises(ValueError, match="repeated"):
        CochainComplex.from_triangles(pos, np.concatenate([faces, [[1, 1, 2]]]), validate=False)
    with pytest.raises(ValueError, match="outside"):
        CochainComplex.from_triangles(pos, np.concatenate([faces, [[1, 2, 99]]]))
    with pytest.raises(ValueError, match="shape"):
        CochainComplex.from_triangles(pos, faces[:, :2])
    with pytest.raises(ValueError, match="shape"):
        CochainComplex.from_triangles(pos[:, :1], faces)
    with pytest.raises(ValueError, match="non-finite"):
        bad = pos.copy()
        bad[0, 0] = np.nan
        CochainComplex.from_triangles(bad, faces)
    # duplicates are a legal (non-regular) CW complex when validation is off
    K = CochainComplex.from_triangles(pos, np.concatenate([faces, faces[:2]]), validate=False)
    assert K.n[2] == len(faces) + 2 and K.check_d2() == 0.0
    with pytest.raises(ValueError, match="cotan"):
        CochainComplex.from_tetrahedra(*mesh_tet_cube(), star="cotan")
    with pytest.raises(ValueError, match="star"):
        CochainComplex.from_triangles(pos, faces, star="circumcentric")


def test_polygon_validation():
    pos, quads = mesh_quad_grid(4, 4)
    bad = np.full((3, 4), -1)
    bad[0, :3] = [0, 1, 1]  # repeated vertex
    bad[1, :2] = [0, 1]  # < 3 vertices
    bad[2] = quads[0][[1, 2, 3, 0]]  # duplicate (cyclic shift)
    with pytest.warns(RuntimeWarning, match="dropped 2 degenerate and 1 duplicate"):
        K = CochainComplex.from_polygons(pos, np.concatenate([quads, bad]))
    assert K.n[2] == len(quads) and K.check_d2() == 0.0
    with pytest.raises(ValueError, match="trailing"):
        CochainComplex.from_polygons(pos, np.array([[0, -1, 1, 5]]))
    with pytest.raises(ValueError, match="validate=True"):
        CochainComplex.from_polygons(pos, np.concatenate([quads, bad[:1]]), validate=False)


def test_input_types_are_equivalent():
    pos, faces = mesh_delaunay_2d(60)
    pos = pos.astype(np.float32).astype(np.float64)  # exactly representable in float32
    Ks = [
        CochainComplex.from_triangles(pos, faces),
        CochainComplex.from_triangles(pos.astype(np.float32), faces.astype(np.int32)),
        CochainComplex.from_triangles(torch.tensor(pos, dtype=torch.float32), torch.tensor(faces, dtype=torch.int32)),
        CochainComplex.from_triangles(pos.tolist(), faces.tolist()),
        CochainComplex.from_triangles(pos, faces.astype(np.float64)),
    ]
    for K in Ks[1:]:
        assert K.n == Ks[0].n and torch.equal(K.cells[2], Ks[0].cells[2])
        assert_same_geometry(K, Ks[0], rtol=1e-5, atol=1e-5)
    Kp = CochainComplex.from_polygons(pos, faces.tolist())
    assert torch.equal(Kp.cells[2], Ks[0].cells[2])
    with pytest.raises(ValueError, match="integer"):
        CochainComplex.from_triangles(pos, faces + 0.5)


def test_planar_2d_equals_embedded_3d():
    pos, faces = mesh_plane_grid(6, 5, jitter=0.2)
    K2 = CochainComplex.from_triangles(pos, faces)
    Q = random_orthogonal(3, seed=4)
    pos3 = np.concatenate([pos, np.zeros((len(pos), 1))], 1) @ Q.T + np.array([0.3, -2.0, 5.0])
    K3 = CochainComplex.from_triangles(pos3, faces)
    assert_same_geometry(K2, K3)
    assert K3.pos.shape[1] == 3 and K2.pos.shape[1] == 2


# ----------------------------------------------------------------------------------------------
# geometry: invariance, stars, boundary, known values
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", MESH_NAMES)
@pytest.mark.parametrize("reflect", [False, True])
def test_geometry_is_E_n_invariant(name, reflect, device):
    pos, cells = raw_mesh(name)
    D = pos.shape[1]
    Q = random_orthogonal(D, seed=11, reflect=reflect)
    t = np.random.default_rng(5).normal(size=D) * 3.0
    K1 = build(name, pos, cells, device)
    K2 = build(name, pos @ Q.T + t, cells, device)
    assert_same_geometry(K1, K2)


@pytest.mark.parametrize("name", MESH_NAMES)
def test_geo_is_scale_invariant_and_stars_scale(name):
    pos, cells = raw_mesh(name)
    lam = 3.7
    K1, K2 = build(name, pos, cells, "cpu"), build(name, lam * pos, cells, "cpu")
    for k in range(K1.dim + 1):
        torch.testing.assert_close(K1.geo[k], K2.geo[k], rtol=1e-5, atol=1e-5)
    powers = [2, 0, -2] if K1.dim == 2 else [3, 1, -1, -3]
    for k, p in enumerate(powers):
        torch.testing.assert_close(K2.star[k], K1.star[k] * lam ** p, rtol=1e-5, atol=0)


@pytest.mark.parametrize("name", MESH_NAMES)
@pytest.mark.parametrize("star", ["default", "barycentric", "unit", "cotan"])
def test_stars_positive_and_finite(name, star, device):
    if star == "cotan" and name == "tets":
        pytest.skip("cotan star is defined for 2-cells only")
    K = make_complex(name, device, star=None if star == "default" else star)
    assert K.meta["star_type"] == ("cotan" if star == "default" and name in TRI_MESHES else
                                   "barycentric" if star == "default" else star)
    for k in range(K.dim + 1):
        assert K.star[k].shape == (K.n[k],) and K.star[k].dtype == torch.float32
        assert bool(torch.isfinite(K.star[k]).all()) and bool((K.star[k] > 0).all())
        assert K.geo[k].shape == (K.n[k], len(GEO_FEATURE_NAMES[k])) and K.geo[k].dtype == torch.float32
        assert bool(torch.isfinite(K.geo[k]).all())
        if star == "unit":
            assert bool((K.star[k] == 1).all())
    assert K.geo_dims == {k: len(GEO_FEATURE_NAMES[k]) for k in range(K.dim + 1)}


def test_square_grid_star_values():
    h = 0.25
    K = CochainComplex.from_grid((5, 4), h, cell="quad")
    nx, ny = 5, 4
    i, j = torch.arange(K.n[0]) // ny, torch.arange(K.n[0]) % ny
    on_x, on_y = (i == 0) | (i == nx - 1), (j == 0) | (j == ny - 1)
    expect0 = h * h * torch.where(on_x, 0.5, 1.0) * torch.where(on_y, 0.5, 1.0)
    torch.testing.assert_close(K.star[0], expect0.float())
    torch.testing.assert_close(K.star[1], torch.where(K.boundary[1], 0.5, 1.0))
    torch.testing.assert_close(K.star[2], torch.full((K.n[2],), 1 / h ** 2))
    assert torch.equal(K.boundary[0], on_x | on_y)
    assert int(K.boundary[1].sum()) == 2 * (nx - 1) + 2 * (ny - 1)
    Kc = CochainComplex.from_grid((5, 4), h, cell="quad", star="cotan")  # circumcentric == barycentric on rectangles
    for k in range(3):
        torch.testing.assert_close(Kc.star[k], K.star[k])
    # anisotropic spacing: star1 = |dual edge| / |edge|
    Ka = CochainComplex.from_grid((4, 4), (0.5, 0.2), cell="quad")
    ev = Ka.edge_vectors().abs()
    interior = ~Ka.boundary[1]
    xe = interior & (ev[:, 0] > 0)
    ye = interior & (ev[:, 1] > 0)
    torch.testing.assert_close(Ka.star[1][xe], torch.full((int(xe.sum()),), 0.2 / 0.5))
    torch.testing.assert_close(Ka.star[1][ye], torch.full((int(ye.sum()),), 0.5 / 0.2))


def test_right_triangle_grid_cotan_clamp():
    K = CochainComplex.from_grid((6, 6), 0.1, cell="tri", diagonal="main")
    ev = K.edge_vectors()
    diag = (ev[:, 0].abs() > 1e-9) & (ev[:, 1].abs() > 1e-9)
    raw = torch.where(diag, 0.0, torch.where(K.boundary[1], 0.5, 1.0))
    floor = KAPPA * float(raw.median())  # torch.median = lower median, as in geometry.py
    torch.testing.assert_close(K.star[1][diag], torch.full((int(diag.sum()),), floor))
    axis = ~diag
    torch.testing.assert_close(K.star[1][axis], torch.where(K.boundary[1][axis], 0.5, 1.0))
    assert K.meta["geometry_stats"]["n_star1_clamped"] == int(diag.sum())
    torch.testing.assert_close(K.geo[1][diag, 2], torch.zeros(int(diag.sum())), atol=1e-6, rtol=0)


def test_equilateral_hexagon_values():
    ang = np.arange(6) * np.pi / 3
    pos = np.concatenate([[[0.0, 0.0]], np.stack([np.cos(ang), np.sin(ang)], 1)])
    faces = np.array([[0, 1 + k, 1 + (k + 1) % 6] for k in range(6)])
    K = CochainComplex.from_triangles(pos, faces)
    c = 1 / math.tan(math.pi / 3)
    torch.testing.assert_close(K.star[1], torch.where(K.boundary[1], c / 2, c).float())
    A = math.sqrt(3) / 4
    torch.testing.assert_close(K.star[0][0], torch.tensor(2 * A, dtype=torch.float32))
    torch.testing.assert_close(K.star[2], torch.full((6,), 1 / A))
    names = GEO_FEATURE_NAMES
    defect = K.geo[0][:, names[0].index("angle_defect")]
    torch.testing.assert_close(defect, torch.tensor([0.0] + [math.pi / 3] * 6), atol=1e-6, rtol=0)
    torch.testing.assert_close(K.geo[2][:, names[2].index("log_aspect")], torch.zeros(6), atol=1e-6, rtol=0)
    torch.testing.assert_close(K.geo[2][:, names[2].index("min_angle")], torch.full((6,), math.pi / 3))
    Kb = CochainComplex.from_triangles(pos, faces, star="barycentric")  # circumcentre == centroid here
    torch.testing.assert_close(Kb.star[1], K.star[1])


def test_cotan_weights_match_reference():
    pos, faces = mesh_delaunay_2d(80, seed=2)
    K = CochainComplex.from_triangles(pos, faces)
    eid = {tuple(e): i for i, e in enumerate(K.cells[1].tolist())}
    ref = np.zeros(K.n[1])
    bary = np.zeros(K.n[1])
    dual = np.zeros(K.n[0])
    for f in faces:
        P = pos[f]
        c = P.mean(0)
        area = 0.5 * abs(cross2(P[1] - P[0], P[2] - P[0]))
        for s in range(3):
            a, b, o = f[s], f[(s + 1) % 3], f[(s + 2) % 3]
            u, w = pos[a] - pos[o], pos[b] - pos[o]
            e = eid[(min(a, b), max(a, b))]
            ref[e] += 0.5 * np.dot(u, w) / abs(cross2(u, w))
            bary[e] += np.linalg.norm(c - 0.5 * (pos[a] + pos[b])) / np.linalg.norm(pos[a] - pos[b])
            dual[f[s]] += area / 3
    cot = torch.tensor(np.clip(ref, -COT_CLAMP, COT_CLAMP), dtype=torch.float32)
    torch.testing.assert_close(K.geo[1][:, GEO_FEATURE_NAMES[1].index("cotan_weight")], cot, rtol=1e-5, atol=1e-5)
    floor = KAPPA * np.sort(ref)[(len(ref) - 1) // 2]  # lower median (torch.median convention)
    torch.testing.assert_close(K.star[1], torch.tensor(np.maximum(ref, floor), dtype=torch.float32), rtol=1e-5, atol=0)
    torch.testing.assert_close(K.star[0], torch.tensor(dual, dtype=torch.float32), rtol=1e-5, atol=0)
    Kb = CochainComplex.from_triangles(pos, faces, star="barycentric")
    torch.testing.assert_close(Kb.star[1], torch.tensor(bary, dtype=torch.float32), rtol=1e-5, atol=0)


def test_gauss_bonnet_and_dihedral():
    defect = GEO_FEATURE_NAMES[0].index("angle_defect")
    dih = GEO_FEATURE_NAMES[1].index("cos_dihedral")
    Ks = make_complex("sphere")
    assert abs(float(Ks.geo[0][:, defect].double().sum()) - 4 * math.pi) < 1e-4
    assert bool((Ks.geo[1][:, dih] < 1).all()) and bool((Ks.geo[1][:, dih] > 0.9).all())
    Kt = make_complex("torus")
    assert abs(float(Kt.geo[0][:, defect].double().sum())) < 1e-4
    Kg = make_complex("grid")  # planar disk: interior defect 0, boundary turning angles sum to 2 pi
    g = Kg.geo[0][:, defect]
    assert float(g[~Kg.boundary[0]].abs().max()) < 1e-5
    assert abs(float(g[Kg.boundary[0]].double().sum()) - 2 * math.pi) < 1e-4
    torch.testing.assert_close(Kg.geo[1][:, dih], torch.ones(Kg.n[1]))
    for name in ("quads", "mixed"):
        Kp = make_complex(name)
        gp = Kp.geo[0][:, defect]
        assert float(gp[~Kp.boundary[0]].abs().max()) < 1e-5
        assert abs(float(gp.double().sum()) - 2 * math.pi) < 1e-4


def test_boundary_flags():
    for name in ("sphere", "torus"):
        K = make_complex(name)
        assert not any(bool(b.any()) for b in K.boundary)
    nx, ny = 7, 6
    K = make_complex("grid", nx=nx, ny=ny)
    i, j = torch.arange(K.n[0]) // ny, torch.arange(K.n[0]) % ny
    assert torch.equal(K.boundary[0], (i == 0) | (i == nx - 1) | (j == 0) | (j == ny - 1))
    assert int(K.boundary[1].sum()) == 2 * (nx - 1) + 2 * (ny - 1)
    E = K.cells[1]
    assert torch.equal(K.boundary[1], K.boundary[0][E[:, 0]] & K.boundary[0][E[:, 1]] &
                       (dense(K.d[1]).abs().sum(0) == 1))
    assert torch.equal(K.boundary[2], (dense(K.d_abs[1]) @ K.boundary[1].double()) > 0)
    # tets: boundary faces = convex hull
    from scipy.spatial import Delaunay

    pts, tets = mesh_tet_cube()
    Kt = CochainComplex.from_tetrahedra(pts, tets)
    hull = {tuple(sorted(f)) for f in Delaunay(pts).convex_hull.tolist()}
    bfaces = {tuple(f) for f in Kt.cells[2][Kt.boundary[2]].tolist()}
    assert bfaces == hull
    assert torch.equal(Kt.boundary[3], (dense(Kt.d_abs[2]) @ Kt.boundary[2].double()) > 0)
    # non-manifold edge: three cofaces, not a boundary edge
    Kn = make_complex("nonmanifold")
    e01 = Kn.cells[1].tolist().index([0, 1])
    assert not bool(Kn.boundary[1][e01])
    torch.testing.assert_close(Kn.geo[1][e01, GEO_FEATURE_NAMES[1].index("log_cofaces")], torch.tensor(math.log(3.0)))
    assert Kn.meta["geometry_stats"]["n_nonmanifold_edges"] == 1


def test_regular_tet_values():
    pos = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1.0]])
    K = CochainComplex.from_tetrahedra(pos, np.array([[0, 1, 2, 3]]))
    torch.testing.assert_close(K.geo[3][:, GEO_FEATURE_NAMES[3].index("log_aspect")], torch.zeros(1), atol=1e-6, rtol=0)
    torch.testing.assert_close(K.geo[2][:, GEO_FEATURE_NAMES[2].index("log_aspect")], torch.zeros(4), atol=1e-6, rtol=0)
    torch.testing.assert_close(K.geo[2][:, GEO_FEATURE_NAMES[2].index("min_angle")], torch.full((4,), math.pi / 3))
    vol = 8.0 / 3.0
    torch.testing.assert_close(K.star[0], torch.full((4,), vol / 4))
    torch.testing.assert_close(K.star[3], torch.tensor([1 / vol]))
    assert all(bool(b.all()) for b in K.boundary)
    assert K.meta["geometry_stats"]["n_negative_cotan_edges"] == 0


def test_isolated_vertex_warns_and_stays_positive():
    pos, faces = mesh_plane_grid(4, 4)
    pos = np.concatenate([pos, [[5.0, 5.0]]])
    with pytest.warns(RuntimeWarning, match="belong to no face"):
        K = CochainComplex.from_triangles(pos, faces)
    assert bool((K.star[0] > 0).all()) and bool(torch.isfinite(K.geo[0]).all())
    assert not bool(K.boundary[0][-1]) and K.meta["geometry_stats"]["n_isolated_vertices"] == 1


def test_polygons_of_triangles_equal_triangles():
    pos, faces = mesh_delaunay_2d(70, seed=4)
    for star in ("barycentric", "cotan", "unit"):
        Kt = CochainComplex.from_triangles(pos, faces, star=star)
        Kp = CochainComplex.from_polygons(pos, faces, star=star)
        assert Kt.n == Kp.n and torch.equal(Kt.cells[1], Kp.cells[1])
        for k in range(2):
            assert torch.equal(dense(Kt.d[k]), dense(Kp.d[k]))
        assert_same_geometry(Kt, Kp, rtol=1e-6, atol=1e-6)


def test_mixed_polygon_complex():
    pos, cells = mesh_mixed_polygons()
    K = CochainComplex.from_polygons(pos, cells)
    L = K.meta["face_lengths"]
    assert set(L.tolist()) == {3, 4, 6} and K.cells[2].shape[1] == 6
    area = 1.0 / K.star[2].double()
    total = torch.tensor(1.0 * 4 / 6, dtype=torch.float64)  # 7 x 5 grid, h = 1/6: [0, 1] x [0, 2/3]
    torch.testing.assert_close(area.sum(), total)
    torch.testing.assert_close(K.star[0].double().sum(), total)  # barycentric dual cells tile the domain
    assert K.check_d2() == 0.0


# ----------------------------------------------------------------------------------------------
# batching, devices, grids
# ----------------------------------------------------------------------------------------------
def _check_batch_equals_parts(Kb, parts):
    assert Kb.num_graphs == len(parts) and Kb.n == [sum(P.n[k] for P in parts) for k in range(Kb.dim + 1)]
    assert Kb.meta["sizes"] == [P.n for P in parts]
    off = [0] * (Kb.dim + 1)
    for g, P in enumerate(parts):
        for k in range(Kb.dim + 1):
            sl = slice(off[k], off[k] + P.n[k])
            assert torch.equal(Kb.geo[k][sl], P.geo[k]) and torch.equal(Kb.star[k][sl], P.star[k])
            assert torch.equal(Kb.boundary[k][sl], P.boundary[k])
            assert bool((Kb.batch[k][sl] == g).all())
            assert Kb.meta["ptr"][k][g].item() == off[k]
            if k >= 1:
                c = P.cells[k]
                cb = Kb.cells[k][sl, : c.shape[1]]
                assert torch.equal(torch.where(c >= 0, c + off[0], c), cb)
                assert bool((Kb.cells[k][sl, c.shape[1]:] == -1).all())
        for k in range(Kb.dim):
            r = slice(off[k + 1], off[k + 1] + P.n[k + 1])
            c = slice(off[k], off[k] + P.n[k])
            for Ab, Ap in ((Kb.d, P.d), (Kb.dT, P.dT), (Kb.d_abs, P.d_abs), (Kb.dT_abs, P.dT_abs)):
                Db = dense(Ab[k])
                if Ab is Kb.dT or Ab is Kb.dT_abs:
                    assert torch.equal(Db[c, r], dense(Ap[k]))
                else:
                    assert torch.equal(Db[r, c], dense(Ap[k]))
        assert torch.equal(Kb.pos[off[0]:off[0] + P.n[0]], P.pos)
        for key, sk, deg in (("face_edges", "face_edge_signs", 1), ("tet_faces", "tet_face_signs", 2)):
            if key in P.meta:
                c, rows = P.meta[key], slice(off[deg + 1], off[deg + 1] + P.n[deg + 1])
                assert torch.equal(Kb.meta[key][rows, : c.shape[1]], torch.where(c >= 0, c + off[deg], c))
                assert torch.equal(Kb.meta[sk][rows, : c.shape[1]], P.meta[sk])
                assert bool((Kb.meta[key][rows, c.shape[1]:] == -1).all())
        off = [o + P.n[k] for k, o in enumerate(off)]
    for k in range(Kb.dim):  # nothing outside the diagonal blocks
        assert dense(Kb.d_abs[k]).sum() == sum(dense(P.d_abs[k]).sum() for P in parts)


def test_batch_equals_individual_complexes(device):
    Kb, parts = make_batch(device)
    _check_batch_equals_parts(Kb, parts)
    x = torch.randn(Kb.n[0], 2, 3, device=device)
    y = Kb.apply_d(0, x)
    o0 = o1 = 0
    for P in parts:
        torch.testing.assert_close(y[o1:o1 + P.n[1]], P.apply_d(0, x[o0:o0 + P.n[0]].contiguous()))
        o0, o1 = o0 + P.n[0], o1 + P.n[1]


def test_batch_mixed_types_nested_and_errors(device):
    Kq, Km, Kt = make_complex("quads", device), make_complex("mixed", device), make_complex("grid", device)
    Kb = CochainComplex.batch([Kq, Km, Kt])
    _check_batch_equals_parts(Kb, [Kq, Km, Kt])
    assert Kb.meta["cell_type"] == "mixed" and Kb.cells[2].shape[1] == 6
    nested = CochainComplex.batch([CochainComplex.batch([Kq, Km]), Kt])
    _check_batch_equals_parts(nested, [Kq, Km, Kt])
    assert torch.equal(nested.batch[2], Kb.batch[2]) and nested.meta["sizes"] == Kb.meta["sizes"]
    one = batch_complexes([Kt])
    assert one.num_graphs == 1 and torch.equal(one.geo[1], Kt.geo[1])
    with pytest.raises(ValueError):
        CochainComplex.batch([Kt, make_complex("tets", device)])
    with pytest.raises(ValueError):
        CochainComplex.batch([Kt, make_complex("sphere", device)])  # 2-D vs 3-D positions
    Kb3 = CochainComplex.batch([make_complex("tets", device), make_complex("tets", device, n=30, seed=1)])
    _check_batch_equals_parts(Kb3, [make_complex("tets", device), make_complex("tets", device, n=30, seed=1)])


def test_to_device_and_dtype():
    K = make_complex("tets")
    for dev in DEVICES:
        K2 = K.to(dev)
        assert K2.device.type == dev and all(A.device.type == dev for A in K2.d + K2.dT + K2.d_abs + K2.dT_abs)
        assert all(t.device.type == dev for t in K2.cells + K2.geo + K2.star + K2.boundary)
        assert K2.meta["face_index"].device.type == dev and K2.meta["ptr"][0].device.type == dev
        K3 = K2.to("cpu")
        for k in range(K.dim):
            assert torch.equal(dense(K3.d[k]), dense(K.d[k]))
        assert_same_geometry(K3, K, rtol=0, atol=0)
    K64 = K.to(dtype=torch.float64)
    assert K64.d[0].dtype == torch.float64 and K64.geo[1].dtype == torch.float64 and K64.star[0].dtype == torch.float64
    assert K64.cells[1].dtype == torch.long and K64.boundary[0].dtype == torch.bool
    x = torch.randn(K.n[1], 2, 3, dtype=torch.float64)
    torch.testing.assert_close(K64.apply_d(1, x), (dense(K.d[1]) @ x.view(K.n[1], -1)).view(-1, 2, 3))


def test_apply_shape_errors():
    K = make_complex("grid")
    with pytest.raises(ValueError, match="rows"):
        K.apply_d(0, torch.randn(K.n[1], 2, 3))
    with pytest.raises(ValueError, match="out of range"):
        K.apply_d(2, torch.randn(K.n[2], 2, 3))
    with pytest.raises(ValueError, match="contiguous"):
        K.apply_d(0, torch.randn(2, K.n[0], 3).transpose(0, 1))
    ev = K.edge_vectors()
    torch.testing.assert_close(ev, K.pos[K.cells[1][:, 1]] - K.pos[K.cells[1][:, 0]])


def test_grid_conventions():
    nx, ny, h = 5, 4, 0.2
    for cell in ("tri", "quad"):
        K = CochainComplex.from_grid((nx, ny), h, cell=cell)
        v = torch.arange(nx * ny)
        torch.testing.assert_close(K.pos, torch.stack([(v // ny) * h, (v % ny) * h], 1).float())
        F = K.cells[2]
        P = K.pos.double()
        a = P[F[:, 1]] - P[F[:, 0]]
        b = P[F[:, 2]] - P[F[:, 0]]
        assert bool(((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) > 0).all())  # CCW
    Kt = CochainComplex.from_grid((nx, ny), h, cell="tri")
    assert Kt.meta["diagonal"] == "v1" and Kt.meta["grid_shape"] == (nx, ny) and Kt.meta["star_type"] == "cotan"
    assert CochainComplex.from_grid((nx, ny), h).meta["star_type"] == "barycentric"
    with pytest.raises(ValueError):
        CochainComplex.from_grid((1, 4))
    with pytest.raises(ValueError):
        CochainComplex.from_grid((3, 4), cell="hex")
    with pytest.raises(ValueError):
        CochainComplex.from_grid((3, 4), cell="tri", diagonal="x")


T1_PATH = os.path.join(ROOT, "datasets", "T1_cns_vorticity.pkl")


@pytest.mark.skipif(not os.path.exists(T1_PATH), reason="T1 dataset not available")
def test_grid_reproduces_v1_T1_triangulation():
    with open(T1_PATH, "rb") as f:
        d = pickle.load(f)
    K = CochainComplex.from_grid((32, 32), 1 / 31, cell="tri")
    assert np.array_equal(K.cells[2].numpy(), np.asarray(d["faces"]))
    assert np.abs(K.pos.numpy() - np.asarray(d["points"])).max() < 1e-6


def test_relabelling_permutes_geometry():
    pos, faces = mesh_delaunay_2d(60, seed=7)
    K = CochainComplex.from_triangles(pos, faces)
    pos2, faces2, perm = relabel_mesh(pos, faces, seed=1)
    K2 = CochainComplex.from_triangles(pos2, faces2)
    torch.testing.assert_close(K2.geo[0], K.geo[0][torch.as_tensor(perm)], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(K2.star[2], K.star[2], rtol=1e-6, atol=0)  # faces keep their order
    # edges map through perm (orientation may flip; even quantities must match)
    old = {tuple(e): i for i, e in enumerate(K.cells[1].tolist())}
    idx = [old[tuple(sorted((perm[a], perm[b])))] for a, b in K2.cells[1].tolist()]
    torch.testing.assert_close(K2.geo[1], K.geo[1][idx], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(K2.star[1], K.star[1][idx], rtol=1e-5, atol=1e-5)


def test_build_is_fast_cpu():
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(0)
    pts = rng.random((10000, 2))
    tri = Delaunay(pts).simplices
    t0 = time.perf_counter()
    K = CochainComplex.from_triangles(pts, tri)
    assert time.perf_counter() - t0 < 5.0 and K.check_d2() == 0.0
    assert K.meta["build_time_s"] < 5.0


def test_nonconvex_polygon_angles():
    # L-shaped hexagon (CCW) with a reflex corner at (1, 1), next to a unit square
    pos = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2], [3, 0], [3, 1.0]])
    faces = [[0, 1, 2, 3, 4, 5], [1, 6, 7, 2]]
    K = CochainComplex.from_polygons(pos, faces)
    assert K.check_d2() == 0.0
    names = GEO_FEATURE_NAMES
    torch.testing.assert_close(1.0 / K.star[2].double(), torch.tensor([3.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(K.geo[2][:, names[2].index("min_angle")], torch.full((2,), math.pi / 2))
    defect = K.geo[0][:, names[0].index("angle_defect")].double()
    assert abs(float(defect[3]) + math.pi / 2) < 1e-6  # reflex boundary corner: pi - 3pi/2
    assert abs(float(defect.sum()) - 2 * math.pi) < 1e-5  # turning number of the boundary
    for flip in (False, True):  # orientation of the input must not change even quantities
        Kf = CochainComplex.from_polygons(pos, [f[::-1] for f in faces] if flip else faces)
        assert_same_geometry(Kf, K)


# ----------------------------------------------------------------------------------------------
# Whitney / Galerkin metric blocks (DESIGN §9.1)
# ----------------------------------------------------------------------------------------------
def _p1_stiffness(pos, cells, n0):
    """Dense P1 (Lagrange) stiffness sum_f |f| grad(l_u).grad(l_v) in float64 from raw positions."""
    from rhmp.geometry import _simplex_measure, barycentric_gradients

    T = torch.as_tensor(cells)
    V = torch.as_tensor(pos, dtype=torch.float64)[T]
    g, m = barycentric_gradients(V), _simplex_measure(V)
    loc = torch.einsum("fid,fjd->fij", g, g) * m[:, None, None]
    S = torch.zeros(n0, n0, dtype=torch.float64)
    for i in range(T.shape[1]):
        for j in range(T.shape[1]):
            S.index_put_((T[:, i], T[:, j]), loc[:, i, j], accumulate=True)
    return S


@pytest.mark.parametrize("name", MESH_NAMES)
def test_whitney_blocks_present(name, device):
    K = make_complex(name, device)
    if name in POLY_MESHES:
        assert K.whitney == {}
        return
    degs = [1, 2] if name == "tets" else [1]
    assert sorted(K.whitney) == degs
    n_top = K.n[K.dim]
    for k in degs:
        W = K.whitney[k]
        mk, m = (3, 3) if K.dim == 2 else ((6, 6) if k == 1 else (4, 6))
        assert W["cells"].shape == (n_top, mk) and W["dir_edges"].shape == (n_top, m)
        assert W["G0"].shape == (n_top, mk, mk) and W["Gk"].shape == (n_top, m, mk, mk)
        assert W["t"].shape == (n_top, m, K.pos.shape[1])
        torch.testing.assert_close(W["t"].norm(dim=-1), torch.ones(n_top, m, device=device))
        assert tuple(W["gather"].shape) == (n_top * mk, K.n[k]) and tuple(W["scatter"].shape) == (K.n[k], n_top * mk)
        assert torch.equal(W["gather"].col_indices(), W["cells"].t().reshape(-1))  # slot-major
        ev0 = torch.linalg.eigvalsh(W["G0"].double())
        evk = torch.linalg.eigvalsh(W["Gk"].double())
        scale = W["G0"].double().abs().amax(dim=(1, 2))
        assert bool((ev0[:, 0] > 0).all()) and bool((evk >= -1e-6 * scale[:, None, None]).all())  # PD / PSD blocks
    if K.dim == 2:
        assert torch.equal(K.whitney[1]["cells"], K.meta["face_edges"])
    else:
        assert torch.equal(K.whitney[2]["cells"], K.meta["tet_faces"])


@pytest.mark.parametrize("name", ["grid", "sphere", "torus", "delaunay", "sliver", "nonmanifold", "tets"])
def test_whitney_p1_stiffness_identity(name):
    """d0^T H1(b=1, a=0) d0 is the P1 FEM stiffness (= cotan Laplacian for triangles)."""
    from rhmp.dec import assemble_whitney_metric

    pos, cells = raw_mesh(name)
    K = build(name, pos, cells, "cpu")
    H1 = dense(assemble_whitney_metric(K, 1))
    D0 = dense(K.d[0])
    S = _p1_stiffness(pos, cells, K.n[0])
    tol = 1e-6 if name != "tets" else 5e-6  # blocks are stored in float32
    assert float((D0.T @ H1 @ D0 - S).abs().max() / S.abs().max()) < tol
    if name in ("grid", "delaunay", "sphere"):  # and the cotan weights of the mesh (no clamp active on interior)
        ref = torch.tensor(np.clip(K.geo[1][:, GEO_FEATURE_NAMES[1].index("cotan_weight")].double().numpy(), -9, 9))
        ok = ref.abs() < 9
        Lc = D0.T @ torch.diag(torch.where(ok, ref, torch.zeros_like(ref))) @ D0
        if bool(ok.all()):
            assert float((Lc - S).abs().max() / S.abs().max()) < 1e-5


def test_whitney_curl_curl_identity():
    """Tets: d1^T H2(b=1, a=0) d1 is the lowest-order Nedelec curl-curl stiffness."""
    from rhmp.dec import assemble_whitney_metric
    from rhmp.geometry import _simplex_measure, barycentric_gradients

    pos, tets = mesh_tet_cube()
    K = CochainComplex.from_tetrahedra(pos, tets)
    T = K.cells[3]
    V = torch.as_tensor(pos, dtype=torch.float64)[T]
    g, vol = barycentric_gradients(V), _simplex_measure(V)
    pairs = torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    W1 = K.whitney[1]
    curl = 2 * torch.linalg.cross(g[:, pairs[:, 0]], g[:, pairs[:, 1]], dim=-1) * W1["signs"].double()[..., None]
    loc = torch.einsum("fid,fjd->fij", curl, curl) * vol[:, None, None]
    Kc = torch.zeros(K.n[1], K.n[1], dtype=torch.float64)
    for i in range(6):
        for j in range(6):
            Kc.index_put_((W1["cells"][:, i], W1["cells"][:, j]), loc[:, i, j], accumulate=True)
    D1 = dense(K.d[1])
    H2 = dense(assemble_whitney_metric(K, 2))
    assert float((D1.T @ H2 @ D1 - Kc).abs().max() / Kc.abs().max()) < 5e-6


@pytest.mark.parametrize("name,k", [("delaunay", 1), ("sphere", 1), ("sliver", 1), ("tets", 1), ("tets", 2)])
def test_whitney_metric_is_spd(name, k):
    from rhmp.dec import assemble_whitney_metric

    K = make_complex(name)
    W = K.whitney[k]
    g = torch.Generator().manual_seed(k)
    for trial in range(3):
        b = torch.exp(torch.randn(W["G0"].shape[0], generator=g, dtype=torch.float64))
        a = 3.0 * torch.rand(W["G0"].shape[0], W["t"].shape[1], generator=g, dtype=torch.float64)
        if trial == 2:
            a = a * (torch.rand(a.shape, generator=g) < 0.3)  # sparse anisotropy
        H = dense(assemble_whitney_metric(K, k, b, a))
        torch.testing.assert_close(H, H.T)
        assert float(torch.linalg.eigvalsh(H)[0]) > 0


@pytest.mark.parametrize("name", ["delaunay", "sphere", "tets"])
@pytest.mark.parametrize("reflect", [False, True])
def test_whitney_blocks_are_E_n_invariant(name, reflect):
    pos, cells = raw_mesh(name)
    D = pos.shape[1]
    Q = random_orthogonal(D, seed=21, reflect=reflect)
    K1 = build(name, pos, cells, "cpu")
    K2 = build(name, pos @ Q.T + 2.5, cells, "cpu")
    for k in K1.whitney:
        W1, W2 = K1.whitney[k], K2.whitney[k]
        for key in ("G0", "Gk", "rowsum_G0", "rowsum_Gk"):
            torch.testing.assert_close(W2[key], W1[key], rtol=1e-5, atol=1e-5 * float(W1[key].abs().max()))
        torch.testing.assert_close(W2["t"], W1["t"] @ torch.tensor(Q.T, dtype=torch.float32), rtol=1e-5, atol=1e-5)
        assert torch.equal(W2["cells"], W1["cells"]) and torch.equal(W2["signs"], W1["signs"])


def test_whitney_and_beta_unit_survive_batch_and_to(device):
    from rhmp.ops import beta_unit

    parts = [make_complex("delaunay", device, n=40), make_complex("grid", device), make_complex("sliver", device)]
    Kb = CochainComplex.batch(parts)
    assert sorted(Kb.whitney) == [1]
    W = Kb.whitney[1]
    o_top = o_e = 0
    for P in parts:
        Wp = P.whitney[1]
        rows = slice(o_top, o_top + P.n[2])
        for key in ("G0", "Gk", "t", "signs", "rowsum_G0"):
            assert torch.equal(W[key][rows], Wp[key])
        assert torch.equal(W["cells"][rows], Wp["cells"] + o_e) and torch.equal(W["dir_edges"][rows], Wp["dir_edges"] + o_e)
        o_top, o_e = o_top + P.n[2], o_e + P.n[1]
    x = torch.randn(Kb.n[1], 2, 3, device=device)
    xl = W["gather"].to_dense() @ x.view(Kb.n[1], -1)  # slot-major rows: j * n_top + f
    assert torch.equal(xl.view(3, -1, 2, 3), x[W["cells"].t()])
    for key, val in Kb.meta["beta_unit"].items():
        assert val.shape == (3,)
        assert torch.equal(val, torch.cat([P.meta["beta_unit"][key] for P in parts]))
    assert CochainComplex.batch([make_complex("quads", device), parts[1]]).whitney == {}
    Kt = make_complex("tets", device)
    for dev in DEVICES:
        K2 = Kt.to(dev)
        assert K2.whitney[2]["G0"].device.type == dev and K2.whitney[2]["gather"].device.type == dev
    K64 = Kt.to(dtype=torch.float64)
    assert K64.whitney[1]["Gk"].dtype == torch.float64 and K64.whitney[1]["cells"].dtype == torch.long
    # precomputed beta_unit matches the function and bounds lambda_max of S^-1/2 A^T A S^-1/2
    K = parts[0]
    bu = K.meta["beta_unit"]["up0_dec"]
    torch.testing.assert_close(bu, beta_unit(K.d_abs[0], K.dT_abs[0], K.star[0].rsqrt()).reshape(-1))
    s = K.star[0].double().cpu().rsqrt()
    D0 = dense(K.d[0])
    lam = torch.linalg.eigvalsh(s[:, None] * (D0.T @ D0) * s[None, :])[-1]
    assert float(bu) >= float(lam) * (1 - 1e-6)
