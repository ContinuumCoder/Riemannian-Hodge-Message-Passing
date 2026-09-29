"""Example 01: build cochain complexes and look at what they cache.

Covers triangle meshes (planar and surfaces), quads and mixed polygons, tetrahedra and regular grids; the exact
coboundaries ``d_k`` (``d_{k+1} d_k = 0``); reference Hodge stars; E(n)-invariant descriptors; boundary flags;
applying ``d_k`` / ``d_k^T`` to ``(n_k, B, C)`` cochains; validation of invalid faces; dtype/device moves; and
block-diagonal batches of several meshes.

Run (CPU, a few seconds):  python examples/01_build_complex.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import Delaunay

try:
    import rhmp  # noqa: F401
except ImportError:  # running from a source checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rhmp import CochainComplex
from rhmp.complex import validate_triangles


def square_points(n: int, seed: int) -> np.ndarray:
    """``n`` random points in the unit square plus its 4 corners."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.random((n, 2)), [[0, 0], [1, 0], [0, 1], [1, 1]]])


def torus_mesh(n_major: int = 16, n_minor: int = 8, R: float = 1.0, r: float = 0.4):
    """Closed triangulated torus in 3-D (genus 1): ``pos (n0, 3)``, ``faces (n2, 3)`` consistently oriented."""
    u = 2 * np.pi * np.arange(n_major) / n_major
    w = 2 * np.pi * np.arange(n_minor) / n_minor
    uu, ww = np.meshgrid(u, w, indexing="ij")
    pos = np.stack([(R + r * np.cos(ww)) * np.cos(uu), (R + r * np.cos(ww)) * np.sin(uu), r * np.sin(ww)], -1)
    i, j = np.meshgrid(np.arange(n_major), np.arange(n_minor), indexing="ij")
    i, j = i.ravel(), j.ravel()
    ip, jp = (i + 1) % n_major, (j + 1) % n_minor
    v00, v10, v11, v01 = i * n_minor + j, ip * n_minor + j, ip * n_minor + jp, i * n_minor + jp
    faces = np.concatenate([np.stack([v00, v10, v11], 1), np.stack([v00, v11, v01], 1)])
    return pos.reshape(-1, 3), faces


def euler_characteristic(K: CochainComplex) -> int:
    """``sum_k (-1)^k n_k`` (1 for a disk or a ball, 0 for a torus)."""
    return sum((-1) ** k * n for k, n in enumerate(K.n))


def describe(name: str, K: CochainComplex) -> None:
    print(f"\n[{name}] {K}")
    print(f"  cells per degree n = {K.n}, Euler characteristic = {euler_characteristic(K)}")
    print(f"  max |d_(k+1) d_k| = {K.check_d2()}  (exactly 0: oriented incidence, integer arithmetic)")
    for k in range(K.dim + 1):
        s = K.star[k]
        print(f"  degree {k}: star in [{float(s.min()):.3g}, {float(s.max()):.3g}], "
              f"{int(K.boundary[k].sum())} boundary cells, geo columns {K.meta['geo_features'][k]}")


def main() -> None:
    torch.manual_seed(0)

    # ------------------------------------------------------------------ 1. a planar triangle mesh
    pts = square_points(60, seed=0)
    tri = Delaunay(pts).simplices                      # numpy int32 is accepted; orientation is kept as given
    K = CochainComplex.from_triangles(pts, tri)        # star='cotan' (default), validate=True
    describe("planar Delaunay mesh", K)
    print(f"  K.geo_dims = {K.geo_dims}   <- pass this to RHMP(cfg, geo_dims)")
    print(f"  edges are stored with src < dst: first edges {K.cells[1][:3].tolist()}")
    print(f"  face 0 = vertices {K.cells[2][0].tolist()}, its edges {K.meta['face_edges'][0].tolist()} "
          f"with orientation signs {K.meta['face_edge_signs'][0].tolist()}")

    # cochains use the layout (n_k, B, C): B samples sharing the mesh, C channels
    u = torch.randn(K.n[0], 2, 3)
    du = K.apply_d(0, u)                                # (n1, B, C): differences along edges
    ddu = K.apply_d(1, du)                              # (n2, B, C): curl of a gradient
    print(f"  d0 u: {tuple(du.shape)}, |d1 (d0 u)|_max = {float(ddu.abs().max()):.1e} "
          "(fp32 round-off of two products; the operator d1 d0 itself is exactly 0)")
    print(f"  d0^T d0 u (graph Laplacian): {tuple(K.apply_dT(0, du).shape)}")
    print(f"  edge vectors pos[dst] - pos[src]: {tuple(K.edge_vectors().shape)}")

    # ------------------------------------------------------------------ 2. a closed surface in 3-D
    pos, faces = torus_mesh()
    describe("torus (surface in 3-D)", CochainComplex.from_triangles(pos, faces))

    # ------------------------------------------------------------------ 3. quads, mixed polygons, grids
    Kq = CochainComplex.from_grid((6, 5), spacing=0.2, cell="quad")        # CCW quads, star='barycentric'
    describe("quad grid", Kq)
    grid = np.array([[x, y] for x in (0.0, 0.5, 1.0) for y in (0.0, 0.5, 1.0)])   # vertex id = 3 i + j
    polys = [[0, 3, 4], [0, 4, 1], [1, 4, 5, 2], [3, 6, 7, 4], [4, 7, 8, 5]]       # triangles + quads
    Kp = CochainComplex.from_polygons(grid, polys)                                 # list of lists, any length
    describe("mixed triangles and quads", Kp)
    print(f"  polygon faces are -1 padded: cells[2] =\n{Kp.cells[2]}")
    Kt1 = CochainComplex.from_grid((32, 32), 1 / 31, cell="tri")           # the T1 triangulation of the paper
    print(f"\n[32x32 triangulated grid (T1 mesh)] {Kt1}")

    # ------------------------------------------------------------------ 4. tetrahedra (degrees 0..3)
    rng = np.random.default_rng(1)
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float)
    pts3 = np.concatenate([rng.random((40, 3)), corners])
    Ktet = CochainComplex.from_tetrahedra(pts3, Delaunay(pts3).simplices)   # star='barycentric'
    describe("tetrahedral mesh of the unit cube", Ktet)
    print(f"  precomputed Whitney (Galerkin) metric blocks for degrees {sorted(Ktet.whitney)}")

    # ------------------------------------------------------------------ 5. invalid faces are dropped by validation
    bad = np.concatenate([tri, tri[:2, ::-1], [[0, 0, 1]]])   # 2 duplicates (reversed) + 1 repeated vertex
    clean, stats = validate_triangles(pts, bad)
    print(f"\n[validation] kept {stats['n_kept']} of {stats['n_input']} faces "
          f"({stats['n_duplicate']} duplicates, {stats['n_degenerate_index']} with repeated vertices)")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Kv = CochainComplex.from_triangles(pts, bad)          # validate=True drops them with a RuntimeWarning
    print(f"  from_triangles warned: {caught[0].message if caught else 'no warning'}")
    print(f"  meta['face_index'] maps kept faces to input rows (first 5: {Kv.meta['face_index'][:5].tolist()})")

    # ------------------------------------------------------------------ 6. dtype / device moves
    K64 = K.to(dtype=torch.float64)                            # float tensors (operators, stars, geo) in fp64
    print(f"\n[to] fp64 star dtype: {K64.star[0].dtype}, operators: {K64.d[0].dtype}")
    if torch.cuda.is_available():
        print(f"  cuda copy: {K.to('cuda')}")

    # ------------------------------------------------------------------ 7. block-diagonal batch of meshes
    parts = [CochainComplex.from_triangles(p, Delaunay(p).simplices)
             for p in (square_points(30, 2), square_points(50, 3), square_points(80, 4))]
    Kb = CochainComplex.batch(parts)                           # class-level call (K.batch is the id list)
    print(f"\n[batch] {Kb}")
    print(f"  per-graph sizes {Kb.meta['sizes']}, vertex offsets meta['ptr'][0] = {Kb.meta['ptr'][0].tolist()}")
    print(f"  K.batch[0] (graph id per vertex) counts: {torch.bincount(Kb.batch[0]).tolist()}, "
          f"max |d d| = {Kb.check_d2()}")

    assert K.check_d2() == 0.0 and Ktet.check_d2() == 0.0 and Kb.check_d2() == 0.0
    assert euler_characteristic(Ktet) == 1 and euler_characteristic(Kq) == 1
    print("\nEXAMPLE 01 OK")


if __name__ == "__main__":
    main()
