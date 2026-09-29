"""Shared test fixtures and importable mesh helpers.

Plain helpers (usable from any test module via ``from conftest import ...`` or through the fixtures):

* ``mesh_plane_grid``, ``mesh_sphere``, ``mesh_torus``, ``mesh_delaunay_2d``, ``mesh_sliver``,
  ``mesh_nonmanifold``, ``mesh_tet_cube``, ``mesh_quad_grid``, ``mesh_mixed_polygons``: return
  ``(pos, cells)`` as numpy arrays (``cells`` is a ``-1``-padded int array for polygons).
* ``make_complex(name, device='cpu', star=None, **kw)``: build a :class:`CochainComplex` for a mesh name
  in :data:`MESH_NAMES` (``'grid', 'sphere', 'torus', 'delaunay', 'sliver', 'nonmanifold', 'tets',
  'quads', 'mixed'``).
* ``make_batch(device='cpu', seed=0)``: ``(block-diagonal batch, [3 parts])`` of 3 different random
  Delaunay meshes.
* ``random_orthogonal(D, seed, reflect)``, ``relabel_mesh(pos, cells, seed)``.
* ``DEVICES``: ``['cpu'] (+ ['cuda'])``.

Fixtures: ``device`` (parametrised over :data:`DEVICES`), ``grid_complex``, ``sphere_complex``,
``torus_complex``, ``delaunay_complex``, ``sliver_complex``, ``tet_complex``, ``quad_complex``,
``mixed_complex``, ``batch_complex`` (all on ``device``), and ``any_complex`` (parametrised over all
mesh names, on ``device``).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rhmp.complex import CochainComplex  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
MESH_NAMES = ("grid", "sphere", "torus", "delaunay", "sliver", "nonmanifold", "tets", "quads", "mixed")


# ----------------------------------------------------------------------------------------------
# raw meshes (numpy)
# ----------------------------------------------------------------------------------------------
def mesh_plane_grid(nx: int = 7, ny: int = 6, jitter: float = 0.15, seed: int = 0, dim: int = 2):
    """Triangulated grid of ``[0,1]^2`` with boundary (alternating diagonals, interior jitter).

    Returns ``pos (nx*ny, dim)``, ``faces (2 (nx-1)(ny-1), 3)`` CCW.
    """
    rng = np.random.default_rng(seed)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    h = 1.0 / (max(nx, ny) - 1)
    pos = np.stack([ii.ravel() * h, jj.ravel() * h], -1).astype(np.float64)
    interior = ((ii > 0) & (ii < nx - 1) & (jj > 0) & (jj < ny - 1)).ravel()
    pos[interior] += jitter * h * rng.uniform(-1, 1, size=(interior.sum(), 2))
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    i, j = i.ravel(), j.ravel()
    v00, v10, v11, v01 = i * ny + j, (i + 1) * ny + j, (i + 1) * ny + j + 1, i * ny + j + 1
    alt = ((i + j) % 2 == 0)[:, None]
    t1 = np.where(alt, np.stack([v00, v10, v11], 1), np.stack([v00, v10, v01], 1))
    t2 = np.where(alt, np.stack([v00, v11, v01], 1), np.stack([v10, v11, v01], 1))
    faces = np.concatenate([t1, t2])
    if dim == 3:
        pos = np.concatenate([pos, np.zeros((pos.shape[0], 1))], 1)
    return pos, faces


def mesh_sphere(subdiv: int = 2, radius: float = 1.0):
    """Icosphere (closed, genus 0, outward CCW faces). ``subdiv=2``: 162 vertices, 320 faces."""
    t = (1.0 + 5 ** 0.5) / 2.0
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t],
                  [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], dtype=np.float64)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4],
                  [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8],
                  [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]], dtype=np.int64)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    for _ in range(subdiv):
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        key = np.minimum(e[:, 0], e[:, 1]) * len(v) + np.maximum(e[:, 0], e[:, 1])
        uk, inv = np.unique(key, return_inverse=True)
        a, b = uk // len(v), uk % len(v)
        mid = v[a] + v[b]
        mid /= np.linalg.norm(mid, axis=1, keepdims=True)
        m = (inv + len(v)).reshape(3, -1).T  # midpoints of (01, 12, 20)
        v = np.concatenate([v, mid])
        f = np.concatenate([np.stack([f[:, 0], m[:, 0], m[:, 2]], 1), np.stack([f[:, 1], m[:, 1], m[:, 0]], 1),
                            np.stack([f[:, 2], m[:, 2], m[:, 1]], 1), m])
    return radius * v, f


def mesh_torus(n_major: int = 12, n_minor: int = 8, R: float = 1.0, r: float = 0.4):
    """Closed torus (genus 1), 3-D."""
    u = 2 * np.pi * np.arange(n_major) / n_major
    w = 2 * np.pi * np.arange(n_minor) / n_minor
    uu, ww = np.meshgrid(u, w, indexing="ij")
    pos = np.stack([(R + r * np.cos(ww)) * np.cos(uu), (R + r * np.cos(ww)) * np.sin(uu), r * np.sin(ww)], -1)
    pos = pos.reshape(-1, 3)
    i, j = np.meshgrid(np.arange(n_major), np.arange(n_minor), indexing="ij")
    i, j = i.ravel(), j.ravel()
    ip, jp = (i + 1) % n_major, (j + 1) % n_minor
    v00, v10, v11, v01 = i * n_minor + j, ip * n_minor + j, ip * n_minor + jp, i * n_minor + jp
    faces = np.concatenate([np.stack([v00, v10, v11], 1), np.stack([v00, v11, v01], 1)])
    return pos, faces


def mesh_delaunay_2d(n: int = 150, seed: int = 0):
    """Random 2-D Delaunay triangulation of the unit square (hull slivers occur naturally)."""
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(seed)
    pts = np.concatenate([rng.random((n - 4, 2)), np.array([[0, 0], [1, 0], [0, 1], [1, 1.0]])])
    return pts, Delaunay(pts).simplices.astype(np.int64)


def mesh_sliver(aspect: float = 1e4, nx: int = 7, ny: int = 6):
    """Plane grid whose second row is squashed onto the first: a strip of triangles with aspect ~``aspect``."""
    pos, faces = mesh_plane_grid(nx, ny, jitter=0.0)
    h = 1.0 / (max(nx, ny) - 1)
    row1 = np.arange(nx) * ny + 1
    pos[row1, 1] = h / aspect
    return pos, faces


def mesh_nonmanifold():
    """Three triangles sharing the edge (0, 1) (a non-manifold 'book'), plus one regular neighbour."""
    pos = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, -1, 0], [0.5, 0.2, 1], [1.5, 1.0, 0.0]], dtype=np.float64)
    faces = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4], [1, 5, 2]], dtype=np.int64)
    return pos, faces


def mesh_tet_cube(n: int = 60, seed: int = 0):
    """Delaunay tetrahedralisation of random points in the unit cube (corners included)."""
    from scipy.spatial import Delaunay

    rng = np.random.default_rng(seed)
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=np.float64)
    pts = np.concatenate([rng.random((n - 8, 3)), corners])
    return pts, Delaunay(pts).simplices.astype(np.int64)


def mesh_quad_grid(nx: int = 6, ny: int = 5, jitter: float = 0.1, seed: int = 0):
    """CCW quads of a (jittered) grid: ``faces (nx-1)(ny-1) x 4``."""
    rng = np.random.default_rng(seed)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    h = 1.0 / (max(nx, ny) - 1)
    pos = np.stack([ii.ravel() * h, jj.ravel() * h], -1).astype(np.float64)
    interior = ((ii > 0) & (ii < nx - 1) & (jj > 0) & (jj < ny - 1)).ravel()
    pos[interior] += jitter * h * rng.uniform(-1, 1, size=(interior.sum(), 2))
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    i, j = i.ravel(), j.ravel()
    faces = np.stack([i * ny + j, (i + 1) * ny + j, (i + 1) * ny + j + 1, i * ny + j + 1], 1)
    return pos, faces


def mesh_mixed_polygons(nx: int = 7, ny: int = 5):
    """Grid with triangles, quads and hexagons (merged quad pairs); ``-1`` padded ``(n2, 6)``."""
    pos, quads = mesh_quad_grid(nx, ny, jitter=0.0)
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    i, j = i.ravel(), j.ravel()
    polys = []
    used = np.zeros(len(quads), bool)
    for q in range(len(quads)):  # small fixture mesh: a Python loop is acceptable here
        if used[q]:
            continue
        a, b, c, d = quads[q]
        # hexagon: merge with the right neighbour (interior rows only, so that the straight-angle vertices keep
        # a third, non-collinear edge; a valence-2 boundary vertex would make per-vertex vector fits rank-1)
        if j[q] % 3 == 1 and 0 < j[q] < ny - 2 and i[q] % 2 == 0 and i[q] + 1 < nx - 1:
            r = q + (ny - 1)
            used[r] = True
            _, b2, c2, _ = quads[r]
            polys.append([a, b, b2, c2, c, d])
        elif (i[q] + j[q]) % 3 == 1:  # two triangles
            polys.append([a, b, c])
            polys.append([a, c, d])
        else:
            polys.append([a, b, c, d])
        used[q] = True
    M = max(map(len, polys))
    out = np.full((len(polys), M), -1, dtype=np.int64)
    for k, p in enumerate(polys):
        out[k, :len(p)] = p
    return pos, out


_MESHES = {
    "grid": mesh_plane_grid, "sphere": mesh_sphere, "torus": mesh_torus, "delaunay": mesh_delaunay_2d,
    "sliver": mesh_sliver, "nonmanifold": mesh_nonmanifold, "tets": mesh_tet_cube, "quads": mesh_quad_grid,
    "mixed": mesh_mixed_polygons,
}


def make_complex(name: str, device="cpu", star: str | None = None, **kw) -> CochainComplex:
    """Build the complex of mesh ``name`` (see :data:`MESH_NAMES`) on ``device``.

    ``star=None`` uses the builder default (triangles 'cotan', polygons/tets 'barycentric').
    Extra keyword arguments go to the mesh generator.
    """
    pos, cells = _MESHES[name](**kw)
    opts = {} if star is None else {"star": star}
    if name == "tets":
        return CochainComplex.from_tetrahedra(pos, cells, device=device, **opts)
    if name in ("quads", "mixed"):
        return CochainComplex.from_polygons(pos, cells, device=device, **opts)
    return CochainComplex.from_triangles(pos, cells, device=device, **opts)


def make_batch(device="cpu", seed: int = 0) -> tuple[CochainComplex, list[CochainComplex]]:
    """Block-diagonal batch of 3 different random Delaunay meshes (30, 50, 80 points)."""
    parts = [make_complex("delaunay", device=device, n=n, seed=seed + s) for s, n in enumerate((30, 50, 80))]
    return CochainComplex.batch(parts), parts


def random_orthogonal(D: int, seed: int = 0, reflect: bool = False) -> np.ndarray:
    """Random ``(D, D)`` orthogonal matrix with ``det = -1`` if ``reflect`` else ``+1``."""
    rng = np.random.default_rng(seed)
    Q, R = np.linalg.qr(rng.standard_normal((D, D)))
    Q = Q * np.sign(np.diag(R))
    if (np.linalg.det(Q) < 0) != reflect:
        Q[:, 0] = -Q[:, 0]
    return Q


def relabel_mesh(pos: np.ndarray, cells: np.ndarray, seed: int = 0):
    """Random vertex relabelling: returns ``(pos_new, cells_new, perm)`` with ``pos_new = pos[perm]``.

    ``cells_new`` refers to the new ids (``-1`` padding kept); old vertex ``perm[i]`` is new vertex ``i``.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(pos.shape[0])
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size)
    cells_new = np.where(cells >= 0, inv[np.maximum(cells, 0)], -1)
    return pos[perm], cells_new, perm


# ----------------------------------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------------------------------
@pytest.fixture(params=DEVICES)
def device(request) -> str:
    """Device string, parametrised over CPU and (if available) CUDA."""
    return request.param


@pytest.fixture
def grid_complex(device):
    return make_complex("grid", device)


@pytest.fixture
def sphere_complex(device):
    return make_complex("sphere", device)


@pytest.fixture
def torus_complex(device):
    return make_complex("torus", device)


@pytest.fixture
def delaunay_complex(device):
    return make_complex("delaunay", device)


@pytest.fixture
def sliver_complex(device):
    return make_complex("sliver", device)


@pytest.fixture
def tet_complex(device):
    return make_complex("tets", device)


@pytest.fixture
def quad_complex(device):
    return make_complex("quads", device)


@pytest.fixture
def mixed_complex(device):
    return make_complex("mixed", device)


@pytest.fixture
def batch_complex(device):
    return make_batch(device)[0]


@pytest.fixture(params=MESH_NAMES)
def any_complex(request, device):
    return make_complex(request.param, device)


@pytest.fixture(autouse=True)
def _restore_tf32_flags():
    """Trainer-based tests switch TF32 on for the whole process; restore the flags after every test."""
    if not torch.cuda.is_available():
        yield
        return
    m, c = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = m, c
