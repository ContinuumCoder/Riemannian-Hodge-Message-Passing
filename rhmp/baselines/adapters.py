"""Adapters between the v2 data layout / complexes and the baseline models.

* :class:`V1ComplexAdapter` - what the v1 baselines need from a :class:`rhmp.complex.CochainComplex` (COO ``d0``,
  ``d1``, ``pos``, ``edges``, ``faces``, ``n0..n2``, ``get_adjacency``), plus lazily cached derived operators used by
  the vectorised baseline cores (directed edge index, normalised edge geometry, sparse adjacencies and Laplacians,
  incidence-mean pooling).  A block-diagonal batch of meshes is just one big complex; everything that needs a
  per-mesh constant (length scale, Laplacian normalisation, global means) is computed per graph, so the prediction
  for one mesh never depends on the other meshes of the batch.  :func:`adapter_for` caches one adapter per complex
  object (weak references: shared-mesh tasks build it once, variable-mesh batches once per batch).
* :class:`NodeInputEncoder` - fixed (parameter-free) map from v2 per-degree inputs to node features for node-based
  baselines: node inputs pass through, even edge/face/tet inputs become incident-cell means, odd edge inputs become
  v1's directional encoding ``(avg, avg*dx, avg*dy[, avg*dz])`` (``encode_edges_to_nodes`` of the v1 generator
  ``gen_T6_wilson_loop.py``).  Optional fixed standardisation (buffers fitted on training samples, never trained).
* :class:`EdgeInputEncoder` - per-directed-edge features for MeshGraphNet.
* Output heads (:class:`NodeHead`, :class:`PairEdgeHead`, :class:`FaceHead`, :class:`CellHead`,
  :class:`ComplexHead`) that map hidden baseline features to the task's target cells.

Orientation conventions (identical to v1 and to ``rhmp.complex``): edges are stored ``src < dst``; the value of an
edge cochain refers to the orientation ``src -> dst``; faces keep the input orientation.  All sparse operators are
CSR with a precomputed transpose (``rhmp.ops.spmm``), built once per complex under ``torch.no_grad()``.
"""
from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rhmp.ops import csr_from_coo, segment_max, segment_mean, spmm

__all__ = [
    "V1ComplexAdapter", "adapter_for", "csr_to_coo", "TaskIO", "task_io",
    "NodeInputEncoder", "EdgeInputEncoder",
    "mlp", "NodeHead", "PairEdgeHead", "FaceHead", "CellHead", "ComplexHead",
    "TRIANGLE_ONLY_MODELS",
]

EDGE_EPS = 1e-12      # v1 encode_edges_to_nodes: unit direction = edge / (|edge| + 1e-12)
STD_FLOOR = 1e-6      # standardisation of encoded features (constant columns keep std 1)
TRIANGLE_ONLY_MODELS = ("ours_v1",)   # v1 code paths that read ``faces`` (n2, 3) directly


# ================================================================================================================
# sparse helpers
# ================================================================================================================
def _rows_from_crow(crow: Tensor) -> Tensor:
    counts = crow[1:] - crow[:-1]
    return torch.repeat_interleave(torch.arange(counts.numel(), device=crow.device), counts)


def csr_to_coo(A: Tensor) -> Tensor:
    """CSR ``(m, n)`` -> coalesced COO with the same values (v1 operators are COO)."""
    rows = _rows_from_crow(A.crow_indices())
    idx = torch.stack([rows, A.col_indices()])
    return torch.sparse_coo_tensor(idx, A.values(), tuple(A.shape)).coalesce()


def _triplets(A: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """``(rows, cols, vals)`` of a CSR matrix."""
    return _rows_from_crow(A.crow_indices()), A.col_indices(), A.values()


def _pair(rows: Tensor, cols: Tensor, vals: Tensor, shape: tuple[int, int]) -> tuple[Tensor, Tensor]:
    """CSR of ``A`` and of ``A^T`` from unique COO triplets."""
    A = csr_from_coo(rows, cols, vals.float(), shape)
    AT = csr_from_coo(cols, rows, vals.float(), (shape[1], shape[0]))
    return A, AT


def _row_normalized(rows: Tensor, cols: Tensor, shape: tuple[int, int]) -> tuple[Tensor, Tensor]:
    """Row-mean operator (every row averages its nonzero columns; empty rows stay zero)."""
    cnt = torch.bincount(rows, minlength=shape[0]).clamp_min(1).to(torch.float32)
    return _pair(rows, cols, 1.0 / cnt[rows], shape)


def _coo(rows: Tensor, cols: Tensor, vals: Tensor, shape: tuple[int, int]) -> Tensor:
    return torch.sparse_coo_tensor(torch.stack([rows, cols]), vals.float(), shape).coalesce()


def _product(A: Tensor, B: Tensor) -> Tensor:
    """Sparse COO product (coalesced)."""
    return torch.sparse.mm(A, B).coalesce()


def _offdiag_pattern(P: Tensor) -> tuple[Tensor, Tensor]:
    """Row/col indices of the off-diagonal nonzeros of a coalesced COO matrix (binary adjacency)."""
    i = P.indices()
    keep = (i[0] != i[1]) & (P.values() != 0)
    return i[0][keep], i[1][keep]


# ================================================================================================================
# complex adapter
# ================================================================================================================
class V1ComplexAdapter:
    """v1 ``CellComplex`` view of a v2 :class:`~rhmp.complex.CochainComplex` plus cached baseline operators.

    v1 attributes (as ``rhmp.baselines.v1.gauge_hodge_mp.cell_complex.CellComplex``):
        pos ``(n0, D)`` float32, edges ``(n1, 2)`` int64 (``src < dst``, lexicographic = v1 order),
        faces ``(n2, 3)`` int64 (triangle complexes only), d0 ``(n1, n0)`` and d1 ``(n2, n1)`` coalesced COO float32
        (built once from the CSR operators, identical to v1's ``from_triangulation``), n0 / n1 / n2,
        ``get_adjacency(0, 0)`` and ``get_adjacency(1, 1)`` (computed sparsely; ``d1`` is never densified).

    Derived operators (lazy, cached): see :meth:`op`, :meth:`directed_edges`, :meth:`directed_geometry`,
    :meth:`edge_length_normalized`, :meth:`face_orientation`, :meth:`grid_shape`.

    Args:
        K: complex (single mesh or block-diagonal batch).  The adapter keeps references to the complex's tensors
            (not to the complex object itself).
    """

    def __init__(self, K: Any) -> None:
        self.dim = int(K.dim)
        self.cell_type = K.meta.get("cell_type")
        self.device = K.pos.device
        self.pos: Tensor = K.pos
        self.edges: Tensor = K.cells[1].long()
        self.cells: list[Tensor] = list(K.cells)
        self.n: list[int] = [int(v) for v in K.n]
        self.n0, self.n1 = self.n[0], self.n[1]
        self.n2 = self.n[2] if self.dim >= 2 else 0
        self.num_graphs = int(K.num_graphs)
        self.batch: list[Tensor] | None = None if K.batch is None else list(K.batch)
        self.d, self.dT = list(K.d), list(K.dT)
        self.d_abs, self.dT_abs = list(K.d_abs), list(K.dT_abs)
        self.star_type = K.meta.get("star_type")
        self._grid_meta = K.meta.get("grid_shape")
        self._face_lengths = K.meta.get("face_lengths")
        self._cache: dict[str, Any] = {}

    def __repr__(self) -> str:
        return (f"V1ComplexAdapter(dim={self.dim}, n={self.n}, cell_type={self.cell_type!r}, "
                f"num_graphs={self.num_graphs}, device={self.device})")

    # ------------------------------------------------------------------ cache
    def _get(self, key: str, fn: Callable[[], Any]) -> Any:
        if key not in self._cache:
            with torch.no_grad():
                self._cache[key] = fn()
        return self._cache[key]

    # ------------------------------------------------------------------ v1 interface
    @property
    def d0(self) -> Tensor:
        """v1 ``d0``: coalesced COO ``(n1, n0)`` (``-1`` at src, ``+1`` at dst)."""
        return self._get("coo_d0", lambda: csr_to_coo(self.d[0]))

    @property
    def d1(self) -> Tensor:
        """v1 ``d1``: coalesced COO ``(n2, n1)`` (face orientation signs)."""
        if self.dim < 2:
            raise NotImplementedError("d1 needs a complex of dimension >= 2")
        return self._get("coo_d1", lambda: csr_to_coo(self.d[1]))

    @property
    def faces(self) -> Tensor:
        """v1 ``faces (n2, 3)``; triangle complexes only."""
        if self.cell_type != "triangle" or self.dim != 2:
            raise NotImplementedError(
                f"faces (n2, 3) exist only for triangle surface complexes (this complex: cell_type="
                f"{self.cell_type!r}, dim={self.dim}); v1 code paths that need them ({', '.join(TRIANGLE_ONLY_MODELS)})"
                " are inapplicable here.  The vectorised baseline cores (gcn, gat, schnet, egnn, mgn, mpsn, sccnn, "
                "cw_net, clifford_smpn, deeponet) work on polygon and tetrahedral complexes; gauge_cnn/gem_cnn need a "
                "2-dimensional complex, fno a regular grid.")
        return self.cells[2].long()

    def get_adjacency(self, dim_from: int, dim_to: int) -> Tensor:
        """v1 ``get_adjacency``: ``(0, 0)`` both edge directions ``(2, 2 n1)``; ``(1, 1)`` edges sharing a face,
        ``(2, E)`` in row-major sorted order (as ``d1_dense.T @ d1_dense`` -> ``nonzero()`` in v1), sparse."""
        if dim_from == 0 and dim_to == 0:
            return self._get("adj00", lambda: torch.stack([
                torch.cat([self.edges[:, 0], self.edges[:, 1]]), torch.cat([self.edges[:, 1], self.edges[:, 0]])]))
        if dim_from == 1 and dim_to == 1:
            def build():
                P = _product(csr_to_coo(self.dT_abs[1]), csr_to_coo(self.d_abs[1]))
                r, c = _offdiag_pattern(P)
                return torch.stack([r, c])
            return self._get("adj11", build)
        raise NotImplementedError(f"Adjacency {dim_from}->{dim_to} not implemented")

    def to(self, device) -> "V1ComplexAdapter":
        """v1 compatibility: adapters live on the device of their complex."""
        if torch.device(device) != self.device and not (torch.device(device).type == self.device.type == "cuda"):
            raise ValueError("V1ComplexAdapter.to: build a new adapter from K.to(device) instead")
        return self

    # ------------------------------------------------------------------ graph ids / geometry
    def graph_ids(self, k: int) -> Tensor:
        """``(n_k,)`` int64 mesh id of every k-cell (zeros for a single complex)."""
        if self.batch is not None:
            return self.batch[k]
        return self._get(f"gid{k}", lambda: torch.zeros(self.n[k], dtype=torch.long, device=self.device))

    def directed_edges(self) -> tuple[Tensor, Tensor]:
        """``(send, recv)`` of the ``2 n1`` directed edges: forward ``src -> dst`` first, then the reverses."""
        def build():
            s, d = self.edges[:, 0], self.edges[:, 1]
            return torch.cat([s, d]).contiguous(), torch.cat([d, s]).contiguous()
        return self._get("dir_edges", build)

    def edge_vectors(self) -> Tensor:
        """``(n1, D)`` ``pos[dst] - pos[src]``."""
        return self._get("edge_vec", lambda: self.pos[self.edges[:, 1]] - self.pos[self.edges[:, 0]])

    def edge_lengths(self) -> Tensor:
        """``(n1,)`` edge lengths."""
        return self._get("edge_len", lambda: self.edge_vectors().norm(dim=-1))

    def edge_units(self) -> Tensor:
        """``(n1, D)`` v1 unit directions ``(pos[dst] - pos[src]) / (|.| + 1e-12)``."""
        return self._get("edge_unit", lambda: self.edge_vectors() / (self.edge_lengths()[:, None] + EDGE_EPS))

    def length_scale(self) -> Tensor:
        """``(num_graphs,)`` mean edge length of every mesh (per-mesh geometric scale)."""
        return self._get("len_scale", lambda: segment_mean(self.edge_lengths(), self.graph_ids(1),
                                                           self.num_graphs).clamp_min(1e-30))

    def edge_length_normalized(self) -> Tensor:
        """``(n1,)`` edge length divided by the mean edge length of its mesh."""
        return self._get("len_norm", lambda: self.edge_lengths() / self.length_scale()[self.graph_ids(1)])

    def directed_geometry(self, normalized: bool = True) -> tuple[Tensor, Tensor]:
        """Relative position ``pos[recv] - pos[send]`` ``(2 n1, D)`` and length ``(2 n1,)`` of the directed edges,
        divided by the mean edge length of the mesh when ``normalized`` (resolution/scale free)."""
        def build():
            ev, ln = self.edge_vectors(), self.edge_lengths()
            if normalized:
                s = self.length_scale()[self.graph_ids(1)]
                ev, ln = ev / s[:, None], ln / s
            return torch.cat([ev, -ev]).contiguous(), torch.cat([ln, ln]).contiguous()
        return self._get(f"dir_geo_{int(normalized)}", build)

    def face_orientation(self) -> Tensor | None:
        """``(n2,)`` +-1 sign of the signed area of every face for planar 2-D complexes (+1 = counter-clockwise; the
        physical orientation of the plane), ``None`` for surfaces in 3-D and volumes."""
        if self.dim != 2 or self.pos.shape[1] != 2:
            return None

        def build():
            Fc = self.cells[2].long()
            n2, m = Fc.shape
            L = (Fc >= 0).sum(1)
            P = self.pos.double()[Fc.clamp_min(0)]                               # (n2, m, 2)
            nxt = (torch.arange(m, device=Fc.device)[None, :] + 1) % L[:, None]
            Pn = torch.gather(P, 1, nxt[..., None].expand(n2, m, 2))
            cross = P[..., 0] * Pn[..., 1] - Pn[..., 0] * P[..., 1]
            cross = torch.where(torch.arange(m, device=Fc.device)[None, :] < L[:, None], cross, 0.0)
            sa = cross.sum(1)
            return torch.where(sa >= 0, 1.0, -1.0).float()
        return self._get("face_sigma", build)

    def grid_shape(self) -> tuple[int, int] | None:
        """``(nx, ny)`` if the vertices form a regular 2-D grid numbered ``i * ny + j`` (v1 T1 convention)."""
        if self._grid_meta is not None:
            return tuple(int(v) for v in self._grid_meta)
        if self.num_graphs != 1 or self.pos.shape[1] != 2:
            return None

        def build():
            p = self.pos.double()
            xs, ys = torch.unique(p[:, 0]), torch.unique(p[:, 1])
            nx, ny = xs.numel(), ys.numel()
            if nx * ny != self.n0 or nx < 2 or ny < 2:
                return None
            gx, gy = torch.meshgrid(xs, ys, indexing="ij")
            ref = torch.stack([gx.reshape(-1), gy.reshape(-1)], 1)
            tol = 1e-6 * float((p.max(0).values - p.min(0).values).max().clamp_min(1e-30))
            return (int(nx), int(ny)) if bool(((ref - p).abs() <= tol).all()) else None
        return self._get("grid_shape", build)

    # ------------------------------------------------------------------ sparse operators
    def op(self, name: str) -> tuple[Tensor, Tensor]:
        """Cached sparse operator ``(A, A^T)`` (CSR float32).

        Names:
            ``gcn_adj``   ``D^-1/2 (A + I) D^-1/2`` on nodes (v1 GCN).
            ``A00``       nodes sharing an edge; ``A11lo`` edges sharing a node; ``A11up`` edges sharing a face;
                          ``A22`` faces sharing an edge (binary, no self loops; v1 MPSN).
            ``L0``, ``L1``, ``L2``   combinatorial Hodge Laplacians ``d0^T d0``, ``d0 d0^T + d1^T d1``,
                          ``d1 d1^T`` (v1 SCCNN; on tetrahedral complexes the 2-skeleton, i.e. no ``d2^T d2``);
                          suffix ``_n``: every mesh block divided by its Gershgorin bound (max abs row sum).
            ``mean_{a}to{b}``  incidence mean from degree ``a`` to degree ``b`` (row = target cell, averaging the
                          incident source cells), e.g. ``mean_1to0`` (node <- incident edges), ``mean_0to2`` (face
                          <- its vertices), ``mean_1to2`` (face <- its edges), ``mean_2to1`` (edge <- cofaces).
        """
        return self._get(f"op_{name}", lambda: self._build_op(name))

    def _vertex_cells(self, k: int) -> tuple[Tensor, Tensor]:
        """``(cell_id, vertex_id)`` pairs of the vertex incidence of degree ``k >= 1`` (padding removed)."""
        C = self.cells[k].long()
        valid = C >= 0
        cid = torch.arange(C.shape[0], device=C.device)[:, None].expand_as(C)[valid]
        return cid, C[valid]

    def _build_op(self, name: str) -> tuple[Tensor, Tensor]:
        n = self.n
        if name == "gcn_adj":
            s, d = self.edges[:, 0], self.edges[:, 1]
            ar = torch.arange(n[0], device=self.device)
            rows, cols = torch.cat([s, d, ar]), torch.cat([d, s, ar])
            deg = torch.bincount(rows, minlength=n[0]).to(torch.float32)
            dis = deg.pow(-0.5)
            dis = torch.where(torch.isinf(dis), torch.zeros_like(dis), dis)
            return _pair(rows, cols, dis[rows] * dis[cols], (n[0], n[0]))
        if name == "A00":
            s, d = self.edges[:, 0], self.edges[:, 1]
            rows, cols = torch.cat([s, d]), torch.cat([d, s])
            return _pair(rows, cols, torch.ones_like(rows, dtype=torch.float32), (n[0], n[0]))
        if name in ("A11lo", "A11up", "A22"):
            if name == "A11lo":
                P = _product(csr_to_coo(self.d_abs[0]), csr_to_coo(self.dT_abs[0]))
                m = n[1]
            elif name == "A11up":
                P = _product(csr_to_coo(self.dT_abs[1]), csr_to_coo(self.d_abs[1]))
                m = n[1]
            else:
                P = _product(csr_to_coo(self.d_abs[1]), csr_to_coo(self.dT_abs[1]))
                m = n[2]
            r, c = _offdiag_pattern(P)
            return _pair(r, c, torch.ones_like(r, dtype=torch.float32), (m, m))
        if name.startswith("L"):
            base, _, norm = name.partition("_")
            if base == "L0":
                P = _product(csr_to_coo(self.dT[0]), csr_to_coo(self.d[0]))
                gid = self.graph_ids(0)
            elif base == "L1":
                P = (_product(csr_to_coo(self.d[0]), csr_to_coo(self.dT[0]))
                     + _product(csr_to_coo(self.dT[1]), csr_to_coo(self.d[1]))).coalesce()
                gid = self.graph_ids(1)
            elif base == "L2":
                P = _product(csr_to_coo(self.d[1]), csr_to_coo(self.dT[1]))
                gid = self.graph_ids(2)
            else:
                raise KeyError(name)
            idx, val = P.indices(), P.values()
            keep = val != 0
            r, c, v = idx[0][keep], idx[1][keep], val[keep]
            if norm == "n":
                m = P.shape[0]
                rs = torch.zeros(m, device=v.device, dtype=v.dtype).index_add_(0, r, v.abs())
                bound = segment_max(rs, gid, self.num_graphs).clamp_min(1e-30)
                v = v / bound[gid[r]]
            elif norm:
                raise KeyError(name)
            return _pair(r, c, v, tuple(P.shape))
        if name.startswith("mean_"):
            a, _, b = name[5:].partition("to")
            a, b = int(a), int(b)
            if a == b or max(a, b) > self.dim:
                raise KeyError(f"no incidence mean {name} on a complex of dimension {self.dim}")
            if a == 0 or b == 0:
                k = max(a, b)
                if k == 1:
                    s, d = self.edges[:, 0], self.edges[:, 1]
                    cid = torch.arange(n[1], device=self.device).repeat_interleave(2)
                    vid = torch.stack([s, d], 1).reshape(-1)
                else:
                    cid, vid = self._vertex_cells(k)
                if b == 0:          # node <- incident k-cells
                    return _row_normalized(vid, cid, (n[0], n[k]))
                return _row_normalized(cid, vid, (n[k], n[0]))   # k-cell <- its vertices
            lo, hi = min(a, b), max(a, b)
            if hi != lo + 1:
                raise KeyError(f"incidence mean {name}: only adjacent degrees or degree 0 are supported")
            r, c, _ = _triplets(self.d_abs[lo])                  # (n_hi, n_lo) pattern
            if b == hi:             # hi-cell <- its boundary lo-cells
                return _row_normalized(r, c, (n[hi], n[lo]))
            return _row_normalized(c, r, (n[lo], n[hi]))          # lo-cell <- its cofaces
        raise KeyError(f"unknown operator {name!r}")

    def apply(self, name: str, x: Tensor) -> Tensor:
        """Apply the cached operator ``name`` to ``x (n, B, C)`` (autograd through the transpose)."""
        A, AT = self.op(name)
        return spmm(A, x.contiguous(), AT)


_ADAPTERS: "weakref.WeakKeyDictionary[Any, V1ComplexAdapter]" = weakref.WeakKeyDictionary()


@torch.compiler.disable
def adapter_for(K: Any) -> V1ComplexAdapter:
    """Cached :class:`V1ComplexAdapter` of a complex (one per complex object, dropped with the complex)."""
    if isinstance(K, V1ComplexAdapter):
        return K
    A = _ADAPTERS.get(K)
    if A is None:
        A = V1ComplexAdapter(K)
        _ADAPTERS[K] = A
    return A


# ================================================================================================================
# task description
# ================================================================================================================
@dataclass
class TaskIO:
    """Everything a baseline builder needs to know about a task's inputs and targets.

    Attributes:
        name: task name.
        target_degree / target_kind: as ``TaskData`` (``cochain`` targets are orientation-odd).
        odd_target: ``target_kind == 'cochain'``.
        out_dim: target channels (``D * fields`` for node vectors).
        in_dims / even_dims / connection_dims: input layout per degree.
        spatial_dim: coordinate width of the complexes (2 = planar).
        top_dim: top degree of the complexes.
        cell_type: ``'triangle' | 'polygon' | 'quad' | 'tetrahedron' | 'mixed'``.
        variable_mesh: one complex per sample.
        native: native-cochain (True) or v1 legacy (False) inputs.
        n: cell counts of the shared complex (``None`` for variable meshes).
        grid: ``(nx, ny)`` for regular-grid shared meshes, else ``None``.
    """
    name: str
    target_degree: int
    target_kind: str
    odd_target: bool
    out_dim: int
    in_dims: dict[int, int]
    even_dims: dict[int, int]
    connection_dims: dict[int, int]
    spatial_dim: int
    top_dim: int
    cell_type: str
    variable_mesh: bool
    native: bool
    n: list[int] | None = None
    grid: tuple[int, int] | None = None
    meta: dict = field(default_factory=dict)


def task_io(td: Any) -> TaskIO:
    """:class:`TaskIO` of a :class:`rhmp.data.TaskData`."""
    K0 = td.K[0] if td.variable_mesh else td.K
    A0 = adapter_for(K0)
    return TaskIO(
        name=td.name, target_degree=int(td.target_degree), target_kind=td.target_kind,
        odd_target=td.target_kind == "cochain", out_dim=int(td.out_dim),
        in_dims={int(k): int(v) for k, v in td.in_dims.items() if int(v) > 0},
        even_dims={int(k): int(v) for k, v in td.even_dims.items()},
        connection_dims={int(k): int(v) for k, v in td.connection_dims.items()},
        spatial_dim=int(K0.pos.shape[1]), top_dim=int(K0.dim), cell_type=str(K0.meta.get("cell_type")),
        variable_mesh=bool(td.variable_mesh), native=bool(td.native),
        n=None if td.variable_mesh else [int(v) for v in K0.n],
        grid=None if td.variable_mesh else A0.grid_shape(),
    )


# ================================================================================================================
# input encoders
# ================================================================================================================
class NodeInputEncoder(nn.Module):
    """Fixed map from v2 per-degree inputs ``{k: (n_k, B, F_k)}`` to node features ``(n0, B, F_node)``.

    Column order: ``[node inputs | degree-1 odd (directional) | degree-1 even | degree-2 odd | degree-2 even |
    degree-3 ...]``.  Per degree, the last ``even_dims[k]`` columns are even, the others odd (connections included).

    * node inputs: passed through (optionally a subset ``node_columns``).
    * even inputs of degree k >= 1: mean over the k-cells incident to the node.
    * odd edge inputs ``v`` (value on the canonical orientation ``src -> dst``): v1 encoding, per node the mean
      over incident edges of ``(v, v*t_x, v*t_y[, v*t_z])`` with ``t`` the unit edge vector ``src -> dst``; the edge
      contributes the *same* ``+v`` (and ``+v t``) to both endpoints, exactly as ``encode_edges_to_nodes``
      (v1 generator ``gen_T6_wilson_loop.py``).  ``v*t`` is independent of the orientation convention (a 1-form times
      its direction is a vector); the ``avg`` column changes sign with it, as in v1.  Order ``[avg(c), avg*dx(c),
      avg*dy(c)(, avg*dz(c))]``.  ``directional=False`` keeps only the ``avg`` columns (v1's rule for EGNN);
      ``odd_avg=False`` drops them and keeps only the orientation-independent ``avg*t`` columns (the one-ring vector
      field; used by MeshGraphNet, which then stays exactly equivariant under vertex relabelling).
    * odd face inputs: mean of ``sigma_f x_f`` (``sigma_f`` = sign of the signed area) on planar complexes, plain mean
      otherwise; odd tet inputs: plain mean.
    * no inputs at all: one constant column.

    Standardisation: ``(f - mean) / std`` with buffers fitted by :meth:`fit` on training samples (identity until
    then, and skipped when only node inputs are present - those are already normalised by the task).

    Args:
        in_dims, even_dims: task input layout.
        spatial_dim: coordinate width ``D`` of the complexes.
        directional: v1 directional encoding of odd edge inputs (else ``avg`` only).
        include_edges: encode degree-1 inputs.
        node_columns: optional subset of the node input columns.
        odd_avg: keep the orientation-convention dependent ``avg`` column of odd edge inputs.
    """

    def __init__(self, in_dims: dict[int, int], even_dims: dict[int, int] | None = None, spatial_dim: int = 2, *,
                 directional: bool = True, include_edges: bool = True,
                 node_columns: list[int] | None = None, odd_avg: bool = True) -> None:
        super().__init__()
        self.in_dims = {int(k): int(v) for k, v in dict(in_dims).items() if int(v) > 0}
        self.even_dims = {int(k): int(v) for k, v in dict(even_dims or {}).items()}
        self.D = int(spatial_dim)
        self.directional = bool(directional)
        self.include_edges = bool(include_edges)
        self.node_columns = None if node_columns is None else [int(c) for c in node_columns]
        self.odd_avg = bool(odd_avg)
        if not (self.directional or self.odd_avg):
            raise ValueError("NodeInputEncoder: odd edge inputs need directional=True or odd_avg=True")
        width = 0
        self.layout: list[tuple[int, str, int]] = []      # (degree, 'node'|'odd'|'even', width)
        for k in sorted(self.in_dims):
            Fk, Ek = self.in_dims[k], self.even_dims.get(k, 0) if k > 0 else 0
            if k == 0:
                w = len(self.node_columns) if self.node_columns is not None else Fk
                self.layout.append((0, "node", w))
                width += w
                continue
            if k == 1 and not self.include_edges:
                continue
            c = Fk - Ek
            if c:
                if k == 1:
                    w = c * ((1 if self.odd_avg else 0) + (self.D if self.directional else 0))
                else:
                    w = c
                self.layout.append((k, "odd", w))
                width += w
            if Ek:
                self.layout.append((k, "even", Ek))
                width += Ek
        self.constant = width == 0
        self.out_dim = max(width, 1)
        self.passthrough = all(kind == "node" for _, kind, _ in self.layout)
        self.register_buffer("mean", torch.zeros(self.out_dim))
        self.register_buffer("std", torch.ones(self.out_dim))

    def extra_repr(self) -> str:
        return (f"layout={self.layout}, out_dim={self.out_dim}, directional={self.directional}, "
                f"odd_avg={self.odd_avg}")

    def encode(self, inputs: dict[int, Tensor], A: V1ComplexAdapter) -> Tensor:
        """Unstandardised node features ``(n0, B, out_dim)``."""
        B = next(iter(inputs.values())).shape[1] if inputs else 1
        parts: list[Tensor] = []
        for k, kind, _ in self.layout:
            x = inputs[k]
            if x.shape[0] != A.n[k]:
                raise ValueError(f"inputs[{k}] has {x.shape[0]} rows, the complex has {A.n[k]} {k}-cells")
            if kind == "node":
                parts.append(x if self.node_columns is None else x[..., self.node_columns])
                continue
            Fk, Ek = x.shape[-1], self.even_dims.get(k, 0)
            if kind == "even":
                parts.append(A.apply(f"mean_{k}to0", x[..., Fk - Ek:]))
                continue
            v = x[..., :Fk - Ek]
            if k == 1:
                cols = [v] if self.odd_avg else []
                if self.directional:
                    t = A.edge_units().to(v.dtype)                            # (n1, D)
                    cols += [v * t[:, d].view(-1, 1, 1) for d in range(t.shape[1])]
                parts.append(A.apply("mean_1to0", torch.cat(cols, dim=-1)))
            elif k == 2:
                sig = A.face_orientation()
                if sig is not None:
                    v = v * sig.to(v.dtype).view(-1, 1, 1)
                parts.append(A.apply("mean_2to0", v))
            else:
                parts.append(A.apply(f"mean_{k}to0", v))
        if not parts:
            p = self.mean
            return torch.ones(A.n0, B, 1, dtype=p.dtype, device=p.device)
        return torch.cat(parts, dim=-1)

    def forward(self, inputs: dict[int, Tensor], A: V1ComplexAdapter) -> Tensor:
        """Standardised node features ``(n0, B, out_dim)``."""
        f = self.encode(inputs, A)
        if f.shape[-1] != self.out_dim:
            raise ValueError(f"encoded width {f.shape[-1]} != expected {self.out_dim}")
        return (f - self.mean.to(f.dtype)) / self.std.to(f.dtype)

    @torch.no_grad()
    def fit(self, td: Any, max_samples: int = 256, max_meshes: int = 64) -> "NodeInputEncoder":
        """Fit the standardisation buffers on training samples of ``td`` (no-op for pure node inputs)."""
        if self.passthrough or self.constant:
            return self
        tr = td.split[0]
        feats = []
        if td.variable_mesh:
            for i in tr[:max_meshes].tolist():
                K = td.K[i]
                x = {k: v.unsqueeze(1).to(K.pos.device) for k, v in td.inputs[i].items()}
                feats.append(self.encode(x, adapter_for(K))[:, 0].float().cpu())
            f = torch.cat(feats, 0)
        else:
            idx = tr[:max_samples].to(td.target.device)
            x = {k: v.index_select(0, idx).transpose(0, 1).contiguous() for k, v in td.inputs.items()}
            f = self.encode(x, adapter_for(td.K)).float().reshape(-1, self.out_dim).cpu()
        mean, std = f.mean(0), f.std(0)
        const = std < STD_FLOOR
        self.mean.copy_(torch.where(const, torch.zeros_like(mean), mean))
        self.std.copy_(torch.where(const, torch.ones_like(std), std))
        return self


class EdgeInputEncoder(nn.Module):
    """Per-directed-edge features for MeshGraphNet, ``(2 n1, B_e, F_edge)`` (``B_e = B`` if the task has edge
    inputs, else 1 and broadcast).

    Columns: ``[odd edge inputs v on the directed edge (+v forward, -v reverse) | v * t (D per odd column, t the unit
    vector of the directed edge; identical for both directions, since both factors flip) | even edge inputs (same
    both ways) | pos[recv] - pos[send] (D) | length (1)]``, geometry divided by the mean edge length of the mesh.

    ``v * t`` is the edge-local analogue of v1's directional node encoding ``(avg, avg*dx, avg*dy)`` that the
    node-feature baselines receive: it hands the network the product of the 1-form with its direction, which a
    narrow ReLU MLP otherwise has to learn (T6, 10 epochs: MGN on raw edge values reaches val R2 0.005, MGN on the
    node encoding 0.83).  Exact orientation equivariance is kept.  ``directional=False`` gives the raw values only.

    Args:
        in_dims, even_dims: task input layout.
        spatial_dim: ``D``.
        normalized_geometry: divide geometry by the per-mesh mean edge length (default) or use raw coordinates.
        directional: append ``v * t`` for the odd edge inputs.
    """

    def __init__(self, in_dims: dict[int, int], even_dims: dict[int, int] | None = None, spatial_dim: int = 2, *,
                 normalized_geometry: bool = True, directional: bool = True) -> None:
        super().__init__()
        F1 = int(dict(in_dims).get(1, 0))
        E1 = int(dict(even_dims or {}).get(1, 0))
        self.odd, self.even = F1 - E1, E1
        self.D = int(spatial_dim)
        self.normalized = bool(normalized_geometry)
        self.directional = bool(directional)
        self.out_dim = self.odd * (1 + (self.D if self.directional else 0)) + self.even + self.D + 1

    def extra_repr(self) -> str:
        return (f"odd={self.odd}, even={self.even}, D={self.D}, directional={self.directional}, "
                f"out_dim={self.out_dim}")

    def forward(self, inputs: dict[int, Tensor], A: V1ComplexAdapter, dtype: torch.dtype = torch.float32) -> Tensor:
        rel, ln = A.directed_geometry(self.normalized)
        geo = torch.cat([rel, ln[:, None]], dim=-1).to(dtype)                  # (2 n1, D + 1)
        if not (self.odd or self.even):
            return geo.unsqueeze(1)
        x = inputs[1]
        if x.shape[0] != A.n1:
            raise ValueError(f"inputs[1] has {x.shape[0]} rows, the complex has {A.n1} edges")
        B = x.shape[1]
        parts = []
        if self.odd:
            o = x[..., :self.odd]
            parts.append(torch.cat([o, -o], 0))
            if self.directional:
                t = A.edge_units().to(o.dtype)                                     # (n1, D), src -> dst
                ot = torch.cat([o * t[:, d].view(-1, 1, 1) for d in range(t.shape[1])], dim=-1)
                parts.append(torch.cat([ot, ot], 0))                               # (-v)(-t) = v t
        if self.even:
            e = x[..., self.odd:]
            parts.append(torch.cat([e, e], 0))
        parts.append(geo.unsqueeze(1).expand(-1, B, -1))
        return torch.cat([p.to(dtype) for p in parts], dim=-1)


# ================================================================================================================
# output heads
# ================================================================================================================
def mlp(sizes: list[int], act: type[nn.Module] = nn.SiLU, layernorm: bool = False) -> nn.Sequential:
    """``Linear -> act -> ... -> Linear`` (optional LayerNorm on the output)."""
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    if layernorm:
        layers.append(nn.LayerNorm(sizes[-1]))
    return nn.Sequential(*layers)


class NodeHead(nn.Module):
    """Node target: ``net(x_0)``, ``(n0, B, H) -> (n0, B, out)`` (``net`` is typically the v1 node head)."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, feats: dict[int, Tensor], A: V1ComplexAdapter) -> Tensor:
        return self.net(feats[0])


class PairEdgeHead(nn.Module):
    """Edge target from node features: ``g([x_a, x_b, l_e])`` with ``(a, b) = (src, dst)`` and ``l_e`` the edge
    length over the mesh's mean edge length; antisymmetrised ``g(a,b) - g(b,a)`` for orientation-odd targets
    (edge cochains: exactly odd under edge-orientation reversal), symmetrised ``(g(a,b) + g(b,a)) / 2`` otherwise.

    Args:
        g: module ``(.., 2H + 1) -> (.., out)`` (the v1 ``edge_mlp`` for ``BaselineBase`` models).
        odd: target parity.
    """

    def __init__(self, g: nn.Module, odd: bool) -> None:
        super().__init__()
        self.g = g
        self.odd = bool(odd)

    def forward(self, feats: dict[int, Tensor], A: V1ComplexAdapter) -> Tensor:
        x = feats[0]
        xa, xb = x.index_select(0, A.edges[:, 0]), x.index_select(0, A.edges[:, 1])
        ln = A.edge_length_normalized().to(x.dtype).view(-1, 1, 1).expand(-1, x.shape[1], 1)
        g1 = self.g(torch.cat([xa, xb, ln], -1))
        g2 = self.g(torch.cat([xb, xa, ln], -1))
        return g1 - g2 if self.odd else 0.5 * (g1 + g2)


class FaceHead(nn.Module):
    """Face target from node features: ``net(mean_{v in f} x_v)``, times ``sigma_f`` (sign of the signed area) for
    orientation-odd targets on planar complexes (the prediction is made in the physical, counter-clockwise frame and
    expressed in the face's own orientation; exactly odd under face-orientation flips).  On 3-D surfaces a node-based
    model has no access to face orientations (documented limitation)."""

    def __init__(self, net: nn.Module, odd: bool) -> None:
        super().__init__()
        self.net = net
        self.odd = bool(odd)

    def forward(self, feats: dict[int, Tensor], A: V1ComplexAdapter) -> Tensor:
        y = self.net(A.apply("mean_0to2", feats[0]))
        sig = A.face_orientation() if self.odd else None
        return y if sig is None else y * sig.to(y.dtype).view(-1, 1, 1)


class CellHead(nn.Module):
    """Target on degree ``k >= 3`` cells (tets) from node features: ``net(mean_{v in c} x_v)``."""

    def __init__(self, net: nn.Module, degree: int) -> None:
        super().__init__()
        self.net = net
        self.degree = int(degree)

    def forward(self, feats: dict[int, Tensor], A: V1ComplexAdapter) -> Tensor:
        return self.net(A.apply(f"mean_0to{self.degree}", feats[0]))


class ComplexHead(nn.Module):
    """Head for baselines with hidden features on several degrees (MPSN, SCCNN, CW Net, Clifford-SMPN).

    Input for target degree ``t`` (every maintained degree is used; other degrees are brought to the target cells by
    incidence means, fixed and parameter-free):

    * ``t = 0``: ``[x_0, mean_{e ∋ v} x_1, mean_{f ∋ v} x_2]``
    * ``t = 1``: ``[x_1, x_a + x_b, x_b - x_a, l_e, mean_{f ∋ e} x_2]`` (``x_b - x_a`` gives access to the edge
      orientation; the internal edge features of these baselines have no definite parity)
    * ``t = 2``: ``[x_2, mean_{v in f} x_0, mean_{e in f} x_1]``, times ``sigma_f`` for odd targets on planar complexes
      (the models' face features are built from unsigned face aggregates or have no parity, as for node models)
    * ``t = 3``: ``[mean_{v in c} x_0]``

    followed by ``MLP(in -> H -> out)``.

    Args:
        degrees: degrees with hidden features (``(0, 1, 2)`` or ``(0, 1)``).
        hidden: feature width ``H`` of every degree.
        out_dim: target channels.
        target_degree, odd: target cells and parity.
    """

    def __init__(self, degrees: tuple[int, ...], hidden: int, out_dim: int, target_degree: int, odd: bool) -> None:
        super().__init__()
        self.degrees = tuple(int(k) for k in degrees)
        self.t = int(target_degree)
        self.odd = bool(odd)
        if self.t == 0:
            w = hidden * len(self.degrees)
        elif self.t == 1:
            w = hidden * (1 + 2) + 1 + (hidden if 2 in self.degrees else 0)
            if 1 not in self.degrees:
                raise ValueError("ComplexHead: edge targets need edge features")
        elif self.t == 2:
            w = hidden * len(self.degrees)
        else:
            w = hidden
        self.net = mlp([w, hidden, out_dim])

    def forward(self, feats: dict[int, Tensor], A: V1ComplexAdapter) -> Tensor:
        t = self.t
        if t == 0:
            z = [feats[0]] + [A.apply(f"mean_{k}to0", feats[k]) for k in self.degrees if k != 0]
        elif t == 1:
            x0 = feats[0]
            xa, xb = x0.index_select(0, A.edges[:, 0]), x0.index_select(0, A.edges[:, 1])
            ln = A.edge_length_normalized().to(x0.dtype).view(-1, 1, 1).expand(-1, x0.shape[1], 1)
            z = [feats[1], xa + xb, xb - xa, ln]
            if 2 in self.degrees:
                z.append(A.apply("mean_2to1", feats[2]))
        elif t == 2:
            z = ([feats[2]] if 2 in self.degrees else []) + [A.apply("mean_0to2", feats[0])]
            if 1 in self.degrees:
                z.append(A.apply("mean_1to2", feats[1]))
        else:
            z = [A.apply(f"mean_0to{t}", feats[0])]
        y = self.net(torch.cat(z, -1))
        if t == 2 and self.odd:
            sig = A.face_orientation()
            if sig is not None:
                y = y * sig.to(y.dtype).view(-1, 1, 1)
        return y


def node_model_head(io: TaskIO, hidden: int, node_net: nn.Module | None = None,
                    edge_net: nn.Module | None = None) -> nn.Module:
    """Target head for a node-feature baseline of width ``hidden``.

    Args:
        io: task description.
        hidden: node feature width.
        node_net: the baseline's own node head (used for node targets; default ``Linear(hidden, out)``).
        edge_net: the baseline's own edge MLP ``(2H + 1) -> out`` (default ``MLP(2H+1 -> H -> out)``).
    Returns:
        :class:`NodeHead` | :class:`PairEdgeHead` | :class:`FaceHead` | :class:`CellHead`.
    """
    t, out = io.target_degree, io.out_dim
    if t == 0:
        return NodeHead(node_net if node_net is not None else nn.Linear(hidden, out))
    if t == 1:
        return PairEdgeHead(edge_net if edge_net is not None else mlp([2 * hidden + 1, hidden, out]), io.odd_target)
    if t == 2:
        return FaceHead(mlp([hidden, hidden, out]), io.odd_target)
    return CellHead(mlp([hidden, hidden, out]), t)
