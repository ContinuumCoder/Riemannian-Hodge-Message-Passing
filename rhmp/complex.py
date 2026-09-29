"""Cochain complexes with exact oriented incidence, reference Hodge stars and cached geometry.

A :class:`CochainComplex` stores, for top degree ``K`` (2 = surfaces, 3 = tetrahedral volumes):

* ``cells[k]``: ``cells[0] = arange(n0)[:, None]``, ``cells[1]`` edges ``(n1, 2)`` with ``src < dst``,
  ``cells[2]`` faces ``(n2, m)`` in the given orientation (``-1`` padded for polygons; canonical sorted
  vertex order for faces of tetrahedra), ``cells[3]`` tets ``(n3, 4)`` in the given order.
* ``d[k]``: coboundary ``(n_{k+1}, n_k)`` as CSR float32; ``dT[k]`` its separately built transpose;
  ``d_abs[k]``, ``dT_abs[k]`` the CSR of ``|d_k|``.  ``d_{k+1} d_k = 0`` holds exactly (integer
  arithmetic); :meth:`CochainComplex.check_d2` verifies it with a sparse product.
* ``star[k] > 0`` reference diagonal Hodge star, ``geo[k]`` E(n)-invariant descriptors
  (see :mod:`rhmp.geometry`), ``boundary[k]`` flags.
* ``batch[k]`` sample id of every cell for block-diagonal batches (``None`` for a single complex).

All construction is vectorised torch (no Python loops over cells) and runs on the target device.
Block-diagonal batches are built with ``CochainComplex.batch([K1, K2, ...])`` (class-level call; the
instance attribute ``K.batch`` holds the per-degree sample ids).
"""
from __future__ import annotations

import itertools
import time
import warnings
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from torch import Tensor

from . import geometry as _geometry
from .ops import beta_unit, csr_from_coo, csr_with_values, sparse_csr, spmm

__all__ = ["CochainComplex", "validate_triangles", "batch_complexes"]

DEGENERATE_REL_MEASURE = 1e-10  # simplex dropped if (k! * measure) / (longest edge)^k is below this
_TET_FACES = torch.tensor([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])  # face i omits local vertex i
_TET_FACE_ALT = torch.tensor([1.0, -1.0, 1.0, -1.0])  # (-1)^i of the simplicial boundary
_TET_EDGES = torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
_TRI_EDGE_SLOTS = torch.tensor([[0, 1], [1, 2], [0, 2]])  # canonical face (a<b<c): edges ab, bc, ac
_TRI_EDGE_SIGN = torch.tensor([1.0, 1.0, -1.0])  # boundary [a,b,c] = [b,c] - [a,c] + [a,b]


# ==============================================================================================
# input conversion
# ==============================================================================================
def _resolve_device(device, *arrays) -> torch.device:
    if device is not None:
        return torch.device(device)
    for a in arrays:
        if isinstance(a, Tensor):
            return a.device
    return torch.device("cpu")


def _to_pos64(pos, device: torch.device) -> Tensor:
    t = pos if isinstance(pos, Tensor) else torch.as_tensor(np.asarray(pos))
    if t.dtype == torch.bool or t.is_complex():
        raise ValueError(f"pos must be real-valued, got dtype {t.dtype}")
    t = t.detach().to(device=device, dtype=torch.float64)
    if t.dim() != 2 or t.shape[1] not in (2, 3):
        raise ValueError(f"pos must have shape (n0, 2) or (n0, 3), got {tuple(t.shape)}")
    if not bool(torch.isfinite(t).all()):
        raise ValueError("pos contains non-finite values")
    return t


def _to_index(a, device: torch.device, name: str) -> Tensor:
    t = a if isinstance(a, Tensor) else torch.as_tensor(np.asarray(a))
    if t.dtype == torch.bool or t.is_complex():
        raise ValueError(f"{name} must be an integer array, got dtype {t.dtype}")
    if t.is_floating_point():
        ti = t.round()
        if not bool((ti == t).all()):
            raise ValueError(f"{name} must contain integer vertex ids")
        t = ti
    return t.detach().to(device=device, dtype=torch.long)


def _check_cells(C: Tensor, n0: int, width: int | None, name: str) -> None:
    if C.dim() != 2 or (width is not None and C.shape[1] != width):
        w = "m" if width is None else width
        raise ValueError(f"{name} must have shape (n, {w}), got {tuple(C.shape)}")
    if C.shape[0] == 0:
        raise ValueError(f"{name} is empty")
    lo = -1 if width is None else 0
    if bool((C < lo).any()) or bool((C >= n0).any()):
        raise ValueError(f"{name} contains vertex ids outside [0, {n0})")


def _pad_polygons(faces, device: torch.device) -> Tensor:
    """list[list[int]] | (n2, M) array (-1 padded) -> (n2, M) int64 tensor, -1 padded."""
    if isinstance(faces, (Tensor, np.ndarray)):
        return _to_index(faces, device, "faces")
    faces = list(faces)
    lengths = np.fromiter(map(len, faces), dtype=np.int64, count=len(faces))
    if len(faces) == 0:
        raise ValueError("faces is empty")
    flat = np.fromiter(itertools.chain.from_iterable(faces), dtype=np.int64, count=int(lengths.sum()))
    M = int(lengths.max())
    out = np.full((len(faces), M), -1, dtype=np.int64)
    rows = np.repeat(np.arange(len(faces)), lengths)
    starts = np.cumsum(lengths) - lengths
    cols = np.arange(flat.size) - np.repeat(starts, lengths)
    out[rows, cols] = flat
    return torch.from_numpy(out).to(device)


# ==============================================================================================
# topology helpers (vectorised)
# ==============================================================================================
def _unique_rows(rows: Tensor, base: int) -> tuple[Tensor, Tensor]:
    """Lexicographically sorted unique rows of an ``(m, r)`` int64 tensor with entries in ``[0, base)``.

    Returns ``(unique_rows (u, r), inverse (m,))``.  Keys are compressed column by column, so no
    overflow for ``m, base < 3e9``.
    """
    m, r = rows.shape
    key = rows[:, 0]
    for j in range(1, r):
        key = key * base + rows[:, j]
        if j < r - 1:
            key = torch.unique(key, sorted=True, return_inverse=True)[1]
    ukey, inv = torch.unique(key, sorted=True, return_inverse=True)
    first = torch.full((ukey.numel(),), m, dtype=torch.long, device=rows.device)
    first = first.scatter_reduce(0, inv, torch.arange(m, device=rows.device), reduce="amin")
    return rows[first], inv


def _first_occurrence(inv: Tensor, n_groups: int) -> Tensor:
    """Bool mask selecting the first element of every group of ``inv``."""
    m = inv.numel()
    ar = torch.arange(m, device=inv.device)
    first = torch.full((n_groups,), m, dtype=torch.long, device=inv.device).scatter_reduce(0, inv, ar, reduce="amin")
    return ar == first[inv]


def _corner_table(F: Tensor, face_len: Tensor) -> dict[str, Tensor]:
    """Corner (= half-edge) table of ``-1``-padded faces, in row-major order of valid slots."""
    n2, M = F.shape
    dev = F.device
    valid = F >= 0
    face = torch.arange(n2, device=dev)[:, None].expand(n2, M)[valid]
    slot = torch.arange(M, device=dev)[None, :].expand(n2, M)[valid]
    v = F[valid]
    start = torch.cumsum(face_len, 0) - face_len
    L = face_len[face]
    nidx = start[face] + (slot + 1) % L
    pidx = start[face] + (slot - 1 + L) % L
    return dict(face=face, slot=slot, v=v, next=v[nidx], prev=v[pidx], prev_idx=pidx, next_idx=nidx)


def _edges_from_halfedges(src: Tensor, dst: Tensor, n0: int) -> tuple[Tensor, Tensor, Tensor]:
    """Canonical edges ``(n1, 2)`` (src < dst, sorted), half-edge -> edge map and orientation signs."""
    lo = torch.minimum(src, dst)
    hi = torch.maximum(src, dst)
    ukey, inv = torch.unique(lo * n0 + hi, sorted=True, return_inverse=True)
    edges = torch.stack([ukey // n0, ukey % n0], dim=1)
    sign = torch.where(src < dst, 1.0, -1.0).to(torch.float32)
    return edges, inv, sign


def _operator_set(row: Tensor, col: Tensor, val: Tensor, shape: tuple[int, int]) -> tuple[Tensor, ...]:
    """CSR of ``A``, ``A^T``, ``|A|``, ``|A|^T`` from COO triplets (no duplicates)."""
    A = csr_from_coo(row, col, val, shape)
    AT = csr_from_coo(col, row, val, (shape[1], shape[0]))
    return A, AT, csr_with_values(A, A.values().abs()), csr_with_values(AT, AT.values().abs())


def _d0_set(edges: Tensor, n0: int) -> tuple[Tensor, ...]:
    n1 = edges.shape[0]
    dev = edges.device
    row = torch.arange(n1, device=dev).repeat_interleave(2)
    val = torch.tensor([-1.0, 1.0], device=dev).repeat(n1)
    return _operator_set(row, edges.reshape(-1), val, (n1, n0))


def _block_diag_csr(mats: list[Tensor]) -> Tensor:
    """Block-diagonal concatenation of CSR matrices (index offsets only; no data movement besides cat)."""
    crows, cols, vals = [], [], []
    nnz = ncol = nrow = 0
    for A in mats:
        crows.append(A.crow_indices()[:-1] + nnz)
        cols.append(A.col_indices() + ncol)
        vals.append(A.values())
        nnz += A.values().numel()
        ncol += A.shape[1]
        nrow += A.shape[0]
    crows.append(torch.tensor([nnz], dtype=torch.long, device=mats[0].device))
    return sparse_csr(torch.cat(crows), torch.cat(cols), torch.cat(vals), (nrow, ncol))


def _cat_offset_padded(complexes: list, key: str, sign_key: str, deg: int) -> tuple[Tensor, Tensor]:
    """Concatenate ``meta[key]`` (ids of degree-``deg`` cells, -1 padded) with offsets, and their signs."""
    width = max(K.meta[key].shape[1] for K in complexes)
    ids, sg, off = [], [], 0
    for K in complexes:
        c, s = K.meta[key], K.meta[sign_key]
        c = torch.where(c >= 0, c + off, c)
        pad = width - c.shape[1]
        if pad:
            c = torch.cat([c, c.new_full((c.shape[0], pad), -1)], dim=1)
            s = torch.cat([s, s.new_zeros((s.shape[0], pad))], dim=1)
        ids.append(c)
        sg.append(s)
        off += K.n[deg]
    return torch.cat(ids), torch.cat(sg)


def _move(obj, device: torch.device, dtype: torch.dtype | None, non_blocking: bool = False):
    """Recursively move tensors (dense and CSR) in lists/tuples/dicts; floats optionally cast (after the copy)."""
    if isinstance(obj, Tensor):
        if obj.layout == torch.sparse_csr:
            val = obj.values().to(device, non_blocking=non_blocking)
            if dtype is not None and val.is_floating_point():
                val = val.to(dtype)
            return sparse_csr(obj.crow_indices().to(device, non_blocking=non_blocking),
                              obj.col_indices().to(device, non_blocking=non_blocking), val, tuple(obj.shape))
        out = obj.to(device, non_blocking=non_blocking)
        return out.to(dtype) if (dtype is not None and out.is_floating_point()) else out
    if isinstance(obj, list):
        return [_move(v, device, dtype, non_blocking) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move(v, device, dtype, non_blocking) for v in obj)
    if isinstance(obj, dict):
        return {k: _move(v, device, dtype, non_blocking) for k, v in obj.items()}
    return obj


def _pin(obj):
    """Recursively replace CPU tensors (dense, and the components of CSR tensors) by page-locked copies.

    Idempotent (already pinned tensors are returned as they are); non-CPU tensors and other objects are unchanged.
    """
    if isinstance(obj, Tensor):
        if obj.device.type != "cpu":
            return obj
        if obj.layout == torch.sparse_csr:
            parts = (obj.crow_indices(), obj.col_indices(), obj.values())
            if all(t.is_pinned() for t in parts):
                return obj
            return sparse_csr(*(t if t.is_pinned() else t.pin_memory() for t in parts), tuple(obj.shape))
        if obj.layout != torch.strided:
            return obj
        return obj if obj.is_pinned() else obj.pin_memory()
    if isinstance(obj, list):
        return [_pin(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_pin(v) for v in obj)
    if isinstance(obj, dict):
        return {k: _pin(v) for k, v in obj.items()}
    return obj


_TENSOR_FIELDS = ("cells", "d", "dT", "d_abs", "dT_abs", "pos", "geo", "star", "boundary", "batch", "meta", "whitney")


# ==============================================================================================
# validation
# ==============================================================================================
def _validate_simplices(P: Tensor, S: Tensor) -> tuple[Tensor, dict]:
    """Drop simplices with repeated vertices, zero measure, or duplicate vertex sets (first kept)."""
    m, r = S.shape
    n0 = P.shape[0]
    Ss = S.sort(dim=1).values
    rep = (Ss[:, 1:] == Ss[:, :-1]).any(dim=1)
    X = _geometry._pad3(P)[S]  # (m, r, 3)
    E = X[:, 1:] - X[:, :1]  # (m, r-1, 3)
    if r == 3:
        meas = torch.linalg.cross(E[:, 0], E[:, 1], dim=-1).norm(dim=-1)
    else:
        meas = torch.linalg.det(E).abs()
    i, j = torch.triu_indices(r, r, offset=1, device=S.device)
    lmax = (X[:, i] - X[:, j]).norm(dim=-1).amax(dim=1)
    flat = (~rep) & (meas <= DEGENERATE_REL_MEASURE * lmax.pow(r - 1))
    keep_idx = (~(rep | flat)).nonzero().squeeze(1)
    if keep_idx.numel() == 0:
        raise ValueError("validation removed every cell (all degenerate)")
    uniq, inv = _unique_rows(Ss[keep_idx], n0)
    kept = keep_idx[_first_occurrence(inv, uniq.shape[0])]
    stats = dict(
        n_input=m,
        n_degenerate_index=int(rep.sum()),
        n_degenerate_measure=int(flat.sum()),
        n_duplicate=int(keep_idx.numel() - kept.numel()),
        n_kept=int(kept.numel()),
        kept_index=kept,
    )
    return S[kept], stats


def validate_triangles(pos, faces) -> tuple[Tensor, dict]:
    """Drop degenerate and duplicate triangles.

    A face is dropped if it repeats a vertex, if it is collinear up to round-off
    (``2 * area < DEGENERATE_REL_MEASURE * longest_edge^2``) or if its vertex set already occurred
    (first occurrence kept, whatever its orientation).  Order and orientation of kept faces are preserved.

    Args:
        pos: ``(n0, D)`` positions (D = 2 or 3; numpy or torch).
        faces: ``(n2, 3)`` integer vertex ids (numpy or torch, int32/int64).
    Returns:
        ``(faces_clean (n2', 3) int64, stats)`` with ``stats`` keys ``n_input, n_degenerate_index,
        n_degenerate_measure, n_duplicate, n_kept`` (ints) and ``kept_index`` (``(n2',)`` int64 indices
        of kept input faces).
    Raises:
        ValueError: on malformed shapes or out-of-range vertex ids.
    """
    dev = _resolve_device(None, pos, faces)
    P = _to_pos64(pos, dev)
    F = _to_index(faces, dev, "faces")
    _check_cells(F, P.shape[0], 3, "faces")
    return _validate_simplices(P, F)


def _validate_polygons(P: Tensor, F: Tensor, validate: bool) -> tuple[Tensor, dict]:
    """Check padding, drop polygons with < 3 vertices, repeated vertices, zero area or duplicates."""
    n2, M = F.shape
    n0 = P.shape[0]
    valid = F >= 0
    if bool((valid[:, 1:] & ~valid[:, :-1]).any()):
        raise ValueError("polygon padding (-1) must be trailing")
    L = valid.sum(1)
    Ss = torch.where(valid, F, n0).sort(dim=1).values  # padding sorts last
    rep = ((Ss[:, 1:] == Ss[:, :-1]) & (Ss[:, 1:] < n0)).any(dim=1)
    short = L < 3
    bad = rep | short
    if not validate:
        if bool(bad.any()):
            raise ValueError(
                f"{int(short.sum())} polygons with < 3 vertices and {int(rep.sum())} with repeated vertices; "
                "pass validate=True to drop them"
            )
        return F, dict(n_input=n2, kept_index=torch.arange(n2, device=F.device))
    Lc = L.clamp_min(3)
    area, _, _ = _geometry.face_measures(_geometry._pad3(P), F, Lc)
    Fz = F.clamp_min(0)
    nxt = torch.gather(Fz, 1, (torch.arange(M, device=F.device)[None, :] + 1) % Lc[:, None])
    P3 = _geometry._pad3(P)
    elen = (P3[nxt] - P3[Fz]).norm(dim=-1) * valid
    lmax = elen.amax(dim=1)
    flat = (~bad) & (2.0 * area <= DEGENERATE_REL_MEASURE * lmax.pow(2))
    keep_idx = (~(bad | flat)).nonzero().squeeze(1)
    if keep_idx.numel() == 0:
        raise ValueError("validation removed every polygon (all degenerate)")
    uniq, inv = _unique_rows(Ss[keep_idx], n0 + 1)
    kept = keep_idx[_first_occurrence(inv, uniq.shape[0])]
    Fk = F[kept]
    Mk = int((Fk >= 0).sum(1).max())
    stats = dict(
        n_input=n2,
        n_degenerate_index=int(bad.sum()),
        n_degenerate_measure=int(flat.sum()),
        n_duplicate=int(keep_idx.numel() - kept.numel()),
        n_kept=int(kept.numel()),
        kept_index=kept,
    )
    return Fk[:, :Mk].contiguous(), stats


def _require_distinct(S: Tensor, name: str) -> None:
    Ss = S.sort(dim=1).values
    if bool((Ss[:, 1:] == Ss[:, :-1]).any()):
        raise ValueError(f"{name} contain cells with repeated vertices; pass validate=True to drop them")


def _warn_dropped(where: str, stats: dict) -> None:
    nd = stats.get("n_degenerate_index", 0) + stats.get("n_degenerate_measure", 0)
    nu = stats.get("n_duplicate", 0)
    if nd or nu:
        warnings.warn(f"{where}: dropped {nd} degenerate and {nu} duplicate cells "
                      f"(kept {stats['n_kept']} of {stats['n_input']}); see meta['validation']",
                      RuntimeWarning, stacklevel=3)


# ==============================================================================================
# the complex
# ==============================================================================================
@dataclass(eq=False, repr=False)
class CochainComplex:
    """Oriented cochain complex with CSR coboundaries, reference stars and E(n)-invariant geometry.

    Fields (``K`` = top degree ``dim``):
        dim: top degree.
        n: ``[n_0, ..., n_K]`` cell counts.
        cells: ``cells[k]`` vertex ids of the k-cells (see module docstring).
        d, dT, d_abs, dT_abs: CSR ``d_k (n_{k+1}, n_k)``, ``d_k^T``, ``|d_k|``, ``|d_k|^T``, k = 0..K-1.
        pos: ``(n0, D)`` float32 vertex positions (only used through edge vectors in vector readouts).
        geo: ``geo[k] (n_k, G_k)`` float32 E(n)-invariant descriptors (names: ``meta['geo_features'][k]``).
        star: ``star[k] (n_k,)`` float32 reference Hodge star, > 0.
        boundary: ``boundary[k] (n_k,)`` bool.
        batch: ``batch[k] (n_k,)`` int64 sample id per cell for block-diagonal batches, else ``None``.
        num_graphs: number of samples (meshes) in the complex.
        whitney: ``{k: dict}`` precomputed Whitney/Galerkin metric blocks of degree k (simplicial complexes only:
            k=1 for triangles, k=1 and 2 for tets; see :func:`rhmp.geometry.whitney_triangles` /
            :func:`rhmp.geometry.whitney_tets` and :mod:`rhmp.dec`), plus CSR ``gather ((m_k n_top), n_k)`` and
            ``scatter (n_k, (m_k n_top))`` maps between k-cells and per-top-cell slots (slot-major rows).
        meta: extra info: ``cell_type, star_type, geo_features, sizes, ptr, validation, face_index,
            geometry_stats, build_time_s``; ``face_edges (n2, m)`` edge ids of every face in cyclic order
            (edge j joins vertex j and j+1 of ``cells[2]``; -1 padded) with ``face_edge_signs`` (+-1, 0 pad);
            tets: ``tet_faces (n3, 4)`` (face i omits local vertex i) with ``tet_face_signs``; polygons:
            ``face_lengths (n2,)``; grids: ``grid_shape``, ``spacing``, ``diagonal``; ``beta_unit``: per-graph
            ``(num_graphs,)`` Gershgorin bounds of ``S^{-1/2} A^T A S^{-1/2}`` with keys ``'up{k}_dec'``
            (A = d_k, S = star_k), ``'up{k}_none'`` (S = 1), ``'down{k}_dec'`` (A = d_{k-1}^T, S = 1/star_k),
            ``'down{k}_none'``.
    """

    dim: int
    n: list[int]
    cells: list[Tensor]
    d: list[Tensor]
    dT: list[Tensor]
    d_abs: list[Tensor]
    dT_abs: list[Tensor]
    pos: Tensor
    geo: list[Tensor]
    star: list[Tensor]
    boundary: list[Tensor]
    batch: list[Tensor] | None
    num_graphs: int
    meta: dict
    whitney: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ basic accessors
    @property
    def device(self) -> torch.device:
        """Device of all tensors of the complex."""
        return self.pos.device

    @property
    def geo_dims(self) -> dict[int, int]:
        """``{k: G_k}`` for all degrees (constructor argument of the model)."""
        return {k: self.geo_dim(k) for k in range(self.dim + 1)}

    def geo_dim(self, k: int) -> int:
        """Number of descriptor columns ``G_k`` of degree ``k``."""
        return int(self.geo[k].shape[1])

    def graph_ids(self, k: int) -> Tensor | None:
        """Graph (sample) id of every k-cell of a block-diagonal batch, ``(n_k,)`` int64; ``None`` for a single complex.

        Use this instead of the instance attribute ``batch`` (whose name is shared with the class-level builder
        ``CochainComplex.batch([...])``).
        """
        if not 0 <= k <= self.dim:
            raise ValueError(f"graph_ids: degree k={k} out of range for a complex of dimension {self.dim}")
        return None if self.batch is None else self.batch[k]

    @classmethod
    def from_list(cls, complexes: list["CochainComplex"]) -> "CochainComplex":
        """Block-diagonal batch of complexes (same as ``CochainComplex.batch([...])`` / :func:`batch_complexes`).

        Args:
            complexes: non-empty list with equal ``dim``, position dimension, descriptor widths and device.
        Returns:
            :class:`CochainComplex` with per-cell graph ids (``graph_ids(k)``), ``num_graphs`` and ``meta['ptr']``.
        """
        return batch_complexes(list(complexes))

    def __repr__(self) -> str:
        return (f"CochainComplex(dim={self.dim}, n={self.n}, cell_type={self.meta.get('cell_type')!r}, "
                f"star={self.meta.get('star_type')!r}, num_graphs={self.num_graphs}, device={self.device})")

    def to(self, device=None, dtype: torch.dtype | None = None, non_blocking: bool | None = None) -> "CochainComplex":
        """Copy of the complex on ``device`` (CSR operators included).

        Host-to-GPU moves (CPU -> CUDA) use page-locked memory and asynchronous copies: the tensors of this (CPU)
        complex are pinned once, in place (pinned CPU tensors behave like pageable ones, so later moves of the same
        complex reuse them), and every copy is issued with ``non_blocking=True`` on the current CUDA stream.  No
        device synchronisation is needed: kernels on that stream run after the copies, and PyTorch's caching host
        allocator keeps the pinned buffers alive until the copies have completed.  (Pageable blocking copies would
        stall the host behind the queued GPU work, about 5 ms per tensor on a busy GPU.)  Moves in all other directions
        use plain copies.

        Args:
            device: target device (default: current).
            dtype: optional floating dtype for ``d*``, ``pos``, ``geo``, ``star`` (e.g. ``torch.float64``
                for numerical tests); index and bool tensors are unchanged.
            non_blocking: ``None`` (default) = pinned asynchronous copies for CPU -> CUDA, plain copies otherwise;
                ``False`` forces the plain blocking copies.
        Returns:
            a new :class:`CochainComplex`.
        """
        dev = self.device if device is None else torch.device(device)
        nb = (dev.type == "cuda" and self.device.type == "cpu") if non_blocking is None else bool(non_blocking)
        nb = nb and dev.type == "cuda" and self.device.type == "cpu"
        if nb:
            for name in _TENSOR_FIELDS:                                   # pin once (idempotent)
                setattr(self, name, _pin(getattr(self, name)))
        mv = lambda o: _move(o, dev, dtype, nb)  # noqa: E731
        mi = lambda o: _move(o, dev, None, nb)  # noqa: E731
        return replace(
            self, cells=mi(self.cells), d=mv(self.d), dT=mv(self.dT), d_abs=mv(self.d_abs),
            dT_abs=mv(self.dT_abs), pos=mv(self.pos), geo=mv(self.geo), star=mv(self.star),
            boundary=mi(self.boundary),
            batch=None if self.batch is None else mi(self.batch),
            meta=mi(self.meta), whitney=mv(self.whitney),
        )

    # ------------------------------------------------------------------ operators
    def _check(self, k: int, x: Tensor, rows: int, what: str) -> None:
        if not 0 <= k < self.dim:
            raise ValueError(f"{what}: degree k={k} out of range for a complex of dimension {self.dim}")
        if x.dim() < 1 or x.shape[0] != rows:
            raise ValueError(f"{what}({k}): expected x with {rows} rows, got shape {tuple(x.shape)}")

    def apply_d(self, k: int, x: Tensor) -> Tensor:
        """Coboundary ``d_k x``: ``(n_k, B, C) -> (n_{k+1}, B, C)`` (any trailing shape; contiguous)."""
        self._check(k, x, self.n[k], "apply_d")
        return spmm(self.d[k], x, self.dT[k])

    def apply_dT(self, k: int, x: Tensor) -> Tensor:
        """Transposed coboundary ``d_k^T x``: ``(n_{k+1}, B, C) -> (n_k, B, C)``."""
        self._check(k, x, self.n[k + 1], "apply_dT")
        return spmm(self.dT[k], x, self.d[k])

    def apply_d_abs(self, k: int, x: Tensor) -> Tensor:
        """``|d_k| x``: ``(n_k, ...) -> (n_{k+1}, ...)`` (Jacobi scalings, Gershgorin bounds)."""
        self._check(k, x, self.n[k], "apply_d_abs")
        return spmm(self.d_abs[k], x, self.dT_abs[k])

    def apply_dT_abs(self, k: int, x: Tensor) -> Tensor:
        """``|d_k|^T x``: ``(n_{k+1}, ...) -> (n_k, ...)``."""
        self._check(k, x, self.n[k + 1], "apply_dT_abs")
        return spmm(self.dT_abs[k], x, self.d_abs[k])

    def edge_vectors(self) -> Tensor:
        """``(n1, D)`` edge vectors ``pos[dst] - pos[src]`` (E(n)-equivariant)."""
        e = self.cells[1]
        return self.pos[e[:, 1]] - self.pos[e[:, 0]]

    def check_d2(self) -> float:
        """``max_k max |d_{k+1} d_k|`` computed with sparse products (0.0 for a valid complex)."""
        worst = 0.0
        for k in range(self.dim - 1):
            prod = torch.sparse.mm(self.d[k + 1], self.d[k])
            vals = prod.values()
            if vals.numel():
                worst = max(worst, float(vals.abs().max()))
        return worst

    # ------------------------------------------------------------------ builders
    @staticmethod
    def from_triangles(pos, faces, *, star: str = "cotan", validate: bool = True, device=None) -> "CochainComplex":
        """Triangle mesh (planar 2-D or surface in 3-D; boundaries and non-manifold edges allowed).

        Args:
            pos: ``(n0, 2|3)`` positions (numpy or torch, any float dtype).
            faces: ``(n2, 3)`` vertex ids (numpy or torch, int32/int64); orientation is kept.
            star: ``'cotan'`` (default) | ``'barycentric'`` | ``'unit'``.
            validate: drop degenerate/duplicate faces with a warning (``meta['validation']``,
                ``meta['face_index']`` maps kept faces to input rows).  With ``False`` faces with repeated
                vertices raise.
            device: target device (default: device of ``pos`` if a tensor, else CPU).
        Returns:
            :class:`CochainComplex` with ``dim = 2``.
        """
        t0 = time.perf_counter()
        dev = _resolve_device(device, pos, faces)
        P = _to_pos64(pos, dev)
        F = _to_index(faces, dev, "faces")
        _check_cells(F, P.shape[0], 3, "faces")
        if validate:
            F, vstats = _validate_simplices(P, F)
            _warn_dropped("from_triangles", vstats)
        else:
            _require_distinct(F, "faces")
            vstats = dict(n_input=F.shape[0], kept_index=torch.arange(F.shape[0], device=dev))
        face_len = torch.full((F.shape[0],), 3, dtype=torch.long, device=dev)
        return _build_surface(P, F, face_len, star=star, cell_type="triangle", vstats=vstats, t0=t0,
                              warn=validate)

    @staticmethod
    def from_polygons(pos, faces, *, star: str = "barycentric", validate: bool = True,
                      device=None) -> "CochainComplex":
        """Polygonal (CW) 2-complex: quads, mixed polygons, triangles.

        Args:
            pos: ``(n0, 2|3)`` positions.
            faces: ``list[list[int]]`` (cyclic vertex order = orientation) or ``(n2, M)`` integer array padded
                with ``-1`` (padding trailing).
            star: ``'barycentric'`` (default) | ``'cotan'`` (cotan terms on triangles, barycentric half dual
                edges on polygons = circumcentric on rectangles) | ``'unit'``.
            validate: drop polygons with < 3 vertices, repeated vertices, zero area, duplicates (warning).
            device: target device.
        Returns:
            :class:`CochainComplex` with ``dim = 2`` and ``cells[2]`` of shape ``(n2, M)`` (``-1`` padded);
            ``d[1]`` rows list the boundary edges in cyclic order with orientation signs.
        """
        t0 = time.perf_counter()
        dev = _resolve_device(device, pos, faces if isinstance(faces, Tensor) else None)
        P = _to_pos64(pos, dev)
        F = _pad_polygons(faces, dev)
        _check_cells(F, P.shape[0], None, "faces")
        F, vstats = _validate_polygons(P, F, validate)
        if validate:
            _warn_dropped("from_polygons", vstats)
        face_len = (F >= 0).sum(1)
        return _build_surface(P, F, face_len, star=star, cell_type="polygon", vstats=vstats, t0=t0,
                              warn=validate)

    @staticmethod
    def from_tetrahedra(pos, tets, *, star: str = "barycentric", validate: bool = True,
                        device=None) -> "CochainComplex":
        """Tetrahedral 3-complex (degrees 0..3).

        Faces are the unique boundary triangles in canonical (sorted) vertex order; ``d_2`` carries the
        orientation sign ``(-1)^i * parity`` of face i (omitting local vertex i) in each tet.

        Args:
            pos: ``(n0, 3)`` positions.
            tets: ``(n3, 4)`` vertex ids (input order is the orientation).
            star: ``'barycentric'`` (default) | ``'unit'``.
            validate: drop tets with repeated vertices, zero volume, duplicates (warning).
            device: target device.
        Returns:
            :class:`CochainComplex` with ``dim = 3``.
        """
        t0 = time.perf_counter()
        dev = _resolve_device(device, pos, tets)
        P = _to_pos64(pos, dev)
        if P.shape[1] != 3:
            raise ValueError(f"from_tetrahedra needs 3-D positions, got {tuple(P.shape)}")
        T = _to_index(tets, dev, "tets")
        _check_cells(T, P.shape[0], 4, "tets")
        if validate:
            T, vstats = _validate_simplices(P, T)
            _warn_dropped("from_tetrahedra", vstats)
        else:
            _require_distinct(T, "tets")
            vstats = dict(n_input=T.shape[0], kept_index=torch.arange(T.shape[0], device=dev))
        return _build_volume(P, T, star=star, vstats=vstats, t0=t0, warn=validate)

    @staticmethod
    def from_grid(shape: tuple[int, int], spacing=1.0, *, cell: str = "quad", star: str | None = None,
                  diagonal: str = "v1", device=None) -> "CochainComplex":
        """Regular 2-D grid with ``nx * ny`` vertices, vertex id ``i * ny + j`` at ``(i*sx, j*sy)``.

        The vertex numbering is the v1 convention (``meshgrid(xs, ys, indexing='ij')``, as in the v1 PDEBench
        loader ``build_regular_grid`` used for T1), so per-vertex data stored as
        ``field[ix, iy].reshape(-1)`` lines up.

        Args:
            shape: ``(nx, ny)`` vertices per axis (>= 2 each).
            spacing: ``h`` or ``(sx, sy)``.  ``from_grid((32, 32), 1/31, cell='tri')`` reproduces T1.
            cell: ``'quad'`` (CCW quads ``[v00, v10, v11, v01]``) or ``'tri'``.
            star: default ``'cotan'`` for triangles, ``'barycentric'`` for quads (identical on rectangles).
            diagonal: triangle split for ``cell='tri'``: ``'v1'`` (default) = scipy/qhull Delaunay of the
                ``linspace(0, 1, n)`` grid exactly as v1 built T1 (degenerate cocircular squares ->
                qhull-chosen diagonals; verified identical to the T1 faces with scipy 1.17), faces made
                CCW; ``'main'`` (v00-v11), ``'anti'`` (v10-v01), ``'alternate'`` (checkerboard) are
                deterministic and need no scipy.
            device: target device.
        Returns:
            :class:`CochainComplex` (``cell_type`` ``'quad'`` or ``'triangle'``) with ``meta['grid_shape']``.
        """
        t0 = time.perf_counter()
        nx, ny = int(shape[0]), int(shape[1])
        if nx < 2 or ny < 2:
            raise ValueError(f"grid shape must be >= (2, 2), got {shape}")
        sx, sy = (float(spacing), float(spacing)) if np.isscalar(spacing) else (float(spacing[0]), float(spacing[1]))
        if sx <= 0 or sy <= 0:
            raise ValueError("grid spacing must be positive")
        dev = torch.device(device) if device is not None else torch.device("cpu")
        ii, jj = torch.meshgrid(torch.arange(nx, dtype=torch.float64), torch.arange(ny, dtype=torch.float64),
                                indexing="ij")
        P = torch.stack([ii.reshape(-1) * sx, jj.reshape(-1) * sy], dim=1).to(dev)
        i, j = torch.meshgrid(torch.arange(nx - 1), torch.arange(ny - 1), indexing="ij")
        i, j = i.reshape(-1), j.reshape(-1)
        v00, v10, v11, v01 = i * ny + j, (i + 1) * ny + j, (i + 1) * ny + j + 1, i * ny + j + 1
        vstats_extra = dict(grid_shape=(nx, ny), spacing=(sx, sy))
        if cell == "quad":
            F = torch.stack([v00, v10, v11, v01], dim=1).to(dev)
            face_len = torch.full((F.shape[0],), 4, dtype=torch.long, device=dev)
            vstats = dict(n_input=F.shape[0], kept_index=torch.arange(F.shape[0], device=dev))
            K = _build_surface(P, F, face_len, star=star or "barycentric", cell_type="quad", vstats=vstats,
                               t0=t0, warn=False)
        elif cell == "tri":
            if diagonal == "v1":
                F = torch.from_numpy(_v1_grid_triangles(nx, ny)).to(dev)
            else:
                main = torch.stack([torch.stack([v00, v10, v11], 1), torch.stack([v00, v11, v01], 1)], 1)
                anti = torch.stack([torch.stack([v00, v10, v01], 1), torch.stack([v10, v11, v01], 1)], 1)
                if diagonal == "main":
                    F = main
                elif diagonal == "anti":
                    F = anti
                elif diagonal == "alternate":
                    F = torch.where(((i + j) % 2 == 0)[:, None, None], main, anti)
                else:
                    raise ValueError(f"unknown diagonal {diagonal!r}; use 'v1', 'main', 'anti' or 'alternate'")
                F = F.reshape(-1, 3).to(dev)
            face_len = torch.full((F.shape[0],), 3, dtype=torch.long, device=dev)
            vstats = dict(n_input=F.shape[0], kept_index=torch.arange(F.shape[0], device=dev))
            K = _build_surface(P, F, face_len, star=star or "cotan", cell_type="triangle", vstats=vstats,
                               t0=t0, warn=False)
            vstats_extra["diagonal"] = diagonal
        else:
            raise ValueError(f"unknown cell {cell!r}; use 'quad' or 'tri'")
        K.meta.update(vstats_extra)
        return K


def _v1_grid_triangles(nx: int, ny: int) -> np.ndarray:
    """Faces of v1's ``build_regular_grid(nx, ny)`` (scipy Delaunay of the unit linspace grid), made CCW."""
    try:
        from scipy.spatial import Delaunay
    except ImportError as e:  # pragma: no cover
        raise ImportError("from_grid(cell='tri', diagonal='v1') needs scipy; use diagonal='main'") from e
    xx, yy = np.meshgrid(np.linspace(0.0, 1.0, nx), np.linspace(0.0, 1.0, ny), indexing="ij")
    pts = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    S = Delaunay(pts).simplices.astype(np.int64)
    a = pts[S[:, 1]] - pts[S[:, 0]]
    b = pts[S[:, 2]] - pts[S[:, 0]]
    cw = (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) < 0
    S[cw] = S[cw][:, [0, 2, 1]]
    return S


# ==============================================================================================
# builders (shared)
# ==============================================================================================
def _finish_meta(meta: dict, t0: float, device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    meta["build_time_s"] = time.perf_counter() - t0


def _single_meta(n: list[int], device: torch.device) -> dict:
    return dict(
        sizes=[list(n)],
        ptr=[torch.tensor([0, nk], dtype=torch.long, device=device) for nk in n],
    )


def _slot_maps(cells: Tensor, n_k: int) -> tuple[Tensor, Tensor]:
    """CSR maps between k-cells and per-top-cell slots in slot-major order (row ``j * n_top + f`` = slot j of top
    cell f): ``gather ((m n_top), n_k)``, ``scatter (n_k, (m n_top))``."""
    R = cells.numel()
    dev = cells.device
    col = cells.t().reshape(-1)
    ones = torch.ones(R, dtype=torch.float32, device=dev)
    gather = sparse_csr(torch.arange(R + 1, device=dev), col, ones, (R, n_k))
    scatter = csr_from_coo(col, torch.arange(R, device=dev), ones, (n_k, R))
    return gather, scatter


def _finish_whitney(w: dict, k: int, top: int, n_k: int) -> dict:
    w["gather"], w["scatter"] = _slot_maps(w["cells"], n_k)
    w.update(k=k, top=top)
    return w


def _beta_unit_meta(K: "CochainComplex") -> dict:
    """Per-graph Gershgorin bounds of ``S^{-1/2} A^T A S^{-1/2}`` for up/down blocks with S = star / 1."""
    out = {}
    b = K.batch
    for k in range(K.dim + 1):
        idx = None if b is None else b[k]
        if k < K.dim:
            out[f"up{k}_dec"] = beta_unit(K.d_abs[k], K.dT_abs[k], K.star[k].rsqrt(), idx, K.num_graphs).reshape(-1)
            out[f"up{k}_none"] = beta_unit(K.d_abs[k], K.dT_abs[k], None, idx, K.num_graphs).reshape(-1)
        if k > 0:
            out[f"down{k}_dec"] = beta_unit(K.dT_abs[k - 1], K.d_abs[k - 1], K.star[k].sqrt(), idx,
                                            K.num_graphs).reshape(-1)
            out[f"down{k}_none"] = beta_unit(K.dT_abs[k - 1], K.d_abs[k - 1], None, idx, K.num_graphs).reshape(-1)
    return out


def _build_surface(P: Tensor, F: Tensor, face_len: Tensor, *, star: str, cell_type: str, vstats: dict,
                   t0: float, warn: bool) -> CochainComplex:
    n0, n2 = P.shape[0], F.shape[0]
    dev = P.device
    corner = _corner_table(F, face_len)
    edges, ce, csign = _edges_from_halfedges(corner["v"], corner["next"], n0)
    corner["edge"], corner["sign"] = ce, csign
    n1 = edges.shape[0]
    D0 = _d0_set(edges, n0)
    D1 = _operator_set(corner["face"], ce, csign, (n2, n1))
    g = _geometry.surface_geometry(P, F, face_len, edges, corner, star=star)
    if warn and g["stats"]["n_isolated_vertices"]:
        warnings.warn(f"{g['stats']['n_isolated_vertices']} vertices belong to no face (kept; star0 set to the "
                      "median dual area)", RuntimeWarning, stacklevel=3)
    n = [n0, n1, n2]
    face_edges = torch.full(F.shape, -1, dtype=torch.long, device=dev)
    face_edges[corner["face"], corner["slot"]] = ce
    face_edge_signs = torch.zeros(F.shape, dtype=torch.float32, device=dev)
    face_edge_signs[corner["face"], corner["slot"]] = csign
    meta = dict(
        cell_type=cell_type, star_type=star, kappa=_geometry.KAPPA,
        geo_features={k: _geometry.GEO_FEATURE_NAMES[k] for k in range(3)},
        face_index=vstats.pop("kept_index"), validation=vstats, geometry_stats=g["stats"],
        face_edges=face_edges, face_edge_signs=face_edge_signs,
        **_single_meta(n, dev),
    )
    if cell_type in ("polygon", "quad"):
        meta["face_lengths"] = face_len
    whitney = {}
    if F.shape[1] == 3:  # simplicial 2-complex: Whitney 1-form Galerkin blocks
        whitney[1] = _finish_whitney(_geometry.whitney_triangles(P, F, face_edges, face_edge_signs), 1, 2, n1)
    K = CochainComplex(
        dim=2, n=n, cells=[torch.arange(n0, device=dev)[:, None], edges, F],
        d=[D0[0], D1[0]], dT=[D0[1], D1[1]], d_abs=[D0[2], D1[2]], dT_abs=[D0[3], D1[3]],
        pos=P.to(torch.float32), geo=g["geo"], star=g["star"], boundary=g["boundary"],
        batch=None, num_graphs=1, meta=meta, whitney=whitney,
    )
    meta["beta_unit"] = _beta_unit_meta(K)
    _finish_meta(meta, t0, dev)
    return K


def _build_volume(P: Tensor, T: Tensor, *, star: str, vstats: dict, t0: float, warn: bool) -> CochainComplex:
    n0, n3 = P.shape[0], T.shape[0]
    dev = P.device
    Fall = T[:, _TET_FACES.to(dev)]  # (n3, 4, 3) tet order
    Fs, perm = Fall.sort(dim=-1)
    inv_cnt = ((perm[..., 0] > perm[..., 1]).long() + (perm[..., 0] > perm[..., 2]).long()
               + (perm[..., 1] > perm[..., 2]).long())
    fsign = (1.0 - 2.0 * (inv_cnt % 2).to(torch.float32)) * _TET_FACE_ALT.to(dev)
    faces, tf = _unique_rows(Fs.reshape(-1, 3), n0)
    tet_face = tf.view(n3, 4)
    n2 = faces.shape[0]
    edges, fe = _unique_rows(faces[:, _TRI_EDGE_SLOTS.to(dev)].reshape(-1, 2), n0)
    face_edge = fe.view(n2, 3)
    n1 = edges.shape[0]
    ekey = edges[:, 0] * n0 + edges[:, 1]
    TE = T[:, _TET_EDGES.to(dev)]
    tkey = torch.minimum(TE[..., 0], TE[..., 1]) * n0 + torch.maximum(TE[..., 0], TE[..., 1])
    tet_edge = torch.searchsorted(ekey, tkey.reshape(-1)).view(n3, 6)
    tri_sign = _TRI_EDGE_SIGN.to(dev)
    D0 = _d0_set(edges, n0)
    D1 = _operator_set(torch.arange(n2, device=dev).repeat_interleave(3), face_edge.reshape(-1),
                       tri_sign.repeat(n2), (n2, n1))
    D2 = _operator_set(torch.arange(n3, device=dev).repeat_interleave(4), tet_face.reshape(-1),
                       fsign.reshape(-1), (n3, n2))
    face_len = torch.full((n2,), 3, dtype=torch.long, device=dev)
    fcorner = _corner_table(faces, face_len)
    fcorner["edge"], fcorner["sign"] = face_edge.reshape(-1), tri_sign.repeat(n2)
    g = _geometry.volume_geometry(P, T, faces, edges, tet_face, tet_edge, face_edge, fcorner, star=star)
    if warn and g["stats"]["n_isolated_vertices"]:
        warnings.warn(f"{g['stats']['n_isolated_vertices']} vertices belong to no tet (kept; star0 set to the "
                      "median dual volume)", RuntimeWarning, stacklevel=3)
    n = [n0, n1, n2, n3]
    meta = dict(
        cell_type="tetrahedron", star_type=star,
        geo_features={k: _geometry.GEO_FEATURE_NAMES[k] for k in range(4)},
        face_index=vstats.pop("kept_index"), validation=vstats, geometry_stats=g["stats"],
        face_edges=face_edge, face_edge_signs=tri_sign.repeat(n2).view(n2, 3),
        tet_faces=tet_face, tet_face_signs=fsign,
        **_single_meta(n, dev),
    )
    w1, w2 = _geometry.whitney_tets(P, T, tet_edge, tet_face)
    whitney = {1: _finish_whitney(w1, 1, 3, n1), 2: _finish_whitney(w2, 2, 3, n2)}
    K = CochainComplex(
        dim=3, n=n, cells=[torch.arange(n0, device=dev)[:, None], edges, faces, T],
        d=[D0[0], D1[0], D2[0]], dT=[D0[1], D1[1], D2[1]], d_abs=[D0[2], D1[2], D2[2]],
        dT_abs=[D0[3], D1[3], D2[3]], pos=P.to(torch.float32), geo=g["geo"], star=g["star"],
        boundary=g["boundary"], batch=None, num_graphs=1, meta=meta, whitney=whitney,
    )
    meta["beta_unit"] = _beta_unit_meta(K)
    _finish_meta(meta, t0, dev)
    return K


# ==============================================================================================
# block-diagonal batching
# ==============================================================================================
def batch_complexes(complexes: list[CochainComplex]) -> CochainComplex:
    """Block-diagonal union of complexes (no re-triangulation; geometry is reused as is).

    Every part keeps its own per-mesh normalised descriptors and stars, so a batch is exactly the
    disjoint union of its parts.  Parts may themselves be batches.

    Args:
        complexes: non-empty list with equal ``dim``, position dimension, descriptor widths and device.
    Returns:
        :class:`CochainComplex` with ``batch[k] (n_k,)`` sample ids, ``num_graphs = sum`` of parts,
        ``meta['sizes']`` (one ``[n_0..n_K]`` per graph) and ``meta['ptr'][k]`` (``(num_graphs+1,)`` offsets).
    """
    if len(complexes) == 0:
        raise ValueError("batch: empty list")
    K0 = complexes[0]
    dim, dev, D = K0.dim, K0.device, K0.pos.shape[1]
    for K in complexes[1:]:
        if K.dim != dim or K.device != dev or K.pos.shape[1] != D:
            raise ValueError("batch: all complexes need the same dim, device and position dimension")
        if any(K.geo_dim(k) != K0.geo_dim(k) for k in range(dim + 1)):
            raise ValueError("batch: descriptor widths differ")
    n = [sum(K.n[k] for K in complexes) for k in range(dim + 1)]
    off0 = np.cumsum([0] + [K.n[0] for K in complexes])[:-1]
    cells = [torch.arange(n[0], device=dev)[:, None]]
    for k in range(1, dim + 1):
        width = max(K.cells[k].shape[1] for K in complexes)
        parts = []
        for K, o in zip(complexes, off0):
            c = K.cells[k]
            c = torch.where(c >= 0, c + int(o), c)
            if c.shape[1] < width:
                c = torch.cat([c, c.new_full((c.shape[0], width - c.shape[1]), -1)], dim=1)
            parts.append(c)
        cells.append(torch.cat(parts))
    goff = np.cumsum([0] + [K.num_graphs for K in complexes])[:-1]
    batch = []
    for k in range(dim + 1):
        parts = []
        for K, g in zip(complexes, goff):
            b = K.batch[k] if K.batch is not None else torch.zeros(K.n[k], dtype=torch.long, device=dev)
            parts.append(b + int(g))
        batch.append(torch.cat(parts))
    sizes = [s for K in complexes for s in K.meta["sizes"]]
    num_graphs = len(sizes)
    ptr = [torch.tensor(np.cumsum([0] + [s[k] for s in sizes]), dtype=torch.long, device=dev) for k in range(dim + 1)]

    def _common(key):
        vals = {K.meta.get(key) for K in complexes}
        return vals.pop() if len(vals) == 1 else "mixed"

    meta = dict(
        cell_type=_common("cell_type"), star_type=_common("star_type"),
        geo_features=K0.meta.get("geo_features"), sizes=sizes, ptr=ptr,
        validation=[K.meta.get("validation") for K in complexes],
    )
    if all("face_lengths" in K.meta for K in complexes):
        meta["face_lengths"] = torch.cat([K.meta["face_lengths"] for K in complexes])
    for key, sgn, deg in (("face_edges", "face_edge_signs", 1), ("tet_faces", "tet_face_signs", 2)):
        if all(key in K.meta for K in complexes):
            meta[key], meta[sgn] = _cat_offset_padded(complexes, key, sgn, deg)
    if all("beta_unit" in K.meta for K in complexes):
        keys = set.intersection(*(set(K.meta["beta_unit"]) for K in complexes))
        meta["beta_unit"] = {key: torch.cat([K.meta["beta_unit"][key] for K in complexes]) for key in sorted(keys)}
    whitney = _merge_whitney(complexes)
    cat = lambda lists: [torch.cat([L[k] for L in lists]) for k in range(len(lists[0]))]  # noqa: E731
    return CochainComplex(
        dim=dim, n=n, cells=cells,
        d=[_block_diag_csr([K.d[k] for K in complexes]) for k in range(dim)],
        dT=[_block_diag_csr([K.dT[k] for K in complexes]) for k in range(dim)],
        d_abs=[_block_diag_csr([K.d_abs[k] for K in complexes]) for k in range(dim)],
        dT_abs=[_block_diag_csr([K.dT_abs[k] for K in complexes]) for k in range(dim)],
        pos=torch.cat([K.pos for K in complexes]),
        geo=cat([K.geo for K in complexes]), star=cat([K.star for K in complexes]),
        boundary=cat([K.boundary for K in complexes]),
        batch=batch, num_graphs=num_graphs, meta=meta, whitney=whitney,
    )


def _merge_whitney(complexes: list[CochainComplex]) -> dict:
    """Concatenate the Whitney blocks of the degrees present in every part (ids offset, CSR maps block-diagonal)."""
    degrees = set.intersection(*(set(K.whitney) for K in complexes))
    out = {}
    for k in sorted(degrees):
        parts = [K.whitney[k] for K in complexes]
        w = {key: torch.cat([p[key] for p in parts]) for key in ("signs", "t", "G0", "Gk", "rowsum_G0", "rowsum_Gk")}
        for key, deg in (("cells", k), ("dir_edges", 1)):
            offs = np.cumsum([0] + [K.n[deg] for K in complexes])[:-1]
            w[key] = torch.cat([p[key] + int(o) for p, o in zip(parts, offs)])
        w["gather"], w["scatter"] = _slot_maps(w["cells"], sum(K.n[k] for K in complexes))  # slot-major rows
        w.update(k=k, top=parts[0]["top"])
        out[k] = w
    return out


# ``batch`` is both a dataclass field (per-degree sample ids, instance attribute) and the class-level
# builder required by the API: ``CochainComplex.batch([K1, K2])``.  The staticmethod is attached after
# the dataclass is created, so instances still carry their own ``batch`` attribute.
CochainComplex.batch = staticmethod(batch_complexes)  # type: ignore[assignment]
