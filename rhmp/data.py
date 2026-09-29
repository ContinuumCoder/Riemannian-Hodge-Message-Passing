"""GPU-resident task containers, v1-compatible splits / normalisation, cochain alignment and minibatching.

``TaskData`` (DESIGN §4) holds everything a run needs, already normalised and on the target device:

* shared-mesh tasks (T1..T7, T6f, ...): ``K`` is one :class:`~rhmp.complex.CochainComplex`, ``inputs[k]`` is
  ``(N, n_k, F_k)`` and ``target`` is ``(N, n_t, out_dim)``.  A minibatch is converted to the model layout
  ``(n_k, B, F_k)`` by :func:`shared_minibatch`.
* variable-mesh tasks (T8, HP, TET): ``K`` is a list of complexes (one per sample), ``inputs`` a list of dicts of
  ``(n_k_i, F_k)`` tensors and ``target`` a list of ``(n_t_i, out_dim)`` tensors.  :func:`mesh_minibatch` builds a
  block-diagonal batch (``CochainComplex.batch``; ``B = 1``, cells concatenated, ``K.batch[k]`` sample ids).

Normalisation follows v1 (the v1 training script ``formal_benchmark.py``): per-feature mean/std over (samples, cells) of the
training part, ``std.clamp(1e-6)``; torch's unbiased std for the shared-mesh pickles, numpy's biased std for T8.
Orientation-odd quantities (cochain inputs/targets on degree >= 1) and vector targets are only *scaled*
(mean 0, RMS scale) so that the normalisation commutes with orientation flips / rotations; see ``Stats``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import Tensor

from rhmp.ops import to_nbc  # (n, B, C) layout helper (single implementation in the core module)

STD_FLOOR = 1e-6  # v1: std.clamp(1e-6)
PIN_MAX_BYTES = 256 * 2 ** 20   # pack_to pins host buffers up to this size (larger ones: one pageable copy)

__all__ = [
    "Stats", "OutputMap", "TaskData", "sequential_split", "holdout_split", "feature_stats", "scale_stats",
    "edge_alignment", "cell_alignment", "shared_minibatch", "mesh_minibatch", "iterate_indices",
    "concat_cells", "to_nbc", "same_device", "tensor_nbytes", "strip_whitney", "pack_to", "add_abs_scale",
]


def same_device(a, b) -> bool:
    """Device equality treating ``cuda`` and ``cuda:<current>`` as the same device."""
    a, b = torch.device(a), torch.device(b)
    if a.type != b.type:
        return False
    if a.type != "cuda":
        return True
    cur = torch.cuda.current_device() if torch.cuda.is_available() else 0
    return (a.index if a.index is not None else cur) == (b.index if b.index is not None else cur)


def tensor_nbytes(o) -> int:
    """Bytes held by the tensors inside ``o`` (tensors incl. sparse CSR, dicts, lists, dataclass fields of a
    complex)."""
    if isinstance(o, Tensor):
        if o.layout == torch.sparse_csr:
            return sum(t.numel() * t.element_size() for t in (o.crow_indices(), o.col_indices(), o.values()))
        return o.numel() * o.element_size()
    if isinstance(o, dict):
        return sum(tensor_nbytes(v) for v in o.values())
    if isinstance(o, (list, tuple)):
        return sum(tensor_nbytes(v) for v in o)
    if hasattr(o, "__dataclass_fields__"):
        return sum(tensor_nbytes(getattr(o, f)) for f in o.__dataclass_fields__)
    return 0


def pack_to(obj, device):
    """Move every tensor inside ``obj`` to ``device`` with one copy per dtype instead of one per tensor.

    ``obj`` may be a tensor, a (nested) list/tuple/dict, or a dataclass such as a ``CochainComplex`` (sparse CSR
    tensors included); the structure is rebuilt with views into a few contiguous device buffers.  Moving e.g. 5000
    variable-mesh complexes (~200K small tensors) this way takes seconds even on a GPU that is time-sliced with other
    processes, where per-tensor synchronous copies took tens of minutes.

    Returns:
        an object of the same structure on ``device`` (non-tensor leaves are shared).
    """
    import dataclasses

    from rhmp.ops import sparse_csr
    dev = torch.device(device)
    flat: dict[torch.dtype, list[Tensor]] = {}

    def collect(o):
        if isinstance(o, Tensor):
            if o.layout == torch.sparse_csr:
                return ("csr", collect(o.crow_indices()), collect(o.col_indices()), collect(o.values()),
                        tuple(o.shape))
            if o.device == dev:
                return ("same", o)
            lst = flat.setdefault(o.dtype, [])
            lst.append(o.detach().reshape(-1))
            return ("t", o.dtype, len(lst) - 1, tuple(o.shape))
        if isinstance(o, dict):
            return ("dict", type(o), {k: collect(v) for k, v in o.items()})
        if isinstance(o, (list, tuple)):
            return ("seq", type(o), [collect(v) for v in o])
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return ("dc", o, {f.name: collect(getattr(o, f.name)) for f in dataclasses.fields(o)})
        return ("leaf", o)

    skel = collect(obj)
    bufs, offs = {}, {}
    for dt, lst in flat.items():
        sizes = [t.numel() for t in lst]
        offs[dt] = [0]
        for n in sizes:
            offs[dt].append(offs[dt][-1] + n)
        host = torch.cat(lst) if lst else torch.empty(0, dtype=dt)
        pinned = False
        if (dev.type == "cuda" and torch.cuda.is_available() and host.device.type == "cpu"
                and host.numel() * host.element_size() <= PIN_MAX_BYTES):
            try:                                  # small (per-batch) buffers: caching pinned host allocator
                host, pinned = host.pin_memory(), True
            except Exception as e:  # noqa: BLE001 - page-locking can fail (limits, other processes)
                import warnings
                warnings.warn(f"pack_to: pin_memory failed ({type(e).__name__}); using a pageable copy",
                              RuntimeWarning, stacklevel=2)
                pinned = False
        bufs[dt] = host.to(dev, non_blocking=pinned)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)

    def build(sk):
        kind = sk[0]
        if kind == "t":
            _, dt, i, shape = sk
            return bufs[dt][offs[dt][i]:offs[dt][i + 1]].view(shape)
        if kind == "same":
            return sk[1]
        if kind == "csr":
            return sparse_csr(build(sk[1]), build(sk[2]), build(sk[3]), sk[4])
        if kind == "dict":
            return sk[1]((k, build(v)) for k, v in sk[2].items())
        if kind == "seq":
            vals = [build(v) for v in sk[2]]
            return vals if sk[1] is list else sk[1](vals)
        if kind == "dc":
            return dataclasses.replace(sk[1], **{k: build(v) for k, v in sk[2].items()})
        return sk[1]

    return build(skel)


def add_abs_scale(task: "TaskData") -> "TaskData":
    """Append a constant EVEN vertex column ``log(median edge length)`` (absolute length units) to ``inputs[0]``.

    The v2 model is scale-free by design (descriptors ``log(x / median x)``, stars normalised per graph), so a task
    whose physics has a fixed length scale (e.g. meshes of different sizes with a fixed diffusion length) must get the
    absolute scale as an input.  The column is constant per mesh, standardised with the training split (a single
    shared mesh gives an all-zero column), listed last and declared even (it also feeds the vertex metric head).
    Extra test sets get the same column with the same statistics.  In place; returns ``task``.
    """
    def scale(K) -> Tensor:
        return torch.log(K.edge_vectors().double().norm(dim=1).median())

    def append(d: dict, col: Tensor) -> None:
        d[0] = torch.cat([d[0], col.to(d[0])], -1).contiguous() if 0 in d else col.contiguous()

    if task.variable_mesh:
        vals = [scale(K) for K in task.K]
        tr = torch.stack([vals[i] for i in task.split[0].tolist()]) if len(task.split[0]) else torch.stack(vals)
        mean, std = tr.mean(), (tr.std() if tr.numel() > 1 else torch.ones((), dtype=tr.dtype)).clamp(min=STD_FLOOR)
        for K, v, d in zip(task.K, vals, task.inputs):
            dev = d[0].device if 0 in d else K.pos.device
            append(d, ((v - mean) / std).float().to(dev).expand(K.n[0], 1))
        for src in task.extra_tests.values():
            for K, d in zip(src["K"], src["inputs"]):
                dev = d[0].device if 0 in d else K.pos.device
                append(d, ((scale(K) - mean) / std).float().to(dev).expand(K.n[0], 1))
    else:
        v = scale(task.K)
        mean, std = v, torch.ones((), dtype=v.dtype)
        N = task.num_samples
        dev = task.target.device
        col = torch.zeros(N, task.K.n[0], 1, device=dev)
        task.inputs[0] = torch.cat([task.inputs[0], col], -1).contiguous() if 0 in task.inputs else col
    old = task.x_stats.get(0)
    m1, s1 = mean.float().reshape(1), std.float().reshape(1)
    task.x_stats[0] = Stats(torch.cat([old.mean, m1.to(old.mean)]) if old is not None else m1,
                            torch.cat([old.std, s1.to(old.std)]) if old is not None else s1,
                            (old.kind if old is not None else "v1") + "+abs_scale")
    task.in_dims = dict(task.in_dims)
    task.in_dims[0] = task.in_dims.get(0, 0) + 1
    task.even_dims = dict(task.even_dims)
    task.even_dims[0] = task.even_dims.get(0, 0) + 1
    task.meta["abs_scale"] = {"mean_log_median_edge": float(mean), "std": float(std)}
    return task


def strip_whitney(obj) -> int:
    """Drop the Whitney/Galerkin blocks (``K.whitney``, only used by tensor metrics) from a complex, a list of
    complexes, or every complex of a :class:`TaskData` and its extra test sets.  Returns the number of bytes freed."""
    freed = 0
    if isinstance(obj, TaskData):
        freed += strip_whitney(obj.K)
        for src in obj.extra_tests.values():
            freed += strip_whitney(src.get("K", []))
        return freed
    if isinstance(obj, (list, tuple)):
        return sum(strip_whitney(K) for K in obj)
    if getattr(obj, "whitney", None):
        freed = tensor_nbytes(obj.whitney)
        obj.whitney = {}
    return freed




# ----------------------------------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class Stats:
    """Per-feature affine normalisation ``x_n = (x - mean) / std``.

    Attributes:
        mean: ``(F,)``.
        std: ``(F,)`` (>= 1e-6).
        kind: free-form description (e.g. ``'v1'``, ``'scale'``, ``'isotropic'``).
    """
    mean: Tensor
    std: Tensor
    kind: str = "v1"

    def normalize(self, x: Tensor) -> Tensor:
        """``(..., F) -> (..., F)``."""
        return (x - self.mean.to(x)) / self.std.to(x)

    def denormalize(self, x: Tensor) -> Tensor:
        """``(..., F) -> (..., F)``."""
        return x * self.std.to(x) + self.mean.to(x)

    def to(self, device) -> "Stats":
        """Copy on ``device``."""
        return Stats(self.mean.to(device), self.std.to(device), self.kind)

    def as_tuple(self) -> tuple[Tensor, Tensor]:
        """``(mean, std)``."""
        return self.mean, self.std

    def to_dict(self) -> dict:
        """JSON-friendly ``{'mean', 'std', 'kind'}``."""
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "kind": self.kind}


def feature_stats(x_train: Tensor | Sequence[Tensor], *, unbiased: bool = True) -> Stats:
    """v1 per-feature statistics over all leading dims (samples x cells).

    Args:
        x_train: ``(N, n, F)`` tensor or list of ``(n_i, F)`` tensors (pooled).
        unbiased: torch default (``True``, v1 shared-mesh pickles) or numpy default (``False``, v1 T8).
    Returns:
        ``Stats`` with ``mean, std`` of shape ``(F,)``, std clamped at 1e-6.
    """
    x = x_train if isinstance(x_train, Tensor) else torch.cat(list(x_train), 0)
    x = x.reshape(-1, x.shape[-1])
    mean = x.mean(0)
    std = x.std(0, unbiased=unbiased) if x.shape[0] > 1 else torch.ones_like(mean)
    return Stats(mean, std.clamp(min=STD_FLOOR), "v1" if unbiased else "v1-numpy")


def scale_stats(x_train: Tensor | Sequence[Tensor], *, isotropic: bool = False) -> Stats:
    """Scale-only statistics for orientation-odd cochains and vector fields: ``mean = 0``, ``std = RMS``.

    Args:
        x_train: ``(N, n, F)`` or list of ``(n_i, F)``.
        isotropic: one common RMS for all ``F`` columns (vector components) instead of one per column.
    Returns:
        ``Stats`` (``(F,)`` tensors).
    """
    x = x_train if isinstance(x_train, Tensor) else torch.cat(list(x_train), 0)
    x = x.reshape(-1, x.shape[-1]).double()
    if isotropic:
        rms = torch.sqrt((x ** 2).mean()).expand(x.shape[-1]).clone()
    else:
        rms = torch.sqrt((x ** 2).mean(0))
    rms = rms.float().clamp(min=STD_FLOOR)
    return Stats(torch.zeros_like(rms), rms, "isotropic" if isotropic else "scale")


# ----------------------------------------------------------------------------------------------------------------
# splits
# ----------------------------------------------------------------------------------------------------------------
def sequential_split(N: int, device=None) -> tuple[Tensor, Tensor, Tensor]:
    """v1 split: first ``int(0.7 N)`` train, next ``int(0.15 N)`` val, rest test (no shuffling)."""
    nt, nv = int(0.7 * N), int(0.15 * N)
    ar = torch.arange(N, device=device)
    return ar[:nt], ar[nt:nt + nv], ar[nt + nv:]


def holdout_split(N: int, train_frac: float = 0.7, val_frac_of_train: float = 0.1,
                  device=None) -> tuple[Tensor, Tensor, Tensor]:
    """v1 T8 split (first 70 % train, last 30 % test) with the last 10 % of the train part held out for model
    selection (v1 selected on the test part; v2 does not)."""
    n_train = int(train_frac * N)
    n_val = int(round(val_frac_of_train * n_train))
    ar = torch.arange(N, device=device)
    return ar[:n_train - n_val], ar[n_train - n_val:n_train], ar[n_train:]


def iterate_indices(idx: Tensor, batch_size: int, *, shuffle: bool, generator: torch.Generator | None = None,
                    drop_last: bool = False):
    """Yield index minibatches (CPU generator for device-independent determinism)."""
    n = idx.numel()
    if shuffle:
        perm = torch.randperm(n, generator=generator)
        idx = idx[perm.to(idx.device)]
    stop = n - (n % batch_size) if drop_last else n
    for s in range(0, stop, batch_size):
        yield idx[s:s + batch_size]


# ----------------------------------------------------------------------------------------------------------------
# cochain alignment (generator orderings -> complex orderings)
# ----------------------------------------------------------------------------------------------------------------
def edge_alignment(K_edges: Tensor, src_edges: Tensor, n0: int) -> tuple[Tensor, Tensor]:
    """Map the edges of a complex to the rows of another edge list.

    Args:
        K_edges: ``(n1, 2)`` edges of the complex (``src < dst``).
        src_edges: ``(n1, 2)`` edges in the data's order and orientation.
        n0: number of vertices.
    Returns:
        ``(perm, sign)``: ``values_K = values_src[perm] * sign`` for an odd edge cochain (``sign = +1`` where the two
        orientations agree), ``values_src[perm]`` for even edge data.
    Raises:
        ValueError: if the two edge sets differ.
    """
    K_edges = K_edges.long().cpu()
    src_edges = src_edges.long().cpu()
    if K_edges.shape != src_edges.shape:
        raise ValueError(f"edge sets differ in size: {tuple(K_edges.shape)} vs {tuple(src_edges.shape)}")
    kk = K_edges.min(1).values * n0 + K_edges.max(1).values
    ks = src_edges.min(1).values * n0 + src_edges.max(1).values
    order = torch.argsort(ks)
    pos = torch.searchsorted(ks[order], kk)
    pos = pos.clamp(max=len(ks) - 1)
    perm = order[pos]
    if not torch.equal(ks[perm], kk):
        raise ValueError("edge sets of the complex and of the data differ")
    same = (src_edges[perm, 0] == K_edges[:, 0])
    sign = torch.where(same, 1.0, -1.0)
    return perm, sign


def _parity_to(a: Tensor, b: Tensor) -> Tensor:
    """For rows of vertex tuples ``a`` and ``b`` with equal vertex sets, +1 if ``b`` is an even permutation of ``a``.

    Args:
        a, b: ``(m, p)`` int.
    Returns:
        ``(m,)`` float +-1.
    """
    # permutation sigma with b[:, j] = a[:, sigma[j]]; parity via counting inversions
    m, p = a.shape
    sigma = (b[:, :, None] == a[:, None, :]).float().argmax(-1)          # (m, p)
    inv = torch.zeros(m, dtype=torch.long)
    for i in range(p):
        for j in range(i + 1, p):
            inv += (sigma[:, i] > sigma[:, j]).long()
    return torch.where(inv % 2 == 0, 1.0, -1.0)


def cell_alignment(K_cells: Tensor, src_cells: Tensor, n0: int, *, oriented: bool = True) -> tuple[Tensor, Tensor]:
    """Map the k-cells (simplices, k>=2) of a complex to the rows of another cell list with the same vertex sets.

    Args:
        K_cells: ``(n_k, p)`` vertex tuples of the complex (``-1`` padding not supported).
        src_cells: ``(n_k, p)`` vertex tuples in the data's order/orientation.
        n0: number of vertices.
        oriented: also return the relative orientation sign.
    Returns:
        ``(perm, sign)``: ``values_K = values_src[perm] * sign`` (odd) or ``values_src[perm]`` (even).
    """
    Kc = K_cells.long().cpu()
    Sc = src_cells.long().cpu()
    if Kc.shape != Sc.shape:
        raise ValueError(f"cell sets differ in size: {tuple(Kc.shape)} vs {tuple(Sc.shape)}")
    if (Kc < 0).any():
        raise ValueError("cell_alignment: padded polygons are not supported")

    def key(c):
        s = torch.sort(c, 1).values
        k = torch.zeros(len(c), dtype=torch.long)
        for j in range(c.shape[1]):
            k = k * n0 + s[:, j]
        return k

    kk, ks = key(Kc), key(Sc)
    order = torch.argsort(ks)
    pos = torch.searchsorted(ks[order], kk).clamp(max=len(ks) - 1)
    perm = order[pos]
    if not torch.equal(ks[perm], kk):
        raise ValueError("cell sets of the complex and of the data differ")
    sign = _parity_to(Kc, Sc[perm]) if oriented else torch.ones(len(Kc))
    return perm, sign


# ----------------------------------------------------------------------------------------------------------------
# output maps (physical orientation supplied by the task)
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class OutputMap:
    """Fixed, parameter-free linear map from the model output to the task's target space.

    The v2 model is exactly equivariant under relabelling of cell orientations and invariant under reflections, so
    it cannot output *pseudo*-scalars/vectors (vorticity, magnetic flux, rotated gradients ``n x grad psi``), which are
    defined relative to the physical orientation of the domain.  The task supplies that orientation here:

    * ``kind='sparse'``: ``y_t = M y_m`` (e.g. the oriented face -> node average
      ``y_i = mean_{f ni i} sigma_f x_f`` with ``sigma_f = +1`` for counter-clockwise faces, applied to a
      ``cochain:2`` readout);
    * ``kind='cross_normal'``: ``v_i = n_i x g_i`` (rotation by +90 deg in the tangent plane of an oriented surface,
      applied to a ``node_vector`` readout);
    * ``kind='direct_vector'``: edge scalars -> vertex vectors ``v_i = sum_{e ni i} w_e t_e / deg_i`` (``t_e`` unit
      src->dst vector; the v1 edge-to-node average), ``M`` of shape ``(D n0, n1)`` (row ``D i + c``), applied to a
      ``grad`` readout (``w = d0 phi``) for T5g.

    Attributes:
        kind: ``'sparse' | 'cross_normal' | 'direct_vector'``.
        model_readout / model_out_dim: readout and output width the model must be configured with.
        M, MT: CSR ``(n_t, n_m)`` and its transpose (``sparse``).
        normals: ``(n0, 3)`` unit vertex normals (``cross_normal``).
        description: human-readable summary.
    """
    kind: str
    model_readout: str
    model_out_dim: int
    M: Tensor | None = None
    MT: Tensor | None = None
    normals: Tensor | None = None
    description: str = ""
    D: int = 0                     # direct_vector: embedding dimension of the output vectors

    def __call__(self, y: Tensor) -> Tensor:
        """``(n_m, B, O) -> (n_t, B, O)``."""
        if self.kind == "sparse":
            from rhmp.ops import spmm
            return spmm(self.M, y.contiguous(), self.MT)
        if self.kind == "cross_normal":
            n = self.normals.to(y.dtype)[:, None, :].expand_as(y)
            return torch.cross(n, y, dim=-1)
        if self.kind == "direct_vector":                    # (n1, B, 1) -> (n0, B, D)
            from rhmp.ops import spmm
            if y.shape[-1] != 1:
                raise ValueError("direct_vector output map expects one edge channel")
            z = spmm(self.M, y.contiguous(), self.MT)       # (D n0, B, 1)
            return z.view(-1, self.D, y.shape[1]).permute(0, 2, 1).contiguous()
        raise ValueError(f"unknown output map kind {self.kind!r}")

    def to(self, device) -> "OutputMap":
        """Copy on ``device``."""
        mv = lambda t: None if t is None else t.to(device)  # noqa: E731
        return OutputMap(self.kind, self.model_readout, self.model_out_dim, mv(self.M), mv(self.MT),
                         mv(self.normals), self.description, self.D)


# ----------------------------------------------------------------------------------------------------------------
# task container
# ----------------------------------------------------------------------------------------------------------------
@dataclass
class TaskData:
    """Everything a run needs, normalised and on ``device`` (DESIGN §4).

    Attributes:
        name: task name (e.g. ``'T6'``, ``'T6f'``, ``'HP_k100'``).
        K: shared complex, or list of complexes (variable meshes).
        inputs: ``{k: (N, n_k, F_k)}`` (shared) or list of ``{k: (n_k_i, F_k)}`` (variable meshes); normalised.
        target: ``(N, n_t, out_dim)`` or list of ``(n_t_i, out_dim)``; normalised.
        target_degree: degree of the output cells (0 for node targets).
        target_kind: ``'node_scalar' | 'node_vector' | 'cochain' | 'even'``.
        in_dims / even_dims / connection_dims: model input configuration (DESIGN §3.4/3.6).
        split: ``(train, val, test)`` sample indices (int64, CPU).
        x_stats: ``{k: Stats}`` input normalisation per degree.
        y_stats: target ``Stats`` (``denormalize`` gives physical units).
        spatial_dim: embedding dimension of the mesh (2 or 3).
        out_dim: number of output channels.
        native: native-cochain (True) or v1-legacy (False) inputs/outputs.
        meta: task defaults and diagnostics: ``v1_C``, ``batch_size``, ``readout``, ``vector_mode``,
            ``r2_std`` (v1 per-feature target std for a v1-comparable R2), ``paper_eval`` (indices of the 100 test
            samples used by the v1 paper tables), ``consistency`` checks, ...
        extra_tests: additional evaluation sets normalised with the same statistics, e.g.
            ``{'fine': {'K': [...], 'inputs': [...], 'target': [...]}}`` (zero-shot resolution transfer).
        output_map: optional :class:`OutputMap` applied to the model output before the loss/metrics; ``None``
            means the model output is compared with ``target`` directly.
        readout: **the** readout the model is built with (single source of truth; ``rhmp.train.build_config`` uses
            it).  ``None`` resolves once at construction to: the legacy ``meta['model_readout']`` if present, else
            ``output_map.model_readout``, else :attr:`target_readout` (the readout that predicts ``target`` on its
            own cells).  Examples: ``'cochain:2'`` + oriented face->node map for the pseudo-scalar node targets of
            T1/T6/T7, ``'grad'`` for the curl-free edge target of HPgrad.
    """
    name: str
    K: Any
    inputs: Any
    target: Any
    target_degree: int
    target_kind: str
    in_dims: dict[int, int]
    even_dims: dict[int, int]
    connection_dims: dict[int, int]
    split: tuple[Tensor, Tensor, Tensor]
    x_stats: dict[int, Stats]
    y_stats: Stats
    spatial_dim: int
    out_dim: int = 1
    native: bool = True
    meta: dict = field(default_factory=dict)
    extra_tests: dict = field(default_factory=dict)
    output_map: OutputMap | None = None
    readout: str | None = None

    def __post_init__(self) -> None:
        if self.readout is None:
            legacy = self.meta.get("model_readout") if isinstance(self.meta, dict) else None
            if legacy:
                self.readout = legacy
            elif self.output_map is not None and getattr(self.output_map, "model_readout", None):
                self.readout = self.output_map.model_readout
            else:
                self.readout = self.target_readout

    # ------------------------------------------------------------------------------------------------------------
    @classmethod
    def from_arrays(cls, K, inputs, target, *, readout: str, even_dims: dict | None = None,
                    connection_dims: dict | None = None, split=(0.7, 0.15, 0.15), stats: str = "auto",
                    name: str = "custom", output_map: "OutputMap | None" = None, meta: dict | None = None
                    ) -> "TaskData":
        """Build a normalised :class:`TaskData` from plain (raw, unnormalised) tensors.

        Args:
            K: one :class:`~rhmp.complex.CochainComplex` shared by all samples, or a list with one complex per sample.
            inputs: ``{k: (N, n_k, F_k)}`` (shared mesh) or a list of ``{k: (n_k_i, F_k)}`` (variable meshes).  On
                degree ``k >= 1`` the columns are ``[connection | odd | even]`` as in ``RHMPConfig`` (even listed last).
            target: ``(N, n_t, O)`` or a list of ``(n_t_i, O)``.
            readout: the model readout (``node_scalar | node_vector | cochain:k | even:k | grad | curl | div[:k]``);
                it also fixes the target cells and kind.
            even_dims / connection_dims: ``{k: count}`` as in ``RHMPConfig``.
            split: ``(train, val, test)`` fractions (sequential, ``int(f * N)`` as v1) or three index tensors.
            stats: ``'auto'``: per column, odd cochain columns get ``scale_stats`` (mean 0, RMS; keeps orientation
                oddness), vertex and even columns ``feature_stats`` (mean/std); the target gets ``scale_stats`` for odd
                readouts (``cochain``, ``grad``, ``curl``, ``div``), isotropic scaling for ``node_vector`` and
                ``feature_stats`` otherwise; statistics from the training split.  ``'none'``: identity.
            name, output_map, meta: passed through (``meta`` gets ``r2_std`` and a default ``batch_size``).
        Returns:
            TaskData (tensors stay on the device they were given on).
        """
        variable = isinstance(K, (list, tuple))
        even_dims = {int(k): int(v) for k, v in dict(even_dims or {}).items()}
        connection_dims = {int(k): int(v) for k, v in dict(connection_dims or {}).items()}
        rd = readout.strip()
        if rd in ("node_scalar", "node_vector"):
            kind, deg = rd, 0
        elif rd == "grad":
            kind, deg = "cochain", 1
        elif rd == "curl":
            kind, deg = "cochain", 2
        elif rd.startswith("div"):
            kk = int(rd.split(":")[1]) if ":" in rd else 1
            kind, deg = ("node_scalar", 0) if kk == 1 else ("cochain", kk - 1)
        elif ":" in rd and rd.split(":")[0] in ("cochain", "even"):
            kind, deg = rd.split(":")[0], int(rd.split(":")[1])
        else:
            raise ValueError(f"unknown readout {readout!r}")
        odd_target = kind == "cochain" or rd.startswith("div")
        N = len(target) if variable else int(target.shape[0])
        if isinstance(split[0], Tensor):
            split = tuple(torch.as_tensor(s_, dtype=torch.long).cpu() for s_ in split)
        else:
            nt, nv = int(split[0] * N), int(split[1] * N)
            ar = torch.arange(N)
            split = (ar[:nt], ar[nt:nt + nv], ar[nt + nv:])
        tr = split[0].tolist()
        degs = sorted(int(k) for k in (inputs[0] if variable else inputs))

        def train_part(k):
            if variable:
                return [inputs[i][k] for i in tr]
            return inputs[k][split[0].to(inputs[k].device)]

        def cols(x, a, b):
            return [t[..., a:b] for t in x] if isinstance(x, list) else x[..., a:b]

        def ident(F, like):
            return Stats(torch.zeros(F, device=like.device), torch.ones(F, device=like.device), "none")

        x_stats: dict[int, Stats] = {}
        for k in degs:
            xt = train_part(k)
            like = xt[0] if isinstance(xt, list) else xt
            F = int(like.shape[-1])
            if stats == "none":
                x_stats[k] = ident(F, like)
                continue
            if stats != "auto":
                raise ValueError(f"stats must be 'auto' or 'none', got {stats!r}")
            if k == 0:
                x_stats[k] = feature_stats(xt)
                continue
            n_odd = F - even_dims.get(k, 0)
            parts = []
            if n_odd > 0:
                parts.append(scale_stats(cols(xt, 0, n_odd)))
            if n_odd < F:
                parts.append(feature_stats(cols(xt, n_odd, F)))
            x_stats[k] = Stats(torch.cat([p_.mean.to(like.device) for p_ in parts]),
                               torch.cat([p_.std.to(like.device) for p_ in parts]), "auto")
        yt = [target[i] for i in tr] if variable else target[split[0].to(target.device)]
        ylike = yt[0] if isinstance(yt, list) else yt
        if stats == "none":
            y_stats = ident(int(ylike.shape[-1]), ylike)
        elif kind == "node_vector":
            y_stats = scale_stats(yt, isotropic=True)
        elif odd_target:
            y_stats = scale_stats(yt)
        else:
            y_stats = feature_stats(yt)
        if variable:
            inputs_n = [{k: x_stats[k].normalize(d[k]) for k in degs} for d in inputs]
            target_n = [y_stats.normalize(y) for y in target]
            K0 = K[0]
        else:
            inputs_n = {k: x_stats[k].normalize(inputs[k]).contiguous() for k in degs}
            target_n = y_stats.normalize(target).contiguous()
            K0 = K
        in_dims = {k: int((inputs[0] if variable else inputs)[k].shape[-1]) for k in degs}
        meta = dict(meta or {})
        meta.setdefault("r2_std", y_stats.std)
        meta.setdefault("batch_size", 8 if variable else 64)
        return cls(name=name, K=K, inputs=inputs_n, target=target_n, target_degree=deg, target_kind=kind,
                   in_dims=in_dims, even_dims=even_dims, connection_dims=connection_dims, split=split,
                   x_stats=x_stats, y_stats=y_stats, spatial_dim=int(K0.pos.shape[1]),
                   out_dim=int(ylike.shape[-1]), native=True, meta=meta, output_map=output_map, readout=rd)

    @property
    def variable_mesh(self) -> bool:
        """True when every sample has its own complex."""
        return isinstance(self.K, (list, tuple))

    @property
    def num_samples(self) -> int:
        """Number of samples ``N``."""
        return len(self.target) if self.variable_mesh else int(self.target.shape[0])

    @property
    def target_readout(self) -> str:
        """Readout that predicts ``target`` on its own cells (``node_scalar | node_vector | cochain:k | even:k``)."""
        if self.target_kind in ("node_scalar", "node_vector"):
            return self.target_kind
        return f"{self.target_kind}:{self.target_degree}"

    @property
    def model_readout(self) -> str:
        """Deprecated alias of :attr:`readout`."""
        return self.readout

    @property
    def geo_dims(self) -> dict[int, int]:
        """``{k: G_k}`` of the task's complexes (model constructor argument)."""
        K0 = self.K[0] if self.variable_mesh else self.K
        return {k: int(K0.geo_dim(k)) for k in range(K0.dim + 1)}

    def summary(self) -> dict:
        """JSON-friendly description (shapes, split sizes, normalisation)."""
        K0 = self.K[0] if self.variable_mesh else self.K
        if self.variable_mesh:
            n_cells = {k: [int(K.n[k]) for K in self.K] for k in range(K0.dim + 1)}
            cells = {k: (min(v), max(v), sum(v) / len(v)) for k, v in n_cells.items()}
            shapes = {k: [None, "var", int(self.inputs[0][k].shape[-1])] for k in self.inputs[0]}
            tshape = [None, "var", int(self.target[0].shape[-1])]
        else:
            cells = {k: int(K0.n[k]) for k in range(K0.dim + 1)}
            shapes = {k: list(v.shape) for k, v in self.inputs.items()}
            tshape = list(self.target.shape)
        return {
            "name": self.name, "native": self.native, "N": self.num_samples, "cells": cells,
            "inputs": shapes, "target": tshape, "target_degree": self.target_degree,
            "target_kind": self.target_kind, "readout": self.readout, "target_readout": self.target_readout,
            "output_map": None if self.output_map is None else self.output_map.description,
            "in_dims": self.in_dims, "even_dims": self.even_dims, "connection_dims": self.connection_dims,
            "split": [int(s.numel()) for s in self.split], "spatial_dim": self.spatial_dim,
            "x_stats": {k: v.to_dict() for k, v in self.x_stats.items()}, "y_stats": self.y_stats.to_dict(),
            "extra_tests": list(self.extra_tests),
            "meta": {k: v for k, v in self.meta.items() if isinstance(v, (int, float, str, bool, dict, list))},
        }


# ----------------------------------------------------------------------------------------------------------------
# minibatches
# ----------------------------------------------------------------------------------------------------------------
def shared_minibatch(inputs: dict[int, Tensor], target: Tensor, idx: Tensor) -> tuple[dict[int, Tensor], Tensor]:
    """Gather a shared-mesh minibatch in the model layout.

    Args:
        inputs: ``{k: (N, n_k, F_k)}``.
        target: ``(N, n_t, O)``.
        idx: ``(B,)`` sample indices.
    Returns:
        ``({k: (n_k, B, F_k)}, (n_t, B, O))`` contiguous.
    """
    dev = target.device
    idx = idx.to(dev)
    xb = {k: to_nbc(v.index_select(0, idx)) for k, v in inputs.items()}
    return xb, to_nbc(target.index_select(0, idx))


def concat_cells(per_sample: Sequence[Tensor]) -> Tensor:
    """Concatenate per-sample cell features ``[(n_i, F)]`` into the block-diagonal layout ``(sum n_i, 1, F)``."""
    return torch.cat(list(per_sample), 0).unsqueeze(1).contiguous()


def mesh_minibatch(Ks: Sequence, inputs: Sequence[dict[int, Tensor]], target: Sequence[Tensor], idx,
                   device=None, cache: dict | None = None):
    """Block-diagonal minibatch of variable meshes.

    Args:
        Ks: list of complexes (any device).
        inputs: list of ``{k: (n_k_i, F_k)}``.
        target: list of ``(n_t_i, O)``.
        idx: sample indices (tensor or list).
        device: device of the returned batch (defaults to the device of the first target).
        cache: optional dict; batched complexes are memoised by the tuple of indices (use for the fixed
            validation/test batches).
    Returns:
        ``(K_batch, {k: (sum n_k, 1, F_k)}, (sum n_t, 1, O))``.
    """
    from rhmp.complex import CochainComplex

    ids = [int(i) for i in (idx.tolist() if isinstance(idx, Tensor) else idx)]
    dev = torch.device(device) if device is not None else target[ids[0]].device
    key = tuple(ids)
    Kb = cache.get(key) if cache is not None else None
    if Kb is None:
        Kb = CochainComplex.batch([Ks[i] for i in ids])
    xb = {k: concat_cells([inputs[i][k] for i in ids]) for k in inputs[ids[0]]}
    yb = concat_cells([target[i] for i in ids])
    if not (same_device(Kb.pos.device, dev) and same_device(yb.device, dev)):
        # CPU-resident data: one pinned, non-blocking copy per dtype for the whole batch (per-tensor pageable
        # copies cost ~5 ms each on a GPU shared with other processes)
        Kb, xb, yb = pack_to((Kb, xb, yb), dev)
    if cache is not None:
        cache[key] = Kb
    return Kb, xb, yb
