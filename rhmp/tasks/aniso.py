"""Anisotropy task suite: PDEs whose operator is the Whitney/Galerkin Hodge star of a misaligned SPD material tensor
(generator ``datasets/generators/gen_aniso.py``; equations, discretisations, sizes and protocol in ``docs/ANISO_TASKS.md``).

===============  ===============  ================================================================  ====================
name             data file        inputs by degree (odd columns first, even last)                    target (readout)
===============  ===============  ================================================================  ====================
AHP_r<R>         AHP_r<R>         0: f | 1: log t^T S t (e) | 2: log det S, log ratio S (e)          u nodes
                                                                                                     (node_scalar)
ASURF_r<R>       ASURF_r<R>       as AHP, closed surfaces in 3-D (tangent-plane fibre tensor)       u (node_scalar)
ASURF_heat_r<R>  ASURF_r<R>       as ASURF                                                           heat flow u_T
                                                                                                     (node_scalar)
ACURL_r<R>       ACURL_r<R>       1: j (odd source 1-cochain), log t^T nu t (e) |                   A edges (cochain:1)
                                  2: log n^T nu n (e) | 3: log det nu, log ratio nu (e)
ACURLb_r<R>      ACURL_r<R>       as ACURL                                                           B = d1 A faces
                                                                                                     (curl: d2 B = 0)
ADARCY_r<R>      ADARCY_r<R>      0: f | 1: log t^T K t (e) | 2: log n^T K^-1 n (e) |               J face fluxes
                                  3: F_T = int_T f (odd), log det K, log ratio K (e)                (cochain:2)
ADARCYp_r<R>     ADARCY_r<R>      as ADARCY                                                          p nodes
                                                                                                     (node_scalar)
===============  ===============  ================================================================  ====================

``R`` in {10, 100} (maximal eigenvalue ratio; the ratio fades to 1 only near the zeros of the direction field).
Variable meshes (one complex per sample, block-diagonal batches), sequential 70/15/15 split, statistics from the
training part (``TaskData.from_arrays``: odd columns / odd targets scale-only, even and vertex columns mean/std),
``extra_tests['fine']`` = the first test samples re-solved on meshes with 4x the nodes (same physical instances),
ASURF also ``test_<family>``.  Triangle complexes use ``star`` (default ``cotan``), tetrahedral complexes
``barycentric`` when ``cotan`` is requested.  ``spatial_dim`` is 2 (AHP) or 3.

``meta['metric_ref']`` suggests the trainer's ``--metric-ref`` for the degree whose metric carries the material
(``1:0`` edges for the scalar problems, ``2:0`` faces for the 2-form problems), ``meta['recovery']`` tells
``scripts/metric_recovery.py`` which stored tensor each learned metric should be compared with, and
``meta['representability']`` summarises the generator's per-cell diagnostics.  :func:`structure_metrics` evaluates
the physics residuals of a model's predictions (discrete PDE residual, boundary conditions, conservation, d2 B = 0).
"""
from __future__ import annotations

import functools
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch import Tensor

from rhmp.data import TaskData, cell_alignment, edge_alignment, pack_to, sequential_split, tensor_nbytes

__all__ = ["ANISO_TASKS", "ANISO_DEFAULTS", "SPECS", "aniso_task_defaults", "parse_name", "load_aniso",
           "structure_metrics", "cell_tensors"]

BUILD_THREADS = int(os.environ.get("RHMP_BUILD_THREADS", 4))   # complexes are built in a thread pool
SURF_FAMILIES = ("ellipsoid", "superquadric", "sphere_pert", "torus", "double_torus")   # codes of gen_surf.py

# kind -> data file, top cells, target, readout, suggested --metric-ref, material degree of the physics metric,
# truth key of the material tensor and whether the physics metric is its inverse
SPECS: dict[str, dict] = {
    "AHP": dict(file="AHP", top="faces", target="u", readout="node_scalar", metric_ref="1:0",
                tensor_km=1, tensor_key="sigma_face", inverse=False, diag_m=1),
    "ASURF": dict(file="ASURF", top="faces", target="u", readout="node_scalar", metric_ref="1:0",
                  tensor_km=1, tensor_key="sigma_face", inverse=False, diag_m=1),
    "ASURF_heat": dict(file="ASURF", top="faces", target="heat", readout="node_scalar", metric_ref="1:0",
                       tensor_km=1, tensor_key="sigma_face", inverse=False, diag_m=1),
    "ACURL": dict(file="ACURL", top="tets", target="A", readout="cochain:1", metric_ref="2:0",
                  tensor_km=2, tensor_key="nu_tet", inverse=False, diag_m=2),
    "ACURLb": dict(file="ACURL", top="tets", target="B", readout="curl", metric_ref="2:0",
                   tensor_km=2, tensor_key="nu_tet", inverse=False, diag_m=2),
    "ADARCY": dict(file="ADARCY", top="tets", target="J", readout="cochain:2", metric_ref="2:0",
                   tensor_km=2, tensor_key="K_tet", inverse=True, diag_m=2),
    "ADARCYp": dict(file="ADARCY", top="tets", target="p", readout="node_scalar", metric_ref="1:0",
                    tensor_km=1, tensor_key="K_tet", inverse=False, diag_m=1),
}
ANISO_DEFAULTS: dict[str, dict] = {k: dict(C=128, layers=4, batch=8) for k in SPECS}   # batch = meshes per step
RATIOS = (10, 100)
SUBSETS_3D = (1500,)       # registered '_n<N>' variants of the tet tasks (first N samples; see load_aniso)
_NAME_RE = re.compile(r"^(AHP|ASURF_heat|ASURF|ACURLb|ACURL|ADARCYp|ADARCY)_r(\d+)(?:_n(\d+))?$")
# degree of every stored key (flat packing with offsets ptr0..ptr3)
_KEY_DEG = {"pos": 0, "f": 0, "u": 0, "heat": 0, "p": 0, "bnd": 0, "j": 1, "A": 1, "bnd_edge": 1, "J": 2, "faces": 2,
            "tets": 3}


def parse_name(name: str) -> tuple[str, int]:
    """``'ACURLb_r100' -> ('ACURLb', 100)`` (an optional ``_n<N>`` subset suffix is ignored here)."""
    m = _NAME_RE.match(name)
    if not m:
        raise KeyError(f"unknown anisotropy task {name!r}; known: {sorted(ANISO_TASKS)}")
    return m.group(1), int(m.group(2))


def subset_of(name: str) -> int | None:
    """``'ACURL_r100_n1500' -> 1500`` (``None`` without the suffix)."""
    m = _NAME_RE.match(name)
    return int(m.group(3)) if (m and m.group(3)) else None


def aniso_task_defaults(name: str) -> dict:
    """Training defaults (``C``, ``layers``, ``batch`` = meshes per block-diagonal batch)."""
    return dict(ANISO_DEFAULTS[parse_name(name)[0]])


# ======================================================================================================================
# data access
# ======================================================================================================================
def _key_deg(key: str) -> int:
    if key in _KEY_DEG:
        return _KEY_DEG[key]
    for suf, d in (("_edge", 1), ("_face", 2), ("_tet", 3)):
        if key.endswith(suf):
            return d
    raise KeyError(f"no degree known for key {key!r}")


@functools.lru_cache(maxsize=2)
def _load_blob(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run CUDA_VISIBLE_DEVICES= python3 -u datasets/generators/gen_aniso.py")
    return torch.load(path, map_location="cpu", weights_only=False)


def _sl(blob: dict, key: str, i: int) -> Tensor:
    p = blob[f"ptr{_key_deg(key)}"]
    return blob[key][int(p[i]):int(p[i + 1])]


def _num_samples(blob: dict) -> int:
    return len(blob["ptr0"]) - 1


def _mesh_edges(cells: Tensor, n0: int) -> Tensor:
    from rhmp.tasks.synthetic import _mesh_edges_torch
    return _mesh_edges_torch(cells, n0)


def _tet_faces(tets: Tensor, n0: int) -> Tensor:
    from rhmp.tasks.synthetic import _tet_faces_torch
    return _tet_faces_torch(tets, n0)


def _build_one(pos: Tensor, top: Tensor, star: str, whitney: bool = True):
    from rhmp.complex import CochainComplex
    if top.shape[1] == 3:
        K = CochainComplex.from_triangles(pos.double(), top.long(), star=star, device="cpu")
    else:
        K = CochainComplex.from_tetrahedra(pos.double(), top.long(), star=star, device="cpu")
    if K.n[K.dim] != top.shape[0]:            # the generator never produces degenerate or duplicate cells
        raise RuntimeError(f"the complex dropped {top.shape[0] - K.n[K.dim]} top cells")
    if not whitney:                           # drop the Galerkin blocks at once (70 % of a tet complex)
        K.whitney = {}
    return K


def _build_complexes(meshes: list, star: str, whitney: bool = True) -> list:
    """CPU complexes of ``[(pos, top), ...]`` in a thread pool (bitwise identical to a serial build)."""
    if BUILD_THREADS <= 1 or len(meshes) < 8:
        return [_build_one(p, t, star, whitney) for p, t in meshes]
    nt = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with ThreadPoolExecutor(BUILD_THREADS) as ex:
            return list(ex.map(lambda m: _build_one(m[0], m[1], star, whitney), meshes))
    finally:
        torch.set_num_threads(nt)


def _alignment(K, blob: dict, i: int, top_key: str) -> dict:
    """Permutations/signs mapping generator cell orders to the complex's (edges; faces or tets; tets)."""
    n0 = K.n[0]
    top = _sl(blob, top_key, i).long()
    eperm, esign = edge_alignment(K.cells[1], _mesh_edges(top, n0), n0)
    out = dict(eperm=eperm, esign=esign)
    if top_key == "faces":
        out["fperm"], _ = cell_alignment(K.cells[2], top, n0, oriented=False)
    else:
        out["fperm"], out["fsign"] = cell_alignment(K.cells[2], _tet_faces(top, n0), n0, oriented=True)
        out["tperm"], out["tsign"] = cell_alignment(K.cells[3], top, n0, oriented=True)
    return out


def _sample_arrays(kind: str, blob: dict, i: int, K, al: dict) -> tuple[dict, Tensor]:
    """Raw (unnormalised) inputs ``{k: (n_k, F_k)}`` and target ``(n_t, 1)`` of sample ``i`` in the complex's order."""
    sl = functools.partial(_sl, blob, i=i)
    e, s = al["eperm"], al["esign"]
    if kind in ("AHP", "ASURF", "ASURF_heat"):
        fp = al["fperm"]
        x = {0: sl("f").float()[:, None], 1: sl("logproj_edge").float()[e][:, None],
             2: torch.stack([sl("logdet_face"), sl("logratio_face")], 1).float()[fp]}
        y = sl("heat" if kind == "ASURF_heat" else "u").float()[:, None]
        return x, y
    fp, fs, tp, ts = al["fperm"], al["fsign"], al["tperm"], al["tsign"]
    tet_even = torch.stack([sl("logdet_tet"), sl("logratio_tet")], 1).float()[tp]
    if kind in ("ACURL", "ACURLb"):
        j = (sl("j").double()[e] * s)
        x = {1: torch.stack([j.float(), sl("logproj_edge").float()[e]], 1), 2: sl("lognn_face").float()[fp][:, None],
             3: tet_even}
        A = sl("A").double()[e] * s
        if kind == "ACURL":
            return x, A.float()[:, None]
        B = K.apply_d(1, A.to(K.star[0].device)[:, None, None].contiguous())[:, 0, :]    # exact d1 A (float64)
        return x, B.float()
    # ADARCY / ADARCYp
    FT = (sl("F_tet").double()[tp] * ts).float()
    x = {0: sl("f").float()[:, None], 1: sl("logproj_edge").float()[e][:, None],
         2: sl("lognninv_face").float()[fp][:, None], 3: torch.cat([FT[:, None], tet_even], 1)}
    if kind == "ADARCY":
        return x, (sl("J").double()[fp] * fs).float()[:, None]
    return x, sl("p").float()[:, None]


def _build_set(kind: str, blob: dict, ids, star: str, whitney: bool = True):
    """Complexes (CPU), raw inputs, raw targets of the samples ``ids``."""
    spec = SPECS[kind]
    ids = list(ids)
    t0 = time.time()
    Ks = _build_complexes([(_sl(blob, "pos", i), _sl(blob, spec["top"], i)) for i in ids], star, whitney)
    xs, ys = [], []
    for i, K in zip(ids, Ks):
        x, y = _sample_arrays(kind, blob, i, K, _alignment(K, blob, i, spec["top"]))
        xs.append(x)
        ys.append(y)
    n = min(20, len(Ks))
    info = dict(build_seconds=time.time() - t0, n=len(Ks),
                n0_mean=float(np.mean([K.n[0] for K in Ks])) if Ks else 0.0,
                mean_MB=sum(tensor_nbytes(K) for K in Ks[:n]) / max(n, 1) / 1e6)
    return Ks, xs, ys, info


def _truncated(split: tuple, max_samples: int | None):
    """Smoke tests: the first ``max_samples // 3`` samples of every split part (as the HP / suite loaders)."""
    N = sum(len(s) for s in split)
    if max_samples is None or max_samples >= N:
        return list(range(N)), split
    per = max(1, max_samples // 3)
    ids = torch.cat([s[:per] for s in split]).tolist()
    sizes = [min(per, len(s)) for s in split]
    off = np.cumsum([0] + sizes)
    return ids, tuple(torch.arange(off[j], off[j + 1]) for j in range(3))


def _representability(blob: dict) -> dict:
    st = blob.get("sample_stats") or []
    keys = ("cone_frac", "inv_cone_frac", "neg_weight_frac", "smax_median", "smax_p90", "frac_smax_le3",
            "frac_smax_le5", "inv_smax_median", "ratio_median", "frac_ratio_gt_half")
    return {k: float(np.mean([s[k] for s in st])) for k in keys if st and k in st[0]}


# ======================================================================================================================
# loader
# ======================================================================================================================
def load_aniso(name: str, droot: str | None = None, *, native: bool = True, device="cuda", star: str = "cotan",
               fine: bool = True, max_samples: int | None = None, keep_on: str = "auto", n_samples: int | None = None,
               fine_max: int | None = None, whitney: bool = True, **_) -> TaskData:
    """Load an anisotropy task (see the module table) as a normalised :class:`TaskData`.

    Args:
        name: e.g. ``'ASURF_r100'``, ``'ACURLb_r10'``; the suffix ``_n<N>`` (registered for the tet tasks:
            ``_n1500``) uses only the first ``N`` samples of the file with the standard sequential 70/15/15 split,
            i.e. the data set that ``gen_aniso.py --n N`` would have produced (the 4x ``fine`` set is unchanged).
            A tet complex with Whitney blocks takes ~45 MB (~14 MB without, for diagonal metrics), so the 3000-sample
            tet sets need ~137 GB of host memory for tensor-metric runs; ``_n1500`` needs ~68 GB.
        droot: repository root or datasets directory (``None``: ``rhmp.tasks.resolve_root``).
        native: must be True (the cochain inputs/targets are the task definition).
        device: device of the statistics (and of the data when it fits ``rhmp.tasks.synthetic.GPU_BUDGET_GB``).
        star: reference star of triangle complexes (tets: ``barycentric`` when ``cotan`` is requested).
        fine: also build the 4x-finer ``fine`` extra test set.
        fine_max: at most this many ``fine`` samples (default: all for triangle tasks, 50 for tet tasks, whose 4x
            complexes take ~170 MB each with Whitney blocks).
        max_samples: smoke tests (first ``max_samples // 3`` samples of each split part; ``fine`` truncated too).
        keep_on: ``'auto' | device | 'cpu'``: where the complexes and tensors live.
        whitney: keep the Whitney/Galerkin blocks (needed by tensor metrics only); ``False`` drops them per complex
            while building (``rhmp.tasks.load_task`` strips them only after loading, so its peak includes them).
    """
    from rhmp.tasks import resolve_root
    from rhmp.tasks.synthetic import GPU_BUDGET_GB
    kind, ratio = parse_name(name)
    if not native:
        raise ValueError(f"{name}: the anisotropy tasks are defined on cochains (native=True only); node-input "
                         "baselines get their encodings from rhmp.baselines")
    spec = SPECS[kind]
    t_all = time.time()
    droot = resolve_root(droot)
    path = os.path.join(droot, "v2", f"{spec['file']}_r{ratio}.pt")
    blob = _load_blob(path)
    star_used = "barycentric" if (spec["top"] == "tets" and star == "cotan") else star
    N = _num_samples(blob)
    n_sub = n_samples if n_samples is not None else subset_of(name)
    if n_sub is not None:
        if not 3 <= n_sub <= N:
            raise ValueError(f"{name}: subset of {n_sub} samples requested, the file has {N}")
        N = int(n_sub)
    ids, split = _truncated(sequential_split(N), max_samples)
    Ks, xs, ys, info = _build_set(kind, blob, ids, star_used, whitney)
    even = {1: 1, 2: 2} if spec["top"] == "faces" else {1: 1, 2: 1, 3: 2}
    meta = dict(v1_C=ANISO_DEFAULTS[kind]["C"], batch_size=ANISO_DEFAULTS[kind]["batch"],
                layers=ANISO_DEFAULTS[kind]["layers"], source=path, generator=blob["meta"], aniso_kind=kind,
                ratio=ratio, star=star_used, star_requested=star, target=spec["target"], sample_ids=ids,
                n_file=_num_samples(blob), n_used=N,
                metric_ref=spec["metric_ref"], representability=_representability(blob),
                recovery=dict(tensor={spec["tensor_km"]: dict(key=spec["tensor_key"], inverse=spec["inverse"],
                                                                surface=kind.startswith("ASURF"))},
                              diag={spec["diag_m"]: dict(degree=spec["diag_m"], column=0)}),
                structure=_STRUCTURE_DOC[kind])
    if kind == "ADARCY":                 # the flux problem also exposes the scalar (P1) material metric
        meta["recovery"]["tensor"][1] = dict(key="K_tet", inverse=False, surface=False)
    if kind in ("AHP", "ADARCYp"):       # Dirichlet P1 problems: K u = M f with M = star0 (trainer --aux-pde, solver mode)
        meta["pde"] = dict(degree=0, source="inputs[0][...,0]", bc="dirichlet")
    td = TaskData.from_arrays(Ks, xs, ys, readout=spec["readout"], even_dims=even, split=split, name=name, meta=meta)
    # extra test sets (normalised with the training statistics, kept on the CPU and moved per batch)
    extra: dict = {}
    fam_sel: dict[str, list[int]] = {}
    if kind.startswith("ASURF"):
        fam = [int(blob["family"][i]) for i in ids]
        te = split[2].tolist()
        for code in sorted({fam[i] for i in te}):
            fam_sel[f"test_{SURF_FAMILIES[code]}"] = [i for i in te if fam[i] == code]
        td.meta["family_of_sample"] = fam
    fpath = path.replace(".pt", "_fine.pt")
    if fine and os.path.exists(fpath):
        fb = _load_blob(fpath)
        nf = _num_samples(fb)
        cap = fine_max if fine_max is not None else (50 if spec["top"] == "tets" else nf)
        if max_samples is not None:
            cap = min(cap, max_samples)
        fids = list(range(min(nf, cap)))
        fK, fx, fy, finfo = _build_set(kind, fb, fids, star_used, whitney)
        extra["fine"] = dict(K=fK, inputs=[{k: td.x_stats[k].normalize(x[k]) for k in x} for x in fx],
                             target=[td.y_stats.normalize(y) for y in fy], info=finfo, source=fpath,
                             sample_ids=fids, coarse_ids=fb["meta"].get("coarse_ids"),
                             fine_factor=fb["meta"].get("fine_factor"))
    # placement of the main data (all tensors incl. Whitney blocks count)
    n = min(50, len(Ks))
    est_gb = sum(tensor_nbytes(K) for K in Ks[:n]) / max(n, 1) * len(Ks) / 1e9
    on = keep_on if keep_on != "auto" else (str(device) if est_gb < GPU_BUDGET_GB else "cpu")
    td.K, td.inputs, td.target = pack_to((td.K, td.inputs, td.target), on)
    for key, sel in fam_sel.items():          # per-family test subsets: views of the (moved) main data
        extra[key] = dict(K=[td.K[i] for i in sel], inputs=[td.inputs[i] for i in sel],
                          target=[td.target[i] for i in sel])
    td.extra_tests = extra
    td.x_stats = {k: s.to(device) for k, s in td.x_stats.items()}
    td.y_stats = td.y_stats.to(device)
    td.meta["complexes"] = dict(info, estimated_GB=est_gb, stored_on=str(on))
    td.meta["load_seconds"] = time.time() - t_all
    _load_blob.cache_clear()                  # the raw files (up to ~4 GB) are only re-read by evaluation helpers
    return td


def _loader(name: str):
    def load(droot=None, **kw):
        return load_aniso(name, droot, **kw)
    load.__name__ = f"load_{name}"
    load.__doc__ = f"``load_aniso({name!r}, droot, **kw)``."
    return load


ANISO_TASKS: dict = {f"{k}_r{r}": _loader(f"{k}_r{r}") for k in SPECS for r in RATIOS}
ANISO_TASKS.update({f"{k}_r{r}_n{n}": _loader(f"{k}_r{r}_n{n}") for k in SPECS if SPECS[k]["top"] == "tets"
                    for r in RATIOS for n in SUBSETS_3D})


# ======================================================================================================================
# ground truth access and structure metrics
# ======================================================================================================================
def cell_tensors(blob: dict, i: int, key: str, K, top_key: str, inverse: bool = False) -> Tensor:
    """Stored per-cell material tensors of sample ``i`` in the complex's top-cell order: ``(n_top, D, D)`` float64
    (``inverse``: their inverses -- full-rank planar / volume tensors only; surface tensors are rank 2)."""
    v = _sl(blob, key, i).double()
    n0 = K.n[0]
    top = _sl(blob, top_key, i).long()
    perm, _ = cell_alignment(K.cells[K.dim], top, n0, oriented=False)
    v = v[perm]
    if v.shape[1] == 3:
        T = torch.stack([torch.stack([v[:, 0], v[:, 1]], -1), torch.stack([v[:, 1], v[:, 2]], -1)], -2)
    else:
        T = torch.stack([torch.stack([v[:, 0], v[:, 1], v[:, 2]], -1), torch.stack([v[:, 1], v[:, 3], v[:, 4]], -1),
                         torch.stack([v[:, 2], v[:, 4], v[:, 5]], -1)], -2)
    return torch.linalg.inv(T) if inverse else T


def _to_scipy(csr: Tensor):
    import scipy.sparse as sp
    c = csr.to_sparse_csr() if csr.layout != torch.sparse_csr else csr
    return sp.csr_matrix((c.values().double().cpu().numpy(), c.col_indices().cpu().numpy(),
                          c.crow_indices().cpu().numpy()), shape=tuple(c.shape))


def _p1_stiffness(K, S: Tensor):
    """P1 stiffness ``sum_T |T| grad(phi)^T S_T grad(phi)`` of a triangle/tet complex (scipy csr, float64)."""
    import scipy.sparse as sp

    from rhmp.geometry import _simplex_measure, barycentric_gradients
    cells = K.cells[K.dim].long().cpu()
    V = K.pos.double().cpu()[cells]
    G = barycentric_gradients(V)
    meas = _simplex_measure(V)
    Kl = torch.einsum("nid,nde,nje->nij", G, S, G) * meas[:, None, None]
    m = cells.shape[1]
    rows = cells[:, :, None].expand(-1, m, m).reshape(-1).numpy()
    cols = cells[:, None, :].expand(-1, m, m).reshape(-1).numpy()
    return sp.csr_matrix((Kl.reshape(-1).numpy(), (rows, cols)), shape=(K.n[0], K.n[0]))


def _whitney_complex(K):
    """A CPU complex with Whitney blocks (rebuilt from the positions when the loaded one had them stripped)."""
    Kc = K.to("cpu")
    if Kc.whitney:
        return Kc
    from rhmp.complex import CochainComplex
    return CochainComplex.from_tetrahedra(Kc.pos.double(), Kc.cells[3].long(), star="barycentric", device="cpu")


def _galerkin(Kw, k: int, T: Tensor):
    """Assembled Whitney Galerkin metric ``H_k(T) = sum_T int w_i . T w_j`` (scipy csr, float64) of per-top-cell
    tensors ``T (n_top, D, D)``: element blocks from ``rhmp.layers.galerkin_blocks`` (float64 basis values, exact,
    the model's own full-tensor assembly), scattered with the slot maps of ``Kw.whitney[k]``."""
    import scipy.sparse as sp
    cells = Kw.whitney[k]["cells"].long().cpu()
    n_top, m = cells.shape
    try:
        from rhmp.layers import galerkin_blocks, galerkin_geometry
        M = galerkin_blocks(galerkin_geometry(Kw, k, torch.float64), T.double()[:, None])[:, 0]
    except ImportError:                                  # older rhmp: signed dyad coefficients, fp32 Whitney blocks
        from rhmp.dec import whitney_blocks
        t = Kw.whitney[k]["t"].double()
        dd = t[:, :, :, None] * t[:, :, None, :]
        iu = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        A = torch.stack([dd[:, :, a, b] for a, b in iu], 1)
        a = torch.linalg.solve(A, torch.stack([T[:, a, b] for a, b in iu], 1)[..., None])[..., 0]
        M = whitney_blocks(Kw, k, torch.zeros(n_top, 1, dtype=torch.float64), a[:, None])[:, 0]
    rows = cells[:, :, None].expand(n_top, m, m).reshape(-1).numpy()
    cols = cells[:, None, :].expand(n_top, m, m).reshape(-1).numpy()
    return sp.csr_matrix((M.reshape(-1).cpu().numpy(), (rows, cols)), shape=(Kw.n[k], Kw.n[k]))


_STRUCTURE_DOC = {
    "AHP": "pde_res = |K(S) u - M f|_I / |M f|_I (interior nodes, M = star0 = lumped mass); bc = |u_bnd| / |u|",
    "ASURF": "pde_res = |(M + K(S)) u - M f| / |M f|; integral = |sum M u - sum M f| / sum M|f| (exact: 0)",
    "ASURF_heat": "integral = |sum M u_T - sum M f| / sum M|f| (heat flow conserves the integral exactly)",
    "ACURL": "pde_res = |(d1^T M2(nu) d1 + eps M1) A - M1 j|_I / |M1 j|_I (interior edges); bc = |A_bnd| / |A| "
             "(PEC: tangential A = 0); gauge = |d0^T M1 A|_I / |M1 A| (weak Coulomb gauge: 0 for the exact solution)",
    "ACURLb": "div = |d2 B| / |B| (exactly 0 for the curl readout); bc = |B_bnd faces| / |B| (normal B = 0)",
    "ADARCY": "cons = |d2 J - F| / |F| (per-tet mass balance); irrot = |d1^T M2(K^-1) J| / |M2(K^-1) J| (K^-1 j is a "
              "gradient)",
    "ADARCYp": "pde_res = |K(K) p - M f|_I / |M f|_I; bc = |p_bnd| / |p|",
}


def _norm(x) -> float:
    return float(np.linalg.norm(x))


def _residuals(kind: str, K, x: dict, y: np.ndarray, blob: dict, sid: int, eps: float) -> dict:
    """Physics residuals of one physical-unit prediction ``y (n_t,)`` (inputs ``x`` in physical units)."""
    spec = SPECS[kind]
    top_key = spec["top"]
    if kind in ("AHP", "ADARCYp", "ASURF", "ASURF_heat"):
        M = K.star[0].double().cpu().numpy()
        f = x[0][:, 0]
        if kind == "ASURF_heat":
            return dict(integral=abs(float((M * y).sum() - (M * f).sum())) / float((M * np.abs(f)).sum()))
        S = cell_tensors(blob, sid, spec["tensor_key"], K, top_key)
        Kop = _p1_stiffness(K, S)
        if kind == "ASURF":
            r = M * y + Kop @ y - M * f
            return dict(pde_res=_norm(r) / _norm(M * f),
                        integral=abs(float((M * y).sum() - (M * f).sum())) / float((M * np.abs(f)).sum()))
        bnd = K.boundary[0].cpu().numpy()
        r = (Kop @ y - M * f)[~bnd]
        return dict(pde_res=_norm(r) / _norm((M * f)[~bnd]), bc=_norm(y[bnd]) / max(_norm(y), 1e-300))
    d1, d2 = _to_scipy(K.d[1]), _to_scipy(K.d[2])
    if kind == "ACURLb":
        bf = K.boundary[2].cpu().numpy()
        return dict(div=_norm(d2 @ y) / max(_norm(y), 1e-300), bc=_norm(y[bf]) / max(_norm(y), 1e-300))
    Kw = _whitney_complex(K)
    if kind == "ACURL":
        nu = cell_tensors(blob, sid, "nu_tet", K, top_key)
        M1 = _galerkin(Kw, 1, torch.eye(3, dtype=torch.float64).expand(Kw.n[3], 3, 3))
        Sop = d1.T @ _galerkin(Kw, 2, nu) @ d1 + eps * M1
        b = M1 @ x[1][:, 0]
        be = K.boundary[1].cpu().numpy()
        r = (Sop @ y - b)[~be]
        bn = K.boundary[0].cpu().numpy()
        My = M1 @ y
        return dict(pde_res=_norm(r) / _norm(b[~be]), bc=_norm(y[be]) / max(_norm(y), 1e-300),
                    gauge=_norm((_to_scipy(K.d[0]).T @ My)[~bn]) / max(_norm(My), 1e-300))
    # ADARCY
    Kinv = cell_tensors(blob, sid, "K_tet", K, top_key, inverse=True)
    M2 = _galerkin(Kw, 2, Kinv)
    F = x[3][:, 0]
    MJ = M2 @ y
    return dict(cons=_norm(d2 @ y - F) / _norm(F), irrot=_norm(d1.T @ MJ) / max(_norm(MJ), 1e-300))


@torch.no_grad()
def structure_metrics(model, task: TaskData, *, source: dict | None = None, idx=None, n: int = 32,
                      batch_size: int = 4, device=None, targets: bool = True) -> dict:
    """Physics residuals of a model's predictions on an anisotropy task (see ``task.meta['structure']``).

    Args:
        model: called as ``model(inputs, K)`` (normalised output; the task's ``readout``).
        source: evaluation set ``{'K', 'inputs', 'target', 'source', 'sample_ids'}`` (default: the task's test split;
            for ``task.extra_tests['fine']`` pass that dict).
        idx: samples of ``source`` (default: the first ``n`` of the test split / of the set).
        targets: also report the residuals of the stored targets (round-off reference).
    Returns:
        ``{metric: {'mean', 'max'}}`` for the predictions and ``target_<metric>`` for the stored targets.
    """
    from rhmp.train import predict
    kind, _ = parse_name(task.name)
    device = torch.device(device) if device is not None else next(model.parameters()).device
    if source is None:
        src = dict(K=task.K, inputs=task.inputs, target=task.target)
        ids_all = task.meta["sample_ids"]
        path = task.meta["source"]
        idx = task.split[2][:n] if idx is None else torch.as_tensor(idx)
    else:
        src, ids_all, path = source, source.get("sample_ids"), source.get("source", task.meta["source"])
        idx = torch.arange(min(n, len(source["target"]))) if idx is None else torch.as_tensor(idx)
    blob = _load_blob(path)
    was = model.training
    model.eval()
    pred, tgt = predict(model, task, idx, batch_size, device, source=source)
    model.train(was)
    eps = float(blob["meta"]["config"].get("eps", 0.0))
    rows, trows = [], []
    for j, i in enumerate(idx.tolist()):
        K = src["K"][i].to("cpu")
        x = {k: task.x_stats[k].to("cpu").denormalize(v.cpu()).double().numpy() for k, v in src["inputs"][i].items()}
        yp = task.y_stats.to("cpu").denormalize(pred[j].cpu()).double().numpy()[:, 0]
        sid = int(ids_all[i]) if ids_all is not None else int(i)
        rows.append(_residuals(kind, K, x, yp, blob, sid, eps))
        if targets:
            yt = task.y_stats.to("cpu").denormalize(tgt[j].cpu()).double().numpy()[:, 0]
            trows.append(_residuals(kind, K, x, yt, blob, sid, eps))
    out = {k: dict(mean=float(np.mean([r[k] for r in rows])), max=float(np.max([r[k] for r in rows])))
           for k in rows[0]}
    for k in (trows[0] if trows else {}):
        out["target_" + k] = dict(mean=float(np.mean([r[k] for r in trows])), max=float(np.max([r[k] for r in trows])))
    out["N"] = len(rows)
    out["definition"] = _STRUCTURE_DOC[kind]
    return out
