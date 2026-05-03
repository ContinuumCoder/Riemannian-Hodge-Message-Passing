"""
Numerical symmetry tests across all 12 baselines + GSHMP.

Protocol (uniform across all models):
  - Test fixture: 2D Delaunay on 6 random points (seed=0) -> n0=6, n1=11, n2=6,
    plus a regular 4x4 grid Delaunay fixture (n0=16, n1=33, n2=18).
  - C = f_in = hidden = 32, n_layers = 2.
  - All models built at random init with seed=42, eval mode (no training).
  - 5 trials per (model, symmetry) using independent random transforms.

Two test scopes (matching the original GSHMP protocol):

  1. WHOLE-NETWORK (E(n), Z_2 orient): apply transform to K, run full forward,
     compare outputs.  Tests end-to-end behavior under geometry/topology
     transformations.

  2. MP-BODY  (O(C), SO(C), S_C, (Z_2)^C):  Bypass each model's encoder /
     readout. Apply transform T to a random hidden state h0 in R^{n0 x C},
     push through the model's first MP layer (or MP body for models without
     a single MP module), un-apply T, compare against the un-transformed run.
     err = || T^{-1} . body(T . h0) - body(h0) ||_F / || body(h0) ||_F
     This matches measure_diagnostics.py's protocol for GSHMP and tests the
     equivariance of message passing in cochain space (which is the design
     property the paper claims).

Symmetry groups tested:
  Spatial E(n):
    T(n)   -- random translation v ~ N(0, 0.1*I_2) on K.pos
    SO(n)  -- random 2D rotation
    O(n)   -- random 2D reflection (det = -1)

  Fiber O(C) (acts on input feature channels; whole-network invariance):
    O(C)        -- random Q from QR(randn(C,C))
    SO(C)       -- as O(C) with det forced +1
    S_C         -- random channel permutation
    (Z_2)^C     -- random channel sign vector

  Topological:
    Z_2 edge orientation -- flip one edge (i,j)->(j,i), rebuild d0/d1, compare
    d^2 = 0              -- algebraic check on K.d1 @ K.d0 = 0 (model uses K.d_k internally)
    U(1) flat            -- by design (architectural property; see paper Section 3)

Outputs:
  ablation/results/baseline_symmetries.json
"""
import os
import sys
import json
import math
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import Delaunay

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS)
sys.path.insert(0, os.path.join(ROOT, "src"))

from gauge_hodge_mp.cell_complex import CellComplex
from gauge_hodge_mp.network import GaugeHodgeNetwork
from baselines_graph import GCNBaseline, GATBaseline, SchNetBaseline, EGNNBaseline
from baselines_topo import MPSNBaseline, SCCNNBaseline
from baselines_advanced import (GaugeEquivCNNBaseline, CWNetBaseline,
                                CliffordNetBaseline, HermesBaseline)
from baselines_operator import FNOWrapper, DeepONetWrapper

warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
C = 32
N_LAYERS = 2
N_TRIALS = 5
INIT_SEED = 42
THRESHOLD_EXACT = 1e-5
THRESHOLD_NUMERICAL = 1e-3


# ---------------------------------------------------------------------------
# Test fixture
# ---------------------------------------------------------------------------

def make_fixture(seed=0, n_points=6):
    """Irregular 2D Delaunay test complex (n=6 seed=0 -> n0=6, n1=11, n2=6)."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, 1.0, size=(n_points, 2))
    tri = Delaunay(pts)
    K = CellComplex.from_triangulation(
        torch.tensor(pts, dtype=torch.float32),
        torch.tensor(tri.simplices, dtype=torch.long),
        device=DEVICE,
    )
    return K


def make_regular_fixture(side=4):
    """Regular grid fixture: side x side lattice, Delaunay triangulated.

    side=4 -> n0=16, n1=33, n2=18. Lets FNO run with grid_size=(side, side)
    and provides a paired benchmark to the irregular fixture.
    """
    g = np.arange(side, dtype=np.float32) / max(side - 1, 1)
    xx, yy = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([xx.flatten(), yy.flatten()], axis=-1)
    tri = Delaunay(pts)
    K = CellComplex.from_triangulation(
        torch.tensor(pts, dtype=torch.float32),
        torch.tensor(tri.simplices, dtype=torch.long),
        device=DEVICE,
    )
    return K, side


def make_input(K, f_in=C, seed=INIT_SEED):
    """Random node-feature input matching the model's expected shape."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.randn(K.n0, f_in, generator=g, device=DEVICE)


# ---------------------------------------------------------------------------
# Geometric transformations on K
# ---------------------------------------------------------------------------

def shifted_complex(K, v):
    """K with translated vertex positions."""
    new_pos = K.pos + v.to(K.pos)
    return CellComplex.from_triangulation(new_pos, K.faces, device=DEVICE)


def transformed_complex(K, M):
    """K with vertex positions transformed by an affine map (pos @ M.T)."""
    new_pos = K.pos @ M.to(K.pos).T
    return CellComplex.from_triangulation(new_pos, K.faces, device=DEVICE)


def random_rotation(seed):
    """Random 2D rotation matrix."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    theta = torch.rand(1, generator=g).item() * 2 * math.pi
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor([[c, -s], [s, c]], dtype=torch.float32)


def random_reflection(seed):
    """Random 2D reflection (Householder along a random axis)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(2, generator=g)
    a = a / a.norm()
    R = torch.eye(2) - 2 * torch.outer(a, a)  # det = -1
    return R.to(torch.float32)


def random_orthogonal(C_dim, seed, det_sign=None):
    """Random Q in O(C); if det_sign is +1 returns SO(C)."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    A = torch.randn(C_dim, C_dim, generator=g, device=DEVICE)
    Q, _ = torch.linalg.qr(A)
    if det_sign is not None:
        d = torch.det(Q)
        if (d.item() > 0) != (det_sign > 0):
            Q[:, 0] = -Q[:, 0]
    return Q


def flip_one_edge(K, edge_idx):
    """Surgically reverse a single edge's orientation.

    Convention:  edge e originally (i -> j) becomes (j -> i).
    Coboundary signs flip accordingly: d0[e,:] negates, d1[:,e] negates.
    The geometry is unchanged; only the edge's orientation convention does.

    A model is Z_2 edge-orientation invariant iff its per-edge output flips
    sign on edge e and is unchanged elsewhere.
    """
    new_edges = K.edges.clone()
    new_edges[edge_idx] = new_edges[edge_idx].flip(0)

    d0_dense = K.d0.to_dense().clone()
    d1_dense = K.d1.to_dense().clone()
    d0_dense[edge_idx, :] = -d0_dense[edge_idx, :]
    d1_dense[:, edge_idx] = -d1_dense[:, edge_idx]

    new_d0 = d0_dense.to_sparse_coo().coalesce()
    new_d1 = d1_dense.to_sparse_coo().coalesce()

    return CellComplex(
        pos=K.pos, edges=new_edges, faces=K.faces,
        d0=new_d0, d1=new_d1,
        n0=K.n0, n1=K.n1, n2=K.n2,
    )


# ---------------------------------------------------------------------------
# Forward + relative-error helpers
# ---------------------------------------------------------------------------

def safe_forward(model, X, K):
    """Run model.forward(X, K) handling both per-sample and batched conventions.

    Returns a 2D tensor (cells, dim). Detaches and moves to CPU at the boundary.
    """
    with torch.no_grad():
        try:
            out = model(X, K)
        except TypeError:
            out = model(X)
    if out.dim() == 3:
        out = out.squeeze(0)
    return out


def rel_err(a, b):
    """|| a - b ||_F / || b ||_F (clamped denominator)."""
    return (a - b).norm().item() / max(b.norm().item(), 1e-12)


# ---------------------------------------------------------------------------
# Per-symmetry test routines
# ---------------------------------------------------------------------------

def test_translation(model, K, X, n_trials=N_TRIALS):
    """T(n): translation invariance (pos += v)."""
    errs = []
    y0 = safe_forward(model, X, K)
    for t in range(n_trials):
        g = torch.Generator(device="cpu").manual_seed(1000 + t)
        v = 0.1 * torch.randn(2, generator=g)
        K_t = shifted_complex(K, v)
        y1 = safe_forward(model, X, K_t)
        errs.append(rel_err(y1, y0))
    return errs


def test_rotation(model, K, X, n_trials=N_TRIALS):
    """SO(n): random 2D rotation."""
    errs = []
    y0 = safe_forward(model, X, K)
    for t in range(n_trials):
        R = random_rotation(2000 + t)
        K_r = transformed_complex(K, R)
        y1 = safe_forward(model, X, K_r)
        errs.append(rel_err(y1, y0))
    return errs


def test_reflection(model, K, X, n_trials=N_TRIALS):
    """O(n): random 2D reflection."""
    errs = []
    y0 = safe_forward(model, X, K)
    for t in range(n_trials):
        Rf = random_reflection(3000 + t)
        K_r = transformed_complex(K, Rf)
        y1 = safe_forward(model, X, K_r)
        errs.append(rel_err(y1, y0))
    return errs


def _mp_body_test(model_name, model, K, h0, transform_fn, inverse_fn):
    """Run one MP-body equivariance trial.

    err = || inverse_fn( body(transform_fn(h0)) ) - body(h0) ||_F / || body(h0) ||_F
    """
    body = MP_BODY[model_name]
    with torch.no_grad():
        y_base = body(model, h0, K)
        y_rot = body(model, transform_fn(h0), K)
        y_unrot = inverse_fn(y_rot)
    return rel_err(y_unrot, y_base)


def test_OC_mp(model_name, model, K, h0, n_trials=N_TRIALS, det_sign=None):
    """O(C) / SO(C) equivariance of MP body, in lifted-feature space."""
    errs = []
    for t in range(n_trials):
        Q = random_orthogonal(h0.size(-1), 4000 + t, det_sign=det_sign)
        errs.append(_mp_body_test(
            model_name, model, K, h0,
            transform_fn=lambda x, Q=Q: x @ Q,
            inverse_fn=lambda y, Q=Q: y @ Q.T,
        ))
    return errs


def test_perm_mp(model_name, model, K, h0, n_trials=N_TRIALS):
    """S_C equivariance of MP body."""
    errs = []
    for t in range(n_trials):
        g = torch.Generator(device="cpu").manual_seed(5000 + t)
        perm = torch.randperm(h0.size(-1), generator=g).to(DEVICE)
        inv_perm = torch.argsort(perm)
        errs.append(_mp_body_test(
            model_name, model, K, h0,
            transform_fn=lambda x, p=perm: x[:, p],
            inverse_fn=lambda y, ip=inv_perm: y[:, ip],
        ))
    return errs


def test_sign_mp(model_name, model, K, h0, n_trials=N_TRIALS):
    """(Z_2)^C equivariance of MP body."""
    errs = []
    for t in range(n_trials):
        g = torch.Generator(device="cpu").manual_seed(6000 + t)
        s = (torch.randint(0, 2, (h0.size(-1),), generator=g) * 2 - 1).float().to(DEVICE)
        errs.append(_mp_body_test(
            model_name, model, K, h0,
            transform_fn=lambda x, s=s: x * s,
            inverse_fn=lambda y, s=s: y * s,
        ))
    return errs


def test_orientation_mp(model_name, model, K, h0, n_trials=N_TRIALS):
    """Z_2 edge orientation invariance of MP body (0-cochain output).

    Cochain-aware MP bodies see d_k an even number of times when producing a
    0-cochain output, so flipping one edge's orientation should leave the
    0-cochain output unchanged. The (sign-broken) readout is not part of
    this test.
    """
    body = MP_BODY[model_name]
    errs = []
    with torch.no_grad():
        y0 = body(model, h0, K)
    for t in range(n_trials):
        g = torch.Generator(device="cpu").manual_seed(7000 + t)
        edge_idx = torch.randint(0, K.n1, (1,), generator=g).item()
        K_f = flip_one_edge(K, edge_idx)
        with torch.no_grad():
            y1 = body(model, h0, K_f)
        errs.append(rel_err(y1, y0))
    return errs


def test_orientation(model, K, X, n_trials=N_TRIALS):
    """Z_2 edge orientation: flip one edge of K (with d0/d1 sign update),
    test whether the model's output transforms as a cochain.

    For edge-task output:  expect y'[e] = -y[e], y'[other] = y[other].
    For node-task output:  expect y' = y (orientation flip doesn't move node-task
                            predictions if the model is cochain-aware).
    """
    y0 = safe_forward(model, X, K)
    out_n = y0.size(0)
    is_edge_output = (out_n == K.n1)
    is_node_output = (out_n == K.n0)
    if not (is_edge_output or is_node_output):
        return [float("nan")]
    errs = []
    for t in range(n_trials):
        g = torch.Generator(device="cpu").manual_seed(7000 + t)
        edge_idx = torch.randint(0, K.n1, (1,), generator=g).item()
        K_f = flip_one_edge(K, edge_idx)
        y1 = safe_forward(model, X, K_f)
        if is_edge_output:
            y_expected = y0.clone()
            y_expected[edge_idx] = -y_expected[edge_idx]
        else:
            y_expected = y0
        errs.append(rel_err(y1, y_expected))
    return errs


# ---------------------------------------------------------------------------
# MP-body adapters (bypass encoder/readout, run only the message passing)
#
# Each adapter takes a hidden state h0 in R^{n0 x C} and returns the post-MP
# hidden state in R^{n0 x C}.  The encoder Linear (which mixes channels) is
# bypassed so we test the architectural O(C)-class equivariance of the MP
# layers themselves.
# ---------------------------------------------------------------------------

import torch.nn.functional as F  # noqa: E402
from gauge_hodge_mp.utils import spmm  # noqa: E402


def _mp_gcn(model, h0, K):
    A_hat = model._build_norm_adj(K)
    h = h0
    # layers = [Linear(C, C)] * n_layers; first layer is "encoder", rest are MP
    for i, lin in enumerate(model.layers[1:], start=1):
        h = A_hat @ h
        h = lin(h)
        if i < len(model.layers) - 1:
            h = F.silu(h)
    return h


def _mp_gat(model, h0, K):
    src, dst = K.edges[:, 0], K.edges[:, 1]
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    h = h0
    for i, (gat, norm) in enumerate(zip(model.layers[1:], model.norms[1:]), start=1):
        h_new = gat(h, edge_index, K.n0)
        if i < len(model.layers) - 1:
            h_new = F.silu(norm(h_new))
        h = h + h_new if h.size(-1) == h_new.size(-1) else h_new
    return h


def _mp_schnet(model, h0, K):
    src_fwd, dst_fwd = K.edges[:, 0], K.edges[:, 1]
    src_idx = torch.cat([src_fwd, dst_fwd])
    dst_idx = torch.cat([dst_fwd, src_fwd])
    diff = K.pos[dst_idx] - K.pos[src_idx]
    rbf = model.rbf(diff.norm(dim=-1))
    h = h0
    for interaction in model.interactions:
        h = interaction(h, rbf, src_idx, dst_idx, K.n0)
    return h


def _mp_egnn(model, h0, K):
    src_fwd, dst_fwd = K.edges[:, 0], K.edges[:, 1]
    src_idx = torch.cat([src_fwd, dst_fwd])
    dst_idx = torch.cat([dst_fwd, src_fwd])
    h = h0
    for layer in model.layers:
        h = layer(h, K.pos, src_idx, dst_idx, K.n0)
    return h


def _mp_mpsn(model, h0, K):
    """MPSN's 0-cochain MP body (MPSN treats 0/1/2-cochains independently)."""
    A_00, _, _, _ = model._build_adjacency(K)
    h = h0
    for i in range(model.n_layers):
        h_new = model.layers_0[i](spmm(A_00, h))
        h = F.silu(model.norms_0[i](h_new + h))
    return h


def _mp_sccnn(model, h0, K):
    L0, _, _ = model._build_laplacians(K)
    h = h0
    for i in range(model.n_layers):
        h = F.silu(model.mix_0[i](model._poly_filter_single(h, L0, model.w_0[i])))
    return h


def _mp_gauge_eq(model, h0, K):
    pos = K.pos[:, :2] if K.pos.size(1) >= 2 else K.pos
    edges = K.edges
    edge_dir, src, dst = model._compute_transport(pos, edges)
    h = h0
    for i in range(model.n_layers):
        x_src = h[src]
        scale = model.transport_scale[i]
        if edge_dir.size(1) >= 2:
            cos_a = edge_dir[:, 0:1]
            sin_a = edge_dir[:, 1:2]
            x_e, x_o = x_src[:, 0::2], x_src[:, 1::2]
            mc = min(x_e.size(1), x_o.size(1))
            x_t = torch.zeros_like(x_src)
            x_t[:, 0::2][:, :mc] = cos_a * x_e[:, :mc] - scale * sin_a * x_o[:, :mc]
            x_t[:, 1::2][:, :mc] = scale * sin_a * x_e[:, :mc] + cos_a * x_o[:, :mc]
        else:
            x_t = x_src
        msg = model.W_msg[i](x_t)
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg); agg.index_add_(0, src, msg)
        cnt = torch.zeros(h.size(0), 1, device=h.device)
        cnt.index_add_(0, dst, torch.ones(len(dst), 1, device=h.device))
        cnt.index_add_(0, src, torch.ones(len(src), 1, device=h.device))
        h = F.silu(model.W_self[i](h) + agg / cnt.clamp(min=1))
    return h


def _mp_gem_cnn(model, h0, K):
    pos = K.pos[:, :2] if K.pos.size(1) >= 2 else K.pos
    src, dst = K.edges[:, 0], K.edges[:, 1]
    ev = pos[dst] - pos[src]
    el = ev.norm(dim=-1, keepdim=True).clamp(1e-8)
    edge_dir = ev / el
    theta = torch.atan2(edge_dir[:, 1] if edge_dir.size(1) >= 2 else torch.zeros_like(edge_dir[:, 0]),
                        edge_dir[:, 0])
    basis = model._compute_basis(theta, h0.device)
    cos_a = edge_dir[:, 0:1] if edge_dir.size(1) >= 2 else torch.ones(len(src), 1, device=h0.device)
    sin_a = edge_dir[:, 1:2] if edge_dir.size(1) >= 2 else torch.zeros(len(src), 1, device=h0.device)
    h = h0
    cnt = torch.zeros(h.size(0), 1, device=h.device)
    cnt.index_add_(0, dst, torch.ones(len(dst), 1, device=h.device))
    cnt = cnt.clamp(min=1)
    for i in range(model.n_layers):
        x_self = model.K_self[i](h)
        x_t = model._parallel_transport(h[src], cos_a, sin_a)
        W = model.K_neigh_weights[i]
        msg = torch.einsum('eb,bhd,ed->eh', basis, W, x_t)
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, msg)
        h = x_self + agg / cnt + model.bias[i]
    return h


def _mp_cwnet(model, h0, K):
    """CWNet 0-cochain MP body. x1, x2 are derived from h0 via K.d_k (no encoder)."""
    d0, d1 = K.d0, K.d1
    h = h0
    x1 = spmm(d0, h)
    x2 = spmm(d1, x1)
    for i in range(model.n_layers):
        m_0 = model.msg_0_from_1[i](spmm(d0.t(), x1))
        m_1_b = model.msg_1_from_0[i](spmm(d0, h))
        m_1_c = model.msg_1_from_2[i](spmm(d1.t(), x2))
        m_2 = model.msg_2_from_1[i](spmm(d1, x1))
        h = F.silu(model.update_0[i](h + m_0))
        x1 = F.silu(model.update_1[i](x1 + m_1_b + m_1_c))
        x2 = F.silu(model.update_2[i](x2 + m_2))
    return h


def _mp_clifford(model, h0, K):
    d0 = K.d0
    h = h0
    x1 = spmm(d0, h)
    for i in range(model.n_layers):
        m0 = spmm(d0.t(), x1)
        m1 = spmm(d0, h)
        h = model._clifford_layer(h + m0, model.W_0_a[i], model.W_0_b[i], model.norms_0[i])
        x1 = model._clifford_layer(x1 + m1, model.W_1_a[i], model.W_1_b[i], model.norms_1[i])
    return h


def _mp_fno(model, h0, K, grid_side):
    """FNO MP body: skip fc0 (encoder), run spectral conv stack, skip fc1/fc2 (decoder)."""
    gx = gy = grid_side
    # h0 is (n0, C); reshape into (1, C, gx, gy)
    x = h0.view(gx, gy, -1).permute(2, 0, 1).unsqueeze(0)
    for conv, w, norm in zip(model.fno.convs, model.fno.ws, model.fno.norms):
        x1 = conv(x)
        x2 = w(x)
        x = norm(x1 + x2)
        x = F.gelu(x)
    return x.squeeze(0).permute(1, 2, 0).reshape(gx * gy, -1)


def _mp_gshmp(model, h0, K):
    """GSHMP first 0-cochain MP layer (matches measure_diagnostics.py protocol).

    Uses single-sample forward (h0 is (n0, C), not batched).
    """
    # Prime any lazily-built parameters
    if model.learned_d and (model._learned_d0_vals is None or model._learned_d1_vals is None):
        with torch.no_grad():
            X_dummy = torch.randn(1, K.n0, model.lifting.node_mlp[0].in_features - 2,
                                  device=h0.device)
            _ = model.forward_batch(X_dummy, K)
    layer = model.layers[0]["mp0"]
    adj_00 = K.get_adjacency(0, 0)
    d0, _ = model._maybe_learned_d(K)
    x1 = spmm(d0, h0)  # synthesize an upper-cochain so the cross-dim path runs
    return layer.forward(
        h0, d_lower=None, d_upper=d0, adj=adj_00,
        x_lower=None, x_upper=x1,
    )


MP_BODY = {
    "GCN":         _mp_gcn,
    "GAT":         _mp_gat,
    "SchNet":      _mp_schnet,
    "EGNN":        _mp_egnn,
    "MPSN":        _mp_mpsn,
    "SCCNN":       _mp_sccnn,
    "GaugeEqCNN":  _mp_gauge_eq,
    "GEM-CNN":     _mp_gem_cnn,
    "CWNet":       _mp_cwnet,
    "Cliff-SMPN":  _mp_clifford,
    # FNO handled separately (needs grid_side)
    # DeepONet has no MP body (function-space outer product) -> N/A
    "GSHMP":       _mp_gshmp,
}


# ---------------------------------------------------------------------------
# Architectural categorical checks
# ---------------------------------------------------------------------------

USES_DK = {
    "GCN": False, "GAT": False, "SchNet": False, "EGNN": False,
    "MPSN": False,                          # uses |d_k| but not signed d_k -> still topology-derived
    "SCCNN": True,                          # builds Hodge Laplacian from d0, d1
    "GaugeEqCNN": False,
    "GEM-CNN": False,
    "CWNet": True,
    "Cliff-SMPN": True,
    "FNO": False, "DeepONet": False,
    "GSHMP": True,
}

# U(1) flat is a property of cochain-level processing
COCHAIN_INPUT = {
    "GCN": False, "GAT": False, "SchNet": False, "EGNN": False,
    "MPSN": True, "SCCNN": True,
    "GaugeEqCNN": False, "GEM-CNN": False,
    "CWNet": True, "Cliff-SMPN": True,
    "FNO": False, "DeepONet": False,
    "GSHMP": True,
}


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(name, K):
    """Construct each model with f_in=C, hidden=C, n_layers=N_LAYERS, edge-task scalar output."""
    torch.manual_seed(INIT_SEED)
    if name == "GCN":
        m = GCNBaseline(f_in=C, hidden=C, n_layers=N_LAYERS, out_dim=1, task="edge")
    elif name == "GAT":
        m = GATBaseline(f_in=C, hidden=C, n_layers=N_LAYERS, out_dim=1, task="edge")
    elif name == "SchNet":
        m = SchNetBaseline(f_in=C, hidden=C, n_layers=N_LAYERS, out_dim=1, task="edge")
    elif name == "EGNN":
        m = EGNNBaseline(f_in=C, hidden=C, n_layers=N_LAYERS, out_dim=1, task="edge")
    elif name == "MPSN":
        m = MPSNBaseline(f_in=C, hidden=C, n_layers=N_LAYERS)
    elif name == "SCCNN":
        m = SCCNNBaseline(f_in=C, hidden=C, n_layers=N_LAYERS)
    elif name == "GaugeEqCNN":
        m = GaugeEquivCNNBaseline(f_in=C, hidden=C, n_layers=N_LAYERS, out_dim=1, task="node")
    elif name == "GEM-CNN":
        m = HermesBaseline(f_in=C, hidden=C, n_layers=N_LAYERS, out_dim=1, task="node")
    elif name == "CWNet":
        m = CWNetBaseline(f_in=C, hidden=C, n_layers=N_LAYERS, out_dim=1, task="node")
    elif name == "Cliff-SMPN":
        m = CliffordNetBaseline(f_in=C, hidden=C, n_layers=N_LAYERS, out_dim=1, task="edge")
    elif name == "FNO":
        m = FNOWrapper(f_in=C, out_dim=1, width=C, modes=2, n_layers=N_LAYERS)
    elif name == "DeepONet":
        m = DeepONetWrapper(f_in=C, out_dim=1, branch_width=C, trunk_width=C, n_basis=C)
    elif name == "GSHMP":
        m = GaugeHodgeNetwork(
            f_in=C, C=C, n_layers=N_LAYERS,
            n0=K.n0, n1=K.n1, n2=K.n2,
            task="edge_scalar", spatial_dim=2, out_dim=1, mp_hidden=16,
            metric_type="diagonal", metric_rank=0,
        )
    else:
        raise ValueError(name)
    return m.to(DEVICE).eval()


MODEL_NAMES = [
    "GCN", "GAT", "SchNet", "EGNN",
    "MPSN", "SCCNN",
    "GaugeEqCNN", "GEM-CNN", "CWNet", "Cliff-SMPN",
    "FNO", "DeepONet",
    "GSHMP",
]


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

# Whole-network groups: take (model, K, X)
GROUPS_WHOLE = [
    ("T(n)",       test_translation, None),
    ("SO(n)",      test_rotation,    None),
    ("O(n)",       test_reflection,  None),
    ("Z_2 orient", test_orientation, None),
]

# MP-body groups: take (model_name, model, K, h0)
GROUPS_MP = [
    ("O(C)",            test_OC_mp,          {"det_sign": None}),
    ("SO(C)",           test_OC_mp,          {"det_sign": +1}),
    ("S_C",             test_perm_mp,        None),
    ("(Z_2)^C",         test_sign_mp,        None),
    ("Z_2 orient (MP)", test_orientation_mp, None),
]


def run_one_model(name, K, fixture_kind="irregular", grid_side=None):
    """Run all tests for a single model. Returns dict {group: {mean, max, n_trials}}."""
    print(f"\n=== {name} ({fixture_kind}) ===")
    out = {}
    all_groups = [g[0] for g in GROUPS_WHOLE] + [g[0] for g in GROUPS_MP]

    # FNO requires a regular grid; on irregular meshes it is N/A by design.
    if name == "FNO" and fixture_kind != "regular":
        for grp in all_groups:
            out[grp] = {"status": "N/A",
                        "reason": "regular-grid spectral method; not applicable to irregular meshes"}
        out["d^2=0"] = {"status": "N/A", "reason": "no internal coboundary operator"}
        out["U(1) flat"] = {"status": "N/A", "reason": "no cochain representation"}
        return out

    try:
        model = build_model(name, K)
        if name == "FNO" and grid_side is not None:
            model.grid_size = (grid_side, grid_side)
    except Exception as e:
        print(f"  build FAILED: {e}")
        for grp in all_groups:
            out[grp] = {"status": "BUILD_FAIL", "reason": str(e)}
        return out

    f_in = C
    X = make_input(K, f_in=f_in)
    # Probe forward once to detect any output issues
    try:
        _ = safe_forward(model, X, K)
    except Exception as e:
        print(f"  forward probe FAILED: {e}")
        for grp in all_groups:
            out[grp] = {"status": "FWD_FAIL", "reason": str(e)}
        out["d^2=0"] = {"status": "exact" if USES_DK[name] else "N/A"}
        out["U(1) flat"] = {"status": "by-design" if COCHAIN_INPUT[name] else "N/A"}
        return out

    # Whole-network tests (E(n) + Z_2 orient)
    for grp, fn, kwargs in GROUPS_WHOLE:
        try:
            errs = fn(model, K, X, **(kwargs or {}))
            errs_clean = [e for e in errs if not math.isnan(e)]
            if not errs_clean:
                out[grp] = {"status": "N/A", "reason": "output shape incompatible"}
                continue
            out[grp] = {"mean": float(np.mean(errs_clean)),
                        "max":  float(np.max(errs_clean)),
                        "n_trials": len(errs_clean), "status": "ok"}
            print(f"  {grp:14s} mean={out[grp]['mean']:.3e}  max={out[grp]['max']:.3e}")
        except Exception as e:
            print(f"  {grp:14s} ERR: {e}")
            out[grp] = {"status": "RUN_FAIL", "reason": str(e)}

    # MP-body tests (channel symmetries) — use mp_body adapter
    if name == "DeepONet":
        # No MP layer in the message-passing sense.
        for grp, _fn, _kw in GROUPS_MP:
            out[grp] = {"status": "N/A",
                        "reason": "no message-passing body; branch+trunk outer product"}
    elif name == "FNO":
        # Special-case: needs grid_side
        if grid_side is None:
            for grp, _fn, _kw in GROUPS_MP:
                out[grp] = {"status": "N/A", "reason": "regular grid required"}
        else:
            g_h = torch.Generator(device=DEVICE).manual_seed(INIT_SEED + 1)
            h0 = torch.randn(K.n0, C, generator=g_h, device=DEVICE)
            for grp, fn, kwargs in GROUPS_MP:
                try:
                    body_with_side = lambda m, h, K, gs=grid_side: _mp_fno(m, h, K, gs)
                    MP_BODY["FNO"] = body_with_side  # patch in
                    errs = fn("FNO", model, K, h0, **(kwargs or {}))
                    out[grp] = {"mean": float(np.mean(errs)),
                                "max":  float(np.max(errs)),
                                "n_trials": len(errs), "status": "ok"}
                    print(f"  {grp:14s} mean={out[grp]['mean']:.3e}  max={out[grp]['max']:.3e}")
                except Exception as e:
                    print(f"  {grp:14s} ERR: {e}")
                    out[grp] = {"status": "RUN_FAIL", "reason": str(e)}
    else:
        g_h = torch.Generator(device=DEVICE).manual_seed(INIT_SEED + 1)
        h0 = torch.randn(K.n0, C, generator=g_h, device=DEVICE)
        for grp, fn, kwargs in GROUPS_MP:
            try:
                errs = fn(name, model, K, h0, **(kwargs or {}))
                out[grp] = {"mean": float(np.mean(errs)),
                            "max":  float(np.max(errs)),
                            "n_trials": len(errs), "status": "ok"}
                print(f"  {grp:14s} mean={out[grp]['mean']:.3e}  max={out[grp]['max']:.3e}")
            except Exception as e:
                print(f"  {grp:14s} ERR: {e}")
                out[grp] = {"status": "RUN_FAIL", "reason": str(e)}

    # Categorical
    out["d^2=0"] = {"status": "exact" if USES_DK[name] else "N/A",
                    "reason": ("inherits K.d1 @ K.d0 = 0" if USES_DK[name]
                               else "model does not use coboundary operators")}
    out["U(1) flat"] = {"status": "by-design" if COCHAIN_INPUT[name] else "N/A",
                        "reason": ("processes 1-cochain features" if COCHAIN_INPUT[name]
                                   else "no cochain representation")}
    return out


def main():
    # Two fixtures, run separately, both saved.
    K_irr = make_fixture(seed=0, n_points=6)
    K_reg, side = make_regular_fixture(side=4)
    print(f"Irregular fixture: n0={K_irr.n0}, n1={K_irr.n1}, n2={K_irr.n2}")
    print(f"Regular   fixture: n0={K_reg.n0}, n1={K_reg.n1}, n2={K_reg.n2} (grid {side}x{side})")

    summary = {
        "config": {"C": C, "n_layers": N_LAYERS, "n_trials": N_TRIALS,
                   "init_seed": INIT_SEED,
                   "thresholds": {"exact": THRESHOLD_EXACT,
                                  "numerical_pass": THRESHOLD_NUMERICAL}},
        "fixtures": {
            "irregular": {"n0": K_irr.n0, "n1": K_irr.n1, "n2": K_irr.n2,
                          "n_points": 6, "seed": 0,
                          "kind": "Delaunay on uniform random points"},
            "regular":   {"n0": K_reg.n0, "n1": K_reg.n1, "n2": K_reg.n2,
                          "side": side,
                          "kind": f"Delaunay on a {side}x{side} regular lattice"},
        },
        "results": {"irregular": {}, "regular": {}},
    }

    for name in MODEL_NAMES:
        summary["results"]["irregular"][name] = run_one_model(
            name, K_irr, fixture_kind="irregular")
    for name in MODEL_NAMES:
        summary["results"]["regular"][name] = run_one_model(
            name, K_reg, fixture_kind="regular", grid_side=side)

    out_path = os.path.join(THIS, "results", "baseline_symmetries.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
