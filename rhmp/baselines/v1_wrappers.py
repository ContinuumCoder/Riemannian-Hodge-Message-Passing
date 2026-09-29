"""Uniform v2 wrapper around the v1 baselines (``rhmp.baselines.v1.baselines_*``) and the v1 model (``ours_v1``).

Every baseline is a :class:`BaselineModel` = ``NodeInputEncoder -> core -> head``:

* the **core** holds the v1 module (``self.v1``, same parameters, same initialisation as v1) and re-executes its
  forward pass in the v2 cochain layout ``(n_k, B, C)``: gathers are ``index_select`` along dim 0, scatters
  ``index_add_``, sparse products ``rhmp.ops.spmm`` on cached CSR operators (``V1ComplexAdapter.op``).  One pass
  serves the whole batch of a shared mesh (no per-sample Python loop, unlike v1's ``BaselineBase.forward_batch``)
  and a block-diagonal batch of different meshes is one big graph, so predictions never depend on the other samples
  of the batch.  With ``v1_exact=True`` (and raw geometry) the cores reproduce v1's outputs to float round-off
  (``tests/test_baselines.py::test_core_fidelity``).
* the **head** maps the hidden features to the task's target cells (``rhmp.baselines.adapters``): node targets
  use the v1 node head, edge / face / tet targets get the heads described in ``docs/BASELINES.md``.

Deliberate deviations from v1 (all switchable, defaults = the stronger/fairer variant):

* ``geometry='normalized'``: SchNet / EGNN distances are divided by the mesh's mean edge length (v1 used raw
  coordinates of unit-square meshes: SchNet's fixed RBF grid on ``[0, 5]`` then sees all distances in its first
  bin, EGNN's ``|x_i - x_j|^2 ~ 1e-3``).  ``'raw'`` reproduces v1.
* GaugeEquivCNN: v1 added the message computed from ``x_src`` to *both* endpoints (the source received its own
  transported feature); the default sends ``T(x_src)`` to ``dst`` and ``T'(x_dst)`` (reverse direction) to ``src``.
* GEM-CNN: v1 aggregated only along canonical edges ``src -> dst`` (``src < dst``: information flowed from lower
  to higher vertex ids only) and had no nonlinearity (a linear model); the default aggregates over both directions
  (reverse direction = angle + pi) and inserts SiLU between convolutions, as the original GEM-CNN does.
* SCCNN: :func:`make_v1_baseline` keeps v1's raw combinatorial Laplacians by default (activations grow by ~1e2 per
  layer, as in v1); the registry uses them only under the v1 edge protocol and otherwise sets ``normalize=True``
  (Laplacians divided per mesh by their Gershgorin bound), which must be combined with ``filter_init='identity'``:
  with v1's ``N(0, 0.1)`` coefficient initialisation the identity term shrinks the signal ~10x per layer and the
  network collapses to a constant (bias-only) predictor (T6: prediction spread 1e-6, gradients 1e-10,
  val R2 -0.0004 after 100 epochs).
* heads: v1 trained MPSN / SCCNN / Clifford-SMPN with a 1-channel edge readout on edge-averaged node targets
  ``0.5 (y_src + y_dst)`` and scored them in edge space (first target component only; v1 evaluation script
  ``compute_all_metrics.py``).  On the v1 paper tasks this *v1 edge protocol* is the default
  (``head='v1_edge'``, the task target is transformed by ``rhmp.baselines.registry.prepare_task``); elsewhere, or
  with ``head='node'``, they predict the task target with the multi-degree ``ComplexHead``.
* EGNN on legacy T7: v1 meant to strip the directional columns (``avg*dx, avg*dy``) but selected columns
  ``0, 3, 6`` of ``[avg(3), avg*dx(3), avg*dy(3)]``; the registry selects the three ``avg`` columns.
"""
from __future__ import annotations

import contextlib
import warnings
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from rhmp.ops import segment_mean, spmm

from .adapters import (ComplexHead, EdgeInputEncoder, NodeInputEncoder, TaskIO, V1ComplexAdapter, adapter_for,
                       mlp, node_model_head)

__all__ = ["import_v1", "BaselineModel", "NODE_CORES", "COMPLEX_CORES", "make_v1_baseline", "make_ours_v1",
           "GCNCore", "GATCore", "SchNetCore", "EGNNCore", "GaugeCNNCore", "GEMCore", "MPSNCore", "SCCNNCore",
           "CWNetCore", "CliffordCore", "DeepONetCore", "FNOCore", "V1OursCore"]

DIR_EPS = 1e-8          # v1 gauge/GEM: edge direction = edge / |edge|.clamp(1e-8)
warnings.filterwarnings("ignore", message=r"index_reduce\(\) is in beta")


def import_v1():
    """Return the vendored v1 modules (``rhmp.baselines.v1``) as a namespace dict."""
    from .v1 import baselines_advanced as adv
    from .v1 import baselines_graph as graph
    from .v1 import baselines_operator as op
    from .v1 import baselines_topo as topo
    from .v1.gauge_hodge_mp import network
    return {"graph": graph, "topo": topo, "adv": adv, "op": op, "network": network}


def _gather(x: Tensor, idx: Tensor) -> Tensor:
    return x.index_select(0, idx)


def _scatter(src: Tensor, idx: Tensor, n: int) -> Tensor:
    return src.new_zeros((n,) + tuple(src.shape[1:])).index_add_(0, idx, src)


def _interleave(even: Tensor, odd: Tensor, H: int) -> Tensor:
    """Channels ``0::2 <- even``, ``1::2 <- odd`` (v1 in-place pair assignment), zero-padded to ``H``."""
    x = torch.stack([even, odd], dim=-1).flatten(-2)
    if x.shape[-1] < H:
        x = torch.cat([x, x.new_zeros(x.shape[:-1] + (H - x.shape[-1],))], dim=-1)
    return x


def _rotate_pairs(x: Tensor, c: Tensor, s: Tensor) -> Tensor:
    """v1 transport: rotate channel pairs ``(x[2i], x[2i+1])`` by ``(c, s)`` (``c, s`` broadcast ``(E, 1, 1)``)."""
    H = x.shape[-1]
    xe, xo = x[..., 0::2], x[..., 1::2]
    m = min(xe.shape[-1], xo.shape[-1])
    xe, xo = xe[..., :m], xo[..., :m]
    return _interleave(c * xe - s * xo, s * xe + c * xo, H)


# ================================================================================================================
# node-feature cores (return {0: x0})
# ================================================================================================================
class GCNCore(nn.Module):
    """v1 GCN: ``h <- lin(D^-1/2 (A+I) D^-1/2 h)``, SiLU between layers (sparse, cached operator)."""

    def __init__(self, v1: nn.Module) -> None:
        super().__init__()
        self.v1 = v1

    def forward(self, f0: Tensor, A: V1ComplexAdapter) -> dict[int, Tensor]:
        h = f0
        L = len(self.v1.layers)
        for i, lin in enumerate(self.v1.layers):
            h = lin(A.apply("gcn_adj", h))
            if i < L - 1:
                h = F.silu(h)
        return {0: h}


class GATCore(nn.Module):
    """v1 GAT (multi-head additive attention over both edge directions, LayerNorm + SiLU, residuals)."""

    def __init__(self, v1: nn.Module) -> None:
        super().__init__()
        self.v1 = v1

    @staticmethod
    def layer(gat: nn.Module, h: Tensor, send: Tensor, recv: Tensor) -> Tensor:
        n, B = h.shape[0], h.shape[1]
        Hh, Dh = gat.n_heads, gat.head_dim
        Wh = gat.W(h).view(n, B, Hh, Dh)
        a_s = (Wh * gat.attn_src).sum(-1)                                   # (n, B, H)
        a_d = (Wh * gat.attn_dst).sum(-1)
        e = F.leaky_relu(_gather(a_s, send) + _gather(a_d, recv), negative_slope=gat.negative_slope)  # (E, B, H)
        # stability shift (v1: zeros + scatter amax with include_self); it cancels in the softmax -> no gradient
        e_max = e.new_zeros(n, B, Hh).index_reduce_(0, recv, e.detach(), "amax", include_self=True)
        ex = torch.exp(e - _gather(e_max, recv))
        den = _scatter(ex, recv, n)
        alpha = ex / _gather(den, recv).clamp(min=1e-12)
        msg = (alpha.unsqueeze(-1) * _gather(Wh, send)).reshape(e.shape[0], B, Hh * Dh)
        return _scatter(msg, recv, n)

    def forward(self, f0: Tensor, A: V1ComplexAdapter) -> dict[int, Tensor]:
        send, recv = A.directed_edges()
        h = f0
        L = len(self.v1.layers)
        for i, (gat, norm) in enumerate(zip(self.v1.layers, self.v1.norms)):
            h_new = self.layer(gat, h, send, recv)
            if i < L - 1:
                h_new = F.silu(norm(h_new))
            h = h + h_new if h.shape[-1] == h_new.shape[-1] else h_new
        return {0: h}


class SchNetCore(nn.Module):
    """v1 SchNet (continuous filters from RBF-expanded distances; filters computed once per layer for the whole
    batch).  ``geometry='normalized'`` divides distances by the mesh's mean edge length."""

    def __init__(self, v1: nn.Module, geometry: str = "normalized") -> None:
        super().__init__()
        self.v1 = v1
        self.geometry = geometry

    def forward(self, f0: Tensor, A: V1ComplexAdapter) -> dict[int, Tensor]:
        send, recv = A.directed_edges()
        _, dist = A.directed_geometry(self.geometry == "normalized")
        rbf = self.v1.rbf(dist.to(f0.dtype))                                  # (E, n_rbf)
        h = self.v1.embed(f0)
        n = h.shape[0]
        for inter in self.v1.interactions:
            W = inter.filter_net(rbf).unsqueeze(1)                           # (E, 1, H)
            agg = _scatter(W * _gather(h, send), recv, n)
            h = inter.norm(h + inter.lin(agg))
        return {0: h}


class EGNNCore(nn.Module):
    """v1 EGNN (no coordinate update).  The first message layer on ``[h_i, h_j, d^2]`` is applied as node-level
    projections gathered per edge (same maths, ~6x fewer flops).  ``geometry='normalized'``: ``d^2 / s^2`` with
    ``s`` the mesh's mean edge length."""

    def __init__(self, v1: nn.Module, geometry: str = "normalized") -> None:
        super().__init__()
        self.v1 = v1
        self.geometry = geometry

    def forward(self, f0: Tensor, A: V1ComplexAdapter) -> dict[int, Tensor]:
        send, recv = A.directed_edges()
        _, dist = A.directed_geometry(self.geometry == "normalized")
        d2 = (dist * dist).view(-1, 1, 1)
        h = self.v1.embed(f0)
        n, Hd = h.shape[0], h.shape[-1]
        for layer in self.v1.layers:
            lin1, lin2 = layer.msg_mlp[0], layer.msg_mlp[2]
            W, b = lin1.weight, lin1.bias
            pre = (_gather(F.linear(h, W[:, :Hd]), send) + _gather(F.linear(h, W[:, Hd:2 * Hd]), recv)
                   + d2.to(h.dtype) * W[:, 2 * Hd] + b)
            msg = lin2(F.silu(pre))
            agg = _scatter(msg, recv, n)
            h = layer.norm(h + layer.node_mlp(torch.cat([h, agg], dim=-1)))
        return {0: h}


class GaugeCNNCore(nn.Module):
    """v1 GaugeEquivCNN: channel pairs rotated by the (2-D projected) edge angle before aggregation.

    ``v1_exact=False`` (default): ``dst`` receives ``W(T_{+t}(x_src))`` and ``src`` receives ``W(T_{-t}(x_dst))``
    (reverse direction); ``v1_exact=True``: v1's aggregation (the message from ``x_src`` goes to both endpoints)."""

    def __init__(self, v1: nn.Module, v1_exact: bool = False) -> None:
        super().__init__()
        self.v1 = v1
        self.v1_exact = bool(v1_exact)

    def forward(self, f0: Tensor, A: V1ComplexAdapter) -> dict[int, Tensor]:
        src, dst = A.edges[:, 0], A.edges[:, 1]
        p2 = A.pos[:, :2]
        ev = p2[dst] - p2[src]
        u = ev / ev.norm(dim=-1, keepdim=True).clamp(min=DIR_EPS)
        c, s = u[:, 0].view(-1, 1, 1), u[:, 1].view(-1, 1, 1)
        x = self.v1.encoder(f0)
        n = x.shape[0]
        deg = torch.bincount(torch.cat([src, dst]), minlength=n).clamp_min(1).to(x.dtype).view(-1, 1, 1)
        for i in range(self.v1.n_layers):
            sc = self.v1.transport_scale[i]
            W = self.v1.W_msg[i]
            m_f = W(_rotate_pairs(_gather(x, src), c.to(x.dtype), (sc * s).to(x.dtype)))
            if self.v1_exact:
                agg = _scatter(m_f, dst, n) + _scatter(m_f, src, n)
            else:
                m_r = W(_rotate_pairs(_gather(x, dst), -c.to(x.dtype), (-sc * s).to(x.dtype)))
                agg = _scatter(m_f, dst, n) + _scatter(m_r, src, n)
            x = F.silu(self.v1.W_self[i](x) + agg / deg)
        return {0: x}


class GEMCore(nn.Module):
    """v1 GEM-CNN ("Hermes"): anisotropic Fourier kernels ``K(theta) = sum_b B_b(theta) W_b`` + pair transport.

    ``v1_exact=False`` (default): messages along both edge directions (reverse angle ``theta + pi``, reverse
    transport), mean over the undirected neighbourhood, SiLU between convolutions.  ``v1_exact=True``: v1 (canonical
    ``src -> dst`` edges only, in-degree mean, linear)."""

    def __init__(self, v1: nn.Module, v1_exact: bool = False) -> None:
        super().__init__()
        self.v1 = v1
        self.v1_exact = bool(v1_exact)

    def _basis(self, theta: Tensor) -> Tensor:
        return self.v1._compute_basis(theta, theta.device)                   # (E, nb)

    @staticmethod
    def _conv(x_t: Tensor, basis: Tensor, W: Tensor) -> Tensor:
        """``msg[e] = sum_b basis[e, b] W_b x_t[e]`` as one matmul: ``(E, B, H) -> (E, B, H)``."""
        E, B, H = x_t.shape
        nb = basis.shape[1]
        Z = (basis.to(x_t.dtype).view(E, 1, nb, 1) * x_t.unsqueeze(2)).reshape(E, B, nb * H)
        Wm = W.permute(0, 2, 1).reshape(nb * H, W.shape[1])                   # [(b, d), h] = W[b, h, d]
        return Z @ Wm.to(Z.dtype)

    def forward(self, f0: Tensor, A: V1ComplexAdapter) -> dict[int, Tensor]:
        src, dst = A.edges[:, 0], A.edges[:, 1]
        p2 = A.pos[:, :2]
        ev = p2[dst] - p2[src]
        u = ev / ev.norm(dim=-1, keepdim=True).clamp(min=DIR_EPS)
        x = self.v1.encoder(f0)
        n = x.shape[0]
        if self.v1_exact:
            send, recv, uu = src, dst, u
            cnt = torch.bincount(dst, minlength=n)
        else:
            send, recv, uu = torch.cat([src, dst]), torch.cat([dst, src]), torch.cat([u, -u])
            cnt = torch.bincount(recv, minlength=n)
        theta = torch.atan2(uu[:, 1], uu[:, 0])
        basis = self._basis(theta)
        c, s = uu[:, 0].view(-1, 1, 1), uu[:, 1].view(-1, 1, 1)
        cnt = cnt.clamp_min(1).to(x.dtype).view(-1, 1, 1)
        L = self.v1.n_layers
        for i in range(L):
            x_self = self.v1.K_self[i](x)
            x_t = _rotate_pairs(_gather(x, send), c.to(x.dtype), s.to(x.dtype))
            agg = _scatter(self._conv(x_t, basis, self.v1.K_neigh_weights[i]), recv, n) / cnt
            x = x_self + agg + self.v1.bias[i]
            if not self.v1_exact and i < L - 1:
                x = F.silu(x)
        return {0: x}


NODE_CORES = {"gcn": GCNCore, "gat": GATCore, "schnet": SchNetCore, "egnn": EGNNCore,
              "gauge_cnn": GaugeCNNCore, "gem_cnn": GEMCore}


# ================================================================================================================
# cell-complex cores (return {0: x0, 1: x1[, 2: x2]})
# ================================================================================================================
def _simple_lift(lift: nn.Module, f0: Tensor, A: V1ComplexAdapter) -> tuple[Tensor, Tensor, Tensor]:
    """v1 ``_SimpleLift``: ``x0 = lift0(f0)``, ``x1 = lift1(mean of endpoints)``, ``x2 = lift2(mean of face edges)``
    (``|d1| x1 / 3`` in v1 = the mean for triangles; the mean is used for polygons)."""
    x0 = lift.lift0(f0)
    x1 = lift.lift1(0.5 * (_gather(x0, A.edges[:, 0]) + _gather(x0, A.edges[:, 1])))
    x2 = lift.lift2(A.apply("mean_1to2", x1))
    return x0, x1, x2


class MPSNCore(nn.Module):
    """v1 MPSN: independent GCN-like updates with binary adjacencies (``A00``, ``A11lo + A11up``, ``A22``)."""

    degrees = (0, 1, 2)

    def __init__(self, v1: nn.Module) -> None:
        super().__init__()
        self.v1 = v1

    def forward(self, f0: Tensor, A: V1ComplexAdapter) -> dict[int, Tensor]:
        m = self.v1
        x0, x1, x2 = _simple_lift(m.lift, f0, A)
        for i in range(m.n_layers):
            x0n = m.layers_0[i](A.apply("A00", x0))
            x1n = m.layers_1[i](A.apply("A11lo", x1) + A.apply("A11up", x1))
            x2n = m.layers_2[i](A.apply("A22", x2))
            x0 = F.silu(m.norms_0[i](x0n + x0))
            x1 = F.silu(m.norms_1[i](x1n + x1))
            x2 = F.silu(m.norms_2[i](x2n + x2))
        return {0: x0, 1: x1, 2: x2}


class SCCNNCore(nn.Module):
    """v1 SCCNN: scalar polynomial filters of the combinatorial Hodge Laplacians + per-degree channel mixing.

    ``normalize=False`` (default, v1): raw Laplacians.  ``normalize=True`` divides every Laplacian block (mesh) by
    its Gershgorin bound; use it with ``filter_init='identity'`` (``w_0 += 1``: every filter starts as the identity
    plus v1's random polynomial part) - with v1's initialisation it collapses to a constant predictor."""

    degrees = (0, 1, 2)

    def __init__(self, v1: nn.Module, normalize: bool = False, filter_init: str = "v1") -> None:
        super().__init__()
        self.v1 = v1
        self.normalize = bool(normalize)
        if filter_init not in ("v1", "identity"):
            raise ValueError(f"filter_init must be 'v1' or 'identity', got {filter_init!r}")
        self.filter_init = filter_init
        if filter_init == "identity":
            with torch.no_grad():
                for ws in (v1.w_0, v1.w_1, v1.w_2):
                    for w in ws:
                        w[0] += 1.0

    @staticmethod
    def _poly(x: Tensor, name: str, w: Tensor, A: V1ComplexAdapter) -> Tensor:
        out = w[0] * x
        Lx = x
        for t in range(1, w.shape[0]):
            Lx = A.apply(name, Lx)
            out = out + w[t] * Lx
        return out

    def forward(self, f0: Tensor, A: V1ComplexAdapter) -> dict[int, Tensor]:
        m = self.v1
        sfx = "_n" if self.normalize else ""
        x0, x1, x2 = _simple_lift(m.lift, f0, A)
        for i in range(m.n_layers):
            x0 = F.silu(m.mix_0[i](self._poly(x0, "L0" + sfx, m.w_0[i], A)))
            x1 = F.silu(m.mix_1[i](self._poly(x1, "L1" + sfx, m.w_1[i], A)))
            x2 = F.silu(m.mix_2[i](self._poly(x2, "L2" + sfx, m.w_2[i], A)))
        return {0: x0, 1: x1, 2: x2}


class CWNetCore(nn.Module):
    """v1 CW Network: boundary / coboundary messages through ``d0``, ``d1`` (and transposes) between degrees."""

    degrees = (0, 1, 2)

    def __init__(self, v1: nn.Module) -> None:
        super().__init__()
        self.v1 = v1

    def forward(self, f0: Tensor, A: V1ComplexAdapter) -> dict[int, Tensor]:
        m = self.v1
        d0, d0T, d1, d1T = A.d[0], A.dT[0], A.d[1], A.dT[1]
        x0 = m.encoder(f0)
        x1 = spmm(d0, x0.contiguous(), d0T)
        x2 = spmm(d1, x1.contiguous(), d1T)
        for i in range(m.n_layers):
            m0 = m.msg_0_from_1[i](spmm(d0T, x1.contiguous(), d0))
            m1b = m.msg_1_from_0[i](spmm(d0, x0.contiguous(), d0T))
            m1c = m.msg_1_from_2[i](spmm(d1T, x2.contiguous(), d1))
            m2 = m.msg_2_from_1[i](spmm(d1, x1.contiguous(), d1T))
            x0 = F.silu(m.update_0[i](x0 + m0))
            x1 = F.silu(m.update_1[i](x1 + m1b + m1c))
            x2 = F.silu(m.update_2[i](x2 + m2))
        return {0: x0, 1: x1, 2: x2}


class CliffordCore(nn.Module):
    """v1 Clifford-SMPN: Cl(2,0) multivector channels, ``d0`` message passing, geometric-product layers."""

    degrees = (0, 1)

    def __init__(self, v1: nn.Module) -> None:
        super().__init__()
        self.v1 = v1

    def forward(self, f0: Tensor, A: V1ComplexAdapter) -> dict[int, Tensor]:
        m = self.v1
        d0, d0T = A.d[0], A.dT[0]
        x0 = m.lift_0(f0)
        x1 = spmm(d0, x0.contiguous(), d0T)
        for i in range(m.n_layers):
            m0 = spmm(d0T, x1.contiguous(), d0)
            m1 = spmm(d0, x0.contiguous(), d0T)
            x0 = m._clifford_layer(x0 + m0, m.W_0_a[i], m.W_0_b[i], m.norms_0[i])
            x1 = m._clifford_layer(x1 + m1, m.W_1_a[i], m.W_1_b[i], m.norms_1[i])
        return {0: x0, 1: x1}


COMPLEX_CORES = {"mpsn": MPSNCore, "sccnn": SCCNNCore, "cw_net": CWNetCore, "clifford_smpn": CliffordCore}


class V1EdgeReadout(nn.Module):
    """v1 edge readout of MPSN / SCCNN / Clifford-SMPN: ``readout(x_1)`` on the edges (the v1 edge protocol; the task
    target has been replaced by the edge-averaged node target, see ``registry.prepare_task``)."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, feats: dict[int, Tensor], A: V1ComplexAdapter) -> Tensor:
        return self.net(feats[1])


# ================================================================================================================
# operator baselines (return the prediction)
# ================================================================================================================
class DeepONetCore(nn.Module):
    """v1 DeepONet: branch on the per-mesh mean of the node features, trunk on node positions.  For block-diagonal
    batches the mean is taken per mesh (v1's ``x.mean(dim=1)`` would mix the meshes of a batch)."""

    def __init__(self, v1: nn.Module) -> None:
        super().__init__()
        self.v1 = v1

    def forward(self, f0: Tensor, A: V1ComplexAdapter) -> Tensor:
        net = self.v1
        n0, B, _ = f0.shape
        pos = A.pos[:, :net.trunk[0].in_features].to(f0.dtype)
        trunk = net.trunk(pos)                                                 # (n0, nb)
        if A.num_graphs == 1:
            br = net.branch(f0.mean(0)).view(B, net.n_basis, net.out_dim)      # (B, nb, O)
            return torch.einsum("nk,bko->nbo", trunk, br) + net.bias
        g = A.graph_ids(0)
        br = net.branch(segment_mean(f0, g, A.num_graphs))                     # (G, B, nb*O)
        br = br.view(A.num_graphs, B, net.n_basis, net.out_dim)
        return torch.einsum("nk,nbko->nbo", trunk, br.index_select(0, g)) + net.bias


class FNOCore(nn.Module):
    """v1 FNO2d on a regular grid (vertex ``i * ny + j`` <-> grid cell ``(i, j)``, v1's reshaping)."""

    def __init__(self, v1: nn.Module, grid: tuple[int, int]) -> None:
        super().__init__()
        self.v1 = v1
        self.grid = (int(grid[0]), int(grid[1]))

    def forward(self, f0: Tensor, A: V1ComplexAdapter) -> Tensor:
        if A.num_graphs != 1 or A.n0 != self.grid[0] * self.grid[1]:
            raise ValueError(f"FNO needs the {self.grid} grid it was built for (got n0={A.n0}, "
                             f"{A.num_graphs} meshes)")
        y = self.v1.fno(f0.transpose(0, 1), self.grid)                         # (B, n0, O)
        return y.transpose(0, 1).contiguous()


class V1OursCore(nn.Module):
    """v1 ``GaugeHodgeNetwork`` (the paper's model; diagonal rank-8 metric, ``mp_hidden=16``) on one shared mesh.

    It has per-cell parameters (metric bases of size ``n_k x 8``), so it only runs on the mesh it was built for.
    Training uses v1's ``forward_batch`` (its batch-mean metric couples the samples of a batch, as when v1 was
    trained); evaluation is per sample (``eval_per_sample=True``, as in the v1 evaluation script
    ``compute_all_metrics.py``).
    """

    def __init__(self, v1: nn.Module, n: tuple[int, int, int], eval_per_sample: bool = True) -> None:
        super().__init__()
        self.v1 = v1
        self.n = tuple(int(v) for v in n)
        self.eval_per_sample = bool(eval_per_sample)

    def forward(self, f0: Tensor, A: V1ComplexAdapter) -> Tensor:
        if A.num_graphs != 1 or (A.n0, A.n1, A.n2) != self.n:
            raise ValueError(f"ours_v1 has per-cell parameters for a mesh with (n0, n1, n2) = {self.n}; got "
                             f"({A.n0}, {A.n1}, {A.n2}) with {A.num_graphs} mesh(es)")
        x = f0.transpose(0, 1).contiguous()                                    # (B, n0, F)
        if self.training or not self.eval_per_sample or x.shape[0] == 1:
            y = self.v1.forward_batch(x, A)
        else:
            y = torch.cat([self.v1.forward_batch(x[i:i + 1], A) for i in range(x.shape[0])])
        return y.transpose(0, 1).contiguous()


# ================================================================================================================
# the uniform wrapper
# ================================================================================================================
class BaselineModel(nn.Module):
    """``forward(inputs: {k: (n_k, B, F_k)}, K) -> (n_out, B, out_dim)`` for any baseline (v2 layout).

    Args:
        name: registry name.
        encoder: node input encoder (``NodeInputEncoder``).
        core: module ``core(f0 (n0, B, F), adapter[, e0]) -> {k: (n_k, B, H)}`` (hidden features) or the final
            prediction tensor when ``head is None``.
        head: target head ``head(feats, adapter) -> (n_t, B, out)`` or ``None``.
        edge_encoder: optional ``EdgeInputEncoder`` whose output is passed to the core as third argument.
        build: keyword arguments that rebuild the model with ``rhmp.baselines.registry.build_model`` (checkpoints).
        info: description (hidden width, parameter budget, representation notes) stored in checkpoints/results.
        amp: bf16 autocast for the dense parts on CUDA (sparse products stay fp32).
    """

    def __init__(self, name: str, encoder: nn.Module, core: nn.Module, head: nn.Module | None = None, *,
                 edge_encoder: nn.Module | None = None, build: dict | None = None, info: dict | None = None,
                 amp: bool = False) -> None:
        super().__init__()
        self.name = name
        self.encoder = encoder
        self.edge_encoder = edge_encoder
        self.core = core
        self.head = head
        self.build = dict(build or {})
        self.info = dict(info or {})
        self.amp = bool(amp)
        self.record_diagnostics = False

    def num_parameters(self) -> int:
        """Number of trainable scalars."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _autocast(self, A: V1ComplexAdapter):
        if self.amp and A.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _check(self, inputs: dict[int, Tensor], A: V1ComplexAdapter) -> None:
        want = getattr(self.encoder, "in_dims", None)
        if want is None:
            return
        got = {int(k): int(v.shape[-1]) for k, v in inputs.items()}
        if got != want:
            raise ValueError(f"{self.name}: inputs {got} do not match the model's input layout {want}")
        for k, v in inputs.items():
            if v.dim() != 3 or v.shape[0] != A.n[int(k)]:
                raise ValueError(f"{self.name}: inputs[{k}] must be (n_{k}={A.n[int(k)]}, B, F), got {tuple(v.shape)}")

    def forward(self, inputs: dict[int, Tensor], K: Any) -> Tensor:
        A = adapter_for(K)
        inputs = {int(k): v for k, v in dict(inputs).items()}
        self._check(inputs, A)
        with self._autocast(A):
            f0 = self.encoder(inputs, A)
            if self.edge_encoder is not None:
                out = self.core(f0, A, self.edge_encoder(inputs, A, dtype=f0.dtype))
            else:
                out = self.core(f0, A)
            y = out if self.head is None else self.head(out, A)
        return y.float()

    def to_checkpoint(self) -> dict:
        """``{'model': name, 'build': kwargs, 'info': ..., 'state_dict': ...}`` (``registry.from_checkpoint``)."""
        return {"model": self.name, "build": dict(self.build), "info": dict(self.info),
                "state_dict": self.state_dict()}


# ================================================================================================================
# construction
# ================================================================================================================
def _v1_class(name: str):
    v1 = import_v1()
    return {
        "gcn": v1["graph"].GCNBaseline, "gat": v1["graph"].GATBaseline, "schnet": v1["graph"].SchNetBaseline,
        "egnn": v1["graph"].EGNNBaseline, "gauge_cnn": v1["adv"].GaugeEquivCNNBaseline,
        "gem_cnn": v1["adv"].HermesBaseline, "cw_net": v1["adv"].CWNetBaseline,
        "clifford_smpn": v1["adv"].CliffordNetBaseline, "mpsn": v1["topo"].MPSNBaseline,
        "sccnn": v1["topo"].SCCNNBaseline,
    }[name]


def make_v1_baseline(name: str, io: TaskIO, hidden: int, *, n_layers: int = 4, geometry: str = "normalized",
                     v1_exact: bool = False, normalize: bool = False, filter_init: str = "v1", head: str = "node",
                     directional: bool = True, node_columns: list[int] | None = None,
                     amp: bool = False) -> BaselineModel:
    """Build a v1 baseline (``NODE_CORES`` or ``COMPLEX_CORES``) for task ``io`` with width ``hidden``.

    Args:
        name: ``gcn | gat | schnet | egnn | gauge_cnn | gem_cnn | mpsn | sccnn | cw_net | clifford_smpn``.
        io: task description.
        hidden: hidden width (the v1 constructor's ``hidden``).
        n_layers: message-passing layers (v1 protocol: 4).
        geometry: ``'normalized' | 'raw'`` distances (SchNet, EGNN).
        v1_exact: v1's aggregation for gauge_cnn / gem_cnn.
        normalize / filter_init: SCCNN Laplacian normalisation and filter initialisation (see ``SCCNNCore``).
        head: MPSN / SCCNN / Clifford-SMPN: ``'node'`` (multi-degree ``ComplexHead`` on the task's target cells) or
            ``'v1_edge'`` (v1's own edge readout on ``x_1``; the task must carry the edge-averaged target).
        directional / node_columns: input encoding (``NodeInputEncoder``).
        amp: bf16 autocast.
    Returns:
        :class:`BaselineModel`.
    """
    enc = NodeInputEncoder(io.in_dims, io.even_dims, io.spatial_dim, directional=directional,
                           node_columns=node_columns)
    f_in = enc.out_dim
    out = io.out_dim
    cls = _v1_class(name)
    build = dict(hidden=hidden, n_layers=n_layers, geometry=geometry, v1_exact=v1_exact, normalize=normalize,
                 filter_init=filter_init, head=head, directional=directional, node_columns=node_columns)
    if name in NODE_CORES:
        v1m = cls(f_in=f_in, hidden=hidden, n_layers=n_layers, out_dim=out, task="node")
        node_net = edge_net = None
        if name in ("gcn", "gat", "schnet", "egnn"):
            node_net, edge_net = v1m.node_mlp, v1m.edge_mlp        # BaselineBase heads
            v1m.node_mlp = None
            v1m.edge_mlp = None
        else:
            node_net = v1m.decoder
            v1m.decoder = None
        if io.target_degree != 0:
            node_net = None
        if io.target_degree != 1:
            edge_net = None
        kw = {}
        if name in ("schnet", "egnn"):
            kw["geometry"] = geometry
        if name in ("gauge_cnn", "gem_cnn"):
            kw["v1_exact"] = v1_exact
        core = NODE_CORES[name](v1m, **kw)
        head = node_model_head(io, hidden, node_net=node_net, edge_net=edge_net)
    elif name in COMPLEX_CORES:
        v1m = cls(f_in=f_in, hidden=hidden, n_layers=n_layers)
        width = hidden
        v1_readout = None
        if name == "clifford_smpn":
            width = v1m.ch * v1m.mv                                  # multivector channels (hidden rounded to 4)
            v1_readout, v1m.readout = v1m.readout, None
        elif name == "cw_net":
            v1m.decoder = None
        else:
            v1_readout, v1m.readout = v1m.readout, None
        if name == "sccnn":
            core = COMPLEX_CORES[name](v1m, normalize=normalize, filter_init=filter_init)
        else:
            core = COMPLEX_CORES[name](v1m)
        if head == "v1_edge":
            if v1_readout is None or io.target_degree != 1:
                raise ValueError(f"head='v1_edge' needs MPSN/SCCNN/Clifford-SMPN and an edge (edge-averaged) target, "
                                 f"got {name} with a degree-{io.target_degree} target (use registry.prepare_task)")
            if out != 1:                                             # v1's readout has one output channel
                v1_readout[-1] = torch.nn.Linear(v1_readout[-1].in_features, out)
            head_mod = V1EdgeReadout(v1_readout)
        elif head == "node":
            head_mod = ComplexHead(core.degrees, width, out, io.target_degree, io.odd_target)
        else:
            raise ValueError(f"head must be 'node' or 'v1_edge', got {head!r}")
        head = head_mod
    else:
        raise KeyError(f"unknown v1 baseline {name!r}")
    info = {"name": name, "hidden": hidden, "n_layers": n_layers, "encoder": enc.layout, "f_in": f_in,
            "head": type(head).__name__}
    if name == "sccnn":
        info.update(normalize=normalize, filter_init=filter_init)
    return BaselineModel(name, enc, core, head, build=build, info=info, amp=amp)


def make_ours_v1(io: TaskIO, C: int, *, directional: bool = True, eval_per_sample: bool = True,
                 amp: bool = False) -> BaselineModel:
    """v1 ``GaugeHodgeNetwork`` exactly as the v1 training script ``formal_benchmark.py`` builds it for the paper tasks
    (``n_layers=4, mp_hidden=16, metric_type='diagonal', metric_rank=8``) on the task's shared triangle mesh.

    Readout: ``node_scalar`` -> v1 ``scalar``; ``node_vector`` -> v1 ``vector`` (EquivariantReconstruction);
    edge targets -> v1 ``edge_scalar``.  Face / tet targets and variable meshes are not supported.
    """
    net = import_v1()["network"]
    enc = NodeInputEncoder(io.in_dims, io.even_dims, io.spatial_dim, directional=directional)
    if io.variable_mesh or io.n is None:
        raise ValueError("ours_v1 has per-cell parameters: shared-mesh tasks only")
    if io.target_kind == "node_vector":
        task, out = "vector", io.out_dim
        if io.out_dim != io.spatial_dim:
            raise ValueError(f"ours_v1 vector readout predicts one D={io.spatial_dim} field, target has {io.out_dim}")
    elif io.target_degree == 0:
        task, out = "scalar", io.out_dim
    elif io.target_degree == 1:
        task, out = "edge_scalar", io.out_dim
    else:
        raise ValueError(f"ours_v1 has no readout for degree-{io.target_degree} targets")
    n0, n1, n2 = io.n[0], io.n[1], io.n[2]
    v1m = net.GaugeHodgeNetwork(f_in=enc.out_dim, C=C, n_layers=4, n0=n0, n1=n1, n2=n2, task=task,
                                spatial_dim=io.spatial_dim, out_dim=out, mp_hidden=16, metric_type="diagonal",
                                metric_rank=8)
    core = V1OursCore(v1m, (n0, n1, n2), eval_per_sample=eval_per_sample)
    info = {"name": "ours_v1", "C": C, "v1_task": task, "encoder": enc.layout, "f_in": enc.out_dim,
            "note": "v1 paper model (per-cell metric bases, batch-coupled forward_batch in training)"}
    return BaselineModel("ours_v1", enc, core, None, build=dict(C=C, directional=directional,
                                                                eval_per_sample=eval_per_sample),
                         info=info, amp=amp)


def make_deeponet(io: TaskIO, hidden: int, *, n_basis: int = 64, amp: bool = False) -> BaselineModel:
    """v1 DeepONet (branch/trunk width ``hidden``, 64 basis functions, trunk on the ``D`` coordinates)."""
    op = import_v1()["op"]
    enc = NodeInputEncoder(io.in_dims, io.even_dims, io.spatial_dim)
    v1m = op.DeepONet(enc.out_dim, io.out_dim, branch_width=hidden, trunk_width=hidden, n_basis=n_basis,
                      pos_dim=io.spatial_dim)
    info = {"name": "deeponet", "hidden": hidden, "n_basis": n_basis, "encoder": enc.layout}
    return BaselineModel("deeponet", enc, DeepONetCore(v1m), None, build=dict(hidden=hidden, n_basis=n_basis),
                         info=info, amp=amp)


def make_fno(io: TaskIO, hidden: int, *, modes: int = 12, n_layers: int = 4, amp: bool = False) -> BaselineModel:
    """v1 FNO2d (width ``hidden``, ``modes`` Fourier modes per axis) on a regular-grid task (T1, T1q)."""
    op = import_v1()["op"]
    if io.grid is None:
        raise ValueError("fno needs a regular grid (vertex i*ny+j at (x_i, y_j)); this task has none")
    enc = NodeInputEncoder(io.in_dims, io.even_dims, io.spatial_dim)
    m = min(modes, io.grid[0] // 2, io.grid[1] // 2 + 1)   # v1 value (12) on the 32 x 32 grid
    v1m = op.FNOWrapper(enc.out_dim, io.out_dim, width=hidden, modes=m, n_layers=n_layers)
    info = {"name": "fno", "hidden": hidden, "modes": m, "n_layers": n_layers, "grid": list(io.grid),
            "encoder": enc.layout}
    return BaselineModel("fno", enc, FNOCore(v1m, io.grid), None, build=dict(hidden=hidden, modes=m,
                                                                             n_layers=n_layers),
                         info=info, amp=amp)
