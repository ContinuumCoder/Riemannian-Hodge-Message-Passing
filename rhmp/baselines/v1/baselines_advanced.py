# Vendored v1 code.  Original: module `baselines_advanced` of the RHMP v1 code base,
#   https://github.com/ContinuumCoder/Riemannian-Hodge-Message-Passing
# which accompanies "Learning Discrete Riemannian Metrics for Physical Fields with Cochain-Frame Equivariance"
# (Zheng & Allen-Blanchette, arXiv:2608.14556).  Kept verbatim except for package-relative imports (`from .gauge_hodge_mp...`).
# Used by rhmp.baselines (the v1 baselines and the v1 model `ours_v1`) and by `python -m rhmp.train --eval-v1`.
"""
Advanced baselines:
  8. GaugeEquivCNN  — Cohen et al., ICML 2019 (gauge equivariant convolution)
  9. CWNet         — Bodnar et al., ICLR 2022 (CW Network on cell complexes)
  10. CliffordNet   — Brandstetter et al., ICLR 2023 (Clifford algebra networks)

All adapted for our benchmark: node-level input → node/edge-level output.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .gauge_hodge_mp.utils import spmm, batch_spmm


# ============================================================
# 8. Gauge Equivariant CNN (Cohen et al., ICML 2019)
# ============================================================
class GaugeEquivCNNBaseline(nn.Module):
    """
    Simplified gauge equivariant convolution on meshes.
    Core idea from Cohen et al. 2019: parallel transport features along edges
    before aggregation, making the convolution gauge-equivariant.

    For each edge (i,j):
      - Compute transport matrix g_{i←j} from edge geometry
      - Transport feature: f_transported = g_{i←j} @ f_j
      - Aggregate with learned kernel

    Simplified: g_{i←j} is a 2D rotation based on edge angle (for 2D meshes)
    or identity + antisymmetric perturbation (for general).
    """

    def __init__(self, f_in: int, hidden: int = 32, n_layers: int = 4,
                 out_dim: int = 1, task: str = 'node'):
        super().__init__()
        self.encoder = nn.Linear(f_in, hidden)
        self.n_layers = n_layers

        # Per-layer: transport-aware message passing
        self.W_msg = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.W_self = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        # Learnable transport scaling per layer
        self.transport_scale = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1)) for _ in range(n_layers)
        ])

        self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, out_dim))

    def _compute_transport(self, pos, edges):
        """Compute parallel transport directions from edge geometry."""
        src, dst = edges[:, 0], edges[:, 1]
        edge_vec = pos[dst] - pos[src]  # (n_edges, d)
        edge_len = edge_vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        edge_dir = edge_vec / edge_len
        return edge_dir, src, dst

    def forward(self, f0, K):
        x = self.encoder(f0)  # (n0, hidden)
        pos = K.pos[:, :2] if K.pos.size(1) >= 2 else K.pos  # 2D positions
        edges = K.edges
        edge_dir, src, dst = self._compute_transport(pos, edges)

        for i in range(self.n_layers):
            x_src = x[src]  # (n_edges, hidden)
            x_dst = x[dst]

            # Gauge transport: rotate feature pairs by edge angle
            scale = self.transport_scale[i]
            if edge_dir.size(1) >= 2:
                cos_a = edge_dir[:, 0:1]  # (n_edges, 1)
                sin_a = edge_dir[:, 1:2]
                h = x_src.size(1)
                x_even = x_src[:, 0::2]
                x_odd = x_src[:, 1::2]
                min_ch = min(x_even.size(1), x_odd.size(1))
                x_rot_even = cos_a * x_even[:, :min_ch] - scale * sin_a * x_odd[:, :min_ch]
                x_rot_odd = scale * sin_a * x_even[:, :min_ch] + cos_a * x_odd[:, :min_ch]
                x_transported = torch.zeros_like(x_src)
                x_transported[:, 0::2][:, :min_ch] = x_rot_even
                x_transported[:, 1::2][:, :min_ch] = x_rot_odd
            else:
                x_transported = x_src

            msg = self.W_msg[i](x_transported)
            agg = torch.zeros_like(x)
            agg.index_add_(0, dst, msg)
            agg.index_add_(0, src, msg)  # undirected: aggregate both directions
            count = torch.zeros(x.size(0), 1, device=x.device)
            count.index_add_(0, dst, torch.ones(len(dst), 1, device=x.device))
            count.index_add_(0, src, torch.ones(len(src), 1, device=x.device))
            count = count.clamp(min=1)
            agg = agg / count

            x = F.silu(self.W_self[i](x) + agg)

        return self.decoder(x)

    def forward_batch(self, f0, K):
        """f0: (B, n0, f_in) — vectorized batch"""
        B = f0.size(0)
        x = self.encoder(f0)  # (B, n0, hidden)
        pos = K.pos[:, :2] if K.pos.size(1) >= 2 else K.pos
        edges = K.edges
        src, dst = edges[:, 0], edges[:, 1]

        edge_vec = pos[dst] - pos[src]
        edge_len = edge_vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        edge_dir = edge_vec / edge_len

        for i in range(self.n_layers):
            x_src = x[:, src]  # (B, n_edges, hidden)
            scale = self.transport_scale[i]

            if edge_dir.size(1) >= 2:
                cos_a = edge_dir[:, 0:1].unsqueeze(0)  # (1, n_edges, 1)
                sin_a = edge_dir[:, 1:2].unsqueeze(0)
                h = x_src.size(-1)
                x_even = x_src[:, :, 0::2]
                x_odd = x_src[:, :, 1::2]
                min_ch = min(x_even.size(-1), x_odd.size(-1))
                x_rot_even = cos_a * x_even[:, :, :min_ch] - scale * sin_a * x_odd[:, :, :min_ch]
                x_rot_odd = scale * sin_a * x_even[:, :, :min_ch] + cos_a * x_odd[:, :, :min_ch]
                x_transported = torch.zeros_like(x_src)
                x_transported[:, :, 0::2][:, :, :min_ch] = x_rot_even
                x_transported[:, :, 1::2][:, :, :min_ch] = x_rot_odd
            else:
                x_transported = x_src

            msg = self.W_msg[i](x_transported)  # (B, n_edges, hidden)
            n0 = x.size(1)
            n_e = msg.size(1)
            H = msg.size(2)
            # Batched scatter via flattening
            offsets = torch.arange(B, device=x.device).unsqueeze(1) * n0  # (B, 1)
            dst_flat = (dst.unsqueeze(0) + offsets).reshape(-1)  # (B*n_edges,)
            src_flat = (src.unsqueeze(0) + offsets).reshape(-1)
            msg_flat = msg.reshape(-1, H)  # (B*n_edges, hidden)
            agg_flat = torch.zeros(B * n0, H, device=x.device)
            agg_flat.index_add_(0, dst_flat, msg_flat)
            agg_flat.index_add_(0, src_flat, msg_flat)
            agg = agg_flat.reshape(B, n0, H)
            count = torch.zeros(n0, 1, device=x.device)
            count.index_add_(0, dst, torch.ones(len(dst), 1, device=x.device))
            count.index_add_(0, src, torch.ones(len(src), 1, device=x.device))
            agg = agg / count.clamp(min=1).unsqueeze(0)

            x = F.silu(self.W_self[i](x) + agg)

        return self.decoder(x)


# ============================================================
# 9. CW Network (Bodnar et al., ICLR 2022)
# ============================================================
class CWNetBaseline(nn.Module):
    """
    CW Network baseline (Bodnar et al., ICLR 2022).

    Message passing on cell complexes using boundary and coboundary operators.
    Each cell receives messages from:
      - Adjacent cells of same dimension (via adjacency)
      - Boundary cells (lower dimension via B)
      - Coboundary cells (higher dimension via B^T)

    Simplified: uses d0, d1 operators from CellComplex for message passing
    between 0-cells (nodes), 1-cells (edges), 2-cells (faces).
    """

    def __init__(self, f_in: int, hidden: int = 32, n_layers: int = 4,
                 out_dim: int = 1, task: str = 'node'):
        super().__init__()
        self.encoder = nn.Linear(f_in, hidden)
        self.n_layers = n_layers

        # Message functions per cell dimension
        self.msg_0_from_1 = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.update_0 = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])

        self.msg_1_from_0 = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.msg_1_from_2 = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.update_1 = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])

        self.msg_2_from_1 = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.update_2 = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])

        self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, out_dim))

    def forward(self, f0, K):
        d0, d1 = K.d0, K.d1  # boundary operators: (n1,n0), (n2,n1)

        x0 = self.encoder(f0)  # (n0, h)
        x1 = spmm(d0, x0)     # lift to edges via coboundary: (n1, h)
        x2 = spmm(d1, x1)     # lift to faces via coboundary: (n2, h)

        for i in range(self.n_layers):
            # d^T: boundary (higher→lower), d: coboundary (lower→higher)
            m_0 = self.msg_0_from_1[i](spmm(d0.t(), x1))   # edges→nodes
            m_1_b = self.msg_1_from_0[i](spmm(d0, x0))      # nodes→edges
            m_1_c = self.msg_1_from_2[i](spmm(d1.t(), x2))  # faces→edges
            m_2 = self.msg_2_from_1[i](spmm(d1, x1))        # edges→faces

            x0 = F.silu(self.update_0[i](x0 + m_0))
            x1 = F.silu(self.update_1[i](x1 + m_1_b + m_1_c))
            x2 = F.silu(self.update_2[i](x2 + m_2))

        return self.decoder(x0)

    def forward_batch(self, f0, K):
        d0, d1 = K.d0, K.d1

        x0 = self.encoder(f0)
        x1 = batch_spmm(d0, x0)
        x2 = batch_spmm(d1, x1)

        for i in range(self.n_layers):
            m_0 = self.msg_0_from_1[i](batch_spmm(d0.t(), x1))
            m_1_b = self.msg_1_from_0[i](batch_spmm(d0, x0))
            m_1_c = self.msg_1_from_2[i](batch_spmm(d1.t(), x2))
            m_2 = self.msg_2_from_1[i](batch_spmm(d1, x1))

            x0 = F.silu(self.update_0[i](x0 + m_0))
            x1 = F.silu(self.update_1[i](x1 + m_1_b + m_1_c))
            x2 = F.silu(self.update_2[i](x2 + m_2))

        return self.decoder(x0)


# ============================================================
# 10. Clifford Network (Brandstetter et al., ICLR 2023)
# ============================================================
class CliffordNetBaseline(nn.Module):
    """
    Clifford Group Equivariant Simplicial Message Passing Networks
    (Liu, Ruhe, Eijkelboom, Forré — ICLR 2024)

    Multivector features in Cl(2,0) on simplicial complex.
    MP via boundary operators d0, d1 (fast batch_spmm).
    Geometric product gp(Wa(x), Wb(x)) for nonlinear Clifford mixing.
    Residual + LayerNorm for training stability.
    """

    def __init__(self, f_in: int, hidden: int = 32, n_layers: int = 4,
                 out_dim: int = 1, task: str = 'node'):
        super().__init__()
        self.mv = 4  # Cl(2,0): {1, e1, e2, e12}
        self.ch = hidden // self.mv
        self.lift_0 = nn.Linear(f_in, self.ch * self.mv)
        self.n_layers = n_layers

        # Per-layer: linear transforms for geometric product + LayerNorm
        self.W_0_a = nn.ModuleList([nn.Linear(self.ch*self.mv, self.ch*self.mv) for _ in range(n_layers)])
        self.W_0_b = nn.ModuleList([nn.Linear(self.ch*self.mv, self.ch*self.mv) for _ in range(n_layers)])
        self.W_1_a = nn.ModuleList([nn.Linear(self.ch*self.mv, self.ch*self.mv) for _ in range(n_layers)])
        self.W_1_b = nn.ModuleList([nn.Linear(self.ch*self.mv, self.ch*self.mv) for _ in range(n_layers)])
        self.norms_0 = nn.ModuleList([nn.LayerNorm(self.ch*self.mv) for _ in range(n_layers)])
        self.norms_1 = nn.ModuleList([nn.LayerNorm(self.ch*self.mv) for _ in range(n_layers)])

        self.readout = nn.Sequential(nn.Linear(self.ch*self.mv, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def _gp(self, a, b):
        """Cl(2,0) geometric product. a,b: (..., ch, 4)"""
        a0,a1,a2,a12 = a[...,0:1],a[...,1:2],a[...,2:3],a[...,3:4]
        b0,b1,b2,b12 = b[...,0:1],b[...,1:2],b[...,2:3],b[...,3:4]
        return torch.cat([
            a0*b0 + a1*b1 + a2*b2 - a12*b12,
            a0*b1 + a1*b0 - a2*b12 + a12*b2,
            a0*b2 + a1*b12 + a2*b0 - a12*b1,
            a0*b12 + a1*b2 - a2*b1 + a12*b0,
        ], dim=-1)

    def _clifford_layer(self, x, Wa, Wb, norm):
        """gp(Wa(x), Wb(x)) + residual + LayerNorm"""
        ch, mv = self.ch, self.mv
        shape = x.shape[:-1]
        a = Wa(x).reshape(*shape, ch, mv)
        b = Wb(x).reshape(*shape, ch, mv)
        return norm(x + self._gp(a, b).reshape(*shape, ch * mv))

    def forward(self, f0, K):
        from .gauge_hodge_mp.utils import spmm
        d0 = K.d0
        x0 = self.lift_0(f0)
        x1 = spmm(d0, x0)
        for i in range(self.n_layers):
            m0 = spmm(d0.t(), x1)
            m1 = spmm(d0, x0)
            x0 = self._clifford_layer(x0 + m0, self.W_0_a[i], self.W_0_b[i], self.norms_0[i])
            x1 = self._clifford_layer(x1 + m1, self.W_1_a[i], self.W_1_b[i], self.norms_1[i])
        return self.readout(x1)

    def forward_batch(self, f0, K):
        from .gauge_hodge_mp.utils import batch_spmm
        d0 = K.d0
        x0 = self.lift_0(f0)
        x1 = batch_spmm(d0, x0)
        for i in range(self.n_layers):
            m0 = batch_spmm(d0.t(), x1)
            m1 = batch_spmm(d0, x0)
            x0 = self._clifford_layer(x0 + m0, self.W_0_a[i], self.W_0_b[i], self.norms_0[i])
            x1 = self._clifford_layer(x1 + m1, self.W_1_a[i], self.W_1_b[i], self.norms_1[i])
        return self.readout(x1)


# ============================================================
# 11. GEM-CNN — Gauge Equivariant Mesh CNNs (de Haan et al., ICLR 2021)
# ============================================================
class HermesBaseline(nn.Module):
    """
    GEM-CNN (de Haan, Weiler, Cohen, Welling — ICLR 2021).
    Faithful implementation: anisotropic LINEAR kernels + parallel transport.

    Convolution (Eq.2 in paper):
      (K*f)_p = K_self @ f_p + Σ_{q∈N_p} K_neigh(θ_pq) @ ρ(g_{q→p}) @ f_q

    K_neigh(θ) = Σ_k w_k * B_k(θ) where B_k are basis kernels (cos/sin harmonics).
    For scalar features (ρ_0→ρ_0): K_neigh is isotropic (no θ dependence).
    We use n_freq Fourier basis terms for the anisotropic kernel.
    All operations are LINEAR (no MLP) — faithful to paper.
    """
    def __init__(self, f_in: int, hidden: int = 32, n_layers: int = 4,
                 out_dim: int = 1, task: str = 'node', n_freq: int = 4):
        super().__init__()
        self.encoder = nn.Linear(f_in, hidden)
        self.n_layers = n_layers
        self.n_freq = n_freq
        # Per-layer: K_self (linear) + K_neigh basis weights
        # K_neigh(θ) = w_0 * I + Σ_{k=1}^{n_freq} (w_{2k-1}*cos(kθ) + w_{2k}*sin(kθ))
        # Each w_k is a (hidden, hidden) linear map
        n_basis = 1 + 2 * n_freq  # 1 (isotropic) + 2*n_freq (anisotropic)
        self.K_self = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(n_layers)])
        self.K_neigh_weights = nn.ParameterList([
            nn.Parameter(torch.randn(n_basis, hidden, hidden) * 0.01) for _ in range(n_layers)])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(hidden)) for _ in range(n_layers)])
        self.decoder = nn.Linear(hidden, out_dim)

    def _compute_basis(self, theta, device):
        """Compute Fourier basis values: [1, cos(θ), sin(θ), cos(2θ), sin(2θ), ...]"""
        basis = [torch.ones_like(theta)]  # isotropic term
        for k in range(1, self.n_freq + 1):
            basis.append(torch.cos(k * theta))
            basis.append(torch.sin(k * theta))
        return torch.stack(basis, dim=-1)  # (n_edges, n_basis)

    def _parallel_transport(self, x_src, cos_a, sin_a):
        """Parallel transport: rotate feature pairs by edge angle."""
        x_t = torch.zeros_like(x_src)
        x_e, x_o = x_src[..., 0::2], x_src[..., 1::2]
        mc = min(x_e.size(-1), x_o.size(-1))
        x_t[..., 0::2][..., :mc] = cos_a * x_e[..., :mc] - sin_a * x_o[..., :mc]
        x_t[..., 1::2][..., :mc] = sin_a * x_e[..., :mc] + cos_a * x_o[..., :mc]
        return x_t

    def forward(self, f0, K):
        x = self.encoder(f0)
        pos = K.pos[:, :2] if K.pos.size(1) >= 2 else K.pos
        src, dst = K.edges[:, 0], K.edges[:, 1]
        edge_vec = pos[dst] - pos[src]
        edge_len = edge_vec.norm(dim=-1, keepdim=True).clamp(1e-8)
        edge_dir = edge_vec / edge_len

        # Edge angles θ for Fourier kernel basis
        theta = torch.atan2(edge_dir[:, 1] if edge_dir.size(1) >= 2 else torch.zeros_like(edge_dir[:, 0]),
                            edge_dir[:, 0])  # (n_edges,)
        basis = self._compute_basis(theta, x.device)  # (n_edges, n_basis)

        cos_a = edge_dir[:, 0:1] if edge_dir.size(1) >= 2 else torch.ones(len(src), 1, device=x.device)
        sin_a = edge_dir[:, 1:2] if edge_dir.size(1) >= 2 else torch.zeros(len(src), 1, device=x.device)

        for i in range(self.n_layers):
            x_self = self.K_self[i](x)
            x_src = x[src]
            x_transported = self._parallel_transport(x_src, cos_a, sin_a)

            # K_neigh(θ) @ x_transported = Σ_k basis_k(θ) * (W_k @ x_transported)
            W = self.K_neigh_weights[i]  # (n_basis, hidden, hidden)
            msg = torch.einsum('eb,bhd,ed->eh', basis, W, x_transported)

            agg = torch.zeros_like(x)
            agg.index_add_(0, dst, msg)
            count = torch.zeros(x.size(0), 1, device=x.device)
            count.index_add_(0, dst, torch.ones(len(dst), 1, device=x.device))
            agg = agg / count.clamp(min=1)

            x = x_self + agg + self.bias[i]

        return self.decoder(x)

    def forward_batch(self, f0, K):
        B = f0.size(0)
        pos = K.pos[:, :2] if K.pos.size(1) >= 2 else K.pos
        src, dst = K.edges[:, 0], K.edges[:, 1]
        edge_vec = pos[dst] - pos[src]
        edge_len = edge_vec.norm(dim=-1, keepdim=True).clamp(1e-8)
        edge_dir = edge_vec / edge_len

        theta = torch.atan2(edge_dir[:, 1] if edge_dir.size(1) >= 2 else torch.zeros_like(edge_dir[:, 0]),
                            edge_dir[:, 0])
        basis = self._compute_basis(theta, f0.device)  # (n_edges, n_basis)
        cos_a = edge_dir[:, 0:1] if edge_dir.size(1) >= 2 else torch.ones(len(src), 1, device=f0.device)
        sin_a = edge_dir[:, 1:2] if edge_dir.size(1) >= 2 else torch.zeros(len(src), 1, device=f0.device)

        x = self.encoder(f0)  # (B, n0, h)
        n0, H = x.size(1), x.size(2)

        count = torch.zeros(n0, 1, device=x.device)
        count.index_add_(0, dst, torch.ones(len(dst), 1, device=x.device))
        count = count.clamp(min=1)

        for i in range(self.n_layers):
            x_self = self.K_self[i](x)  # (B, n0, h)

            x_src = x[:, src]  # (B, n_edges, h)
            x_transported = self._parallel_transport(x_src, cos_a.unsqueeze(0), sin_a.unsqueeze(0))

            W = self.K_neigh_weights[i]  # (n_basis, h, h)
            msg = torch.einsum('eb,bhd,Bed->Beh', basis, W, x_transported)  # (B, n_edges, h)

            offsets = torch.arange(B, device=x.device).unsqueeze(1) * n0
            dst_flat = (dst.unsqueeze(0) + offsets).reshape(-1)
            agg_flat = torch.zeros(B * n0, H, device=x.device)
            agg_flat.index_add_(0, dst_flat, msg.reshape(-1, H))
            agg = agg_flat.reshape(B, n0, H)
            agg = agg / count.unsqueeze(0)

            x = x_self + agg + self.bias[i]

        return self.decoder(x)
