"""
Topological Baseline Models (pure PyTorch, no PyG)

Three message-passing baselines on simplicial complexes:
  5. MPSNBaseline   — Message Passing Simplicial Networks
  6. SCCNNBaseline  — Simplicial Complex CNN (fixed Hodge Laplacian)
  7. HodgeAwareBaseline — Hodge-Aware GNN (separate up/down Laplacian + learnable mixing)

Unified interface: model.forward(f0, K) -> (n1, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from gauge_hodge_mp.utils import spmm, scatter_sum, batch_spmm


# ============================================================
# Helper: Feature Lifting (node features -> 0/1/2-cochain initial features)
# ============================================================

class _SimpleLift(nn.Module):
    """Simple lifting: f0 -> x0, x1, x2 initial features."""

    def __init__(self, f_in: int, hidden: int):
        super().__init__()
        self.lift0 = nn.Sequential(nn.Linear(f_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.lift1 = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.lift2 = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, f0: torch.Tensor, K):
        """
        Args:
            f0: (n0, f_in)
            K: CellComplex
        Returns:
            x0 (n0, hidden), x1 (n1, hidden), x2 (n2, hidden)
        """
        x0 = self.lift0(f0)

        # 1-cochain: edge endpoint average
        src_feat = x0[K.edges[:, 0]]
        dst_feat = x0[K.edges[:, 1]]
        x1 = self.lift1(0.5 * (src_feat + dst_feat))

        # 2-cochain: aggregate edge features to faces via |d1|
        d1_abs = _sparse_abs(K.d1)  # (n2, n1), entries in {0, 1}
        face_feat_sum = spmm(d1_abs, x1)
        x2 = self.lift2(face_feat_sum / 3.0)  # each face has 3 edges

        return x0, x1, x2

    def forward_batch(self, f0: torch.Tensor, K):
        """Batched: f0 (B, n0, f_in) -> x0 (B, n0, h), x1 (B, n1, h), x2 (B, n2, h)"""
        x0 = self.lift0(f0)

        src_feat = x0[:, K.edges[:, 0]]
        dst_feat = x0[:, K.edges[:, 1]]
        x1 = self.lift1(0.5 * (src_feat + dst_feat))

        d1_abs = _sparse_abs(K.d1)
        face_feat_sum = batch_spmm(d1_abs, x1)
        x2 = self.lift2(face_feat_sum / 3.0)

        return x0, x1, x2


# ============================================================
# Helper Functions
# ============================================================

def _sparse_abs(sp: torch.Tensor) -> torch.Tensor:
    """Element-wise absolute value of a sparse tensor (preserves sparse format)."""
    sp = sp.coalesce()
    return torch.sparse_coo_tensor(sp.indices(), sp.values().abs(), sp.size(),
                                   device=sp.device, dtype=sp.dtype).coalesce()


def _sparse_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Sparse @ sparse -> dense -> re-sparsify. Feasible at small scale."""
    dense = torch.sparse.mm(a, b)
    if dense.is_sparse:
        return dense.coalesce()
    return dense.to_sparse_coo().coalesce()


def _remove_self_loops_sparse(sp: torch.Tensor) -> torch.Tensor:
    """Remove diagonal elements from a sparse matrix."""
    sp = sp.coalesce()
    idx = sp.indices()
    vals = sp.values()
    mask = idx[0] != idx[1]
    new_idx = idx[:, mask]
    new_vals = vals[mask]
    return torch.sparse_coo_tensor(new_idx, new_vals, sp.size(),
                                   device=sp.device, dtype=sp.dtype).coalesce()


def _binary_sparse(sp: torch.Tensor) -> torch.Tensor:
    """Binarize a sparse matrix (set all nonzero values to 1)."""
    sp = sp.coalesce()
    vals = torch.ones_like(sp.values())
    return torch.sparse_coo_tensor(sp.indices(), vals, sp.size(),
                                   device=sp.device, dtype=sp.dtype).coalesce()


# ============================================================
# 5. MPSN (Message Passing Simplicial Networks)
# ============================================================

class MPSNBaseline(nn.Module):
    """
    Message Passing Simplicial Networks baseline.

    Independent GCN-like message passing on 0/1/2-cochains using
    adjacency matrices (not coboundary d_k) to define neighborhoods.
    No cross-dimension message passing.

    Adjacency relations:
      A_00 = |d0|^T @ |d0|  (nodes sharing an edge)
      A_11_lower = |d0| @ |d0|^T  (edges sharing a node)
      A_11_upper = |d1|^T @ |d1|  (edges sharing a face)
      A_22 = |d1| @ |d1|^T  (faces sharing an edge)

    Symmetries:
      x E(n)  — no geometric invariants used
      x O(C)  — linear layers mix channels
      x Z_2   — uses |d_k|, discards sign information, not orientation-equivariant
      x d^2=0 — does not use coboundary operators
    """

    def __init__(self, f_in: int, hidden: int = 32, n_layers: int = 4):
        super().__init__()
        self.lift = _SimpleLift(f_in, hidden)

        self.layers_0 = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.layers_1 = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.layers_2 = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])

        self.norms_0 = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.norms_1 = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.norms_2 = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])

        self.n_layers = n_layers
        self.readout = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def _build_adjacency(self, K):
        """Precompute adjacency matrices (binarized, self-loops removed)."""
        d0_abs = _sparse_abs(K.d0)  # (n1, n0)
        d1_abs = _sparse_abs(K.d1)  # (n2, n1)

        # A_00 = |d0|^T @ |d0|, remove diagonal
        A_00 = _remove_self_loops_sparse(_binary_sparse(
            _sparse_matmul(d0_abs.t(), d0_abs)))

        # A_11_lower = |d0| @ |d0|^T (edges sharing a node)
        A_11_lo = _remove_self_loops_sparse(_binary_sparse(
            _sparse_matmul(d0_abs, d0_abs.t())))

        # A_11_upper = |d1|^T @ |d1| (edges sharing a face)
        A_11_up = _remove_self_loops_sparse(_binary_sparse(
            _sparse_matmul(d1_abs.t(), d1_abs)))

        # A_22 = |d1| @ |d1|^T (faces sharing an edge)
        A_22 = _remove_self_loops_sparse(_binary_sparse(
            _sparse_matmul(d1_abs, d1_abs.t())))

        return A_00, A_11_lo, A_11_up, A_22

    def forward(self, f0: torch.Tensor, K) -> torch.Tensor:
        """
        Args:
            f0: (n0, f_in) node features
            K: CellComplex
        Returns:
            (n1, 1) edge-level predictions
        """
        x0, x1, x2 = self.lift(f0, K)
        A_00, A_11_lo, A_11_up, A_22 = self._build_adjacency(K)

        for i in range(self.n_layers):
            # 0-cochain: x0' = sigma(A_00 x0 W0)
            x0_new = spmm(A_00, x0)
            x0_new = self.layers_0[i](x0_new)
            x0 = F.silu(self.norms_0[i](x0_new + x0))

            # 1-cochain: x1' = sigma((A_11_lo + A_11_up) x1 W1)
            x1_new = spmm(A_11_lo, x1) + spmm(A_11_up, x1)
            x1_new = self.layers_1[i](x1_new)
            x1 = F.silu(self.norms_1[i](x1_new + x1))

            # 2-cochain: x2' = sigma(A_22 x2 W2)
            x2_new = spmm(A_22, x2)
            x2_new = self.layers_2[i](x2_new)
            x2 = F.silu(self.norms_2[i](x2_new + x2))

        return self.readout(x1)

    def forward_batch(self, f0: torch.Tensor, K) -> torch.Tensor:
        """Batched: f0 (B, n0, f_in) -> (B, n1, 1)"""
        x0, x1, x2 = self.lift.forward_batch(f0, K)
        A_00, A_11_lo, A_11_up, A_22 = self._build_adjacency(K)
        for i in range(self.n_layers):
            x0_new = self.layers_0[i](batch_spmm(A_00, x0))
            x0 = F.silu(self.norms_0[i](x0_new + x0))
            x1_new = self.layers_1[i](batch_spmm(A_11_lo, x1) + batch_spmm(A_11_up, x1))
            x1 = F.silu(self.norms_1[i](x1_new + x1))
            x2_new = self.layers_2[i](batch_spmm(A_22, x2))
            x2 = F.silu(self.norms_2[i](x2_new + x2))
        return self.readout(x1)


# ============================================================
# 6. SCCNN (Simplicial Complex CNN)
# ============================================================

class SCCNNBaseline(nn.Module):
    """
    Faithful SCCNN baseline (Yang et al., ICASSP 2022).

    Polynomial filter on combined Hodge Laplacian:
      H_k(L_k) = sum_t w_{k,t} L_k^t
    where w_{k,t} are scalar filter coefficients, L_k = L_{k,down} + L_{k,up}.

    Single Linear layer per k-cochain for channel mixing after filtering.
    No LayerNorm, no per-direction weight matrices.
    """

    def __init__(self, f_in: int, hidden: int = 32, n_layers: int = 4,
                 T: int = 3):
        super().__init__()
        self.lift = _SimpleLift(f_in, hidden)
        self.n_layers = n_layers
        self.T = T  # polynomial order

        # Scalar polynomial coefficients per layer per k-cochain
        self.w_0 = nn.ParameterList([nn.Parameter(torch.randn(T + 1) * 0.1) for _ in range(n_layers)])
        self.w_1 = nn.ParameterList([nn.Parameter(torch.randn(T + 1) * 0.1) for _ in range(n_layers)])
        self.w_2 = nn.ParameterList([nn.Parameter(torch.randn(T + 1) * 0.1) for _ in range(n_layers)])

        # Channel mixing (single Linear per layer per k)
        self.mix_0 = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.mix_1 = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.mix_2 = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])

        self.readout = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    @staticmethod
    def _build_laplacians(K):
        d0, d1 = K.d0, K.d1
        def _ensure_sparse(x):
            return x.coalesce() if x.is_sparse else x.to_sparse_coo().coalesce()
        L0 = _ensure_sparse(_sparse_matmul(d0.t(), d0))
        L1_lo = _sparse_matmul(d0, d0.t())
        L1_up = _sparse_matmul(d1.t(), d1)
        L1 = (L1_lo.to_dense() + L1_up.to_dense()).to_sparse_coo().coalesce()
        L2 = _ensure_sparse(_sparse_matmul(d1, d1.t()))
        return L0, L1, L2

    def _poly_filter_single(self, x, L, w):
        """H(L)x = w[0]*x + w[1]*Lx + w[2]*L^2*x + ..."""
        out = w[0] * x
        Lx = x
        for t in range(1, len(w)):
            Lx = spmm(L, Lx)
            out = out + w[t] * Lx
        return out

    def _poly_filter_batch(self, x, L, w):
        out = w[0] * x
        Lx = x
        for t in range(1, len(w)):
            Lx = batch_spmm(L, Lx)
            out = out + w[t] * Lx
        return out

    def forward(self, f0: torch.Tensor, K) -> torch.Tensor:
        x0, x1, x2 = self.lift(f0, K)
        L0, L1, L2 = self._build_laplacians(K)
        for i in range(self.n_layers):
            x0 = F.silu(self.mix_0[i](self._poly_filter_single(x0, L0, self.w_0[i])))
            x1 = F.silu(self.mix_1[i](self._poly_filter_single(x1, L1, self.w_1[i])))
            x2 = F.silu(self.mix_2[i](self._poly_filter_single(x2, L2, self.w_2[i])))
        return self.readout(x1)

    def forward_batch(self, f0: torch.Tensor, K) -> torch.Tensor:
        x0, x1, x2 = self.lift.forward_batch(f0, K)
        L0, L1, L2 = self._build_laplacians(K)
        for i in range(self.n_layers):
            x0 = F.silu(self.mix_0[i](self._poly_filter_batch(x0, L0, self.w_0[i])))
            x1 = F.silu(self.mix_1[i](self._poly_filter_batch(x1, L1, self.w_1[i])))
            x2 = F.silu(self.mix_2[i](self._poly_filter_batch(x2, L2, self.w_2[i])))
        return self.readout(x1)


# ============================================================
# 7. SAN — Simplicial Attention Network (Giusti et al., 2022/2024)
# ============================================================

class SANBaseline(nn.Module):
    """
    Simplicial Attention Network (SAN) baseline.
      Giusti et al., "Simplicial Attention Neural Networks" (2022, arXiv:2203.07485)
      + Generalized version: Giusti et al. (2024, arXiv:2309.02138)

    Masked self-attention on simplicial complexes.
    For each k-simplex, attends to upper and lower neighbors
    with learned attention weights.

    Key: attention-based (like GAT on simplices), NOT spectral/polynomial.
    Simpler than Hodge decomposition, no separated Laplacians.
    """

    def __init__(self, f_in: int, hidden: int = 32, n_layers: int = 4):
        super().__init__()
        self.lift = _SimpleLift(f_in, hidden)
        self.n_layers = n_layers

        # Additive attention a^T [Wq x_i || Wk x_j] (GATv2-style)
        self.W_q_down = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.W_k_down = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.a_down = nn.ParameterList([nn.Parameter(torch.randn(hidden)) for _ in range(n_layers)])
        self.W_q_up = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.W_k_up = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.a_up = nn.ParameterList([nn.Parameter(torch.randn(hidden)) for _ in range(n_layers)])
        self.W_val = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.W_self = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])

        self.readout = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    @staticmethod
    def _build_adj(K):
        d0, d1 = K.d0, K.d1
        A_down = _sparse_matmul(d0, d0.t()).coalesce()
        A_up = _sparse_matmul(d1.t(), d1).coalesce()
        return A_down, A_up

    def _attn_msg(self, x, adj, W_q, W_k, a_vec, W_val):
        """Additive attention: score = a^T LeakyReLU(Wq x_i + Wk x_j)"""
        idx = adj.indices()
        src, dst = idx[0], idx[1]
        q = W_q(x)
        k = W_k(x)
        scores = (F.leaky_relu(q[dst] + k[src], 0.2) * a_vec).sum(-1)
        # Softmax per destination node
        scores_max = torch.zeros(x.size(0), device=x.device)
        scores_max.scatter_reduce_(0, dst, scores, reduce='amax', include_self=False)
        alpha = torch.exp(scores - scores_max[dst])
        denom = torch.zeros(x.size(0), device=x.device)
        denom.scatter_add_(0, dst, alpha)
        alpha = alpha / (denom[dst] + 1e-8)
        # Aggregate
        val = W_val(x[src]) * alpha.unsqueeze(-1)
        out = torch.zeros_like(x)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(val), val)
        return out

    def forward(self, f0: torch.Tensor, K) -> torch.Tensor:
        x0, x1, x2 = self.lift(f0, K)
        if not hasattr(self, '_A_d') or self._cached_dev != K.d0.device:
            self._A_d, self._A_u = self._build_adj(K)
            self._cached_dev = K.d0.device

        for i in range(self.n_layers):
            md = self._attn_msg(x1, self._A_d, self.W_q_down[i], self.W_k_down[i], self.a_down[i], self.W_val[i])
            mu = self._attn_msg(x1, self._A_u, self.W_q_up[i], self.W_k_up[i], self.a_up[i], self.W_val[i])
            x1 = F.silu(self.W_self[i](x1) + md + mu)

        return self.readout(x1)

    def forward_batch(self, f0: torch.Tensor, K) -> torch.Tensor:
        x0, x1, x2 = self.lift.forward_batch(f0, K)
        if not hasattr(self, '_A_d') or self._cached_dev != K.d0.device:
            self._A_d, self._A_u = self._build_adj(K)
            self._cached_dev = K.d0.device
        B, n1, H = x1.shape

        for i in range(self.n_layers):
            # Flatten (B, n1, H) -> (B*n1, H) with offset indices for batched attention
            idx_d = self._A_d.indices()
            idx_u = self._A_u.indices()
            x_flat = x1.reshape(B * n1, H)
            offsets = torch.arange(B, device=x1.device).unsqueeze(1) * n1

            out_d = torch.zeros_like(x_flat)
            out_u = torch.zeros_like(x_flat)
            q_d = self.W_q_down[i](x_flat); k_d = self.W_k_down[i](x_flat)
            q_u = self.W_q_up[i](x_flat); k_u = self.W_k_up[i](x_flat)
            val_all = self.W_val[i](x_flat)

            for adj_idx, q, k, a_vec, out in [
                (idx_d, q_d, k_d, self.a_down[i], out_d),
                (idx_u, q_u, k_u, self.a_up[i], out_u)]:
                s0, d0_ = adj_idx[0], adj_idx[1]
                s_exp = (s0.unsqueeze(0) + offsets).reshape(-1)
                d_exp = (d0_.unsqueeze(0) + offsets).reshape(-1)
                scores = (F.leaky_relu(q[d_exp] + k[s_exp], 0.2) * a_vec).sum(-1)
                sm = torch.zeros(B * n1, device=x1.device)
                sm.scatter_reduce_(0, d_exp, scores, reduce='amax', include_self=False)
                alpha = torch.exp(scores - sm[d_exp])
                dn = torch.zeros(B * n1, device=x1.device)
                dn.scatter_add_(0, d_exp, alpha)
                alpha = alpha / (dn[d_exp] + 1e-8)
                v = val_all[s_exp] * alpha.unsqueeze(-1)
                out.scatter_add_(0, d_exp.unsqueeze(-1).expand_as(v), v)

            x1 = F.silu(self.W_self[i](x_flat) + out_d + out_u).reshape(B, n1, H)

        return self.readout(x1)
