"""
Graph-level baseline models (pure PyTorch, no PyG dependency).

Implements 4 baselines:
  1. GCN  — Isotropic message passing, topology only, no geometry
  2. GAT  — Dynamic attention message passing
  3. SchNet — E(3)-invariant scalar network (RBF continuous filters)
  4. EGNN — E(n)-equivariant message passing

Unified interface: forward(f0, K) -> (n1, 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from gauge_hodge_mp.cell_complex import CellComplex


# ---------------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------------

def scatter_sum(src, index, dim=0, dim_size=None):
    """Pure PyTorch scatter_add, replacing torch_scatter."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    shape = list(src.shape)
    shape[dim] = dim_size
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    idx = index.unsqueeze(-1).expand_as(src) if src.dim() > 1 else index
    return out.scatter_add_(dim, idx, src)


def scatter_mean(src, index, dim=0, dim_size=None):
    """scatter_mean = scatter_sum / count."""
    s = scatter_sum(src, index, dim=dim, dim_size=dim_size)
    count = scatter_sum(torch.ones_like(src), index, dim=dim, dim_size=dim_size)
    return s / count.clamp(min=1)


def scatter_softmax(src, index, dim=0, dim_size=None):
    """Group-wise softmax over index (used for GAT attention)."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    max_per_group = torch.zeros(dim_size, 1, dtype=src.dtype, device=src.device)
    max_per_group.scatter_reduce_(0, index.unsqueeze(-1).expand_as(src), src, reduce='amax',
                                  include_self=True)
    src_stable = src - max_per_group[index]
    exp_src = torch.exp(src_stable)
    exp_sum = scatter_sum(exp_src, index, dim=dim, dim_size=dim_size)
    return exp_src / exp_sum[index].clamp(min=1e-12)


# ---------------------------------------------------------------------------
#  Base class
# ---------------------------------------------------------------------------

class BaselineBase(nn.Module):
    """
    Base class for all baselines.

    Subclasses implement encode(f0, K) -> (n0, hidden) for node encoding;
    the base class handles edge_readout to map node features to edge predictions (n1, 1).
    """

    def __init__(self, hidden: int, out_dim: int = 1, task: str = "edge"):
        super().__init__()
        self.task = task
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden * 2 + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        self.node_mlp = nn.Linear(hidden, out_dim)

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        """Auto-adapt node_mlp structure to match checkpoint."""
        has_2layer = any(k.startswith('node_mlp.0.') for k in state_dict)
        current_2layer = isinstance(self.node_mlp, nn.Sequential)
        if has_2layer and not current_2layer:
            h = state_dict['node_mlp.0.weight'].shape[0]
            od = state_dict['node_mlp.2.weight'].shape[0]
            self.node_mlp = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, od))
        elif not has_2layer and current_2layer:
            od = state_dict['node_mlp.weight'].shape[0]
            h = state_dict['node_mlp.weight'].shape[1]
            self.node_mlp = nn.Linear(h, od)
        super().load_state_dict(state_dict, strict=strict, **kwargs)

    def edge_readout(self, x0: torch.Tensor, K: CellComplex) -> torch.Tensor:
        """Node features (n0, hidden) -> edge predictions (n1, out_dim)."""
        src, dst = K.edges[:, 0], K.edges[:, 1]
        diff = K.pos[dst] - K.pos[src]                     # (n1, dim)
        edge_lengths = diff.norm(dim=-1, keepdim=True)      # (n1, 1)
        edge_feat = torch.cat([x0[src], x0[dst], edge_lengths], dim=-1)
        return self.edge_mlp(edge_feat)

    def node_readout(self, x0: torch.Tensor) -> torch.Tensor:
        """Node features (n0, hidden) -> node predictions (n0, out_dim)."""
        return self.node_mlp(x0)

    def encode(self, f0: torch.Tensor, K: CellComplex) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, f0: torch.Tensor, K: CellComplex) -> torch.Tensor:
        """Unified interface: task="edge" -> (n1, out_dim), task="node" -> (n0, out_dim)."""
        x0 = self.encode(f0, K)
        if self.task == "node":
            return self.node_readout(x0)
        return self.edge_readout(x0, K)

    def forward_batch(self, f0_batch: torch.Tensor, K: CellComplex) -> torch.Tensor:
        """Batched forward. Default per-sample loop; subclasses may override."""
        B = f0_batch.size(0)
        return torch.stack([self.forward(f0_batch[i], K) for i in range(B)], dim=0)


# ===========================================================================
#  1. GCN — Graph Convolutional Network
# ===========================================================================

class GCNBaseline(BaselineBase):
    """
    GCN (Kipf & Welling, 2017).

    Symmetries:
      - Permutation equivariant (node reordering)
      - No rotation/translation invariance (topology only, no coordinates)
      - No higher-order structure (ignores faces / 2-cells)
      - Fixed aggregation weights (degree-normalized, not learnable)

    h' = sigma(D^{-1/2} A D^{-1/2} h W)
    """

    def __init__(self, f_in: int, hidden: int = 32, n_layers: int = 4, out_dim: int = 1, task: str = "edge"):
        super().__init__(hidden, out_dim, task=task)
        self.layers = nn.ModuleList()
        dims = [f_in] + [hidden] * n_layers
        for i in range(n_layers):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

    def _build_norm_adj(self, K: CellComplex) -> torch.Tensor:
        """Build symmetric normalized adjacency: A_hat = D^{-1/2}(A+I)D^{-1/2}."""
        n = K.n0
        device = K.pos.device
        A = torch.zeros(n, n, device=device)
        src, dst = K.edges[:, 0], K.edges[:, 1]
        A[src, dst] = 1.0
        A[dst, src] = 1.0
        A = A + torch.eye(n, device=device)
        deg = A.sum(dim=1)  # (n,)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
        # D^{-1/2} A D^{-1/2}
        D = torch.diag(deg_inv_sqrt)
        A_hat = D @ A @ D
        return A_hat

    def encode(self, f0: torch.Tensor, K: CellComplex) -> torch.Tensor:
        A_hat = self._build_norm_adj(K)
        h = f0
        for i, lin in enumerate(self.layers):
            h = A_hat @ h
            h = lin(h)
            if i < len(self.layers) - 1:
                h = F.silu(h)
        return h

    def forward_batch(self, f0_batch: torch.Tensor, K: CellComplex) -> torch.Tensor:
        """GCN encode natively supports batching (A_hat @ h broadcasts, nn.Linear broadcasts)."""
        x0 = self.encode(f0_batch, K)  # (B, n0, hidden)
        if self.task == "node":
            return self.node_mlp(x0)
        B = f0_batch.size(0)
        return torch.stack([self.edge_readout(x0[i], K) for i in range(B)], dim=0)


# ===========================================================================
#  2. GAT — Graph Attention Network
# ===========================================================================

class GATLayer(nn.Module):
    """Single GAT layer (Velickovic et al., 2018) with multi-head attention."""

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4,
                 negative_slope: float = 0.2):
        super().__init__()
        assert out_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        # Attention parameters a: per-head 2*head_dim -> 1
        self.attn_src = nn.Parameter(torch.empty(n_heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.empty(n_heads, self.head_dim))
        self.negative_slope = negative_slope
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.attn_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.attn_dst.unsqueeze(0))

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
        """
        Args:
            h: (n, in_dim)
            edge_index: (2, E) — bidirectional edges
            n_nodes: number of nodes
        Returns:
            (n, out_dim)
        """
        src_idx, dst_idx = edge_index[0], edge_index[1]
        Wh = self.W(h).view(-1, self.n_heads, self.head_dim)  # (n, H, D)

        # Attention scores: a_src^T Wh_i + a_dst^T Wh_j
        e_src = (Wh[src_idx] * self.attn_src).sum(-1)  # (E, H)
        e_dst = (Wh[dst_idx] * self.attn_dst).sum(-1)  # (E, H)
        e = F.leaky_relu(e_src + e_dst, negative_slope=self.negative_slope)  # (E, H)

        # Per-group softmax with numerical stabilization
        e_max = torch.zeros(n_nodes, self.n_heads, device=h.device)
        e_max.scatter_reduce_(0, dst_idx.unsqueeze(-1).expand_as(e), e,
                              reduce='amax', include_self=True)
        e_stable = e - e_max[dst_idx]
        exp_e = torch.exp(e_stable)
        exp_sum = scatter_sum(exp_e, dst_idx, dim=0, dim_size=n_nodes)  # (n, H)
        alpha = exp_e / exp_sum[dst_idx].clamp(min=1e-12)  # (E, H)

        # Weighted aggregation
        msg = alpha.unsqueeze(-1) * Wh[src_idx]  # (E, H, D)
        msg_flat = msg.view(msg.size(0), -1)       # (E, out_dim)
        out = scatter_sum(msg_flat, dst_idx, dim=0, dim_size=n_nodes)
        return out


class GATBaseline(BaselineBase):
    """
    GAT (Velickovic et al., 2018).

    Symmetries:
      - Permutation equivariant (node reordering)
      - No rotation/translation invariance (topology only, no coordinates)
      - No higher-order structure
      - Learnable dynamic attention weights (vs. GCN's fixed weights)
    """

    def __init__(self, f_in: int, hidden: int = 32, n_layers: int = 4,
                 n_heads: int = 4, out_dim: int = 1, task: str = "edge"):
        super().__init__(hidden, out_dim, task=task)
        self.layers = nn.ModuleList()
        dims = [f_in] + [hidden] * n_layers
        for i in range(n_layers):
            self.layers.append(GATLayer(dims[i], dims[i + 1], n_heads=n_heads))
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])

    def encode(self, f0: torch.Tensor, K: CellComplex) -> torch.Tensor:
        src, dst = K.edges[:, 0], K.edges[:, 1]
        edge_index = torch.stack([
            torch.cat([src, dst]),
            torch.cat([dst, src])
        ], dim=0)  # (2, 2*n1)

        h = f0
        for i, (gat, norm) in enumerate(zip(self.layers, self.norms)):
            h_new = gat(h, edge_index, K.n0)
            if i < len(self.layers) - 1:
                h_new = F.silu(norm(h_new))
            if h.size(-1) == h_new.size(-1):
                h = h + h_new
            else:
                h = h_new
        return h


# ===========================================================================
#  3. SchNet — Continuous-filter convolutional network
# ===========================================================================

class GaussianRBF(nn.Module):
    """Gaussian radial basis functions: phi_k(d) = exp(-gamma (d - mu_k)^2)."""

    def __init__(self, n_rbf: int = 16, cutoff: float = 5.0):
        super().__init__()
        self.register_buffer('centers', torch.linspace(0.0, cutoff, n_rbf))
        self.gamma = 0.5 / ((cutoff / n_rbf) ** 2)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """dist: (...,) -> (..., n_rbf)"""
        return torch.exp(-self.gamma * (dist.unsqueeze(-1) - self.centers) ** 2)


class SchNetInteraction(nn.Module):
    """SchNet cfconv interaction layer."""

    def __init__(self, hidden: int, n_rbf: int):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.lin = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, rbf: torch.Tensor,
                src_idx: torch.Tensor, dst_idx: torch.Tensor,
                n_nodes: int) -> torch.Tensor:
        """
        h: (n, hidden)
        rbf: (E, n_rbf) — RBF features per edge (bidirectional)
        src_idx, dst_idx: (E,)
        """
        W = self.filter_net(rbf)          # (E, hidden) — continuous filter
        msg = W * h[src_idx]              # (E, hidden)
        agg = scatter_sum(msg, dst_idx, dim=0, dim_size=n_nodes)
        return self.norm(h + self.lin(agg))


class SchNetBaseline(BaselineBase):
    """
    SchNet (Schutt et al., 2017).

    Symmetries:
      - E(3)-invariant (uses only scalar distances ||r_i - r_j||)
      - Permutation equivariant
      - No higher-order structure
      - Continuous filters: edge weights generated from RBF-expanded distances
    """

    def __init__(self, f_in: int, hidden: int = 32, n_layers: int = 4,
                 n_rbf: int = 16, cutoff: float = 5.0, out_dim: int = 1, task: str = "edge"):
        super().__init__(hidden, out_dim, task=task)
        self.embed = nn.Linear(f_in, hidden)
        self.rbf = GaussianRBF(n_rbf, cutoff)
        self.interactions = nn.ModuleList([
            SchNetInteraction(hidden, n_rbf) for _ in range(n_layers)
        ])

    def encode(self, f0: torch.Tensor, K: CellComplex) -> torch.Tensor:
        src_fwd, dst_fwd = K.edges[:, 0], K.edges[:, 1]
        src_idx = torch.cat([src_fwd, dst_fwd])
        dst_idx = torch.cat([dst_fwd, src_fwd])

        diff = K.pos[dst_idx] - K.pos[src_idx]
        dist = diff.norm(dim=-1)  # (2*n1,)
        rbf = self.rbf(dist)      # (2*n1, n_rbf)

        h = self.embed(f0)
        for interaction in self.interactions:
            h = interaction(h, rbf, src_idx, dst_idx, K.n0)
        return h


# ===========================================================================
#  4. EGNN — E(n) Equivariant Graph Neural Network
# ===========================================================================

class EGNNLayer(nn.Module):
    """
    Single EGNN layer (Satorras et al., ICML 2021) with LayerNorm.
    Does not update coordinates.
    """

    def __init__(self, hidden: int):
        super().__init__()
        # phi_e: message MLP
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden * 2 + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        # phi_h: node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, pos, src_idx, dst_idx, n_nodes):
        diff = pos[dst_idx] - pos[src_idx]
        dist_sq = (diff ** 2).sum(dim=-1, keepdim=True)

        msg = self.msg_mlp(torch.cat([h[src_idx], h[dst_idx], dist_sq], dim=-1))
        agg = scatter_sum(msg, dst_idx, dim=0, dim_size=n_nodes)

        return self.norm(h + self.node_mlp(torch.cat([h, agg], dim=-1)))


class EGNNBaseline(BaselineBase):
    """
    EGNN (Satorras et al., 2021).

    Symmetries:
      - E(n)-equivariant (uses relative coordinates / squared distances)
      - Permutation equivariant
      - No higher-order structure
      - Does not update coordinates (scalar feature updates only)
      - Compared to SchNet: messages include h_i, h_j (not just distances)
    """

    def __init__(self, f_in: int, hidden: int = 32, n_layers: int = 4, out_dim: int = 1, task: str = "edge"):
        super().__init__(hidden, out_dim, task=task)
        self.embed = nn.Linear(f_in, hidden)
        self.layers = nn.ModuleList([EGNNLayer(hidden) for _ in range(n_layers)])

    def encode(self, f0: torch.Tensor, K: CellComplex) -> torch.Tensor:
        src_fwd, dst_fwd = K.edges[:, 0], K.edges[:, 1]
        src_idx = torch.cat([src_fwd, dst_fwd])
        dst_idx = torch.cat([dst_fwd, src_fwd])

        h = self.embed(f0)
        for layer in self.layers:
            h = layer(h, K.pos, src_idx, dst_idx, K.n0)
        return h


# ---------------------------------------------------------------------------
#  Factory function
# ---------------------------------------------------------------------------

BASELINES = {
    'gcn': GCNBaseline,
    'gat': GATBaseline,
    'schnet': SchNetBaseline,
    'egnn': EGNNBaseline,
}


def build_baseline(name: str, f_in: int, **kwargs) -> BaselineBase:
    """Create a baseline model by name.

    Args:
        name: 'gcn' | 'gat' | 'schnet' | 'egnn'
        f_in: input feature dimension
        **kwargs: additional arguments passed to the model constructor

    Returns:
        BaselineBase subclass instance
    """
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline: {name}. Choose from {list(BASELINES.keys())}")
    return BASELINES[name](f_in=f_in, **kwargs)
