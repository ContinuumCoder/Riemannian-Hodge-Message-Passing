"""
GaugeHodgeNetwork -- complete network.

Three-stage architecture:
  Stage 1: Lifting encoder -- E(n)-invariants -> cochain features
  Stage 2: Hodge MP x L layers -- d^2=0 + O(C) + E(n)
  Stage 3: Readout/reconstruction -- scalar (invariant) or vector (equivariant)

Full symmetry group: G = E(n) x O(C) x {d^2=0}
"""

import torch
import torch.nn as nn
from .cell_complex import CellComplex
from .lifting import InvariantLifting
from .hodge_mp import HodgeMPLayer
from .reconstruction import EquivariantReconstruction, ScalarReadout


class GaugeHodgeNetwork(nn.Module):
    """
    Gauge-Structured Hodge Message Passing Network.

    Simultaneously enforces:
    1. d^2=0 (Hodge structure) -- d_k fixed as CW complex coboundary
    2. O(C) fibre gauge equivariance -- spatial metric H + invariant inputs
    3. E(n) spatial invariance/equivariance -- cochain representation + geometric reconstruction
    4. SPD positive-definite metric -- Cholesky parameterisation
    """

    def __init__(self, f_in: int, C: int, n_layers: int,
                 n0: int, n1: int, n2: int,
                 task: str = "scalar",
                 spatial_dim: int = 3,
                 out_dim: int = 1,
                 mp_hidden: int = 64,
                 metric_type: str = "diagonal",
                 metric_rank: int = 0,
                 identity_metric: bool = False,
                 no_cross: bool = False,
                 relu_gate: bool = False,
                 learned_d: bool = False):
        """
        Args:
            f_in: raw node feature dimension
            C: cochain feature dimension
            n_layers: number of Hodge MP layers
            n0, n1, n2: vertex / edge / face counts (for metric sizing)
            task: "scalar", "edge_scalar", or "vector"
            spatial_dim: spatial dimension
            out_dim: output dimension
            mp_hidden: MP-layer MLP hidden dimension
            metric_type: "diagonal" (O(n)) or "full" (SPD, O(n^2))
            metric_rank: 0 = standard, >0 = low-rank metric (rank-k basis expansion)
        """
        super().__init__()
        self.task = task
        self.metric_type = metric_type
        self.n_layers = n_layers
        self.identity_metric = identity_metric
        self.no_cross = no_cross
        self.relu_gate = relu_gate
        self.learned_d = learned_d
        # Learnable coboundary values (init at +/-1 matching the fixed d); only
        # the values on existing sparsity pattern are learnable -- shape stays the same.
        # Created lazily on first forward when we see K.
        self._learned_d0_vals = None
        self._learned_d1_vals = None

        # Stage 1: Lifting encoder
        self.lifting = InvariantLifting(f_in, C)

        # Stage 2: Hodge MP layers
        geo_dims = {'mp0_up': 5, 'mp1_low': 3, 'mp1_up': 3, 'mp2_low': 5} if metric_type == "local_rich" else {}
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.ModuleDict({
                "mp0": HodgeMPLayer(C, n_lower=0, n_upper=n1,
                                    hidden=mp_hidden, metric_type=metric_type,
                                    metric_rank=metric_rank,
                                    geo_dim=geo_dims.get('mp0_up', 2),
                                    identity_metric=identity_metric,
                                    no_cross=no_cross, relu_gate=relu_gate),
                "mp1": HodgeMPLayer(C, n_lower=n0, n_upper=n2,
                                    hidden=mp_hidden, metric_type=metric_type,
                                    metric_rank=metric_rank,
                                    geo_dim=max(geo_dims.get('mp1_low', 2), geo_dims.get('mp1_up', 2)),
                                    identity_metric=identity_metric,
                                    no_cross=no_cross, relu_gate=relu_gate),
                "mp2": HodgeMPLayer(C, n_lower=n1, n_upper=0,
                                    hidden=mp_hidden, metric_type=metric_type,
                                    metric_rank=metric_rank,
                                    geo_dim=geo_dims.get('mp2_low', 2),
                                    identity_metric=identity_metric,
                                    no_cross=no_cross, relu_gate=relu_gate),
            })
            self.layers.append(layer)

        # LayerNorm to prevent feature explosion across MP layers
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.norms.append(nn.ModuleDict({
                "ln0": nn.LayerNorm(C),
                "ln1": nn.LayerNorm(C),
                "ln2": nn.LayerNorm(C),
            }))

        # Learnable coboundary parameters (learned_d ablation).
        # We keep the sparsity pattern of d0/d1 but let the +/-1 entries be
        # learnable scalars -- this breaks d^2 = 0 in general while preserving
        # the same edge/face incidence structure.
        if learned_d:
            # Lazy init: sizes depend on K, not available here.
            self._d0_val_count = None
            self._d1_val_count = None

        # Stage 3: Readout
        if task == "scalar":
            self.readout = ScalarReadout(C, out_dim)
        elif task == "edge_scalar":
            self.readout = ScalarReadout(C, out_dim)
        elif task == "vector":
            self.readout = EquivariantReconstruction(C, spatial_dim)
        else:
            raise ValueError(f"Unknown task: {task}. Use 'scalar', 'edge_scalar', or 'vector'.")

    def _maybe_learned_d(self, K: CellComplex):
        """Return (d0, d1) to use. If learned_d, build sparse tensors with
        learnable values on the same sparsity pattern as K.d0 / K.d1."""
        if not self.learned_d:
            return K.d0, K.d1

        d0_c = K.d0.coalesce()
        d1_c = K.d1.coalesce()
        dev = d0_c.device

        if self._learned_d0_vals is None or self._learned_d0_vals.numel() != d0_c.values().numel():
            init_d0 = d0_c.values().detach().clone()
            self._learned_d0_vals = torch.nn.Parameter(init_d0.to(dev))
            self.register_parameter("learned_d0_vals", self._learned_d0_vals)
        if self._learned_d1_vals is None or self._learned_d1_vals.numel() != d1_c.values().numel():
            init_d1 = d1_c.values().detach().clone()
            self._learned_d1_vals = torch.nn.Parameter(init_d1.to(dev))
            self.register_parameter("learned_d1_vals", self._learned_d1_vals)

        d0_learn = torch.sparse_coo_tensor(
            d0_c.indices(), self._learned_d0_vals, size=d0_c.shape).coalesce()
        d1_learn = torch.sparse_coo_tensor(
            d1_c.indices(), self._learned_d1_vals, size=d1_c.shape).coalesce()
        return d0_learn, d1_learn

    def forward(self, f0: torch.Tensor, K: CellComplex) -> torch.Tensor:
        """
        Args:
            f0: (n0, f_in) raw node features
            K: CellComplex instance (with pos, edges, faces, d0, d1)

        Returns:
            Scalar task: (n0, out_dim)
            Vector task: (n0, spatial_dim)
        """
        adj_00 = K.get_adjacency(0, 0)
        d0, d1 = self._maybe_learned_d(K)

        # Per-cell geometric invariants (E(n)-invariant, for local metric)
        from .utils import compute_edge_lengths, compute_node_degrees, compute_face_areas
        geo_0 = compute_node_degrees(K.edges, K.n0).unsqueeze(-1)
        geo_1 = compute_edge_lengths(K.pos, K.edges).unsqueeze(-1)
        geo_2 = compute_face_areas(K.pos, K.faces).unsqueeze(-1)

        if self.metric_type == "local_rich":
            edge_mid = (K.pos[K.edges[:, 0]] + K.pos[K.edges[:, 1]]) / 2
            edge_dir = K.pos[K.edges[:, 1]] - K.pos[K.edges[:, 0]]
            edge_dir_norm = edge_dir / (edge_dir.norm(dim=-1, keepdim=True) + 1e-8)
            geo_1 = torch.cat([geo_1, edge_mid[:, :2], edge_dir_norm[:, :2]], dim=-1)  # (n1, 5)
            geo_0 = torch.cat([geo_0, K.pos[:, :2]], dim=-1)  # (n0, 3)
            face_centroid = K.pos[K.faces].mean(dim=1)
            geo_2 = torch.cat([geo_2, face_centroid[:, :2]], dim=-1)  # (n2, 3)

        # Stage 1: Lifting
        x0, x1, x2 = self.lifting(f0, K)

        # Stage 2: Hodge MP x L layers with cross-dimension transfer
        for layer, norm in zip(self.layers, self.norms):
            x0_in, x1_in, x2_in = x0, x1, x2

            # 0-cochain: upper adjacency via d0 (Hodge + cross from x1)
            x0 = norm["ln0"](layer["mp0"](x0_in, d_lower=None, d_upper=d0,
                              adj=adj_00, x_lower=None, x_upper=x1_in,
                              geo_lower=None, geo_upper=geo_1))
            # 1-cochain: lower via d0 (from x0), upper via d1 (from x2)
            x1 = norm["ln1"](layer["mp1"](x1_in, d_lower=d0, d_upper=d1,
                              adj=None, x_lower=x0_in, x_upper=x2_in,
                              geo_lower=geo_0, geo_upper=geo_2))
            # 2-cochain: lower adjacency via d1 (from x1)
            x2 = norm["ln2"](layer["mp2"](x2_in, d_lower=d1, d_upper=None,
                              adj=None, x_lower=x1_in, x_upper=None,
                              geo_lower=geo_1, geo_upper=None))

        # Stage 3: Readout
        if self.task == "scalar":
            return self.readout(x0)
        elif self.task == "edge_scalar":
            return self.readout(x1)
        else:
            return self.readout(x1, K.pos, K.edges, K.n0, x0=x0)

    def forward_batch(self, f0_batch: torch.Tensor, K: CellComplex) -> torch.Tensor:
        """
        Batched forward: all samples share the same CellComplex.
        Exploits spmm linearity by folding the batch into the feature dimension.

        Args:
            f0_batch: (B, n0, f_in) batched node features
            K: shared CellComplex

        Returns:
            Scalar: (B, n0, out_dim)  Vector: (B, n0, spatial_dim)
        """
        adj_00 = K.get_adjacency(0, 0)
        d0, d1 = self._maybe_learned_d(K)

        # Stage 1: Batched lifting
        x0, x1, x2 = self.lifting.forward_batch(f0_batch, K)

        # Stage 2: Hodge MP x L layers (batched)
        for layer, norm in zip(self.layers, self.norms):
            x0_in, x1_in, x2_in = x0, x1, x2

            x0 = norm["ln0"](layer["mp0"].forward_batch(
                x0_in, d_lower=None, d_upper=d0,
                adj=adj_00, x_lower=None, x_upper=x1_in))
            x1 = norm["ln1"](layer["mp1"].forward_batch(
                x1_in, d_lower=d0, d_upper=d1,
                adj=None, x_lower=x0_in, x_upper=x2_in))
            x2 = norm["ln2"](layer["mp2"].forward_batch(
                x2_in, d_lower=d1, d_upper=None,
                adj=None, x_lower=x1_in, x_upper=None))

        # Stage 3: Readout (nn.Linear broadcasts over batch)
        if self.task == "scalar":
            return self.readout(x0)
        elif self.task == "edge_scalar":
            return self.readout(x1)
        else:
            # Vector task: per-sample reconstruction (involves scatter to vertices)
            B = f0_batch.size(0)
            outs = []
            for i in range(B):
                outs.append(self.readout(x1[i], K.pos, K.edges, K.n0, x0=x0[i]))
            return torch.stack(outs, dim=0)
