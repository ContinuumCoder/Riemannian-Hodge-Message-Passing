"""MeshGraphNet (Pfaff, Fortunato, Sanchez-Gonzalez, Battaglia; ICLR 2021), vectorised for the v2 layout.

Architecture (as the reference ``meshgraphnets/core_model.py``): encode - process - decode.

* encoder: node MLP and edge MLP, each ``in -> H -> H -> H`` (2 hidden layers, ReLU) + LayerNorm;
* processor: ``L`` blocks on the *directed* mesh graph (every mesh edge in both directions);
  edge update ``h_ij = LN(MLP_e([e_ij, x_i, x_j]))``, node update ``x_i <- x_i + LN(MLP_v([x_i, sum_j h_ij]))``,
  ``e_ij <- e_ij + h_ij`` (aggregation of the new messages, residuals afterwards, as the reference code);
* decoder: ``H -> H -> H -> out`` without LayerNorm.  Node targets decode node latents; edge targets decode the
  two directed edge latents, ``dec(e_ab) - dec(e_ba)`` for orientation-odd targets (exactly odd under edge
  reversal) and their mean otherwise; face / tet targets decode the mean of their vertices' latents (times the
  face orientation sign for odd targets on planar meshes).

Inputs: node features = the fixed node encoding of the node-input baselines (``rhmp.baselines.adapters.
NodeInputEncoder``: node inputs, incident means of even edge/face/tet inputs, and for odd edge inputs the
orientation-independent part ``mean_e v_e t_e`` of v1's directional encoding - its orientation-convention dependent
``avg`` column is dropped, so MGN stays exactly equivariant under vertex relabelling; ``node_edge_encoding=False``
leaves edge inputs to the edge features only);
edge features per directed edge = ``[odd edge inputs (+v forward, -v reverse), v * t (t the unit vector of the
directed edge), even edge inputs, (x_recv - x_send) / s, |x_recv - x_send| / s]`` with ``s`` the mesh's mean edge
length (``EdgeInputEncoder``).  No absolute positions (translation invariant, as MGN).  T6, 10 epochs: MGN with raw
edge values only stays at val R2 ~ 0; with ``v * t`` edge features it takes off after 6 epochs (0.57 at epoch 10);
with v1's node encoding (legacy inputs) it reaches 0.83 - the node encoding hands every node its one-ring vector
field, which the edge MLPs otherwise have to assemble.

Implementation:
* the directed edge index, the geometry and all sparse operators are built once per complex and cached on its
  adapter (``adapter_for``);
* gathers are ``index_select`` along dim 0 of ``(n, B, H)`` tensors (one gather per operand per block for the whole
  batch of a shared mesh), scatters ``index_add_``; no Python loop over edges, nodes or samples;
* the first layer of the edge MLP on ``[e, x_i, x_j]`` is split into ``W_e e + (W_r x)[recv] + (W_s x)[send]``:
  node-level projections are gathered instead of multiplying the ``2 n1``-row concatenation (identical maths);
* variable meshes are block-diagonal batches (one big graph);
* ``amp=True``: bf16 autocast for the MLPs on CUDA; LayerNorm outputs, residual streams and aggregations stay fp32;
* ``torch.compile``: adapter look-ups and sparse pooling run in ``torch.compiler.disable``-d prepare/decode steps,
  the processor is pure tensor code without data-dependent control flow.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .adapters import EdgeInputEncoder, NodeInputEncoder, TaskIO, V1ComplexAdapter, adapter_for, mlp
from .v1_wrappers import BaselineModel

__all__ = ["GNBlock", "MeshGraphNet", "make_mgn", "mgn_param_count"]


def _mlp(sizes: list[int], act: type[nn.Module], layernorm: bool) -> nn.Sequential:
    return mlp(sizes, act=act, layernorm=layernorm)


class GNBlock(nn.Module):
    """One MeshGraphNet processor block (edge update, sum aggregation, node update, residuals).

    Args:
        H: latent width.
        mlp_layers: hidden layers per MLP (2 in the paper: ``in -> H -> H -> H``).
        act: activation class (ReLU in the paper).
    """

    def __init__(self, H: int, mlp_layers: int = 2, act: type[nn.Module] = nn.ReLU) -> None:
        super().__init__()
        # edge MLP on [e, x_recv, x_send]: first layer split into three blocks (one bias)
        self.e_in = nn.Linear(H, H)
        self.e_recv = nn.Linear(H, H, bias=False)
        self.e_send = nn.Linear(H, H, bias=False)
        self.e_rest = self._rest(H, mlp_layers, act)
        self.e_norm = nn.LayerNorm(H)
        # node MLP on [x, sum_j h_ij]
        self.n_self = nn.Linear(H, H)
        self.n_agg = nn.Linear(H, H, bias=False)
        self.n_rest = self._rest(H, mlp_layers, act)
        self.n_norm = nn.LayerNorm(H)

    @staticmethod
    def _rest(H: int, mlp_layers: int, act: type[nn.Module]) -> nn.Sequential:
        layers: list[nn.Module] = []
        for _ in range(mlp_layers):
            layers += [act(), nn.Linear(H, H)]
        return nn.Sequential(*layers)

    def forward(self, x: Tensor, e: Tensor, send: Tensor, recv: Tensor) -> tuple[Tensor, Tensor]:
        """``x (n, B, H)``, ``e (2 n1, B_e, H)`` -> updated ``(x, e)`` (``e`` becomes ``(2 n1, B, H)``)."""
        h = self.e_in(e) + self.e_recv(x).index_select(0, recv) + self.e_send(x).index_select(0, send)
        h = self.e_norm(self.e_rest(h))                                         # (2 n1, B, H)
        agg = h.new_zeros((x.shape[0],) + tuple(h.shape[1:])).index_add_(0, recv, h)
        x = x + self.n_norm(self.n_rest(self.n_self(x) + self.n_agg(agg)))
        return x, e + h


class MeshGraphNet(BaselineModel):
    """MeshGraphNet with the :class:`BaselineModel` interface: ``forward(inputs, K) -> (n_out, B, out_dim)``.

    Args:
        io: task description (inputs, target degree/parity, spatial dimension).
        hidden: latent width ``H``.
        n_layers: processor steps ``L``.
        mlp_layers: hidden layers per MLP.
        act: ``'relu'`` (paper) or ``'silu'``.
        normalized_geometry: edge geometry divided by the mesh's mean edge length.
        amp: bf16 autocast for the MLPs on CUDA.
        name: registry name (``mgn`` / ``mgn_fast``).
    """

    def __init__(self, io: TaskIO, hidden: int, n_layers: int = 15, *, mlp_layers: int = 2, act: str = "relu",
                 normalized_geometry: bool = True, directional_edges: bool = True, node_edge_encoding: bool = True,
                 amp: bool = False, name: str = "mgn") -> None:
        act_cls = {"relu": nn.ReLU, "silu": nn.SiLU}[act]
        node_enc = NodeInputEncoder(io.in_dims, io.even_dims, io.spatial_dim, include_edges=node_edge_encoding,
                                    odd_avg=False)
        edge_enc = EdgeInputEncoder(io.in_dims, io.even_dims, io.spatial_dim, normalized_geometry=normalized_geometry,
                                    directional=directional_edges)
        H = int(hidden)
        hid = [H] * mlp_layers
        core = nn.Module()
        core.node_mlp = _mlp([node_enc.out_dim] + hid + [H], act_cls, True)
        core.edge_mlp = _mlp([edge_enc.out_dim] + hid + [H], act_cls, True)
        core.blocks = nn.ModuleList([GNBlock(H, mlp_layers, act_cls) for _ in range(n_layers)])
        t = io.target_degree
        info = {"name": name, "hidden": H, "n_layers": int(n_layers), "mlp_layers": mlp_layers, "act": act,
                "node_encoder": node_enc.layout, "node_in": node_enc.out_dim, "edge_in": edge_enc.out_dim,
                "target_degree": t, "odd_target": io.odd_target}
        build = dict(hidden=H, n_layers=int(n_layers), mlp_layers=mlp_layers, act=act,
                     normalized_geometry=normalized_geometry, directional_edges=directional_edges,
                     node_edge_encoding=node_edge_encoding)
        super().__init__(name, node_enc, core, None, edge_encoder=edge_enc, build=build, info=info, amp=amp)
        self.t = t
        self.odd = io.odd_target
        self.decoder = _mlp([H] + hid + [io.out_dim], act_cls, False)

    # ------------------------------------------------------------------ steps
    @torch.compiler.disable
    def _prepare(self, inputs: dict[int, Tensor], K: Any) -> tuple[V1ComplexAdapter, Tensor, Tensor, Tensor, Tensor]:
        """Adapter, node features ``(n0, B, F_n)``, edge features ``(2 n1, B_e, F_e)``, ``send``, ``recv``."""
        A = adapter_for(K)
        inputs = {int(k): v for k, v in dict(inputs).items()}
        self._check(inputs, A)
        f0 = self.encoder(inputs, A)
        e0 = self.edge_encoder(inputs, A, dtype=f0.dtype)
        send, recv = A.directed_edges()
        return A, f0, e0, send, recv

    def _process(self, f0: Tensor, e0: Tensor, send: Tensor, recv: Tensor) -> tuple[Tensor, Tensor]:
        """Encoder + processor (pure tensor code)."""
        x = self.core.node_mlp(f0)
        e = self.core.edge_mlp(e0)
        for blk in self.core.blocks:
            x, e = blk(x, e, send, recv)
        return x, e

    @torch.compiler.disable
    def _cell_decode(self, A: V1ComplexAdapter, x: Tensor) -> Tensor:
        """Face / tet targets: decode the mean latent of the cell's vertices (x sigma_f for odd planar faces)."""
        y = self.decoder(A.apply(f"mean_0to{self.t}", x))
        if self.t == 2 and self.odd:
            sig = A.face_orientation()
            if sig is not None:
                y = y * sig.to(y.dtype).view(-1, 1, 1)
        return y

    def _edge_decode(self, e: Tensor, B: int) -> Tensor:
        """Edge targets from the two directed latents (forward half first, see ``directed_edges``)."""
        e_f, e_r = torch.chunk(e, 2, dim=0)
        if e_f.shape[1] != B:
            e_f, e_r = e_f.expand(-1, B, -1), e_r.expand(-1, B, -1)
        y_f, y_r = self.decoder(e_f), self.decoder(e_r)
        return y_f - y_r if self.odd else 0.5 * (y_f + y_r)

    def forward(self, inputs: dict[int, Tensor], K: Any) -> Tensor:
        A, f0, e0, send, recv = self._prepare(inputs, K)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp and f0.is_cuda):
            x, e = self._process(f0, e0, send, recv)
            if self.t == 0:
                y = self.decoder(x)
            elif self.t == 1:
                y = self._edge_decode(e, x.shape[1])
            else:
                y = self._cell_decode(A, x)
        return y.float()


def make_mgn(io: TaskIO, hidden: int, n_layers: int = 15, *, name: str = "mgn", amp: bool = False,
             **kw) -> MeshGraphNet:
    """Build a :class:`MeshGraphNet` for task ``io``."""
    return MeshGraphNet(io, hidden, n_layers, amp=amp, name=name, **kw)


def mgn_param_count(node_in: int, edge_in: int, out: int, H: int, L: int, mlp_layers: int = 2) -> int:
    """Closed-form parameter count (checks / docs): encoders + ``L`` blocks (``~9 H^2`` each) + decoder."""
    def mlp_n(i, o, ln):
        sizes = [i] + [H] * mlp_layers + [o]
        n = sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))
        return n + (2 * o if ln else 0)
    block = (3 * H * H + H) + mlp_layers * (H * H + H) + 2 * H + (2 * H * H + H) + mlp_layers * (H * H + H) + 2 * H
    return mlp_n(node_in, H, True) + mlp_n(edge_in, H, True) + L * block + mlp_n(H, out, False)
