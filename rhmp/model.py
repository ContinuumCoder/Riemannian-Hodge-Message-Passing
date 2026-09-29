"""RHMP v2 model: configuration, lifting -> metric Hodge message passing -> readout (DESIGN §3, §4).

Inputs are a dict ``{k: (n_k, B, F_k)}`` of raw cochains on any degree (``cfg.in_dims``); the complex ``K`` is a
``rhmp.complex.CochainComplex`` shared by the ``B`` samples (or a block-diagonal batch of meshes with ``B = 1``).
The output of :meth:`RHMP.forward` is ``(n_out, B, out_dim)`` (``out_dim * D`` for ``node_vector``).

Checkpoints: ``{'cfg': cfg.to_dict(), 'geo_dims': {k: G_k}, 'state_dict': ...}`` (:meth:`RHMP.to_checkpoint`,
:meth:`RHMP.from_checkpoint`).  No parameter shape depends on the number of cells.
"""
from __future__ import annotations

import contextlib
import dataclasses
import math
import os
import pickle
import types
import warnings
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as _checkpoint

from .layers import LayerContext, RHMPLayer, make_context
from .lifting import CochainLifting, input_layout
from .readout import build_readout, parse_readout

__all__ = ["RHMPConfig", "RHMP"]

_DICT_FIELDS = ("in_dims", "even_dims", "connection_dims", "metric_reference", "latent_dims", "material_dims")


@dataclass
class RHMPConfig:
    """Hyper-parameters of :class:`RHMP`.

    Attributes:
        in_dims: ``{k: F_k}`` input width per degree (degrees absent or 0 have no inputs).
        even_dims: ``{k: E_k}`` number of even columns (listed last) of ``inputs[k]``; they also feed the metric
            head of degree ``k``.
        connection_dims: ``{k: c_k}`` number of gauge-connection columns (listed first) of ``inputs[k]``; used only
            through ``d_k A`` (exact gauge invariance), ``k < K``.
        C: hidden channels.
        n_layers: number of message-passing layers.
        poly_order: order ``P`` of the scalar polynomial filters.
        metric_hidden: hidden width of the metric MLPs.
        log_range: ``a``; ``H/star`` is confined to ``[e^-a, e^a]``.
        tie_metrics: one metric per degree (up uses ``H_m``, down uses ``1/H_m``) vs. separate up/down heads.
        scaling: ``'dec' | 'jacobi' | 'none'`` operator scaling.
        cross: cross-degree transport terms.
        gate: ``'norm'`` (radial gate), ``'relu'`` (ablation) or ``'none'`` (no nonlinearity; used by
            :meth:`solver_preset`).
        identity_metric: every metric ``H = 1`` (ablation of the learned metric, DESIGN §3.2): no learned correction
            and no reference star inside ``H``.  With ``scaling='dec'`` the blocks still carry the DEC stars of their
            own degree through the symmetric scaling ``S`` (``S_up = star_k``, ``S_down = 1/star_k``), so the geometric
            (physics) prior survives at that level; the fully combinatorial variant builds the complex with
            ``star='unit'`` (all reference stars 1) or uses ``scaling='none'``.  The default model is the pure DEC
            operator (``H = star``) at initialisation.
        readout: ``'node_scalar' | 'node_vector' | 'cochain:k' | 'even:k' | 'grad' | 'curl' | 'div' | 'div:k' |
            'mdiv' | 'mdiv:k'`` (see ``rhmp.readout``; ``grad``/``curl``/``div`` satisfy ``d d = 0`` exactly).
        out_dim: output channels (number of vector fields for ``node_vector``).
        vector_mode: ``'ls' | 'direct'`` (``node_vector`` only).
        checkpoint_layers: activation checkpointing per layer (training memory).
        amp: run the dense MLPs under bf16 autocast on CUDA (sparse products stay fp32).
        connection_odd: connection columns also enter the ordinary odd path (non-Abelian connections).
        lift_hidden: hidden width of the even lifting gates ``e_k``.
        fused: fused polynomial-block kernels (``False`` = plain autograd reference path, same maths).
        layers: per-layer types, e.g. ``['poly', 'poly', 'resolvent', 'poly']`` (DESIGN §9.2); ``None`` means
            ``['poly'] * n_layers``.  When given it defines the depth (``n_layers`` is set to ``len(layers)``).
        resolvent_iters: CG iterations of resolvent layers (default 20).  The solve acts identically on every
            channel and commutes with channel mixing (``M^{-1}(x Q) = (M^{-1} x) Q``), so no equivariant channel
            reduction lowers its cost; fewer iterations or warm starts do.
        resolvent_warm_start: start each resolvent layer's CG from the previous resolvent layer's solution of the same
            degree (same shape), solving only the correction (default True; exact implicit gradients).
        learn_metric: ``False`` removes all metric heads: ``H_m = star_m exp(ref_m)`` exactly (tied or untied, down
            blocks use the inverse; tensor metric ``b = 1, a = 0`` = the Galerkin/Whitney star, times the reference
            scaling).  The model is then the pure DEC/FEEC physics prior with only scalar polynomial, cross, gate,
            lifting and readout parameters (the ``dec_fixed`` baseline).  ``identity_metric=True`` takes precedence
            (``H = 1``).
        latent_dims: ``{k: r}``: ``r`` learnable per-cell input columns ``Z_k`` ``(n_k, r)`` on degree ``k`` (a learned
            hidden field, e.g. a velocity 1-form the data do not provide), initialised ``N(0, 0.1^2)`` and inserted
            before the even columns of ``inputs[k]`` (layout ``[connection | odd | latent | even]``): odd cochains for
            ``k >= 1`` (they enter the odd path and the lifting gates, never the metric heads), vertex columns for
            ``k = 0``.  These are the only mesh-sized parameters, so they tie the model to *one* mesh (shared-mesh
            tasks; a different ``n_k`` or a block-diagonal batch raises).  They are created by ``model.init_latents(K)``
            (call it before building the optimizer) or lazily, with a warning, by the first forward; checkpoints store
            them.
            All symmetry statements hold conditional on the latent field.
        material_dims: ``{k: m}``: the last ``m`` even columns of ``inputs[k]`` are *material* columns (e.g.
            ``log sigma``): they feed *only* the metric heads (``psi`` of the degree-k metric, the tensor-metric
            descriptors) and ``metric_reference``, never the lifting, a gate or any feature path, so the only route
            from the material to the output is the metric ``H``.
        lifting: ``'mlp'`` (default) or ``'linear'`` (``x_0 = W f_0``, ``x_k = Lin(d x_{k-1}) + Lin(f_k)``: no geometry,
            biases or even gates; linear in the field inputs).
        resolvent_normalize: ``False``: resolvent layers use the un-normalised physical operator
            ``(I + tau L^2 Delta_H)^{-1}`` (identifies the metric's magnitude; see ``RHMPLayer``).
        solve_iters, solve_tol, solve_bc: ``'solve'`` layers (``layers=[..., 'solve', ...]``): CG budget (the CG
            count grows like the square root of the mesh condition number; warm starts reuse the previous solve layer)
            and boundary condition (``'dirichlet'``: ``K.meta['dirichlet'][k]`` or ``K.boundary[k]`` fixed; ``'none'``:
            no fixed cells; ``'neumann'``: no fixed cells and vertex solves return the zero-mean (lumped mass)
            solution of the pure Neumann problem, as in the T5 generator).
            A solve layer computes ``y_k = (Delta_H + lam / L^2)^{-1} x_k`` with the un-normalised metric Hodge
            Laplacian ``Delta_H`` (FEM stiffness / lumped mass for k = 0) and ``lam = softplus(log_lam)`` (init 1e-3,
            ``L^D`` = domain measure).  See :meth:`solver_preset`.
        readout_head: ``'mlp'`` (default) or ``'linear'``: the potential head of the ``grad`` readout
            (``phi = MLP(x_0)`` or ``phi = Lin_nobias(x_0)``); the solver preset uses ``'linear'`` so that the model
            stays linear in the field inputs.
        solve_precond: CG preconditioner of solve layers: ``'none'`` (Jacobi, default) or ``'twolevel'`` (Jacobi plus
            an additive coarse correction on per-graph spatial aggregates of about ``rhmp.layers.COARSE_AGG_SIZE``
            vertices, assembled per sample and graph; degree 0 only, other degrees keep Jacobi).  Same solution, far
            fewer iterations on large meshes; O(C)-equivariant and per sample.
        tensor_param: tensor-metric parameterisation: ``'full'`` (default) ``sigma_f = b_f expm(sum_j s_j t_j t_j^T)``
            with signed bounded ``s`` (every SPD tensor with bounded condition number, incl. anisotropy misaligned with
            the edges) or ``'cone'`` (``b_f I + sum_j a_j t_j t_j^T``, ``a >= 0``: M-matrix stiffness only, the same
            class as the diagonal metric on triangles).  Configurations saved without a ``tensor_param`` key (with
            ``metric_type='tensor'``) load as ``'cone'``, the parameterisation they were trained with.
        resolvent_grad: ``'implicit'`` (adjoint CG solve, memory independent of the iterations; default) or
            ``'unrolled'`` (autograd through the checkpointed iterations; memory grows with ``resolvent_iters``).
        metric_type: ``'diag'`` (diagonal metrics, default) or ``'tensor'`` (Whitney/Galerkin material-tensor metric
            on the intermediate degrees for up blocks and cross-up terms, DESIGN §9.1).
        metric_reference: ``{m: column}``: absolute index of an *even* column of ``inputs[m]`` holding a log-scale
            material reference ``ref_m`` (e.g. ``log t^T Sigma t`` on edges).  It is a fixed offset of the log-metric,
            ``H_m = star_m exp(ref_m) exp(a tanh(MLP))`` (down blocks use the inverse), so the bounded correction is
            relative to the known coefficient.  Tensor metrics scale the material tensor instead: ``b_f`` by
            ``exp(mean of ref over the m-cells of f)`` and, for the edge metric, ``a_{f,j}`` by ``exp(ref of direction
            edge j)``.  Ignored with ``identity_metric``.
    """

    in_dims: dict[int, int]
    even_dims: dict[int, int] = field(default_factory=dict)
    connection_dims: dict[int, int] = field(default_factory=dict)
    C: int = 128
    n_layers: int = 4
    poly_order: int = 2
    metric_hidden: int = 32
    log_range: float = 2.0
    tie_metrics: bool = True
    scaling: str = "dec"
    cross: bool = True
    gate: str = "norm"
    identity_metric: bool = False
    readout: str = "node_scalar"
    out_dim: int = 1
    vector_mode: str = "ls"
    checkpoint_layers: bool = False
    amp: bool = False
    connection_odd: bool = False
    lift_hidden: int = 64
    fused: bool = True
    layers: list[str] | None = None
    resolvent_iters: int = 20
    resolvent_grad: str = "implicit"
    metric_type: str = "diag"
    metric_reference: dict[int, int] = field(default_factory=dict)
    resolvent_warm_start: bool = True
    learn_metric: bool = True
    latent_dims: dict[int, int] = field(default_factory=dict)
    material_dims: dict[int, int] = field(default_factory=dict)
    lifting: str = "mlp"
    resolvent_normalize: bool = True
    solve_iters: int = 64
    solve_tol: float = 1e-6
    solve_bc: str = "dirichlet"
    tensor_param: str = "full"
    solve_precond: str = "none"
    readout_head: str = "mlp"

    def __post_init__(self) -> None:
        for name in _DICT_FIELDS:
            raw = getattr(self, name) or {}
            clean = {int(k): int(v) for k, v in dict(raw).items()}
            if any(k < 0 or v < 0 for k, v in clean.items()):
                raise ValueError(f"{name} must map non-negative degrees to non-negative widths, got {raw}")
            setattr(self, name, clean)
        for k in set(self.even_dims) | set(self.connection_dims):
            F, E, c = self.in_dims.get(k, 0), self.even_dims.get(k, 0), self.connection_dims.get(k, 0)
            if E + c > F:
                raise ValueError(f"degree {k}: even_dims ({E}) + connection_dims ({c}) exceed in_dims ({F})")
        if self.C < 1 or self.n_layers < 0 or self.poly_order < 1 or self.out_dim < 1:
            raise ValueError("need C >= 1, n_layers >= 0, poly_order >= 1, out_dim >= 1")
        if not self.log_range > 0:
            raise ValueError("log_range must be > 0")
        if self.scaling not in ("dec", "jacobi", "none"):
            raise ValueError(f"scaling must be 'dec', 'jacobi' or 'none', got {self.scaling!r}")
        if self.gate not in ("norm", "relu", "none"):
            raise ValueError(f"gate must be 'norm', 'relu' or 'none', got {self.gate!r}")
        if self.vector_mode not in ("ls", "direct"):
            raise ValueError(f"vector_mode must be 'ls' or 'direct', got {self.vector_mode!r}")
        parse_readout(self.readout)
        self.log_range = float(self.log_range)
        if self.layers is not None:
            self.layers = [str(t) for t in self.layers]
            bad = [t for t in self.layers if t not in ("poly", "resolvent", "solve")]
            if bad or not self.layers:
                raise ValueError(f"layers must be a non-empty list of 'poly'/'resolvent'/'solve', got {self.layers}")
            self.n_layers = len(self.layers)
        if self.resolvent_iters < 1:
            raise ValueError("resolvent_iters must be >= 1")
        if self.resolvent_grad not in ("implicit", "unrolled"):
            raise ValueError(f"resolvent_grad must be 'implicit' or 'unrolled', got {self.resolvent_grad!r}")
        if self.metric_type not in ("diag", "tensor"):
            raise ValueError(f"metric_type must be 'diag' or 'tensor', got {self.metric_type!r}")
        if self.tensor_param not in ("cone", "full"):
            raise ValueError(f"tensor_param must be 'cone' or 'full', got {self.tensor_param!r}")
        if self.lifting not in ("mlp", "linear"):
            raise ValueError(f"lifting must be 'mlp' or 'linear', got {self.lifting!r}")
        if self.solve_bc not in ("dirichlet", "none", "neumann"):
            raise ValueError(f"solve_bc must be 'dirichlet', 'none' or 'neumann', got {self.solve_bc!r}")
        if self.readout_head not in ("mlp", "linear"):
            raise ValueError(f"readout_head must be 'mlp' or 'linear', got {self.readout_head!r}")
        if self.solve_precond not in ("none", "twolevel"):
            raise ValueError(f"solve_precond must be 'none' or 'twolevel', got {self.solve_precond!r}")
        if self.solve_iters < 1 or not self.solve_tol >= 0:
            raise ValueError("need solve_iters >= 1 and solve_tol >= 0")
        for k, M in self.material_dims.items():
            if M > self.even_dims.get(k, 0):
                raise ValueError(f"material_dims[{k}] = {M} exceeds even_dims[{k}] = {self.even_dims.get(k, 0)}: "
                                 f"material columns are the last even columns")
        for m, col in self.metric_reference.items():
            F, E = self.in_dims.get(m, 0), self.even_dims.get(m, 0)
            if not F - E <= col < F:
                raise ValueError(f"metric_reference[{m}] = {col} must index an even column of inputs[{m}] "
                                 f"(columns {F - E}..{F - 1})")

    @classmethod
    def solver_preset(cls, **kw: Any) -> "RHMPConfig":
        """A model that is linear in the field inputs and whose only nonlinearity is the metric map.

        ``lifting='linear'``, ``layers=['solve']``, ``gate='none'`` (the solve layer returns ``y = (Delta_H +
        lam / L^2)^{-1} x``), ``cross=False``, linear readout ``'cochain:0'`` (or any linear readout given in ``kw``),
        ``C=4``, ``readout_head='linear'`` (``grad``: ``E = d_0 Lin_nobias(x_0)``); everything can be overridden through
        ``kw`` (``in_dims`` is required).

        For P1 finite-element data of ``-div(sigma grad u) = f`` (lumped-mass right-hand side,
        homogeneous Dirichlet boundary) this class contains the exact discrete solution operator: with
        ``metric_type='tensor'`` (``H_1`` = Whitney/Galerkin star, ``b_f = sigma_f``: e.g. ``sigma`` as a material
        column on the top degree with ``metric_reference`` and ``learn_metric=False``, or learned), and on triangle
        meshes whose P1 stiffness is an M-matrix (non-negative cotan-type edge weights) also with the diagonal metric
        (``H_1`` = the P1 edge weights).  With ``learn_metric=True`` the metric heads learn the material map.
        """
        base = dict(layers=["solve"], gate="none", lifting="linear", readout="cochain:0", cross=False, C=4,
                    readout_head="linear")
        base.update(kw)
        return cls(**base)

    @property
    def layer_types(self) -> list[str]:
        """Per-layer types (``['poly'] * n_layers`` unless ``layers`` is given)."""
        return list(self.layers) if self.layers is not None else ["poly"] * self.n_layers

    def to_dict(self) -> dict[str, Any]:
        """Plain (JSON- and ``torch.save``-friendly) dict; :meth:`from_dict` inverts it."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RHMPConfig":
        """Inverse of :meth:`to_dict` (accepts string degree keys, e.g. after a JSON round trip).

        Raises:
            ValueError: on unknown keys.
        """
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown RHMPConfig keys: {sorted(unknown)}")
        d = dict(d)
        if d.get("metric_type") == "tensor" and "tensor_param" not in d:
            d["tensor_param"] = "cone"                                       # saved without tensor_param: cone
        return cls(**d)


class RHMP(nn.Module):
    """Riemannian Hodge Message Passing network (v2).

    Args:
        cfg: :class:`RHMPConfig` (or its dict).
        geo_dims: ``{k: G_k}`` descriptor widths of the complexes the model will see (``K.geo_dims``); the keys
            ``0..K`` also fix the top degree ``K``.
    """

    def __init__(self, cfg: RHMPConfig | dict, geo_dims: dict[int, int]) -> None:
        super().__init__()
        if isinstance(cfg, dict):
            cfg = RHMPConfig.from_dict(cfg)
        gd = {int(k): int(v) for k, v in dict(geo_dims).items()}
        if not gd:
            raise ValueError("geo_dims is empty")
        top = max(gd)
        if sorted(gd) != list(range(top + 1)) or top < 1:
            raise ValueError(f"geo_dims must have keys 0..K with K >= 1, got {sorted(gd)}")
        for name in _DICT_FIELDS:
            bad = [k for k, v in getattr(cfg, name).items() if k > top and v > 0]
            if bad:
                raise ValueError(f"cfg.{name} refers to degrees {bad} above the top degree {top}")
        self.cfg = cfg
        self.geo_dims = gd
        self.top = top
        self.lifting = CochainLifting(self._lifting_config(cfg), gd)
        self.latent = nn.ParameterDict()                                   # sized by init_latents(K)
        self.layers = nn.ModuleList([
            RHMPLayer(top, gd, cfg.even_dims, C=cfg.C, poly_order=cfg.poly_order, metric_hidden=cfg.metric_hidden,
                      log_range=cfg.log_range, tie_metrics=cfg.tie_metrics, scaling=cfg.scaling, cross=cfg.cross,
                      gate=cfg.gate, identity_metric=cfg.identity_metric, fused=cfg.fused, kind=kind,
                      resolvent_iters=cfg.resolvent_iters, resolvent_grad=cfg.resolvent_grad,
                      metric_type=cfg.metric_type if (kind in ("poly", "solve") or
                                                      (kind == "resolvent" and not cfg.resolvent_normalize)) else "diag",
                      resolvent_warm_start=cfg.resolvent_warm_start, learn_metric=cfg.learn_metric,
                      resolvent_normalize=cfg.resolvent_normalize, solve_iters=cfg.solve_iters,
                      solve_tol=cfg.solve_tol, solve_bc=cfg.solve_bc, tensor_param=cfg.tensor_param,
                      solve_precond=cfg.solve_precond)
            for kind in cfg.layer_types
        ])
        self.readout = build_readout(cfg, gd)
        self.record_diagnostics = True

    # ---------------------------------------------------------------------------------------------------------
    @staticmethod
    def _lifting_config(cfg: RHMPConfig) -> Any:
        """Config view for the lifting: input widths including the latent columns (``in_dims[k] + latent_dims[k]``)."""
        if not cfg.latent_dims:
            return cfg
        eff = dict(cfg.in_dims)
        for k, r in cfg.latent_dims.items():
            eff[k] = eff.get(k, 0) + r
        return types.SimpleNamespace(**{**{f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)},
                                        "in_dims": eff})

    def init_latents(self, K: Any) -> "RHMP":
        """Create (or check) the learnable latent input fields of ``cfg.latent_dims`` for the mesh of ``K``.

        Idempotent; call it before building the optimizer.  Latents are tied to one mesh.

        Raises:
            ValueError: if ``K`` is a block-diagonal batch of several meshes, or if existing latents were sized for a
                different number of cells.
        """
        if not self.cfg.latent_dims:
            return self
        if K.batch is not None and int(K.num_graphs) > 1:
            raise ValueError("latent_dims: latent fields are tied to one mesh; block-diagonal batches of several "
                             "meshes (variable-mesh tasks) are not supported")
        p = next(self.parameters())
        for k, r in sorted(self.cfg.latent_dims.items()):
            key = str(k)
            if key in self.latent:
                if self.latent[key].shape[0] != K.n[k]:
                    raise ValueError(f"latent_dims: the latent field on degree {k} was sized for "
                                     f"{self.latent[key].shape[0]} cells but this complex has n_{k} = {K.n[k]}; latent "
                                     f"fields are tied to one mesh (shared-mesh tasks only)")
                continue
            g = torch.Generator().manual_seed(1234 + k)
            z = 0.1 * torch.randn(K.n[k], r, generator=g, dtype=torch.float64)
            self.latent[key] = nn.Parameter(z.to(device=p.device, dtype=p.dtype))
        return self

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):  # noqa: D102
        for key, val in state_dict.items():                                  # create checkpointed latent fields
            if key.startswith("latent.") and key.split(".", 1)[1] not in self.latent:
                p = next(self.parameters())
                self.latent[key.split(".", 1)[1]] = nn.Parameter(torch.empty(val.shape, dtype=p.dtype, device=p.device))
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _with_latents(self, inputs: dict[int, Tensor], K: Any, B: int) -> dict[int, Tensor]:
        """``inputs`` with the latent columns inserted before the even block of each latent degree."""
        if not self.cfg.latent_dims:
            return inputs
        missing = [k for k in self.cfg.latent_dims if str(k) not in self.latent]
        if missing:
            warnings.warn("latent fields created lazily by the first forward; build the optimizer after calling "
                          "model.init_latents(K) (or after a first forward) so that they are trained", RuntimeWarning,
                          stacklevel=3)
        self.init_latents(K)
        out = dict(inputs)
        for k in sorted(self.cfg.latent_dims):
            z = self.latent[str(k)].unsqueeze(1).expand(-1, B, -1)
            if k in out:
                t = out[k]
                E = self.cfg.even_dims.get(k, 0)
                F = t.shape[-1]
                out[k] = torch.cat([t[..., :F - E], z.to(t.dtype), t[..., F - E:]], dim=-1)
            else:
                out[k] = z.contiguous()
        return out

    @staticmethod
    def geo_dims_of(K: Any) -> dict[int, int]:
        """``{k: G_k}`` of a complex (constructor argument)."""
        return {k: K.geo_dim(k) for k in range(K.dim + 1)}

    @property
    def output_degree(self) -> int:
        """Degree of the cells the output lives on (0 node readouts, 1 ``grad``, 2 ``curl``, k-1 ``div:k``, ...)."""
        return int(self.readout.output_degree)

    @property
    def feature_degree(self) -> int:
        """Degree of the hidden cochains the readout reads."""
        return int(self.readout.feature_degree)

    def num_parameters(self) -> int:
        """Number of trainable scalars."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _autocast(self, K: Any):
        if self.cfg.amp and K.pos.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _prepare(self, inputs: dict[int, Tensor], K: Any) -> tuple[dict[int, Tensor], LayerContext]:
        """Validate inputs against the config and the complex; build the forward context."""
        if int(K.dim) != self.top:
            raise ValueError(f"model built for top degree {self.top}, complex has dim {K.dim}")
        for k in range(self.top + 1):
            if K.geo_dim(k) != self.geo_dims[k]:
                raise ValueError(f"geo width mismatch on degree {k}: model {self.geo_dims[k]}, complex {K.geo_dim(k)}")
        p = next(self.parameters())
        dtype, device = p.dtype, p.device
        if K.pos.device != device:
            raise ValueError(f"complex is on {K.pos.device} but the model is on {device}; use K.to(device)")
        if K.star[0].dtype != dtype:
            raise ValueError(f"complex dtype {K.star[0].dtype} != model dtype {dtype}; use K.to(dtype=...)")
        inputs = {int(k): v for k, v in dict(inputs).items()}
        B = None
        for k, t in inputs.items():
            if not 0 <= k <= self.top:
                raise ValueError(f"inputs[{k}]: degree out of range 0..{self.top}")
            F = self.cfg.in_dims.get(k, 0)
            if F == 0:
                raise ValueError(f"inputs[{k}] given but cfg.in_dims[{k}] is 0")
            if t.dim() != 3 or t.shape[0] != K.n[k] or t.shape[2] != F:
                raise ValueError(f"inputs[{k}] must have shape (n_{k}={K.n[k]}, B, F_{k}={F}), got {tuple(t.shape)}")
            if B is None:
                B = int(t.shape[1])
            elif t.shape[1] != B:
                raise ValueError(f"inconsistent batch sizes across input degrees ({B} vs {t.shape[1]})")
            if t.device != device:
                raise ValueError(f"inputs[{k}] is on {t.device}, model on {device}")
        for k, F in self.cfg.in_dims.items():
            if F > 0 and k not in inputs:
                raise ValueError(f"cfg.in_dims[{k}] = {F} but inputs has no degree {k}")
        B = 1 if B is None else B
        inputs = {k: (t if t.dtype == dtype else t.to(dtype)) for k, t in inputs.items()}
        even, ref = [], []
        for k in range(self.top + 1):
            lay = input_layout(self.cfg, k)
            even.append(inputs[k][..., lay["F"] - lay["E"]:] if lay["E"] > 0 else None)
            col = self.cfg.metric_reference.get(k)
            ref.append(inputs[k][..., col] if (col is not None and not self.cfg.identity_metric) else None)
        ctx = make_context(K, scaling=self.cfg.scaling, B=B, dtype=dtype, even=even, record=self.record_diagnostics)
        ctx.ref = ref
        ctx.resolvent_prev = {}
        return self._with_latents(inputs, K, B), ctx

    def _propagate(self, x: list[Tensor], ctx: LayerContext, last_degrees: list[int] | None = None) -> list[Tensor]:
        """Run the layers; the last one only updates ``last_degrees`` (all degrees if ``None``)."""
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            degrees = last_degrees if i == n - 1 else None
            if self.cfg.checkpoint_layers and torch.is_grad_enabled():
                x = _checkpoint(layer, x, ctx, degrees, use_reentrant=False)
            else:
                x = layer(x, ctx, degrees)
        return x

    # ---------------------------------------------------------------------------------------------------------
    def forward(self, inputs: dict[int, Tensor], K: Any) -> Tensor:
        """Predict.

        Args:
            inputs: ``{k: (n_k, B, F_k)}`` for every degree with ``cfg.in_dims[k] > 0``.
            K: ``CochainComplex`` on the model's device and dtype.

        Returns:
            ``(n_out, B, out_dim)`` (``(n_0, B, out_dim * D)`` for ``node_vector``), in the model dtype.
        """
        inputs, ctx = self._prepare(inputs, K)
        with self._autocast(K):
            x = self._propagate(self.lifting(inputs, ctx), ctx, last_degrees=[self.feature_degree])
            y = self.readout(x, ctx)
        return y.to(ctx.dtype)

    def lift(self, inputs: dict[int, Tensor], K: Any) -> dict[int, Tensor]:
        """Lifting only: ``{k: (n_k, B, C)}``."""
        inputs, ctx = self._prepare(inputs, K)
        with self._autocast(K):
            x = self.lifting(inputs, ctx)
        return dict(enumerate(x))

    def propagate(self, x: dict[int, Tensor], K: Any, inputs: dict[int, Tensor] | None = None) -> dict[int, Tensor]:
        """Message-passing stack only (O(C)-equivariant): ``{k: (n_k, B, C)} -> {k: (n_k, B, C)}``.

        Args:
            x: hidden cochains for every degree ``0..K``.
            K: complex.
            inputs: raw inputs (only their even columns are used, by the metric heads); required if the config
                has even columns.
        """
        if sorted(int(k) for k in x) != list(range(self.top + 1)):
            raise ValueError(f"propagate needs features for degrees 0..{self.top}")
        xs = [x[k] for k in range(self.top + 1)]
        B = xs[0].shape[1]
        if inputs is None:
            if any(v > 0 for v in self.cfg.in_dims.values()):
                p = next(self.parameters())
                inputs = {k: torch.zeros(K.n[k], B, F, dtype=p.dtype, device=p.device)
                          for k, F in self.cfg.in_dims.items() if F > 0}
            else:
                inputs = {}
        _, ctx = self._prepare(inputs, K)
        for k, v in enumerate(xs):
            if v.shape != (K.n[k], ctx.B, self.cfg.C):
                raise ValueError(f"x[{k}] must have shape {(K.n[k], ctx.B, self.cfg.C)}, got {tuple(v.shape)}")
        with self._autocast(K):
            out = self._propagate([v.to(ctx.dtype).contiguous() for v in xs], ctx)
        return dict(enumerate(out))

    @torch.no_grad()
    def metric_fields(self, inputs: dict[int, Tensor], K: Any) -> list[dict]:
        """Learned metrics of every layer for these inputs (public accessor for analysis scripts).

        Returns:
            One dict per layer (see ``RHMPLayer.metric_fields``): ``{'log_ratio': {m: (n_m, B)}, 'phi': {m: (n_m, B)},
            'tensor': {km: (b (n_top, B), a (n_top, B, m))}, 'sigma': {km: (n_top, B, D, D)}}``.
        """
        inputs, ctx = self._prepare(inputs, K)
        with self._autocast(K):
            x = self.lifting(inputs, ctx)
            out = []
            for layer in self.layers:
                out.append(layer.metric_fields(x, ctx))
                x = layer(x, ctx)
        return out

    def operator_residual(self, inputs: dict[int, Tensor], K: Any, u: Tensor, f: Tensor, k: int = 0, layer: int = -1,
                          bc: str | None = None) -> Tensor:
        """Operator-identification residual ``|| S_H u - M f ||^2 / || M f ||^2`` per sample (auxiliary PDE loss).

        ``S_H`` is the weak (FEM) metric Hodge stiffness of degree ``k`` built from the metrics that layer ``layer``
        uses on these inputs (``d_0^T H_1 d_0`` for k = 0; ``+ star_k d_{k-1} H^down d_{k-1}^T star_k`` for k >= 1;
        tensor metrics through the Whitney blocks), in physical units, and ``M = star_k`` the lumped mass.  Rows of
        fixed cells are excluded under ``bc='dirichlet'`` (``K.meta['dirichlet'][k]`` or ``K.boundary[k]``; default
        ``cfg.solve_bc``).  For the diagonal metric the residual is linear in ``H``, so the loss is convex in the metric
        (convex identification of the material from (u, f) pairs).

        Args:
            inputs: model inputs (as for :meth:`forward`).
            K: complex (``scaling='dec'`` models).
            u: solution cochain ``(n_k, B)`` or ``(n_k, B, 1)`` (physical units, incl. its boundary values).
            f: source ``(n_k, B)`` or ``(n_k, B, 1)`` (physical units).
            k: degree of the equation.
            layer: index of the layer whose metric is used (default: the last layer).
            bc: ``'dirichlet' | 'none' | 'neumann'`` (default ``cfg.solve_bc``; only ``'dirichlet'`` excludes rows).

        Returns:
            ``(1, B)`` (single complex) or ``(num_graphs, B)``; differentiable w.r.t. the model parameters.
        """
        from .layers import SharedCoboundaries, dirichlet_free, physical_hodge, sample_norm2
        if self.cfg.scaling != "dec":
            raise ValueError("operator_residual needs scaling='dec'")
        inputs, ctx = self._prepare(inputs, K)
        ctx.record = False
        n = len(self.layers)
        if n == 0:
            raise ValueError("operator_residual needs at least one layer")
        li = layer % n
        top = self.top
        with self._autocast(K):
            x = self.lifting(inputs, ctx)
            for i in range(li):
                x = self.layers[i](x, ctx)
            lay = self.layers[li]
            q = SharedCoboundaries(x, ctx)
            tensor_up = k < top and (k + 1) in lay.tensor_degrees
            need_up = [k + 1] if (k < top and not tensor_up) else []
            logH_up, logH_dn = lay.metrics(x, q, ctx, need_up, [k - 1] if k > 0 else [])
            tens = {k + 1: lay.tensor_metric(k + 1, x, q, ctx)} if tensor_up else {}
        op = physical_hodge(ctx, k, logH_up, logH_dn, tens)
        dt = ctx.dtype
        u3 = (u if u.dim() == 3 else u.unsqueeze(-1)).to(dt)
        f3 = (f if f.dim() == 3 else f.unsqueeze(-1)).to(dt)
        st = K.star[k].to(dt).view(-1, 1, 1)
        r = st.sqrt() * op.apply((st.sqrt() * u3).contiguous())
        mf = st * f3
        free = dirichlet_free(ctx, k, bc or self.cfg.solve_bc)
        r = r - mf
        if free is not None:
            r, mf = r * free, mf * free
        den = sample_norm2(ctx, k, mf)
        return sample_norm2(ctx, k, r) / torch.where(den > 0, den, torch.ones_like(den))

    def hidden(self, inputs: dict[int, Tensor], K: Any) -> dict[int, Tensor]:
        """Pre-readout features ``{k: (n_k, B, C)}`` (lifting + message passing)."""
        inputs, ctx = self._prepare(inputs, K)
        with self._autocast(K):
            x = self._propagate(self.lifting(inputs, ctx), ctx)
        return dict(enumerate(x))

    @torch.no_grad()
    def fit_linear_readout_(self, inputs: dict[int, Tensor], K: Any, target: Tensor, output_map: Any = None,
                            rtol: float = 1e-8) -> float:
        """Least-squares weights of a linear readout from one batch (in place; minimum-norm, channels may be collinear).

        Linear readouts: ``cochain:k`` (``h_k W^T``), ``grad`` with ``readout_head='linear'`` (``d_0 (h_0 W^T)``),
        ``curl`` (``d_1 (h_1 W^T)``), ``div[:k]`` (``d_{k-1}^T (h_k W^T)``), ``mdiv[:k]`` (``S^{-1} d_{k-1}^T (h_k W^T)``);
        the output is ``F W^T`` with the features ``F = Op(h)``.  With a (linear) task ``output_map`` the fit is made
        in the target space through the composed map ``output_map o Op`` (one output column).  The solver class
        (``solver_preset``) is exact up to the overall gain of lifting x readout; this initialises that gain from data
        (the trainer does it for ``--solver-mode``).

        Args:
            inputs, K: one batch (as for :meth:`forward`); target: model output units ``(n_out, B, out_dim)`` or, with
                ``output_map``, the task's target ``(n_t, B, O_t)``.
            output_map: optional ``rhmp.data.OutputMap`` applied to the model output.
            rtol: relative singular-value cut-off of the pseudo-inverse.

        Returns:
            Relative residual of the fit.
        """
        from . import ops
        from .readout import CochainReadout, CurlReadout, DivReadout, GradReadout, MassDivReadout
        ro = self.readout
        inputs_p, ctx = self._prepare(inputs, K)
        with self._autocast(K):
            h = dict(enumerate(self._propagate(self.lifting(inputs_p, ctx), ctx)))
        if isinstance(ro, MassDivReadout):
            F = ops.spmm(K.dT[ro.k - 1], h[ro.k].contiguous(), K.d[ro.k - 1])
            F = F * torch.exp(-ctx.log_star[ro.k - 1]).to(F.dtype).view(-1, 1, 1)
        elif isinstance(ro, CochainReadout):
            F = h[ro.degree]
        elif isinstance(ro, GradReadout) and ro.linear:
            F = ops.spmm(K.d[0], h[0].contiguous(), K.dT[0])
        elif isinstance(ro, CurlReadout):
            F = ops.spmm(K.d[1], h[1].contiguous(), K.dT[1])
        elif isinstance(ro, DivReadout):
            F = ops.spmm(K.dT[ro.k - 1], h[ro.k].contiguous(), K.d[ro.k - 1])
        else:
            raise ValueError(f"fit_linear_readout_ needs a linear readout (cochain:k, curl, div, mdiv, or grad with "
                             f"readout_head='linear'), got {self.cfg.readout!r}")
        lin = ro.lin
        C = F.shape[-1]
        if output_map is not None:
            if lin.weight.shape[0] != 1:
                raise ValueError("fit_linear_readout_ with an output map supports one output column")
            Ft = torch.stack([output_map(F[..., c:c + 1].contiguous()) for c in range(C)], -1)   # (n_t, B, O_t, C)
            if Ft.shape[:-1] != target.shape:
                raise ValueError(f"target {tuple(target.shape)} does not match the mapped output {tuple(Ft.shape[:-1])}")
            Hm, T = Ft.reshape(-1, C).double(), target.reshape(-1, 1).double()
        else:
            if target.shape[:2] != F.shape[:2]:
                raise ValueError(f"target must be (n, B, out_dim) with (n, B) = {tuple(F.shape[:2])}, "
                                 f"got {tuple(target.shape)}")
            Hm, T = F.reshape(-1, C).double(), target.reshape(-1, target.shape[-1]).double()
        W = torch.linalg.pinv(Hm.T @ Hm, rtol=rtol, hermitian=True) @ (Hm.T @ T)            # (C, out_dim)
        lin.weight.copy_(W.T.to(lin.weight))
        return float((Hm @ W - T).norm() / T.norm().clamp_min(1e-300))

    # ---------------------------------------------------------------------------------------------------------
    @property
    def diagnostics(self) -> dict[str, float]:
        """Statistics of the last forward pass (with ``record_diagnostics = True``), as python floats.

        One key scheme, ``layer{l}.<name>``:

        * ``layer{l}.H{m}.<stat>`` for the degree-m metric (untied metrics: ``H{m}.up_<stat>`` / ``H{m}.down_<stat>``)
          with ``<stat>`` in ``mean, std, min, max`` (of the learned bounded part ``phi = log(H/(star exp(ref)))``),
          ``cond_learned`` (max over samples of ``max exp(phi) / min exp(phi)``: 1 at initialisation),
          ``cond_total`` (max over samples of ``max H / min H`` including the reference star),
          ``clamp_fraction`` (fraction of cells and samples with ``|tanh| > 0.99``) and ``sat`` (``|phi| > 0.95 a``);
        * ``layer{l}.H{m}.tensor_<stat>`` for tensor metrics: ``logb_mean, aniso_mean, aniso_max`` (eigenvalue ratio of
          the learned ``sigma_f``, tangent plane on surfaces), ``clamp_b`` and ``s_absmax, clamp_s``
          (``tensor_param='full'``) or ``a_mean, a_max, clamp_a`` (``'cone'``);
        * ``layer{l}.beta_up{k}`` / ``layer{l}.beta_down{k}``: mean Gershgorin normaliser of each block;
        * resolvent layers: ``layer{l}.tau_up{k}`` / ``layer{l}.tau_down{k}`` and ``layer{l}.cg_res{k}`` (final relative
          CG residual, max over samples);
        * solve layers: ``layer{l}.solve_res{k}`` (final relative CG residual, max over samples and graphs),
          ``layer{l}.solve_it{k}`` (CG iterations run) and ``layer{l}.lam{k}``.

        Calling it synchronises the device.
        """
        stats = ("mean", "std", "min", "max", "cond_learned", "cond_total", "clamp_fraction", "sat")
        out: dict[str, float] = {}
        for i, layer in enumerate(self.layers):
            for name, t in layer.last_diagnostics.items():
                if t.dim() == 0:
                    out[f"layer{i}.{name}"] = float(t)
                    continue
                base, _, variant = name.partition(".")
                prefix = f"layer{i}.{base}." + (f"{variant}_" if variant else "")
                for stat, v in zip(stats, (float(v) for v in t.tolist())):
                    out[prefix + stat] = math.exp(min(v, 700.0)) if stat.startswith("cond") else v
        return out

    def to_checkpoint(self) -> dict[str, Any]:
        """``{'cfg': cfg.to_dict(), 'geo_dims': {k: G_k}, 'state_dict': state_dict}`` for ``torch.save``."""
        return {"cfg": self.cfg.to_dict(), "geo_dims": dict(self.geo_dims), "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, ckpt: dict | str | os.PathLike, map_location: Any = None, strict: bool = True,
                        weights_only: bool | None = None) -> "RHMP":
        """Rebuild a model from a checkpoint (dict or path) and return it.

        Accepted formats:

        * :meth:`to_checkpoint` output ``{'cfg', 'geo_dims', 'state_dict'}``;
        * the trainer's ``best.pt`` (the same keys plus epoch / validation / task metadata);
        * the trainer's ``last.pt`` (the model under the ``'model'`` key next to optimizer, scheduler and RNG state).

        Args:
            ckpt: checkpoint dict or path.
            map_location: device for loading (the returned model is moved there too).
            strict: ``load_state_dict`` strictness.
            weights_only: ``torch.load`` mode for paths.  ``None`` (default) first tries the safe ``weights_only=True``
                and falls back to full unpickling for files holding non-tensor state (e.g. the numpy RNG state of a
                trainer ``last.pt``): only load checkpoint files you trust.

        Raises:
            ValueError: if the file does not contain an RHMP model.
        """
        if not isinstance(ckpt, dict):
            if weights_only is None:
                try:
                    ckpt = torch.load(ckpt, map_location=map_location, weights_only=True)
                except (pickle.UnpicklingError, RuntimeError):
                    ckpt = torch.load(ckpt, map_location=map_location, weights_only=False)
            else:
                ckpt = torch.load(ckpt, map_location=map_location, weights_only=weights_only)
        if "cfg" not in ckpt and isinstance(ckpt.get("model"), dict) and "cfg" in ckpt["model"]:
            ckpt = ckpt["model"]                                              # trainer last.pt
        missing = [k for k in ("cfg", "geo_dims", "state_dict") if k not in ckpt]
        if missing:
            raise ValueError(f"not an RHMP checkpoint: missing {missing} (expected to_checkpoint() output, a trainer "
                             f"best.pt, or a trainer last.pt with the model under 'model')")
        model = cls(RHMPConfig.from_dict(ckpt["cfg"]), ckpt["geo_dims"])
        model.load_state_dict(ckpt["state_dict"], strict=strict)
        if map_location is not None:
            model = model.to(map_location)
        return model
