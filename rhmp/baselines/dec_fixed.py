"""Physics-prior controls built from the v2 model itself (same lifting, message passing, readouts, output maps):

==========  ==========================================================================================  =========
name        operators                                                                                   data star
==========  ==========================================================================================  =========
rhmp        learned bounded metric around the reference DEC Hodge star (the v2 model)                   cotan
dec_fixed   ``H = star`` frozen: fixed DEC Hodge operators, no metric learning ("geometry prior only")   cotan
unit_star   ``star = 1`` + learned metric (combinatorial Laplacians as the prior; = ``--star unit``)     unit
unit_fixed  ``star = 1``, ``H = 1`` frozen: combinatorial Hodge Laplacians (SCCNN-like operators) in the    unit
            v2 stack ("neither geometry nor learned metric")
==========  ==========================================================================================  =========

The 2 x 2 table (star in {cotan, unit}) x (metric in {learned, fixed}) separates the value of the geometric prior
from the value of metric learning.

``dec_fixed`` is *not* ``RHMPConfig(identity_metric=True)``: in ``rhmp.layers`` the identity-metric ablation sets
``log H = 0``, i.e. ``H = 1`` while the operator scaling still uses the star (``scaling='dec'``), which is neither
the DEC nor the combinatorial operator.  Instead the metric heads are frozen: their last layer is zero-initialised,
so ``H = star * exp(a * tanh(0)) = star`` exactly, for every input and throughout training (tested).  Frozen heads
are excluded from the trainable-parameter count.  ``unit_*`` need complexes built with ``star='unit'``
(``rhmp.train --model unit_star`` loads the task that way).
"""
from __future__ import annotations

from typing import Any

import torch

from rhmp.model import RHMP, RHMPConfig

__all__ = ["FixedMetricRHMP", "rhmp_config", "make_rhmp_variant", "VARIANTS"]

VARIANTS = {
    "rhmp": dict(freeze_metric=False, star=None),
    "dec_fixed": dict(freeze_metric=True, star=None),
    "unit_star": dict(freeze_metric=False, star="unit"),
    "unit_fixed": dict(freeze_metric=True, star="unit"),
}


class FixedMetricRHMP(RHMP):
    """:class:`rhmp.model.RHMP` whose metric heads are frozen at their zero initialisation (``H = star``).

    Args:
        cfg: :class:`RHMPConfig` (``identity_metric`` must be False).
        geo_dims: ``{k: G_k}``.
        variant: registry name recorded in checkpoints.
    """

    def __init__(self, cfg: RHMPConfig | dict, geo_dims: dict[int, int], variant: str = "dec_fixed") -> None:
        super().__init__(cfg, geo_dims)
        if self.cfg.identity_metric:
            raise ValueError("FixedMetricRHMP freezes the metric heads; identity_metric must be False")
        self.variant = variant
        for layer in self.layers:
            for head in layer.heads.values():
                if head.fc2.weight.abs().max() != 0 or head.fc2.bias.abs().max() != 0:
                    raise RuntimeError("metric head output layer is not zero-initialised; H = star would not hold")
                for p in head.parameters():
                    p.requires_grad_(False)

    def to_checkpoint(self) -> dict[str, Any]:
        ck = super().to_checkpoint()
        ck["model"] = self.variant
        return ck


def rhmp_config(td: Any, args: Any | None = None, **cfg_overrides) -> RHMPConfig:
    """The :class:`RHMPConfig` that ``rhmp.train`` builds for task ``td`` (``args``: trainer namespace; default:
    the trainer's defaults for this task)."""
    from rhmp.train import build_config, parse_args
    if args is None:
        args = parse_args(["--task", str(td.name)])
    cfg = build_config(td, args)
    if cfg_overrides:
        d = cfg.to_dict()
        d.update(cfg_overrides)
        cfg = RHMPConfig.from_dict(d)
    return cfg


def make_rhmp_variant(name: str, td: Any, args: Any | None = None, **cfg_overrides) -> RHMP:
    """Build ``rhmp`` / ``dec_fixed`` / ``unit_star`` / ``unit_fixed`` for task ``td``.

    Raises:
        ValueError: if the task's complexes do not use the star the variant needs.
    """
    if name not in VARIANTS:
        raise KeyError(f"unknown RHMP variant {name!r}; known: {sorted(VARIANTS)}")
    v = VARIANTS[name]
    K0 = td.K[0] if td.variable_mesh else td.K
    st = K0.meta.get("star_type")
    if v["star"] == "unit" and st != "unit":
        raise ValueError(f"{name} needs complexes with star='unit' (task loaded with star={st!r}); "
                         "load_task(..., star='unit') - rhmp.train --model does this automatically")
    if v["star"] is None and st == "unit":
        raise ValueError(f"{name} is defined on the reference DEC star; the task was loaded with star='unit' "
                         f"(use {'unit_fixed' if v['freeze_metric'] else 'unit_star'})")
    cfg = rhmp_config(td, args, **cfg_overrides)
    if v["freeze_metric"]:
        model = FixedMetricRHMP(cfg, td.geo_dims, variant=name)
    else:
        model = RHMP(cfg, td.geo_dims)
    model.info = {"name": name, "C": cfg.C, "n_layers": cfg.n_layers, "readout": cfg.readout,
                  "star": st, "metric": "frozen (H = star)" if v["freeze_metric"] else "learned"}
    return model


@torch.no_grad()
def metric_is_star(model: RHMP, inputs: dict, K: Any, atol: float = 0.0) -> bool:
    """True if every recorded ``log(H/star)`` statistic of the last forward pass is within ``atol`` of 0."""
    rec = model.record_diagnostics
    model.record_diagnostics = True
    model(inputs, K)
    model.record_diagnostics = rec
    d = model.diagnostics
    vals = [abs(v) for k, v in d.items() if k.endswith((".mean", ".min", ".max")) and ".H" in k]
    return bool(vals) and max(vals) <= atol
