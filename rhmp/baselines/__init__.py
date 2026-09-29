"""Baselines for the RHMP v2 task suite (same data, targets, loss, metrics and trainer as ``rhmp.RHMP``).

    from rhmp.baselines import build_model, applicable, MODELS
    ok, why = applicable("mgn", task)                       # task: rhmp.data.TaskData
    model, info = build_model("mgn", task, target_params)   # param-matched to the v2 model (v1 rule)
    y = model(inputs, K)                                    # {k: (n_k, B, F_k)}, complex -> (n_out, B, out_dim)

Modules:
    adapters      v1-compatible complex adapter (+ cached operators), input encoders, output heads
    v1_wrappers   uniform wrapper ``BaselineModel`` around the v1 baselines and the v1 model (``ours_v1``)
    mgn           MeshGraphNet (Pfaff et al., ICLR 2021), vectorised
    dec_fixed     fixed-metric RHMP controls (``dec_fixed``, ``unit_star``, ``unit_fixed``)
    param_match   v1 parameter-matching rule with a JSON cache
    registry      ``MODELS`` (name -> builder), ``applicable``, ``build_model``, benchmark CLI

Imports are lazy so that importing :mod:`rhmp.baselines` never pulls in the v1 tree.
"""
from __future__ import annotations

from typing import Any

__all__ = ["MODELS", "SPECS", "applicable", "build_model", "model_names"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import registry

        return getattr(registry, name)
    raise AttributeError(f"module 'rhmp.baselines' has no attribute {name!r}")
