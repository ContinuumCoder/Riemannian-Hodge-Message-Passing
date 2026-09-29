"""RHMP v2: Riemannian Hodge Message Passing on cochain complexes.

Public API (names are resolved lazily, so that each one imports only its own sub-module)::

    from rhmp import RHMP, RHMPConfig, CochainComplex          # model, configuration, complexes
    from rhmp import cg_solve, hodge_decompose, sharp, flat    # DEC toolkit (rhmp.dec)
    from rhmp import load_task                                  # task registry (rhmp.tasks)
"""
from __future__ import annotations

import importlib
from typing import Any

__version__ = "2.0.0"

_LAZY = {
    "RHMP": ("model", "RHMP"),
    "RHMPConfig": ("model", "RHMPConfig"),
    "CochainComplex": ("complex", "CochainComplex"),
    "cg_solve": ("dec", "cg_solve"),
    "hodge_decompose": ("dec", "hodge_decompose"),
    "sharp": ("dec", "sharp"),
    "flat": ("dec", "flat"),
    "load_task": ("tasks", "load_task"),
}

__all__ = ["__version__", *_LAZY]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module, attr = _LAZY[name]
        return getattr(importlib.import_module(f".{module}", __name__), attr)
    raise AttributeError(f"module 'rhmp' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
