"""Unpickling shim for machines without pyvista.

Some v1 datasets (T3/T4 ellipsoid meshes) were pickled with pyvista objects inside. The loaders only use the
numpy arrays next to them, so on a machine without pyvista we expose dummy classes that absorb the pickled state.
Enable with PYTHONPATH=<repo>/shims (never on machines where the real pyvista is installed).
"""
import sys as _sys

import numpy as _np


class _NDArray(_np.ndarray):
    """Stand-in for pyvista_ndarray (an ndarray subclass); numpy reconstructs it like a plain array."""


def _pick(name):
    return _NDArray if "ndarray" in name.lower() else _Dummy


class _Dummy:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state

    def __reduce__(self):
        return (_Dummy, ())


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return _pick(name)


class _ShimModule(type(_sys)):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _pick(name)


# make every submodule import (pyvista.core.pointset, ...) resolve to a shim module
class _Finder:
    @staticmethod
    def find_spec(fullname, path=None, target=None):
        if fullname == "pyvista" or fullname.startswith("pyvista."):
            import importlib.machinery as m
            return m.ModuleSpec(fullname, _Loader(), is_package=True)
        return None


class _Loader:
    @staticmethod
    def create_module(spec):
        return _ShimModule(spec.name)

    @staticmethod
    def exec_module(module):
        module.__path__ = []


_sys.meta_path.insert(0, _Finder())
