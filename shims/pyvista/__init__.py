"""Unpickling shim for datasets that were pickled with pyvista, for use on machines without pyvista.

Some datasets (for example the T3 ellipsoid pickle of the v1 paper) contain pyvista objects next to plain numpy
arrays.  The loaders only use the numpy arrays, so this package stands in for every ``pyvista`` module and class:
array subclasses are reconstructed as ndarrays and all other objects as inert placeholders that absorb their
pickled state.  Enable it with ``PYTHONPATH=<repo>/shims``, and only on machines where pyvista is not installed
(the shim would otherwise shadow the real package).
"""
import sys as _sys

import numpy as _np


class _NDArray(_np.ndarray):
    """Stand-in for pyvista_ndarray (an ndarray subclass); numpy reconstructs it like a plain array."""


def _pick(name):
    return _NDArray if "ndarray" in name.lower() else _Dummy


class _Dummy:
    """Inert placeholder for any other pyvista class; it keeps the unpickled state as attributes."""

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
    """Module object whose attributes resolve to the stand-in classes."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _pick(name)


# Every import of pyvista or one of its submodules (pyvista.core.pointset, ...) resolves to a shim module.
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
