"""Helpers to load the compiled pybind11 extension modules.

The native modules (``pyQES._winds``, ``pyQES._plume``, ``pyQES._fire``,
``pyQES._util``) are produced by the CMake build. Importing them lazily lets the
pure-Python parts of pyQES be used (config, XML I/O, geospatial preprocessing)
even in an environment where the extensions were not compiled.
"""

from __future__ import annotations

import importlib
from types import ModuleType

__all__ = ["load_extension", "solver_code"]

_BUILD_HINT = (
    "The pyQES native extension '{name}' is not available. Build the wheel with "
    "`pip install .` (requires a C++ toolchain plus GDAL, NetCDF and Boost), or "
    "install a prebuilt wheel from PyPI."
)

_SOLVERS = {
    "cpu": 1,
    "gpu": 2,
    "dynamic": 2,
    "global": 3,
    "shared": 4,
}


def load_extension(name: str) -> ModuleType:
    """Import a compiled extension module by short name (e.g. ``"_winds"``)."""
    try:
        return importlib.import_module(f"pyQES.{name}")
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(_BUILD_HINT.format(name=name)) from exc


def solver_code(solver: str | int) -> int:
    """Map a solver name/alias to the QES solver-type integer."""
    if isinstance(solver, int):
        return solver
    try:
        return _SOLVERS[solver.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown solver '{solver}'. Use one of {sorted(_SOLVERS)} or an int."
        ) from None
