"""pyQES: Python wrapper for the Quick Environmental Simulation (QES) suite.

Submodules:
  - :mod:`pyQES.pywinds` -- QES-Winds solver interface.
  - :mod:`pyQES.pyplume` -- QES-Plume dispersion interface.
  - :mod:`pyQES.pyfire`  -- coupled QES-Fire interface.
  - :mod:`pyQES.util`    -- config models, XML/JSON I/O and geospatial helpers.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from . import pyfire, pyplume, pywinds, util

try:
    __version__ = version("pyqes")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0"

__all__ = ["pywinds", "pyplume", "pyfire", "util", "__version__"]
