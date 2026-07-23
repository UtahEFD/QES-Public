"""Shared utilities for pyQES: config models, XML I/O and geospatial helpers."""

from __future__ import annotations

# Geospatial helpers depend on optional native libraries (rasterio/pyproj/
# geopandas); import lazily so `import pyQES.util` works without them.
from . import geo  # noqa: F401  (re-exported module)
from .config import (
    BuildingsParams,
    FileOptions,
    MetParams,
    SensorParameters,
    SimulationParameters,
    TimeSeries,
    TurbParams,
    WindsParameters,
)
from .paths import resolve_path, resolve_work_dir
from .xml_io import (
    from_qes_xml,
    from_sensor_xml,
    to_qes_xml,
    to_sensor_xml,
    write_qes_xml,
    write_sensor_xml,
)

__all__ = [
    "SimulationParameters",
    "MetParams",
    "BuildingsParams",
    "TurbParams",
    "FileOptions",
    "WindsParameters",
    "TimeSeries",
    "SensorParameters",
    "to_qes_xml",
    "from_qes_xml",
    "to_sensor_xml",
    "from_sensor_xml",
    "write_qes_xml",
    "write_sensor_xml",
    "resolve_work_dir",
    "resolve_path",
    "geo",
]
