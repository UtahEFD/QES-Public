"""Tests for NetCDF → GeoTIFF export helpers."""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("netCDF4")
pytest.importorskip("rasterio")

from pyQES.util.outputs import default_mag_tif_name  # noqa: E402


def test_default_mag_tif_name():
    assert default_mag_tif_name("umep_larochelle", 1.5, 180.0) == (
        "umep_larochelle_Vmag_z1.5m_dir180.tif"
    )
    assert default_mag_tif_name("run", 10.0, 270.0) == "run_Vmag_z10m_dir270.tif"
