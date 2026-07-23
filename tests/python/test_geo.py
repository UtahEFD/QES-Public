"""Tests for the geospatial preprocessing helpers.

These mirror the arithmetic of ``run_qeswinds.sh`` and require the geo extras
(rasterio / pyproj / geopandas).
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("rasterio")
pytest.importorskip("pyproj")

from pyQES.util import geo  # noqa: E402
from pyQES.util.config import WindsParameters  # noqa: E402


def test_dem_extent(dem_path):
    ext = geo.dem_extent(dem_path)
    assert ext.width_px > 0 and ext.height_px > 0
    assert ext.width_m == ext.width_px * abs(ext.res_x)
    assert ext.height_m == ext.height_px * abs(ext.res_y)


def test_compute_domain_origin(dem_path):
    ext = geo.dem_extent(dem_path)
    origin = geo.compute_domain_origin_from_dem(dem_path)
    assert origin.utm_x == round(ext.sw_x, 2)
    assert origin.utm_y == round(ext.sw_y, 2)
    assert 1 <= origin.utm_zone <= 60
    assert origin.utm_zone_letter in geo._UTM_LAT_BANDS


def test_compute_domain_cells_formula(dem_path):
    params = WindsParameters()
    params.simulation_parameters.halo_x = 40.0
    params.simulation_parameters.halo_y = 40.0
    params.simulation_parameters.cell_size = (2.0, 2.0, 0.5)

    ext = geo.dem_extent(dem_path)
    nx, ny, nz = geo.compute_domain_cells(params, dem_path, shp=None, z_margin=20.0)

    assert nx == math.ceil((ext.width_m + 80.0) / 2.0)
    assert ny == math.ceil((ext.height_m + 80.0) / 2.0)
    assert nz >= 1


def test_compute_sensor_north_coords(dem_path):
    ext = geo.dem_extent(dem_path)
    site_x, site_y = geo.compute_sensor_north_qes_coords(dem_path)
    assert site_x == round(ext.width_m / 2.0, 1)
    assert site_y == round(ext.height_m, 1)


def test_check_domain_rotation_raises():
    params = WindsParameters()
    params.simulation_parameters.domain_rotation = 90.0
    with pytest.raises(ValueError):
        geo.check_domain_rotation(params)


def test_prepare_buildings_clipped(dem_path, buildings_src, mask_shp, tmp_path):
    pytest.importorskip("geopandas")
    out = geo.prepare_buildings_clipped(
        dem_path, buildings_src, mask_shp, tmp_path / "buildings_clipped.shp"
    )
    assert out.is_file()

    import geopandas as gpd

    clipped = gpd.read_file(out)
    dem_crs = geo.get_dem_srs(dem_path)
    assert clipped.crs == dem_crs
