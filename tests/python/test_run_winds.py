"""End-to-end QES-Winds run test.

Marked ``slow`` and skipped unless the native extension has been compiled and the
geospatial dependencies are available.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pyQES._winds")
pytest.importorskip("rasterio")
pytest.importorskip("pyproj")

from pyQES import pywinds  # noqa: E402
from pyQES.util.config import WindsParameters  # noqa: E402
from pyQES.util.xml_io import from_sensor_xml  # noqa: E402


@pytest.mark.slow
def test_run_winds_flat_dem(dem_path, sensor_xml, tmp_path):
    sensor = from_sensor_xml(sensor_xml)

    params = WindsParameters()
    sim = params.simulation_parameters
    sim.dem = str(dem_path)
    sim.cell_size = (2.0, 2.0, 1.0)
    sim.halo_x = 20.0
    sim.halo_y = 20.0
    sim.domain_rotation = 0.0
    sim.total_time_increments = 1

    result = pywinds.run(
        config=params,
        solver="cpu",
        out_basename="test_winds",
        winds_out=True,
        workspace=True,
        turb=False,
        auto_preprocess=True,
        work_dir=tmp_path,
        sensor=sensor,
    )

    assert Path(result.winds_out).is_file()
    assert Path(result.winds_wk).is_file()
