"""Tests for pydantic <-> QES XML conversion."""

from __future__ import annotations

from pyQES.util.xml_io import (
    from_qes_xml,
    from_sensor_xml,
    to_qes_xml,
    write_qes_xml,
)


def test_parse_umep_winds_xml(winds_xml):
    params = from_qes_xml(winds_xml)
    sim = params.simulation_parameters
    assert sim.domain == (146, 99, 89)
    assert sim.cell_size == (2.0, 2.0, 0.5)
    assert sim.halo_x == 40.0
    assert sim.halo_y == 40.0
    assert sim.domain_rotation == 90.0

    assert params.buildings_params.shp_height_field == "hauteur"
    assert params.buildings_params.shp_building_layer == "buildings_clipped"
    assert params.met_params.sensor_names == ["sensor_umep.xml"]
    assert "mag" in params.file_options.output_fields


def test_winds_xml_round_trip(winds_xml, tmp_path):
    params = from_qes_xml(winds_xml)
    out = write_qes_xml(params, tmp_path / "round_trip.xml")
    restored = from_qes_xml(out)
    assert restored == params


def test_to_qes_xml_contains_expected_tags():
    from pyQES.util.config import WindsParameters

    params = WindsParameters()
    params.simulation_parameters.domain = (12, 34, 56)
    xml = to_qes_xml(params)
    assert "<QESWindsParameters>" in xml
    assert "<simulationParameters>" in xml
    assert "<domain>12 34 56</domain>" in xml


def test_parse_sensor_xml(sensor_xml):
    sensor = from_sensor_xml(sensor_xml)
    assert sensor.site_coord_flag == 1
    assert len(sensor.time_series) == 1
    ts = sensor.time_series[0]
    assert ts.speed == 3.0
    assert ts.direction == 270.0
