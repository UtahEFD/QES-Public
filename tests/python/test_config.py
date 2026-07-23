"""Unit tests for the pyQES pydantic configuration models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pyQES.util.config import SimulationParameters, WindsParameters


def test_defaults_and_types():
    params = WindsParameters()
    assert params.simulation_parameters.domain == (1, 1, 1)
    assert params.simulation_parameters.cell_size == (1.0, 1.0, 1.0)
    assert params.file_options.output_fields == ["all"]
    assert params.turb_params is None


def test_json_round_trip():
    params = WindsParameters()
    params.simulation_parameters.domain = (10, 20, 30)
    params.simulation_parameters.halo_x = 40.0
    restored = WindsParameters.from_json(params.to_json())
    assert restored == params


def test_alias_population_and_space_separated_vector():
    params = WindsParameters.model_validate(
        {"simulationParameters": {"domain": "10 20 30", "cellSize": "2 2 0.5"}}
    )
    assert params.simulation_parameters.domain == (10, 20, 30)
    assert params.simulation_parameters.cell_size == (2.0, 2.0, 0.5)


def test_field_name_population():
    sim = SimulationParameters(halo_x=15.0, domain=(5, 6, 7))
    assert sim.halo_x == 15.0
    assert sim.domain == (5, 6, 7)


def test_invalid_domain_raises():
    with pytest.raises(ValidationError):
        SimulationParameters(domain=(1, 2))  # wrong arity
