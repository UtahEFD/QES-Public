"""Shared pytest fixtures for the pyQES test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
UMEP_DIR = REPO_ROOT / "data" / "umep_workflow"
QES_DIR = UMEP_DIR / "qes"


@pytest.fixture(scope="session")
def umep_dir() -> Path:
    return UMEP_DIR


@pytest.fixture(scope="session")
def dem_path() -> Path:
    path = UMEP_DIR / "DEM_flat_zero.tif"
    if not path.is_file():
        pytest.skip(f"DEM asset missing: {path}")
    return path


@pytest.fixture(scope="session")
def winds_xml() -> Path:
    path = QES_DIR / "umep_larochelle.xml"
    if not path.is_file():
        pytest.skip(f"QES XML asset missing: {path}")
    return path


@pytest.fixture(scope="session")
def sensor_xml() -> Path:
    path = QES_DIR / "sensor_umep.xml"
    if not path.is_file():
        pytest.skip(f"Sensor XML asset missing: {path}")
    return path


@pytest.fixture(scope="session")
def buildings_src() -> Path:
    path = UMEP_DIR / "buildings.shp"
    if not path.is_file():
        pytest.skip(f"Buildings shapefile missing: {path}")
    return path


@pytest.fixture(scope="session")
def mask_shp() -> Path:
    path = UMEP_DIR / "mask.shp"
    if not path.is_file():
        pytest.skip(f"Mask shapefile missing: {path}")
    return path
