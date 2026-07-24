"""QES-Winds Python interface.

Provides the :class:`~pyQES.util.config.WindsParameters` model plus :func:`run`,
which accepts a QES XML file, a JSON document, a config object or direct keyword
arguments, optionally runs the geospatial preprocessing (DEM origin, domain cell
counts, north-centered sensor placement, buildings clipping) and then calls the
native QES-Winds solver.

After :func:`run`, :func:`to_tif` exports wind magnitude from the last run's
``windsOut.nc`` to a georeferenced GeoTIFF using the in-memory simulation
parameters (not re-parsed from XML).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .._extensions import load_extension, solver_code
from ..util import geo
from ..util.config import SensorParameters, WindsParameters
from ..util.paths import resolve_work_dir
from ..util.xml_io import from_qes_xml, from_sensor_xml, write_qes_xml, write_sensor_xml

__all__ = ["WindsParameters", "SensorParameters", "run", "to_tif"]


@dataclass
class _WindsRunContext:
    """Geometry and sensor state retained after the latest :func:`run`."""

    winds_out: str
    out_basename: str
    params: WindsParameters
    sensors: list[SensorParameters] = field(default_factory=list)


_last_run: _WindsRunContext | None = None


def _resolve_params(
    config: WindsParameters | dict | None,
    xml: str | Path | None,
    json: str | bytes | dict | None,
) -> tuple[WindsParameters, Path]:
    """Build a WindsParameters model and a base directory for relative paths."""
    if sum(x is not None for x in (config, xml, json)) > 1:
        raise ValueError("Provide only one of 'config', 'xml' or 'json'.")

    if xml is not None:
        return from_qes_xml(xml), Path(xml).resolve().parent
    if json is not None:
        if isinstance(json, dict):
            return WindsParameters.model_validate(json), Path.cwd()
        if isinstance(json, bytes):
            return WindsParameters.from_json(json), Path.cwd()
        # str: either a path to a .json file or a JSON document.
        try:
            payload = Path(json)
            is_file = payload.is_file()
        except OSError:
            is_file = False
        if is_file:
            return WindsParameters.from_json(payload.read_text()), payload.resolve().parent
        return WindsParameters.from_json(json), Path.cwd()
    if config is not None:
        if isinstance(config, dict):
            return WindsParameters.model_validate(config), Path.cwd()
        return config, Path.cwd()
    return WindsParameters(), Path.cwd()


def _apply_overrides(params: WindsParameters, overrides: dict[str, Any]) -> None:
    """Apply keyword overrides; dotted keys traverse, plain keys are searched."""
    submodels = [
        params.simulation_parameters,
        params.met_params,
        params.buildings_params,
        params.file_options,
    ]
    if params.turb_params is not None:
        submodels.append(params.turb_params)

    for key, value in overrides.items():
        if "." in key:
            head, _, tail = key.partition(".")
            target = getattr(params, head)
            setattr(target, tail, value)
            continue
        if hasattr(params, key) and key in params.model_fields:
            setattr(params, key, value)
            continue
        for model in submodels:
            if key in type(model).model_fields:
                setattr(model, key, value)
                break
        else:
            raise AttributeError(f"Unknown parameter override: {key!r}")


def _absolute(path: str | Path, base: Path) -> str:
    """Resolve ``path`` to an absolute string, relative to ``base`` if needed."""
    p = Path(path)
    return str(p if p.is_absolute() else (base / p).resolve())


def _collect_sensors(
    sensor: SensorParameters | str | Path | list | None,
    params: WindsParameters,
    base_dir: Path,
) -> list[tuple[SensorParameters, str]]:
    """Resolve the sensor inputs into (model, output-filename) pairs."""
    if sensor is None:
        pairs: list[tuple[SensorParameters, str]] = []
        for name in params.met_params.sensor_names:
            path = Path(_absolute(name, base_dir))
            pairs.append((from_sensor_xml(path), path.name))
        return pairs

    items = sensor if isinstance(sensor, list) else [sensor]
    pairs = []
    for i, item in enumerate(items):
        if isinstance(item, SensorParameters):
            pairs.append((item, f"sensor_{i}.xml" if len(items) > 1 else "sensor.xml"))
        else:
            path = Path(_absolute(item, base_dir))
            pairs.append((from_sensor_xml(path), path.name))
    return pairs


def _write_sensors(
    pairs: list[tuple[SensorParameters, str]],
    work_dir: Path,
    params: WindsParameters,
) -> list[SensorParameters]:
    """Write sensor XMLs into ``work_dir`` and update ``params.met_params``."""
    written: list[str] = []
    sensors: list[SensorParameters] = []
    for sensor_model, name in pairs:
        out_path = work_dir / name
        write_sensor_xml(sensor_model, out_path)
        written.append(str(out_path))
        sensors.append(sensor_model)
    if written:
        params.met_params.sensor_names = written
    return sensors


def _preprocess(
    params: WindsParameters,
    work_dir: Path,
    base_dir: Path,
    *,
    buildings_src: str | Path | None,
    buildings_mask: str | Path | None,
    sensor: SensorParameters | str | Path | list | None,
    z_margin: float,
) -> list[SensorParameters]:
    """Run the DEM/shapefile preprocessing and patch the model in place."""
    sim = params.simulation_parameters
    geo.check_domain_rotation(params)

    if sim.dem is None:
        raise ValueError("No DEM set; provide 'dem=' or <DEM> in the config.")
    dem = _absolute(sim.dem, base_dir)
    sim.dem = dem

    if buildings_src is not None and buildings_mask is not None:
        out_shp = work_dir / "buildings_clipped.shp"
        geo.prepare_buildings_clipped(dem, buildings_src, buildings_mask, out_shp)
        params.buildings_params.shp_file = str(out_shp)
        params.buildings_params.shp_building_layer = out_shp.stem
    elif params.buildings_params.shp_file is not None:
        params.buildings_params.shp_file = _absolute(params.buildings_params.shp_file, base_dir)

    origin = geo.compute_domain_origin_from_dem(dem)
    sim.utm_x = origin.utm_x
    sim.utm_y = origin.utm_y
    sim.utm_zone = origin.utm_zone
    sim.utm_zone_letter = origin.utm_zone_letter

    sim.domain = geo.compute_domain_cells(params, dem, params.buildings_params.shp_file, z_margin)

    site_x, site_y = geo.compute_sensor_north_qes_coords(
        dem, sim.dem_distance_x, sim.dem_distance_y
    )
    pairs = _collect_sensors(sensor, params, base_dir)
    for sensor_model, _name in pairs:
        sensor_model.site_x_coord = site_x
        sensor_model.site_y_coord = site_y
    return _write_sensors(pairs, work_dir, params)


def _finalize_paths(
    params: WindsParameters,
    work_dir: Path,
    base_dir: Path,
    sensor: SensorParameters | str | Path | list | None,
) -> list[SensorParameters]:
    """Without preprocessing, still resolve inputs to absolute paths."""
    sim = params.simulation_parameters
    if sim.dem is not None:
        sim.dem = _absolute(sim.dem, base_dir)
    if params.buildings_params.shp_file is not None:
        params.buildings_params.shp_file = _absolute(params.buildings_params.shp_file, base_dir)

    pairs = _collect_sensors(sensor, params, base_dir)
    return _write_sensors(pairs, work_dir, params)


def _sensors_from_params(params: WindsParameters, base_dir: Path) -> list[SensorParameters]:
    """Load sensor models listed in ``params.met_params.sensor_names``."""
    sensors: list[SensorParameters] = []
    for name in params.met_params.sensor_names:
        sensors.append(from_sensor_xml(_absolute(name, base_dir)))
    return sensors


def _direction_from_sensors(sensors: list[SensorParameters], time_idx: int) -> float:
    """Return wind direction (degrees) from the first sensor at ``time_idx``."""
    if not sensors:
        raise ValueError("No sensors in the last run context; cannot resolve wind direction.")
    series = sensors[0].time_series
    if not series:
        raise ValueError("First sensor has no timeSeries; cannot resolve wind direction.")
    if time_idx < 0 or time_idx >= len(series):
        raise ValueError(
            f"time index {time_idx} out of range for sensor timeSeries (0..{len(series) - 1})"
        )
    return series[time_idx].direction


def run(
    config: WindsParameters | dict | None = None,
    *,
    xml: str | Path | None = None,
    json: str | bytes | dict | None = None,
    solver: str | int = "cpu",
    out_basename: str = "qes",
    winds_out: bool = True,
    workspace: bool = True,
    turb: bool = False,
    auto_preprocess: bool = True,
    work_dir: str | Path | None = None,
    dem: str | Path | None = None,
    buildings_src: str | Path | None = None,
    buildings_mask: str | Path | None = None,
    sensor: SensorParameters | str | Path | list | None = None,
    z_margin: float = 20.0,
    **overrides: Any,
):
    """Run QES-Winds and return the output NetCDF paths.

    Exactly one of ``config``, ``xml`` or ``json`` may be given as the base
    configuration (or none, to build purely from keyword overrides).

    Stores a module-level context used by :func:`to_tif`.

    :param solver: ``"cpu"``/``"gpu"`` (or the QES solver-type int).
    :param out_basename: Basename for the produced NetCDF files.
    :param auto_preprocess: Compute origin/domain/sensor and clip buildings.
    :param dem: DEM raster path (overrides ``<DEM>``).
    :param buildings_src, buildings_mask: Raw buildings + mask to clip.
    :param sensor: Sensor model/path (or list) to place at the DEM north center.
    :param overrides: Extra parameter overrides (dotted or plain keys).
    :returns: The native ``WindsRunResult`` (``winds_out``/``winds_wk``/``turb_out``).
    """
    global _last_run

    ext = load_extension("_winds")
    work = resolve_work_dir(work_dir)

    direct_xml = (
        xml is not None
        and not auto_preprocess
        and config is None
        and json is None
        and not overrides
        and dem is None
        and sensor is None
    )
    if direct_xml:
        assert xml is not None  # guaranteed by direct_xml
        xml_path = str(Path(xml).resolve())
        params = from_qes_xml(xml_path)
        sensors = _sensors_from_params(params, Path(xml_path).parent)
    else:
        params, base_dir = _resolve_params(config, xml, json)
        _apply_overrides(params, overrides)
        if dem is not None:
            params.simulation_parameters.dem = str(Path(dem).resolve())

        if auto_preprocess:
            sensors = _preprocess(
                params,
                work,
                base_dir,
                buildings_src=buildings_src,
                buildings_mask=buildings_mask,
                sensor=sensor,
                z_margin=z_margin,
            )
        else:
            sensors = _finalize_paths(params, work, base_dir, sensor)

        xml_path = str(write_qes_xml(params, work / f"{out_basename}.xml"))

    out_prefix = str(work / out_basename)
    result = ext.run_winds(
        xml_path=xml_path,
        solve_type=solver_code(solver),
        out_basename=out_prefix,
        visu_output=winds_out,
        wksp_output=workspace,
        turb_output=turb,
    )
    _last_run = _WindsRunContext(
        winds_out=getattr(result, "winds_out", "") or "",
        out_basename=out_basename,
        params=params,
        sensors=sensors,
    )
    return result


def to_tif(
    z: float = 1.5,
    *,
    time_idx: int = 0,
    mask_buildings: bool = True,
    output_path: str | Path | None = None,
) -> str:
    """Export velocity magnitude at ``z`` m AGL from the last :func:`run`.

    Geometry (``dx``, ``dy``, ``halo_x``, ``halo_y``, DEM) is taken from the
    in-memory :class:`~pyQES.util.config.SimulationParameters` of that run.
    Wind direction for the default filename comes from the first sensor's
    ``timeSeries[time_idx].direction``.

    :returns: Absolute path of the written GeoTIFF
        (``{out_basename}_Vmag_z{z}m_dir{dir}.tif`` by default).
    """
    if _last_run is None:
        raise RuntimeError("No pywinds.run() context; call run() before to_tif().")

    winds_out = _last_run.winds_out
    if not winds_out:
        raise RuntimeError("No winds_out NetCDF in the last run; call run(winds_out=True).")
    if not Path(winds_out).is_file():
        raise FileNotFoundError(f"winds_out NetCDF not found: {winds_out}")

    sim = _last_run.params.simulation_parameters
    if sim.dem is None:
        raise ValueError("No DEM in the last run parameters; cannot georeference the GeoTIFF.")

    from ..util.outputs import default_mag_tif_name, mag_to_tif

    dx, dy = sim.cell_size[0], sim.cell_size[1]
    direction = _direction_from_sensors(_last_run.sensors, time_idx)

    if output_path is None:
        out_dir = Path(winds_out).resolve().parent
        output_path = out_dir / default_mag_tif_name(_last_run.out_basename, z, direction)

    return mag_to_tif(
        winds_out,
        dx=dx,
        dy=dy,
        halo_x=sim.halo_x,
        halo_y=sim.halo_y,
        dem=sim.dem,
        output_path=output_path,
        agl_height=z,
        time_idx=time_idx,
        mask_buildings=mask_buildings,
    )
