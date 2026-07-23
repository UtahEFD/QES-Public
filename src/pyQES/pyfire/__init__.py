"""QES-Fire Python interface.

Runs the coupled QES-Winds / QES-Fire (optionally QES-Turb and smoke QES-Plume)
workflow. The winds/fire parameters come from a QES XML file; an optional plume
XML enables smoke transport. Reuses :func:`pyQES.pywinds.run`-style preprocessing
when ``auto_preprocess`` is requested.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .._extensions import load_extension, solver_code
from ..pywinds import _apply_overrides, _finalize_paths, _preprocess, _resolve_params
from ..util.config import SensorParameters, WindsParameters
from ..util.paths import resolve_work_dir
from ..util.xml_io import write_qes_xml

__all__ = ["run"]


def run(
    config: WindsParameters | dict | None = None,
    *,
    xml: str | Path | None = None,
    json: str | bytes | dict | None = None,
    plume_xml: str | Path | None = None,
    solver: str | int = "cpu",
    out_basename: str = "qes",
    turb: bool = False,
    fire_winds_off: bool = False,
    auto_preprocess: bool = True,
    work_dir: str | Path | None = None,
    dem: str | Path | None = None,
    buildings_src: str | Path | None = None,
    buildings_mask: str | Path | None = None,
    sensor: SensorParameters | str | Path | list | None = None,
    z_margin: float = 20.0,
    **overrides: Any,
):
    """Run the coupled QES-Fire workflow and return the output NetCDF paths.

    :param plume_xml: QES-Plume XML enabling smoke transport (optional).
    :param fire_winds_off: Disable fire-induced winds.
    See :func:`pyQES.pywinds.run` for the shared configuration/preprocessing args.
    :returns: The native ``FireRunResult`` (``fire_out``/``plume_out``).
    """
    ext = load_extension("_fire")
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
    else:
        params, base_dir = _resolve_params(config, xml, json)
        _apply_overrides(params, overrides)
        if dem is not None:
            params.simulation_parameters.dem = str(Path(dem).resolve())

        if auto_preprocess:
            _preprocess(
                params,
                work,
                base_dir,
                buildings_src=buildings_src,
                buildings_mask=buildings_mask,
                sensor=sensor,
                z_margin=z_margin,
            )
        else:
            _finalize_paths(params, work, base_dir, sensor)

        xml_path = str(write_qes_xml(params, work / f"{out_basename}.xml"))

    out_prefix = str(work / out_basename)
    return ext.run_fire(
        winds_xml_path=xml_path,
        plume_xml_path=str(Path(plume_xml).resolve()) if plume_xml else "",
        solve_type=solver_code(solver),
        out_basename=out_prefix,
        comp_turb=turb,
        fire_winds_off=fire_winds_off,
    )
