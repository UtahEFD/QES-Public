#!/usr/bin/env python3
"""Run QES-Winds with all La Rochelle parameters as CLI / Python args (no XML input).

Defaults mirror ``qes/umep_larochelle.xml`` and ``qes/sensor_umep.xml``.
DEM defaults to ``DEM_clip.tif`` (``../DEM.tif`` in the XML is missing).
``domainRotation`` defaults to ``0`` (XML has ``90``, but QES-Winds crashes
when rotation != 0 — same guard as ``run_qeswinds.sh``).

Usage (from the repo root, after ``uv sync``)::

    uv run python data/umep_workflow/run_qeswinds_args.py
    uv run python data/umep_workflow/run_qeswinds_args.py --speed 5 --direction 180
    uv run python data/umep_workflow/run_qeswinds_args.py --no-preprocess
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DEM = HERE / "DEM_clip.tif"
DEFAULT_SHP = HERE / "qes" / "buildings_clipped.shp"
DEFAULT_BUILDINGS_SRC = HERE / "buildings.shp"
DEFAULT_MASK = HERE / "mask.shp"
DEFAULT_OUT = HERE / "output"
OUT_BASENAME = "umep_larochelle"


def _triple_int(text: str) -> tuple[int, int, int]:
    parts = text.split()
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected 3 ints, got {text!r}")
    return int(parts[0]), int(parts[1]), int(parts[2])


def _triple_float(text: str) -> tuple[float, float, float]:
    parts = text.split()
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected 3 floats, got {text!r}")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run QES-Winds for umep_workflow without reading XML — "
            "all parameters from umep_larochelle.xml / sensor_umep.xml as arguments."
        )
    )

    # --- run options ---
    p.add_argument("--solver", choices=("cpu", "gpu"), default="cpu")
    p.add_argument("--work-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip origin/domain/sensor placement and buildings clip.",
    )
    p.add_argument("--buildings-src", type=Path, default=DEFAULT_BUILDINGS_SRC)
    p.add_argument("--buildings-mask", type=Path, default=DEFAULT_MASK)

    # --- simulationParameters ---
    sim = p.add_argument_group("simulationParameters")
    sim.add_argument("--dem", type=Path, default=DEFAULT_DEM)
    sim.add_argument("--halo-x", type=float, default=40.0)
    sim.add_argument("--halo-y", type=float, default=40.0)
    sim.add_argument("--domain", type=_triple_int, default=(146, 99, 89), metavar="NX NY NZ")
    sim.add_argument(
        "--cell-size", type=_triple_float, default=(2.0, 2.0, 0.5), metavar="DX DY DZ"
    )
    sim.add_argument("--vertical-stretching", type=int, default=0)
    sim.add_argument("--total-time-increments", type=int, default=1)
    sim.add_argument("--rooftop-flag", type=int, default=1)
    sim.add_argument("--upwind-cavity-flag", type=int, default=1)
    sim.add_argument("--street-canyon-flag", type=int, default=1)
    sim.add_argument("--street-intersection-flag", type=int, default=0)
    sim.add_argument("--wake-flag", type=int, default=2)
    sim.add_argument("--high-rise-flag", type=int, default=0)
    sim.add_argument("--sidewall-flag", type=int, default=1)
    sim.add_argument("--log-law-flag", type=int, default=1)
    sim.add_argument("--max-iterations", type=int, default=500)
    sim.add_argument("--tolerance", type=float, default=1e-9)
    sim.add_argument("--mesh-type-flag", type=int, default=0)
    sim.add_argument(
        "--domain-rotation",
        type=float,
        default=0.0,
        help="XML has 90; must be 0 for a successful run (default: 0).",
    )
    sim.add_argument("--origin-flag", type=int, default=0)
    sim.add_argument("--dem-distance-x", type=float, default=0.0)
    sim.add_argument("--dem-distance-y", type=float, default=0.0)
    sim.add_argument("--utm-x", type=float, default=0)
    sim.add_argument("--utm-y", type=float, default=0)
    sim.add_argument("--utm-zone", type=int, default=30)
    sim.add_argument("--utm-zone-letter", type=str, default="T")
    sim.add_argument("--read-coefficients-flag", type=int, default=0)

    # --- metParams ---
    met = p.add_argument_group("metParams")
    met.add_argument("--z0-domain-flag", type=int, default=0)

    # --- buildingsParams ---
    bld = p.add_argument_group("buildingsParams")
    bld.add_argument("--wall-roughness", type=float, default=0.01)
    bld.add_argument("--bld-rooftop-flag", type=int, default=1)
    bld.add_argument("--bld-upwind-cavity-flag", type=int, default=2)
    bld.add_argument("--bld-street-canyon-flag", type=int, default=1)
    bld.add_argument("--bld-street-intersection-flag", type=int, default=0)
    bld.add_argument("--bld-wake-flag", type=int, default=2)
    bld.add_argument("--bld-high-rise-flag", type=int, default=0)
    bld.add_argument("--bld-sidewall-flag", type=int, default=1)
    bld.add_argument("--shp-file", type=Path, default=DEFAULT_SHP)
    bld.add_argument("--shp-building-layer", type=str, default="buildings_clipped")
    bld.add_argument("--shp-height-field", type=str, default="hauteur")
    bld.add_argument("--height-factor", type=float, default=1.0)

    # --- turbParams ---
    turb = p.add_argument_group("turbParams")
    turb.add_argument("--turb-method", type=int, default=0)

    # --- fileOptions ---
    fo = p.add_argument_group("fileOptions")
    fo.add_argument("--output-flag", type=int, default=1)
    fo.add_argument(
        "--output-fields",
        nargs="+",
        default=["all", "u", "v", "w", "icell", "mag"],
        metavar="FIELD",
    )

    # --- sensor ---
    sens = p.add_argument_group("sensor (sensor_umep.xml)")
    sens.add_argument("--site-coord-flag", type=int, default=1)
    sens.add_argument("--site-xcoord", type=float, default=511.8)
    sens.add_argument("--site-ycoord", type=float, default=603.4)
    sens.add_argument("--time-stamp", type=str, default="2025-07-01T12:00:00")
    sens.add_argument("--boundary-layer-flag", type=int, default=2)
    sens.add_argument("--site-z0", type=float, default=0.24)
    sens.add_argument("--reciprocal", type=float, default=0.0)
    sens.add_argument("--height", type=float, default=10.0)
    sens.add_argument("--speed", type=float, default=3.0)
    sens.add_argument("--direction", type=float, default=270.0)
    sens.add_argument("--canopy-height", type=float, default=3.0)
    sens.add_argument("--attenuation-coefficient", type=float, default=1.0)

    return p.parse_args()


def build_config(args: argparse.Namespace):
    """Build WindsParameters + SensorParameters from parsed CLI args."""
    from pyQES.util.config import (
        BuildingsParams,
        FileOptions,
        MetParams,
        SensorParameters,
        SimulationParameters,
        TimeSeries,
        TurbParams,
        WindsParameters,
    )

    params = WindsParameters(
        simulation_parameters=SimulationParameters(
            dem=str(args.dem.resolve()),
            halo_x=args.halo_x,
            halo_y=args.halo_y,
            domain=args.domain,
            cell_size=args.cell_size,
            vertical_stretching=args.vertical_stretching,
            total_time_increments=args.total_time_increments,
            rooftop_flag=args.rooftop_flag,
            upwind_cavity_flag=args.upwind_cavity_flag,
            street_canyon_flag=args.street_canyon_flag,
            street_intersection_flag=args.street_intersection_flag,
            wake_flag=args.wake_flag,
            high_rise_flag=args.high_rise_flag,
            sidewall_flag=args.sidewall_flag,
            log_law_flag=args.log_law_flag,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            mesh_type_flag=args.mesh_type_flag,
            domain_rotation=args.domain_rotation,
            origin_flag=args.origin_flag,
            dem_distance_x=args.dem_distance_x,
            dem_distance_y=args.dem_distance_y,
            utm_x=args.utm_x,
            utm_y=args.utm_y,
            utm_zone=args.utm_zone,
            utm_zone_letter=args.utm_zone_letter,
            read_coefficients_flag=args.read_coefficients_flag,
        ),
        met_params=MetParams(z0_domain_flag=args.z0_domain_flag, sensor_names=[]),
        buildings_params=BuildingsParams(
            wall_roughness=args.wall_roughness,
            rooftop_flag=args.bld_rooftop_flag,
            upwind_cavity_flag=args.bld_upwind_cavity_flag,
            street_canyon_flag=args.bld_street_canyon_flag,
            street_intersection_flag=args.bld_street_intersection_flag,
            wake_flag=args.bld_wake_flag,
            high_rise_flag=args.bld_high_rise_flag,
            sidewall_flag=args.bld_sidewall_flag,
            shp_file=str(args.shp_file.resolve()),
            shp_building_layer=args.shp_building_layer,
            shp_height_field=args.shp_height_field,
            height_factor=args.height_factor,
        ),
        turb_params=TurbParams(method=args.turb_method),
        file_options=FileOptions(
            output_flag=args.output_flag,
            output_fields=list(args.output_fields),
        ),
    )

    sensor = SensorParameters(
        site_coord_flag=args.site_coord_flag,
        site_x_coord=args.site_xcoord,
        site_y_coord=args.site_ycoord,
        time_series=[
            TimeSeries(
                time_stamp=args.time_stamp,
                boundary_layer_flag=args.boundary_layer_flag,
                site_z0=args.site_z0,
                reciprocal=args.reciprocal,
                height=args.height,
                speed=args.speed,
                direction=args.direction,
                canopy_height=args.canopy_height,
                attenuation_coefficient=args.attenuation_coefficient,
            )
        ],
    )
    return params, sensor


def main() -> int:
    args = _parse_args()
    dem = args.dem.resolve()
    if not dem.is_file():
        print(f"Error: DEM not found: {dem}", file=sys.stderr)
        return 1

    try:
        from pyQES import pywinds
    except ImportError as exc:
        print(
            "Error: pyQES is not installed. From the repo root run:\n"
            "  uv sync\n"
            f"({exc})",
            file=sys.stderr,
        )
        return 1

    params, sensor = build_config(args)
    auto_preprocess = not args.no_preprocess

    kwargs: dict = {
        "config": params,
        "sensor": sensor,
        "solver": args.solver,
        "out_basename": OUT_BASENAME,
        "winds_out": True,
        "workspace": True,
        "turb": False,
        "auto_preprocess": auto_preprocess,
        "work_dir": args.work_dir.resolve(),
    }
    if auto_preprocess:
        kwargs["buildings_src"] = args.buildings_src.resolve()
        kwargs["buildings_mask"] = args.buildings_mask.resolve()

    if auto_preprocess:
        from pyQES.util import geo

        origin = geo.compute_domain_origin_from_dem(dem)
        utm_x, utm_y = origin.utm_x, origin.utm_y
        utm_zone, utm_letter = origin.utm_zone, origin.utm_zone_letter
    else:
        sim = params.simulation_parameters
        utm_x, utm_y = sim.utm_x, sim.utm_y
        utm_zone, utm_letter = sim.utm_zone, sim.utm_zone_letter

    print(f"DEM:      {dem}")
    print(f"SHP:      {params.buildings_params.shp_file}")
    print(f"Origin:   UTMx={utm_x} UTMy={utm_y} UTMZone={utm_zone}{utm_letter}")
    print(f"Solver:   {args.solver}")
    print(f"Wind:     {args.speed} m/s @ {args.direction}°")
    print(f"Preproc:  {auto_preprocess}")
    print(f"Work dir: {kwargs['work_dir']}")

    result = pywinds.run(**kwargs)

    print("Done.")
    if getattr(result, "winds_out", None):
        print(f"  winds_out: {result.winds_out}")
    if getattr(result, "winds_wk", None):
        print(f"  winds_wk:  {result.winds_wk}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
