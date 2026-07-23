#!/usr/bin/env python3
"""Export wind magnitude at AGL height from windsOut.nc to GeoTIFF.

Workflow example for the umep La Rochelle case. Delegates to
``windsout_mag_to_tif.windsout_mag_to_tif``.

Usage (from the repo root, after ``uv sync``)::

    uv run python data/umep_workflow/run_mag_to_tif.py -z 1.5
    uv run python data/umep_workflow/run_mag_to_tif.py -z 10 -o data/umep_workflow/output/mag_10m.tif
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_NC = HERE / "output" / "umep_larochelle_windsOut.nc"
DEFAULT_XML = HERE / "qes" / "umep_larochelle.xml"
DEFAULT_DEM = HERE / "DEM_clip.tif"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extract velocity magnitude at a fixed height above ground from "
            "a QES-Winds windsOut.nc and write a georeferenced GeoTIFF."
        )
    )
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_NC,
        help=f"Input windsOut.nc (default: {DEFAULT_NC.relative_to(HERE)}).",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output GeoTIFF (default: <input_basename>_mag_<HEIGHT>m.tif).",
    )
    p.add_argument(
        "-x",
        "--xml",
        type=Path,
        default=DEFAULT_XML,
        help=(
            "QES XML for cellSize, halo and <DEM> path "
            f"(domain SW = DEM SW − halo; default: {DEFAULT_XML.relative_to(HERE)})."
        ),
    )
    p.add_argument(
        "-z",
        "--agl-height",
        type=float,
        default=1.5,
        help="Height above ground in metres (default: 1.5).",
    )
    p.add_argument(
        "-t",
        "--time-index",
        type=int,
        default=0,
        help="Time index in NetCDF (default: 0).",
    )
    p.add_argument(
        "--dem",
        type=Path,
        default=DEFAULT_DEM,
        help=(
            f"Reference DEM for CRS and domain SW origin (default: {DEFAULT_DEM.name})."
        ),
    )
    p.add_argument(
        "--no-mask-buildings",
        action="store_true",
        help="Keep building/terrain cells (default: mask as NoData).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    nc_path = args.input.resolve()
    xml_path = args.xml.resolve()
    dem_path = args.dem.resolve()

    if not nc_path.is_file():
        print(f"Error: NetCDF not found: {nc_path}", file=sys.stderr)
        return 1
    if not xml_path.is_file():
        print(f"Error: XML not found: {xml_path}", file=sys.stderr)
        return 1
    if not dem_path.is_file():
        print(f"Error: DEM not found: {dem_path}", file=sys.stderr)
        return 1

    try:
        from windsout_mag_to_tif import windsout_mag_to_tif
    except ImportError as exc:
        print(
            "Error: could not import windsout_mag_to_tif (needs numpy, netCDF4, "
            "rasterio). From the repo root run:\n"
            "  uv sync\n"
            f"({exc})",
            file=sys.stderr,
        )
        return 1

    output_path = args.output.resolve() if args.output is not None else None

    print(f"Input:  {nc_path}")
    print(f"XML:    {xml_path}")
    print(f"DEM:    {dem_path}")
    print(f"AGL:    {args.agl_height:g} m")
    print(f"Time:   {args.time_index}")

    out = windsout_mag_to_tif(
        nc_path=str(nc_path),
        xml_path=str(xml_path),
        output_path=str(output_path) if output_path is not None else None,
        agl_height=args.agl_height,
        time_idx=args.time_index,
        mask_buildings=not args.no_mask_buildings,
        dem_path=str(dem_path),
    )
    print(f"Done: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
