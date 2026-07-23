#!/usr/bin/env python3
"""Run QES-Winds on the umep_workflow La Rochelle case via pyQES.

Mirror of ``run_qeswinds.sh`` using :func:`pyQES.pywinds.run`.

Usage (from the repo root, after ``uv sync``)::

    uv run python data/umep_workflow/run_qeswinds.py
    uv run python data/umep_workflow/run_qeswinds.py --solver cpu --dem DEM_flat.tif
    uv run python data/umep_workflow/run_qeswinds.py --no-preprocess
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_XML = HERE / "qes" / "umep_larochelle.xml"
DEFAULT_DEM = HERE / "DEM_clip.tif"
DEFAULT_BUILDINGS = HERE / "buildings.shp"
DEFAULT_MASK = HERE / "mask.shp"
DEFAULT_OUT = HERE / "output"
OUT_BASENAME = "umep_larochelle"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run QES-Winds for the umep_workflow La Rochelle case (pyQES)."
    )
    p.add_argument(
        "--solver",
        choices=("cpu", "gpu"),
        default="cpu",
        help="Solver type (default: cpu).",
    )
    p.add_argument(
        "--dem",
        type=Path,
        default=DEFAULT_DEM,
        help=f"DEM GeoTIFF (default: {DEFAULT_DEM.name}).",
    )
    p.add_argument(
        "--xml",
        type=Path,
        default=DEFAULT_XML,
        help=f"QES-Winds XML (default: {DEFAULT_XML.relative_to(HERE)}).",
    )
    p.add_argument(
        "--buildings-src",
        type=Path,
        default=DEFAULT_BUILDINGS,
        help=f"Source buildings shapefile (default: {DEFAULT_BUILDINGS.name}).",
    )
    p.add_argument(
        "--buildings-mask",
        type=Path,
        default=DEFAULT_MASK,
        help=f"Clip mask shapefile (default: {DEFAULT_MASK.name}).",
    )
    p.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip DEM origin / domain / buildings clip; use paths already in the XML.",
    )
    p.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT.name}/).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    xml = args.xml.resolve()
    dem = args.dem.resolve()
    if not xml.is_file():
        print(f"Error: XML not found: {xml}", file=sys.stderr)
        return 1
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

    auto_preprocess = not args.no_preprocess
    kwargs: dict = {
        "xml": xml,
        "dem": dem,
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

    print(f"XML:      {xml}")
    print(f"DEM:      {dem}")
    print(f"Solver:   {args.solver}")
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
