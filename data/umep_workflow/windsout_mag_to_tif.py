#!/usr/bin/env python3
"""Export QES-Winds windsOut.nc velocity magnitude to a georeferenced GeoTIFF.

Extracts mag at a fixed height above ground (AGL) on the native QES grid.
Dependencies: numpy, netCDF4, rasterio
"""

from __future__ import annotations

import argparse
import os
import warnings
import xml.etree.ElementTree as ET
from typing import Optional

import netCDF4
import numpy as np
import rasterio
from rasterio.transform import from_origin

NODATA = -9999.0

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
DEFAULT_NC = os.path.join(_REPO_ROOT, "data/umep_workflow/output/umep_larochelle_windsOut.nc")
DEFAULT_XML = os.path.join(_REPO_ROOT, "data/umep_workflow/qes/umep_larochelle.xml")


def _parse_qes_xml(xml_path: str) -> tuple[float, float, float, float, str]:
    """Return dx, dy, halo_x, halo_y, and relative DEM path from a QES XML.

    Domain origin (UTMx/UTMy) is intentionally not read: the template XML is not
    updated by run_qeswinds.py / run_qeswinds_args.py. Use DEM SW minus halo
    (DEM content is inset by halo in the QES mesh — see DTEHeightField.cpp).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def find_text(tag: str) -> str:
        for elem in root.iter(tag):
            if elem.text and elem.text.strip():
                return elem.text.strip()
        raise ValueError(f"Could not read <{tag}> from {xml_path}")

    cell_size = find_text("cellSize").split()
    if len(cell_size) < 2:
        raise ValueError(f"Invalid <cellSize> in {xml_path}: expected 'dx dy'")
    dx = float(cell_size[0])
    dy = float(cell_size[1])
    halo_x = float(find_text("halo_x"))
    halo_y = float(find_text("halo_y"))
    dem_rel = find_text("DEM")
    return dx, dy, halo_x, halo_y, dem_rel


def _resolve_dem_path(xml_path: str, dem_rel: str, dem_override: Optional[str] = None) -> str:
    if dem_override:
        dem_path = os.path.abspath(dem_override)
        if not os.path.isfile(dem_path):
            raise FileNotFoundError(f"DEM not found: {dem_path}")
        return dem_path

    qes_dir = os.path.dirname(os.path.abspath(xml_path))
    dem_path = os.path.abspath(os.path.join(qes_dir, dem_rel))
    if os.path.isfile(dem_path):
        return dem_path

    umep_dir = os.path.dirname(qes_dir)
    candidates = [
        os.path.join(umep_dir, os.path.basename(dem_rel)),
        os.path.join(umep_dir, "DEM_clip.tif"),
        os.path.join(umep_dir, "DEM_flat_zero.tif"),
        os.path.join(umep_dir, "DEM_flat.tif"),
    ]
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isfile(candidate):
            warnings.warn(
                f"DEM from XML not found ({dem_path}), using {candidate}",
                stacklevel=2,
            )
            return candidate

    return dem_path


def _default_output_path(nc_path: str, agl_height: float) -> str:
    base = os.path.basename(nc_path)
    for suffix in ("_windsOut.nc", ".nc"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return os.path.join(
        os.path.dirname(os.path.abspath(nc_path)),
        f"{base}_mag_{agl_height:g}m.tif",
    )


def _load_terrain(
    ds: netCDF4.Dataset,
    dem_path: str,
    ny: int,
    nx: int,
) -> np.ndarray:
    if "terrain" in ds.variables:
        terrain = np.asarray(ds.variables["terrain"][:], dtype=np.float64)
        if terrain.shape != (ny, nx):
            raise ValueError(f"terrain shape {terrain.shape} != mag spatial shape ({ny}, {nx})")
        return terrain

    with rasterio.open(dem_path) as dem_ds:
        if dem_ds.height != ny or dem_ds.width != nx:
            raise ValueError(
                f"terrain not in NetCDF and DEM shape ({dem_ds.height}, {dem_ds.width}) "
                f"!= mag shape ({ny}, {nx})"
            )
        warnings.warn(
            "terrain not in NetCDF, using elevations from DEM",
            stacklevel=2,
        )
        # DEM is north-up; QES mag/terrain NetCDF are south-up (j=0 at south).
        return np.flipud(dem_ds.read(1).astype(np.float64))


def _select_mag_at_agl(
    mag: np.ndarray,
    z_levels: np.ndarray,
    terrain: np.ndarray,
    agl_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select mag at nearest z level to terrain + agl_height for each pixel."""
    target_z = terrain + agl_height
    k_idx = np.abs(z_levels[:, None, None] - target_z[None, :, :]).argmin(axis=0)
    out = np.take_along_axis(mag, k_idx[None, :, :], axis=0)[0].astype(np.float32)
    return out, k_idx


def _apply_icell_mask(
    out: np.ndarray,
    icell: np.ndarray,
    k_idx: np.ndarray,
) -> np.ndarray:
    icell_sel = np.take_along_axis(icell, k_idx[None, :, :], axis=0)[0]
    return np.where(icell_sel == 1, out, NODATA)


def _write_geotiff(
    output_path: str,
    array: np.ndarray,
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    crs,
    agl_height: float,
) -> None:
    """Write GeoTIFF. ``x0, y0`` = SW corner of the full QES domain (DEM SW − halo)."""
    ny, nx = array.shape
    # north-up transform: NW corner = (x0, y0 + ny*dy)
    transform = from_origin(x0, y0 + ny * dy, dx, dy)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": nx,
        "height": ny,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "nodata": NODATA,
        "compress": "deflate",
        "tiled": True,
    }
    with rasterio.open(output_path, "w", **profile) as dst:
        # QES j=0 at south; GeoTIFF line 0 must be north.
        dst.write(np.flipud(array), 1)
        dst.set_band_description(1, f"velocity magnitude at {agl_height} m AGL (m/s)")
        dst.update_tags(1, units="m/s")


def windsout_mag_to_tif(
    nc_path: str,
    xml_path: str,
    output_path: Optional[str] = None,
    agl_height: float = 1.5,
    time_idx: int = 0,
    mask_buildings: bool = True,
    dem_path: Optional[str] = None,
) -> str:
    """Export wind magnitude at AGL height from a QES windsOut.nc to GeoTIFF."""
    nc_path = os.path.abspath(nc_path)
    xml_path = os.path.abspath(xml_path)
    if not os.path.isfile(nc_path):
        raise FileNotFoundError(f"Input NetCDF not found: {nc_path}")
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"XML not found: {xml_path}")

    if output_path is None:
        output_path = _default_output_path(nc_path, agl_height)
    else:
        output_path = os.path.abspath(output_path)

    dx, dy, halo_x, halo_y, dem_rel = _parse_qes_xml(xml_path)
    dem_path = _resolve_dem_path(xml_path, dem_rel, dem_override=dem_path)
    if not os.path.isfile(dem_path):
        umep_dir = os.path.dirname(os.path.dirname(xml_path))
        raise FileNotFoundError(
            f"DEM not found: {dem_path}\n"
            f"  XML <DEM> resolves to: {dem_rel}\n"
            f"  Available in {umep_dir}: "
            f"{', '.join(sorted(f for f in os.listdir(umep_dir) if f.lower().endswith('.tif'))) or '(none)'}\n"
            f"  Use --dem to specify the reference raster for CRS."
        )

    with netCDF4.Dataset(nc_path) as ds:
        nt = ds.dimensions["t"].size
        if time_idx < 0 or time_idx >= nt:
            raise ValueError(f"time index {time_idx} out of range (0..{nt - 1})")

        z_levels = np.asarray(ds.variables["z"][:], dtype=np.float64)
        mag = np.asarray(ds.variables["mag"][time_idx], dtype=np.float32)

        if mag.ndim != 3:
            raise ValueError(f"Expected mag shape (z, y, x), got {mag.shape}")
        nz, ny, nx = mag.shape
        if len(z_levels) != nz:
            raise ValueError(f"z level count ({len(z_levels)}) != mag z count ({nz})")

        terrain = _load_terrain(ds, dem_path, ny, nx)

        out, k_idx = _select_mag_at_agl(mag, z_levels, terrain, agl_height)

        if mask_buildings:
            icell = np.asarray(ds.variables["icell"][time_idx], dtype=np.float32)
            if icell.shape != mag.shape:
                raise ValueError(f"icell shape {icell.shape} != mag shape {mag.shape}")
            out = _apply_icell_mask(out, icell, k_idx)

    out[np.isnan(out)] = NODATA

    with rasterio.open(dem_path) as dem_ds:
        crs = dem_ds.crs
        # DEM is inset by halo in the QES mesh; mag covers the full domain.
        x0 = dem_ds.bounds.left - halo_x
        y0 = dem_ds.bounds.bottom - halo_y
    if crs is None:
        raise ValueError(f"Could not read CRS from {dem_path}")

    _write_geotiff(output_path, out, x0, y0, dx, dy, crs, agl_height)

    selected_z = z_levels[k_idx]
    agl_actual = selected_z - terrain
    target_z = terrain + agl_height
    valid = out[out != NODATA]

    print(f"windsOut.nc → GeoTIFF (mag @ {agl_height} m AGL)")
    print(f"  Input  : {nc_path}")
    print(f"  Output : {output_path}")
    print(f"  XML    : {xml_path}")
    print(f"  DEM    : {dem_path}")
    print(
        f"  Origin : domain SW x0={x0} y0={y0} "
        f"(DEM SW − halo {halo_x}x{halo_y})  cellSize={dx}x{dy} m"
    )
    print(f"  Mask buildings: {'yes' if mask_buildings else 'no'}")
    print(f"  Grid   : {nx} x {ny} pixels, {len(z_levels)} z levels")
    print(
        f"  z target (AMSL): min={target_z.min():.2f} max={target_z.max():.2f} "
        f"mean={target_z.mean():.2f} m"
    )
    print(
        f"  z selected (AMSL): min={selected_z.min():.2f} max={selected_z.max():.2f} "
        f"mean={selected_z.mean():.2f} m"
    )
    print(
        f"  AGL at selected z: min={agl_actual.min():.2f} max={agl_actual.max():.2f} "
        f"mean={agl_actual.mean():.2f} m"
    )
    unique_k, counts = np.unique(k_idx, return_counts=True)
    top = sorted(zip(counts, unique_k), reverse=True)[:5]
    print(
        "  Top z-band usage (count, z_m):",
        ", ".join(f"{c}@{z_levels[k]:.2f}" for c, k in top),
    )
    if valid.size:
        print(
            f"  mag    : min={valid.min():.3f} max={valid.max():.3f} "
            f"mean={valid.mean():.3f} m/s ({valid.size} valid pixels)"
        )
    else:
        print("  mag    : no valid pixels after masking")
    print(f"Success: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract velocity magnitude (mag) at a fixed height above ground from "
            "a QES-Winds windsOut.nc file and write a georeferenced GeoTIFF."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        default=DEFAULT_NC,
        help=f"Input NetCDF (default: {DEFAULT_NC})",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output GeoTIFF (default: <input_basename>_mag_<HEIGHT>m.tif)",
    )
    parser.add_argument(
        "-x",
        "--xml",
        default=DEFAULT_XML,
        help=(
            f"QES XML for cellSize, halo_x/halo_y and <DEM> path "
            f"(domain SW = DEM SW − halo; default: {DEFAULT_XML})"
        ),
    )
    parser.add_argument(
        "-z",
        "--agl-height",
        type=float,
        default=1.5,
        help="Height above ground in metres (default: 1.5)",
    )
    parser.add_argument(
        "-t",
        "--time-index",
        type=int,
        default=0,
        help="Time index in NetCDF (default: 0)",
    )
    parser.add_argument(
        "--no-mask-buildings",
        action="store_true",
        help="Keep building/terrain cells (default: mask as NoData)",
    )
    parser.add_argument(
        "--dem",
        default=None,
        help=(
            "Reference DEM GeoTIFF for CRS and domain SW origin "
            "(default: <DEM> from XML, with fallback to DEM_clip.tif)"
        ),
    )
    args = parser.parse_args()

    windsout_mag_to_tif(
        nc_path=args.input,
        xml_path=args.xml,
        output_path=args.output,
        agl_height=args.agl_height,
        time_idx=args.time_index,
        mask_buildings=not args.no_mask_buildings,
        dem_path=args.dem,
    )


if __name__ == "__main__":
    main()
