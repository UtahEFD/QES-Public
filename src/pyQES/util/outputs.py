"""Export QES-Winds NetCDF fields to georeferenced rasters.

Geometry (``dx``, ``dy``, ``halo_x``, ``halo_y``, DEM path) is supplied by the
caller — typically the in-memory :class:`~pyQES.util.config.SimulationParameters`
of the current :mod:`pyQES.pywinds` run — not re-parsed from XML.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import netCDF4
import numpy as np
import rasterio
from rasterio.transform import from_origin

__all__ = ["mag_to_tif", "default_mag_tif_name"]

NODATA = -9999.0


def default_mag_tif_name(basename: str, agl_height: float, direction: float) -> str:
    """Return ``{basename}_Vmag_z{h}m_dir{dir}.tif``."""
    return f"{basename}_Vmag_z{agl_height:g}m_dir{direction:.0f}.tif"


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


def _log_summary(
    *,
    nc_path: str,
    output_path: str,
    dem_path: str,
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    halo_x: float,
    halo_y: float,
    mask_buildings: bool,
    nx: int,
    ny: int,
    z_levels: np.ndarray,
    terrain: np.ndarray,
    agl_height: float,
    k_idx: np.ndarray,
    out: np.ndarray,
) -> None:
    selected_z = z_levels[k_idx]
    agl_actual = selected_z - terrain
    target_z = terrain + agl_height
    valid = out[out != NODATA]

    print(f"windsOut.nc → GeoTIFF (mag @ {agl_height} m AGL)")
    print(f"  Input  : {nc_path}")
    print(f"  Output : {output_path}")
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


def mag_to_tif(
    nc_path: str | Path,
    *,
    dx: float,
    dy: float,
    halo_x: float,
    halo_y: float,
    dem: str | Path,
    output_path: str | Path | None = None,
    agl_height: float = 1.5,
    time_idx: int = 0,
    mask_buildings: bool = True,
    verbose: bool = False,
) -> str:
    """Export wind magnitude at AGL height from a QES windsOut.nc to GeoTIFF.

    ``dx``, ``dy``, ``halo_x``, ``halo_y`` and ``dem`` must come from the active
    run binding (e.g. :class:`~pyQES.util.config.SimulationParameters`), not from
    re-parsing a QES XML file.
    """
    nc_path = os.path.abspath(str(nc_path))
    dem_path = os.path.abspath(str(dem))
    if not os.path.isfile(nc_path):
        raise FileNotFoundError(f"Input NetCDF not found: {nc_path}")
    if not os.path.isfile(dem_path):
        raise FileNotFoundError(f"DEM not found: {dem_path}")

    if output_path is None:
        base = os.path.basename(nc_path)
        for suffix in ("_windsOut.nc", ".nc"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        output_path = os.path.join(
            os.path.dirname(nc_path),
            f"{base}_mag_{agl_height:g}m.tif",
        )
    else:
        output_path = os.path.abspath(str(output_path))

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
        x0 = dem_ds.bounds.left - halo_x
        y0 = dem_ds.bounds.bottom - halo_y
    if crs is None:
        raise ValueError(f"Could not read CRS from {dem_path}")

    _write_geotiff(output_path, out, x0, y0, dx, dy, crs, agl_height)

    if verbose:
        _log_summary(
            nc_path=nc_path,
            output_path=output_path,
            dem_path=dem_path,
            x0=x0,
            y0=y0,
            dx=dx,
            dy=dy,
            halo_x=halo_x,
            halo_y=halo_y,
            mask_buildings=mask_buildings,
            nx=nx,
            ny=ny,
            z_levels=z_levels,
            terrain=terrain,
            agl_height=agl_height,
            k_idx=k_idx,
            out=out,
        )
    return output_path
