#!/usr/bin/env bash
# Extract wind magnitude at a fixed height above ground from QES-Winds windsOut.nc → GeoTIFF.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEFAULT_XML="$REPO_ROOT/data/umep_workflow/qes/umep_larochelle.xml"
DEFAULT_NC="$REPO_ROOT/data/umep_workflow/output/umep_larochelle_windsOut.nc"
QES_DIR="$(dirname "$DEFAULT_XML")"

NC_IN="$DEFAULT_NC"
XML="$DEFAULT_XML"
OUT_TIF=""
AGL_HEIGHT="1.5"
TIME_IDX="0"
MASK_BUILDINGS=1

usage() {
  cat <<EOF
Usage: $(basename "$0") [-i windsOut.nc] [-o output.tif] [-x xml] [-z HEIGHT] [-t INDEX] [--no-mask-buildings]

Extract velocity magnitude (mag) at a fixed height above ground from a QES-Winds
windsOut.nc file and write a georeferenced GeoTIFF.

The z axis in windsOut.nc is absolute altitude (m AMSL). For each pixel the script
selects the nearest vertical level where z ≈ terrain + HEIGHT using the terrain field.

Options:
  -i FILE     Input NetCDF (default: $DEFAULT_NC)
  -o FILE     Output GeoTIFF (default: <input_basename>_mag_<HEIGHT>m.tif)
  -x FILE     QES XML for cellSize and <DEM> path (origin = DEM SW corner; default: umep_larochelle.xml)
  -z HEIGHT   Height above ground in metres (default: 1.5)
  -t INDEX    Time index in NetCDF (default: 0)
  --no-mask-buildings   Keep building/terrain cells (default: mask as NoData)

Requires: python3 with numpy + osgeo (GDAL), gdalinfo on PATH
  Set QES_PYTHON to pick a specific interpreter (e.g. python3.14 on Homebrew).
EOF
}

read_xml_tag() {
  local file="$1" tag="$2"
  sed -n "s/.*<${tag}>[[:space:]]*\\([^<]*\\)[[:space:]]*<\\/${tag}>.*/\\1/p" "$file" | head -1
}

resolve_dem_path() {
  local xml="$1"
  local dem_rel
  dem_rel="$(sed -n 's/.*<DEM>[[:space:]]*\([^<]*\)[[:space:]]*<\/DEM>.*/\1/p' "$xml" | head -1)"
  if [[ -z "$dem_rel" ]]; then
    echo "Error: could not read <DEM> from $xml" >&2
    return 1
  fi
  echo "$(cd "$QES_DIR" && cd "$(dirname "$dem_rel")" && pwd)/$(basename "$dem_rel")"
}

check_dependencies() {
  PYTHON=""
  if [[ -n "${QES_PYTHON:-}" ]] && "$QES_PYTHON" -c "import numpy; from osgeo import gdal" 2>/dev/null; then
    PYTHON="$QES_PYTHON"
  else
    for candidate in python3.14 python3.13 python3.12 python3.11 python3; do
      if command -v "$candidate" >/dev/null 2>&1 \
        && "$candidate" -c "import numpy; from osgeo import gdal" 2>/dev/null; then
        PYTHON="$candidate"
        break
      fi
    done
  fi
  if [[ -z "$PYTHON" ]]; then
    echo "Error: no python with numpy and osgeo (GDAL) found." >&2
    echo "Install GDAL Python bindings or set QES_PYTHON to a suitable interpreter." >&2
    exit 1
  fi
  if ! command -v gdalinfo >/dev/null 2>&1; then
    echo "Error: gdalinfo not found (install GDAL)" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -i) NC_IN="$2"; shift 2 ;;
    -o) OUT_TIF="$2"; shift 2 ;;
    -x) XML="$2"; QES_DIR="$(dirname "$2")"; shift 2 ;;
    -z) AGL_HEIGHT="$2"; shift 2 ;;
    -t) TIME_IDX="$2"; shift 2 ;;
    --no-mask-buildings) MASK_BUILDINGS=0; shift ;;
    -h) usage; exit 0 ;;
    *) echo "Error: unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

check_dependencies

if [[ ! -f "$NC_IN" ]]; then
  echo "Error: input NetCDF not found: $NC_IN" >&2
  exit 1
fi
if [[ ! -f "$XML" ]]; then
  echo "Error: XML not found: $XML" >&2
  exit 1
fi

if [[ -z "$OUT_TIF" ]]; then
  base="$(basename "$NC_IN")"
  base="${base%_windsOut.nc}"
  base="${base%.nc}"
  OUT_TIF="$(dirname "$NC_IN")/${base}_mag_${AGL_HEIGHT}m.tif"
fi

DEM_ABS="$(resolve_dem_path "$XML")"
if [[ ! -f "$DEM_ABS" ]]; then
  echo "Error: DEM not found at: $DEM_ABS" >&2
  exit 1
fi

CELL_SIZE="$(read_xml_tag "$XML" cellSize)"
DX="$(echo "$CELL_SIZE" | awk '{print $1}')"
DY="$(echo "$CELL_SIZE" | awk '{print $2}')"
HALO_X="$(read_xml_tag "$XML" halo_x)"
HALO_Y="$(read_xml_tag "$XML" halo_y)"
HALO_X="${HALO_X:-0}"
HALO_Y="${HALO_Y:-0}"

# Domain SW = DEM SW − halo (DEM is inset by halo in the QES mesh).
read -r X0 Y0 < <(
  python3 - "$DEM_ABS" "$HALO_X" "$HALO_Y" <<'PY'
import json, subprocess, sys
dem, hx, hy = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
info = json.loads(subprocess.check_output(["gdalinfo", "-json", dem], text=True))
c = info["cornerCoordinates"]["lowerLeft"]
print(c[0] - hx, c[1] - hy)
PY
)

mkdir -p "$(dirname "$OUT_TIF")"

echo "windsOut.nc → GeoTIFF (mag @ ${AGL_HEIGHT} m AGL)"
echo "  Input  : $NC_IN"
echo "  Output : $OUT_TIF"
echo "  XML    : $XML"
echo "  DEM    : $DEM_ABS"
echo "  Origin : domain SW x0=$X0 y0=$Y0 (DEM SW − halo ${HALO_X}x${HALO_Y})  cellSize=${DX}x${DY} m"
echo "  Mask buildings: $([[ "$MASK_BUILDINGS" -eq 1 ]] && echo yes || echo no)"

"$PYTHON" - "$NC_IN" "$OUT_TIF" "$DEM_ABS" "$X0" "$Y0" "$DX" "$DY" "$AGL_HEIGHT" "$TIME_IDX" "$MASK_BUILDINGS" <<'PY'
import json
import re
import subprocess
import sys

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

nc_path, out_tif, dem_path, x0, y0, dx, dy, agl_height, time_idx, mask_buildings = sys.argv[1:11]
x0 = float(x0)
y0 = float(y0)
dx = float(dx)
dy = float(dy)
agl_height = float(agl_height)
time_idx = int(time_idx)
mask_buildings = int(mask_buildings)

NODATA = -9999.0


def netcdf_subdataset(path: str, var: str) -> str:
    return f'NETCDF:"{path}":{var}'


def parse_z_levels(mag_ds) -> np.ndarray:
    meta = mag_ds.GetMetadata()
    raw = meta.get("NETCDF_DIM_z_VALUES", "")
    if not raw:
        raise SystemExit("Error: could not read NETCDF_DIM_z_VALUES from mag subdataset metadata")
    values = [float(v) for v in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", raw)]
    if not values:
        raise SystemExit("Error: parsed z levels list is empty")
    return np.asarray(values, dtype=np.float64)


def read_band_stack(ds, time_idx: int) -> np.ndarray:
    """Read mag or icell as (nz, ny, nx), selecting the requested time slice."""
    nz = ds.RasterCount
    if nz == 0:
        raise SystemExit(f"Error: no bands in dataset {ds.GetDescription()}")

    # GDAL exposes mag as nz bands for a single time step when t has one value.
    # When multiple time steps exist, bands are ordered t-major then z.
    meta = ds.GetMetadata()
    t_values_raw = meta.get("NETCDF_DIM_t_VALUES", "0")
    t_values = [float(v) for v in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", t_values_raw)]
    nt = max(1, len(t_values))
    if nz % nt != 0:
        raise SystemExit(f"Error: band count {nz} is not divisible by time count {nt}")
    nz_per_t = nz // nt
    if time_idx < 0 or time_idx >= nt:
        raise SystemExit(f"Error: time index {time_idx} out of range (0..{nt - 1})")

    start_band = time_idx * nz_per_t
    stack = np.empty((nz_per_t, ds.RasterYSize, ds.RasterXSize), dtype=np.float32)
    for k in range(nz_per_t):
        band = ds.GetRasterBand(start_band + k + 1)
        stack[k] = band.ReadAsArray()
    return stack


def read_terrain(path: str) -> np.ndarray:
    ds = gdal.Open(netcdf_subdataset(path, "terrain"), gdal.GA_ReadOnly)
    if ds is None:
        raise SystemExit("Error: could not open terrain subdataset")
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float64)
    ds = None
    return arr


def read_dem_wkt(dem_path: str) -> str:
    info = json.loads(subprocess.check_output(["gdalinfo", "-json", dem_path], text=True))
    wkt = info.get("coordinateSystem", {}).get("wkt")
    if not wkt:
        raise SystemExit(f"Error: could not read CRS WKT from {dem_path}")
    return wkt


mag_ds = gdal.Open(netcdf_subdataset(nc_path, "mag"), gdal.GA_ReadOnly)
if mag_ds is None:
    raise SystemExit("Error: could not open mag subdataset")

z_levels = parse_z_levels(mag_ds)
mag_stack = read_band_stack(mag_ds, time_idx)
mag_ds = None

ny, nx = mag_stack.shape[1], mag_stack.shape[2]
if len(z_levels) != mag_stack.shape[0]:
    raise SystemExit(
        f"Error: z level count ({len(z_levels)}) != mag band count ({mag_stack.shape[0]})"
    )

terrain = read_terrain(nc_path)
if terrain.shape != (ny, nx):
    raise SystemExit(f"Error: terrain shape {terrain.shape} != mag shape ({ny}, {nx})")

target_z = terrain + agl_height
k_idx = np.abs(z_levels[:, None, None] - target_z[None, :, :]).argmin(axis=0).astype(np.int32)
out = np.take_along_axis(mag_stack, k_idx[None, :, :], axis=0)[0].astype(np.float32)

selected_z = z_levels[k_idx]
agl_actual = selected_z - terrain

if mask_buildings:
    icell_ds = gdal.Open(netcdf_subdataset(nc_path, "icell"), gdal.GA_ReadOnly)
    if icell_ds is None:
        raise SystemExit("Error: could not open icell subdataset (required for --mask-buildings)")
    icell_stack = read_band_stack(icell_ds, time_idx)
    icell_ds = None
    if icell_stack.shape[0] != mag_stack.shape[0]:
        raise SystemExit("Error: icell vertical level count does not match mag")
    icell_sel = np.take_along_axis(icell_stack, k_idx[None, :, :], axis=0)[0]
    out = np.where(icell_sel == 1, out, NODATA)

out[np.isnan(out)] = NODATA

driver = gdal.GetDriverByName("GTiff")
if driver is None:
    raise SystemExit("Error: GTiff driver not available")

if gdal.VSIStatL(out_tif) is not None:
    driver.Delete(out_tif)

# QES j=0 at south; GeoTIFF line 0 must be north.
out_north = np.flipud(out)
gt = (x0, dx, 0.0, y0 + ny * dy, 0.0, -dy)
dst = driver.Create(out_tif, nx, ny, 1, gdal.GDT_Float32, options=["COMPRESS=DEFLATE", "TILED=YES"])
if dst is None:
    raise SystemExit(f"Error: could not create {out_tif}")

dst.SetGeoTransform(gt)
dst.SetProjection(read_dem_wkt(dem_path))
band = dst.GetRasterBand(1)
band.SetNoDataValue(NODATA)
band.SetDescription(f"velocity magnitude at {agl_height} m AGL (m/s)")
band.SetUnitType("m/s")
band.WriteArray(out_north)
band.ComputeStatistics(False)
band.FlushCache()
dst = None

valid = out[out != NODATA]
print(f"  Grid   : {nx} x {ny} pixels, {len(z_levels)} z levels")
print(f"  z target (AMSL): min={target_z.min():.2f} max={target_z.max():.2f} mean={target_z.mean():.2f} m")
print(f"  z selected (AMSL): min={selected_z.min():.2f} max={selected_z.max():.2f} mean={selected_z.mean():.2f} m")
print(f"  AGL at selected z: min={agl_actual.min():.2f} max={agl_actual.max():.2f} mean={agl_actual.mean():.2f} m")
unique_k, counts = np.unique(k_idx, return_counts=True)
top = sorted(zip(counts, unique_k), reverse=True)[:5]
print("  Top z-band usage (count, z_m):", ", ".join(f"{c}@{z_levels[k]:.2f}" for c, k in top))
if valid.size:
    print(f"  mag    : min={valid.min():.3f} max={valid.max():.3f} mean={valid.mean():.3f} m/s ({valid.size} valid pixels)")
else:
    print("  mag    : no valid pixels after masking")
print(f"Success: {out_tif}")
PY

echo "Done."
