#!/usr/bin/env bash
# Shared launcher for QES-Winds umep_workflow (La Rochelle).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${QES_BUILD_DIR:-$REPO_ROOT/build}"
XML="$REPO_ROOT/data/umep_workflow/qes/umep_larochelle.xml"
QES_DIR="$(dirname "$XML")"
OUT_DIR="$REPO_ROOT/data/umep_workflow/output"
QES_BIN="$BUILD_DIR/qesWinds/qesWinds"
OUTPUT_NAME="umep_larochelle"

usage() {
  cat <<EOF
Usage: $(basename "$0") [-s SOLVER]

Run QES-Winds for the umep_workflow La Rochelle case.

Options:
  -s SOLVER   Solver type (default: ${SOLVER_TYPE:-1})
              1 = CPU, 2 = GPU dynamic parallel

Environment:
  QES_BUILD_DIR       Path to CMake build directory (default: \$REPO_ROOT/build)
  QES_SENSOR_AUTO     Auto-place sensor at DEM north center (default: 1, set 0 to skip)
  QES_BUILDINGS_SRC   Source buildings shapefile (default: batiments_urock_0.shp)
  QES_BUILDINGS_MASK  Clip mask shapefile (default: mask.shp)
  QES_BUILDINGS_AUTO  Reproject+clip buildings to DEM CRS (default: 1, set 0 to skip)

Build QES first if needed:
  mkdir -p build && cd build && cmake .. && make
EOF
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

get_dem_srs() {
  local dem="$1"
  if ! command -v gdalinfo >/dev/null 2>&1; then
    echo "EPSG:2154"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$dem" <<'PY'
import json
import subprocess
import sys

info = json.loads(subprocess.check_output(["gdalinfo", "-json", sys.argv[1]], text=True))
wkt = info.get("coordinateSystem", {}).get("wkt")
if wkt:
    print(wkt)
else:
    print("EPSG:2154")
PY
    return 0
  fi
  echo "EPSG:2154"
}

prepare_buildings_clipped() {
  local dem="$1"
  local src="${QES_BUILDINGS_SRC:-$REPO_ROOT/data/umep_workflow/batiments_urock_0.shp}"
  local mask="${QES_BUILDINGS_MASK:-$REPO_ROOT/data/umep_workflow/mask.shp}"
  local out="$REPO_ROOT/data/umep_workflow/qes/buildings_clipped.shp"

  if [[ "${QES_BUILDINGS_AUTO:-1}" == "0" ]]; then
    echo "  Buildings: auto clip/reproject skipped (QES_BUILDINGS_AUTO=0)"
    return 0
  fi

  if ! command -v ogr2ogr >/dev/null 2>&1; then
    echo "Error: ogr2ogr not found (install GDAL)" >&2
    return 1
  fi

  if [[ ! -f "$src" ]]; then
    if [[ -f "$out" ]]; then
      echo "  Buildings: source missing ($src), using existing $out" >&2
      return 0
    fi
    echo "Error: buildings source not found at: $src" >&2
    return 1
  fi

  if [[ ! -f "$mask" ]]; then
    echo "Error: clip mask not found at: $mask" >&2
    return 1
  fi

  local dem_srs
  dem_srs="$(get_dem_srs "$dem")"
  ogr2ogr -overwrite -t_srs "$dem_srs" -clipsrc "$mask" "$out" "$src" -nln buildings_clipped
  echo "  Buildings: clipped+reprojected to DEM CRS -> $out"
}

compute_domain_origin_from_dem() {
  local dem="$1"
  if [[ ! -f "$dem" ]]; then
    echo "Error: DEM not found at: $dem" >&2
    return 1
  fi
  if ! command -v gdalinfo >/dev/null 2>&1; then
    echo "Error: gdalinfo not found (install GDAL)" >&2
    return 1
  fi

  if command -v python3 >/dev/null 2>&1; then
    python3 - "$dem" <<'PY'
import json
import math
import subprocess
import sys

dem = sys.argv[1]
info = json.loads(subprocess.check_output(["gdalinfo", "-json", dem], text=True))
gt = info["geoTransform"]
w, h = info["size"]
sw_x = gt[0]
sw_y = gt[3] + h * gt[5]

srs = "EPSG:2154"
wkt = info.get("coordinateSystem", {}).get("wkt")
if wkt:
    srs = wkt

out = subprocess.check_output(
    ["gdaltransform", "-s_srs", srs, "-t_srs", "EPSG:4326"],
    input=f"{sw_x} {sw_y}\n",
    text=True,
)
lon, lat = map(float, out.split()[:2])
utm_zone = int(math.floor((lon + 180.0) / 6.0) + 1)
lat_bands = "CDEFGHJKLMNPQRSTUVWX"
band_idx = int(math.floor((lat + 80.0) / 8.0))
band_idx = max(0, min(band_idx, len(lat_bands) - 1))
utm_letter = lat_bands[band_idx]

print(f"{sw_x:.2f}")
print(f"{sw_y:.2f}")
print(utm_zone)
print(utm_letter)
PY
    return 0
  fi

  # Fallback: parse gdalinfo Lower Left + gdaltransform for zone
  local sw_x sw_y lon lat
  read -r sw_x sw_y < <(
    gdalinfo "$dem" | awk '/Lower Left/ {
      gsub(/[(),]/, "", $5)
      gsub(/[(),]/, "", $6)
      print $5, $6
    }'
  )
  if [[ -z "${sw_x:-}" ]]; then
    echo "Error: could not parse DEM corner from gdalinfo" >&2
    return 1
  fi
  read -r lon lat < <(echo "$sw_x $sw_y" | gdaltransform -s_srs EPSG:2154 -t_srs EPSG:4326 | awk '{print $1, $2}')
  local utm_zone utm_letter
  utm_zone="$(awk -v lon="$lon" 'BEGIN { printf "%d", int((lon + 180) / 6) + 1 }')"
  utm_letter="$(awk -v lat="$lat" 'BEGIN {
    bands = "CDEFGHJKLMNPQRSTUVWX"
    idx = int((lat + 80) / 8)
    if (idx < 0) idx = 0
    if (idx > 19) idx = 19
    printf "%s", substr(bands, idx + 1, 1)
  }')"
  printf '%s\n%s\n%s\n%s\n' "$sw_x" "$sw_y" "$utm_zone" "$utm_letter"
}

patch_xml_tag() {
  local file="$1" tag="$2" value="$3"
  if [[ "$(uname)" == Darwin ]]; then
    sed -i '' -E "s|(<${tag}>)[^<]*(</${tag}>)|\\1 ${value} \\2|" "$file"
  else
    sed -i -E "s|(<${tag}>)[^<]*(</${tag}>)|\\1 ${value} \\2|" "$file"
  fi
}

read_xml_tag() {
  local file="$1" tag="$2"
  sed -n "s/.*<${tag}>[[:space:]]*\\([^<]*\\)[[:space:]]*<\\/${tag}>.*/\\1/p" "$file" | head -1
}

patch_xml_domain() {
  local file="$1" nx="$2" ny="$3" nz="$4"
  if [[ "$(uname)" == Darwin ]]; then
    sed -i '' -E "s|(<domain>)[^<]*(</domain>)|\\1 ${nx} ${ny} ${nz} \\2|" "$file"
  else
    sed -i -E "s|(<domain>)[^<]*(</domain>)|\\1 ${nx} ${ny} ${nz} \\2|" "$file"
  fi
}

compute_domain_cells() {
  local xml="$1" dem="$2" shp="$3"
  local z_margin="${QES_DOMAIN_Z_MARGIN:-20}"

  if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 required to compute domain cells" >&2
    return 1
  fi

  python3 - "$xml" "$dem" "$shp" "$z_margin" <<'PY'
import json
import math
import re
import subprocess
import sys

xml_path, dem_path, shp_path, z_margin = sys.argv[1:5]
z_margin = float(z_margin)

def read_tag(path, tag):
    text = open(path, encoding="utf-8").read()
    m = re.search(rf"<{tag}>\s*([^<]+?)\s*</{tag}>", text)
    if not m:
        raise SystemExit(f"Missing <{tag}> in {path}")
    return m.group(1).strip()

halo_x = float(read_tag(xml_path, "halo_x"))
halo_y = float(read_tag(xml_path, "halo_y"))
cell_parts = read_tag(xml_path, "cellSize").split()
dx, dy, dz = map(float, cell_parts[:3])

info = json.loads(subprocess.check_output(["gdalinfo", "-json", dem_path], text=True))
gt = info["geoTransform"]
w, h = info["size"]
width_m = w * abs(gt[1])
height_m = h * abs(gt[5])

nx = int(math.ceil((width_m + 2.0 * halo_x) / dx))
ny = int(math.ceil((height_m + 2.0 * halo_y) / dy))

stats = subprocess.check_output(
    ["gdalinfo", "-stats", dem_path], text=True, stderr=subprocess.STDOUT
)
dem_max = float(re.search(r"STATISTICS_MAXIMUM=([0-9.+-eE]+)", stats).group(1))

max_h = 0.0
try:
    out = subprocess.check_output(
        [
            "ogrinfo",
            "-sql",
            "SELECT MAX(hauteur) AS max_h FROM buildings_clipped",
            shp_path,
        ],
        text=True,
        stderr=subprocess.STDOUT,
    )
    m = re.search(r"max_h \(Real\) = ([0-9.]+)", out)
    if m:
        max_h = float(m.group(1))
except subprocess.CalledProcessError:
    pass

top_z = dem_max + max_h + z_margin
nz = max(1, int(math.ceil(top_z / dz)))

print(nx)
print(ny)
print(nz)
print(f"width_m={width_m:.2f} height_m={height_m:.2f} dem_max={dem_max:.2f} max_h={max_h:.2f}")
PY
}

update_domain_origin_in_xml() {
  local xml="$1" dem="$2"
  local origin_lines utmx utmy utm_zone utm_letter
  origin_lines="$(compute_domain_origin_from_dem "$dem")"
  utmx="$(echo "$origin_lines" | sed -n '1p')"
  utmy="$(echo "$origin_lines" | sed -n '2p')"
  utm_zone="$(echo "$origin_lines" | sed -n '3p')"
  utm_letter="$(echo "$origin_lines" | sed -n '4p')"

  patch_xml_tag "$xml" UTMx "$utmx"
  patch_xml_tag "$xml" UTMy "$utmy"
  patch_xml_tag "$xml" UTMZone "$utm_zone"
  patch_xml_tag "$xml" UTMZoneLetter "$utm_letter"

  echo "  DEM    : $dem"
  echo "  Origin : UTMx=$utmx UTMy=$utmy UTMZone=$utm_zone UTMZoneLetter=$utm_letter"
}

update_domain_cells_in_xml() {
  local xml="$1" dem="$2" shp="$3"
  local domain_lines nx ny nz domain_info
  domain_lines="$(compute_domain_cells "$xml" "$dem" "$shp")"
  nx="$(echo "$domain_lines" | sed -n '1p')"
  ny="$(echo "$domain_lines" | sed -n '2p')"
  nz="$(echo "$domain_lines" | sed -n '3p')"
  domain_info="$(echo "$domain_lines" | sed -n '4p')"

  patch_xml_domain "$xml" "$nx" "$ny" "$nz"

  echo "  Domain : nx=$nx ny=$ny nz=$nz ($domain_info)"
}

resolve_sensor_paths() {
  local xml="$1"
  local sensor_rel sensor_abs
  while IFS= read -r sensor_rel; do
    [[ -z "$sensor_rel" ]] && continue
    sensor_abs="$(cd "$QES_DIR" && cd "$(dirname "$sensor_rel")" && pwd)/$(basename "$sensor_rel")"
    echo "$sensor_abs"
  done < <(sed -n 's/.*<sensorName>[[:space:]]*\([^<]*\)[[:space:]]*<\/sensorName>.*/\1/p' "$xml")
}

compute_sensor_north_qes_coords() {
  local dem="$1" dem_distance_x="$2" dem_distance_y="$3"

  if [[ ! -f "$dem" ]]; then
    echo "Error: DEM not found at: $dem" >&2
    return 1
  fi
  if ! command -v gdalinfo >/dev/null 2>&1; then
    echo "Error: gdalinfo not found (install GDAL)" >&2
    return 1
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 required to compute sensor coordinates" >&2
    return 1
  fi

  python3 - "$dem" "$dem_distance_x" "$dem_distance_y" <<'PY'
import json
import subprocess
import sys

dem_path, dem_distance_x, dem_distance_y = sys.argv[1:4]
dem_distance_x = float(dem_distance_x)
dem_distance_y = float(dem_distance_y)

info = json.loads(subprocess.check_output(["gdalinfo", "-json", dem_path], text=True))
gt = info["geoTransform"]
w, h = info["size"]
width_m = w * abs(gt[1])
height_m = h * abs(gt[5])

site_x = width_m / 2.0 - dem_distance_x
site_y = height_m - dem_distance_y

print(f"{site_x:.1f}")
print(f"{site_y:.1f}")
PY
}

update_sensor_site_in_xml() {
  local xml="$1" dem="$2"
  local dem_dist_x dem_dist_y coord_lines site_x site_y sensor_path found=0

  if [[ "${QES_SENSOR_AUTO:-1}" == "0" ]]; then
    echo "  Sensor : auto-placement skipped (QES_SENSOR_AUTO=0)"
    return 0
  fi

  dem_dist_x="$(read_xml_tag "$xml" DEMDistancex)"
  dem_dist_y="$(read_xml_tag "$xml" DEMDistancey)"
  dem_dist_x="${dem_dist_x:-0.0}"
  dem_dist_y="${dem_dist_y:-0.0}"

  coord_lines="$(compute_sensor_north_qes_coords "$dem" "$dem_dist_x" "$dem_dist_y")"
  site_x="$(echo "$coord_lines" | sed -n '1p')"
  site_y="$(echo "$coord_lines" | sed -n '2p')"

  while IFS= read -r sensor_path; do
    [[ -z "$sensor_path" ]] && continue
    if [[ ! -f "$sensor_path" ]]; then
      echo "Error: sensor file not found at: $sensor_path" >&2
      return 1
    fi
    patch_xml_tag "$sensor_path" site_xcoord "$site_x"
    patch_xml_tag "$sensor_path" site_ycoord "$site_y"
    echo "  Sensor : $sensor_path site_xcoord=$site_x site_ycoord=$site_y (DEM north center, QES local, halo excluded)"
    found=1
  done < <(resolve_sensor_paths "$xml")

  if [[ "$found" -eq 0 ]]; then
    echo "Error: no <sensorName> found in $xml" >&2
    return 1
  fi
}

check_domain_rotation() {
  local xml="$1"
  local rotation
  rotation="$(read_xml_tag "$xml" domainRotation)"
  rotation="${rotation:-0}"
  if awk -v r="$rotation" 'BEGIN { exit (r == 0 || r == 0.0) ? 0 : 1 }'; then
    return 0
  fi
  echo "Error: domainRotation=$rotation is not supported by this workflow." >&2
  echo "QES-Winds crashes when domainRotation != 0 (sensor UTM conversion in WindProfilerSensorType)." >&2
  echo "Set <domainRotation>0</domainRotation> in $xml" >&2
  return 1
}

SOLVER_TYPE="${QES_SOLVER:-1}"
while getopts ":s:h" opt; do
  case "$opt" in
    s) SOLVER_TYPE="$OPTARG" ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
done

if [[ ! -x "$QES_BIN" ]]; then
  echo "Error: qesWinds binary not found at: $QES_BIN" >&2
  echo "Build QES with: mkdir -p build && cd build && cmake .. && make" >&2
  exit 1
fi

if [[ ! -f "$XML" ]]; then
  echo "Error: XML input not found at: $XML" >&2
  exit 1
fi

DEM_ABS="$(resolve_dem_path "$XML")"
check_domain_rotation "$XML"
prepare_buildings_clipped "$DEM_ABS"

SHP="$REPO_ROOT/data/umep_workflow/qes/buildings_clipped.shp"
if [[ ! -f "$SHP" ]]; then
  echo "Error: clipped buildings shapefile not found at: $SHP" >&2
  echo "Set QES_BUILDINGS_SRC or run ogr2ogr -t_srs EPSG:2154 -clipsrc mask.shp qes/buildings_clipped.shp batiments_urock_0.shp" >&2
  exit 1
fi

update_domain_origin_in_xml "$XML" "$DEM_ABS"
update_domain_cells_in_xml "$XML" "$DEM_ABS" "$SHP"
update_sensor_site_in_xml "$XML" "$DEM_ABS"

mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/run_${SOLVER_TYPE}_$(date +%Y%m%d_%H%M%S).log"

echo "QES-Winds umep_workflow run"
echo "  Binary : $QES_BIN"
echo "  XML    : $XML"
echo "  Solver : $SOLVER_TYPE"
echo "  Output : $OUT_DIR/${OUTPUT_NAME}_windsOut.nc"
echo "  Log    : $LOG_FILE"

cd "$OUT_DIR"
"$QES_BIN" -q "$XML" -s "$SOLVER_TYPE" -w -o "$OUTPUT_NAME" 2>&1 | tee "$LOG_FILE"

if [[ -f "${OUTPUT_NAME}_windsOut.nc" ]]; then
  echo "Success: ${OUT_DIR}/${OUTPUT_NAME}_windsOut.nc"
  echo "Workspace: ${OUT_DIR}/${OUTPUT_NAME}_windsWk.nc"
else
  echo "Error: expected output file not created: ${OUT_DIR}/${OUTPUT_NAME}_windsOut.nc" >&2
  exit 1
fi
