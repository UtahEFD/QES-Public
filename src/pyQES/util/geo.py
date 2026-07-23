"""Geospatial preprocessing for QES domains.

Python re-implementation of the DEM/shapefile preprocessing done by the
``run_qeswinds.sh`` launcher, using rasterio / pyproj / geopandas instead of the
GDAL command-line tools. These helpers compute the domain origin (UTM), the
domain cell counts, the north-centered sensor position and the clipped buildings
shapefile aligned to the DEM CRS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .config import WindsParameters

__all__ = [
    "Origin",
    "DEMExtent",
    "get_dem_srs",
    "dem_extent",
    "compute_domain_origin_from_dem",
    "compute_domain_cells",
    "compute_sensor_north_qes_coords",
    "prepare_buildings_clipped",
    "check_domain_rotation",
]

_UTM_LAT_BANDS = "CDEFGHJKLMNPQRSTUVWX"


@dataclass(frozen=True)
class Origin:
    """SW corner of the DEM expressed as UTM origin metadata."""

    utm_x: float
    utm_y: float
    utm_zone: int
    utm_zone_letter: str


@dataclass(frozen=True)
class DEMExtent:
    """Raster extent expressed in pixels and metres."""

    width_px: int
    height_px: int
    res_x: float
    res_y: float
    sw_x: float
    sw_y: float

    @property
    def width_m(self) -> float:
        return self.width_px * abs(self.res_x)

    @property
    def height_m(self) -> float:
        return self.height_px * abs(self.res_y)


def _require(module: str):
    """Import an optional geospatial dependency with an actionable error."""
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            f"'{module}' is required for pyQES geospatial preprocessing. "
            f"Install the geo extras, e.g. `pip install pyqes[geo]`."
        ) from exc


def get_dem_srs(dem: str | Path):
    """Return the coordinate reference system of a DEM raster."""
    rasterio = _require("rasterio")
    with rasterio.open(str(dem)) as ds:
        if ds.crs is None:
            raise ValueError(f"DEM has no CRS: {dem}")
        return ds.crs


def dem_extent(dem: str | Path) -> DEMExtent:
    """Read pixel size and SW corner of a DEM raster."""
    rasterio = _require("rasterio")
    dem = Path(dem)
    if not dem.is_file():
        raise FileNotFoundError(f"DEM not found: {dem}")
    with rasterio.open(str(dem)) as ds:
        gt = ds.transform
        return DEMExtent(
            width_px=ds.width,
            height_px=ds.height,
            res_x=gt.a,
            res_y=gt.e,
            sw_x=ds.bounds.left,
            sw_y=ds.bounds.bottom,
        )


def compute_domain_origin_from_dem(dem: str | Path) -> Origin:
    """Compute the UTM origin (SW corner) and UTM zone/band of a DEM.

    Mirrors ``compute_domain_origin_from_dem`` in ``run_qeswinds.sh``: the SW
    corner is reprojected to WGS84 to derive the UTM zone number and latitude
    band letter, while UTMx/UTMy keep the DEM's native projected coordinates.
    """
    pyproj = _require("pyproj")
    ext = dem_extent(dem)
    crs = get_dem_srs(dem)

    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(ext.sw_x, ext.sw_y)

    utm_zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    band_idx = int(math.floor((lat + 80.0) / 8.0))
    band_idx = max(0, min(band_idx, len(_UTM_LAT_BANDS) - 1))

    return Origin(
        utm_x=round(ext.sw_x, 2),
        utm_y=round(ext.sw_y, 2),
        utm_zone=utm_zone,
        utm_zone_letter=_UTM_LAT_BANDS[band_idx],
    )


def _max_building_height(shp: str | Path, height_field: str) -> float:
    """Return the maximum value of ``height_field`` in a buildings shapefile."""
    shp = Path(shp)
    if not shp.is_file():
        return 0.0
    gpd = _require("geopandas")
    gdf = gpd.read_file(str(shp))
    if height_field not in gdf.columns or gdf.empty:
        return 0.0
    return float(gdf[height_field].max())


def compute_domain_cells(
    params: WindsParameters,
    dem: str | Path,
    shp: str | Path | None = None,
    z_margin: float = 20.0,
) -> tuple[int, int, int]:
    """Compute the (nx, ny, nz) cell counts for a QES domain.

    Mirrors ``compute_domain_cells`` in ``run_qeswinds.sh``:
    ``nx = ceil((width_m + 2*halo_x) / dx)`` (idem ny), and nz derives from the
    max terrain elevation plus the max building height plus a vertical margin.
    """
    sim = params.simulation_parameters
    halo_x, halo_y = sim.halo_x, sim.halo_y
    dx, dy, dz = sim.cell_size

    ext = dem_extent(dem)
    nx = math.ceil((ext.width_m + 2.0 * halo_x) / dx)
    ny = math.ceil((ext.height_m + 2.0 * halo_y) / dy)

    rasterio = _require("rasterio")
    with rasterio.open(str(dem)) as ds:
        band = ds.read(1, masked=True)
        dem_max = float(band.max())

    max_h = 0.0
    if shp is not None:
        field = params.buildings_params.shp_height_field or "hauteur"
        max_h = _max_building_height(shp, field)

    top_z = dem_max + max_h + z_margin
    nz = max(1, math.ceil(top_z / dz))
    return nx, ny, nz


def compute_sensor_north_qes_coords(
    dem: str | Path,
    dem_distance_x: float = 0.0,
    dem_distance_y: float = 0.0,
) -> tuple[float, float]:
    """Compute the north-center sensor site in QES-local coordinates.

    Mirrors ``compute_sensor_north_qes_coords`` in ``run_qeswinds.sh``:
    ``site_x = width_m/2 - dem_distance_x`` and ``site_y = height_m - dem_distance_y``.
    """
    ext = dem_extent(dem)
    site_x = ext.width_m / 2.0 - dem_distance_x
    site_y = ext.height_m - dem_distance_y
    return round(site_x, 1), round(site_y, 1)


def prepare_buildings_clipped(
    dem: str | Path,
    src: str | Path,
    mask: str | Path,
    out: str | Path,
) -> Path:
    """Reproject buildings to the DEM CRS and clip them to a mask polygon.

    Mirrors ``prepare_buildings_clipped`` in ``run_qeswinds.sh`` (ogr2ogr
    ``-t_srs <dem_crs> -clipsrc mask``) using geopandas. The output layer name is
    the file stem, matching ``<SHPBuildingLayer>`` in the QES XML.
    """
    gpd = _require("geopandas")
    src, mask, out = Path(src), Path(mask), Path(out)
    if not src.is_file():
        raise FileNotFoundError(f"Buildings source not found: {src}")
    if not mask.is_file():
        raise FileNotFoundError(f"Clip mask not found: {mask}")

    dem_crs = get_dem_srs(dem)
    buildings = gpd.read_file(str(src)).to_crs(dem_crs)
    mask_gdf = gpd.read_file(str(mask)).to_crs(dem_crs)

    clipped = gpd.clip(buildings, mask_gdf)
    out.parent.mkdir(parents=True, exist_ok=True)
    clipped.to_file(str(out))
    return out


def check_domain_rotation(params: WindsParameters) -> None:
    """Raise if the domain rotation is non-zero (unsupported by the workflow).

    QES-Winds crashes when ``domainRotation != 0`` (sensor UTM conversion in
    ``WindProfilerSensorType``), so the launcher refuses to run in that case.
    """
    rotation = params.simulation_parameters.domain_rotation
    if rotation != 0:
        raise ValueError(
            f"domainRotation={rotation} is not supported: QES-Winds crashes when "
            "domainRotation != 0. Set <domainRotation>0</domainRotation>."
        )
