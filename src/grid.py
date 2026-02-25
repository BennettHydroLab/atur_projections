"""
WRF d01 grid coordinate loading and subbasin spatial mask construction.

The WRF d01 grid uses a Lambert Conformal projection; lat/lon are stored as
2D arrays (XLAT, XLONG) in the coordinate NetCDF file. Subbasin masks are
built by spatially joining grid-cell centroids to subbasin polygons.
"""

from __future__ import annotations

import netCDF4
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import s3fs
from shapely.geometry import Point

from .catalog import get_wrf_coord_path


def load_wrf_coords(fs: s3fs.S3FileSystem) -> tuple[np.ndarray, np.ndarray]:
    """Load 2D latitude and longitude arrays from the WRF d01 coordinate file.

    Args:
        fs: Anonymous s3fs filesystem.

    Returns:
        Tuple of (xlat, xlon) as 2D float64 numpy arrays with shape (ny, nx).
    """
    coord_path = get_wrf_coord_path()
    print(f"Loading WRF coordinates from {coord_path}...")
    # The netcdf4 engine doesn't support storage_options for S3 URIs.
    # Instead, read the file bytes via s3fs and open as an in-memory NetCDF4 dataset.
    # The coord file is only ~0.7 MB so this is fast.
    data = fs.cat(coord_path)
    nc = netCDF4.Dataset("in_memory.nc", memory=data)
    ds = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))
    print(ds)
    with ds:
        # WRF coordinate files store lat/lon as XLAT and XLONG
        # Try common variable name variants
        lat_name = 'lat2d'
        lon_name = 'lon2d'

        if lat_name in ds:
            xlat = ds[lat_name].values
        else:
            raise KeyError(f"Could not find latitude variable in {list(ds.data_vars)}")

        if lon_name in ds:
            xlon = ds[lon_name].values
        else:
            raise KeyError(f"Could not find longitude variable in {list(ds.data_vars)}")

    # Drop any leading time/ensemble dimensions — keep only (ny, nx)
    while xlat.ndim > 2:
        xlat = xlat[0]
    while xlon.ndim > 2:
        xlon = xlon[0]

    print('Min max and delta of lat:', xlat.min(), xlat.max(), np.mean(np.diff(xlat)))
    print('Min max and delta of lon:', xlon.min(), xlon.max(), np.mean(np.diff(xlon)))
    return xlat.astype(np.float64), xlon.astype(np.float64)


def build_subbasin_masks(
    shapefile_path: str,
    xlat: np.ndarray,
    xlon: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Create boolean grid masks for each subbasin and a cosine-latitude weight array.

    Strategy:
        1. Reproject subbasin polygons from EPSG:26912 to EPSG:4326 (WGS84).
        2. Create a flat GeoDataFrame of WRF grid-cell centroid Points.
        3. Spatial join: assign each grid point to its containing subbasin.
        4. Reconstruct 2D boolean masks (ny × nx) per subbasin.

    Args:
        shapefile_path: Path to the GW_SubBasins shapefile directory or .shp file.
        xlat: 2D latitude array from WRF grid, shape (ny, nx).
        xlon: 2D longitude array from WRF grid, shape (ny, nx).

    Returns:
        masks: Dict mapping subbasin name → 2D boolean numpy array (ny, nx).
               True where the grid cell centroid falls within the subbasin.
        lat_weights: 2D float array (ny, nx) of cosine-latitude weights,
                     normalised to sum to 1 within each subbasin mask.
    """
    ny, nx = xlat.shape

    # Load and reproject subbasin polygons to WGS84
    basins = gpd.read_file(shapefile_path).to_crs("EPSG:4326")

    # Use SUBBASIN_N as the primary identifier; fall back to index if blank
    if "SUBBASIN_N" in basins.columns:
        basins["_label"] = basins["SUBBASIN_N"].fillna(
            pd.Series(basins.index.astype(str), index=basins.index)
        )
    else:
        basins["_label"] = basins.index.astype(str)

    # Build flat GeoDataFrame of all WRF grid-cell centroids
    flat_lats = xlat.ravel()
    flat_lons = xlon.ravel()
    points = gpd.GeoDataFrame(
        {"row": np.repeat(np.arange(ny), nx), "col": np.tile(np.arange(nx), ny)},
        geometry=[Point(lon, lat) for lat, lon in zip(flat_lats, flat_lons)],
        crs="EPSG:4326",
    )

    # Spatial join: each point gets the label of the polygon it falls within
    joined = gpd.sjoin(points, basins[["_label", "geometry"]], how="left", predicate="within")

    # Build cosine-latitude weights (unnormalised; normalised per subbasin below)
    cos_weights = np.cos(np.deg2rad(xlat))

    masks: dict[str, np.ndarray] = {}
    for label in basins["_label"].unique():
        subset = joined[joined["_label"] == label]
        mask = np.zeros((ny, nx), dtype=bool)
        mask[subset["row"].values, subset["col"].values] = True
        if mask.any():
            masks[str(label)] = mask

    return masks, cos_weights


def weighted_basin_mean(
    data: np.ndarray,
    mask: np.ndarray,
    cos_weights: np.ndarray,
) -> float:
    """Compute the cosine-latitude-weighted spatial mean over a subbasin mask.

    Args:
        data: 2D array (ny, nx) of values to average.
        mask: 2D boolean array (ny, nx) for the subbasin.
        cos_weights: 2D cosine-latitude weight array (ny, nx).

    Returns:
        Scalar weighted mean, or NaN if mask is empty.
    """
    w = cos_weights[mask]
    v = data[mask]
    if w.sum() == 0:
        return float("nan")
    return float(np.average(v, weights=w))
