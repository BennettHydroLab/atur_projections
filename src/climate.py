"""
Climate data loading, monthly climatology computation, and delta calculation.

Tier 3 time coordinate is an integer array encoded as yyyymmdd (e.g. 19800115).
Files are opened lazily via xarray + dask; the full 30-year stack is processed
one variable at a time to keep memory manageable.
"""

from __future__ import annotations

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
import s3fs

from .catalog import (
    build_s3_path,
    GCM_CATALOG,
    EXP_HIST,
    EXP_SSP370,
    HIST_YEARS,
    FUTURE_YEARS,
    ALL_HIST_YEARS,
    ALL_SSP370_YEARS,
)
from .grid import weighted_basin_mean

# Variables of interest
VARIABLES = ["t2", "prec"]


# ---------------------------------------------------------------------------
# Time decoding
# ---------------------------------------------------------------------------

def decode_yyyymmdd(ds: xr.Dataset) -> xr.Dataset:
    """Normalise the time coordinate to datetime64 and rename it to 'time'.

    Handles two encodings found across WUS-D3 GCMs:

    1. yyyymmdd integer (most models) — the time/day variable contains plain
       integers like 19800115 or 19800115.0.  Cast to int, then parse with
       strptime format "%Y%m%d".

    2. CF convention (e.g. noresm2-mm) — xarray's default decode_times=True
       already converted the variable to datetime64[ns] when the file has a
       standard "days since …" units attribute.  In this case no further
       parsing is needed; just rename if necessary.

    The time coordinate may be named 'time' or 'day' depending on the file.

    Args:
        ds: Dataset as returned by xr.open_dataset.

    Returns:
        Dataset with a 'time' coord of np.datetime64 values.
    """
    time_var = "time" if "time" in ds.coords else "day"
    raw = ds[time_var].values

    # Case 1: already datetime64 — xarray decoded the CF time units for us
    if np.issubdtype(raw.dtype, np.datetime64):
        if time_var != "time":
            ds = ds.rename({time_var: "time"})
        return ds

    # Case 2: yyyymmdd integer encoding — cast to int, strip decimal part
    dates = pd.to_datetime(raw.astype(int).astype(str), format="%Y%m%d")
    if time_var != "time":
        ds = ds.rename({time_var: "time"})
    return ds.assign_coords(time=dates.values)


# ---------------------------------------------------------------------------
# Single-file loading
# ---------------------------------------------------------------------------

def open_annual_file(
    fs: s3fs.S3FileSystem,
    gcm_info: dict,
    scenario: str,
    variable: str,
    year: int,
) -> xr.DataArray | None:
    """Open a single Tier 3 NetCDF file and return the variable DataArray.

    Returns None if the file does not exist on S3.

    Args:
        fs: Anonymous s3fs filesystem.
        gcm_info: Entry from GCM_CATALOG.
        scenario: EXP_HIST or EXP_SSP370.
        variable: 't2' or 'prec'.
        year: 4-digit year.

    Returns:
        DataArray with decoded datetime64 time coordinate, or None.
    """
    path = build_s3_path(gcm_info, scenario, variable, year)
    if not fs.exists(path):
        return None
    try:
        # Read bytes via s3fs then open as in-memory NetCDF4 dataset.
        # This avoids the storage_options limitation of the xarray netcdf4 backend.
        data = fs.cat(path)
        nc = netCDF4.Dataset("in_memory.nc", memory=data)
        ds = xr.open_dataset(xr.backends.NetCDF4DataStore(nc))
        ds = decode_yyyymmdd(ds)
        # Force load into numpy so the netCDF4 object can be closed safely.
        return ds[variable].load()
    except Exception as exc:
        print(f"  [warn] Could not open {path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# 30-year climatology
# ---------------------------------------------------------------------------

def build_monthly_climatology(
    fs: s3fs.S3FileSystem,
    gcm_info: dict,
    scenario: str,
    variable: str,
    years: list[int],
) -> xr.DataArray | None:
    """Compute a 30-year monthly climatology for a single GCM/scenario/variable.

    Opens each annual file, concatenates along time, then groups by calendar
    month to produce a (12, ny, nx) mean array.

    Args:
        fs: Anonymous s3fs filesystem.
        gcm_info: Entry from GCM_CATALOG.
        scenario: EXP_HIST or EXP_SSP370.
        variable: 't2' or 'prec'.
        years: List of years to include (e.g. HIST_YEARS or FUTURE_YEARS).

    Returns:
        DataArray with shape (12, ny, nx) and 'month' coordinate (1–12),
        or None if no files could be opened.
    """
    arrays = []
    for year in years:
        da = open_annual_file(fs, gcm_info, scenario, variable, year)
        if da is not None:
            arrays.append(da)

    if not arrays:
        return None

    combined = xr.concat(arrays, dim="time")
    climatology = combined.groupby("time.month").mean("time")
    return climatology


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------

def compute_delta(
    hist_clim: xr.DataArray,
    fut_clim: xr.DataArray,
    variable: str,
) -> xr.DataArray:
    """Compute end-of-century minus historical delta on the full 2D grid.

    Args:
        hist_clim: Historical climatology, shape (12, ny, nx).
        fut_clim: Future climatology, shape (12, ny, nx).
        variable: 't2' or 'prec'. Controls delta type.

    Returns:
        Delta DataArray shape (12, ny, nx).
        - t2:   absolute difference in K (equivalent to °C change)
        - prec: absolute difference in mm day⁻¹ (fut - hist)
    """
    if variable == "t2":
        delta = fut_clim - hist_clim
        delta.attrs["units"] = "K"
        delta.attrs["long_name"] = "2-m temperature change (future - historical)"
    elif variable == "prec":
        delta = fut_clim - hist_clim
        delta.attrs["units"] = "mm day-1"
        delta.attrs["long_name"] = "Precipitation change (future - historical)"
    else:
        delta = fut_clim - hist_clim
        delta.attrs["long_name"] = f"{variable} change (future - historical)"
    return delta


# ---------------------------------------------------------------------------
# Subbasin aggregation
# ---------------------------------------------------------------------------

def aggregate_to_subbasins(
    delta: xr.DataArray,
    masks: dict[str, np.ndarray],
    cos_weights: np.ndarray,
) -> pd.DataFrame:
    """Aggregate a (12, ny, nx) delta array to per-subbasin monthly values.

    Args:
        delta: DataArray with 'month' as first dimension (1–12).
        masks: Dict of subbasin_name → 2D boolean mask.
        cos_weights: 2D cosine-latitude weight array (ny, nx).

    Returns:
        DataFrame with shape (12, n_subbasins).
        Index: month integers 1–12.
        Columns: subbasin names.
    """
    delta_np = delta.values  # shape (12, ny, nx)
    months = delta["month"].values  # should be 1..12

    records = {}
    for name, mask in masks.items():
        monthly_means = []
        for m_idx in range(len(months)):
            val = weighted_basin_mean(delta_np[m_idx], mask, cos_weights)
            monthly_means.append(val)
        records[name] = monthly_means

    df = pd.DataFrame(records, index=months)
    df.index.name = "month"
    return df


# ---------------------------------------------------------------------------
# Per-GCM processing
# ---------------------------------------------------------------------------

def process_gcm(
    fs: s3fs.S3FileSystem,
    gcm_info: dict,
    masks: dict[str, np.ndarray],
    cos_weights: np.ndarray,
) -> dict[str, pd.DataFrame] | None:
    """Run the full historical→future climatology→delta→subbasin pipeline for one GCM.

    Args:
        fs: Anonymous s3fs filesystem.
        gcm_info: Entry from GCM_CATALOG.
        masks: Subbasin boolean masks from grid.build_subbasin_masks.
        cos_weights: Cosine-latitude weights from grid.build_subbasin_masks.

    Returns:
        Dict mapping variable name → DataFrame (12 months × 85 subbasins),
        or None if data could not be loaded.
    """
    gcm_label = f"{gcm_info['dir_base']}"
    results = {}

    for variable in VARIABLES:
        print(f"  [{gcm_label}] {variable}: loading historical climatology ...")
        hist_clim = build_monthly_climatology(fs, gcm_info, EXP_HIST, variable, HIST_YEARS)
        if hist_clim is None:
            print(f"  [{gcm_label}] {variable}: no historical data — skipping")
            continue

        print(f"  [{gcm_label}] {variable}: loading future climatology ...")
        fut_clim = build_monthly_climatology(fs, gcm_info, EXP_SSP370, variable, FUTURE_YEARS)
        if fut_clim is None:
            print(f"  [{gcm_label}] {variable}: no future data — skipping")
            continue

        print(f"  [{gcm_label}] {variable}: computing delta and aggregating ...")
        delta = compute_delta(hist_clim, fut_clim, variable)
        df = aggregate_to_subbasins(delta, masks, cos_weights)
        results[variable] = df

    return results if results else None


# ---------------------------------------------------------------------------
# Full-record annual timeseries
# ---------------------------------------------------------------------------

def build_annual_timeseries(
    fs: s3fs.S3FileSystem,
    gcm_info: dict,
    masks: dict[str, np.ndarray],
    cos_weights: np.ndarray,
) -> dict[str, pd.DataFrame] | None:
    """Load the full historical + SSP370 record and return annual subbasin means.

    Iterates over ALL_HIST_YEARS (historical_bc scenario) then ALL_SSP370_YEARS
    (ssp370_bc scenario), skipping any year whose file is absent on S3.  For
    each present year the daily DataArray is averaged over time to produce an
    annual mean, which is then aggregated to subbasins via weighted_basin_mean.

    Args:
        fs: Anonymous s3fs filesystem.
        gcm_info: Entry from GCM_CATALOG.
        masks: Subbasin boolean masks from grid.build_subbasin_masks.
        cos_weights: Cosine-latitude weights from grid.build_subbasin_masks.

    Returns:
        Dict mapping variable → DataFrame with shape (n_years, n_subbasins).
        Index: integer years (sorted).  Columns: subbasin names.
        Returns None if no data could be loaded for any variable.
    """
    gcm_label = gcm_info["dir_base"]
    year_scenario_pairs = (
        [(y, EXP_HIST) for y in ALL_HIST_YEARS]
        + [(y, EXP_SSP370) for y in ALL_SSP370_YEARS]
    )

    results = {}
    for variable in VARIABLES:
        print(f"  [{gcm_label}] {variable}: loading full timeseries ...")
        records: dict[int, dict[str, float]] = {}
        for year, scenario in year_scenario_pairs:
            da = open_annual_file(fs, gcm_info, scenario, variable, year)
            if da is None:
                continue
            annual_mean_2d = da.mean(dim="time").values  # (ny, nx)
            records[year] = {
                name: weighted_basin_mean(annual_mean_2d, mask, cos_weights)
                for name, mask in masks.items()
            }

        if not records:
            continue

        df = pd.DataFrame.from_dict(records, orient="index")
        df.index.name = "year"
        df.sort_index(inplace=True)
        results[variable] = df

    return results if results else None
