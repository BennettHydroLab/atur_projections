"""
Results saving and visualisation for climate delta analysis.

Output directory layout:
    results/
        <gcm_dir_base>/
            t2_delta.csv        # (12 months) × (85 subbasins), units: K
            prec_delta.csv      # (12 months) × (85 subbasins), units: %
        ensemble_t2_delta_summary.csv
        ensemble_prec_delta_summary.csv
        figures/
            ensemble_t2_delta_<month>.png
            ensemble_prec_delta_<month>.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.io.shapereader as shpreader
import geopandas as gpd

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


# ---------------------------------------------------------------------------
# Per-GCM CSV saving
# ---------------------------------------------------------------------------

def save_gcm_results(
    gcm_label: str,
    gcm_results: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Save per-GCM monthly delta DataFrames to CSV.

    Args:
        gcm_label: Directory-base name of the GCM (used as subfolder name).
        gcm_results: Dict variable → DataFrame (12 × n_subbasins).
        output_dir: Root results directory.
    """
    gcm_dir = output_dir / gcm_label
    gcm_dir.mkdir(parents=True, exist_ok=True)
    for variable, df in gcm_results.items():
        path = gcm_dir / f"{variable}_delta.csv"
        df.to_csv(path)


# ---------------------------------------------------------------------------
# Per-GCM map plotting
# ---------------------------------------------------------------------------

def plot_gcm_maps(
    gcm_label: str,
    gcm_results: dict[str, pd.DataFrame],
    shapefile_path: str,
    output_dir: Path,
) -> None:
    """Generate one map per variable per month for a single GCM.

    Args:
        gcm_label: Directory-base name of the GCM (used as subfolder name).
        gcm_results: Dict variable → DataFrame (12 months × n_subbasins).
        shapefile_path: Path to GW_SubBasins shapefile.
        output_dir: Root results directory; figures go in output_dir/gcm_label/figures/.
    """
    fig_dir = output_dir / gcm_label / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    basins = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
    if "SUBBASIN_N" in basins.columns:
        basins["_label"] = basins["SUBBASIN_N"].fillna(
            pd.Series(basins.index.astype(str), index=basins.index)
        )
    else:
        basins["_label"] = basins.index.astype(str)

    for variable, df in gcm_results.items():
        for month in range(1, 13):
            if month not in df.index:
                continue
            month_row = df.loc[month]  # Series: subbasin → delta value
            basins_month = basins.copy()
            basins_month["delta"] = basins_month["_label"].map(month_row)

            month_name = MONTH_NAMES[month - 1]
            title = (
                f"{gcm_label} — {variable} delta — {month_name}\n"
                f"SSP3-7.0  |  2070–2099 vs 1980–2009"
            )
            fname = fig_dir / f"{variable}_delta_{month:02d}_{month_name}.png"
            plot_spatial_delta(basins_month, title, fname, variable)
            print(f"  Saved: {fname.name}")


# ---------------------------------------------------------------------------
# Ensemble summary
# ---------------------------------------------------------------------------

def build_ensemble_summary(
    all_results: dict[str, dict[str, pd.DataFrame]],
    variable: str,
) -> pd.DataFrame:
    """Stack per-GCM results and compute ensemble statistics.

    Args:
        all_results: {gcm_label: {variable: DataFrame}}.
        variable: 't2' or 'prec'.

    Returns:
        Long-format DataFrame with columns:
            subbasin, month, mean_delta, std_delta, p10_delta, p90_delta
    """
    # Stack all GCMs into a 3D structure: (n_gcm, 12, n_subbasins)
    frames = []
    for gcm_label, results in all_results.items():
        if variable not in results:
            continue
        df = results[variable].copy()
        df["gcm"] = gcm_label
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    # Wide → long: columns are subbasins, index is month, extra col is gcm
    stacked = pd.concat(frames)  # (n_gcm * 12) × (n_subbasins + 1)
    subbasin_cols = [c for c in stacked.columns if c != "gcm"]

    records = []
    for month in range(1, 13):
        month_data = stacked[stacked.index == month][subbasin_cols]
        for subbasin in subbasin_cols:
            vals = month_data[subbasin].dropna().values
            records.append({
                "subbasin": subbasin,
                "month": month,
                "mean_delta": float(np.mean(vals)) if len(vals) else np.nan,
                "std_delta": float(np.std(vals)) if len(vals) else np.nan,
                "p10_delta": float(np.percentile(vals, 10)) if len(vals) else np.nan,
                "p90_delta": float(np.percentile(vals, 90)) if len(vals) else np.nan,
                "n_gcm": len(vals),
            })

    return pd.DataFrame(records)


def save_ensemble_summary(
    all_results: dict[str, dict[str, pd.DataFrame]],
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Compute and save ensemble summary CSVs for all variables.

    Returns:
        Dict variable → summary DataFrame (for downstream plotting).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for variable in ["t2", "prec"]:
        summary = build_ensemble_summary(all_results, variable)
        if summary.empty:
            continue
        path = output_dir / f"ensemble_{variable}_delta_summary.csv"
        summary.to_csv(path, index=False)
        summaries[variable] = summary
    return summaries


# ---------------------------------------------------------------------------
# Spatial map plotting
# ---------------------------------------------------------------------------

def plot_spatial_delta(
    basins_gdf: gpd.GeoDataFrame,
    title: str,
    output_path: Path,
    variable: str,
) -> None:
    """Plot a choropleth map with each subbasin polygon filled by its delta value.

    Uses a plain matplotlib axes (no CartoPy rendering) to avoid a CartoPy 0.25
    / Python 3.13 incompatibility with WeakValueDictionary and NoneType geometries.
    Context lines (states, borders, coastline) are drawn via geopandas using
    Natural Earth shapefiles resolved through CartoPy's data cache.

    Args:
        basins_gdf: GeoDataFrame in EPSG:4326 with a 'delta' column of values to plot.
        title: Figure title string.
        output_path: File path for the saved PNG.
        variable: 't2' or 'prec' — controls colormap and label.
    """
    if variable == "t2":
        cmap_name = "RdBu_r"
        cbar_label = "Temperature change (K)"
    else:
        cmap_name = "BrBG"
        cbar_label = "Precipitation change (%)"

    vals = basins_gdf["delta"].dropna()
    vmax = max(abs(vals.min()), abs(vals.max())) if len(vals) > 0 else 1.0
    vmax = vmax if vmax > 0 else 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Choropleth — subbasin polygons filled by delta value
    basins_gdf.plot(
        ax=ax,
        column="delta",
        cmap=cmap_name,
        norm=norm,
        edgecolor="black",
        linewidth=0.5,
        legend=False,
        missing_kwds={"color": (0.85, 0.85, 0.85), "edgecolor": "black", "linewidth": 0.5},
    )

    # Context lines from Natural Earth (read via geopandas — no CartoPy rendering)
    _NE = [
        ("cultural", "admin_1_states_provinces_lines", "gray",  0.5),
        ("cultural", "admin_0_boundary_lines_land",    "black", 0.5),
        ("physical", "coastline",                      "black", 0.5),
    ]
    for category, name, color, lw in _NE:
        path = shpreader.natural_earth(resolution="50m", category=category, name=name)
        gpd.read_file(path).plot(ax=ax, color=color, linewidth=lw)

    lon_min, lat_min, lon_max, lat_max = basins_gdf.total_bounds
    ax.set_xlim(lon_min - 1, lon_max + 1)
    ax.set_ylim(lat_min - 1, lat_max + 1)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(linewidth=0.3, color="gray", alpha=0.5)

    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=cbar_label, shrink=0.7, pad=0.05)
    ax.set_title(title, fontsize=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ensemble_maps(
    summaries: dict[str, pd.DataFrame],
    shapefile_path: str,
    output_dir: Path,
) -> None:
    """Generate one map per variable per month showing ensemble-mean delta.

    Args:
        summaries: Output of save_ensemble_summary.
        shapefile_path: Path to GW_SubBasins shapefile.
        output_dir: Root results directory.
    """
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    basins = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
    if "SUBBASIN_N" in basins.columns:
        basins["_label"] = basins["SUBBASIN_N"].fillna(
            pd.Series(basins.index.astype(str), index=basins.index)
        )
    else:
        basins["_label"] = basins.index.astype(str)

    for variable, summary in summaries.items():
        for month in range(1, 13):
            month_data = summary[summary["month"] == month].set_index("subbasin")["mean_delta"]
            basins_month = basins.copy()
            basins_month["delta"] = basins_month["_label"].map(month_data)

            month_name = MONTH_NAMES[month - 1]
            title = (
                f"Ensemble-mean {variable} delta — {month_name}\n"
                f"SSP3-7.0  |  2070–2099 vs 1980–2009"
            )
            fname = fig_dir / f"ensemble_{variable}_delta_{month:02d}_{month_name}.png"
            plot_spatial_delta(basins_month, title, fname, variable)
            print(f"  Saved: {fname.name}")
