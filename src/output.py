"""
Results saving and visualisation for climate delta analysis.

Output directory layout:
    results/
        <gcm_dir_base>/
            t2_delta.csv        # (12 months) × (52 basins), units: K
            prec_delta.csv      # (12 months) × (52 basins), units: mm day-1
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
from .catalog import HIST_YEARS
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
    """Generate one 4×3 monthly grid figure per variable for a single GCM.

    Args:
        gcm_label: Directory-base name of the GCM (used as subfolder name).
        gcm_results: Dict variable → DataFrame (12 months × n_subbasins).
        shapefile_path: Path to GW_Shapefile shapefile.
        output_dir: Root results directory; figures go in output_dir/gcm_label/figures/.
    """
    fig_dir = output_dir / gcm_label / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    basins = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
    if "BASIN_NAME" in basins.columns:
        basins["_label"] = basins["BASIN_NAME"].fillna(
            pd.Series(basins.index.astype(str), index=basins.index)
        )
    else:
        basins["_label"] = basins.index.astype(str)

    for variable, df in gcm_results.items():
        basins_by_month = {}
        for month in range(1, 13):
            if month not in df.index:
                continue
            basins_month = basins.copy()
            basins_month["delta"] = basins_month["_label"].map(df.loc[month])
            basins_by_month[month] = basins_month

        suptitle = (
            f"{gcm_label} — {variable} delta\n"
            f"SSP3-7.0  |  2070–2099 vs 1980–2009"
        )
        fname = fig_dir / f"{variable}_delta_all_months.png"
        plot_monthly_grid(basins_by_month, variable, suptitle, fname)


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
        cbar_label = "Precipitation change (mm day\u207b\u00b9)"

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


def _draw_basin_map(
    ax: plt.Axes,
    basins_gdf: gpd.GeoDataFrame,
    norm: mcolors.Normalize,
    cmap_name: str,
    ne_layers: list,
    extent: tuple,
) -> None:
    """Draw a choropleth + Natural Earth context lines into an existing axes."""
    basins_gdf.plot(
        ax=ax,
        column="delta",
        cmap=cmap_name,
        norm=norm,
        edgecolor="black",
        linewidth=0.3,
        legend=False,
        missing_kwds={"color": (0.85, 0.85, 0.85), "edgecolor": "black", "linewidth": 0.3},
    )
    lon_min, lat_min, lon_max, lat_max = extent
    for gdf, color, lw in ne_layers:
        gdf.plot(ax=ax, color=color, linewidth=lw)
    ax.set_xlim(lon_min - 1, lon_max + 1)
    ax.set_ylim(lat_min - 1, lat_max + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(linewidth=0.3, color="gray", alpha=0.5)


def plot_monthly_grid(
    basins_by_month: dict[int, gpd.GeoDataFrame],
    variable: str,
    suptitle: str,
    output_path: Path,
) -> None:
    """Plot all 12 months in a 4-column × 3-row grid with a shared colorbar.

    Args:
        basins_by_month: Mapping month (1–12) → GeoDataFrame with a 'delta' column.
        variable: 't2' or 'prec' — selects colormap and colorbar label.
        suptitle: Figure-level title.
        output_path: Destination PNG path.
    """
    if variable == "t2":
        cmap_name = "RdBu_r"
        cbar_label = "Temperature change (K)"
    else:
        cmap_name = "BrBG"
        cbar_label = "Precipitation change (mm day\u207b\u00b9)"

    # Shared symmetric vmax across all months
    all_vals = pd.concat(
        [gdf["delta"].dropna() for gdf in basins_by_month.values()]
    )
    vmax = max(abs(all_vals.min()), abs(all_vals.max())) if len(all_vals) > 0 else 1.0
    vmax = vmax if vmax > 0 else 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # Load Natural Earth context layers once for all subplots
    _NE_DEFS = [
        ("cultural", "admin_1_states_provinces_lines", "gray",  0.4),
        ("cultural", "admin_0_boundary_lines_land",    "black", 0.4),
        ("physical", "coastline",                      "black", 0.4),
    ]
    ne_layers = []
    for category, name, color, lw in _NE_DEFS:
        shp_path = shpreader.natural_earth(resolution="50m", category=category, name=name)
        ne_layers.append((gpd.read_file(shp_path), color, lw))

    first_gdf = next(iter(basins_by_month.values()))
    extent = first_gdf.total_bounds  # (lon_min, lat_min, lon_max, lat_max)

    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    for idx in range(12):
        month = idx + 1
        row, col = divmod(idx, 4)
        ax = axes[row, col]
        ax.set_title(MONTH_NAMES[idx], fontsize=10)
        if month in basins_by_month:
            _draw_basin_map(ax, basins_by_month[month], norm, cmap_name, ne_layers, extent)

    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label=cbar_label, shrink=0.6, pad=0.02, aspect=40)
    fig.suptitle(suptitle, fontsize=14)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_ensemble_maps(
    summaries: dict[str, pd.DataFrame],
    shapefile_path: str,
    output_dir: Path,
) -> None:
    """Generate one 4×3 monthly grid figure per variable showing ensemble-mean delta.

    Args:
        summaries: Output of save_ensemble_summary.
        shapefile_path: Path to GW_Shapefile shapefile.
        output_dir: Root results directory.
    """
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    basins = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
    if "BASIN_NAME" in basins.columns:
        basins["_label"] = basins["BASIN_NAME"].fillna(
            pd.Series(basins.index.astype(str), index=basins.index)
        )
    else:
        basins["_label"] = basins.index.astype(str)

    for variable, summary in summaries.items():
        basins_by_month = {}
        for month in range(1, 13):
            month_data = summary[summary["month"] == month].set_index("subbasin")["mean_delta"]
            basins_month = basins.copy()
            basins_month["delta"] = basins_month["_label"].map(month_data)
            basins_by_month[month] = basins_month

        suptitle = (
            f"Ensemble-mean {variable} delta\n"
            f"SSP3-7.0  |  2070–2099 vs 1980–2009"
        )
        fname = fig_dir / f"ensemble_{variable}_delta_all_months.png"
        plot_monthly_grid(basins_by_month, variable, suptitle, fname)


# ---------------------------------------------------------------------------
# Spaghetti plots
# ---------------------------------------------------------------------------

def plot_spaghetti(
    df: pd.DataFrame,
    variable: str,
    title: str,
    output_path: Path,
) -> None:
    """Monthly spaghetti plot: one thin line per subbasin + bold domain-mean overlay.

    Args:
        df: DataFrame with month index (1–12) and one column per subbasin.
        variable: 't2' or 'prec' — controls y-axis label and mean line colour.
        title: Figure title.
        output_path: Destination PNG path.
    """
    if variable == "t2":
        ylabel = "Temperature change (K)"
        mean_color = "#d73027"
    else:
        ylabel = "Precipitation change (mm day\u207b\u00b9)"
        mean_color = "#1a9850"

    months = list(range(1, 13))
    domain_mean = df.mean(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))

    for col in df.columns:
        ax.plot(months, df[col].values, color="gray", linewidth=0.6, alpha=0.35)

    ax.plot(months, domain_mean.values, color=mean_color, linewidth=2.5,
            label="Domain mean", zorder=5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=4)
    ax.set_xlim(1, 12)
    ax.set_xticks(months)
    ax.set_xticklabels(MONTH_NAMES, fontsize=9)
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    ax.grid(linewidth=0.3, color="gray", alpha=0.5)
    ax.set_title(title, fontsize=12)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_gcm_spaghetti(
    gcm_label: str,
    gcm_results: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Generate one spaghetti figure per variable for a single GCM.

    Args:
        gcm_label: Directory-base name of the GCM (used as subfolder name).
        gcm_results: Dict variable → DataFrame (12 months × n_subbasins).
        output_dir: Root results directory; figures go in output_dir/gcm_label/figures/.
    """
    fig_dir = output_dir / gcm_label / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for variable, df in gcm_results.items():
        title = (
            f"{gcm_label} — {variable} delta by subbasin\n"
            f"SSP3-7.0  |  2070–2099 vs 1980–2009"
        )
        fname = fig_dir / f"{variable}_delta_spaghetti.png"
        plot_spaghetti(df, variable, title, fname)


def plot_ensemble_spaghetti(
    summaries: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Generate one spaghetti figure per variable using ensemble-mean deltas.

    Args:
        summaries: Output of save_ensemble_summary.
        output_dir: Root results directory; figures go in output_dir/figures/.
    """
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for variable, summary in summaries.items():
        df = summary.pivot(index="month", columns="subbasin", values="mean_delta")
        title = (
            f"Ensemble-mean {variable} delta by subbasin\n"
            f"SSP3-7.0  |  2070–2099 vs 1980–2009"
        )
        fname = fig_dir / f"ensemble_{variable}_delta_spaghetti.png"
        plot_spaghetti(df, variable, title, fname)


# ---------------------------------------------------------------------------
# Full-record annual deviation timeseries
# ---------------------------------------------------------------------------

def _deviation_panels(
    axes,
    var_cfg: dict,
    get_series_iter,  # callable(variable) → iterable of (label, Series, alpha, lw, color)
    domain_mean_per_var: dict,
) -> None:
    """Shared panel-drawing logic for both per-GCM and ensemble timeseries figures."""
    for ax, (variable, cfg) in zip(axes, var_cfg.items()):
        series_iter = get_series_iter(variable)
        if series_iter is None:
            ax.set_visible(False)
            continue
        for _lbl, ser, alpha, lw, color in series_iter:
            ax.plot(ser.index, ser.values, color=color, linewidth=lw, alpha=alpha)
        dmean = domain_mean_per_var.get(variable)
        if dmean is not None:
            ax.plot(dmean.index, dmean.values, color=cfg["color"],
                    linewidth=2.5, label="Domain mean", zorder=5)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=4)
        ax.axvline(2014.5, color="black", linewidth=1.0, linestyle=":", alpha=0.6,
                   label="Hist / SSP3-7.0 boundary")
        ax.set_ylabel(cfg["ylabel"])
        ax.set_title(cfg["title"], fontsize=11)
        ax.grid(linewidth=0.3, color="gray", alpha=0.5)
        ax.legend(fontsize=9)


def plot_gcm_timeseries(
    gcm_label: str,
    ts_data: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Plot annual deviation timeseries for a single GCM with per-subbasin lines.

    Two-panel figure:
        Top:    Annual temperature deviation per subbasin + bold domain mean (K)
        Bottom: Annual precipitation deviation per subbasin + bold domain mean
                (mm day⁻¹)

    Each subbasin's deviation is computed relative to its own 1980–2009 mean, so
    spatial gradients are removed and only temporal change is shown.

    Args:
        gcm_label: GCM directory-base name (used as subfolder name).
        ts_data: {variable: DataFrame(year × subbasin)}.
        output_dir: Root results directory; figure saved to
                    output_dir/gcm_label/figures/.
    """
    _BASELINE = set(HIST_YEARS)

    var_cfg = {
        "t2": {
            "ylabel": "Temperature deviation (K)",
            "color": "#d73027",
            "title": "Annual temperature deviation from 1980\u20132009 mean",
        },
        "prec": {
            "ylabel": "Precipitation deviation (mm day\u207b\u00b9)",
            "color": "#1a9850",
            "title": "Annual precipitation deviation from 1980\u20132009 mean",
        },
    }

    domain_means = {}
    precomputed: dict[str, pd.DataFrame] = {}
    for variable, df in ts_data.items():
        if variable not in var_cfg:
            continue
        baseline_means = df[df.index.isin(_BASELINE)].mean(axis=0)
        dev_df = df - baseline_means
        precomputed[variable] = dev_df
        domain_means[variable] = dev_df.mean(axis=1)

    def get_series_iter(variable):
        if variable not in precomputed:
            return None
        dev_df = precomputed[variable]
        return (
            (col, dev_df[col], 0.2, 0.5, "gray")
            for col in dev_df.columns
        )

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    _deviation_panels(axes, var_cfg, get_series_iter, domain_means)
    axes[1].set_xlabel("Year")
    fig.suptitle(
        f"{gcm_label} \u2014 annual deviations from 1980\u20132009 baseline",
        fontsize=13,
    )
    plt.tight_layout()

    fig_dir = output_dir / gcm_label / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    output_path = fig_dir / "timeseries_annual_deviations.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")


def plot_timeseries_deviations(
    all_ts: dict[str, dict[str, pd.DataFrame]],
    output_dir: Path,
) -> None:
    """Plot annual deviation timeseries for all GCMs across the full record.

    Two-panel figure:
        Top:    Annual domain-mean temperature deviation from 1980–2009 baseline (K)
        Bottom: Annual domain-mean precipitation deviation from 1980–2009 baseline
                (mm day⁻¹)

    Each GCM is drawn as a thin semi-transparent line; the ensemble mean is a bold
    line.  Deviations are computed per GCM by subtracting that GCM's own 1980–2009
    domain mean so that model biases do not inflate spread.  A vertical dotted line
    marks the historical/SSP3-7.0 boundary (end of 2014).

    Args:
        all_ts: {gcm_label: {variable: DataFrame(year × subbasin)}}.
        output_dir: Root results directory; figure saved to output_dir/figures/.
    """
    _BASELINE = set(HIST_YEARS)  # 1980–2009

    var_cfg = {
        "t2": {
            "ylabel": "Temperature deviation (K)",
            "color": "#d73027",
            "title": "Annual temperature deviation from 1980\u20132009 mean",
        },
        "prec": {
            "ylabel": "Precipitation deviation (mm day\u207b\u00b9)",
            "color": "#1a9850",
            "title": "Annual precipitation deviation from 1980\u20132009 mean",
        },
    }

    # Collect per-GCM domain-mean deviation series
    gcm_devs: dict[str, dict[str, pd.Series]] = {v: {} for v in var_cfg}
    for gcm_label, ts_data in all_ts.items():
        for variable, df in ts_data.items():
            if variable not in var_cfg:
                continue
            domain_ts = df.mean(axis=1)  # average across subbasins → Series(year)
            baseline_vals = domain_ts[domain_ts.index.isin(_BASELINE)]
            if baseline_vals.empty:
                continue
            gcm_devs[variable][gcm_label] = domain_ts - baseline_vals.mean()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for ax, (variable, cfg) in zip(axes, var_cfg.items()):
        devs = gcm_devs[variable]
        if not devs:
            ax.set_visible(False)
            continue

        for dev in devs.values():
            ax.plot(dev.index, dev.values, color=cfg["color"],
                    linewidth=0.7, alpha=0.25)

        ens_mean = pd.DataFrame(devs).mean(axis=1)
        ax.plot(ens_mean.index, ens_mean.values, color=cfg["color"],
                linewidth=2.5, label="Ensemble mean", zorder=5)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=4)
        ax.axvline(2014.5, color="black", linewidth=1.0, linestyle=":", alpha=0.6,
                   label="Hist / SSP3-7.0 boundary")
        ax.set_ylabel(cfg["ylabel"])
        ax.set_title(cfg["title"], fontsize=11)
        ax.grid(linewidth=0.3, color="gray", alpha=0.5)
        ax.legend(fontsize=9)

    axes[1].set_xlabel("Year")
    fig.suptitle(
        "WUS-D3 annual deviations from 1980\u20132009 baseline \u2014 SSP3-7.0",
        fontsize=13,
    )
    plt.tight_layout()

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    output_path = fig_dir / "timeseries_annual_deviations.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}")
