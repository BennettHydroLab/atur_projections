"""
Climate delta analysis — WUS-D3 Tier 3 bias-corrected projections.

Computes monthly climatology deltas (SSP3-7.0 end-of-century 2070–2099 vs
historical 1980–2009) for temperature (t2) and precipitation (prec), then
aggregates results to the 85 GW_SubBasins groundwater subbasins.

Usage:
    uv run main.py [--gcm-index N] [--dry-run]

    --gcm-index N   Process only the Nth GCM (0-based) — useful for testing.
    --dry-run       Print S3 paths for the first GCM/year without fetching data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import s3fs

from src.catalog import GCM_CATALOG, build_s3_path, EXP_HIST, EXP_SSP370, HIST_YEARS, FUTURE_YEARS
from src.grid import load_wrf_coords, build_subbasin_masks
from src.climate import process_gcm, build_annual_timeseries
from src.output import (
    save_gcm_results, plot_gcm_maps, plot_gcm_spaghetti,
    save_ensemble_summary, plot_ensemble_maps, plot_ensemble_spaghetti,
    plot_gcm_timeseries, plot_timeseries_deviations,
)

SHAPEFILE = Path(__file__).parent / "GW_SubBasins"
RESULTS_DIR = Path(__file__).parent / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WUS-D3 climate delta analysis")
    parser.add_argument(
        "--gcm-index",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="Process only the GCMs at these indices in GCM_CATALOG (0-based). "
             "Accepts one or more values (e.g. --gcm-index 0 1 2 3). "
             "Omit to run all 16 GCMs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print example S3 paths for the first GCM without fetching data.",
    )
    return parser.parse_args()


def dry_run(fs: s3fs.S3FileSystem) -> None:
    """Print example S3 paths and check whether they exist, then exit."""
    gcm = GCM_CATALOG[0]
    print(f"\nDry run — GCM: {gcm['dir_base']}\n")
    for scenario, years in [(EXP_HIST, HIST_YEARS[:2]), (EXP_SSP370, FUTURE_YEARS[:2])]:
        for variable in ["t2", "prec"]:
            for year in years:
                path = build_s3_path(gcm, scenario, variable, year)
                exists = fs.exists(path)
                print(f"  {'OK' if exists else 'MISSING'} {path.split('/')[-1]}")
    print()


def main() -> None:
    args = parse_args()

    print("Initialising anonymous S3 filesystem ...")
    fs = s3fs.S3FileSystem(anon=True)

    if args.dry_run:
        dry_run(fs)
        return

    # Step 1: Load WRF d01 grid coordinates
    print("Loading WRF d01 grid coordinates ...")
    xlat, xlon = load_wrf_coords(fs)
    print(f"  Grid shape: {xlat.shape}")

    # Step 2: Build subbasin spatial masks
    print(f"Building subbasin masks from {SHAPEFILE} ...")
    masks, cos_weights = build_subbasin_masks(str(SHAPEFILE), xlat, xlon)
    print(f"  {len(masks)} subbasins with at least one grid cell")

    # Step 3: Determine which GCMs to process
    gcms_to_run = GCM_CATALOG if args.gcm_index is None else [GCM_CATALOG[i] for i in args.gcm_index]

    # Step 4: Process each GCM
    all_results: dict[str, dict] = {}
    all_ts: dict[str, dict] = {}
    for i, gcm_info in enumerate(gcms_to_run):
        label = gcm_info["dir_base"]
        print(f"\n[{i + 1}/{len(gcms_to_run)}] Processing {label} ...")
        result = process_gcm(fs, gcm_info, masks, cos_weights)
        if result is None:
            print(f"  No data returned for {label} — skipping")
            continue
        all_results[label] = result
        save_gcm_results(label, result, RESULTS_DIR)
        print(f"  Generating per-model figures for {label} ...")
        plot_gcm_maps(label, result, str(SHAPEFILE), RESULTS_DIR)
        plot_gcm_spaghetti(label, result, RESULTS_DIR)
        ts = build_annual_timeseries(fs, gcm_info, masks, cos_weights)
        if ts is not None:
            all_ts[label] = ts
            plot_gcm_timeseries(label, ts, RESULTS_DIR)
        print(f"  Saved results for {label}")

    if not all_results:
        print("\nNo results produced — check S3 connectivity and GCM availability.")
        return

    # Step 5: Ensemble summary
    print("\nComputing ensemble summary statistics ...")
    summaries = save_ensemble_summary(all_results, RESULTS_DIR)
    print(f"  Saved ensemble summaries to {RESULTS_DIR}")

    # Step 6: Spatial maps and spaghetti plots
    print("Generating spatial delta maps ...")
    plot_ensemble_maps(summaries, str(SHAPEFILE), RESULTS_DIR)
    plot_ensemble_spaghetti(summaries, RESULTS_DIR)

    if all_ts:
        print("Generating ensemble timeseries figure ...")
        plot_timeseries_deviations(all_ts, RESULTS_DIR)

    print(f"\nDone. Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
