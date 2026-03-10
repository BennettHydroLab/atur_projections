"""
Climate delta analysis — WUS-D3 Tier 3 bias-corrected projections.

Computes monthly climatology deltas (SSP3-7.0 end-of-century 2070–2099 vs
historical 1980–2009) for temperature (t2) and precipitation (prec), then
aggregates results to the 52 GW_Shapefile groundwater basins.

GCMs are processed in parallel using Dask Distributed.  Each worker creates
its own anonymous S3 connection; results are saved and plotted serially in
the main process after each future completes.

Usage:
    uv run main.py [--gcm-index N [N ...]] [--workers N] [--dry-run]

    --gcm-index N   Process only the GCMs at these indices in GCM_CATALOG
                    (0-based). Accepts one or more values. Omit for all 16.
    --workers N     Number of parallel Dask workers (default: 4).
    --dry-run       Print S3 paths for the first GCM/year without fetching data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import s3fs
from distributed import Client, LocalCluster, as_completed

from src.catalog import GCM_CATALOG, build_s3_path, EXP_HIST, EXP_SSP370, HIST_YEARS, FUTURE_YEARS
from src.grid import load_wrf_coords, build_subbasin_masks
from src.output import (
    save_gcm_results, plot_gcm_maps, plot_gcm_spaghetti,
    save_ensemble_summary, plot_ensemble_maps, plot_ensemble_spaghetti,
    plot_gcm_timeseries, plot_timeseries_deviations,
)
from src.worker import process_gcm_task

SHAPEFILE = Path(__file__).parent / "GW_Shapefile"
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
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Number of parallel Dask workers (default: 4).",
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
    gcms_to_run = (
        GCM_CATALOG if args.gcm_index is None
        else [GCM_CATALOG[i] for i in args.gcm_index]
    )
    n_workers = min(args.workers, len(gcms_to_run))

    # Step 4: Process GCMs in parallel
    print(f"\nSpawning {n_workers} Dask worker(s) for {len(gcms_to_run)} GCM(s) ...")

    all_results: dict[str, dict] = {}
    all_ts: dict[str, dict] = {}

    with LocalCluster(n_workers=n_workers, threads_per_worker=1, silence_logs=True) as cluster:
        with Client(cluster) as client:
            print(f"  Dashboard: {client.dashboard_link}\n")

            # Scatter shared read-only arrays to workers once (broadcast avoids
            # re-sending per-task; ~35 MB total for 52 basins on a ~600×1000 grid).
            masks_ref = client.scatter(masks, broadcast=True)
            cos_weights_ref = client.scatter(cos_weights, broadcast=True)

            # Submit all GCM tasks
            futures: dict = {}
            for gcm_info in gcms_to_run:
                f = client.submit(process_gcm_task, gcm_info, masks_ref, cos_weights_ref)
                futures[f] = gcm_info

            n_total = len(futures)
            n_done = 0

            # Collect results as they complete; save/plot serially in main process
            for future in as_completed(futures):
                gcm_info = futures[future]
                label = gcm_info["dir_base"]
                n_done += 1
                print(f"[{n_done}/{n_total}] Received {label}")

                try:
                    result, ts = future.result()
                except Exception as exc:
                    print(f"  ERROR in {label}: {exc} — skipping")
                    continue

                if result is None:
                    print(f"  No data returned for {label} — skipping")
                    continue

                all_results[label] = result
                save_gcm_results(label, result, RESULTS_DIR)
                print(f"  Generating per-model figures for {label} ...")
                plot_gcm_maps(label, result, str(SHAPEFILE), RESULTS_DIR)
                plot_gcm_spaghetti(label, result, RESULTS_DIR)

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
