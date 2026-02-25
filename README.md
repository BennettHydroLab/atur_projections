# ATUR Climate Projections

Calculates end-of-century vs. historical climate deltas for precipitation and temperature using the [WUS-D3 downscaled CMIP6 projections](https://dept.atmos.ucla.edu/alexhall/downscaling-cmip6), aggregated to the Arizona groundwater subbasins defined by the GW_SubBasins shapefile.

## What it does

For each of the 16 available bias-corrected GCMs:

1. Loads **Tier 3 daily data** (bias-corrected, 9-km d02 domain) from the public AWS S3 bucket `s3://wrf-cmip6-noversioning`
2. Computes **monthly climatologies** (mean value for each calendar month) over two 30-year periods:
   - **Historical**: 1980–2009
   - **End of century**: 2070–2099 (SSP3-7.0)
3. Computes **deltas** between the two periods:
   - Temperature (`t2`): absolute change in K
   - Precipitation (`prec`): percent change
4. Aggregates grid-cell values to each subbasin using cosine-latitude weighting
5. Outputs per-GCM CSVs, per-GCM choropleth maps, and ensemble summary statistics (mean, std, 10th/90th percentile across all 16 GCMs)
6. Generates ensemble-mean choropleth maps for each month

## Data source

[WUS-D3](https://dept.atmos.ucla.edu/alexhall/downscaling-cmip6) (Rahimi & Huang, updated 2023) — dynamically downscaled CMIP6 GCMs using WRF. 

## Setup

Requires Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/). To install dependencies:

```bash
uv sync
```

And then to activate the environment:

```bash
source .venv/bin/activate
```

No AWS credentials are needed — the bucket is publicly accessible.

## Usage
Once the environment is set up and activated you can use the command line interface to run the analysis. Here are some example commands:

```bash
# Verify S3 paths and file availability for one GCM (no data downloaded)
uv run main.py --dry-run

# Process a single GCM (0-based index into GCM_CATALOG) — good for testing
uv run main.py --gcm-index 0   

# Full run — all 16 GCMs
uv run main.py
```

## Output

```
results/
  <gcm_dir_base>/
    t2_delta.csv          # 12 rows (months) × N columns (subbasins), units: K
    prec_delta.csv        # 12 rows (months) × N columns (subbasins), units: %
    figures/
      t2_delta_01_Jan.png       # choropleth map per variable per month
      prec_delta_01_Jan.png
      ...
  ensemble_t2_delta_summary.csv     # long-format: subbasin, month, mean/std/p10/p90
  ensemble_prec_delta_summary.csv
  figures/
    ensemble_t2_delta_01_Jan.png    # ensemble-mean choropleth per variable per month
    ...
```

Maps are choropleth plots — each subbasin polygon is filled with its delta value. Subbasins with no grid cell coverage are shown in gray.

## Project structure

| File | Purpose |
|------|---------|
| `main.py` | Orchestrator and CLI entry point |
| `src/catalog.py` | S3 path builder; GCM registry; domain constant (`DOMAIN = "d02"`) |
| `src/grid.py` | WRF grid coordinate loading; subbasin spatial mask construction |
| `src/climate.py` | Data loading; monthly climatology; delta computation; subbasin aggregation |
| `src/output.py` | CSV saving; choropleth map generation (geopandas + Natural Earth boundaries) |
| `GW_SubBasins/` | Groundwater subbasin polygons (EPSG:26912) |
