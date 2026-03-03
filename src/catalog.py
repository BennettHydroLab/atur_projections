"""
S3 path building and GCM catalog for WUS-D3 Tier 3 bias-corrected data.

Bucket: s3://wrf-cmip6-noversioning (public/anonymous access)
Path:   downscaled_products/gcm/<gcm_dir>/postprocess/<domain>/
File:   <var>.daily.<gcm>.bc.<variant>.<exp_id>.<domain>.<year>.nc
"""

import s3fs

BUCKET = "wrf-cmip6-noversioning"
GCM_BASE = f"{BUCKET}/downscaled_products/gcm"
COORD_BASE = f"{BUCKET}/downscaled_products/wrf_coordinates"

DOMAIN = "d02"  # 9-km domain

# Historical and future time periods (30-year climatology windows)
HIST_YEARS = list(range(1980, 2010))   # 1980–2009
FUTURE_YEARS = list(range(2070, 2100)) # 2070–2099

# Full available record for timeseries analysis
# Historical BC files typically cover 1950–2014; SSP370 BC covers 2015–2100.
# open_annual_file returns None for any year not present on S3, so these ranges
# are inclusive upper bounds rather than guarantees.
ALL_HIST_YEARS = list(range(1950, 2015))    # 1950–2014 (historical_bc)
ALL_SSP370_YEARS = list(range(2015, 2101))  # 2015–2100 (ssp370_bc)

# All 16 GCMs available with both historical_bc and ssp370_bc variants.
# Each entry: (gcm_label_in_filename, variant, directory_base_name)
# The directory name is: <dir_base>_historical_bc  /  <dir_base>_ssp370_bc
GCM_CATALOG = [
    {"gcm": "access-cm2",      "variant": "r5i1p1f1",  "dir_base": "access-cm2_r5i1p1f1"},
    {"gcm": "canesm5",         "variant": "r1i1p2f1",  "dir_base": "canesm5_r1i1p2f1"},
    {"gcm": "cesm2",           "variant": "r11i1p1f1", "dir_base": "cesm2_r11i1p1f1"},
    {"gcm": "cnrm-esm2-1",     "variant": "r1i1p1f2",  "dir_base": "cnrm-esm2-1_r1i1p1f2"},
    {"gcm": "ec-earth3-veg",   "variant": "r1i1p1f1",  "dir_base": "ec-earth3-veg_r1i1p1f1"},
    {"gcm": "ec-earth3",       "variant": "r1i1p1f1",  "dir_base": "ec-earth3_r1i1p1f1"},
    {"gcm": "ec-earth3",       "variant": "r1i1p1f1",  "dir_base": "ec-earth3_r1i1p1f1_2"},
    {"gcm": "fgoals-g3",       "variant": "r1i1p1f1",  "dir_base": "fgoals-g3_r1i1p1f1"},
    {"gcm": "giss-e2-1-g",     "variant": "r1i1p1f2",  "dir_base": "giss-e2-1-g_r1i1p1f2"},
    {"gcm": "miroc6",          "variant": "r1i1p1f1",  "dir_base": "miroc6_r1i1p1f1"},
    {"gcm": "mpi-esm1-2-hr",   "variant": "r3i1p1f1",  "dir_base": "mpi-esm1-2-hr_r3i1p1f1"},
    {"gcm": "mpi-esm1-2-hr",   "variant": "r7i1p1f1",  "dir_base": "mpi-esm1-2-hr_r7i1p1f1"},
    {"gcm": "mpi-esm1-2-lr",   "variant": "r7i1p1f1",  "dir_base": "mpi-esm1-2-lr_r7i1p1f1"},
    {"gcm": "noresm2-mm",      "variant": "r1i1p1f1",  "dir_base": "noresm2-mm_r1i1p1f1"},
    {"gcm": "taiesm1",         "variant": "r1i1p1f1",  "dir_base": "taiesm1_r1i1p1f1"},
    {"gcm": "ukesm1-0-ll",     "variant": "r2i1p1f2",  "dir_base": "ukesm1-0-ll_r2i1p1f2"},
]

# Experiment ID strings used in filenames and directory suffixes
EXP_HIST = "hist"
EXP_SSP370 = "ssp370"


def get_gcm_dir(gcm_info: dict, scenario: str) -> str:
    """Return the S3 directory name for a given GCM and scenario.

    Args:
        gcm_info: Entry from GCM_CATALOG.
        scenario: Either EXP_HIST or EXP_SSP370.

    Returns:
        Full S3 path to the postprocess/<domain>/ directory.
    """
    suffix = "historical_bc" if scenario == EXP_HIST else "ssp370_bc"
    return f"{GCM_BASE}/{gcm_info['dir_base']}_{suffix}/postprocess/{DOMAIN}"


def build_s3_path(gcm_info: dict, scenario: str, variable: str, year: int) -> str:
    """Construct the S3 path for a single Tier 3 NetCDF file.

    File naming convention for bias-corrected files:
        <var>.daily.<gcm>.bc.<variant>.<exp_id>.<domain>.<year>.nc

    Args:
        gcm_info: Entry from GCM_CATALOG.
        scenario: Either EXP_HIST or EXP_SSP370.
        variable: Variable label, e.g. 't2' or 'prec'.
        year: 4-digit year integer.

    Returns:
        Full S3 URI string.
    """
    gcm = gcm_info["gcm"]
    variant = gcm_info["variant"]
    dir_path = get_gcm_dir(gcm_info, scenario)
    filename = f"{variable}.daily.{gcm}.{variant}.{scenario}.bias-correct.{DOMAIN}.{year}.nc"
    return f"s3://{dir_path}/{filename}"


def list_available_years(
    fs: s3fs.S3FileSystem,
    gcm_info: dict,
    scenario: str,
    variable: str,
) -> list[int]:
    """Return sorted list of years actually present on S3 for a GCM/scenario/variable.

    Args:
        fs: Authenticated (anon) s3fs filesystem.
        gcm_info: Entry from GCM_CATALOG.
        scenario: Either EXP_HIST or EXP_SSP370.
        variable: Variable label.

    Returns:
        Sorted list of integer years found on S3.
    """
    dir_path = get_gcm_dir(gcm_info, scenario)
    gcm = gcm_info["gcm"]
    variant = gcm_info["variant"]
    pattern = f"{dir_path}/{variable}.daily.{gcm}.{variant}.{scenario}.bias-correct.{DOMAIN}.*.nc"
    paths = fs.glob(pattern)
    years = []
    for p in paths:
        fname = p.split("/")[-1]
        # Extract year from second-to-last dot-separated token
        try:
            year = int(fname.split(".")[-2])
            years.append(year)
        except (ValueError, IndexError):
            continue
    return sorted(years)


def get_wrf_coord_path() -> str:
    """Return S3 path to the WRF coordinate NetCDF file for the active domain."""
    return f"s3://{COORD_BASE}/wrfinput_{DOMAIN}_coord.nc"
