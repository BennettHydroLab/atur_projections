"""
Top-level worker function for distributed GCM processing.

Must be defined at module level (not inside another function) so that Dask
workers spawned via LocalCluster can import and serialise it correctly.
"""

from __future__ import annotations

import numpy as np
import s3fs

from src.climate import process_gcm, build_annual_timeseries


def process_gcm_task(
    gcm_info: dict,
    masks: dict[str, np.ndarray],
    cos_weights: np.ndarray,
) -> tuple[dict | None, dict | None]:
    """Run process_gcm + build_annual_timeseries for one GCM.

    Each Dask worker calls this in its own process and creates its own
    anonymous S3FileSystem so the non-serialisable filesystem object never
    has to cross the process boundary.

    Args:
        gcm_info:    Entry from GCM_CATALOG.
        masks:       Subbasin boolean masks (scattered to workers).
        cos_weights: Cosine-latitude weight array (scattered to workers).

    Returns:
        (result, ts) tuple — each is either a dict of DataFrames or None.
    """
    fs = s3fs.S3FileSystem(anon=True)
    result = process_gcm(fs, gcm_info, masks, cos_weights)
    ts = build_annual_timeseries(fs, gcm_info, masks, cos_weights) if result is not None else None
    return result, ts
