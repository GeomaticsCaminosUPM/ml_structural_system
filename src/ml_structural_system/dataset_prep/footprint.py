"""Compute missing footprint features with ``footprint_attributes.run()``."""

from __future__ import annotations

import geopandas as gpd


def compute_missing_columns(gdf: gpd.GeoDataFrame, missing_columns: list[str]) -> gpd.GeoDataFrame:
    """Compute ``missing_columns`` on ``gdf`` with ``footprint_attributes.run()``.

    Geographic data is reprojected to a local Universal Transverse Mercator (UTM)
    CRS for the computation and back to its original CRS afterwards.

    Args:
        gdf: Building footprints.
        missing_columns: Feature columns to compute.

    Returns:
        The footprints with the computed columns added.
    """
    import footprint_attributes

    original_crs = gdf.crs
    working = gdf
    if original_crs is not None and original_crs.is_geographic:
        working = gdf.to_crs(gdf.estimate_utm_crs())

    result = footprint_attributes.run(working, config={"columns": missing_columns})

    if original_crs is not None and result.crs != original_crs:
        result = result.to_crs(original_crs)
    return result
