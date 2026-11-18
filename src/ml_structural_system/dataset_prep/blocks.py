"""Group touching buildings into blocks, the unit of the spatially grouped split."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

#: Footprints closer than this (in meters) are considered to touch.
TOUCH_TOLERANCE_M = 0.05


def add_block_ids(
    gdf: gpd.GeoDataFrame,
    prefix: str = "",
    column: str = "block_id",
    tolerance: float = TOUCH_TOLERANCE_M,
) -> gpd.GeoDataFrame:
    """Group touching footprints into blocks.

    Two buildings belong to the same block if their footprints are closer than
    ``tolerance``. Blocks are the connected components of that graph, so a
    chain of touching buildings forms one block. Splitting by block keeps
    neighbors, which share contact features, in the same partition.

    Args:
        gdf: Building footprints. Reprojected to a metric CRS internally if
            the CRS is geographic.
        prefix: Text prepended to every block ID, e.g. the source file name,
            so IDs stay unique when several files are combined.
        column: Name of the output column.
        tolerance: Distance in meters under which two footprints touch.

    Returns:
        A copy of ``gdf`` with a string block ID column.
    """
    geoms = gdf.geometry
    if gdf.crs is not None and gdf.crs.is_geographic:
        geoms = geoms.to_crs(gdf.estimate_utm_crs())
    geoms = geoms.reset_index(drop=True)
    left, right = geoms.sindex.query(geoms.buffer(tolerance), predicate="intersects")
    n = len(geoms)
    graph = coo_matrix((np.ones(len(left)), (left, right)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    out = gdf.copy()
    out[column] = [f"{prefix}{k}" for k in labels]
    return out
