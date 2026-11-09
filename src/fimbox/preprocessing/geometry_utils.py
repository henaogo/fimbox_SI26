"""
Shared defensive geometry-repair helper.

Real coastline/stream-network data (high vertex density, complex detail) can
produce geometry that reports ``.is_valid == True`` but is still numerically
fragile enough to throw ``TopologyException: side location conflict`` (or
similar GEOS robustness errors) the next time ANYTHING calls a set operation
on it -- union_all(), unary_union(), dissolve(), even set_precision() itself
on sufficiently messy input. This has surfaced independently in at least two
places (AOI boundary construction in preprocess_area.py, and level-path
dissolving in branch_derivation.py) with the same underlying cause, so it's a
shared helper rather than a one-off fix in either place.
"""
from __future__ import annotations

import logging

import geopandas as gpd
import shapely

log = logging.getLogger(__name__)

# Grid sizes to try, coarsest-last. Units match the input CRS (assumed to be
# a projected, meters-based CRS such as EPSG:5070 wherever this is called).
_PRECISION_GRID_SIZES_M = (0.01, 0.1, 1.0)


def repair_geometry(geoseries: gpd.GeoSeries) -> gpd.GeoSeries:
    """
    Best-effort repair for a GeoSeries that may be numerically fragile.
    Tries set_precision() at increasingly coarse grid sizes (each also
    validity-repairs via mode="valid_output"), then make_valid(), then
    buffer(0). Never raises -- worst case returns the input unchanged and
    lets the caller's own error handling deal with whatever happens next,
    same as before this helper existed.
    """
    values = geoseries.values

    for grid_size in _PRECISION_GRID_SIZES_M:
        try:
            return gpd.GeoSeries(
                shapely.set_precision(values, grid_size=grid_size, mode="valid_output"),
                index=geoseries.index,
                crs=geoseries.crs,
            )
        except Exception as exc:
            log.debug("repair_geometry: set_precision(grid_size=%s) failed: %s",
                     grid_size, exc)

    try:
        return geoseries.make_valid()
    except Exception as exc:
        log.debug("repair_geometry: make_valid() failed: %s", exc)

    try:
        return gpd.GeoSeries(
            shapely.buffer(values, 0), index=geoseries.index, crs=geoseries.crs
        )
    except Exception as exc:
        log.warning("repair_geometry: all repair attempts failed (%s); "
                    "returning original geometry unchanged", exc)
        return geoseries
