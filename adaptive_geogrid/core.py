from __future__ import annotations

from pathlib import Path
from typing import Iterable
import warnings

import geopandas as gpd
import numpy as np

from ._power import build_power_polygons, fit_capacity_power_diagram
from ._utils import (
    as_coordinate_array,
    boundary_mask,
    choose_projected_crs,
    load_boundary,
    transform_xy,
)
from .grid import AdaptiveGeoGrid


def tessellate(
    *,
    points: Iterable[tuple[float, float]] | np.ndarray,
    boundary,
    target_points: int | None = None,
    n_tiles: int | None = None,
    projected_crs=None,
    random_state: int = 42,
    lloyd_iterations: int = 4,
    balance_tolerance: float = 0.05,
    chunk_size: int = 20_000,
    verbose: bool = False,
) -> AdaptiveGeoGrid:
    """Generate a density-adaptive, capacity-balanced geographic tessellation.

    Parameters
    ----------
    points:
        WGS84 ``(longitude, latitude)`` coordinate pairs.
    boundary:
        Polygon/MultiPolygon GeoJSON path, Shapely geometry, or GeoDataFrame.
    target_points:
        Desired approximate number of input points per fine tile. Exactly one of
        ``target_points`` and ``n_tiles`` must be supplied.
    n_tiles:
        Desired number of fine tiles.
    projected_crs:
        Optional projected CRS used for metric geometry. If omitted, a local UTM
        CRS is estimated from the boundary.
    balance_tolerance:
        Maximum desired relative deviation from each integer tile capacity.
    """
    if (target_points is None) == (n_tiles is None):
        raise ValueError("Specify exactly one of target_points or n_tiles")
    if target_points is not None and target_points <= 0:
        raise ValueError("target_points must be > 0")
    if n_tiles is not None and n_tiles <= 0:
        raise ValueError("n_tiles must be > 0")
    if not 0 <= balance_tolerance < 1:
        raise ValueError("balance_tolerance must be in [0, 1)")

    lonlat = as_coordinate_array(points)
    boundary_gdf = load_boundary(boundary)
    boundary_wgs84 = boundary_gdf.geometry.iloc[0]

    inside = boundary_mask(boundary_wgs84, lonlat)
    if not np.all(inside):
        dropped = int(np.sum(~inside))
        warnings.warn(f"Ignoring {dropped} input point(s) outside the boundary.", stacklevel=2)
        lonlat = lonlat[inside]
    if len(lonlat) == 0:
        raise ValueError("No input points fall inside the supplied boundary")

    if target_points is not None:
        k = max(1, int(round(len(lonlat) / target_points)))
    else:
        k = int(n_tiles)
    k = min(k, len(lonlat))

    crs = choose_projected_crs(boundary_gdf, projected_crs)
    boundary_projected_gdf = boundary_gdf.to_crs(crs)
    boundary_projected = boundary_projected_gdf.geometry.iloc[0]
    points_projected = transform_xy(lonlat, "EPSG:4326", crs)

    def progress(message: str):
        if verbose:
            print(message, flush=True)

    progress(f"Fitting {k} capacity-balanced tile(s) from {len(lonlat)} in-boundary points")
    sites, weights, assignments, counts, capacities, scale, origin = fit_capacity_power_diagram(
        points_projected,
        k,
        random_state=random_state,
        lloyd_iterations=lloyd_iterations,
        balance_tolerance=balance_tolerance,
        chunk_size=chunk_size,
        progress=progress,
    )

    normalized_sites = (sites - origin) / scale
    normalized_weights = weights / (scale * scale)
    progress("Constructing bounded power polygons")
    polygons = build_power_polygons(
        boundary_projected,
        sites,
        weights,
        normalized_sites=normalized_sites,
        normalized_weights=normalized_weights,
    )

    rows = []
    for tile_id, (geom, count, capacity) in enumerate(zip(polygons, counts, capacities)):
        rows.append(
            {
                "tile_id": tile_id,
                "level": 0,
                "point_count": int(count),
                "target_count": int(capacity),
                "count_error": int(count - capacity),
                "geometry": geom,
            }
        )
    fine_projected = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    fine_wgs84 = fine_projected.to_crs("EPSG:4326")

    return AdaptiveGeoGrid(
        fine_tiles_projected=fine_projected,
        fine_tiles_wgs84=fine_wgs84,
        boundary_wgs84=boundary_wgs84,
        boundary_projected=boundary_projected,
        projected_crs=crs,
        sites_projected=sites,
        weights_projected=weights,
        training_point_count=len(lonlat),
    )
