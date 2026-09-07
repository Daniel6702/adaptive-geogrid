from __future__ import annotations

from pathlib import Path
from typing import Iterable
import warnings

import geopandas as gpd
import numpy as np
from pyproj import CRS, Transformer
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely import covers, points as shapely_points


def as_coordinate_array(points: Iterable[tuple[float, float]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("points must have shape (N, 2) and contain (longitude, latitude) pairs")
    if len(arr) == 0:
        raise ValueError("points cannot be empty")
    if not np.isfinite(arr).all():
        raise ValueError("points contain NaN or infinite values")
    return arr


def load_boundary(boundary: str | Path | BaseGeometry | gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if isinstance(boundary, gpd.GeoDataFrame):
        gdf = boundary.copy()
    elif isinstance(boundary, BaseGeometry):
        gdf = gpd.GeoDataFrame({"geometry": [boundary]}, crs="EPSG:4326")
    else:
        gdf = gpd.read_file(boundary)

    if gdf.empty:
        raise ValueError("boundary is empty")
    if gdf.crs is None:
        # RFC 7946 GeoJSON coordinates are WGS84. For non-GeoJSON inputs callers
        # should explicitly provide a CRS before passing a GeoDataFrame.
        warnings.warn("Boundary has no CRS; assuming EPSG:4326.", stacklevel=2)
        gdf = gdf.set_crs("EPSG:4326")

    gdf = gdf.to_crs("EPSG:4326")
    geom = gdf.geometry.union_all()
    if geom.is_empty:
        raise ValueError("boundary geometry is empty")
    if geom.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError(f"boundary must be Polygon or MultiPolygon, got {geom.geom_type}")
    if not geom.is_valid:
        geom = geom.buffer(0)
    return gpd.GeoDataFrame({"geometry": [geom]}, crs="EPSG:4326")


def choose_projected_crs(boundary_wgs84: gpd.GeoDataFrame, projected_crs=None) -> CRS:
    if projected_crs is not None:
        crs = CRS.from_user_input(projected_crs)
        if not crs.is_projected:
            raise ValueError("projected_crs must be a projected CRS")
        return crs

    try:
        crs = boundary_wgs84.estimate_utm_crs()
    except Exception:
        crs = None
    if crs is None:
        warnings.warn(
            "Could not estimate a local UTM CRS; falling back to EPSG:3857. "
            "For very large/multi-zone areas, pass projected_crs explicitly.",
            stacklevel=2,
        )
        crs = CRS.from_epsg(3857)
    return CRS.from_user_input(crs)


def transform_xy(arr: np.ndarray, source_crs, target_crs) -> np.ndarray:
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    x, y = transformer.transform(arr[:, 0], arr[:, 1])
    return np.column_stack([x, y]).astype(np.float64, copy=False)


def boundary_mask(boundary_geom: BaseGeometry, lonlat: np.ndarray) -> np.ndarray:
    pts = shapely_points(lonlat[:, 0], lonlat[:, 1])
    return np.asarray(covers(boundary_geom, pts), dtype=bool)
