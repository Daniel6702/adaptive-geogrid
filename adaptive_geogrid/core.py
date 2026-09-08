from __future__ import annotations

from typing import Iterable
import warnings

import geopandas as gpd
import numpy as np

from ._geodesic import build_geodesic_polygons, fit_capacity_geodesic_voronoi
from ._graph import build_graph_polygons, fit_capacity_graph_partition
from ._power import build_power_polygons, fit_capacity_power_diagram
from ._utils import (
    as_coordinate_array,
    boundary_mask,
    choose_projected_crs,
    load_boundary,
    transform_xy,
)
from ._warp import SinusoidalWarp, make_sinusoidal_warp
from .grid import AdaptiveGeoGrid


_TESSELLATION_MODES = ("power", "warped", "geodesic", "graph")
_MODE_ALIASES = {
    "laguerre": "power",
    "power_laguerre": "power",
    "warped_laguerre": "warped",
    "warped_power": "warped",
    "geodesic_voronoi": "geodesic",
    "isotropic_geodesic": "geodesic",
    "balanced_geodesic": "geodesic",
    "graph_partition": "graph",
    "balanced_graph": "graph",
    "spatial_graph": "graph",
}


def _normalize_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    normalized = _MODE_ALIASES.get(normalized, normalized)
    if normalized not in _TESSELLATION_MODES:
        choices = ", ".join(repr(value) for value in _TESSELLATION_MODES)
        raise ValueError(f"Unsupported tessellation mode {mode!r}. Available modes: {choices}")
    return normalized


def _fit_power_space(
    points: np.ndarray,
    *,
    k: int,
    random_state: int,
    lloyd_iterations: int,
    balance_tolerance: float,
    chunk_size: int,
    progress,
):
    sites, weights, assignments, counts, capacities, scale, origin = fit_capacity_power_diagram(
        points,
        k,
        random_state=random_state,
        lloyd_iterations=lloyd_iterations,
        balance_tolerance=balance_tolerance,
        chunk_size=chunk_size,
        progress=progress,
    )
    normalized_sites = (sites - origin) / scale
    normalized_weights = weights / (scale * scale)
    return (
        sites,
        weights,
        assignments,
        counts,
        capacities,
        normalized_sites,
        normalized_weights,
    )


def _build_warped_polygons(
    *,
    boundary_projected,
    warp: SinusoidalWarp,
    sites: np.ndarray,
    weights: np.ndarray,
    normalized_sites: np.ndarray,
    normalized_weights: np.ndarray,
):
    # Construct the exact power partition in warped space inside a generous box.
    # We intentionally do not clip by a warped approximation of the geographic
    # boundary. Instead, inverse-warp the cells and clip them by the original
    # boundary, which preserves the exact dataset outline.
    warped_clip = warp.tessellation_clip_box(boundary_projected)
    warped_cells = build_power_polygons(
        warped_clip,
        sites,
        weights,
        normalized_sites=normalized_sites,
        normalized_weights=normalized_weights,
    )

    polygons = []
    for cell in warped_cells:
        geographic_cell = warp.inverse_geometry(cell)
        polygons.append(geographic_cell.intersection(boundary_projected))
    return polygons


def tessellate(
    *,
    points: Iterable[tuple[float, float]] | np.ndarray,
    boundary,
    target_points: int | None = None,
    n_tiles: int | None = None,
    mode: str = "power",
    projected_crs=None,
    random_state: int = 42,
    lloyd_iterations: int = 4,
    balance_tolerance: float = 0.05,
    chunk_size: int = 20_000,
    warp_strength: float = 0.12,
    warp_octaves: int = 4,
    warp_random_state: int | None = None,
    geodesic_strength: float = 0.35,
    geodesic_octaves: int = 4,
    geodesic_grid_size: int | None = None,
    geodesic_simplify: float = 1.0,
    geodesic_random_state: int | None = None,
    graph_compactness: float = 1.0,
    graph_boundary_weight: float = 0.35,
    graph_organic_strength: float = 0.15,
    graph_octaves: int = 4,
    graph_refine_iterations: int = 8,
    graph_simplify: float = 0.5,
    graph_random_state: int | None = None,
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
    mode:
        Tessellation backend. ``"power"`` is the original capacity-constrained
        power/Laguerre tessellation. ``"warped"`` first applies a smooth,
        invertible multi-scale spatial warp, fits the same capacity-constrained
        Laguerre tessellation in warped space, and maps the cells back to produce
        curved geographic borders. ``"geodesic"`` uses an isotropic scalar cost
        field and capacity-balanced weighted geodesic Voronoi competition on an
        adaptive raster graph. Its Lloyd step remains in ordinary projected
        coordinates to favor geographically compact point groups. ``"graph"``
        builds a Delaunay adjacency graph over the input points, then directly
        refines a connected balanced partition for geographic compactness and
        boundary regularity. Final graph cells are unions of the points' ordinary
        Voronoi microcells, so this mode introduces no raster discretization.
    projected_crs:
        Optional projected CRS used for metric geometry. If omitted, a local UTM
        CRS is estimated from the boundary.
    balance_tolerance:
        Maximum desired relative deviation from each integer tile capacity.
    warp_strength:
        Organic deformation strength for ``mode="warped"``. ``0`` gives an
        identity-like warp; values around ``0.08``-``0.18`` are useful starting
        points. Larger values intentionally produce stronger distortion.
    warp_octaves:
        Number of spatial scales used by the warped mode.
    warp_random_state:
        Optional independent seed controlling only the warp geometry. If omitted,
        ``random_state`` is reused.
    geodesic_strength:
        Scalar cost-field strength for ``mode="geodesic"``. The local metric is
        isotropic; the cost multiplier is bounded to approximately
        ``exp(-strength) .. exp(+strength)``.
    geodesic_octaves:
        Number of spatial scales in the geodesic scalar cost field.
    geodesic_grid_size:
        Number of raster cells across the longest boundary dimension. ``None``
        chooses a value automatically from the requested number of tiles.
    geodesic_simplify:
        Topology-preserving vector simplification tolerance measured in raster
        cell widths. ``0`` keeps exact raster-cell edges; around ``1`` removes
        most staircase artifacts while retaining the shared-edge coverage.
    geodesic_random_state:
        Optional independent seed controlling only the geodesic cost field.
    graph_compactness:
        Weight of within-tile geographic point dispersion for ``mode="graph"``.
    graph_boundary_weight:
        Weight of graph-cut boundary regularity. Short Delaunay edges are more
        expensive to cut, explicitly preferring nearby points in the same tile.
    graph_organic_strength:
        Strength of a weak smooth multiscale boundary-cost field. This may bend
        boundaries organically but remains subordinate to compactness/capacity.
    graph_octaves:
        Number of spatial scales in the graph boundary-cost field.
    graph_refine_iterations:
        Number of connectivity-preserving boundary-swap refinement passes.
    graph_simplify:
        Topology-preserving final coverage simplification as a fraction of the
        median Delaunay edge length. ``0`` retains exact point-Voronoi microcells.
    graph_random_state:
        Optional independent seed controlling graph boundary refinement/field.
    """
    mode = _normalize_mode(mode)

    if (target_points is None) == (n_tiles is None):
        raise ValueError("Specify exactly one of target_points or n_tiles")
    if target_points is not None and target_points <= 0:
        raise ValueError("target_points must be > 0")
    if n_tiles is not None and n_tiles <= 0:
        raise ValueError("n_tiles must be > 0")
    if not 0 <= balance_tolerance < 1:
        raise ValueError("balance_tolerance must be in [0, 1)")
    if warp_strength < 0:
        raise ValueError("warp_strength must be >= 0")
    if warp_octaves < 1:
        raise ValueError("warp_octaves must be >= 1")
    if geodesic_strength < 0:
        raise ValueError("geodesic_strength must be >= 0")
    if geodesic_octaves < 1:
        raise ValueError("geodesic_octaves must be >= 1")
    if geodesic_grid_size is not None and geodesic_grid_size < 32:
        raise ValueError("geodesic_grid_size must be >= 32")
    if geodesic_simplify < 0:
        raise ValueError("geodesic_simplify must be >= 0")
    if graph_compactness < 0:
        raise ValueError("graph_compactness must be >= 0")
    if graph_boundary_weight < 0:
        raise ValueError("graph_boundary_weight must be >= 0")
    if graph_organic_strength < 0:
        raise ValueError("graph_organic_strength must be >= 0")
    if graph_octaves < 1:
        raise ValueError("graph_octaves must be >= 1")
    if graph_refine_iterations < 0:
        raise ValueError("graph_refine_iterations must be >= 0")
    if graph_simplify < 0:
        raise ValueError("graph_simplify must be >= 0")

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

    progress(
        f"Fitting {k} capacity-balanced tile(s) from {len(lonlat)} in-boundary points "
        f"using mode={mode!r}"
    )

    warp: SinusoidalWarp | None = None
    geodesic_metric = None
    graph_state = None

    if mode == "graph":
        progress(
            "Fitting spatially constrained balanced graph partition with "
            f"compactness={graph_compactness:g}, boundary_weight={graph_boundary_weight:g}"
        )
        (
            sites,
            weights,
            assignments,
            counts,
            capacities,
            graph_state,
        ) = fit_capacity_graph_partition(
            points_projected,
            boundary_projected,
            k,
            random_state=random_state,
            lloyd_iterations=lloyd_iterations,
            balance_tolerance=balance_tolerance,
            chunk_size=chunk_size,
            compactness_weight=float(graph_compactness),
            boundary_weight=float(graph_boundary_weight),
            organic_strength=float(graph_organic_strength),
            organic_octaves=int(graph_octaves),
            refine_iterations=int(graph_refine_iterations),
            graph_random_state=(
                random_state if graph_random_state is None else int(graph_random_state)
            ),
            progress=progress,
        )
        progress("Constructing graph partitions from point-owned Voronoi microcells")
        polygons = build_graph_polygons(
            graph_state,
            boundary_projected,
            k,
            simplify=float(graph_simplify),
        )
    elif mode == "geodesic":
        progress(
            "Fitting capacity-balanced isotropic geodesic Voronoi cells with "
            f"strength={geodesic_strength:g}, octaves={geodesic_octaves}"
        )
        (
            sites,
            weights,
            assignments,
            counts,
            capacities,
            geodesic_metric,
        ) = fit_capacity_geodesic_voronoi(
            points_projected,
            boundary_projected,
            k,
            random_state=random_state,
            lloyd_iterations=lloyd_iterations,
            balance_tolerance=balance_tolerance,
            grid_size=geodesic_grid_size,
            strength=float(geodesic_strength),
            octaves=int(geodesic_octaves),
            field_random_state=(
                random_state if geodesic_random_state is None else int(geodesic_random_state)
            ),
            progress=progress,
        )
        progress("Polygonizing geodesic Voronoi labels")
        polygons = build_geodesic_polygons(
            geodesic_metric,
            boundary_projected,
            k,
            simplify_cells=float(geodesic_simplify),
        )
    else:
        classifier_points = points_projected
        if mode == "warped":
            warp = make_sinusoidal_warp(
                boundary_projected,
                strength=float(warp_strength),
                octaves=int(warp_octaves),
                random_state=random_state if warp_random_state is None else int(warp_random_state),
            )
            classifier_points = warp.forward(points_projected)
            progress(
                "Warped projected space with "
                f"strength={warp_strength:g}, octaves={warp_octaves}"
            )

        (
            sites,
            weights,
            assignments,
            counts,
            capacities,
            normalized_sites,
            normalized_weights,
        ) = _fit_power_space(
            classifier_points,
            k=k,
            random_state=random_state,
            lloyd_iterations=lloyd_iterations,
            balance_tolerance=balance_tolerance,
            chunk_size=chunk_size,
            progress=progress,
        )

        if mode == "power":
            progress("Constructing bounded power polygons")
            polygons = build_power_polygons(
                boundary_projected,
                sites,
                weights,
                normalized_sites=normalized_sites,
                normalized_weights=normalized_weights,
            )
        else:
            progress("Constructing power polygons in warped space and mapping them back")
            polygons = _build_warped_polygons(
                boundary_projected=boundary_projected,
                warp=warp,
                sites=sites,
                weights=weights,
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
        tessellation_mode=mode,
        warp=warp,
        geodesic_metric=geodesic_metric,
        graph_state=graph_state,
    )
