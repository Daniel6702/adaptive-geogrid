from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable
import warnings

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree
from shapely import contains_xy, union_all
from shapely.geometry import Polygon, shape
from shapely.geometry.base import BaseGeometry
from sklearn.cluster import KMeans


@dataclass
class GeodesicMetric:
    """Raster approximation of an isotropic Riemannian/geodesic metric.

    ``cost`` is a positive scalar field. Moving by an infinitesimal Euclidean
    distance ``ds`` around location ``x`` costs ``cost(x) * ds`` regardless of
    direction, so the metric is locally isotropic. The raster graph uses 8-way
    connectivity and trapezoidal edge integration to approximate shortest-path
    geodesic distance.

    ``labels`` stores the final weighted geodesic Voronoi class of every raster
    pixel. Pixels whose centers fall outside the boundary are filled from the
    nearest in-boundary graph node; polygons and classification are subsequently
    clipped/masked by the exact geographic boundary.
    """

    minx: float
    maxy: float
    dx: float
    dy: float
    rows: int
    cols: int
    valid_mask: np.ndarray
    cost: np.ndarray
    labels: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.rows), int(self.cols)

    @property
    def max_cell_size(self) -> float:
        return max(float(self.dx), float(self.dy))

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        xs = self.minx + (np.arange(self.cols, dtype=np.float64) + 0.5) * self.dx
        ys = self.maxy - (np.arange(self.rows, dtype=np.float64) + 0.5) * self.dy
        return xs, ys

    def projected_to_rc(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")
        col = np.floor((arr[:, 0] - self.minx) / self.dx).astype(np.int64)
        row = np.floor((self.maxy - arr[:, 1]) / self.dy).astype(np.int64)
        np.clip(col, 0, self.cols - 1, out=col)
        np.clip(row, 0, self.rows - 1, out=row)
        return row, col

    def classify_projected(self, points: np.ndarray) -> np.ndarray:
        if self.labels is None:
            raise ValueError("Geodesic metric does not contain fitted labels")
        row, col = self.projected_to_rc(points)
        return np.asarray(self.labels[row, col], dtype=np.int64)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "type": "isotropic_raster_geodesic",
            "minx": float(self.minx),
            "maxy": float(self.maxy),
            "dx": float(self.dx),
            "dy": float(self.dy),
            "rows": int(self.rows),
            "cols": int(self.cols),
        }

    @classmethod
    def from_metadata(
        cls,
        data: dict[str, Any],
        *,
        valid_mask: np.ndarray,
        cost: np.ndarray,
        labels: np.ndarray,
    ) -> "GeodesicMetric":
        if data.get("type") != "isotropic_raster_geodesic":
            raise ValueError(f"Unsupported geodesic metric type: {data.get('type')!r}")
        rows = int(data["rows"])
        cols = int(data["cols"])
        expected = (rows, cols)
        if valid_mask.shape != expected or cost.shape != expected or labels.shape != expected:
            raise ValueError("Serialized geodesic raster arrays do not match metadata shape")
        return cls(
            minx=float(data["minx"]),
            maxy=float(data["maxy"]),
            dx=float(data["dx"]),
            dy=float(data["dy"]),
            rows=rows,
            cols=cols,
            valid_mask=np.asarray(valid_mask, dtype=bool),
            cost=np.asarray(cost, dtype=np.float64),
            labels=np.asarray(labels, dtype=np.int32),
        )


@dataclass(frozen=True)
class _GraphData:
    rows: np.ndarray
    cols: np.ndarray
    values: np.ndarray
    id_map: np.ndarray
    node_rows: np.ndarray
    node_cols: np.ndarray
    node_xy: np.ndarray

    @property
    def n_nodes(self) -> int:
        return int(len(self.node_rows))


def _auto_grid_size(k: int) -> int:
    # Maintain enough raster support per expected cell that curvilinear borders
    # are visible rather than stair-stepped, without allowing very large K to
    # explode quadratically without bound.
    return int(np.clip(math.ceil(16.0 * math.sqrt(max(k, 1))), 384, 1024))


def make_geodesic_metric(
    boundary: BaseGeometry,
    *,
    k: int,
    grid_size: int | None = None,
    strength: float = 0.35,
    octaves: int = 4,
    random_state: int = 42,
) -> GeodesicMetric:
    """Create a deterministic positive scalar cost field over ``boundary``.

    The field is a normalized sum of broad sinusoidal waves at several spatial
    scales. Unlike the warped backend this never prefers one local direction over
    another: at a given point the same scalar multiplier applies to every travel
    direction. ``strength`` controls the maximum cost contrast; 0 reduces the
    metric to ordinary Euclidean distance on the raster graph.
    """
    if strength < 0:
        raise ValueError("geodesic_strength must be >= 0")
    if octaves < 1:
        raise ValueError("geodesic_octaves must be >= 1")
    if grid_size is not None and grid_size < 32:
        raise ValueError("geodesic_grid_size must be >= 32")

    minx, miny, maxx, maxy = map(float, boundary.bounds)
    width = max(maxx - minx, 1.0)
    height = max(maxy - miny, 1.0)
    longest = max(width, height)
    longest_cells = _auto_grid_size(k) if grid_size is None else int(grid_size)
    dx = longest / longest_cells
    dy = dx
    cols = max(2, int(math.ceil(width / dx)))
    rows = max(2, int(math.ceil(height / dy)))
    # Recompute spacing so the raster bounds end exactly at the source bbox.
    dx = width / cols
    dy = height / rows

    xs = minx + (np.arange(cols, dtype=np.float64) + 0.5) * dx
    ys = maxy - (np.arange(rows, dtype=np.float64) + 0.5) * dy
    xx, yy = np.meshgrid(xs, ys)
    valid = contains_xy(boundary, xx, yy)
    if not np.any(valid):
        raise ValueError("Geodesic raster contains no cell centers inside the boundary")

    # Normalized coordinates are dimensionless, allowing the same frequency
    # settings to behave similarly for a city and for a much larger region.
    xn = (xx - (minx + maxx) * 0.5) / longest
    yn = (yy - (miny + maxy) * 0.5) / longest
    rng = np.random.default_rng(random_state)
    field = np.zeros((rows, cols), dtype=np.float64)
    amplitude_sum = 0.0

    for octave in range(octaves):
        wavelength = 0.70 / (2**octave)
        amplitude = 0.78**octave
        # Two differently oriented components per octave remove obvious stripes.
        for _ in range(2):
            angle = float(rng.uniform(0.0, 2.0 * math.pi))
            phase = float(rng.uniform(0.0, 2.0 * math.pi))
            direction = math.cos(angle) * xn + math.sin(angle) * yn
            field += amplitude * np.sin(2.0 * math.pi * direction / wavelength + phase)
            amplitude_sum += amplitude

    if amplitude_sum > 0:
        field /= amplitude_sum
    max_abs = float(np.max(np.abs(field[valid])))
    if max_abs > 1e-12:
        field /= max_abs

    # cost in [exp(-strength), exp(+strength)] -> global ratio exp(2*strength).
    # This intentionally bounds how strongly the organic field can fight the
    # ordinary geographic compactness induced by centroidal site relaxation.
    cost = np.exp(float(strength) * field)
    cost[~valid] = np.nan

    return GeodesicMetric(
        minx=minx,
        maxy=maxy,
        dx=float(dx),
        dy=float(dy),
        rows=rows,
        cols=cols,
        valid_mask=np.asarray(valid, dtype=bool),
        cost=cost,
    )


def _build_graph(metric: GeodesicMetric) -> _GraphData:
    valid = metric.valid_mask
    node_rows, node_cols = np.nonzero(valid)
    n = len(node_rows)
    id_map = np.full(valid.shape, -1, dtype=np.int64)
    id_map[node_rows, node_cols] = np.arange(n, dtype=np.int64)

    xs = metric.minx + (node_cols.astype(np.float64) + 0.5) * metric.dx
    ys = metric.maxy - (node_rows.astype(np.float64) + 0.5) * metric.dy
    node_xy = np.column_stack([xs, ys])

    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []

    # Only create each undirected edge once here; add both directions below.
    for dr, dc, length in (
        (0, 1, metric.dx),
        (1, 0, metric.dy),
        (1, 1, math.hypot(metric.dx, metric.dy)),
        (1, -1, math.hypot(metric.dx, metric.dy)),
    ):
        r0_start = max(0, -dr)
        r0_stop = min(metric.rows, metric.rows - dr)
        c0_start = max(0, -dc)
        c0_stop = min(metric.cols, metric.cols - dc)
        r0, c0 = np.nonzero(
            valid[r0_start:r0_stop, c0_start:c0_stop]
            & valid[r0_start + dr:r0_stop + dr, c0_start + dc:c0_stop + dc]
        )
        r0 = r0 + r0_start
        c0 = c0 + c0_start
        r1 = r0 + dr
        c1 = c0 + dc
        a = id_map[r0, c0]
        b = id_map[r1, c1]
        edge_cost = float(length) * 0.5 * (metric.cost[r0, c0] + metric.cost[r1, c1])

        row_parts.extend([a, b])
        col_parts.extend([b, a])
        value_parts.extend([edge_cost, edge_cost])

    rows = (
        np.concatenate(row_parts).astype(np.int64, copy=False)
        if row_parts
        else np.empty(0, dtype=np.int64)
    )
    cols = (
        np.concatenate(col_parts).astype(np.int64, copy=False)
        if col_parts
        else np.empty(0, dtype=np.int64)
    )
    values = (
        np.concatenate(value_parts).astype(np.float64, copy=False)
        if value_parts
        else np.empty(0, dtype=np.float64)
    )

    # The center-sampled raster of a MultiPolygon, narrow isthmus, or thin piece
    # of boundary can contain several disconnected graph components. Geodesic
    # competition still needs one globally defined metric even when there are
    # fewer sites than raster components. Join components with the shortest
    # projected-space virtual bridges. These bridges affect only distance
    # propagation; polygonization is still clipped to the exact source boundary.
    if n > 1:
        matrix = coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()
        n_components, component_id = connected_components(
            matrix, directed=False, return_labels=True
        )
        if n_components > 1:
            component_nodes = [
                np.flatnonzero(component_id == component)
                for component in range(n_components)
            ]
            base_component = int(np.argmax([len(nodes) for nodes in component_nodes]))
            connected_nodes = component_nodes[base_component].copy()
            remaining = {
                component for component in range(n_components) if component != base_component
            }
            bridge_rows: list[int] = []
            bridge_cols: list[int] = []
            bridge_values: list[float] = []

            while remaining:
                tree = cKDTree(node_xy[connected_nodes])
                best = None
                for component in remaining:
                    nodes = component_nodes[component]
                    distances, nearest = tree.query(node_xy[nodes], k=1)
                    local = int(np.argmin(distances))
                    candidate = (
                        float(distances[local]),
                        int(connected_nodes[int(nearest[local])]),
                        int(nodes[local]),
                        int(component),
                    )
                    if best is None or candidate[0] < best[0]:
                        best = candidate

                assert best is not None
                distance, a, b, component = best
                ar, ac = int(node_rows[a]), int(node_cols[a])
                br, bc = int(node_rows[b]), int(node_cols[b])
                local_cost = 0.5 * (float(metric.cost[ar, ac]) + float(metric.cost[br, bc]))
                bridge_cost = max(distance * local_cost, 1e-9)
                bridge_rows.extend([a, b])
                bridge_cols.extend([b, a])
                bridge_values.extend([bridge_cost, bridge_cost])
                connected_nodes = np.concatenate([connected_nodes, component_nodes[component]])
                remaining.remove(component)

            rows = np.concatenate([rows, np.asarray(bridge_rows, dtype=np.int64)])
            cols = np.concatenate([cols, np.asarray(bridge_cols, dtype=np.int64)])
            values = np.concatenate([values, np.asarray(bridge_values, dtype=np.float64)])

    return _GraphData(
        rows=rows,
        cols=cols,
        values=values,
        id_map=id_map,
        node_rows=node_rows.astype(np.int64, copy=False),
        node_cols=node_cols.astype(np.int64, copy=False),
        node_xy=node_xy,
    )


def _snap_unique_sites(sites: np.ndarray, graph: _GraphData) -> tuple[np.ndarray, np.ndarray]:
    k = len(sites)
    if k > graph.n_nodes:
        raise ValueError(
            "Geodesic raster is too coarse for the requested number of tiles. "
            "Increase geodesic_grid_size."
        )
    tree = cKDTree(graph.node_xy)
    query_k = min(max(8, min(32, k + 2)), graph.n_nodes)
    distances, candidates = tree.query(sites, k=query_k)
    if query_k == 1:
        candidates = candidates[:, None]
        distances = distances[:, None]

    chosen = np.full(k, -1, dtype=np.int64)
    used: set[int] = set()
    order = np.argsort(distances[:, 0])
    for i in order:
        for candidate in np.atleast_1d(candidates[i]):
            node = int(candidate)
            if node not in used:
                chosen[i] = node
                used.add(node)
                break
        if chosen[i] < 0:
            # Rare collision fallback. Query all nodes in distance order only for
            # this site rather than allocating a huge K x N matrix.
            _, all_candidates = tree.query(sites[i], k=graph.n_nodes)
            for candidate in np.atleast_1d(all_candidates):
                node = int(candidate)
                if node not in used:
                    chosen[i] = node
                    used.add(node)
                    break
    return chosen, graph.node_xy[chosen].copy()


def _weighted_multisource_labels(
    graph: _GraphData,
    seed_nodes: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one weighted multi-source geodesic Voronoi solve.

    A synthetic super-source is connected to site ``i`` with cost ``C-w_i``.
    Adding the same constant C to all competitors changes no argmin, therefore
    shortest path from the super-source is exactly ``argmin_i d_geo(i,x)-w_i``.
    """
    n = graph.n_nodes
    k = len(seed_nodes)
    super_node = n
    max_weight = float(np.max(weights)) if k else 0.0
    epsilon = max(float(np.median(graph.values)) * 1e-9, 1e-9)
    offsets = max_weight - np.asarray(weights, dtype=np.float64) + epsilon

    rows = np.concatenate(
        [graph.rows, np.full(k, super_node, dtype=np.int64), seed_nodes.astype(np.int64)]
    )
    cols = np.concatenate(
        [graph.cols, seed_nodes.astype(np.int64), np.full(k, super_node, dtype=np.int64)]
    )
    values = np.concatenate([graph.values, offsets, offsets])
    matrix = coo_matrix((values, (rows, cols)), shape=(n + 1, n + 1)).tocsr()

    distances, predecessors = dijkstra(
        matrix,
        directed=False,
        indices=super_node,
        return_predecessors=True,
    )
    distances = np.asarray(distances[:-1], dtype=np.float64)
    predecessors = np.asarray(predecessors[:-1], dtype=np.int64)

    seed_to_tile = {int(node): int(tile) for tile, node in enumerate(seed_nodes)}
    labels = np.full(n, -1, dtype=np.int32)

    # Predecessor distances strictly decrease on positive graph edges, so sorting
    # by distance ensures a predecessor's class has already been propagated.
    order = np.argsort(distances, kind="stable")
    for node in order:
        if not np.isfinite(distances[node]):
            continue
        pred = int(predecessors[node])
        if pred == super_node:
            labels[node] = seed_to_tile.get(int(node), -1)
        elif 0 <= pred < n:
            labels[node] = labels[pred]

    # _build_graph() explicitly bridges disconnected raster components, so every
    # graph node must be reachable from the super-source. Reaching this branch is
    # an implementation error rather than a condition to hide with a different
    # Euclidean classifier.
    if np.any(labels < 0):
        raise RuntimeError(
            "Geodesic graph contains unreachable nodes after component bridging"
        )

    return labels, distances


def _point_nodes(points: np.ndarray, metric: GeodesicMetric, graph: _GraphData) -> np.ndarray:
    row, col = metric.projected_to_rc(points)
    nodes = graph.id_map[row, col].copy()
    missing = nodes < 0
    if np.any(missing):
        # Map boundary-edge pixels whose center lies just outside to the nearest
        # valid graph pixel. This same extrapolation is used for the final label
        # raster, so training and later classification stay consistent.
        _, nearest = ndimage.distance_transform_edt(
            ~metric.valid_mask,
            return_distances=True,
            return_indices=True,
        )
        rr = nearest[0, row[missing], col[missing]]
        cc = nearest[1, row[missing], col[missing]]
        nodes[missing] = graph.id_map[rr, cc]
    return nodes.astype(np.int64, copy=False)


def _refine_weights(
    *,
    graph: _GraphData,
    seed_nodes: np.ndarray,
    point_nodes: np.ndarray,
    capacities: np.ndarray,
    initial_weights: np.ndarray,
    tolerance: float,
    max_iter: int = 180,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k = len(seed_nodes)
    weights = np.asarray(initial_weights, dtype=np.float64).copy()
    target_scale = np.maximum(capacities.astype(np.float64), 1.0)

    if k > 1:
        site_xy = graph.node_xy[seed_nodes]
        tree = cKDTree(site_xy)
        nearest = tree.query(site_xy, k=2)[0][:, 1]
        characteristic = float(np.median(nearest))
        step = max(0.12 * characteristic, 0.25 * np.median(graph.values))
    else:
        step = max(float(np.median(graph.values)), 1.0)

    best_error = math.inf
    best_weights = weights.copy()
    best_labels = None
    best_assignments = None
    best_counts = None

    for _ in range(max_iter):
        labels, _ = _weighted_multisource_labels(graph, seed_nodes, weights)
        assignments = labels[point_nodes].astype(np.int64, copy=False)
        counts = np.bincount(assignments, minlength=k)
        rel_error = (capacities - counts) / target_scale
        max_error = float(np.max(np.abs(rel_error)))

        if max_error < best_error:
            best_error = max_error
            best_weights = weights.copy()
            best_labels = labels.copy()
            best_assignments = assignments.copy()
            best_counts = counts.copy()

        if max_error <= tolerance:
            return weights, labels, assignments, counts

        # Under-full cells receive larger additive weight, lowering d_geo - w.
        weights += step * rel_error
        weights -= np.mean(weights)
        step *= 0.988

    assert best_labels is not None and best_assignments is not None and best_counts is not None
    return best_weights, best_labels, best_assignments, best_counts


def _fill_label_raster(metric: GeodesicMetric, graph: _GraphData, node_labels: np.ndarray) -> np.ndarray:
    raster = np.full(metric.shape, -1, dtype=np.int32)
    raster[graph.node_rows, graph.node_cols] = node_labels.astype(np.int32, copy=False)
    outside = raster < 0
    if np.any(outside):
        _, nearest = ndimage.distance_transform_edt(
            ~metric.valid_mask,
            return_distances=True,
            return_indices=True,
        )
        raster[outside] = raster[nearest[0, outside], nearest[1, outside]]
    return raster


def fit_capacity_geodesic_voronoi(
    points_m: np.ndarray,
    boundary: BaseGeometry,
    k: int,
    *,
    random_state: int = 42,
    lloyd_iterations: int = 4,
    balance_tolerance: float = 0.05,
    grid_size: int | None = None,
    strength: float = 0.35,
    octaves: int = 4,
    field_random_state: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    GeodesicMetric,
]:
    """Fit a capacity-balanced centroidal geodesic Voronoi tessellation.

    Capacity is enforced by additive geodesic weights. Site movement is performed
    toward the ordinary projected-coordinate centroid of each tile's assigned
    points. This retains the organic geodesic boundary prior while explicitly
    pulling sites toward geographically compact point groups.
    """
    points_m = np.asarray(points_m, dtype=np.float64)
    n = len(points_m)
    if not 1 <= k <= n:
        raise ValueError(f"n_tiles must be between 1 and number of in-boundary points ({n})")

    capacities = np.full(k, n // k, dtype=np.int64)
    capacities[: n % k] += 1

    metric = make_geodesic_metric(
        boundary,
        k=k,
        grid_size=grid_size,
        strength=strength,
        octaves=octaves,
        random_state=random_state if field_random_state is None else int(field_random_state),
    )
    graph = _build_graph(metric)
    point_nodes = _point_nodes(points_m, metric, graph)

    if progress:
        progress(
            f"Geodesic raster: {metric.cols}x{metric.rows}, "
            f"{graph.n_nodes} in-boundary graph nodes"
        )

    if k == 1:
        labels = np.zeros(graph.n_nodes, dtype=np.int32)
        assignments = np.zeros(n, dtype=np.int64)
        counts = np.array([n], dtype=np.int64)
        metric.labels = _fill_label_raster(metric, graph, labels)
        site = np.mean(points_m, axis=0, keepdims=True)
        return site, np.zeros(1), assignments, counts, capacities, metric

    km = KMeans(n_clusters=k, init="k-means++", n_init=1, random_state=random_state, max_iter=100)
    km.fit(points_m)
    sites = km.cluster_centers_.astype(np.float64)
    weights = np.zeros(k, dtype=np.float64)

    assignments = np.zeros(n, dtype=np.int64)
    counts = np.zeros(k, dtype=np.int64)
    node_labels = np.zeros(graph.n_nodes, dtype=np.int32)

    for outer in range(max(1, lloyd_iterations)):
        if progress:
            progress(
                f"Balancing geodesic cells: Lloyd iteration {outer + 1}/{max(1, lloyd_iterations)}"
            )

        seed_nodes, snapped_sites = _snap_unique_sites(sites, graph)
        weights, node_labels, assignments, counts = _refine_weights(
            graph=graph,
            seed_nodes=seed_nodes,
            point_nodes=point_nodes,
            capacities=capacities,
            initial_weights=weights,
            tolerance=balance_tolerance,
        )

        # Geographic compactness term: unlike warped mode, the centroidal update
        # is measured in the original projected CRS, not in a transformed metric.
        new_sites = snapped_sites.copy()
        for i in range(k):
            members = points_m[assignments == i]
            if len(members):
                new_sites[i] = members.mean(axis=0)
        move = float(np.max(np.linalg.norm(new_sites - sites, axis=1)))
        sites = new_sites
        if move < 0.15 * metric.max_cell_size:
            break

    # One final balance after the final centroid movement.
    seed_nodes, sites = _snap_unique_sites(sites, graph)
    weights, node_labels, assignments, counts = _refine_weights(
        graph=graph,
        seed_nodes=seed_nodes,
        point_nodes=point_nodes,
        capacities=capacities,
        initial_weights=weights,
        tolerance=balance_tolerance,
    )

    max_rel_error = float(np.max(np.abs(counts - capacities) / np.maximum(capacities, 1)))
    if max_rel_error > max(balance_tolerance * 2, 0.10):
        warnings.warn(
            f"Geodesic capacity optimizer stopped with maximum relative tile-count error "
            f"{max_rel_error:.1%}. Consider increasing geodesic_grid_size, using more "
            "points per tile, or loosening balance_tolerance.",
            stacklevel=2,
        )

    metric.labels = _fill_label_raster(metric, graph, node_labels)
    return sites, weights, assignments, counts, capacities, metric


def build_geodesic_polygons(
    metric: GeodesicMetric,
    boundary: BaseGeometry,
    k: int,
    *,
    simplify_cells: float = 1.0,
) -> list[BaseGeometry]:
    """Polygonize the final geodesic label raster and clip to ``boundary``.

    A small topology-preserving coverage simplification removes the visual
    staircase caused by rasterization while keeping all shared edges matched.
    ``simplify_cells`` is the tolerance in raster-cell widths; set it to 0 to
    retain the exact raster-cell geometry. Classification remains defined by the
    stored geodesic label raster.
    """
    if simplify_cells < 0:
        raise ValueError("geodesic_simplify must be >= 0")
    if metric.labels is None:
        raise ValueError("Geodesic metric does not contain fitted labels")

    try:
        from rasterio.features import shapes
        from rasterio.transform import from_origin
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise ImportError(
            "The geodesic tessellation backend requires rasterio>=1.4."
        ) from exc

    transform = from_origin(metric.minx, metric.maxy, metric.dx, metric.dy)
    parts: list[list[BaseGeometry]] = [[] for _ in range(k)]
    raster = np.asarray(metric.labels, dtype=np.int32)

    for mapping, value in shapes(raster, transform=transform):
        tile_id = int(value)
        if 0 <= tile_id < k:
            parts[tile_id].append(shape(mapping))

    polygons: list[BaseGeometry] = []
    for tile_id in range(k):
        geom = union_all(parts[tile_id]) if parts[tile_id] else Polygon()
        # Exact source boundary is authoritative; rasterization only determines
        # the internal geodesic competition.
        geom = geom.intersection(boundary)
        polygons.append(geom)

    if simplify_cells > 0 and len(polygons) > 1:
        try:
            from shapely import coverage_simplify

            tolerance = float(simplify_cells) * metric.max_cell_size
            simplified = coverage_simplify(
                np.asarray(polygons, dtype=object),
                tolerance,
                simplify_boundary=False,
            )
            polygons = list(simplified)
        except (ValueError, TypeError):
            # If a pathological boundary prevents Shapely from recognizing the
            # clipped cells as a coverage, retain the exact raster polygons.
            pass

    return polygons
