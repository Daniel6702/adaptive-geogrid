from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely import set_precision, union_all


def _geometry_tolerance(gdf: gpd.GeoDataFrame) -> float:
    bounds = gdf.total_bounds
    scale = max(math.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1]), 1.0)
    return max(scale * 1e-8, 1e-6)


def _build_adjacency_graph(gdf: gpd.GeoDataFrame) -> nx.Graph:
    """Build a polygon region-adjacency graph.

    Two cells are adjacent only when they share a non-trivial boundary segment.
    A very small tolerance is used because independently clipped power cells can
    differ by floating-point noise along an otherwise identical edge.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(len(gdf)))
    geoms = list(gdf.geometry)
    sindex = gdf.sindex
    tol = _geometry_tolerance(gdf)

    for i, geom in enumerate(geoms):
        search_geom = geom.buffer(tol)
        for raw_j in sindex.query(search_geom, predicate="intersects"):
            j = int(raw_j)
            if j <= i:
                continue
            other = geoms[j]
            if geom.distance(other) > tol:
                continue

            exact_shared = float(geom.boundary.intersection(other.boundary).length)
            if exact_shared > tol * 10:
                shared = exact_shared
            else:
                # Near-coincident edges form a long thin overlap after buffering;
                # a corner contact forms only a tiny disk-like overlap.
                overlap_area = float(
                    geom.boundary.buffer(tol).intersection(other.boundary.buffer(tol)).area
                )
                shared = overlap_area / max(2.0 * tol, 1e-12)

            if shared > tol * 100:
                graph.add_edge(i, j, shared=shared)

    return graph


def _allocate_component_groups(
    graph: nx.Graph,
    point_counts: np.ndarray,
    target_groups: int,
) -> list[tuple[list[int], int]]:
    """Allocate the requested number of parents over disconnected components."""
    components = [sorted(c) for c in nx.connected_components(graph)]
    allocations = [1] * len(components)

    while sum(allocations) < target_groups:
        candidates = []
        for idx, nodes in enumerate(components):
            if allocations[idx] >= len(nodes):
                continue
            load = float(point_counts[nodes].sum())
            candidates.append((load / allocations[idx], idx))
        if not candidates:
            break
        _, idx = max(candidates)
        allocations[idx] += 1

    return list(zip(components, allocations))


@dataclass
class _Region:
    members: set[int]
    point_count: float
    child_count: int
    centroid: np.ndarray
    area: float
    perimeter: float


def _merge_score(
    a: _Region,
    b: _Region,
    *,
    shared: float,
    target_load: float,
    target_children: float,
    diag: float,
) -> float:
    """Cost for one adjacent agglomerative merge.

    The hard structural rule is adjacency. The score then favors compact regions
    with a strong common border while strongly discouraging parents that grow far
    beyond the desired point load or desired number of immediate children.
    """
    combined_load = a.point_count + b.point_count
    combined_children = a.child_count + b.child_count

    load_ratio = combined_load / max(target_load, 1.0)
    child_ratio = combined_children / max(target_children, 1.0)

    # Being under target is fine during agglomeration; overshooting is expensive.
    load_overshoot = max(0.0, load_ratio - 1.0)
    child_overshoot = max(0.0, child_ratio - 1.0)

    merged_area = max(a.area + b.area, 1e-12)
    merged_perimeter = max(a.perimeter + b.perimeter - 2.0 * shared, 1e-12)
    compactness = max(
        0.0,
        merged_perimeter * merged_perimeter / (4.0 * math.pi * merged_area) - 1.0,
    )
    compactness = min(compactness, 25.0)

    distance = float(np.linalg.norm(a.centroid - b.centroid) / max(diag, 1.0))
    shared_fraction = shared / max(min(a.perimeter, b.perimeter), 1e-12)

    return (
        9.0 * load_overshoot * load_overshoot
        + 6.0 * child_overshoot * child_overshoot
        + 0.55 * distance
        + 0.10 * compactness
        - 1.25 * shared_fraction
    )


def _agglomerate_component(
    graph: nx.Graph,
    nodes: list[int],
    n_groups: int,
    gdf: gpd.GeoDataFrame,
    point_counts: np.ndarray,
    centroids: np.ndarray,
) -> list[list[int]]:
    """Merge adjacent child regions until ``n_groups`` contiguous parents remain.

    Unlike region-growing from several seeds, this procedure can never create a
    disconnected parent: every operation merges two regions that are adjacent in
    the current region-adjacency graph.
    """
    if n_groups <= 0:
        raise ValueError("n_groups must be positive")
    if n_groups == 1:
        return [sorted(nodes)]
    if n_groups >= len(nodes):
        return [[node] for node in nodes]

    node_set = set(nodes)
    total_load = float(point_counts[nodes].sum())
    target_load = total_load / n_groups
    target_children = len(nodes) / n_groups

    component_pts = centroids[nodes]
    diag = max(
        math.hypot(
            float(component_pts[:, 0].max() - component_pts[:, 0].min()),
            float(component_pts[:, 1].max() - component_pts[:, 1].min()),
        ),
        1.0,
    )

    regions: dict[int, _Region] = {}
    neighbors: dict[int, dict[int, float]] = {}

    for node in nodes:
        geom = gdf.geometry.iloc[node]
        regions[node] = _Region(
            members={node},
            point_count=float(point_counts[node]),
            child_count=1,
            centroid=np.asarray(centroids[node], dtype=float),
            area=float(geom.area),
            perimeter=float(geom.length),
        )
        neighbors[node] = {}

    for u, v, data in graph.subgraph(nodes).edges(data=True):
        shared = float(data.get("shared", 0.0))
        neighbors[int(u)][int(v)] = shared
        neighbors[int(v)][int(u)] = shared

    next_region_id = max(node_set) + 1
    heap: list[tuple[float, int, int]] = []

    def push_edge(a_id: int, b_id: int) -> None:
        if a_id == b_id or a_id not in regions or b_id not in regions:
            return
        if b_id not in neighbors.get(a_id, {}):
            return
        lo, hi = sorted((a_id, b_id))
        shared = neighbors[lo][hi] if hi in neighbors[lo] else neighbors[hi][lo]
        score = _merge_score(
            regions[lo],
            regions[hi],
            shared=shared,
            target_load=target_load,
            target_children=target_children,
            diag=diag,
        )
        heapq.heappush(heap, (score, lo, hi))

    for u in nodes:
        for v in neighbors[u]:
            if u < v:
                push_edge(u, v)

    while len(regions) > n_groups:
        while heap:
            _, a_id, b_id = heapq.heappop(heap)
            if (
                a_id in regions
                and b_id in regions
                and b_id in neighbors.get(a_id, {})
            ):
                break
        else:
            raise RuntimeError(
                "Hierarchy aggregation lost adjacency inside a connected component. "
                "This indicates invalid polygon topology rather than a merge fallback."
            )

        a = regions[a_id]
        b = regions[b_id]
        shared_ab = neighbors[a_id][b_id]

        combined_load = a.point_count + b.point_count
        if combined_load > 0:
            centroid = (
                a.centroid * a.point_count + b.centroid * b.point_count
            ) / combined_load
        else:
            centroid = (a.centroid + b.centroid) / 2.0

        merged = _Region(
            members=a.members | b.members,
            point_count=combined_load,
            child_count=a.child_count + b.child_count,
            centroid=centroid,
            area=a.area + b.area,
            perimeter=max(a.perimeter + b.perimeter - 2.0 * shared_ab, 0.0),
        )

        merged_neighbors: dict[int, float] = {}
        for neighbor_id, shared in neighbors[a_id].items():
            if neighbor_id == b_id or neighbor_id not in regions:
                continue
            merged_neighbors[neighbor_id] = merged_neighbors.get(neighbor_id, 0.0) + shared
        for neighbor_id, shared in neighbors[b_id].items():
            if neighbor_id == a_id or neighbor_id not in regions:
                continue
            merged_neighbors[neighbor_id] = merged_neighbors.get(neighbor_id, 0.0) + shared

        del regions[a_id]
        del regions[b_id]
        neighbors.pop(a_id, None)
        neighbors.pop(b_id, None)
        for neighbor_id in merged_neighbors:
            neighbors[neighbor_id].pop(a_id, None)
            neighbors[neighbor_id].pop(b_id, None)

        new_id = next_region_id
        next_region_id += 1
        regions[new_id] = merged
        neighbors[new_id] = {}

        for neighbor_id, shared in merged_neighbors.items():
            if neighbor_id not in regions:
                continue
            neighbors[new_id][neighbor_id] = shared
            neighbors[neighbor_id][new_id] = shared
            push_edge(new_id, neighbor_id)

    return [sorted(region.members) for _, region in sorted(regions.items())]


def aggregate_level(
    gdf: gpd.GeoDataFrame,
    *,
    branching_factor: int,
) -> tuple[gpd.GeoDataFrame, dict[int, int]]:
    """Aggregate adjacent polygons into balanced, contiguous parent regions.

    The hierarchy is strictly bottom-up. Input polygons are immutable atomic
    children and each returned parent is exactly the union of adjacent children.
    No nearest-region fallback is used, so a parent cannot become disconnected
    merely to satisfy the requested group count.
    """
    if branching_factor < 2:
        raise ValueError("branching_factor must be >= 2")

    gdf = gdf.reset_index(drop=True)
    m = len(gdf)
    if m <= 1:
        if m == 1:
            tile_id = int(gdf.iloc[0]["tile_id"])
            return gdf.copy(), {tile_id: 0}
        return gdf.copy(), {}

    graph = _build_adjacency_graph(gdf)
    component_count = nx.number_connected_components(graph)
    target_groups = max(component_count, math.ceil(m / branching_factor))

    point_counts = gdf["point_count"].to_numpy(dtype=float)
    centroids = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])
    allocation = _allocate_component_groups(graph, point_counts, target_groups)

    groups: list[list[int]] = []
    for nodes, n_groups in allocation:
        groups.extend(
            _agglomerate_component(
                graph,
                nodes,
                n_groups,
                gdf,
                point_counts,
                centroids,
            )
        )

    # Stable IDs make repeated builds deterministic for the same input geometry.
    groups.sort(key=lambda children: min(children))

    # Power-cell boundaries can differ by sub-millimetre floating-point noise.
    # Snap every child to one shared metric precision grid before unioning them.
    # This removes artificial microscopic gaps without changing the meaningful
    # geography, and makes parent polygons topologically match their adjacency.
    precision = _geometry_tolerance(gdf)

    rows = []
    child_to_parent: dict[int, int] = {}
    for parent_id, children in enumerate(groups):
        snapped_children = [
            set_precision(gdf.geometry.iloc[child], grid_size=precision)
            for child in children
        ]
        geometry = union_all(snapped_children)
        point_count = int(gdf["point_count"].iloc[children].sum())
        rows.append(
            {
                "tile_id": parent_id,
                "point_count": point_count,
                "child_count": len(children),
                "geometry": geometry,
            }
        )
        for child_row in children:
            child_tile_id = int(gdf.iloc[child_row]["tile_id"])
            child_to_parent[child_tile_id] = parent_id

    parent_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=gdf.crs)
    return parent_gdf, child_to_parent
