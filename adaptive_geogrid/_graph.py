from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable
import warnings

import numpy as np
from scipy.spatial import Delaunay, QhullError, cKDTree
from shapely import union_all, voronoi_polygons
from shapely.geometry import MultiPoint, Polygon
from shapely.geometry.base import BaseGeometry

from ._power import fit_capacity_power_diagram


@dataclass
class GraphPartitionState:
    """Classifier state for the point-graph partition backend.

    Each unique training coordinate owns its ordinary Euclidean Voronoi microcell.
    ``generator_labels`` groups those microcells into the final graph partitions.
    Consequently classification is exact and inexpensive: find the nearest stored
    generator point, then return that generator's partition label.
    """

    generator_points: np.ndarray
    generator_labels: np.ndarray
    characteristic_spacing: float
    _tree: cKDTree | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        points = np.asarray(self.generator_points, dtype=np.float64)
        labels = np.asarray(self.generator_labels, dtype=np.int32)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("graph generator_points must have shape (N, 2)")
        if labels.ndim != 1 or len(labels) != len(points):
            raise ValueError("graph generator_labels must have one label per point")
        if len(points) == 0:
            raise ValueError("graph partition state cannot be empty")
        self.generator_points = points
        self.generator_labels = labels
        self.characteristic_spacing = float(max(self.characteristic_spacing, 1e-12))

    def classify_projected(self, points: np.ndarray) -> np.ndarray:
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")
        if self._tree is None:
            self._tree = cKDTree(self.generator_points)
        _, nearest = self._tree.query(arr, k=1)
        return self.generator_labels[np.asarray(nearest, dtype=np.int64)].astype(
            np.int64, copy=False
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "type": "delaunay_voronoi_graph_partition",
            "characteristic_spacing": float(self.characteristic_spacing),
        }

    @classmethod
    def from_metadata(
        cls,
        data: dict[str, Any],
        *,
        generator_points: np.ndarray,
        generator_labels: np.ndarray,
    ) -> "GraphPartitionState":
        if data.get("type") != "delaunay_voronoi_graph_partition":
            raise ValueError(f"Unsupported graph partition type: {data.get('type')!r}")
        return cls(
            generator_points=np.asarray(generator_points, dtype=np.float64),
            generator_labels=np.asarray(generator_labels, dtype=np.int32),
            characteristic_spacing=float(data["characteristic_spacing"]),
        )


@dataclass(frozen=True)
class _PointGraph:
    edge_u: np.ndarray
    edge_v: np.ndarray
    edge_cut_cost: np.ndarray
    adjacency: tuple[np.ndarray, ...]
    adjacency_cost: tuple[np.ndarray, ...]
    characteristic_spacing: float

    @property
    def n_nodes(self) -> int:
        return len(self.adjacency)


def _unique_points_with_mass(points: np.ndarray):
    unique, first, inverse, counts = np.unique(
        np.asarray(points, dtype=np.float64),
        axis=0,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    return (
        unique,
        first.astype(np.int64, copy=False),
        inverse.astype(np.int64, copy=False),
        counts.astype(np.int64, copy=False),
    )


def _delaunay_edges(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(points)
    if n <= 1:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if n == 2:
        return np.array([0], dtype=np.int64), np.array([1], dtype=np.int64)

    edges: set[tuple[int, int]] = set()
    try:
        triangulation = Delaunay(points, qhull_options="QJ Qbb Qc")
        for simplex in np.asarray(triangulation.simplices, dtype=np.int64):
            m = len(simplex)
            for i in range(m):
                for j in range(i + 1, m):
                    a, b = int(simplex[i]), int(simplex[j])
                    if a == b or a >= n or b >= n:
                        continue
                    edges.add((a, b) if a < b else (b, a))
    except QhullError:
        edges.clear()

    # Degenerate/collinear point sets can defeat Delaunay. A symmetric local kNN
    # graph keeps the backend usable while retaining spatial locality.
    if not edges:
        tree = cKDTree(points)
        q = min(7, n)
        _, neighbors = tree.query(points, k=q)
        neighbors = np.asarray(neighbors)
        if neighbors.ndim == 1:
            neighbors = neighbors[:, None]
        for i in range(n):
            for candidate in neighbors[i]:
                j = int(candidate)
                if i == j:
                    continue
                edges.add((i, j) if i < j else (j, i))

    # Delaunay should already be connected. For numerical degeneracies, explicitly
    # connect components by nearest-point bridges.
    adjacency_sets = [set() for _ in range(n)]
    for a, b in edges:
        adjacency_sets[a].add(b)
        adjacency_sets[b].add(a)

    def components():
        seen = np.zeros(n, dtype=bool)
        result = []
        for start in range(n):
            if seen[start]:
                continue
            stack = [start]
            seen[start] = True
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adjacency_sets[u]:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
            result.append(np.asarray(comp, dtype=np.int64))
        return result

    comps = components()
    while len(comps) > 1:
        base = max(comps, key=len)
        base_tree = cKDTree(points[base])
        best = None
        for comp in comps:
            if comp is base:
                continue
            distances, nearest = base_tree.query(points[comp], k=1)
            local = int(np.argmin(distances))
            candidate = (
                float(distances[local]),
                int(base[int(nearest[local])]),
                int(comp[local]),
            )
            if best is None or candidate[0] < best[0]:
                best = candidate
        assert best is not None
        _, a, b = best
        edges.add((a, b) if a < b else (b, a))
        adjacency_sets[a].add(b)
        adjacency_sets[b].add(a)
        comps = components()

    ordered = sorted(edges)
    u = np.fromiter((a for a, _ in ordered), dtype=np.int64, count=len(ordered))
    v = np.fromiter((b for _, b in ordered), dtype=np.int64, count=len(ordered))
    return u, v


def _organic_edge_field(
    midpoints: np.ndarray,
    *,
    bounds: tuple[float, float, float, float],
    strength: float,
    octaves: int,
    random_state: int,
) -> np.ndarray:
    if len(midpoints) == 0 or strength <= 0:
        return np.ones(len(midpoints), dtype=np.float64)

    minx, miny, maxx, maxy = bounds
    longest = max(maxx - minx, maxy - miny, 1.0)
    xn = (midpoints[:, 0] - 0.5 * (minx + maxx)) / longest
    yn = (midpoints[:, 1] - 0.5 * (miny + maxy)) / longest
    rng = np.random.default_rng(random_state)
    field = np.zeros(len(midpoints), dtype=np.float64)
    amplitude_sum = 0.0

    for octave in range(octaves):
        wavelength = 0.70 / (2**octave)
        amplitude = 0.78**octave
        for _ in range(2):
            angle = float(rng.uniform(0.0, 2.0 * math.pi))
            phase = float(rng.uniform(0.0, 2.0 * math.pi))
            direction = math.cos(angle) * xn + math.sin(angle) * yn
            field += amplitude * np.sin(2.0 * math.pi * direction / wavelength + phase)
            amplitude_sum += amplitude

    if amplitude_sum:
        field /= amplitude_sum
    max_abs = float(np.max(np.abs(field))) if len(field) else 0.0
    if max_abs > 1e-12:
        field /= max_abs
    return np.exp(float(strength) * field)


def _build_point_graph(
    points: np.ndarray,
    boundary: BaseGeometry,
    *,
    organic_strength: float,
    organic_octaves: int,
    random_state: int,
) -> _PointGraph:
    edge_u, edge_v = _delaunay_edges(points)
    n = len(points)

    if len(edge_u):
        lengths = np.linalg.norm(points[edge_u] - points[edge_v], axis=1)
        positive = lengths[lengths > 1e-12]
        spacing = float(np.median(positive)) if len(positive) else 1.0
        # Cutting a very short Delaunay edge is expensive: close points should
        # preferentially stay together. The clipping prevents isolated numerical
        # distances from dominating the entire partition.
        closeness = np.clip(spacing / np.maximum(lengths, spacing * 1e-6), 0.25, 4.0)
        midpoint = 0.5 * (points[edge_u] + points[edge_v])
        organic = _organic_edge_field(
            midpoint,
            bounds=tuple(map(float, boundary.bounds)),
            strength=organic_strength,
            octaves=organic_octaves,
            random_state=random_state,
        )
        cut_cost = closeness * organic
    else:
        spacing = 1.0
        cut_cost = np.empty(0, dtype=np.float64)

    neighbors: list[list[int]] = [[] for _ in range(n)]
    costs: list[list[float]] = [[] for _ in range(n)]
    for a, b, cost in zip(edge_u, edge_v, cut_cost):
        ai, bi = int(a), int(b)
        c = float(cost)
        neighbors[ai].append(bi)
        costs[ai].append(c)
        neighbors[bi].append(ai)
        costs[bi].append(c)

    return _PointGraph(
        edge_u=edge_u,
        edge_v=edge_v,
        edge_cut_cost=np.asarray(cut_cost, dtype=np.float64),
        adjacency=tuple(np.asarray(v, dtype=np.int64) for v in neighbors),
        adjacency_cost=tuple(np.asarray(v, dtype=np.float64) for v in costs),
        characteristic_spacing=max(spacing, 1e-9),
    )


def _label_components(label: int, labels: np.ndarray, graph: _PointGraph) -> list[np.ndarray]:
    nodes = np.flatnonzero(labels == label)
    if len(nodes) <= 1:
        return [nodes]
    node_set = set(map(int, nodes))
    seen: set[int] = set()
    result: list[np.ndarray] = []
    for start in nodes:
        s = int(start)
        if s in seen:
            continue
        stack = [s]
        seen.add(s)
        comp: list[int] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in graph.adjacency[u]:
                vi = int(v)
                if vi in node_set and vi not in seen:
                    seen.add(vi)
                    stack.append(vi)
        result.append(np.asarray(comp, dtype=np.int64))
    return result


def _repair_connectivity(
    labels: np.ndarray,
    masses: np.ndarray,
    points: np.ndarray,
    capacities: np.ndarray,
    graph: _PointGraph,
) -> np.ndarray:
    labels = labels.copy()
    k = len(capacities)

    # Moving every minor disconnected component to an adjacent region makes each
    # label connected without inventing non-local geometry. Iterate because a
    # moved component can also repair/alter the receiver's component structure.
    for _ in range(max(2, k * 2)):
        changed = False
        counts = np.bincount(labels, weights=masses, minlength=k).astype(np.float64)
        centroids = np.zeros((k, 2), dtype=np.float64)
        for label in range(k):
            idx = np.flatnonzero(labels == label)
            if len(idx):
                w = masses[idx].astype(np.float64)
                centroids[label] = np.average(points[idx], axis=0, weights=w)

        for label in range(k):
            comps = _label_components(label, labels, graph)
            if len(comps) <= 1:
                continue
            comp_mass = [int(np.sum(masses[c])) for c in comps]
            keep = int(np.argmax(comp_mass))
            for ci, comp in enumerate(comps):
                if ci == keep or len(comp) == 0:
                    continue
                neighbor_labels: dict[int, int] = {}
                for u in comp:
                    for v in graph.adjacency[int(u)]:
                        other = int(labels[int(v)])
                        if other != label:
                            neighbor_labels[other] = neighbor_labels.get(other, 0) + 1
                if not neighbor_labels:
                    continue
                mass = float(np.sum(masses[comp]))
                comp_center = np.average(points[comp], axis=0, weights=masses[comp])

                def target_score(other: int):
                    # Prefer a strongly adjacent target, then the target that most
                    # improves capacity, then geographic proximity.
                    adjacency_score = -neighbor_labels[other]
                    capacity_before = abs(counts[other] - capacities[other])
                    capacity_after = abs(counts[other] + mass - capacities[other])
                    distance = np.linalg.norm(comp_center - centroids[other])
                    return (adjacency_score, capacity_after - capacity_before, distance)

                target = min(neighbor_labels, key=target_score)
                labels[comp] = int(target)
                counts[label] -= mass
                counts[target] += mass
                changed = True
        if not changed:
            break
    return labels


def _can_remove_node(
    node: int,
    label: int,
    labels: np.ndarray,
    graph: _PointGraph,
    region_node_count: int,
) -> bool:
    if region_node_count <= 1:
        return False
    same = [int(v) for v in graph.adjacency[node] if int(labels[int(v)]) == label]
    if len(same) <= 1:
        return True

    target_count = region_node_count - 1
    seen = {same[0]}
    stack = [same[0]]
    while stack:
        u = stack.pop()
        for v in graph.adjacency[u]:
            vi = int(v)
            if vi == node or vi in seen or int(labels[vi]) != label:
                continue
            seen.add(vi)
            stack.append(vi)
    return len(seen) == target_count


def _rebalance_connected(
    labels: np.ndarray,
    masses: np.ndarray,
    capacities: np.ndarray,
    graph: _PointGraph,
    *,
    tolerance: float,
    random_state: int,
    max_passes: int = 24,
) -> np.ndarray:
    labels = labels.copy()
    k = len(capacities)
    counts = np.bincount(labels, weights=masses, minlength=k).astype(np.int64)
    node_counts = np.bincount(labels, minlength=k).astype(np.int64)
    rng = np.random.default_rng(random_state)

    for _ in range(max_passes):
        rel = np.abs(counts - capacities) / np.maximum(capacities, 1)
        if float(np.max(rel)) <= tolerance:
            break
        changed = False
        order = rng.permutation(len(graph.edge_u))
        for edge_idx in order:
            u = int(graph.edge_u[edge_idx])
            v = int(graph.edge_v[edge_idx])
            a, b = int(labels[u]), int(labels[v])
            if a == b:
                continue

            best = None
            for node, donor, receiver in ((u, a, b), (v, b, a)):
                mass = int(masses[node])
                if counts[donor] - mass <= 0:
                    continue
                before = float((counts[donor] - capacities[donor]) ** 2)
                before += float((counts[receiver] - capacities[receiver]) ** 2)
                after = float((counts[donor] - mass - capacities[donor]) ** 2)
                after += float((counts[receiver] + mass - capacities[receiver]) ** 2)
                gain = before - after
                if gain <= 0:
                    continue
                if not _can_remove_node(
                    node,
                    donor,
                    labels,
                    graph,
                    int(node_counts[donor]),
                ):
                    continue
                candidate = (gain, node, donor, receiver, mass)
                if best is None or candidate[0] > best[0]:
                    best = candidate

            if best is None:
                continue
            _, node, donor, receiver, mass = best
            labels[node] = receiver
            counts[donor] -= mass
            counts[receiver] += mass
            node_counts[donor] -= 1
            node_counts[receiver] += 1
            changed = True

        if not changed:
            break
    return labels


def _cluster_stats(
    labels: np.ndarray,
    masses: np.ndarray,
    points: np.ndarray,
    k: int,
):
    counts = np.bincount(labels, weights=masses, minlength=k).astype(np.float64)
    sums = np.zeros((k, 2), dtype=np.float64)
    sumsq = np.zeros(k, dtype=np.float64)
    weighted = points * masses[:, None]
    np.add.at(sums, labels, weighted)
    np.add.at(sumsq, labels, np.sum(points * points, axis=1) * masses)
    return counts, sums, sumsq


def _sse(count: float, total: np.ndarray, sumsq: float) -> float:
    if count <= 0:
        return 0.0
    return float(sumsq - np.dot(total, total) / count)


def _local_cut_delta_for_swap(
    u: int,
    v: int,
    labels: np.ndarray,
    graph: _PointGraph,
) -> float:
    a, b = int(labels[u]), int(labels[v])
    touched: dict[tuple[int, int], float] = {}
    for node in (u, v):
        for neighbor, cost in zip(graph.adjacency[node], graph.adjacency_cost[node]):
            ni = int(neighbor)
            key = (node, ni) if node < ni else (ni, node)
            touched[key] = float(cost)

    def after_label(node: int) -> int:
        if node == u:
            return b
        if node == v:
            return a
        return int(labels[node])

    before = 0.0
    after = 0.0
    for (x, y), cost in touched.items():
        if int(labels[x]) != int(labels[y]):
            before += cost
        if after_label(x) != after_label(y):
            after += cost
    return after - before


def _refine_equal_mass_swaps(
    labels: np.ndarray,
    masses: np.ndarray,
    points: np.ndarray,
    graph: _PointGraph,
    *,
    k: int,
    compactness_weight: float,
    boundary_weight: float,
    iterations: int,
    random_state: int,
) -> np.ndarray:
    labels = labels.copy()
    counts, sums, sumsq = _cluster_stats(labels, masses, points, k)
    node_counts = np.bincount(labels, minlength=k).astype(np.int64)
    rng = np.random.default_rng(random_state)
    scale2 = graph.characteristic_spacing**2

    for _ in range(iterations):
        accepted = 0
        touched_nodes: set[int] = set()
        for edge_idx in rng.permutation(len(graph.edge_u)):
            u = int(graph.edge_u[edge_idx])
            v = int(graph.edge_v[edge_idx])
            if u in touched_nodes or v in touched_nodes:
                continue
            a, b = int(labels[u]), int(labels[v])
            if a == b or int(masses[u]) != int(masses[v]):
                continue

            # After exchanging the two nodes, each inserted node must touch the
            # receiver independently of the counterpart that is being removed.
            if not any(int(n) != v and int(labels[int(n)]) == b for n in graph.adjacency[u]):
                continue
            if not any(int(n) != u and int(labels[int(n)]) == a for n in graph.adjacency[v]):
                continue
            if not _can_remove_node(u, a, labels, graph, int(node_counts[a])):
                continue
            if not _can_remove_node(v, b, labels, graph, int(node_counts[b])):
                continue

            mass = float(masses[u])
            old_sse = _sse(counts[a], sums[a], sumsq[a]) + _sse(
                counts[b], sums[b], sumsq[b]
            )
            new_sum_a = sums[a] - mass * points[u] + mass * points[v]
            new_sum_b = sums[b] - mass * points[v] + mass * points[u]
            new_sq_a = sumsq[a] - mass * float(np.dot(points[u], points[u])) + mass * float(
                np.dot(points[v], points[v])
            )
            new_sq_b = sumsq[b] - mass * float(np.dot(points[v], points[v])) + mass * float(
                np.dot(points[u], points[u])
            )
            new_sse = _sse(counts[a], new_sum_a, new_sq_a) + _sse(
                counts[b], new_sum_b, new_sq_b
            )
            compact_delta = (new_sse - old_sse) / max(scale2, 1e-12)
            boundary_delta = _local_cut_delta_for_swap(u, v, labels, graph)
            score = compactness_weight * compact_delta + boundary_weight * boundary_delta
            if score >= -1e-10:
                continue

            labels[u], labels[v] = b, a
            sums[a], sums[b] = new_sum_a, new_sum_b
            sumsq[a], sumsq[b] = new_sq_a, new_sq_b
            touched_nodes.add(u)
            touched_nodes.add(v)
            accepted += 1

        if accepted == 0:
            break
    return labels


def _centroids_from_labels(
    points: np.ndarray,
    masses: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> np.ndarray:
    result = np.zeros((k, 2), dtype=np.float64)
    for label in range(k):
        idx = np.flatnonzero(labels == label)
        if len(idx):
            result[label] = np.average(points[idx], axis=0, weights=masses[idx])
    return result


def fit_capacity_graph_partition(
    points_m: np.ndarray,
    boundary: BaseGeometry,
    k: int,
    *,
    random_state: int = 42,
    lloyd_iterations: int = 4,
    balance_tolerance: float = 0.05,
    chunk_size: int = 20_000,
    compactness_weight: float = 1.0,
    boundary_weight: float = 0.35,
    organic_strength: float = 0.15,
    organic_octaves: int = 4,
    refine_iterations: int = 8,
    graph_random_state: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    GraphPartitionState,
]:
    """Fit a spatially constrained, capacity-balanced point-graph partition.

    The initial power solution supplies a strong balanced/compact starting point.
    The actual final partition is then optimized on the Delaunay adjacency graph:
    disconnected fragments are repaired, capacity errors are moved through legal
    connected boundary transfers, and equal-mass boundary swaps minimize a direct
    combination of geographic within-tile dispersion and graph-cut boundary cost.

    Final geometry/classification is not a power diagram. Each training location
    owns its ordinary Voronoi microcell, and graph labels merge those microcells
    into the final connected regions.
    """
    points_m = np.asarray(points_m, dtype=np.float64)
    n = len(points_m)
    if not 1 <= k <= n:
        raise ValueError(f"n_tiles must be between 1 and number of in-boundary points ({n})")
    if compactness_weight < 0:
        raise ValueError("graph_compactness must be >= 0")
    if boundary_weight < 0:
        raise ValueError("graph_boundary_weight must be >= 0")
    if organic_strength < 0:
        raise ValueError("graph_organic_strength must be >= 0")
    if organic_octaves < 1:
        raise ValueError("graph_octaves must be >= 1")
    if refine_iterations < 0:
        raise ValueError("graph_refine_iterations must be >= 0")

    capacities = np.full(k, n // k, dtype=np.int64)
    capacities[: n % k] += 1

    if progress:
        progress("Building balanced power initialization for graph partition")
    (
        _power_sites,
        _power_weights,
        initial_assignments,
        _power_counts,
        _power_capacities,
        _scale,
        _origin,
    ) = fit_capacity_power_diagram(
        points_m,
        k,
        random_state=random_state,
        lloyd_iterations=lloyd_iterations,
        balance_tolerance=balance_tolerance,
        chunk_size=chunk_size,
        progress=progress,
    )

    unique_points, first, inverse, masses = _unique_points_with_mass(points_m)
    labels = np.asarray(initial_assignments[first], dtype=np.int32)
    graph_seed = random_state if graph_random_state is None else int(graph_random_state)

    if progress:
        progress(
            f"Building Delaunay point graph from {len(unique_points)} unique coordinate(s)"
        )
    graph = _build_point_graph(
        unique_points,
        boundary,
        organic_strength=float(organic_strength),
        organic_octaves=int(organic_octaves),
        random_state=graph_seed,
    )

    if k > 1:
        if progress:
            progress("Repairing graph connectivity and capacity after initialization")
        labels = _repair_connectivity(labels, masses, unique_points, capacities, graph)
        labels = _rebalance_connected(
            labels,
            masses,
            capacities,
            graph,
            tolerance=balance_tolerance,
            random_state=graph_seed + 1,
        )

        if refine_iterations > 0:
            if progress:
                progress(
                    "Optimizing graph partition for geographic compactness and boundary regularity"
                )
            labels = _refine_equal_mass_swaps(
                labels,
                masses,
                unique_points,
                graph,
                k=k,
                compactness_weight=float(compactness_weight),
                boundary_weight=float(boundary_weight),
                iterations=int(refine_iterations),
                random_state=graph_seed + 2,
            )

    assignments = labels[inverse].astype(np.int64, copy=False)
    counts = np.bincount(assignments, minlength=k).astype(np.int64)
    max_rel_error = float(np.max(np.abs(counts - capacities) / np.maximum(capacities, 1)))
    if max_rel_error > max(balance_tolerance * 2, 0.10):
        warnings.warn(
            f"Graph partition stopped with maximum relative tile-count error "
            f"{max_rel_error:.1%}. Increase target_points, loosen balance_tolerance, "
            "or reduce graph regularization strength.",
            stacklevel=2,
        )

    # Ensure every final graph label is connected. The refinement operations are
    # connectivity-preserving; this assertion catches only unexpected numerical/
    # implementation failures rather than silently returning fragmented regions.
    for label in range(k):
        comps = _label_components(label, labels, graph)
        if len(comps) != 1:
            raise RuntimeError(f"Graph partition label {label} is disconnected after refinement")

    sites = _centroids_from_labels(unique_points, masses, labels, k)
    weights = np.zeros(k, dtype=np.float64)
    state = GraphPartitionState(
        generator_points=unique_points,
        generator_labels=labels,
        characteristic_spacing=graph.characteristic_spacing,
    )
    return sites, weights, assignments, counts, capacities, state


def build_graph_polygons(
    state: GraphPartitionState,
    boundary: BaseGeometry,
    k: int,
    *,
    simplify: float = 0.5,
) -> list[BaseGeometry]:
    """Build final graph tiles as unions of point-owned Voronoi microcells.

    This gives exact complete coverage without rasterization. ``simplify`` is a
    topology-preserving coverage simplification tolerance measured as a fraction
    of the median Delaunay spacing. Set it to 0 for exact microcell boundaries.
    """
    if simplify < 0:
        raise ValueError("graph_simplify must be >= 0")

    points = state.generator_points
    labels = state.generator_labels
    if len(points) == 1:
        return [boundary] + [Polygon() for _ in range(max(0, k - 1))]

    cells_collection = voronoi_polygons(
        MultiPoint(points),
        extend_to=boundary.envelope,
        ordered=True,
    )
    cells = list(cells_collection.geoms)
    if len(cells) != len(points):
        raise RuntimeError("Voronoi microcell construction did not preserve generator ordering")

    parts: list[list[BaseGeometry]] = [[] for _ in range(k)]
    for cell, label in zip(cells, labels):
        clipped = cell.intersection(boundary)
        if not clipped.is_empty:
            parts[int(label)].append(clipped)

    polygons: list[BaseGeometry] = []
    for label in range(k):
        geom = union_all(parts[label]) if parts[label] else Polygon()
        polygons.append(geom.intersection(boundary))

    if simplify > 0 and len(polygons) > 1:
        try:
            from shapely import coverage_simplify

            tolerance = float(simplify) * state.characteristic_spacing
            polygons = list(
                coverage_simplify(
                    np.asarray(polygons, dtype=object),
                    tolerance,
                    simplify_boundary=False,
                )
            )
        except (ValueError, TypeError):
            # Preserve exact Voronoi coverage if a pathological source boundary
            # prevents Shapely from treating the clipped cells as one coverage.
            pass

    return polygons
