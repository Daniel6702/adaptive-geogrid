from __future__ import annotations

import math
from typing import Callable
import warnings

import numpy as np
from scipy.special import logsumexp
from scipy.spatial import ConvexHull, QhullError
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from sklearn.cluster import KMeans


def _squared_distances(points: np.ndarray, sites: np.ndarray) -> np.ndarray:
    # (a-b)^2 expanded this way is faster and avoids a (N,K,2) temporary.
    p2 = np.sum(points * points, axis=1)[:, None]
    s2 = np.sum(sites * sites, axis=1)[None, :]
    d2 = p2 + s2 - 2.0 * (points @ sites.T)
    np.maximum(d2, 0.0, out=d2)
    return d2


def power_assign(
    points: np.ndarray,
    sites: np.ndarray,
    weights: np.ndarray,
    *,
    chunk_size: int = 20_000,
) -> np.ndarray:
    out = np.empty(len(points), dtype=np.int64)
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        d2 = _squared_distances(points[start:stop], sites)
        out[start:stop] = np.argmin(d2 - weights[None, :], axis=1)
    return out


def _hard_counts(
    points: np.ndarray,
    sites: np.ndarray,
    weights: np.ndarray,
    k: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    assignments = power_assign(points, sites, weights, chunk_size=chunk_size)
    counts = np.bincount(assignments, minlength=k)
    return assignments, counts


def _sinkhorn_weights(
    points: np.ndarray,
    sites: np.ndarray,
    capacities: np.ndarray,
    *,
    initial_weights: np.ndarray | None = None,
    temperatures: tuple[float, ...] = (0.08, 0.03, 0.012, 0.005),
    iterations_per_temperature: int = 80,
    chunk_size: int = 20_000,
) -> np.ndarray:
    """Approximate Laguerre weights with annealed entropic OT.

    Distances are expected in normalized coordinates. The resulting weights are
    squared-distance offsets in the same normalized coordinate system.
    """
    k = len(sites)
    weights = np.zeros(k, dtype=np.float64) if initial_weights is None else initial_weights.copy()
    log_capacity = np.log(capacities.astype(np.float64))

    for tau in temperatures:
        log_v = weights / tau
        log_v -= np.mean(log_v)

        for _ in range(iterations_per_temperature):
            # Given v, compute row scaling u and then the unscaled column sums K^T u.
            log_col_base = np.full(k, -np.inf, dtype=np.float64)

            for start in range(0, len(points), chunk_size):
                stop = min(start + chunk_size, len(points))
                d2 = _squared_distances(points[start:stop], sites)
                log_k = -d2 / tau
                log_u = -logsumexp(log_k + log_v[None, :], axis=1)
                chunk_cols = logsumexp(log_k + log_u[:, None], axis=0)
                log_col_base = np.logaddexp(log_col_base, chunk_cols)

            new_log_v = log_capacity - log_col_base
            new_log_v -= np.mean(new_log_v)
            delta = float(np.max(np.abs(new_log_v - log_v)))
            log_v = new_log_v
            if delta < 1e-7:
                break

        weights = tau * log_v
        weights -= np.mean(weights)

    return weights


def _refine_hard_weights(
    points: np.ndarray,
    sites: np.ndarray,
    capacities: np.ndarray,
    weights: np.ndarray,
    *,
    tolerance: float,
    max_iter: int = 250,
    chunk_size: int = 20_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subgradient refinement against empirical (hard) point counts."""
    k = len(sites)
    target_scale = np.maximum(capacities.astype(float), 1.0)
    # A useful weight step is tied to the squared spacing between neighboring
    # sites. A fixed step is unstable because normalized grids can still vary
    # substantially in local cell scale.
    if k > 1:
        site_d2 = _squared_distances(sites, sites)
        np.fill_diagonal(site_d2, np.inf)
        characteristic = float(np.median(np.min(site_d2, axis=1)))
        step = max(0.15 * characteristic, 1e-7)
    else:
        step = 1e-3
    best_weights = weights.copy()
    best_assignments, best_counts = _hard_counts(points, sites, weights, k, chunk_size)
    best_error = float(np.max(np.abs(best_counts - capacities) / target_scale))

    for iteration in range(max_iter):
        assignments, counts = _hard_counts(points, sites, weights, k, chunk_size)
        rel_error = (capacities - counts) / target_scale
        max_error = float(np.max(np.abs(rel_error)))

        if max_error < best_error:
            best_error = max_error
            best_weights = weights.copy()
            best_assignments = assignments.copy()
            best_counts = counts.copy()

        if max_error <= tolerance:
            return weights, assignments, counts

        # Under-full cells receive a larger weight, lowering their power distance.
        weights += step * rel_error
        weights -= np.mean(weights)
        step *= 0.992

    return best_weights, best_assignments, best_counts


def fit_capacity_power_diagram(
    points_m: np.ndarray,
    k: int,
    *,
    random_state: int = 42,
    lloyd_iterations: int = 4,
    balance_tolerance: float = 0.05,
    chunk_size: int = 20_000,
    progress: Callable[[str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Fit a capacity-balanced centroidal power diagram to empirical points.

    Returns projected sites, projected squared-distance weights, assignments,
    counts, capacities, normalization scale, and normalization origin.
    """
    n = len(points_m)
    if not 1 <= k <= n:
        raise ValueError(f"n_tiles must be between 1 and number of in-boundary points ({n})")

    capacities = np.full(k, n // k, dtype=np.int64)
    capacities[: n % k] += 1

    origin = np.mean(points_m, axis=0)
    spread = np.ptp(points_m, axis=0)
    scale = float(max(np.max(spread), np.max(np.std(points_m, axis=0)), 1.0))
    points = (points_m - origin) / scale

    if k == 1:
        site = np.mean(points, axis=0, keepdims=True)
        assignments = np.zeros(n, dtype=np.int64)
        counts = np.array([n], dtype=np.int64)
        return site * scale + origin, np.zeros(1), assignments, counts, capacities, scale, origin

    km = KMeans(n_clusters=k, init="k-means++", n_init=1, random_state=random_state, max_iter=100)
    km.fit(points)
    sites = km.cluster_centers_.astype(np.float64)
    weights = np.zeros(k, dtype=np.float64)

    assignments = None
    counts = None

    for outer in range(max(1, lloyd_iterations)):
        if progress:
            progress(f"Balancing power cells: Lloyd iteration {outer + 1}/{max(1, lloyd_iterations)}")

        weights = _sinkhorn_weights(
            points,
            sites,
            capacities,
            initial_weights=weights,
            chunk_size=chunk_size,
        )
        weights, assignments, counts = _refine_hard_weights(
            points,
            sites,
            capacities,
            weights,
            tolerance=balance_tolerance,
            chunk_size=chunk_size,
        )

        # Centroidal relaxation, while retaining each site's capacity weight.
        new_sites = sites.copy()
        for i in range(k):
            members = points[assignments == i]
            if len(members):
                new_sites[i] = members.mean(axis=0)
        move = np.max(np.linalg.norm(new_sites - sites, axis=1))
        sites = new_sites
        if move < 1e-6:
            break

    # Final rebalance after final site movement.
    weights = _sinkhorn_weights(
        points,
        sites,
        capacities,
        initial_weights=weights,
        chunk_size=chunk_size,
    )
    weights, assignments, counts = _refine_hard_weights(
        points,
        sites,
        capacities,
        weights,
        tolerance=balance_tolerance,
        chunk_size=chunk_size,
    )

    max_rel_error = float(np.max(np.abs(counts - capacities) / np.maximum(capacities, 1)))
    if max_rel_error > max(balance_tolerance * 2, 0.10):
        warnings.warn(
            f"Capacity optimizer stopped with maximum relative tile-count error {max_rel_error:.1%}. "
            "The grid is still valid, but consider more points per tile or a looser tolerance.",
            stacklevel=2,
        )

    sites_m = sites * scale + origin
    weights_m2 = weights * (scale * scale)
    return sites_m, weights_m2, assignments, counts, capacities, scale, origin


def _regular_neighbors(sites: np.ndarray, weights: np.ndarray) -> list[set[int]]:
    """Neighbors in a 2-D power diagram via the lower convex hull lifting."""
    k = len(sites)
    neighbors = [set() for _ in range(k)]
    if k == 1:
        return neighbors
    if k == 2:
        neighbors[0].add(1)
        neighbors[1].add(0)
        return neighbors

    z = np.sum(sites * sites, axis=1) - weights
    lifted = np.column_stack([sites, z])
    try:
        hull = ConvexHull(lifted, qhull_options="QJ")
    except QhullError:
        for i in range(k):
            neighbors[i].update(j for j in range(k) if j != i)
        return neighbors

    for simplex, equation in zip(hull.simplices, hull.equations):
        if equation[2] >= -1e-10:
            continue  # upper hull facet
        a, b, c = map(int, simplex)
        for u, v in ((a, b), (b, c), (c, a)):
            neighbors[u].add(v)
            neighbors[v].add(u)
    return neighbors


def _box_halfplane(bounds, normal: np.ndarray, offset: float) -> Polygon:
    """Clip an axis-aligned bounding box by normal·x <= offset."""
    minx, miny, maxx, maxy = bounds
    vertices = [
        np.array([minx, miny], dtype=float),
        np.array([maxx, miny], dtype=float),
        np.array([maxx, maxy], dtype=float),
        np.array([minx, maxy], dtype=float),
    ]

    def inside(p):
        return float(np.dot(normal, p)) <= offset + 1e-8

    def intersection(p, q):
        d = q - p
        denom = float(np.dot(normal, d))
        if abs(denom) < 1e-15:
            return p
        t = (offset - float(np.dot(normal, p))) / denom
        return p + t * d

    output: list[np.ndarray] = []
    for idx, current in enumerate(vertices):
        previous = vertices[idx - 1]
        curr_in = inside(current)
        prev_in = inside(previous)
        if curr_in:
            if not prev_in:
                output.append(intersection(previous, current))
            output.append(current)
        elif prev_in:
            output.append(intersection(previous, current))

    if len(output) < 3:
        return Polygon()
    return Polygon([(float(p[0]), float(p[1])) for p in output])


def build_power_polygons(
    boundary: BaseGeometry,
    sites_m: np.ndarray,
    weights_m2: np.ndarray,
    *,
    normalized_sites: np.ndarray | None = None,
    normalized_weights: np.ndarray | None = None,
) -> list[BaseGeometry]:
    """Construct bounded Laguerre/power cells clipped to ``boundary``."""
    k = len(sites_m)
    if k == 1:
        return [boundary]

    # Convex-hull lifting is numerically happier in normalized coordinates.
    if normalized_sites is None:
        origin = np.mean(sites_m, axis=0)
        scale = max(float(np.ptp(sites_m, axis=0).max()), 1.0)
        normalized_sites = (sites_m - origin) / scale
        normalized_weights = weights_m2 / (scale * scale)
    neighbors = _regular_neighbors(normalized_sites, normalized_weights)

    bounds = boundary.bounds
    cells: list[BaseGeometry] = []
    for i in range(k):
        cell = boundary
        si = sites_m[i]
        wi = weights_m2[i]
        candidate_neighbors = neighbors[i]
        if not candidate_neighbors:
            candidate_neighbors = {j for j in range(k) if j != i}

        for j in candidate_neighbors:
            sj = sites_m[j]
            wj = weights_m2[j]
            normal = 2.0 * (sj - si)
            offset = float(np.dot(sj, sj) - np.dot(si, si) + wi - wj)
            hp = _box_halfplane(bounds, normal, offset)
            if hp.is_empty:
                cell = Polygon()
                break
            cell = cell.intersection(hp)
            if cell.is_empty:
                break
        cells.append(cell)

    return cells
