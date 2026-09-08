import json

import geopandas as gpd
import matplotlib
import numpy as np
from shapely.geometry import box

matplotlib.use("Agg")

from adaptive_geogrid import AdaptiveGeoGrid, load_grid, tessellate


def _small_grid(seed=4):
    rng = np.random.default_rng(seed)
    points = rng.uniform([10.0, 56.0], [10.2, 56.2], size=(400, 2))
    boundary = box(9.99, 55.99, 10.21, 56.21)
    grid = tessellate(
        points=points,
        boundary=boundary,
        n_tiles=8,
        projected_crs="EPSG:32632",
        lloyd_iterations=2,
        balance_tolerance=0.15,
    )
    return grid, points


def test_tessellate_classify_and_hierarchy():
    grid, points = _small_grid()
    assert len(grid.tiles) == 8
    assert grid.tiles.geometry.notna().all()
    assert all(not g.is_empty for g in grid.tiles.geometry)

    classified = grid.classify(points)
    assert classified.shape == (400,)
    assert np.all(classified >= 0)

    grid.build_hierarchy(branching_factor=4, levels=2)
    classified_h = grid.classify(points[:10], include_hierarchy=True)
    assert classified_h.shape == (10, grid.n_levels)
    assert np.array_equal(classified_h[:, 0], grid.classify(points[:10]))
    assert grid.n_levels >= 2
    assert grid.n_classes[0] == 8

    as_df = grid.classify(points[:10], include_hierarchy=True, as_dataframe=True)
    assert list(as_df.columns) == [f"level_{i}" for i in range(grid.n_levels)]
    assert np.array_equal(as_df.to_numpy(), classified_h)


def test_class_paths_are_training_friendly():
    grid, points = _small_grid(seed=6)
    grid.build_hierarchy(branching_factor=4, levels=2)

    fine = grid.classify(points[:30])
    paths_from_ids = grid.class_paths(fine)
    paths_from_coords = grid.classify(points[:30], include_hierarchy=True)
    assert np.array_equal(paths_from_ids, paths_from_coords)

    single = grid.class_path(int(fine[0]))
    assert single.shape == (grid.n_levels,)
    assert np.array_equal(single, paths_from_coords[0])

    table = grid.hierarchy_table()
    assert table.shape == (grid.n_classes[0], grid.n_levels)
    for fine_id in np.unique(fine[:10]):
        assert np.array_equal(table.iloc[int(fine_id)].to_numpy(), grid.class_path(int(fine_id)))


def test_clustered_points_are_balanced_and_cover_boundary():
    rng = np.random.default_rng(10)
    dense = rng.normal([10.08, 56.08], [0.018, 0.018], size=(700, 2))
    sparse = rng.uniform([10.0, 56.0], [10.2, 56.2], size=(300, 2))
    points = np.vstack([dense, sparse])
    mask = (
        (points[:, 0] >= 10.0)
        & (points[:, 0] <= 10.2)
        & (points[:, 1] >= 56.0)
        & (points[:, 1] <= 56.2)
    )
    points = points[mask]

    grid = tessellate(
        points=points,
        boundary=box(10.0, 56.0, 10.2, 56.2),
        n_tiles=10,
        projected_crs="EPSG:32632",
        lloyd_iterations=2,
        balance_tolerance=0.12,
    )

    target = len(points) / 10
    assert np.max(np.abs(grid.tiles.point_count.to_numpy() - target)) <= target * 0.20
    union = grid._levels_projected[0].geometry.union_all()
    assert abs(union.area - grid.boundary_projected.area) / grid.boundary_projected.area < 1e-8

    grid.build_hierarchy(branching_factor=5, levels=1)
    assert len(grid.level(1)) == 2
    result = grid.classify(points[:20], include_hierarchy=True)
    assert result.shape == (20, 2)


def test_outside_classification_is_minus_one():
    grid = tessellate(
        points=[(10.0, 56.0), (10.1, 56.0), (10.0, 56.1), (10.1, 56.1)],
        boundary=box(9.9, 55.9, 10.2, 56.2),
        n_tiles=2,
        projected_crs="EPSG:32632",
        balance_tolerance=0.5,
    )
    grid.build_hierarchy(branching_factor=2, levels=1)
    result = grid.classify([(10.0, 56.0), (20.0, 60.0)], include_hierarchy=True)
    assert result[0, 0] >= 0
    assert np.all(result[1] == -1)


def test_geojson_all_levels_and_single_level(tmp_path):
    grid, _ = _small_grid(seed=8)
    grid.build_hierarchy(branching_factor=4, levels=2)

    all_path = tmp_path / "all.geojson"
    grid.to_geojson(all_path)
    exported = gpd.read_file(all_path)
    assert set(exported["level"].astype(int)) == set(range(grid.n_levels))
    assert len(exported) == sum(grid.n_classes)

    fine_path = tmp_path / "fine.geojson"
    grid.to_geojson(fine_path, level=0)
    fine = gpd.read_file(fine_path)
    assert len(fine) == grid.n_classes[0]
    assert set(fine["level"].astype(int)) == {0}


def test_save_load_round_trip_for_training(tmp_path):
    grid, points = _small_grid(seed=12)
    grid.build_hierarchy(branching_factor=4, levels=2)
    expected = grid.classify(points[:50], include_hierarchy=True)

    archive = tmp_path / "aarhus.aggrid"
    grid.save(archive)
    loaded = load_grid(archive)
    assert isinstance(loaded, AdaptiveGeoGrid)
    assert loaded.n_classes == grid.n_classes
    assert loaded.n_levels == grid.n_levels
    assert np.array_equal(loaded.classify(points[:50], include_hierarchy=True), expected)
    assert np.array_equal(loaded.hierarchy_table().to_numpy(), grid.hierarchy_table().to_numpy())

    directory = tmp_path / "grid_dir"
    grid.to_directory(directory)
    loaded_dir = AdaptiveGeoGrid.load(directory)
    assert np.array_equal(loaded_dir.classify(points[:50], include_hierarchy=True), expected)


def test_plot_multiple_levels(tmp_path):
    grid, _ = _small_grid(seed=14)
    grid.build_hierarchy(branching_factor=4, levels=2)
    ax = grid.plot()
    output = tmp_path / "grid.png"
    ax.figure.savefig(output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_hierarchy_uses_overlapping_boundaries_and_points_can_be_sampled(tmp_path):
    grid, points = _small_grid(seed=21)
    grid.build_hierarchy(branching_factor=2, levels=2)

    boundaries = grid._hierarchy_boundaries_projected(list(range(grid.n_levels)))
    # A coarser hierarchy border should also exist in every finer grid level.
    # This overlap is intentional: plot() renders it as nested color bands.
    for coarse_level in range(1, grid.n_levels):
        coarse = boundaries[coarse_level]
        assert not coarse.is_empty
        for fine_level in range(coarse_level):
            fine = boundaries[fine_level]
            tol = max(
                np.hypot(*(
                    grid._levels_projected[0].total_bounds[2:]
                    - grid._levels_projected[0].total_bounds[:2]
                )) * 1e-7,
                1e-5,
            )
            covered = coarse.difference(fine.buffer(tol))
            assert covered.length < max(tol * 20, coarse.length * 1e-6)

    ax = grid.plot(
        points=points,
        point_sample=50,
        point_random_state=7,
        hierarchy_lane_width=1.3,
        hierarchy_lane_gap=0.45,
    )
    output = tmp_path / "banded_grid_with_points.png"
    ax.figure.savefig(output, dpi=140)
    assert output.exists()
    assert output.stat().st_size > 0



def test_warped_mode_classifies_training_points_consistently():
    rng = np.random.default_rng(31)
    points = rng.uniform([10.0, 56.0], [10.2, 56.2], size=(600, 2))
    boundary = box(9.99, 55.99, 10.21, 56.21)

    grid = tessellate(
        points=points,
        boundary=boundary,
        n_tiles=12,
        mode="warped",
        projected_crs="EPSG:32632",
        random_state=11,
        lloyd_iterations=2,
        balance_tolerance=0.15,
        warp_strength=0.12,
        warp_octaves=4,
    )

    assert grid.tessellation_mode == "warped"
    classified = grid.classify(points)
    counts = np.bincount(classified, minlength=12)
    assert np.array_equal(counts, grid.tiles.point_count.to_numpy(dtype=np.int64))
    assert all(not geom.is_empty for geom in grid.tiles.geometry)
    assert grid._levels_projected[0].geometry.is_valid.all()

    union = grid._levels_projected[0].geometry.union_all()
    relative_missing = grid.boundary_projected.difference(union).area / grid.boundary_projected.area
    relative_extra = union.difference(grid.boundary_projected).area / grid.boundary_projected.area
    assert relative_missing < 1e-7
    assert relative_extra < 1e-10


def test_warped_mode_save_load_preserves_classification(tmp_path):
    rng = np.random.default_rng(41)
    points = rng.uniform([10.0, 56.0], [10.2, 56.2], size=(300, 2))
    grid = tessellate(
        points=points,
        boundary=box(9.99, 55.99, 10.21, 56.21),
        n_tiles=6,
        mode="warped",
        projected_crs="EPSG:32632",
        random_state=13,
        lloyd_iterations=2,
        balance_tolerance=0.20,
        warp_strength=0.14,
    )
    grid.build_hierarchy(branching_factor=3, levels=1)
    expected = grid.classify(points[:80], include_hierarchy=True)

    path = tmp_path / "warped.aggrid"
    grid.save(path)
    loaded = load_grid(path)

    assert loaded.tessellation_mode == "warped"
    assert np.array_equal(loaded.classify(points[:80], include_hierarchy=True), expected)


def test_tessellation_mode_validation():
    with np.testing.assert_raises_regex(ValueError, "Unsupported tessellation mode"):
        tessellate(
            points=[(10.0, 56.0), (10.1, 56.1)],
            boundary=box(9.9, 55.9, 10.2, 56.2),
            n_tiles=1,
            mode="future_magic",
            projected_crs="EPSG:32632",
        )


def test_geodesic_mode_balances_and_covers_boundary():
    rng = np.random.default_rng(51)
    points = rng.uniform([10.0, 56.0], [10.2, 56.2], size=(600, 2))
    boundary = box(9.99, 55.99, 10.21, 56.21)

    grid = tessellate(
        points=points,
        boundary=boundary,
        n_tiles=12,
        mode="geodesic",
        projected_crs="EPSG:32632",
        random_state=17,
        lloyd_iterations=2,
        balance_tolerance=0.15,
        geodesic_strength=0.35,
        geodesic_octaves=4,
        geodesic_grid_size=120,
        geodesic_simplify=1.0,
    )

    assert grid.tessellation_mode == "geodesic"
    assert grid._geodesic_metric is not None
    assert grid._geodesic_metric.labels.shape == grid._geodesic_metric.shape
    assert np.nanmin(grid._geodesic_metric.cost) > 0

    classified = grid.classify(points)
    counts = np.bincount(classified, minlength=12)
    assert np.array_equal(counts, grid.tiles.point_count.to_numpy(dtype=np.int64))
    assert np.max(np.abs(counts - 50)) <= 10

    assert all(not geom.is_empty for geom in grid.tiles.geometry)
    assert grid._levels_projected[0].geometry.is_valid.all()
    union = grid._levels_projected[0].geometry.union_all()
    relative_missing = grid.boundary_projected.difference(union).area / grid.boundary_projected.area
    relative_extra = union.difference(grid.boundary_projected).area / grid.boundary_projected.area
    assert relative_missing < 1e-7
    assert relative_extra < 1e-7


def test_geodesic_mode_hierarchy_and_save_load(tmp_path):
    rng = np.random.default_rng(61)
    points = rng.uniform([10.0, 56.0], [10.2, 56.2], size=(360, 2))
    grid = tessellate(
        points=points,
        boundary=box(9.99, 55.99, 10.21, 56.21),
        n_tiles=8,
        mode="geodesic",
        projected_crs="EPSG:32632",
        random_state=19,
        lloyd_iterations=2,
        balance_tolerance=0.20,
        geodesic_strength=0.30,
        geodesic_grid_size=100,
    )
    grid.build_hierarchy(branching_factor=2, levels=2)
    expected = grid.classify(points[:80], include_hierarchy=True)

    path = tmp_path / "geodesic.aggrid"
    grid.save(path)
    loaded = load_grid(path)

    assert loaded.tessellation_mode == "geodesic"
    assert loaded._geodesic_metric is not None
    assert loaded._geodesic_metric.shape == grid._geodesic_metric.shape
    assert np.array_equal(loaded._geodesic_metric.labels, grid._geodesic_metric.labels)
    assert np.array_equal(loaded.classify(points[:80], include_hierarchy=True), expected)


def test_geodesic_zero_strength_is_positive_uniform_metric():
    rng = np.random.default_rng(71)
    points = rng.uniform([10.0, 56.0], [10.1, 56.1], size=(180, 2))
    grid = tessellate(
        points=points,
        boundary=box(9.99, 55.99, 10.11, 56.11),
        n_tiles=4,
        mode="geodesic",
        projected_crs="EPSG:32632",
        random_state=3,
        lloyd_iterations=1,
        balance_tolerance=0.25,
        geodesic_strength=0.0,
        geodesic_grid_size=80,
        geodesic_simplify=0.0,
    )
    valid_cost = grid._geodesic_metric.cost[grid._geodesic_metric.valid_mask]
    assert np.allclose(valid_cost, 1.0)


def test_graph_mode_balances_covers_and_classifies_consistently():
    rng = np.random.default_rng(81)
    points = rng.uniform([10.0, 56.0], [10.2, 56.2], size=(600, 2))
    boundary = box(9.99, 55.99, 10.21, 56.21)

    grid = tessellate(
        points=points,
        boundary=boundary,
        n_tiles=12,
        mode="graph",
        projected_crs="EPSG:32632",
        random_state=23,
        lloyd_iterations=2,
        balance_tolerance=0.15,
        graph_compactness=1.0,
        graph_boundary_weight=0.35,
        graph_organic_strength=0.20,
        graph_refine_iterations=5,
        graph_simplify=0.0,
    )

    assert grid.tessellation_mode == "graph"
    assert grid._graph_state is not None
    classified = grid.classify(points)
    counts = np.bincount(classified, minlength=12)
    assert np.array_equal(counts, grid.tiles.point_count.to_numpy(dtype=np.int64))
    assert np.max(np.abs(counts - 50)) <= 10

    assert all(not geom.is_empty for geom in grid.tiles.geometry)
    assert grid._levels_projected[0].geometry.is_valid.all()
    union = grid._levels_projected[0].geometry.union_all()
    relative_missing = grid.boundary_projected.difference(union).area / grid.boundary_projected.area
    relative_extra = union.difference(grid.boundary_projected).area / grid.boundary_projected.area
    assert relative_missing < 1e-9
    assert relative_extra < 1e-9


def test_graph_mode_hierarchy_and_save_load(tmp_path):
    rng = np.random.default_rng(91)
    points = rng.uniform([10.0, 56.0], [10.2, 56.2], size=(360, 2))
    grid = tessellate(
        points=points,
        boundary=box(9.99, 55.99, 10.21, 56.21),
        n_tiles=8,
        mode="graph",
        projected_crs="EPSG:32632",
        random_state=29,
        lloyd_iterations=2,
        balance_tolerance=0.20,
        graph_refine_iterations=4,
    )
    grid.build_hierarchy(branching_factor=2, levels=2)
    expected = grid.classify(points[:80], include_hierarchy=True)

    path = tmp_path / "graph.aggrid"
    grid.save(path)
    loaded = load_grid(path)

    assert loaded.tessellation_mode == "graph"
    assert loaded._graph_state is not None
    assert np.array_equal(
        loaded._graph_state.generator_labels,
        grid._graph_state.generator_labels,
    )
    assert np.allclose(
        loaded._graph_state.generator_points,
        grid._graph_state.generator_points,
    )
    assert np.array_equal(loaded.classify(points[:80], include_hierarchy=True), expected)


def test_geodesic_disconnected_components_need_no_euclidean_fallback_warning():
    import warnings
    from shapely.geometry import MultiPolygon

    boundary = MultiPolygon([
        box(10.00, 56.00, 10.04, 56.04),
        box(10.08, 56.00, 10.12, 56.04),
        box(10.16, 56.00, 10.20, 56.04),
    ])
    rng = np.random.default_rng(111)
    chunks = []
    for minx in (10.00, 10.08, 10.16):
        chunks.append(rng.uniform([minx + 0.002, 56.002], [minx + 0.038, 56.038], size=(80, 2)))
    points = np.vstack(chunks)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        grid = tessellate(
            points=points,
            boundary=boundary,
            n_tiles=2,
            mode="geodesic",
            projected_crs="EPSG:32632",
            random_state=31,
            lloyd_iterations=1,
            balance_tolerance=0.30,
            geodesic_grid_size=120,
            geodesic_simplify=0.0,
        )

    assert not any("contain no site" in str(item.message) for item in caught)
    assert np.all(grid.classify(points) >= 0)
