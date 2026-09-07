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
        hierarchy_band_width=1.3,
        hierarchy_separator_width=0.45,
    )
    output = tmp_path / "banded_grid_with_points.png"
    ax.figure.savefig(output, dpi=140)
    assert output.exists()
    assert output.stat().st_size > 0

