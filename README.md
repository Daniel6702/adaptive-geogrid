# adaptive-geogrid

`adaptive-geogrid` builds geographic cells whose size adapts to the density of input coordinates.

The fine grid is an approximate **capacity-constrained power/Laguerre tessellation**: dense areas get smaller polygons, sparse areas get larger polygons, and the optimizer aims to place approximately the same number of input points in each fine tile.

An optional **bottom-up hierarchy** merges adjacent child polygons into progressively coarser parent regions.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Generate a grid

```python
from adaptive_geogrid import tessellate

coordinates = [
    (10.2039, 56.1629),
    (10.1812, 56.1513),
    # ...
]

grid = tessellate(
    points=coordinates,
    boundary="aarhus_kommune.geojson",
    target_points=500,
)
```

Instead of `target_points`, specify the desired number of fine cells:

```python
grid = tessellate(
    points=coordinates,
    boundary="aarhus_kommune.geojson",
    n_tiles=100,
)
```

Exactly one of `target_points` and `n_tiles` is required.

## Bottom-up hierarchy

Level `0` is always the finest grid. Larger level numbers are progressively coarser.

```python
grid.build_hierarchy(
    branching_factor=8,
    levels=3,
)

print(grid.n_classes)
# e.g. (4096, 512, 64, 8)
```

Parents are formed only by merging adjacent child polygons. Every parent geometry is therefore an exact union of its children.

## Classification

Fine-only classification returns a one-dimensional NumPy array:

```python
fine_ids = grid.classify(coordinates)
# shape: (N,)
```

Hierarchical classification returns one **class vector per point**, ordered fine -> coarse:

```python
labels = grid.classify(
    coordinates,
    include_hierarchy=True,
)
# shape: (N, n_levels)
# row example: [183, 21, 3, 0]
```

For named columns:

```python
labels_df = grid.classify(
    coordinates,
    include_hierarchy=True,
    as_dataframe=True,
)
# level_0, level_1, level_2, ...
```

Coordinates outside the boundary receive `-1` at every level by default.

### From a predicted fine class to all ancestors

This is useful during hierarchical model training/evaluation when the model outputs a fine class rather than a coordinate:

```python
path = grid.class_path(183)
# [183, 21, 3, 0]
```

Batch form:

```python
paths = grid.class_paths([183, 917, 42])
```

A complete fine-class lookup table can also be materialized once:

```python
lookup = grid.hierarchy_table()
```

Thus both common model outputs are supported:

```python
# Model predicts coordinate:
path_from_coordinate = grid.classify(
    [(pred_lon, pred_lat)],
    include_hierarchy=True,
)[0]

# Model predicts lowest-level class:
path_from_fine_class = grid.class_path(predicted_fine_class)
```

## Save and load for preprocessing/training

GeoJSON alone is intended for GIS/visualization. For later classification, save the complete fitted grid, including the power sites/weights and hierarchy mappings:

```python
grid.save("aarhus_grid.aggrid")
```

Load it in another process:

```python
from adaptive_geogrid import load_grid

grid = load_grid("aarhus_grid.aggrid")
labels = grid.classify(batch_coordinates, include_hierarchy=True)
```

`AdaptiveGeoGrid.load(...)` is equivalent.

A directory representation is also supported:

```python
grid.to_directory("aarhus_grid")
grid = load_grid("aarhus_grid")
```

The directory contains the GeoJSON layers, hierarchy mapping CSV files, boundary, metadata, and fitted classification state.

## GeoJSON export

After building a hierarchy:

```python
grid.to_geojson("grid.geojson")
```

writes **all hierarchy levels into one GeoJSON FeatureCollection**. Every feature contains its `level` and `tile_id`, and non-top levels contain `parent_id`.

Export one level only with:

```python
grid.to_geojson("fine.geojson", level=0)
grid.to_geojson("coarse.geojson", level=2)
```

IDs are local to each level, so `(level, tile_id)` uniquely identifies a hierarchical cell.

## Visualization

Plot polygon outlines for every hierarchy level:

```python
import matplotlib.pyplot as plt

ax = grid.plot()
ax.set_title("Aarhus adaptive grid")
plt.show()
```

Or selected levels:

```python
grid.plot(levels=[0, 2, 3])
```

Coarser layers are drawn with progressively thicker outlines.

## Aarhus example

`examples/aarhus_example.py` expects:

```text
aarhus_kommune.geojson
```

in the working directory. It generates synthetic demo points inside the municipality solely so the example can run without a separate point dataset; replace them with your real coordinate list.

```bash
python examples/aarhus_example.py
```

## Current scope

- Inputs/outputs use WGS84 `(longitude, latitude)` coordinates.
- Geometry optimization is done in a projected metric CRS. A local UTM CRS is estimated automatically; pass `projected_crs=` for very large or unusual extents.
- Capacity balancing is approximate for discrete empirical points. `balance_tolerance` controls the desired relative count error.
- Fine cells are power/Laguerre cells rather than ordinary Voronoi cells; their weighted boundaries are what allow density balancing.
- Hierarchy generation is bottom-up and adjacency constrained.

## License

MIT.
