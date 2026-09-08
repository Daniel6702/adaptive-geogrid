# adaptive-geogrid

`adaptive-geogrid` turns WGS84 `(longitude, latitude)` points into density-adaptive, capacity-balanced geographic cells. Dense areas receive smaller cells; sparse areas receive larger ones. Optionally, cells can be merged into an adjacency-constrained hierarchy.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install plotting support with `pip install -e '.[plot]'`.

## Quick start

See [`examples/aarhus_example.py`](examples/aarhus_example.py) for a complete workflow: load points, build a grid and hierarchy, export GeoJSON, save the fitted classifier, and plot the result.

```bash
python examples/aarhus_example.py
```

The example uses `examples/aarhus_kommune.geojson` and `examples/points.csv`. Replace the CSV with your own coordinates.

## Build a grid

```python
from adaptive_geogrid import tessellate

grid = tessellate(
    points=coordinates,  # [(longitude, latitude), ...]
    boundary="boundary.geojson",
    target_points=500,
)
```

`boundary` accepts a GeoJSON path, Shapely polygon, or GeoDataFrame. Points outside the boundary are ignored.

Choose exactly one capacity option:

| Option | Meaning |
| --- | --- |
| `target_points=500` | Approximate number of input points per fine cell |
| `n_tiles=100` | Exact requested number of fine cells |

### Tessellation modes

| Mode | Use when | Key options |
| --- | --- | --- |
| `power` | You want compact, straight-edged capacity-balanced cells. This is the default. | `lloyd_iterations`, `balance_tolerance` |
| `warped` | You want curved, organic power-cell borders. | `warp_strength=0.12`, `warp_octaves=4` |
| `geodesic` | You want organic boundaries with a geographic compactness bias. | `geodesic_strength=0.35`, `geodesic_grid_size`, `geodesic_simplify=1.0` |
| `graph` | You want compact connected cells without raster discretization. | `graph_compactness=1.0`, `graph_boundary_weight=0.35`, `graph_simplify=0.5` |

For example, create curved cells with the warped backend:

```python
grid = tessellate(
    points=coordinates,
    boundary="boundary.geojson",
    n_tiles=100,
    mode="warped",
    warp_strength=0.12,
    random_state=42,
)
```

`random_state` makes fitting reproducible. Pass `projected_crs=` to override the automatically selected local UTM CRS, and use `verbose=True` to print progress.

## Work with the grid

```python
# Merge adjacent fine cells into coarser levels.
grid.build_hierarchy(branching_factor=8, levels=3)

# Fine-cell IDs, or a fine-to-coarse ID path for each point.
fine_ids = grid.classify(query_points)
paths = grid.classify(query_points, include_hierarchy=True)

# Export all levels, persist the fitted classifier, and reload it later.
grid.to_geojson("grid.geojson")
grid.save("grid.aggrid")
```

```python
from adaptive_geogrid import load_grid

grid = load_grid("grid.aggrid")
ax = grid.plot()  # Requires the plot extra.
```

`level=0` selects the finest cells in `grid.to_geojson(...)`; without it, every hierarchy level is exported. `classify(..., include_hierarchy=True)` returns IDs ordered fine to coarse. Coordinates outside the boundary receive `-1`.

## Notes

- Input and output coordinates are WGS84 `(longitude, latitude)`.
- Capacity balancing is approximate because points are discrete; tune `balance_tolerance` when needed.
- A saved `.aggrid` bundle contains the fitted classification state. GeoJSON is intended for GIS interchange and visualization.

## License

MIT.
