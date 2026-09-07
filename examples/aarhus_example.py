"""End-to-end example using ``aarhus_kommune.geojson``.

Put the boundary GeoJSON in the working directory. ``make_demo_points`` only
exists so the example can run without a separate coordinate dataset; replace it
with your real ``[(lon, lat), ...]`` list.
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely import contains_xy

from adaptive_geogrid import load_grid, tessellate

BOUNDARY = Path("examples/aarhus_kommune.geojson")


def make_demo_points(boundary_path: Path, n: int = 12_800, seed: int = 42):
    area = gpd.read_file(boundary_path).to_crs("EPSG:4326").geometry.union_all()
    minx, miny, maxx, maxy = area.bounds
    cx, cy = area.centroid.x, area.centroid.y
    rng = np.random.default_rng(seed)

    result = []
    while len(result) < n:
        batch = max(2_000, n - len(result))
        dense_count = int(batch * 0.70)
        sparse_count = batch - dense_count

        dense_x = rng.normal(cx, max((maxx - minx) * 0.12, 1e-6), dense_count)
        dense_y = rng.normal(cy, max((maxy - miny) * 0.12, 1e-6), dense_count)
        sparse_x = rng.uniform(minx, maxx, sparse_count)
        sparse_y = rng.uniform(miny, maxy, sparse_count)
        x = np.concatenate([dense_x, sparse_x])
        y = np.concatenate([dense_y, sparse_y])

        mask = contains_xy(area, x, y)
        result.extend(zip(x[mask], y[mask]))
    return result[:n]


# Real input can simply be:
# coordinates = [(10.2039, 56.1629), (10.1812, 56.1513), ...]
coordinates = make_demo_points(BOUNDARY)

grid = tessellate(
    points=coordinates,
    boundary=BOUNDARY,
    target_points=50,
    verbose=True,
)

# Bottom-up hierarchy: roughly 4 children per parent, three times.
grid.build_hierarchy(branching_factor=4, levels=4)
print(grid.summary())
print("Classes per level:", grid.n_classes)

# One GeoJSON containing every level. Use level=0 to export only fine cells.
grid.to_geojson("aarhus_all_levels.geojson")

# Complete classification bundle for preprocessing/training-time reuse.
grid.save("aarhus_grid.aggrid")

# Classification of coordinates returns one class vector per coordinate:
# [fine_class, parent_class, grandparent_class, ...]
query_points = [(10.2039, 56.1629), (10.2107, 56.1572)]
class_vectors = grid.classify(query_points, include_hierarchy=True)
print("Coordinate class vectors:\n", class_vectors)

# If a model predicts only a fine class, retrieve all upper-level classes cheaply.
predicted_fine_class = int(class_vectors[0, 0])
print("Predicted class path:", grid.class_path(predicted_fine_class))

# Load later in a dataset preprocessing script or training process.
loaded_grid = load_grid("aarhus_grid.aggrid")
assert np.array_equal(
    loaded_grid.classify(query_points, include_hierarchy=True),
    class_vectors,
)

# Grid Visualization
ax = grid.plot(
    points=coordinates,
    point_sample=2_000,
    linewidth=0.50,
    hierarchy_lane_alpha=0.82,
    hierarchy_lane_gap=0.0,
    point_size=5.0,
    point_alpha=0.20,
)
ax.set_title("Grid Visual")
plt.tight_layout()
plt.show()