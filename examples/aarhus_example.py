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
import csv
from adaptive_geogrid import load_grid, tessellate

BOUNDARY = Path("examples/aarhus_kommune.geojson")

coordinates = []
with open('examples/points.csv', mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    header = next(reader)
    for row in reader:
        coordinates.append((float(row[1]),float(row[0])))

grid = tessellate(
    points=coordinates,
    boundary=BOUNDARY,
    target_points=100,
    mode="geodesic",
    verbose=True
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