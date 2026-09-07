"""adaptive-geogrid public API."""

from pathlib import Path

from .core import tessellate
from .grid import AdaptiveGeoGrid


def load_grid(path: str | Path) -> AdaptiveGeoGrid:
    """Load a previously saved adaptive-geogrid bundle."""
    return AdaptiveGeoGrid.load(path)


__all__ = ["AdaptiveGeoGrid", "load_grid", "tessellate"]
__version__ = "0.2.0"
