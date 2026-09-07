from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Iterable, Sequence
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS

from ._power import power_assign
from ._utils import as_coordinate_array, boundary_mask, transform_xy
from .hierarchy import aggregate_level


_BUNDLE_FORMAT = "adaptive-geogrid"
_BUNDLE_FORMAT_VERSION = 1


class AdaptiveGeoGrid:
    """A fitted fine-level power grid plus optional bottom-up hierarchy.

    Level 0 is always the finest level. Increasing level numbers are progressively
    coarser. Tile IDs are local to a level, so a hierarchical class is represented
    by ``(level, tile_id)`` or by a complete fine-to-coarse class path.
    """

    def __init__(
        self,
        *,
        fine_tiles_projected: gpd.GeoDataFrame,
        fine_tiles_wgs84: gpd.GeoDataFrame,
        boundary_wgs84,
        boundary_projected,
        projected_crs,
        sites_projected: np.ndarray,
        weights_projected: np.ndarray,
        training_point_count: int,
    ):
        self._levels_projected = [fine_tiles_projected.reset_index(drop=True)]
        self._levels_wgs84 = [fine_tiles_wgs84.reset_index(drop=True)]
        self._parent_maps: list[dict[int, int]] = []
        self.boundary_wgs84 = boundary_wgs84
        self.boundary_projected = boundary_projected
        self.projected_crs = CRS.from_user_input(projected_crs)
        self.sites_projected = np.asarray(sites_projected, dtype=float)
        self.weights_projected = np.asarray(weights_projected, dtype=float)
        self.training_point_count = int(training_point_count)

    @property
    def tiles(self) -> gpd.GeoDataFrame:
        """Finest-level tiles in WGS84."""
        return self._levels_wgs84[0].copy()

    @property
    def n_levels(self) -> int:
        """Number of hierarchy levels, including the fine level."""
        return len(self._levels_wgs84)

    @property
    def n_classes(self) -> tuple[int, ...]:
        """Number of classes at each level, ordered fine -> coarse."""
        return tuple(len(gdf) for gdf in self._levels_wgs84)

    def level(self, level: int = 0) -> gpd.GeoDataFrame:
        """Return one hierarchy level in WGS84; level 0 is the finest."""
        return self._levels_wgs84[level].copy()

    def class_paths(
        self,
        fine_tile_ids: Iterable[int] | np.ndarray,
        *,
        outside_value: int = -1,
    ) -> np.ndarray:
        """Map fine-tile IDs to complete fine-to-coarse class paths.

        Parameters
        ----------
        fine_tile_ids:
            Fine-level tile IDs, typically model predictions or labels produced
            during preprocessing. ``outside_value`` is propagated through all
            levels.

        Returns
        -------
        np.ndarray
            Integer array with shape ``(N, n_levels)``. Column 0 is the fine
            class and subsequent columns contain its ancestors.
        """
        ids = np.asarray(fine_tile_ids, dtype=np.int64)
        if ids.ndim == 0:
            ids = ids.reshape(1)
        if ids.ndim != 1:
            raise ValueError("fine_tile_ids must be a one-dimensional sequence")

        valid = ids != outside_value
        if np.any(valid):
            n_fine = len(self._levels_wgs84[0])
            bad = (ids[valid] < 0) | (ids[valid] >= n_fine)
            if np.any(bad):
                invalid = int(ids[valid][np.flatnonzero(bad)[0]])
                raise ValueError(f"Invalid fine tile ID {invalid}; expected 0 <= id < {n_fine}")

        paths = np.full((len(ids), self.n_levels), outside_value, dtype=np.int64)
        paths[:, 0] = ids
        current = ids.copy()

        for level_idx, mapping in enumerate(self._parent_maps, start=1):
            parent = np.full(len(current), outside_value, dtype=np.int64)
            current_valid = current != outside_value
            if np.any(current_valid):
                parent[current_valid] = np.fromiter(
                    (mapping[int(v)] for v in current[current_valid]),
                    dtype=np.int64,
                    count=int(np.sum(current_valid)),
                )
            paths[:, level_idx] = parent
            current = parent

        return paths

    def class_path(self, fine_tile_id: int, *, outside_value: int = -1) -> np.ndarray:
        """Return the fine-to-coarse class vector for one fine tile ID."""
        return self.class_paths([fine_tile_id], outside_value=outside_value)[0]

    def hierarchy_table(self) -> pd.DataFrame:
        """Return the class path of every fine tile.

        This table can be cached with a training dataset. If a model predicts a
        fine class ``c``, ``hierarchy_table().iloc[c]`` gives every upper-level
        class corresponding to that prediction.
        """
        fine_ids = self._levels_wgs84[0]["tile_id"].to_numpy(dtype=np.int64)
        paths = self.class_paths(fine_ids)
        return pd.DataFrame(paths, columns=[f"level_{i}" for i in range(self.n_levels)])

    def classify(
        self,
        points: Iterable[tuple[float, float]] | np.ndarray,
        *,
        include_hierarchy: bool = False,
        as_dataframe: bool = False,
        outside_value: int = -1,
        chunk_size: int = 20_000,
    ):
        """Classify WGS84 ``(longitude, latitude)`` points into the grid.

        By default this returns a one-dimensional fine-class array of shape
        ``(N,)``. With ``include_hierarchy=True`` it returns an integer class
        matrix of shape ``(N, n_levels)``, ordered from fine to coarse. Set
        ``as_dataframe=True`` for named ``level_0``, ``level_1``, ... columns.
        """
        lonlat = as_coordinate_array(points)
        inside = boundary_mask(self.boundary_wgs84, lonlat)
        fine = np.full(len(lonlat), outside_value, dtype=np.int64)

        if np.any(inside):
            projected = transform_xy(lonlat[inside], "EPSG:4326", self.projected_crs)
            fine[inside] = power_assign(
                projected,
                self.sites_projected,
                self.weights_projected,
                chunk_size=chunk_size,
            )

        result = self.class_paths(fine, outside_value=outside_value) if include_hierarchy else fine

        if as_dataframe:
            if result.ndim == 1:
                return pd.DataFrame({"level_0": result})
            return pd.DataFrame(result, columns=[f"level_{i}" for i in range(self.n_levels)])
        return result

    def build_hierarchy(
        self,
        *,
        branching_factor: int | Sequence[int] = 8,
        levels: int = 1,
    ) -> "AdaptiveGeoGrid":
        """Build coarser levels bottom-up by merging adjacent polygons.

        ``branching_factor=8, levels=3`` aims to reduce tile count by about 8x at
        each of three successive levels. A sequence such as ``[4, 8, 8]`` uses a
        different branching factor for each level.
        """
        if levels < 1:
            raise ValueError("levels must be >= 1")

        if isinstance(branching_factor, int):
            factors = [branching_factor] * levels
        else:
            factors = list(branching_factor)
            if levels != 1 and levels != len(factors):
                raise ValueError("When branching_factor is a sequence, omit levels or make it match")

        if not factors:
            raise ValueError("branching_factor sequence cannot be empty")
        if any(int(factor) < 2 for factor in factors):
            raise ValueError("Every branching factor must be >= 2")

        # Rebuild hierarchy from the fine level each time, avoiding stale state.
        fine_projected = self._levels_projected[0].drop(columns=["parent_id"], errors="ignore").copy()
        fine_wgs84 = fine_projected.to_crs("EPSG:4326")
        self._levels_projected = [fine_projected]
        self._levels_wgs84 = [fine_wgs84]
        self._parent_maps = []

        current = self._levels_projected[0].copy()
        for factor in factors:
            if len(current) <= 1:
                break
            parent, mapping = aggregate_level(current, branching_factor=int(factor))
            self._parent_maps.append(mapping)

            # Annotate the child level with its parent id. Parent IDs are local
            # to the next level, matching class_paths/classify output.
            child = self._levels_projected[-1].copy()
            child["parent_id"] = child["tile_id"].map(mapping).astype("Int64")
            self._levels_projected[-1] = child
            self._levels_wgs84[-1] = child.to_crs("EPSG:4326")

            parent["level"] = len(self._levels_projected)
            self._levels_projected.append(parent)
            self._levels_wgs84.append(parent.to_crs("EPSG:4326"))
            current = parent

        if len(self._levels_projected) > 1:
            top = self._levels_projected[-1].copy()
            top["parent_id"] = pd.Series([pd.NA] * len(top), dtype="Int64")
            self._levels_projected[-1] = top
            self._levels_wgs84[-1] = top.to_crs("EPSG:4326")
        return self

    def _combined_levels(self) -> gpd.GeoDataFrame:
        frames = []
        for level_idx, gdf in enumerate(self._levels_wgs84):
            frame = gdf.copy()
            frame["level"] = level_idx
            frames.append(frame)
        return gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True, sort=False),
            geometry="geometry",
            crs="EPSG:4326",
        )

    def to_geojson(self, path: str | Path, *, level: int | None = None) -> None:
        """Write the grid to GeoJSON in WGS84.

        By default all hierarchy levels are written into one FeatureCollection;
        each feature carries a ``level`` property. Pass ``level=0`` (or another
        level number) to export only that level.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        gdf = self._combined_levels() if level is None else self._levels_wgs84[level]
        gdf.to_file(path, driver="GeoJSON")

    def _write_bundle_directory(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        metadata = {
            "format": _BUNDLE_FORMAT,
            "format_version": _BUNDLE_FORMAT_VERSION,
            "projected_crs_wkt": self.projected_crs.to_wkt(),
            "training_point_count": self.training_point_count,
            "n_levels": self.n_levels,
        }
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        np.savez_compressed(
            path / "state.npz",
            sites_projected=self.sites_projected,
            weights_projected=self.weights_projected,
        )

        boundary = gpd.GeoDataFrame({"geometry": [self.boundary_wgs84]}, crs="EPSG:4326")
        boundary.to_file(path / "boundary.geojson", driver="GeoJSON")

        for i, gdf in enumerate(self._levels_wgs84):
            gdf.to_file(path / f"level_{i}.geojson", driver="GeoJSON")
        for i, mapping in enumerate(self._parent_maps):
            pd.DataFrame(
                {"child_id": list(mapping.keys()), "parent_id": list(mapping.values())}
            ).to_csv(path / f"level_{i}_to_{i + 1}.csv", index=False)

    def to_directory(self, path: str | Path) -> None:
        """Write a complete, loadable grid directory.

        Besides GeoJSON levels and mapping CSV files, the directory contains the
        fitted power-diagram state required for fast classification. Reload it
        with :meth:`AdaptiveGeoGrid.load` or ``load_grid(...)``.
        """
        self._write_bundle_directory(Path(path))

    def save(self, path: str | Path) -> None:
        """Save the complete grid as one portable ``.aggrid`` ZIP archive."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="adaptive_geogrid_") as tmp:
            bundle_dir = Path(tmp) / "grid"
            self._write_bundle_directory(bundle_dir)
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for file in sorted(bundle_dir.rglob("*")):
                    if file.is_file():
                        archive.write(file, arcname=file.relative_to(bundle_dir))

    @classmethod
    def _load_directory(cls, path: Path) -> "AdaptiveGeoGrid":
        metadata_path = path / "metadata.json"
        state_path = path / "state.npz"
        boundary_path = path / "boundary.geojson"
        if not metadata_path.exists() or not state_path.exists() or not boundary_path.exists():
            raise ValueError(
                "Not a complete adaptive-geogrid bundle: expected metadata.json, state.npz, "
                "and boundary.geojson"
            )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format") != _BUNDLE_FORMAT:
            raise ValueError(f"Unsupported grid bundle format: {metadata.get('format')!r}")
        if int(metadata.get("format_version", -1)) != _BUNDLE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported grid bundle version {metadata.get('format_version')!r}; "
                f"expected {_BUNDLE_FORMAT_VERSION}"
            )

        n_levels = int(metadata["n_levels"])
        if n_levels < 1:
            raise ValueError("Serialized grid contains no levels")

        crs = CRS.from_wkt(metadata["projected_crs_wkt"])
        levels_wgs84 = []
        for i in range(n_levels):
            level_path = path / f"level_{i}.geojson"
            if not level_path.exists():
                raise ValueError(f"Serialized grid is missing {level_path.name}")
            gdf = gpd.read_file(level_path).to_crs("EPSG:4326").reset_index(drop=True)
            levels_wgs84.append(gdf)

        boundary_gdf = gpd.read_file(boundary_path).to_crs("EPSG:4326")
        boundary_wgs84 = boundary_gdf.geometry.union_all()
        boundary_projected = boundary_gdf.to_crs(crs).geometry.union_all()

        with np.load(state_path) as state:
            sites = np.asarray(state["sites_projected"], dtype=float)
            weights = np.asarray(state["weights_projected"], dtype=float)

        grid = cls(
            fine_tiles_projected=levels_wgs84[0].to_crs(crs),
            fine_tiles_wgs84=levels_wgs84[0],
            boundary_wgs84=boundary_wgs84,
            boundary_projected=boundary_projected,
            projected_crs=crs,
            sites_projected=sites,
            weights_projected=weights,
            training_point_count=int(metadata["training_point_count"]),
        )
        grid._levels_wgs84 = levels_wgs84
        grid._levels_projected = [gdf.to_crs(crs) for gdf in levels_wgs84]
        grid._parent_maps = []

        for i in range(n_levels - 1):
            mapping_path = path / f"level_{i}_to_{i + 1}.csv"
            if mapping_path.exists():
                mapping_df = pd.read_csv(mapping_path)
                mapping = {
                    int(child): int(parent)
                    for child, parent in zip(mapping_df["child_id"], mapping_df["parent_id"])
                }
            elif "parent_id" in levels_wgs84[i].columns:
                mapping = {
                    int(child): int(parent)
                    for child, parent in zip(
                        levels_wgs84[i]["tile_id"], levels_wgs84[i]["parent_id"]
                    )
                    if pd.notna(parent)
                }
            else:
                raise ValueError(f"Serialized grid is missing hierarchy mapping for level {i}")
            grid._parent_maps.append(mapping)

        return grid

    @classmethod
    def load(cls, path: str | Path) -> "AdaptiveGeoGrid":
        """Load a grid saved by :meth:`save` or :meth:`to_directory`."""
        path = Path(path)
        if path.is_dir():
            return cls._load_directory(path)
        if not path.is_file():
            raise FileNotFoundError(path)

        with tempfile.TemporaryDirectory(prefix="adaptive_geogrid_load_") as tmp:
            tmp_path = Path(tmp)
            with zipfile.ZipFile(path, "r") as archive:
                for member in archive.infolist():
                    member_path = Path(member.filename)
                    if member_path.is_absolute() or ".." in member_path.parts:
                        raise ValueError("Unsafe path found in grid archive")
                archive.extractall(tmp_path)
            return cls._load_directory(tmp_path)

    def _internal_boundary_projected(self, level: int):
        """Return only internal tile borders for one level in the metric CRS.

        Polygon boundaries also contain the outer dataset boundary. For hierarchy
        visualization we want only boundaries that separate two classes; the outer
        boundary is rendered separately by :meth:`plot`.
        """
        linework = self._levels_projected[level].boundary.union_all()
        bounds = self._levels_projected[0].total_bounds
        scale = max(
            float(np.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1])),
            1.0,
        )
        tolerance = max(scale * 1e-8, 1e-6)
        return linework.difference(self.boundary_projected.boundary.buffer(tolerance))

    def _hierarchy_boundaries_projected(self, levels: Sequence[int]):
        """Return internal boundary linework for each selected hierarchy level."""
        return {
            int(level): self._internal_boundary_projected(int(level))
            for level in levels
        }

    def _auto_hierarchy_lane_width_projected(self) -> float:
        """Choose a visible hierarchy-ribbon width in the metric CRS.

        The width is tied primarily to a typical fine-cell scale, but it is also
        prevented from becoming vanishingly small relative to the full map.  The
        result is only a plotting heuristic; callers can override it through
        ``hierarchy_lane_width`` in :meth:`plot`.
        """
        areas = self._levels_projected[0].geometry.area.to_numpy(dtype=float)
        areas = areas[np.isfinite(areas) & (areas > 0)]

        bounds = self._levels_projected[0].total_bounds
        map_diag = max(
            float(np.hypot(bounds[2] - bounds[0], bounds[3] - bounds[1])),
            1e-9,
        )

        if len(areas):
            fine_scale = float(np.median(np.sqrt(areas)))
        else:
            fine_scale = map_diag / 20.0

        # Keep hierarchy lanes deliberately thin. They only need to remain
        # individually distinguishable; they should not dominate the map.
        candidate = max(fine_scale * 0.035, map_diag * 0.00045)
        return max(min(candidate, fine_scale * 0.08), 1e-9)

    def _hierarchy_lane_bands_projected(
        self,
        *,
        level: int,
        offset: float,
        width: float,
    ) -> gpd.GeoSeries:
        """Return inward boundary ribbons for all polygons at one level.

        Every hierarchy level gets a *different geometric lane* instead of being
        painted on the same physical edge.  A ribbon is constructed as the strip
        between two negative buffers of each parent polygon.  Portions along the
        outer dataset boundary are removed, leaving only hierarchy-separating
        borders.
        """
        if offset < 0:
            raise ValueError("offset must be >= 0")
        if width <= 0:
            raise ValueError("width must be > 0")

        # Remove the ribbon where a parent border merely follows the outer dataset
        # boundary.  The small extra margin avoids hairline remnants from floating
        # point noise after buffering/difference operations.
        outer_clearance = offset + width * 1.15
        outer_exclusion = self.boundary_projected.boundary.buffer(outer_clearance)

        bands = []
        for geom in self._levels_projected[level].geometry:
            if geom is None or geom.is_empty:
                continue

            outer = geom.buffer(-offset) if offset > 0 else geom
            if outer.is_empty:
                continue

            inner = geom.buffer(-(offset + width))
            band = outer if inner.is_empty else outer.difference(inner)
            band = band.difference(outer_exclusion)
            if not band.is_empty:
                bands.append(band)

        return gpd.GeoSeries(bands, crs=self.projected_crs)

    def plot(
        self,
        *,
        levels: int | Sequence[int] | None = None,
        ax=None,
        figsize: tuple[float, float] = (9.0, 9.0),
        linewidth: float = 0.55,
        hierarchy_lane_width: float | None = None,
        hierarchy_lane_gap: float = 0.0,
        hierarchy_lane_alpha: float = 0.82,
        colors: Sequence[str] | None = None,
        show_boundary: bool = True,
        boundary_linewidth: float = 1.8,
        legend: bool = True,
        points: Iterable[tuple[float, float]] | np.ndarray | None = None,
        point_sample: int | float | None = None,
        point_size: float = 5.0,
        point_alpha: float = 0.22,
        point_color: str = "black",
        point_random_state: int = 42,
        label_level: int | None = None,
        label_fontsize: float = 7.0,
    ):
        """Plot the hierarchy using spatially separated inward boundary lanes.

        The important design rule is that hierarchy levels are **not drawn on top
        of the same edge**.  The finest selected level is drawn exactly on its
        polygon boundaries.  Each successively coarser level is rendered as a
        colored ribbon slightly farther inside its parent polygons::

            exact child edge | L1 lane | L2 lane | L3 lane

        Therefore an edge shared by levels 0, 1, 2 and 3 visibly contains four
        independent visual signals.  An ordinary fine-only edge contains only the
        thin fine line.  Middle levels can no longer disappear underneath the
        finest or coarsest stroke.

        Parameters
        ----------
        levels:
            Level number or sequence of levels to draw. ``None`` draws all levels.
            The finest selected level is the exact thin boundary; every selected
            coarser level becomes an inward ribbon.
        linewidth:
            Width in points of the exact boundary for the finest selected level.
        hierarchy_lane_width:
            Ribbon width in units of the projected metric CRS. ``None`` chooses a
            value automatically from the fine-cell and map scale.
        hierarchy_lane_gap:
            Optional distance between neighboring hierarchy ribbons in projected
            CRS units. Defaults to ``0.0``, so the lanes touch with no empty gaps.
        hierarchy_lane_alpha:
            Opacity of hierarchy ribbons.
        points:
            Optional WGS84 ``(longitude, latitude)`` point cloud to overlay.
        point_sample:
            Optional deterministic subset of ``points``. An integer selects at
            most that many points; a float in ``(0, 1]`` selects that fraction.
        label_level:
            Optional level whose tile IDs are drawn at representative points.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Matplotlib is required for plotting. Install adaptive-geogrid[plot] "
                "or matplotlib>=3.8."
            ) from exc

        if levels is None:
            selected = list(range(self.n_levels))
        elif isinstance(levels, int):
            selected = [int(levels)]
        else:
            selected = [int(level) for level in levels]

        if not selected:
            raise ValueError("levels cannot be empty")
        if len(set(selected)) != len(selected):
            raise ValueError("levels cannot contain duplicates")
        if any(level < 0 or level >= self.n_levels for level in selected):
            raise ValueError(f"levels must be between 0 and {self.n_levels - 1}")
        if linewidth <= 0:
            raise ValueError("linewidth must be > 0")
        if not 0 <= hierarchy_lane_alpha <= 1:
            raise ValueError("hierarchy_lane_alpha must be in [0, 1]")
        if boundary_linewidth <= 0:
            raise ValueError("boundary_linewidth must be > 0")
        if not 0 <= point_alpha <= 1:
            raise ValueError("point_alpha must be in [0, 1]")

        selected = sorted(selected)
        base_level = selected[0]
        lane_levels = selected[1:]

        if hierarchy_lane_width is None:
            lane_width = self._auto_hierarchy_lane_width_projected()
        else:
            lane_width = float(hierarchy_lane_width)
            if lane_width <= 0:
                raise ValueError("hierarchy_lane_width must be > 0")

        lane_gap = float(hierarchy_lane_gap)
        if lane_gap < 0:
            raise ValueError("hierarchy_lane_gap must be >= 0")

        if label_level is not None:
            label_level = int(label_level)
            if label_level < 0 or label_level >= self.n_levels:
                raise ValueError(f"label_level must be between 0 and {self.n_levels - 1}")

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        # Distinct level colors. The exact base grid remains neutral so the colored
        # lanes read immediately as hierarchy rather than as arbitrary cell edges.
        if colors is None:
            lane_palette = [
                "tab:blue",
                "tab:orange",
                "tab:purple",
                "tab:green",
                "tab:red",
                "tab:brown",
                "tab:pink",
                "tab:olive",
                "tab:cyan",
            ]
            level_colors = {base_level: "0.28"}
            for rank, level in enumerate(lane_levels):
                level_colors[level] = lane_palette[rank % len(lane_palette)]
        else:
            supplied = list(colors)
            if len(supplied) < len(selected):
                raise ValueError("colors must contain at least one color per selected level")
            level_colors = {
                level: supplied[i]
                for i, level in enumerate(selected)
            }

        handles = []

        # ------------------------------------------------------------------
        # 1) Spatially separated hierarchy lanes. L1 is nearest the exact edge,
        #    L2 immediately follows it, etc. By default the lanes touch directly,
        #    with no separator gap, while still occupying different geometry.
        # ------------------------------------------------------------------
        for rank, level in reversed(list(enumerate(lane_levels))):
            offset = lane_gap + rank * (lane_width + lane_gap)
            bands_projected = self._hierarchy_lane_bands_projected(
                level=level,
                offset=offset,
                width=lane_width,
            )
            if len(bands_projected):
                bands_projected.to_crs("EPSG:4326").plot(
                    ax=ax,
                    color=level_colors[level],
                    alpha=hierarchy_lane_alpha,
                    edgecolor="none",
                    zorder=10 + rank,
                )

        for level in lane_levels:
            handles.append(
                Patch(
                    facecolor=level_colors[level],
                    edgecolor="none",
                    alpha=hierarchy_lane_alpha,
                    label=f"Level {level} boundary lane",
                )
            )

        # ------------------------------------------------------------------
        # 2) Original point cloud above the broad colored ribbons but below the
        #    exact fine-grid linework.
        # ------------------------------------------------------------------
        if points is not None:
            point_array = as_coordinate_array(points)
            n_points = len(point_array)

            if point_sample is None:
                sampled = point_array
            else:
                if isinstance(point_sample, (float, np.floating)):
                    fraction = float(point_sample)
                    if not 0 < fraction <= 1:
                        raise ValueError("float point_sample must be in (0, 1]")
                    sample_count = max(1, int(round(n_points * fraction))) if n_points else 0
                elif isinstance(point_sample, (int, np.integer)):
                    sample_count = int(point_sample)
                    if sample_count <= 0:
                        raise ValueError("integer point_sample must be > 0")
                    sample_count = min(sample_count, n_points)
                else:
                    raise TypeError("point_sample must be an int, float, or None")

                if sample_count >= n_points:
                    sampled = point_array
                elif sample_count == 0:
                    sampled = point_array[:0]
                else:
                    rng = np.random.default_rng(point_random_state)
                    indices = rng.choice(n_points, size=sample_count, replace=False)
                    sampled = point_array[indices]

            if len(sampled):
                ax.scatter(
                    sampled[:, 0],
                    sampled[:, 1],
                    s=point_size,
                    alpha=point_alpha,
                    c=point_color,
                    linewidths=0,
                    zorder=30,
                )
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="None",
                        markerfacecolor=point_color,
                        markeredgewidth=0,
                        alpha=point_alpha,
                        markersize=max(3.0, float(np.sqrt(max(point_size, 0.1)))),
                        label=f"Points ({len(sampled):,}/{n_points:,})",
                    )
                )

        # ------------------------------------------------------------------
        # 3) Draw only the finest selected boundary on its true geometric edge.
        #    This is the reference line from which all hierarchy lanes are offset.
        # ------------------------------------------------------------------
        base_line = self._internal_boundary_projected(base_level)
        if not base_line.is_empty:
            gpd.GeoSeries([base_line], crs=self.projected_crs).to_crs("EPSG:4326").plot(
                ax=ax,
                color=level_colors[base_level],
                linewidth=linewidth,
                alpha=0.90,
                zorder=50,
            )
        handles.append(
            Line2D(
                [0],
                [0],
                color=level_colors[base_level],
                linewidth=max(linewidth, 1.5),
                label=f"Level {base_level} exact boundary",
            )
        )

        if label_level is not None:
            label_gdf = self._levels_wgs84[label_level]
            representatives = label_gdf.geometry.representative_point()
            for tile_id, point in zip(label_gdf["tile_id"], representatives):
                ax.text(
                    point.x,
                    point.y,
                    str(int(tile_id)),
                    ha="center",
                    va="center",
                    fontsize=label_fontsize,
                    color="black",
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.70,
                    },
                    zorder=80,
                )

        if show_boundary:
            boundary_series = gpd.GeoSeries([self.boundary_wgs84], crs="EPSG:4326")
            boundary_series.boundary.plot(
                ax=ax,
                color="black",
                linewidth=boundary_linewidth,
                zorder=100,
            )
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linewidth=boundary_linewidth,
                    label="Dataset boundary",
                )
            )

        if legend:
            ax.legend(handles=handles, loc="best", framealpha=0.92)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="datalim")
        return ax

    def summary(self) -> pd.DataFrame:
        rows = []
        for level, gdf in enumerate(self._levels_wgs84):
            counts = gdf["point_count"].to_numpy(dtype=float)
            rows.append(
                {
                    "level": level,
                    "tiles": len(gdf),
                    "min_points": int(np.min(counts)) if len(counts) else 0,
                    "mean_points": float(np.mean(counts)) if len(counts) else 0.0,
                    "max_points": int(np.max(counts)) if len(counts) else 0,
                }
            )
        return pd.DataFrame(rows)
