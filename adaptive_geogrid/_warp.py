from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from shapely import segmentize
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry


@dataclass(frozen=True)
class _ShearStep:
    """One analytically invertible sinusoidal shear in normalized coordinates."""

    axis: int
    amplitude: float
    wavelength: float
    phase: float
    angle: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "axis": int(self.axis),
            "amplitude": float(self.amplitude),
            "wavelength": float(self.wavelength),
            "phase": float(self.phase),
            "angle": float(self.angle),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_ShearStep":
        return cls(
            axis=int(data["axis"]),
            amplitude=float(data["amplitude"]),
            wavelength=float(data["wavelength"]),
            phase=float(data["phase"]),
            angle=float(data.get("angle", 0.0)),
        )


@dataclass(frozen=True)
class SinusoidalWarp:
    """Smooth, globally invertible 2-D warp used by warped Laguerre mode.

    The transform is a composition of alternating sinusoidal shears. Every
    individual shear has a closed-form inverse, so the complete mapping remains
    one-to-one even when the deformation is visually strong. Coordinates are
    normalized before deformation so the same parameters work across regions of
    different physical size.
    """

    origin_x: float
    origin_y: float
    scale: float
    angle: float
    steps: tuple[_ShearStep, ...]

    @property
    def origin(self) -> np.ndarray:
        return np.array([self.origin_x, self.origin_y], dtype=np.float64)

    @property
    def minimum_wavelength(self) -> float:
        return min((step.wavelength for step in self.steps), default=1.0)

    @property
    def geometry_segment_length(self) -> float:
        # Curved images of straight Laguerre edges are represented as polylines.
        # Twenty-four samples per shortest wavelength is smooth enough for
        # rendering while keeping large grids practical.
        return max(self.scale * self.minimum_wavelength / 24.0, self.scale * 1e-5, 1e-6)

    def _rotate(self, points: np.ndarray, angle: float) -> np.ndarray:
        c = math.cos(angle)
        s = math.sin(angle)
        x = points[:, 0].copy()
        y = points[:, 1].copy()
        points[:, 0] = c * x - s * y
        points[:, 1] = s * x + c * y
        return points

    def forward(self, points: np.ndarray) -> np.ndarray:
        """Map projected geographic coordinates into warped tessellation space."""
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")

        q = (arr - self.origin) / self.scale
        q = q.copy()
        self._rotate(q, self.angle)

        for step in self.steps:
            self._rotate(q, step.angle)
            frequency = 2.0 * math.pi / step.wavelength
            if step.axis == 0:
                q[:, 0] += step.amplitude * np.sin(frequency * q[:, 1] + step.phase)
            else:
                q[:, 1] += step.amplitude * np.sin(frequency * q[:, 0] + step.phase)
            self._rotate(q, -step.angle)

        self._rotate(q, -self.angle)
        return q * self.scale + self.origin

    def inverse(self, points: np.ndarray) -> np.ndarray:
        """Map warped tessellation coordinates back to projected geography."""
        arr = np.asarray(points, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")

        q = (arr - self.origin) / self.scale
        q = q.copy()
        self._rotate(q, self.angle)

        for step in reversed(self.steps):
            self._rotate(q, step.angle)
            frequency = 2.0 * math.pi / step.wavelength
            if step.axis == 0:
                q[:, 0] -= step.amplitude * np.sin(frequency * q[:, 1] + step.phase)
            else:
                q[:, 1] -= step.amplitude * np.sin(frequency * q[:, 0] + step.phase)
            self._rotate(q, -step.angle)

        self._rotate(q, -self.angle)
        return q * self.scale + self.origin

    def _geometry_transform(self, geometry: BaseGeometry, *, inverse: bool) -> BaseGeometry:
        if geometry.is_empty:
            return geometry

        dense = segmentize(geometry, max_segment_length=self.geometry_segment_length)
        mapper = self.inverse if inverse else self.forward

        def callback(x, y, z=None):
            x_arr = np.asarray(x, dtype=np.float64)
            y_arr = np.asarray(y, dtype=np.float64)
            shape = x_arr.shape
            coords = np.column_stack([x_arr.ravel(), y_arr.ravel()])
            mapped = mapper(coords)
            mapped_x = mapped[:, 0].reshape(shape)
            mapped_y = mapped[:, 1].reshape(shape)
            if z is None:
                return mapped_x, mapped_y
            return mapped_x, mapped_y, z

        return transform_geometry(callback, dense)

    def forward_geometry(self, geometry: BaseGeometry) -> BaseGeometry:
        return self._geometry_transform(geometry, inverse=False)

    def inverse_geometry(self, geometry: BaseGeometry) -> BaseGeometry:
        return self._geometry_transform(geometry, inverse=True)

    def tessellation_clip_box(self, boundary: BaseGeometry) -> BaseGeometry:
        """Return a warped-space box whose inverse safely contains ``boundary``.

        Fine cells are first constructed as ordinary power cells inside this box,
        then inverse-warped and clipped by the exact geographic boundary. This
        avoids approximating the dataset boundary in warped space.
        """
        warped_boundary = self.forward_geometry(boundary)
        minx, miny, maxx, maxy = warped_boundary.bounds
        width = maxx - minx
        height = maxy - miny
        margin = max(width, height, self.scale) * 0.15
        return box(minx - margin, miny - margin, maxx + margin, maxy + margin)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "sinusoidal_shear",
            "origin": [float(self.origin_x), float(self.origin_y)],
            "scale": float(self.scale),
            "angle": float(self.angle),
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SinusoidalWarp":
        if data.get("type") != "sinusoidal_shear":
            raise ValueError(f"Unsupported warp type: {data.get('type')!r}")
        origin = data["origin"]
        return cls(
            origin_x=float(origin[0]),
            origin_y=float(origin[1]),
            scale=float(data["scale"]),
            angle=float(data["angle"]),
            steps=tuple(_ShearStep.from_dict(step) for step in data["steps"]),
        )


def make_sinusoidal_warp(
    boundary: BaseGeometry,
    *,
    strength: float = 0.12,
    octaves: int = 4,
    random_state: int = 42,
) -> SinusoidalWarp:
    """Create a deterministic multi-scale organic warp for one boundary.

    ``strength`` is dimensionless. Roughly speaking, each octave displaces space
    by ``strength * wavelength``. Because the transform is built from shears, it
    remains analytically invertible rather than relying on numerical inversion.
    """
    if strength < 0:
        raise ValueError("warp_strength must be >= 0")
    if octaves < 1:
        raise ValueError("warp_octaves must be >= 1")

    minx, miny, maxx, maxy = boundary.bounds
    width = maxx - minx
    height = maxy - miny
    scale = max(float(width), float(height), 1.0)
    origin_x = float((minx + maxx) / 2.0)
    origin_y = float((miny + maxy) / 2.0)

    rng = np.random.default_rng(random_state)
    angle = float(rng.uniform(0.0, math.pi))

    # Start with a feature spanning around half the dataset, then halve the
    # wavelength at each octave. Two differently phased shears per octave avoid
    # a conspicuous axis-aligned wave pattern.
    steps: list[_ShearStep] = []
    for octave in range(octaves):
        wavelength = 0.55 / (2**octave)
        octave_strength = strength * (0.88**octave)
        amplitude = octave_strength * wavelength

        if strength == 0:
            amp_x = amp_y = 0.0
        else:
            amp_x = amplitude * float(rng.uniform(0.80, 1.20))
            amp_y = amplitude * float(rng.uniform(0.80, 1.20))

        steps.append(
            _ShearStep(
                axis=0,
                amplitude=amp_x,
                wavelength=wavelength,
                phase=float(rng.uniform(0.0, 2.0 * math.pi)),
                angle=float(rng.uniform(0.0, math.pi)),
            )
        )
        steps.append(
            _ShearStep(
                axis=1,
                amplitude=amp_y,
                wavelength=wavelength * float(rng.uniform(0.90, 1.10)),
                phase=float(rng.uniform(0.0, 2.0 * math.pi)),
                angle=float(rng.uniform(0.0, math.pi)),
            )
        )

    return SinusoidalWarp(
        origin_x=origin_x,
        origin_y=origin_y,
        scale=scale,
        angle=angle,
        steps=tuple(steps),
    )
