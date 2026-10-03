# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Optical sensor models behind ``DataAcquire`` (6.15.1): point, line, area and camera.

Pure numpy; the scene arrives as a :class:`~pyippdme.twin.depthbuffer.DepthBuffer` for the
sensor's viewing direction, so these models work with any scene that can provide one.

* ``point``: a laser point or confocal sensor, one distance per control point.
* ``line``: a laser line profiler, a profile of ``points_per_line`` points per station.
* ``area``: a structured-light / fringe-projection scanner, a grid of points per shot.
* ``camera``: a telecentric 2D camera that returns the edges (silhouette and depth
  steps) it sees, as a video measuring system does.

What the models reproduce: the sensor only sees within its depth range around the
stand-off; surfaces tilted more than ``max_angle_deg`` against the viewing direction return
nothing; with a triangulation angle, surfaces hidden from the camera by a step or wall are
shadowed; noise grows with the distance from the stand-off and with the tilt; points at depth
steps are unreliable (flying pixels); a share of points drops out.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pyippdme.twin.depthbuffer import DepthBuffer
from pyippdme.twin.pointtypes import POINT_TYPES, to_gsl, to_qsp
from pyippdme.types.pointcloud import MeasPoint, PointCloud, PointCloudSet, PointSet
from pyippdme.types.vec3 import Vec3, norm, normalize, sub

Array = NDArray[np.float64]
#: Provides the depth buffer for a viewing direction (cached by the scene's owner).
BufferProvider = Callable[[Vec3], DepthBuffer]

KINDS = ("point", "line", "area", "camera")
#: Depth steps larger than this (mm) mark an edge in line, area and camera data.
EDGE_STEP = 1.0


@dataclass(frozen=True, slots=True)
class OpticalSpec:
    """Typical numbers of an optical sensor on a CMM (not one specific product)."""

    kind: str = "line"
    #: Distance from the sensor window to the middle of the measuring range, and its depth (mm).
    standoff: float = 60.0
    depth_range: float = 30.0
    #: Width of the line, or of the field of an area sensor or camera (mm).
    line_width: float = 30.0
    #: The other dimension of an area sensor's or camera's field (mm).
    field_height: float = 22.0
    #: Points per line, or columns of the field.
    points_per_line: int = 320
    #: Spacing of stations when sweeping between control points (mm).
    line_pitch: float = 1.0
    #: Noise (standard deviation, mm) at the stand-off, looking straight at the surface.
    noise_mm: float = 0.003
    #: Share of points lost at random (dark or shiny surface).
    dropout: float = 0.005
    #: Surfaces tilted more than this against the viewing direction return nothing (degrees).
    max_angle_deg: float = 75.0
    #: Angle between laser and camera; 0 switches shadowing off (degrees).
    triangulation_deg: float = 30.0
    max_points: int = 400_000
    #: What the sensor delivers (Figure 40): ``RSL`` raw scan lines, ``GSL`` gridded scan lines or
    #: ``QSP`` qualified surface points (averaged on the grid), and the grid pitch (mm).
    point_type: str = "RSL"
    grid_pitch: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {', '.join(KINDS)}")
        if self.point_type not in POINT_TYPES:
            raise ValueError(f"point_type must be one of {', '.join(POINT_TYPES)}")


#: Name kept from the first, line-only version of this module.
LineScannerSpec = OpticalSpec


@dataclass(frozen=True, slots=True)
class Station:
    """Where the sensor is for one shot: window position, viewing direction and travel."""

    position: Vec3
    view: Vec3
    travel: Vec3


def stations(
    positions: list[Vec3],
    directions: list[Vec3],
    acquisition_type: str,
    spacing: float,
) -> list[Station]:
    """Sensor stations for ``DataAcquire``'s control points.

    ``directions`` are the ``primary`` vectors of ``DataAcquire``'s control points (6.15.1,
    Table 95), anti-parallel to the tool axis, i.e. pointing away from the surface: the sensor
    looks along their inverse. A zero vector means looking down (``-Z``).

    ``SingleShot`` and ``MultiShot`` use the control points as they are; ``Sweep``
    places a station every ``spacing`` mm along the polyline of the control points.
    """
    base: list[Station] = []
    for i, pos in enumerate(positions):
        primary = directions[i]
        view = (-primary[0], -primary[1], -primary[2]) if norm(primary) > 1e-9 else (0.0, 0.0, -1.0)
        nxt = positions[min(i + 1, len(positions) - 1)]
        prev = positions[max(i - 1, 0)]
        travel = sub(nxt, prev) if nxt != prev else (1.0, 0.0, 0.0)
        base.append(Station(pos, normalize(view), travel))
    if acquisition_type != "Sweep" or len(base) < 2:
        return base
    dense: list[Station] = []
    for a, b in zip(base, base[1:], strict=False):
        length = norm(sub(b.position, a.position))
        count = max(1, math.ceil(length / spacing))
        for k in range(count):
            t = k / count
            pos = (
                a.position[0] + (b.position[0] - a.position[0]) * t,
                a.position[1] + (b.position[1] - a.position[1]) * t,
                a.position[2] + (b.position[2] - a.position[2]) * t,
            )
            view = normalize(tuple(a.view[j] + (b.view[j] - a.view[j]) * t for j in range(3)))  # type: ignore[arg-type]
            dense.append(Station(pos, view, sub(b.position, a.position)))
    dense.append(base[-1])
    return dense


def field_axes(view: Vec3, travel: Vec3) -> tuple[Array, Array]:
    """The field's axes: ``u`` across the line (perpendicular to view and travel), ``v`` along."""
    d = np.asarray(view, dtype=float)
    u = np.cross(d, np.asarray(travel, dtype=float))
    if np.linalg.norm(u) < 1e-9:
        u = np.cross(d, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(u) < 1e-9:
            u = np.array([0.0, 1.0, 0.0])
    u = u / np.linalg.norm(u)
    return u, np.cross(u, d)


class OpticalSensor:
    """Samples a scene like an optical sensor; satisfies :class:`~pyippdme.server.surface.RawSensor`."""

    def __init__(
        self,
        buffers: BufferProvider,
        spec: OpticalSpec | None = None,
        *,
        seed: int | None = None,
        noise: Callable[[], bool] = lambda: True,
        on_acquired: Callable[[PointCloudSet], None] | None = None,
    ) -> None:
        self.fixed_spec = spec or OpticalSpec()
        #: Called before each acquisition for the spec to use (the active tool's, if optical).
        self.spec_provider: Callable[[], OpticalSpec | None] | None = None
        self._buffers = buffers
        self._rng = np.random.default_rng(seed)
        self._noise_enabled = noise
        self._on_acquired = on_acquired

    @property
    def spec(self) -> OpticalSpec:
        provided = self.spec_provider() if self.spec_provider is not None else None
        return provided or self.fixed_spec

    # -- RawSensor ---------------------------------------------------------------------

    def acquire(
        self,
        acq_name: str,
        acquisition_type: str,
        positions: list[Vec3],
        directions: list[Vec3],
    ) -> PointCloudSet | None:
        del acq_name
        s = self.spec
        spacing = s.line_pitch if s.kind in ("point", "line") else max(s.field_height * 0.85, 1.0)
        shots = stations(positions, directions, acquisition_type, spacing)
        points: list[MeasPoint] = []
        for station in shots:
            points.extend(self.shoot(station, s))
            if len(points) >= s.max_points:
                break
        points = self._deliver(points, shots[0].view, s)
        cloud = PointCloudSet(
            (PointCloud((PointSet((0.0, 0.0, 1.0), shots[0].view, tuple(points)),)),)
        )
        if self._on_acquired is not None:
            self._on_acquired(cloud)
        return cloud

    @staticmethod
    def _deliver(points: list[MeasPoint], view: Vec3, s: OpticalSpec) -> list[MeasPoint]:
        """Turn the raw points into what the sensor is set to deliver (RSL, GSL or QSP)."""
        if s.point_type == "RSL" or not points:
            return points
        raw = np.array([[p.x, p.y, p.z] for p in points])
        if s.point_type == "GSL":
            processed = to_gsl(raw, view, s.grid_pitch)
        else:
            processed = to_qsp(raw, view, s.grid_pitch)
        return [MeasPoint(float(p[0]), float(p[1]), float(p[2]), 0) for p in processed]

    # -- one shot ----------------------------------------------------------------------

    def shoot(self, station: Station, s: OpticalSpec) -> list[MeasPoint]:
        """The points one shot returns."""
        buffer = self._buffers(station.view)
        u, v = field_axes(station.view, station.travel)
        origin = np.asarray(station.position, dtype=float)
        view = np.asarray(station.view, dtype=float)
        cols = 1 if s.kind == "point" else max(s.points_per_line, 2)
        rows = 1
        if s.kind in ("area", "camera"):
            pixel = s.line_width / max(cols - 1, 1)
            rows = max(int(round(s.field_height / pixel)), 2)
        a = np.zeros(1) if cols == 1 else (np.arange(cols) / (cols - 1) - 0.5) * s.line_width
        b = np.zeros(1) if rows == 1 else (np.arange(rows) / (rows - 1) - 0.5) * s.field_height
        grid_a, grid_b = np.meshgrid(a, b)  # (rows, cols)
        origins = origin + grid_a.reshape(-1, 1) * u + grid_b.reshape(-1, 1) * v
        hits = buffer.hit(origins)
        depth = hits.distance.reshape(rows, cols)
        valid = hits.valid.reshape(rows, cols)
        gate = np.abs(depth - s.standoff) <= s.depth_range / 2.0
        if s.kind == "camera":
            return self._camera(s, hits.point.reshape(rows, cols, 3), depth, valid, u, v, view)
        ok = valid & gate
        cos_angle = np.abs(hits.normal @ view).reshape(rows, cols)
        ok &= cos_angle >= math.cos(math.radians(s.max_angle_deg))
        if s.triangulation_deg > 0.0 and s.kind in ("line", "area"):
            ok &= ~self._shadowed(buffer, hits.point, v, s, view).reshape(rows, cols)
        edge = self._edges(depth, valid) if s.kind in ("line", "area") else np.zeros_like(ok)
        noisy = self._noise_enabled()
        if noisy:
            ok &= self._rng.random(ok.shape) >= s.dropout
            ok &= ~(edge & (self._rng.random(ok.shape) < 0.3))
        points = hits.point.reshape(rows, cols, 3)
        if noisy:
            offset = (np.abs(depth - s.standoff) / (s.depth_range / 2.0)).clip(0, 1)
            sigma = s.noise_mm * (1.0 + 4.0 * offset**2) / np.maximum(cos_angle, 0.2)
            sigma = np.where(edge, sigma * 10.0, sigma)
            points = points - view * (self._rng.normal(size=ok.shape) * sigma)[..., None]
        quality = np.clip((1.0 - cos_angle) * 255.0, 0, 255).astype(int)
        return [
            MeasPoint(float(p[0]), float(p[1]), float(p[2]), int(q))
            for p, q in zip(points[ok], quality[ok], strict=True)
        ]

    @staticmethod
    def _edges(depth: Array, valid: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Where the depth steps by more than :data:`EDGE_STEP` along a row, column or at a gap."""
        edge = np.zeros_like(valid)
        for axis in (0, 1):
            if depth.shape[axis] < 2:
                continue
            step = np.abs(np.diff(depth, axis=axis))
            both = np.diff(valid.astype(int), axis=axis) == 0
            jump = both & valid.take(range(depth.shape[axis] - 1), axis=axis) & (step > EDGE_STEP)
            gap = np.diff(valid.astype(int), axis=axis) != 0
            mark = jump | gap
            lo = [slice(None)] * 2
            hi = [slice(None)] * 2
            lo[axis], hi[axis] = slice(0, -1), slice(1, None)
            edge[tuple(lo)] |= mark
            edge[tuple(hi)] |= mark
        return edge & valid

    @staticmethod
    def _shadowed(
        buffer: DepthBuffer, points: Array, along: Array, s: OpticalSpec, view: Array
    ) -> NDArray[np.bool_]:
        """Hidden from the camera: the line from the surface to the camera runs into the part.

        The camera sits ``standoff * tan(triangulation)`` to the side of the sensor window
        (along the travel). The test walks that line over the buffer's heights.
        """
        baseline = s.standoff * math.tan(math.radians(s.triangulation_deg))
        camera_offset = along * baseline - view * s.standoff  # from the surface towards the camera
        shadow = np.zeros(len(points), dtype=bool)
        for fraction in np.linspace(0.08, 1.0, 12):
            q = points + camera_offset * fraction
            shadow |= buffer.height_at(q) > -(q @ view) + 0.02
        return shadow

    def _camera(
        self,
        s: OpticalSpec,
        points: Array,
        depth: Array,
        valid: NDArray[np.bool_],
        u: Array,
        v: Array,
        view: Array,
    ) -> list[MeasPoint]:
        """Edge points of the image: silhouette pixels and depth steps, with pixel-scale noise."""
        seen = valid & (np.abs(depth - s.standoff) <= s.depth_range / 2.0)
        edge = self._edges(np.where(seen, depth, np.nan), seen)
        # The image border is not an edge: pretend the scene continues beyond it.
        padded = np.pad(seen, 1, constant_values=True)
        border = seen & ~(
            padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
        )
        pick = edge | border
        out = points[pick]
        if len(out) == 0:
            return []
        pixel = s.line_width / max(s.points_per_line - 1, 1)
        if self._noise_enabled():
            out = out + (
                self._rng.normal(size=(len(out), 1)) * u * pixel * 0.15
                + self._rng.normal(size=(len(out), 1)) * v * pixel * 0.15
            )
        return [MeasPoint(float(p[0]), float(p[1]), float(p[2]), 0) for p in out]


#: The line scanner of the first version of this module.
LineScanner = OpticalSensor
