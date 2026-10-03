# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""A simulated laser line scanner behind ``DataAcquire`` (6.15.1)."""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pyippdme.types.pointcloud import MeasPoint, PointCloud, PointCloudSet, PointSet
from pyippdme.types.vec3 import Vec3, norm, normalize, sub

#: ``(origin, unit direction) -> (hit point, distance)`` against everything touchable.
Caster = Callable[[Vec3, Vec3], tuple[Vec3, float] | None]


@dataclass(frozen=True, slots=True)
class LineScannerSpec:
    """Typical numbers of a laser line profiler on a CMM (not one specific product)."""

    #: Points per profile and width of the laser line at the stand-off distance (mm).
    points_per_line: int = 320
    line_width: float = 30.0
    #: Distance from the sensor to the middle of the measuring range and the range itself (mm).
    standoff: float = 60.0
    depth_range: float = 30.0
    #: Spacing of profiles when sweeping between control points (mm).
    line_pitch: float = 0.5
    #: Standard deviation of a point along the viewing direction (mm).
    noise_mm: float = 0.003
    #: Share of points lost (dark/shiny surface, occlusion).
    dropout: float = 0.005
    #: Hits more oblique than this to the viewing direction are lost (degrees); needs normals.
    max_points: int = 400_000


class LineScanner:
    """Samples the scene with a line of rays per control point, like an optical profiler.

    Control points are sensor positions with the viewing direction ``IJK``
    (``-Z`` when zero). A ``Sweep`` (6.15.1) places a profile every
    ``line_pitch`` mm along the polyline of the control points; the line is
    laid out perpendicular to both the viewing direction and the travel.
    """

    def __init__(
        self,
        cast: Caster,
        spec: LineScannerSpec | None = None,
        *,
        seed: int | None = None,
        noise: Callable[[], bool] = lambda: True,
        on_acquired: Callable[[PointCloudSet], None] | None = None,
    ) -> None:
        self.spec = spec or LineScannerSpec()
        self._cast = cast
        self._rng = random.Random(seed)  # noqa: S311 # nosec B311 (simulation noise)
        self._noise_enabled = noise
        self._on_acquired = on_acquired

    def acquire(
        self,
        acq_name: str,
        acquisition_type: str,
        positions: list[Vec3],
        directions: list[Vec3],
    ) -> PointCloudSet | None:
        del acq_name
        s = self.spec
        stations: list[tuple[Vec3, Vec3, Vec3]] = []  # (sensor position, view dir, travel dir)
        for i, pos in enumerate(positions):
            view = directions[i] if norm(directions[i]) > 1e-9 else (0.0, 0.0, -1.0)
            nxt = positions[min(i + 1, len(positions) - 1)]
            prev = positions[max(i - 1, 0)]
            travel = sub(nxt, prev) if nxt != prev else (1.0, 0.0, 0.0)
            stations.append((pos, normalize(view), travel))
        if acquisition_type == "Sweep" and len(positions) >= 2:
            dense: list[tuple[Vec3, Vec3, Vec3]] = []
            for a, b in itertools.pairwise(stations):
                length = norm(sub(b[0], a[0]))
                count = max(1, math.ceil(length / s.line_pitch))
                for k in range(count):
                    t = k / count
                    p = tuple(a[0][j] + (b[0][j] - a[0][j]) * t for j in range(3))
                    v = normalize(tuple(a[1][j] + (b[1][j] - a[1][j]) * t for j in range(3)))  # type: ignore[arg-type]
                    dense.append((p, v, sub(b[0], a[0])))  # type: ignore[arg-type]
            dense.append(stations[-1])
            stations = dense
        points: list[MeasPoint] = []
        noisy = self._noise_enabled()
        for pos, view, travel in stations:
            across = np.cross(np.asarray(view), np.asarray(travel))
            if np.linalg.norm(across) < 1e-9:
                across = np.cross(np.asarray(view), np.array([1.0, 0.0, 0.0]))
                if np.linalg.norm(across) < 1e-9:
                    across = np.array([0.0, 1.0, 0.0])
            across = across / np.linalg.norm(across)
            for k in range(s.points_per_line):
                u = (k / max(s.points_per_line - 1, 1) - 0.5) * s.line_width
                origin = tuple(pos[j] + across[j] * u for j in range(3))
                hit = self._cast(origin, view)
                if hit is None:
                    continue
                point, distance = hit
                if abs(distance - s.standoff) > s.depth_range / 2:
                    continue
                if noisy and self._rng.random() < s.dropout:
                    continue
                if noisy:
                    e = self._rng.gauss(0.0, s.noise_mm)
                    point = tuple(point[j] - view[j] * e for j in range(3))  # type: ignore[assignment]
                points.append(MeasPoint(point[0], point[1], point[2], 0))
                if len(points) >= s.max_points:
                    break
        cloud = PointCloudSet(
            (PointCloud((PointSet((0.0, 0.0, 1.0), stations[0][1], tuple(points)),)),)
        )
        if self._on_acquired is not None:
            self._on_acquired(cloud)
        return cloud
