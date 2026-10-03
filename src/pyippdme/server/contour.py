# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Scanning an unknown contour (6.13.2.2, Figures 35-39), and an algorithm that follows it.

The five commands ``ScanInPlaneEndIsSphere``/``EndIsPlane``/``EndIsCyl`` and
``ScanInCylEndIsSphere``/``EndIsPlane`` have the machine follow the real surface of a part, which
the server does not know in advance, step by step from a start point in a given direction, until a
stop element (sphere, plane or cylinder) is reached. The path stays in a **constraint**: a scanning
plane (``ScanInPlane``) or a cylindrical surface around an axis (``ScanInCyl``).

:class:`ContourScan` describes one such scan; a backend that can probe a surface (see
:meth:`~pyippdme.server.backend.MachineBackend.scan_contour`) passes it to
:func:`trace_contour`, which needs nothing but a function that probes the surface along a ray.
Pure Python and numpy.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from pyippdme.types.vec3 import Vec3, add, dot, norm, normalize, scale, sub

#: Probes the surface: from ``origin`` along ``direction`` to ``(point, normal)``, or ``None``.
#: The normal points away from the material, towards the probe.
SurfaceProbe = Callable[[Vec3, Vec3], tuple[Vec3, Vec3] | None]


@dataclass(frozen=True, slots=True)
class ContourConstraint:
    """What the path stays in: a plane (``ScanInPlane``) or a cylinder (``ScanInCyl``)."""

    kind: str  # "plane" or "cylinder"
    #: The scanning plane's normal (``Ni, Nj, Nk``).
    normal: Vec3 | None = None
    #: A point on the cylinder axis and its direction (``Cx, Cy, Cz`` and ``Ci, Cj, Ck``).
    axis_point: Vec3 | None = None
    axis: Vec3 | None = None


@dataclass(frozen=True, slots=True)
class ContourStop:
    """What ends the scan: a sphere, a plane or a cylinder."""

    kind: str  # "sphere", "plane" or "cylinder"
    #: Sphere centre, a point of the plane, or a point on the cylinder axis.
    point: Vec3
    #: The plane's normal or the cylinder's axis direction.
    direction: Vec3 | None = None
    #: The sphere's diameter ``Dia`` or the cylinder's diameter ``d``.
    size: float = 0.0


@dataclass(frozen=True, slots=True)
class ContourScan:
    """One unknown-contour scan, in the coordinates of the machine it runs on."""

    constraint: ContourConstraint
    start: Vec3
    #: ``Si, Sj, Sk``: from the surface towards the probe at the start.
    probe: Vec3
    #: ``Dx, Dy, Dz``: a point that gives the direction of travel from the start.
    direction_point: Vec3
    #: ``StepW``: the distance between points along the path.
    step: float
    stop: ContourStop
    #: ``n``: how often the stop element has to be reached.
    count: int = 1
    #: ``Ei, Ej, Ek``: the probing direction at the end.
    end_direction: Vec3 = (0.0, 0.0, 0.0)


def stop_reached(stop: ContourStop, point: Vec3, start: Vec3) -> bool:
    """Whether ``point`` is in the stop element (a plane: on or beyond it, seen from the start)."""
    if stop.kind == "sphere":
        return norm(sub(point, stop.point)) <= stop.size / 2.0
    assert stop.direction is not None  # noqa: S101 (plane and cylinder always have one)
    unit = normalize(stop.direction)
    if stop.kind == "plane":
        before = dot(sub(start, stop.point), unit)
        now = dot(sub(point, stop.point), unit)
        return now * (1.0 if before >= 0 else -1.0) <= 0.0
    along = dot(sub(point, stop.point), unit)
    radial = sub(sub(point, stop.point), scale(unit, along))
    return norm(radial) <= stop.size / 2.0


class StopTracker:
    """Counts how often the stop element has been reached, as 6.13.2.2 defines it.

    * Sphere: an entry counts when the distance to the centre has a local minimum inside the
      sphere. If the scan starts inside, it first has to leave.
    * Plane: a pass through the plane counts, but only after the scan has moved further from
      the start than the direction point is (``skip``).
    * Cylinder: an entry into the cylinder counts. If the scan starts inside, it first has to
      leave.

    Feed every measured point to :meth:`update`.
    """

    def __init__(self, stop: ContourStop, start: Vec3, skip: float = 0.0, needed: int = 1) -> None:
        self._stop = stop
        self._start = start
        self._skip = skip
        self._needed = max(needed, 1)
        self.count = 0
        self._armed = not self._inside(start)  # started inside: wait until it has left
        self._counted = False
        self._previous_distance = math.inf
        self._last_side = self._signed(start) if stop.kind == "plane" else 0.0

    def _inside(self, point: Vec3) -> bool:
        stop = self._stop
        if stop.kind == "plane":
            return False
        if stop.kind == "sphere":
            return norm(sub(point, stop.point)) <= stop.size / 2.0
        assert stop.direction is not None  # noqa: S101
        unit = normalize(stop.direction)
        offset = sub(point, stop.point)
        return norm(sub(offset, scale(unit, dot(offset, unit)))) <= stop.size / 2.0

    def _signed(self, point: Vec3) -> float:
        assert self._stop.direction is not None  # noqa: S101
        return dot(sub(point, self._stop.point), normalize(self._stop.direction))

    def update(self, point: Vec3) -> tuple[bool, bool]:
        """Take the next measured point; return ``(finished, keep this point)``."""
        if self._stop.kind == "plane":
            return self._update_plane(point)
        if not self._inside(point):
            self._armed = True
            self._counted = False
            self._previous_distance = math.inf
            return False, True
        if not self._armed or self._counted:
            return False, True
        if self._stop.kind == "cylinder":
            self._counted = True
            self.count += 1
            return self.count >= self._needed, True
        distance = norm(sub(point, self._stop.point))
        increasing = distance > self._previous_distance
        self._previous_distance = distance
        if not increasing:
            return False, True
        # The previous point was the minimum; this one is already moving away from the centre.
        self._counted = True
        self.count += 1
        return self.count >= self._needed, False

    def _update_plane(self, point: Vec3) -> tuple[bool, bool]:
        side = self._signed(point)
        # A point exactly on the plane is no pass yet: the next one on the other side is.
        crossed = side != 0.0 and self._last_side != 0.0 and (side > 0.0) != (self._last_side > 0.0)
        if side != 0.0:
            self._last_side = side
        if crossed and norm(sub(point, self._start)) > self._skip:
            self.count += 1
            return self.count >= self._needed, True
        return False, True


def _toward(vector: Vec3, reference: Vec3) -> Vec3:
    return vector if dot(vector, reference) >= 0.0 else scale(vector, -1.0)


def _cylinder_radius_vector(constraint: ContourConstraint, point: Vec3) -> Vec3:
    assert constraint.axis_point is not None  # noqa: S101
    assert constraint.axis is not None  # noqa: S101
    axis = normalize(constraint.axis)
    offset = sub(point, constraint.axis_point)
    return sub(offset, scale(axis, dot(offset, axis)))


def _next_heading(constraint: ContourConstraint, point: Vec3, normal: Vec3, heading: Vec3) -> Vec3:
    """Return the direction of travel that is tangent to the surface and stays in the constraint."""
    if constraint.kind == "plane":
        assert constraint.normal is not None  # noqa: S101
        across = np.cross(normalize(constraint.normal), normal)
    else:
        radial = _cylinder_radius_vector(constraint, point)
        if norm(radial) < 1e-12:
            return heading
        across = np.cross(normalize(radial), normal)
    length = float(np.linalg.norm(across))
    if length < 1e-6:  # the surface is parallel to the constraint: keep going straight on
        if constraint.kind == "plane":
            return heading
        radial_unit = normalize(_cylinder_radius_vector(constraint, point))
        keep = sub(heading, scale(radial_unit, dot(heading, radial_unit)))
        return normalize(keep) if norm(keep) > 1e-9 else heading
    unit = (float(across[0] / length), float(across[1] / length), float(across[2] / length))
    return _toward(unit, heading)


def _constrain(constraint: ContourConstraint, point: Vec3, radius: float) -> Vec3:
    """Pull a nominal point back onto the cylinder (a plane needs nothing: heading is in it)."""
    if constraint.kind != "cylinder":
        return point
    assert constraint.axis_point is not None  # noqa: S101
    assert constraint.axis is not None  # noqa: S101
    axis = normalize(constraint.axis)
    offset = sub(point, constraint.axis_point)
    along = dot(offset, axis)
    radial = sub(offset, scale(axis, along))
    if norm(radial) < 1e-12:
        return point
    return add(constraint.axis_point, add(scale(axis, along), scale(normalize(radial), radius)))


def trace_contour(
    scan: ContourScan,
    probe_surface: SurfaceProbe,
    *,
    lookahead: float = 3.0,
    max_points: int = 20_000,
    max_length: float = 10_000.0,
) -> Iterator[Vec3]:
    """Follow the surface from the start until the stop element is reached; yields measured points.

    At each step the surface is probed from ``lookahead`` above the nominal point, along the last
    known normal. The measured point and normal give the next direction of travel: tangent to the
    surface and inside the constraint (the intersection of the surface's tangent plane with the
    scanning plane, or with the cylinder's), continued in the sense of the previous step; the
    next nominal point lies one ``step`` along it. The scan ends when the stop element has been
    reached ``count`` times, when the surface is lost, or at ``max_points``/``max_length``.
    """
    constraint = scan.constraint
    probe = normalize(scan.probe)
    heading = normalize(sub(scan.direction_point, scan.start))
    radius = 0.0
    if constraint.kind == "cylinder":
        radius = norm(_cylinder_radius_vector(constraint, scan.start))
    point = scan.start
    tracker = StopTracker(
        scan.stop, scan.start, norm(sub(scan.direction_point, scan.start)), scan.count
    )
    travelled = 0.0
    for _ in range(max_points):
        hit = probe_surface(add(point, scale(probe, lookahead)), scale(probe, -1.0))
        if hit is None:  # a little further out, in case the surface curved away
            hit = probe_surface(add(point, scale(probe, 3.0 * lookahead)), scale(probe, -1.0))
        if hit is None:
            return
        measured, normal = hit
        done, keep = tracker.update(measured)
        if keep:
            yield measured
        if done:
            return
        normal = _toward(normalize(normal), probe)
        heading = _next_heading(constraint, measured, normal, heading)
        point = _constrain(constraint, add(measured, scale(heading, scan.step)), radius)
        probe = normal
        if constraint.kind == "plane":
            assert constraint.normal is not None  # noqa: S101
            plane_normal = normalize(constraint.normal)
            in_plane = sub(normal, scale(plane_normal, dot(normal, plane_normal)))
            if norm(in_plane) > 0.2:  # a surface parallel to the plane: keep the probing direction
                probe = normalize(in_plane)
        travelled += scan.step
        if travelled > max_length or math.isnan(point[0]):
            return
