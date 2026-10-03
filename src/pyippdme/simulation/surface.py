# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

r"""Synthetic sample-surface geometry: concrete :class:`~pyippdme.server.surface.SampleSurface`\ s.

This is also the intended extension point for a future "load a CAD mesh as
the sample" GUI feature - that surface would satisfy the same one-method
Protocol as the three simple ones here (see :mod:`pyippdme.server.surface`).
"""

from __future__ import annotations

import math

from pyippdme.server.surface import SampleSurface
from pyippdme.types.vec3 import Vec3, add, dot, norm, normalize, scale, sub

#: Below this, a ray is treated as parallel to a surface rather than
#: technically intersecting it at a huge, meaningless distance.
_PARALLEL_TOLERANCE = 1e-9


class PlaneSurface:
    """An infinite flat plane through ``point``, perpendicular to ``normal``."""

    def __init__(self, point: Vec3, normal: Vec3) -> None:
        self._point = point
        self._normal = normalize(normal)

    def intersect(self, origin: Vec3, direction: Vec3) -> Vec3 | None:
        unit = normalize(direction)
        denom = dot(self._normal, unit)
        if abs(denom) < _PARALLEL_TOLERANCE:
            return None
        t = dot(sub(self._point, origin), self._normal) / denom
        if t < 0:
            return None
        return add(origin, scale(unit, t))


class SphereSurface:
    """A sphere centered at ``center`` with the given ``radius`` (mm)."""

    def __init__(self, center: Vec3, radius: float) -> None:
        self._center = center
        self._radius = radius

    def intersect(self, origin: Vec3, direction: Vec3) -> Vec3 | None:
        unit = normalize(direction)
        to_center = sub(origin, self._center)
        b = dot(to_center, unit)
        c = dot(to_center, to_center) - self._radius * self._radius
        discriminant = b * b - c
        if discriminant < 0:
            return None
        sqrt_discriminant = math.sqrt(discriminant)
        t = _nearest_nonnegative(-b - sqrt_discriminant, -b + sqrt_discriminant)
        if t is None:
            return None
        return add(origin, scale(unit, t))


class CylinderSurface:
    """An infinite cylinder of the given ``radius`` (mm).

    Centered on the line through ``point`` along ``axis``.
    """

    def __init__(self, point: Vec3, axis: Vec3, radius: float) -> None:
        self._point = point
        self._axis = normalize(axis)
        self._radius = radius

    def intersect(self, origin: Vec3, direction: Vec3) -> Vec3 | None:
        unit = normalize(direction)
        direction_perp = sub(unit, scale(self._axis, dot(unit, self._axis)))
        offset = sub(origin, self._point)
        offset_perp = sub(offset, scale(self._axis, dot(offset, self._axis)))

        a = dot(direction_perp, direction_perp)
        if a < _PARALLEL_TOLERANCE:
            return None  # ray runs parallel to the cylinder's axis
        b = 2.0 * dot(offset_perp, direction_perp)
        c = dot(offset_perp, offset_perp) - self._radius * self._radius
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0:
            return None
        sqrt_discriminant = math.sqrt(discriminant)
        t = _nearest_nonnegative(
            (-b - sqrt_discriminant) / (2.0 * a), (-b + sqrt_discriminant) / (2.0 * a)
        )
        if t is None:
            return None
        return add(origin, scale(unit, t))


class CompositeSurface:
    """Several surfaces as one: a probing ray meets the nearest of them."""

    def __init__(self, surfaces: list[SampleSurface]) -> None:
        self._surfaces = list(surfaces)

    def intersect(self, origin: Vec3, direction: Vec3) -> Vec3 | None:
        hits = [h for s in self._surfaces if (h := s.intersect(origin, direction)) is not None]
        if not hits:
            return None
        return min(hits, key=lambda h: norm(sub(h, origin)))


def parse_surface(spec: str) -> SampleSurface:
    """Parse ``plane:px,py,pz:nx,ny,nz``, ``sphere:cx,cy,cz:r`` or ``cylinder:p:a:r``."""
    kind, *parts = spec.split(":")

    def vector(text: str) -> Vec3:
        x, y, z = (float(v) for v in text.split(","))
        return (x, y, z)

    try:
        if kind == "plane" and len(parts) == 2:
            return PlaneSurface(vector(parts[0]), vector(parts[1]))
        if kind == "sphere" and len(parts) == 2:
            return SphereSurface(vector(parts[0]), float(parts[1]))
        if kind == "cylinder" and len(parts) == 3:
            return CylinderSurface(vector(parts[0]), vector(parts[1]), float(parts[2]))
    except ValueError as exc:
        raise ValueError(f"bad numbers in surface {spec!r}: {exc}") from None
    raise ValueError(
        f"bad surface {spec!r}; use plane:P:N, sphere:C:R or cylinder:P:AXIS:R with points as x,y,z"
    )


def _nearest_nonnegative(t1: float, t2: float) -> float | None:
    """Return the smaller of ``t1``/``t2`` that is ``>= 0``, or ``None`` if neither is."""
    lo, hi = (t1, t2) if t1 <= t2 else (t2, t1)
    if lo >= 0:
        return lo
    if hi >= 0:
        return hi
    return None
