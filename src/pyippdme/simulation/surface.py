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

from pyippdme.types.vec3 import Vec3, add, dot, normalize, scale, sub

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


def _nearest_nonnegative(t1: float, t2: float) -> float | None:
    """Return the smaller of ``t1``/``t2`` that is ``>= 0``, or ``None`` if neither is."""
    lo, hi = (t1, t2) if t1 <= t2 else (t2, t1)
    if lo >= 0:
        return lo
    if hi >= 0:
        return hi
    return None
