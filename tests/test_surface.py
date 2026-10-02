# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for pyippdme.server.surface's SampleSurface implementations."""

from __future__ import annotations

import math

import pytest

from pyippdme.simulation.surface import CylinderSurface, PlaneSurface, SphereSurface


def test_plane_surface_hits_straight_on() -> None:
    plane = PlaneSurface(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0))
    hit = plane.intersect(origin=(1.0, 2.0, 10.0), direction=(0.0, 0.0, -1.0))
    assert hit == pytest.approx((1.0, 2.0, 0.0))


def test_plane_surface_misses_when_parallel() -> None:
    plane = PlaneSurface(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0))
    assert plane.intersect(origin=(0.0, 0.0, 10.0), direction=(1.0, 0.0, 0.0)) is None


def test_plane_surface_misses_when_behind_the_ray() -> None:
    plane = PlaneSurface(point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0))
    assert plane.intersect(origin=(0.0, 0.0, 10.0), direction=(0.0, 0.0, 1.0)) is None


def test_sphere_surface_hits_the_near_side() -> None:
    sphere = SphereSurface(center=(0.0, 0.0, 0.0), radius=5.0)
    hit = sphere.intersect(origin=(0.0, 0.0, 20.0), direction=(0.0, 0.0, -1.0))
    assert hit == pytest.approx((0.0, 0.0, 5.0))


def test_sphere_surface_reports_the_far_side_from_inside() -> None:
    sphere = SphereSurface(center=(0.0, 0.0, 0.0), radius=5.0)
    hit = sphere.intersect(origin=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0))
    assert hit == pytest.approx((0.0, 0.0, 5.0))


def test_sphere_surface_misses_when_the_ray_passes_beside_it() -> None:
    sphere = SphereSurface(center=(0.0, 0.0, 0.0), radius=5.0)
    assert sphere.intersect(origin=(20.0, 20.0, 0.0), direction=(0.0, 0.0, -1.0)) is None


def test_cylinder_surface_hits_perpendicular_to_its_axis() -> None:
    # A cylinder along Z through the origin, radius 5.
    cylinder = CylinderSurface(point=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), radius=5.0)
    hit = cylinder.intersect(origin=(20.0, 0.0, 3.0), direction=(-1.0, 0.0, 0.0))
    assert hit == pytest.approx((5.0, 0.0, 3.0))


def test_cylinder_surface_misses_when_parallel_to_its_axis() -> None:
    cylinder = CylinderSurface(point=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), radius=5.0)
    assert cylinder.intersect(origin=(20.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0)) is None


def test_cylinder_surface_hits_at_an_angle() -> None:
    cylinder = CylinderSurface(point=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), radius=5.0)
    # A ray angled 45 degrees off the axis, aimed through the cylinder wall.
    hit = cylinder.intersect(origin=(20.0, 0.0, 0.0), direction=(-1.0, 0.0, 1.0))
    assert hit is not None
    x, y, _z = hit
    assert math.hypot(x, y) == pytest.approx(5.0)
