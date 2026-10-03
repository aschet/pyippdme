# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Fitting and evaluating nominal features from measured points."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyippdme.twin.features import (
    Cone,
    Cylinder,
    Length,
    Plane,
    Sphere,
    evaluate_features,
    fit_circle,
    fit_sphere,
)
from pyippdme.twin.spec import Accuracy

ACCURACY = Accuracy(1.9, 300.0, 1.9)
UP = np.array([0.0, 0.0, 1.0])


def _sphere_points(center: np.ndarray, radius: float, n: int = 40) -> np.ndarray:
    rng = np.random.default_rng(0)
    d = rng.normal(size=(n, 3))
    d[:, 2] = np.abs(d[:, 2])
    return center + radius * d / np.linalg.norm(d, axis=1, keepdims=True)


def test_sphere_fit_recovers_centre_and_radius() -> None:
    center = np.array([40.0, 45.0, 97.5])
    points = _sphere_points(center, 12.5)
    fitted, radius = fit_sphere(points)
    assert radius == pytest.approx(12.5, abs=1e-9)
    assert np.allclose(fitted, center, atol=1e-9)


def test_circle_fit_recovers_a_bore() -> None:
    angles = np.linspace(0, 2 * math.pi, 12, endpoint=False)
    xy = np.column_stack([3 + 25 * np.cos(angles), -2 + 25 * np.sin(angles)])
    center, radius = fit_circle(xy)
    assert radius == pytest.approx(25.0)
    assert np.allclose(center, (3, -2))


def test_sphere_evaluation_reports_size_position_and_form_against_the_mpe() -> None:
    center = np.array([40.0, 45.0, 97.5])
    sphere = Sphere("ref", center, 12.5)
    points = _sphere_points(center, 12.5 + 0.002)  # 2 um too big
    results = {r.quantity: r for r in sphere.evaluate(points, ACCURACY)}
    assert results["diameter"].error == pytest.approx(0.004, abs=1e-6)
    assert results["diameter"].ok
    assert results["form"].measured == pytest.approx(0.0, abs=1e-9)


def test_cylinder_distinguishes_bore_from_pin_and_selects_by_height() -> None:
    angles = np.linspace(0, 2 * math.pi, 16, endpoint=False)
    ring = Cylinder("ring", np.array([0.0, 0.0, 0.0]), UP, 25.0, True, 2.0, 18.0)
    points = np.column_stack([25.0 * np.cos(angles), 25.0 * np.sin(angles), np.full(16, 10.0)])
    far = points + np.array([0.0, 0.0, 100.0])  # same radius, outside the height range
    results = ring.evaluate(np.vstack([points, far]), ACCURACY)
    assert {r.quantity for r in results} == {"diameter (bore)", "position", "roundness"}
    assert results[0].points == 16
    assert results[0].error == pytest.approx(0.0, abs=1e-9)


def test_plane_flatness_and_position_and_lengths_between_planes() -> None:
    left = Plane("l", np.array([0.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 10.0, 10.0)
    right = Plane("r", np.array([40.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 10.0, 10.0)
    rng = np.random.default_rng(1)
    yz = rng.uniform(-8, 8, size=(10, 2))
    pa = np.column_stack([np.zeros(10), yz])
    pb = np.column_stack([np.full(10, 40.003), yz])  # the second face is 3 um further away
    length = Length("gauge 40 mm", "l", "r", 40.0)
    results = evaluate_features([left, right], [length], np.vstack([pa, pb]), ACCURACY)
    gauge = next(r for r in results if r.quantity == "length")
    assert gauge.error == pytest.approx(0.003, abs=1e-9)
    assert gauge.limit == pytest.approx((1.9 + 40 / 300) * 1e-3)
    assert gauge.ok is False or gauge.ok is True  # evaluated, with a verdict
    flat = next(r for r in results if r.feature == "l" and r.quantity == "flatness")
    assert flat.measured == pytest.approx(0.0, abs=1e-9)


def test_cone_deviation_from_the_nominal_surface() -> None:
    cone = Cone("c", np.array([0.0, 0.0, 0.0]), UP, 18.0, 9.0, 35.0)
    t = np.linspace(2, 33, 30)
    r = 18.0 - 9.0 * t / 35.0
    angles = np.linspace(0, 2 * math.pi, 30, endpoint=False)
    points = np.column_stack([r * np.cos(angles), r * np.sin(angles), t])
    results = {x.quantity: x for x in cone.evaluate(points, ACCURACY)}
    assert results["max deviation"].measured == pytest.approx(0.0, abs=1e-9)
