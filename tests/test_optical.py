# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Optical sensor models on a synthetic scene: a plate with a tall block (a step and a wall)."""

from __future__ import annotations

import numpy as np
import pytest

from pyippdme.twin.depthbuffer import DepthBuffer
from pyippdme.twin.optical import OpticalSensor, OpticalSpec, stations


def _box(
    lo: tuple[float, float, float], hi: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], dtype=float)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    f = []
    for a, b, c, d in quads:
        f += [(a, b, c), (a, c, d)]
    return v, np.array(f, dtype=np.int64)


def _scene() -> DepthBuffer:
    plate_v, plate_f = _box((0, 0, 0), (200, 100, 10))
    block_v, block_f = _box((80, 30, 10), (120, 70, 40))
    vertices = np.vstack([plate_v, block_v])
    faces = np.vstack([plate_f, block_f + len(plate_v)])
    return DepthBuffer(vertices, faces, np.array([0.0, 0.0, -1.0]), 0.25)


def _sensor(**spec: object) -> OpticalSensor:
    buffer = _scene()
    return OpticalSensor(lambda _: buffer, OpticalSpec(**spec), seed=3)  # type: ignore[arg-type]


def test_depth_buffer_hits_are_exact_on_planes() -> None:
    hits = _scene().hit(np.array([[20.0, 50.0, 200.0], [100.0, 50.0, 200.0], [500.0, 50.0, 200.0]]))
    assert hits.valid.tolist() == [True, True, False]
    assert hits.point[0, 2] == pytest.approx(10.0)
    assert hits.point[1, 2] == pytest.approx(40.0)
    assert hits.normal[0] == pytest.approx((0.0, 0.0, 1.0))


def test_a_point_sensor_returns_one_distance_inside_its_depth_window() -> None:
    sensor = _sensor(kind="point", standoff=30.0, depth_range=10.0, noise_mm=0.0, dropout=0.0)
    cloud = sensor.acquire("a", "SingleShot", [(20.0, 50.0, 40.0)], [(0.0, 0.0, 1.0)])
    assert cloud is not None
    points = cloud.point_clouds[0].point_sets[0].points
    assert len(points) == 1 and points[0].z == pytest.approx(10.0)
    too_far = sensor.acquire("b", "SingleShot", [(20.0, 50.0, 80.0)], [(0.0, 0.0, 1.0)])
    assert too_far is not None and not too_far.point_clouds[0].point_sets[0].points


def test_a_line_scanner_profiles_a_step_and_marks_unreliable_edges() -> None:
    sensor = _sensor(
        kind="line",
        standoff=40.0,
        depth_range=60.0,
        line_width=100.0,
        points_per_line=400,
        noise_mm=0.0,
        dropout=0.0,
        triangulation_deg=0.0,
    )
    # The line runs along x (travel along y); the block is in the middle of it.
    cloud = sensor.acquire(
        "a", "SingleShot", [(100.0, 50.0, 80.0), (100.0, 60.0, 80.0)], [(0.0, 0.0, 1.0)] * 2
    )
    assert cloud is not None
    z = np.array([p.z for p in cloud.point_clouds[0].point_sets[0].points])
    assert set(np.round(z, 3)) == {10.0, 40.0}  # plate and block top, nothing in between


def test_steep_walls_return_nothing_and_triangulation_shadows_behind_a_step() -> None:
    # Looking along the wall: surfaces more than max_angle tilted are dropped.
    sensor = _sensor(
        kind="area",
        standoff=30.0,
        depth_range=100.0,
        line_width=40.0,
        field_height=40.0,
        points_per_line=60,
        noise_mm=0.0,
        dropout=0.0,
        triangulation_deg=0.0,
        max_angle_deg=10.0,
    )
    cloud = sensor.acquire("a", "SingleShot", [(100.0, 50.0, 70.0)], [(0.0, 0.0, 1.0)])
    assert cloud is not None
    tilted = [p for p in cloud.point_clouds[0].point_sets[0].points if p.q > 40]
    assert not tilted  # only faces that look at the sensor remain
    # With a triangulation camera the plate right behind the block is hidden from it.
    shadowed = _sensor(
        kind="line",
        standoff=60.0,
        depth_range=120.0,
        line_width=20.0,
        points_per_line=40,
        noise_mm=0.0,
        dropout=0.0,
        triangulation_deg=20.0,
    )
    near_wall = shadowed.acquire(
        "b", "SingleShot", [(100.0, 15.0, 70.0), (100.0, 25.0, 70.0)], [(0.0, 0.0, 1.0)] * 2
    )
    assert near_wall is not None
    # The camera sits ahead of the line (along the travel), behind the block's wall.
    assert 0 < len(near_wall.point_clouds[0].point_sets[0].points) < 80


def test_a_camera_returns_edges_only() -> None:
    sensor = _sensor(
        kind="camera",
        standoff=40.0,
        depth_range=100.0,
        line_width=100.0,
        field_height=60.0,
        points_per_line=200,
        noise_mm=0.0,
        dropout=0.0,
    )
    cloud = sensor.acquire("a", "SingleShot", [(100.0, 50.0, 80.0)], [(0.0, 0.0, 1.0)])
    assert cloud is not None
    points = cloud.point_clouds[0].point_sets[0].points
    assert points
    xs = np.array([p.x for p in points])
    ys = np.array([p.y for p in points])
    # Every point lies on the outline of the block (the depth step) or the field border.
    on_block = (
        (np.abs(xs - 80) < 1)
        | (np.abs(xs - 120) < 1)
        | (np.abs(ys - 30) < 1)
        | (np.abs(ys - 70) < 1)
    )
    assert np.mean(on_block) > 0.6


def test_a_sweep_places_stations_along_the_path_and_noise_follows_the_sensor() -> None:
    shots = stations([(0.0, 0.0, 50.0), (10.0, 0.0, 50.0)], [(0.0, 0.0, 1.0)] * 2, "Sweep", 2.0)
    assert len(shots) == 6
    assert shots[0].view == (0.0, 0.0, -1.0)  # primary points away from the surface
    noisy = _sensor(kind="point", standoff=30.0, depth_range=10.0, noise_mm=0.01, dropout=0.0)
    zs = []
    for _ in range(30):
        cloud = noisy.acquire("n", "SingleShot", [(20.0, 50.0, 40.0)], [(0.0, 0.0, 1.0)])
        assert cloud is not None
        zs.append(cloud.point_clouds[0].point_sets[0].points[0].z)
    assert 0.002 < float(np.std(zs)) < 0.03
