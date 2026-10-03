# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Stop criteria of the unknown-contour scans (6.13.2.2) and the contour tracer, without a twin."""

from __future__ import annotations

from pyippdme.server.contour import (
    ContourConstraint,
    ContourScan,
    ContourStop,
    StopTracker,
    trace_contour,
)
from pyippdme.types.vec3 import Vec3


def _run(tracker: StopTracker, points: list[Vec3]) -> list[Vec3]:
    kept = []
    for p in points:
        done, keep = tracker.update(p)
        if keep:
            kept.append(p)
        if done:
            break
    return kept


def _line(x0: float, x1: float, step: float = 1.0) -> list[Vec3]:
    n = int(abs(x1 - x0) / step)
    return [(x0 + (x1 - x0) * i / n, 0.0, 0.0) for i in range(n + 1)]


def test_a_sphere_stops_at_the_local_minimum_of_the_distance() -> None:
    stop = ContourStop("sphere", (10.0, 1.0, 0.0), None, 6.0)  # passes 1 mm beside the centre
    kept = _run(StopTracker(stop, (0.0, 0.0, 0.0)), _line(0.0, 20.0))
    assert kept[-1] == (10.0, 0.0, 0.0)  # the minimum itself, not the point after it


def test_a_sphere_that_contains_the_start_is_left_first() -> None:
    stop = ContourStop("sphere", (0.0, 1.0, 0.0), None, 6.0)
    tracker = StopTracker(stop, (0.0, 0.0, 0.0))
    kept = _run(tracker, _line(0.0, 30.0))
    assert tracker.count == 0  # it never came back
    assert len(kept) == 31
    again = _line(0.0, 8.0) + _line(8.0, 0.0)[1:] + _line(0.0, 3.0)[1:]
    tracker = StopTracker(stop, (0.0, 0.0, 0.0))
    _run(tracker, again)
    assert tracker.count >= 1  # leaving and coming back counts


def test_a_plane_counts_only_after_the_direction_point_distance() -> None:
    stop = ContourStop("plane", (3.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    # The plane at x = 3 is crossed after 3 mm; the direction point is 5 mm away: not counted.
    tracker = StopTracker(stop, (0.0, 0.0, 0.0), skip=5.0)
    assert _run(tracker, _line(0.0, 10.0))[-1] == (10.0, 0.0, 0.0)
    assert tracker.count == 0
    far = ContourStop("plane", (8.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    tracker = StopTracker(far, (0.0, 0.0, 0.0), skip=5.0)
    assert _run(tracker, _line(0.0, 10.0))[-1] == (9.0, 0.0, 0.0)  # seen beyond x = 8


def test_n_passes_through_a_plane_are_needed() -> None:
    stop = ContourStop("plane", (5.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    path = _line(0.0, 10.0) + _line(10.0, 0.0)[1:] + _line(0.0, 10.0)[1:]
    tracker = StopTracker(stop, (0.0, 0.0, 0.0), needed=3)
    kept = _run(tracker, path)
    assert tracker.count == 3
    assert kept[-1] == (6.0, 0.0, 0.0)  # the third pass is seen on the first point beyond x = 5


def test_a_cylinder_counts_entries() -> None:
    stop = ContourStop("cylinder", (10.0, 0.0, 0.0), (0.0, 0.0, 1.0), 4.0)
    tracker = StopTracker(stop, (0.0, 0.0, 0.0))
    kept = _run(tracker, _line(0.0, 20.0))
    assert kept[-1] == (8.0, 0.0, 0.0)  # first point inside the radius 2 around x = 10


def test_tracing_a_flat_sheet_follows_it_to_the_stop_plane() -> None:
    def sheet(origin: Vec3, direction: Vec3) -> tuple[Vec3, Vec3] | None:
        if direction[2] >= 0:
            return None
        return (origin[0], origin[1], 0.0), (0.0, 0.0, 1.0)

    scan = ContourScan(
        ContourConstraint("plane", normal=(0.0, 1.0, 0.0)),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (5.0, 0.0, 0.0),
        2.0,
        ContourStop("plane", (11.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    )
    points = list(trace_contour(scan, sheet))
    assert points[-1][0] >= 11.0
    assert all(abs(p[2]) < 1e-9 and abs(p[1]) < 1e-9 for p in points)
    assert [round(p[0]) for p in points[:4]] == [0, 2, 4, 6]
