# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The deprecated FeatureExtraction class (Annex J.2): regions of interest and FeatureExtract."""

from __future__ import annotations

import math
from collections.abc import AsyncIterator

import pytest

from pyippdme import IppDmeMachine, VirtualCMM
from pyippdme.client.builders import AcquisitionPoint
from pyippdme.client.features import (
    CircleShape,
    CylinderShape,
    LineShape,
    PointShape,
    Polygon2D,
    SphereShape,
)
from pyippdme.exceptions import IppDmeServerError
from pyippdme.protocol.network import MemoryNetwork
from pyippdme.types.csy import InMemoryCsyStore

UP = (0.0, 0.0, 1.0)
X = (1.0, 0.0, 0.0)


@pytest.fixture
async def machine(network: MemoryNetwork) -> AsyncIterator[IppDmeMachine]:
    server = VirtualCMM(network=network, csy_store=InMemoryCsyStore())
    port = await server.start()
    m = await IppDmeMachine.connect("x", port, network=network)
    await m.start_session()
    try:
        yield m
    finally:
        await m.close()
        await server.close()


async def _ring(machine: IppDmeMachine, name: str = "Ring") -> None:
    """Acquire 24 points on a circle of radius 10 about (50, 40, 5), and 3 stray ones."""
    ring = [
        (50.0 + 10.0 * math.cos(a * math.pi / 12), 40.0 + 10.0 * math.sin(a * math.pi / 12), 5.0)
        for a in range(24)
    ]
    stray = [(120.0, 0.0, 5.0), (121.0, 0.0, 5.0), (122.0, 1.0, 5.0)]
    points = [AcquisitionPoint(p, UP, X) for p in (*ring, *stray)]
    await machine.raw_data.data_acquire(name, "MultiShot", "Settings", points)


async def test_a_circle_is_fitted_inside_its_region_only(machine: IppDmeMachine) -> None:
    await _ring(machine)
    await machine.feature_extraction.roi(
        "Around", CircleShape((50.0, 40.0, 5.0), UP, X, 15.0), include=True
    )
    nominal = CircleShape((50.0, 40.0, 5.0), UP, X, 10.0)
    (report,) = await machine.feature_extraction.feature_extract(
        ["Ring"], nominal, ["Around"], "Settings"
    )
    circle = dict(report)["Circle"]
    assert isinstance(circle, tuple)
    x, y, z, i, j, k, _, _, _, radius = circle
    assert (x, y, z) == pytest.approx((50.0, 40.0, 5.0), abs=0.01)  # the stray points are outside
    assert (i, j, k) == pytest.approx(UP)
    assert radius == pytest.approx(10.0, abs=0.01)


async def test_a_line_a_point_and_a_cylinder_can_be_extracted(machine: IppDmeMachine) -> None:
    track = [(float(x), 5.0, 3.0) for x in range(0, 41, 4)]
    await machine.raw_data.data_acquire(
        "Track", "MultiShot", "S", [AcquisitionPoint(p, UP, X) for p in track]
    )
    await machine.feature_extraction.roi("All", SphereShape((20.0, 5.0, 3.0), UP, X, 100.0))
    (line,) = await machine.feature_extraction.feature_extract(
        ["Track"], LineShape((20.0, 5.0, 3.0), X, 40.0), ["All"], "S"
    )
    values = dict(line)["Line"]
    assert isinstance(values, tuple)
    assert values[:3] == pytest.approx((20.0, 5.0, 3.0), abs=0.01)
    assert abs(values[3]) == pytest.approx(1.0, abs=1e-3)
    assert values[6] == pytest.approx(40.0, abs=0.05)
    (point,) = await machine.feature_extraction.feature_extract(
        ["Track"], PointShape((0.0, 0.0, 0.0), UP), ["All"], "S"
    )
    found = dict(point)["Point"]
    assert isinstance(found, tuple)
    assert found[:3] == pytest.approx((20.0, 5.0, 3.0), abs=0.01)
    await _ring(machine)
    (cylinder,) = await machine.feature_extraction.feature_extract(
        ["Ring"], CylinderShape((50.0, 40.0, 5.0), UP, X, 10.0, 8.0), ["All"], "S"
    )
    solid = dict(cylinder)["Cylinder"]
    assert isinstance(solid, tuple)
    assert solid[9] == pytest.approx(10.0, abs=0.2)


async def test_the_included_and_excluded_regions_decide_which_points_count(
    machine: IppDmeMachine,
) -> None:
    await _ring(machine)
    await machine.feature_extraction.roi(
        "Left", SphereShape((50.0, 40.0, 5.0), UP, X, 30.0), include=True
    )
    await machine.feature_extraction.roi(
        "Slice",
        Polygon2D([(40.0, 30.0, 5.0), (60.0, 30.0, 5.0), (60.0, 40.0, 5.0), (40.0, 40.0, 5.0)]),
        include=False,
    )
    points = [
        p
        async for p in machine.feature_extraction.feature_extract_points(
            ["Ring"], CircleShape((50.0, 40.0, 5.0), UP, X, 10.0), ["Left", "Slice"], "S", 0.0
        )
    ]
    assert points
    assert all(math.dist(p[:2], (50.0, 40.0)) < 31.0 for p in points)  # the stray ones are out
    assert all(p[1] > 39.0 for p in points)  # the lower half of the ring is excluded


async def test_errors_for_unknown_acquisitions_and_regions(machine: IppDmeMachine) -> None:
    await _ring(machine)
    shape = CircleShape((50.0, 40.0, 5.0), UP, X, 10.0)
    await machine.feature_extraction.roi("All", SphereShape((50.0, 40.0, 5.0), UP, X, 500.0))
    with pytest.raises(IppDmeServerError) as error:
        await machine.feature_extraction.feature_extract(["Nope"], shape, ["All"], "S")
    assert error.value.error.number == "2003"
    await machine.server.clear_all_errors()
    with pytest.raises(IppDmeServerError) as error:
        await machine.feature_extraction.feature_extract(["Ring"], shape, ["Missing"], "S")
    assert error.value.error.number == "0509"
    await machine.server.clear_all_errors()
    await machine.feature_extraction.roi(
        "Far", SphereShape((900.0, 900.0, 900.0), UP, X, 1.0), include=True
    )
    with pytest.raises(IppDmeServerError) as error:
        await machine.feature_extraction.feature_extract(["Ring"], shape, ["Far"], "S")
    assert error.value.error.number == "1006"
