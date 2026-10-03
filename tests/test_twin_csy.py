# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""A client works in its own coordinate system; the twin places the machine in machine coordinates."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

pytest.importorskip("OCP")

from pyippdme import IppDmeMachine
from pyippdme.exceptions import IppDmeServerError
from pyippdme.protocol.network import MemoryNetwork
from pyippdme.twin import DigitalTwin, demo_sample
from pyippdme.types.csy import CoordinateTransform

# The demo block stands at x 310..390, y 320..380, its top face is at z = 20 (machine coordinates).
ORIGIN = (310.0, 320.0, 20.0)
Rig = tuple[DigitalTwin, IppDmeMachine]


@pytest.fixture
async def rig() -> AsyncIterator[Rig]:
    twin = DigitalTwin(time_scale=0.0, seed=4)
    twin.noise_enabled = False
    twin.place_sample(demo_sample())
    network = MemoryNetwork()
    server = twin.create_server(network=network)
    port = await server.start("127.0.0.1", 0)
    machine = await IppDmeMachine.connect("127.0.0.1", port, network=network)
    await machine.start_session()
    await machine.dme.home()
    try:
        yield twin, machine
    finally:
        await machine.close()
        await server.close()
        twin.toolkit.unregister()


async def test_a_part_system_moves_the_machine_where_the_client_says(rig: Rig) -> None:
    twin, machine = rig
    await machine.cart_cmm.set_csy_transformation("PartCsy", CoordinateTransform(*ORIGIN, 0, 0, 0))
    await machine.cart_cmm.set_coord_system("PartCsy")
    await machine.cart_cmm.go_to(x=10, y=10, z=40)
    assert twin.snapshot().position == pytest.approx((320.0, 330.0, 60.0))  # machine coordinates
    assert await machine.cart_cmm.get_position() == pytest.approx(
        (10.0, 10.0, 40.0)
    )  # the client's
    report = await machine.cart_cmm.pt_meas(x=10, y=10, z=0, ijk=(0, 0, 1))
    assert (report.number("X"), report.number("Y"), report.number("Z")) == pytest.approx(
        (10.0, 10.0, 0.0), abs=1e-6
    )
    # Back in the machine system the same point has machine coordinates.
    await machine.cart_cmm.set_coord_system("MachineCsy")
    position = await machine.cart_cmm.get_position()
    assert position[:2] == pytest.approx((320.0, 330.0))
    assert position[2] == pytest.approx(
        20.0 + twin.toolkit.spec("RefTool").ball_radius + 2.0, abs=0.05
    )


async def test_a_rotated_part_system_rotates_points_and_probing_directions(rig: Rig) -> None:
    twin, machine = rig
    # Phi = 90: the part's x axis is the machine's y axis, its y axis the machine's -x axis.
    await machine.cart_cmm.set_csy_transformation("PartCsy", CoordinateTransform(*ORIGIN, 0, 0, 90))
    await machine.cart_cmm.set_coord_system("PartCsy")
    # Machine point (320, 330, 20) is the part point (10, -10, 0).
    await machine.cart_cmm.go_to(x=10, y=-10, z=30)
    assert twin.snapshot().position == pytest.approx((320.0, 330.0, 50.0))
    report = await machine.cart_cmm.pt_meas(x=10, y=-10, z=0, ijk=(0, 0, 1))
    assert report.number("Z") == pytest.approx(0.0, abs=1e-6)
    # The block's +x face (machine x = 390, normal +x) is the part system's y = -80 face, and its
    # normal is the part's -y axis: both the point and the probing direction are rotated.
    await machine.cart_cmm.go_to(z=40)
    await machine.cart_cmm.go_to(x=30, y=-120)
    await machine.cart_cmm.go_to(z=-5)
    report = await machine.cart_cmm.pt_meas(x=30, y=-80, z=-5, ijk=(0, -1, 0))
    assert (report.number("X"), report.number("Y"), report.number("Z")) == pytest.approx(
        (30.0, -80.0, -5.0), abs=1e-6
    )
    assert twin.contacts[-1] == pytest.approx(
        (390.0, 350.0, 15.0), abs=1e-6
    )  # in machine coordinates


async def test_the_chain_places_the_rotary_table_system_between_machine_and_part(rig: Rig) -> None:
    twin, machine = rig
    await machine.cart_cmm.set_csy_transformation(
        "RotaryTableFixCsy", CoordinateTransform(100.0, 0.0, 0.0, 0, 0, 0)
    )
    await machine.cart_cmm.set_csy_transformation(
        "PartCsy", CoordinateTransform(0.0, 50.0, 0.0, 0, 0, 0)
    )
    await machine.cart_cmm.set_coord_system("PartCsy")
    await machine.cart_cmm.go_to(x=0, y=0, z=0)
    assert twin.snapshot().position == pytest.approx((100.0, 50.0, 0.0))  # fix, then part
    await machine.cart_cmm.set_coord_system("RotaryTableFixCsy")
    assert await machine.cart_cmm.get_position() == pytest.approx(
        (0.0, 50.0, 0.0)
    )  # part is above it


async def test_limits_apply_to_machine_coordinates_not_the_clients(rig: Rig) -> None:
    _, machine = rig
    await machine.cart_cmm.set_csy_transformation(
        "PartCsy", CoordinateTransform(600.0, 0.0, 0.0, 0, 0, 0)
    )
    await machine.cart_cmm.set_coord_system("PartCsy")
    await machine.cart_cmm.go_to(x=-500)  # machine x = 100: inside
    with pytest.raises(IppDmeServerError) as error:
        await machine.cart_cmm.go_to(x=200)  # machine x = 800: outside the 700 mm travel
    assert "1008" in str(error.value)


async def test_home_leaves_the_machine_at_its_origin_in_the_clients_coordinates() -> None:
    twin = DigitalTwin(time_scale=0.0)
    network = MemoryNetwork()
    server = twin.create_server(network=network)
    port = await server.start("127.0.0.1", 0)
    machine = await IppDmeMachine.connect("127.0.0.1", port, network=network)
    await machine.start_session()
    await machine.cart_cmm.set_csy_transformation(
        "PartCsy", CoordinateTransform(100.0, 200.0, 0.0, 0, 0, 0)
    )
    await machine.cart_cmm.set_coord_system("PartCsy")
    await machine.dme.home()
    assert await machine.cart_cmm.get_position() == pytest.approx((-100.0, -200.0, 0.0))
    await machine.close()
    await server.close()
    twin.toolkit.unregister()
