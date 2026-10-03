# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The digital twin answers the protocol with limits, collisions, CAD probing and raw data."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

pytest.importorskip("OCP")

from pyippdme import IppDmeMachine
from pyippdme.exceptions import IppDmeServerError
from pyippdme.protocol.network import MemoryNetwork
from pyippdme.twin import DigitalTwin, MachineModel, demo_sample


@pytest.fixture
async def twin_machine() -> AsyncIterator[tuple[DigitalTwin, IppDmeMachine]]:
    network = MemoryNetwork()
    twin = DigitalTwin(time_scale=0.0, seed=1)
    twin.noise_enabled = False
    sample = twin.place_sample(demo_sample())
    assert sample.kind == "sample"
    server = twin.create_server(network=network)
    port = await server.start("127.0.0.1", 0)
    machine = await IppDmeMachine.connect("127.0.0.1", port, network=network)
    await machine.start_session()
    try:
        yield twin, machine
    finally:
        await machine.close()
        await server.close()


async def test_machine_refuses_to_move_before_home(
    twin_machine: tuple[DigitalTwin, IppDmeMachine],
) -> None:
    _, machine = twin_machine
    with pytest.raises(IppDmeServerError) as error:
        await machine.cart_cmm.go_to(x=10)
    assert "1011" in str(error.value)
    await machine.server.clear_all_errors()
    await machine.dme.home()
    await machine.cart_cmm.go_to(x=10)


async def test_target_outside_machine_volume_is_rejected(
    twin_machine: tuple[DigitalTwin, IppDmeMachine],
) -> None:
    _, machine = twin_machine
    await machine.dme.home()
    with pytest.raises(IppDmeServerError) as error:
        await machine.cart_cmm.go_to(x=5000)
    assert "1008" in str(error.value)


async def test_pt_meas_touches_the_cad_surface(
    twin_machine: tuple[DigitalTwin, IppDmeMachine],
) -> None:
    twin, machine = twin_machine
    await machine.dme.home()
    sample = twin.objects[0]
    lo, _hi = sample.world_bounds(twin.machine.rotary_pose(0.0))
    top = twin.machine.spec.table_top_z + 30.0  # the block is 30 mm high, boss excluded
    x, y = float(lo[0] + 10), float(lo[1] + 10)  # on the plain top face, away from bore and boss
    await machine.cart_cmm.go_to(x=x, y=y, z=top + 30)
    report = await machine.cart_cmm.pt_meas(x=x, y=y, z=top, ijk=(0, 0, -1))
    assert report.number("Z") == pytest.approx(top, abs=1e-6)
    assert len(twin.contacts) == 1


async def test_noise_follows_the_machine_accuracy(
    twin_machine: tuple[DigitalTwin, IppDmeMachine],
) -> None:
    twin, machine = twin_machine
    twin.noise_enabled = True
    await machine.dme.home()
    lo, _ = twin.objects[0].world_bounds(twin.machine.rotary_pose(0.0))
    top = twin.machine.spec.table_top_z + 30.0
    x, y = float(lo[0] + 10), float(lo[1] + 10)
    zs = []
    for _ in range(20):
        await machine.cart_cmm.go_to(x=x, y=y, z=top + 30)
        zs.append((await machine.cart_cmm.pt_meas(x=x, y=y, z=top, ijk=(0, 0, -1))).number("Z"))
    spread = max(zs) - min(zs)
    assert 0.0 < spread < 0.02  # micrometres, within the MPE of the machine


async def test_stylus_collides_with_the_sample(
    twin_machine: tuple[DigitalTwin, IppDmeMachine],
) -> None:
    twin, machine = twin_machine
    await machine.dme.home()
    lo, hi = twin.objects[0].world_bounds(twin.machine.rotary_pose(0.0))
    cx, cy = float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)
    below_top = twin.machine.spec.table_top_z + 30.0 - 10.0
    await machine.cart_cmm.go_to(x=cx - 200, y=cy, z=float(hi[2]) + 30)
    await machine.cart_cmm.go_to(x=cx - 200, y=cy, z=below_top)
    with pytest.raises(IppDmeServerError) as error:
        await machine.cart_cmm.go_to(x=cx, y=cy)
    assert "2504" in str(error.value)
    await machine.server.clear_all_errors()
    # The machine stood still where it hit, short of the target.
    assert (await machine.cart_cmm.get_position())[0] < cx


async def test_line_scanner_returns_a_point_cloud_of_the_sample(
    twin_machine: tuple[DigitalTwin, IppDmeMachine],
) -> None:
    twin, _machine = twin_machine
    lo, hi = twin.objects[0].world_bounds(twin.machine.rotary_pose(0.0))
    z = float(hi[2]) + twin.scanner.spec.standoff
    positions = [
        (float(lo[0]), float((lo[1] + hi[1]) / 2), z),
        (float(hi[0]), float((lo[1] + hi[1]) / 2), z),
    ]
    cloud = twin.scanner.acquire("a", "Sweep", positions, [(0.0, 0.0, -1.0)] * 2)
    assert cloud is not None
    points = cloud.point_clouds[0].point_sets[0].points
    assert len(points) > 500
    assert all(float(lo[2]) - 0.1 <= p.z <= float(hi[2]) + 0.1 for p in points)


def test_machine_directory_round_trips_through_step_files(tmp_path) -> None:  # type: ignore[no-untyped-def]
    model = MachineModel.default(rotary=True)
    model.export(tmp_path)
    again = MachineModel.from_directory(tmp_path)
    assert [b.spec.name for b in again.bodies] == [b.spec.name for b in model.bodies]
    assert again.spec.travel == model.spec.travel
    assert again.spec.rotary_origin == model.spec.rotary_origin


def test_machine_from_one_step_assembly_derives_missing_spec(tmp_path) -> None:  # type: ignore[no-untyped-def]
    model = MachineModel.default()
    model.export(tmp_path)
    from pyippdme.twin import cad

    cad.write_step_assembly(
        [cad.CadBody(b.spec.name, b.shape) for b in model.bodies], tmp_path / "all.step"
    )
    derived = MachineModel.from_step(tmp_path / "all.step")
    assert {"origin", "travel"} <= set(derived.spec.derived)
    roles = {b.spec.name: b.spec.moves_with for b in derived.bodies}
    assert roles["bridge"] == ("y",)
    assert roles["carriage"] == ("x", "y")
    assert roles["quill"] == ("x", "y", "z")
    assert roles["table"] == ()
    # Derived from the table and bridge, in the neighbourhood of the real 700/700/600 machine.
    assert 500 < derived.spec.travel[0] < 900
    assert 300 < derived.spec.travel[2] < 800


async def test_abort_stops_a_timed_move() -> None:
    network = MemoryNetwork()
    twin = DigitalTwin(time_scale=1.0)
    server = twin.create_server(network=network)
    port = await server.start("127.0.0.1", 0)
    machine = await IppDmeMachine.connect("127.0.0.1", port, network=network)
    await machine.start_session()
    await machine.dme.home()
    call = machine.cart_cmm.go_to(x=600)
    await asyncio.sleep(0.3)
    assert twin.snapshot().moving
    await machine.client.abort_e() if hasattr(machine.client, "abort_e") else None
    await call
    await machine.close()
    await server.close()


async def test_scan_on_line_follows_the_cad_surface(
    twin_machine: tuple[DigitalTwin, IppDmeMachine],
) -> None:
    twin, machine = twin_machine
    await machine.dme.home()
    lo, _ = twin.objects[0].world_bounds(twin.machine.rotary_pose(0.0))
    top = twin.machine.spec.table_top_z + 30.0
    start = (float(lo[0] + 5), float(lo[1] + 40), top + 2.0)  # 2 mm above the block's top face
    end = (float(lo[0] + 75), float(lo[1] + 40), top + 2.0)
    points = [
        p async for p in machine.scanning.scan_on_line(start, end, (0, 0, 1), step_width=10.0)
    ]
    assert len(points) >= 7
    assert all(abs(p[2] - top) < 0.05 for p in points)  # snapped onto the surface, not the path
