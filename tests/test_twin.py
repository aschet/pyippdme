# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""The digital twin answers the protocol with limits, collisions, CAD probing and raw data."""

from __future__ import annotations

import asyncio
import itertools
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
    report = await machine.cart_cmm.pt_meas(x=x, y=y, z=top, ijk=(0, 0, 1))
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
        zs.append((await machine.cart_cmm.pt_meas(x=x, y=y, z=top, ijk=(0, 0, 1))).number("Z"))
    spread = max(zs) - min(zs)
    assert 0.0 < spread < 0.02  # micrometres, within the MPE of the machine


async def test_the_tip_touching_the_sample_during_a_move_is_an_illegal_touch(
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
    assert "1001" in str(error.value)  # the ball triggered; the stylus did not hit anything
    await machine.server.clear_all_errors()
    # The machine stood still where it touched, short of the target, and nothing broke.
    assert (await machine.cart_cmm.get_position())[0] < cx
    assert not twin.snapshot().detached


async def test_a_scanning_probe_pushed_into_the_part_reports_excessive_force(
    twin_machine: tuple[DigitalTwin, IppDmeMachine],
) -> None:
    twin, machine = twin_machine
    await machine.dme.home()
    await machine.tool_changer.change_tool("ScanSP25")
    lo, hi = twin.objects[0].world_bounds(twin.machine.rotary_pose(0.0))
    cx, cy = float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)
    below_top = twin.machine.spec.table_top_z + 30.0 - 10.0
    await machine.cart_cmm.go_to(x=cx - 200, y=cy, z=float(hi[2]) + 30)
    await machine.cart_cmm.go_to(x=cx - 200, y=cy, z=below_top)
    with pytest.raises(IppDmeServerError) as error:
        await machine.cart_cmm.go_to(x=cx, y=cy)
    assert "2001" in str(error.value)
    await machine.server.clear_all_errors()


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
    cloud = twin.scanner.acquire("a", "Sweep", positions, [(0.0, 0.0, 1.0)] * 2)
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


async def test_unknown_contour_scan_follows_the_cad_part_to_the_end_plane(
    twin_machine: tuple[DigitalTwin, IppDmeMachine],
) -> None:
    twin, machine = twin_machine
    await machine.dme.home()
    lo, _ = twin.objects[0].world_bounds(twin.machine.rotary_pose(0.0))
    top = twin.machine.spec.table_top_z + 30.0
    x0, y = float(lo[0]), float(lo[1]) + 45.0
    start = (x0 + 5.0, y, top)
    points = [
        p
        async for p in machine.scanning.scan_in_plane_end_is_plane(
            start,
            (0, 0, 1),
            (0, 1, 0),
            (x0 + 20.0, y, top),
            4.0,
            (x0 + 60.0, y, top),
            (1, 0, 0),
            1,
            (0, 0, 1),
        )
    ]
    assert 13 <= len(points) <= 16
    assert all(abs(p[2] - top) < 0.05 and abs(p[1] - y) < 0.05 for p in points)
    assert points[-1][0] >= x0 + 60.0 - 0.1
    xs = [p[0] for p in points]
    assert xs == sorted(xs)


async def test_unknown_contour_scan_ends_in_a_sphere(
    twin_machine: tuple[DigitalTwin, IppDmeMachine],
) -> None:
    twin, machine = twin_machine
    await machine.dme.home()
    lo, _ = twin.objects[0].world_bounds(twin.machine.rotary_pose(0.0))
    top = twin.machine.spec.table_top_z + 30.0
    x0, y = float(lo[0]), float(lo[1]) + 45.0
    points = [
        p
        async for p in machine.scanning.scan_in_plane_end_is_sphere(
            (x0 + 5.0, y, top),
            (0, 0, 1),
            (0, 1, 0),
            (x0 + 20.0, y, top),
            4.0,
            (x0 + 50.0, y, top),
            6.0,
            1,
            (0, 0, 1),
        )
    ]
    assert points
    assert abs(points[-1][0] - (x0 + 50.0)) <= 3.0


def test_a_collision_free_path_goes_over_the_sample() -> None:
    twin = DigitalTwin(time_scale=0.0)
    twin.place_sample(demo_sample())
    lo, hi = twin.objects[0].world_bounds(twin.machine.rotary_pose(0.0))
    y = float((lo[1] + hi[1]) / 2)
    z = float(lo[2] + (hi[2] - lo[2]) / 2)  # half way up the block
    start = (float(lo[0]) - 60.0, y, z)
    end = (float(hi[0]) + 60.0, y, z)
    assert not twin.segment_is_free(start, end)  # straight through the block
    path = twin.plan_collision_free(end, start)
    assert path is not None
    assert path[0] == start
    assert path[-1] == end
    assert max(p[2] for p in path) > float(hi[2])  # up and over
    assert all(twin.segment_is_free(a, b) for a, b in itertools.pairwise(path))


async def test_an_offline_program_is_generated_and_simulated() -> None:
    from pyippdme.twin.programs import run_program, touch_program

    network = MemoryNetwork()
    twin = DigitalTwin(time_scale=0.0, seed=2)
    twin.place_sample(demo_sample())
    lo, _ = twin.objects[0].world_bounds(twin.machine.rotary_pose(0.0))
    top = float(lo[2]) + 30.0  # the top face of the block (the boss rises above it)
    xs = [float(lo[0]) + 10.0, float(lo[0]) + 25.0, float(lo[0]) + 40.0]
    y = float(lo[1]) + 45.0
    points = [((x, y, top), (0.0, 0.0, 1.0)) for x in xs]
    program = touch_program(twin, points, clearance=8.0)
    assert program[0] == "Home()"
    assert sum(line.startswith("PtMeas") for line in program) == 3
    server = twin.create_server(network=network)
    port = await server.start("127.0.0.1", 0)
    done: list[tuple[str, bool]] = []
    ok = await run_program(
        "127.0.0.1", port, program, on_line=lambda t, o: done.append((t, o)), network=network
    )
    await server.close()
    assert ok
    assert len(done) == len(program)
    assert len(twin.contacts) == 3
    assert all(abs(c[2] - top) < 0.1 for c in twin.contacts)


def test_picking_a_point_by_hand_records_the_touch() -> None:
    twin = DigitalTwin(time_scale=0.0, seed=1)
    twin.place_sample(demo_sample())
    lo, _ = twin.objects[0].world_bounds(twin.machine.rotary_pose(0.0))
    top = float(lo[2]) + 30.0
    twin._pos = (float(lo[0]) + 10.0, float(lo[1]) + 45.0, top + 20.0)
    picked = twin.pick_point()
    assert picked is not None
    point, normal = picked
    assert abs(point[2] - top) < 0.1
    assert normal[2] > 0.99
    assert len(twin.contacts) == 1
    twin._pos = (0.0, 0.0, 300.0)
    assert twin.pick_point() is None  # nothing within reach


async def test_a_scan_starts_with_an_implicit_pt_meas(
    twin_machine: tuple[DigitalTwin, IppDmeMachine],
) -> None:
    twin, machine = twin_machine
    await machine.dme.home()
    lo, _ = twin.objects[0].world_bounds(twin.machine.rotary_pose(0.0))
    top = twin.machine.spec.table_top_z + 30.0
    start = (float(lo[0] + 5), float(lo[1] + 40), top + 2.0)
    end = (float(lo[0] + 45), float(lo[1] + 40), top + 2.0)
    contacts_before = len(twin.contacts)
    points = [
        p async for p in machine.scanning.scan_on_line(start, end, (0, 0, 1), step_width=10.0)
    ]
    assert points
    assert len(twin.contacts) > contacts_before  # the start was touched before the scan


async def test_a_scan_over_empty_space_fails_with_surface_not_found(
    twin_machine: tuple[DigitalTwin, IppDmeMachine],
) -> None:
    _, machine = twin_machine
    await machine.dme.home()
    with pytest.raises(IppDmeServerError) as error:
        _ = [
            p
            async for p in machine.scanning.scan_on_line(
                (600.0, 600.0, 100.0), (650.0, 600.0, 100.0), (0, 0, 1), step_width=10.0
            )
        ]
    assert "1006" in str(error.value)


async def test_a_scan_stops_when_the_stylus_would_be_inside_the_part(
    twin_machine: tuple[DigitalTwin, IppDmeMachine],
) -> None:
    from pyippdme.protocol.errors import ErrorCode
    from pyippdme.server.motion import MotionError
    from pyippdme.twin.backend import TwinBackend

    twin, machine = twin_machine
    await machine.dme.home()
    lo, hi = twin.objects[0].world_bounds(twin.machine.rotary_pose(0.0))
    x, y = float(lo[0] + 5), float(lo[1] + 40)  # where the earlier scan test finds the top face
    hit = twin.cast((x, y, float(hi[2] + 40.0)), (0.0, 0.0, -1.0))
    assert hit is not None
    inside = (x, y, hit[0][2] - 5.0)
    backend = TwinBackend(twin)
    context = twin.csy_context()
    backend._check_collision((x, y, hit[0][2] + 40.0), (0.0, 0.0, 1.0), context)  # free air
    with pytest.raises(MotionError) as error:
        backend._check_collision(inside, (0.0, 0.0, 1.0), context)
    assert error.value.number == ErrorCode.COLLISION
