# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tools, probing cycle, safety, qualification and tool changes of the twin, over the protocol."""

from __future__ import annotations

from collections.abc import AsyncIterator

import numpy as np
import pytest

pytest.importorskip("OCP")

from pyippdme import IppDmeMachine
from pyippdme.client.builders import AcquisitionPoint
from pyippdme.exceptions import IppDmeServerError
from pyippdme.protocol.network import MemoryNetwork
from pyippdme.twin import DigitalTwin, demo_sample
from pyippdme.twin.artifact import build_check_artifact, build_reference_sphere
from pyippdme.twin.check import CheckOptions, run_check_program
from pyippdme.types.tool_id import ToolIdOptical2DRs, ToolIdTactileMeasuring

TOP = -10.0 + 30.0  # top of the demo block (the table is at z = -10, the block is 30 mm high)
Rig = tuple[DigitalTwin, IppDmeMachine]


async def _rig(twin: DigitalTwin) -> AsyncIterator[Rig]:
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


@pytest.fixture
async def rig() -> AsyncIterator[Rig]:
    twin = DigitalTwin(time_scale=0.0, seed=7)
    twin.place_sample(demo_sample())
    async for item in _rig(twin):
        yield item


async def _error(call: object) -> str:
    with pytest.raises(IppDmeServerError) as error:
        await call  # type: ignore[misc]
    return str(error.value)


async def test_the_protocol_offers_all_tools_and_their_descriptions(rig: Rig) -> None:
    _, machine = rig
    tools = await machine.tool_changer.enum_tools()
    assert {"RefTool", "StarXP", "ScanSP25", "RevoHeadTouch", "LaserLine", "Camera2D"} <= set(tools)
    await machine.tool_changer.change_tool("ScanSP25")
    assert isinstance(await machine.tool.get_id(), ToolIdTactileMeasuring)
    await machine.tool_changer.change_tool("Camera2D")
    assert isinstance(await machine.tool.get_id(), ToolIdOptical2DRs)


async def test_a_tool_that_lacks_a_function_gets_error_2002(rig: Rig) -> None:
    twin, machine = rig
    await machine.tool_changer.change_tool("LaserLine")
    assert "2002" in await _error(machine.cart_cmm.pt_meas(x=100, y=100, z=10, ijk=(0, 0, 1)))
    await machine.server.clear_all_errors()
    await machine.tool_changer.change_tool("StarDown")  # a touch probe cannot scan
    assert "2002" in await _error(
        _drain(machine.scanning.scan_on_line((0, 0, 0), (10, 0, 0), (0, 0, 1), 1.0))
    )
    await machine.server.clear_all_errors()
    assert twin.toolkit.spec("StarDown").mode == "touch"


async def _drain(stream: object) -> None:
    async for _ in stream:  # type: ignore[attr-defined]
        pass


async def test_the_probing_cycle_stops_one_ball_radius_off_the_surface(rig: Rig) -> None:
    twin, machine = rig
    lo, _ = twin.objects[0].world_bounds(twin.machine.rotary_pose(0.0))
    x, y = float(lo[0]) + 10, float(lo[1]) + 10
    await machine.cart_cmm.go_to(x=x, y=y, z=TOP + 30)
    report = await machine.cart_cmm.pt_meas(x=x, y=y, z=TOP, ijk=(0, 0, 1))
    assert report.number("Z") == pytest.approx(TOP, abs=0.05)
    # The machine rests at ball radius + retract (2 mm default) above the surface.
    ball = twin.toolkit.spec("RefTool").ball_radius
    assert (await machine.cart_cmm.get_position())[2] == pytest.approx(TOP + ball + 2.0, abs=0.05)


async def test_a_surface_beyond_approach_and_search_is_not_found(rig: Rig) -> None:
    twin, machine = rig
    lo, _ = twin.objects[0].world_bounds(twin.machine.rotary_pose(0.0))
    x, y = float(lo[0]) + 10, float(lo[1]) + 10
    await machine.cart_cmm.go_to(x=x, y=y, z=TOP + 30)
    far = (
        TOP + 12.0
    )  # the nominal point is 12 mm above the real surface: approach 5 + search 2 < 12
    assert "1006" in await _error(machine.cart_cmm.pt_meas(x=x, y=y, z=far, ijk=(0, 0, 1)))
    await machine.server.clear_all_errors()
    # A larger search finds it.
    await machine.server.set_prop("Tool.PtMeasPar.Search", 12.0)
    report = await machine.cart_cmm.pt_meas(x=x, y=y, z=far, ijk=(0, 0, 1))
    assert report.number("Z") == pytest.approx(TOP, abs=0.05)


async def test_emergency_stop_and_air_pressure_stop_the_machine_and_lose_the_reference(
    rig: Rig,
) -> None:
    twin, machine = rig
    twin.set_estop(True)
    assert "1000" in await _error(machine.cart_cmm.go_to(x=10))
    await machine.server.clear_all_errors()
    twin.set_estop(False)
    assert not await machine.dme.is_homed()  # the reference is lost
    assert "1011" in await _error(machine.cart_cmm.go_to(x=10))
    await machine.server.clear_all_errors()
    twin.set_air_ok(False)
    assert "1005" in await _error(machine.dme.home())  # Home needs the air supply
    await machine.server.clear_all_errors()
    twin.set_air_ok(True)
    await machine.dme.home()
    await machine.cart_cmm.go_to(x=10)


async def test_a_crash_makes_the_module_break_away_until_it_is_changed(rig: Rig) -> None:
    twin, machine = rig
    lo, hi = twin.objects[0].world_bounds(twin.machine.rotary_pose(0.0))
    cx, cy = float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)
    await machine.cart_cmm.go_to(x=cx - 200, y=cy, z=TOP - 10)
    assert "2504" in await _error(machine.cart_cmm.go_to(x=cx, y=cy))
    await machine.server.clear_all_errors()
    assert twin.snapshot().detached
    assert "1501" in await _error(machine.cart_cmm.pt_meas(x=cx - 100, y=cy, z=TOP, ijk=(0, 0, 1)))
    await machine.server.clear_all_errors()
    await machine.tool_changer.change_tool("RefTool2")
    await machine.tool_changer.change_tool("RefTool")
    assert not twin.snapshot().detached


async def test_a_tool_must_be_qualified_on_a_reference_sphere(rig: Rig) -> None:
    twin, machine = rig
    assert not twin.snapshot().qualified
    assert "1006" in await _error(machine.tool.re_qualify())  # no sphere in the volume yet
    await machine.server.clear_all_errors()
    sphere = build_reference_sphere()
    sphere.pose[:3, 3] = (150.0, 150.0, -10.0)
    twin.add_object(sphere)
    await machine.tool.re_qualify()
    assert twin.snapshot().qualified
    await machine.tool_changer.change_tool("StarDown")
    assert not twin.snapshot().qualified  # a different tool is qualified separately


async def test_tips_of_a_star_share_the_stylus_but_not_the_tool_centre_point(rig: Rig) -> None:
    twin, machine = rig
    tcp = (200.0, 200.0, 100.0)
    down = twin.placement(tcp, "StarDown")
    side = twin.placement(tcp, "StarXP")
    assert side.pivot[0] == pytest.approx(down.pivot[0] - twin.toolkit.spec("StarXP").arm_length)
    assert twin.toolkit.spec("StarXP").rack_key == twin.toolkit.spec("StarDown").rack_key == "Star"
    # The quill follows the pivot: a longer tool lifts it.
    assert twin.head_shift(tcp, twin.placement(tcp, "IndexedTP200"))[2] > 50.0


async def test_an_indexing_head_snaps_and_is_qualified_per_position(rig: Rig) -> None:
    twin, machine = rig
    await machine.tool_changer.change_tool("IndexedTP200")
    await machine.tool.align_tool((0.2, 0.0, 0.98), 0.0)
    position = twin.snapshot().head_position
    assert position is not None and position[0] % 7.5 == 0.0
    await machine.tool.align_tool((0.0, 0.0, 1.0), 0.0)
    assert twin.snapshot().head_position == (0.0, 0.0)


async def test_tool_change_drives_to_the_rack_and_back() -> None:
    twin = DigitalTwin(time_scale=3000.0, seed=1)
    async for _, machine in _rig(twin):
        await machine.cart_cmm.go_to(x=300, y=300, z=150)
        events: list[str] = []
        twin.add_listener(
            lambda e: events.append(e.data.get("stage", "")) if e.kind == "tool_change" else None
        )
        await machine.tool_changer.change_tool("ScanSP25")
        assert events == ["released", "taken"]
        assert (await machine.cart_cmm.get_position()) == pytest.approx(
            (300.0, 300.0, 150.0), abs=1e-6
        )
        assert twin.snapshot().tool_name == "ScanSP25"
        stored = {i.key.split(":")[1] for i in twin.draw_items() if i.kind == "stored"}
        assert (
            "RefTool" in stored and "ScanSP25" not in stored
        )  # the old module is back in its port


async def test_check_artifact_touch_program_measures_the_reference_sphere() -> None:
    twin = DigitalTwin(time_scale=0.0, seed=11)
    artifact = build_check_artifact()
    twin.place_sample(artifact)
    async for _, machine in _rig(twin):
        data = artifact.artifact.placed(artifact.world_pose(np.eye(4)))  # type: ignore[union-attr]
        data.touch = data.touch[:25]  # the 25 points of the reference sphere
        await run_check_program(machine, data, CheckOptions(modes=("touch",)))
        results = {
            r.quantity: r
            for r in twin.evaluate(artifact)["touch"]
            if r.feature == "reference sphere"
        }
        assert results["diameter"].measured == pytest.approx(25.0, abs=0.01)
        assert results["diameter"].ok


async def test_optical_sweep_over_the_artifact_returns_points_in_the_laser_mode() -> None:
    twin = DigitalTwin(time_scale=0.0, seed=2)
    artifact = build_check_artifact()
    twin.place_sample(artifact)
    async for _, machine in _rig(twin):
        await machine.tool_changer.change_tool("LaserLine")
        top = artifact.artifact.top + float(artifact.pose[2, 3])  # type: ignore[union-attr]
        z = top + 60.0
        x0 = float(artifact.pose[0, 3])
        y = float(artifact.pose[1, 3]) + 180.0
        points = [
            AcquisitionPoint((x0 + 130.0, y, z), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
            AcquisitionPoint((x0 + 190.0, y, z), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        ]
        await machine.raw_data.data_acquire("a", "Sweep", "default", points)
        assert len(twin.points["laser"]) > 1000
        zs = np.array([p[2] for p in twin.points["laser"]])
        assert zs.max() == pytest.approx(top + 10.0, abs=0.05)  # the dome's peak
