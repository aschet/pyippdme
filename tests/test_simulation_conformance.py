# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Simulation behaviour the standard prescribes (session defaults, tool change, state kept)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from pyippdme import IppDmeMachine, VirtualCMM
from pyippdme.client.builders import ToolAlignment
from pyippdme.exceptions import IppDmeServerError
from pyippdme.protocol.ast import BasicName, EventTag, Items, NamedValue, Number
from pyippdme.protocol.commands import CommandName
from pyippdme.protocol.network import MemoryNetwork
from pyippdme.types.csy import InMemoryCsyStore


@pytest.fixture
async def server(network: MemoryNetwork) -> AsyncIterator[VirtualCMM]:
    cmm = VirtualCMM(network=network, csy_store=InMemoryCsyStore())
    await cmm.start()
    try:
        yield cmm
    finally:
        await cmm.close()


async def _connect(server: VirtualCMM, network: MemoryNetwork) -> IppDmeMachine:
    assert server.port is not None
    machine = await IppDmeMachine.connect("x", server.port, network=network)
    await machine.start_session()
    return machine


@pytest.fixture
async def machine(server: VirtualCMM, network: MemoryNetwork) -> AsyncIterator[IppDmeMachine]:
    m = await _connect(server, network)
    try:
        yield m
    finally:
        await m.close()


# -- what is kept between clients ------------------------------------------------------


async def test_the_next_client_finds_tool_coordinate_system_and_position(
    server: VirtualCMM, network: MemoryNetwork
) -> None:
    """6.3.1 (EndSession): active tool, active coordinate system and part are preserved."""
    first = await _connect(server, network)
    await first.dme.home()
    await first.tool_changer.change_tool("AlignProbe")
    await first.cart_cmm.set_coord_system("PartCsy")
    await first.tool.align_tool((1, 0, 0), 0)
    await first.cart_cmm.go_to(x=4, y=5, z=6)
    await first.end_session()
    await first.close()

    second = await _connect(server, network)
    try:
        assert await second.tool.get_name() == "AlignProbe"
        assert await second.cart_cmm.get_coord_system() == "PartCsy"
        assert await second.cart_cmm.get_position() == (4.0, 5.0, 6.0)
        assert await second.tool.get_alignment() == pytest.approx((1.0, 0.0, 0.0))
        assert await second.dme.is_homed() is True
    finally:
        await second.close()


async def test_start_session_resets_the_documented_defaults(machine: IppDmeMachine) -> None:
    await machine.cart_cmm.on_pt_meas_report("X")
    await machine.scanning.on_scan_report("X")
    await machine.part.set_temperature(30)
    await machine.part.set_approach(4)
    await machine.end_session()
    await machine.start_session()
    # 6.3.1: the defaults are (X,Y,Z) and (X,Y,Z,Q); 6.24.2: 20 degrees, approach 0.
    assert set(await machine.cart_cmm.pt_meas(1, 2, 3)) == {"X", "Y", "Z"}
    points = [
        p
        async for p in machine.scanning.reporting("X").scan_on_line(
            (0, 0, 0), (2, 0, 0), (0, 0, 1), 1.0
        )
    ]
    assert len(points) == 3
    assert await machine.part.get_temperature() == 20.0
    assert await machine.part.get_approach() == 0.0


async def test_the_scan_report_default_includes_the_quality(machine: IppDmeMachine) -> None:
    await machine.scanning.on_scan_report("X")
    await machine.end_session()
    await machine.start_session()
    data = await machine.client.call(
        CommandName.SCAN_ON_LINE, *(Number.of(v) for v in (0, 0, 0, 2, 0, 0, 0, 0, 1, 1))
    )
    assert len(data[0].to_wire().split(",")) == 4  # X, Y, Z, Q


# -- tools --------------------------------------------------------------------------------


async def test_a_tool_change_resets_the_tool_parameters_and_unlocks_the_axes(
    machine: IppDmeMachine,
) -> None:
    """6.10.4: Act values start from Def for the new tool; 6.6.1: ChangeTool unlocks everything."""
    await machine.tool.set_parameter("PtMeasPar", "Retract", 10)
    await machine.form_tester.lock_axis("X", "Y")
    await machine.mover.enable_user()
    await machine.tool_changer.change_tool("RefTool2")
    assert (await machine.tool.get_parameter("PtMeasPar", "Retract")).act == 2.0
    await machine.cart_cmm.go_to(x=3)  # X is free again
    assert (await machine.cart_cmm.get_position())[0] == 3.0
    await machine.tool_changer.change_tool("RefTool")
    assert await machine.mover.is_user_enabled() is False  # 6.7.1: ChangeTool disables the user


async def test_set_tool_also_starts_from_the_default_parameters(machine: IppDmeMachine) -> None:
    await machine.tool.set_parameter("GoToPar", "Speed", 7)
    await machine.tool_changer.set_tool("RefTool2")
    assert (await machine.tool.get_parameter("GoToPar", "Speed")).act == 100.0


async def test_with_no_tool_the_machine_can_move_but_not_measure(machine: IppDmeMachine) -> None:
    """6.10.1: with NoTool "the machine can only move but not measure"."""
    await machine.tool_changer.change_tool("NoTool")
    await machine.cart_cmm.go_to(x=1)
    with pytest.raises(IppDmeServerError) as excinfo:
        await machine.cart_cmm.pt_meas(x=1)
    assert excinfo.value.error.number == "1503"
    await machine.server.clear_all_errors()
    with pytest.raises(IppDmeServerError) as scan_error:
        await machine.client.call(
            CommandName.SCAN_ON_LINE, *(Number.of(v) for v in (0, 0, 0, 2, 0, 0, 0, 0, 1, 1))
        )
    assert scan_error.value.error.number == "1503"


async def test_the_tool_catalog_describes_what_the_tools_can_do(machine: IppDmeMachine) -> None:
    tool_id = await machine.tool.get_id()
    assert {"GoTo", "PtMeas", "ScanOnLine", "ScanOnCurve", "PTMeasSelfCenter"} <= set(
        tool_id.basic_functions
    )
    await machine.tool_changer.change_tool("NoTool")
    assert (await machine.tool.get_id()).basic_functions == ("GoTo",)


async def test_align_tool_disables_the_user(machine: IppDmeMachine) -> None:
    await machine.tool_changer.change_tool("AlignProbe")
    await machine.mover.enable_user()
    await machine.tool.align_tool((0, 0, 1), 0)
    assert await machine.mover.is_user_enabled() is False


# -- moves ---------------------------------------------------------------------------------


async def test_pt_meas_ijk_needs_a_coordinate(machine: IppDmeMachine) -> None:
    """6.12.1: IJK(..) without any linear axis coordinate "is currently not allowed"."""
    with pytest.raises(IppDmeServerError) as excinfo:
        await machine.cart_cmm.pt_meas(ijk=(0, 0, 1))
    assert excinfo.value.error.number == "0502"


async def test_moves_reject_arguments_they_do_not_take(machine: IppDmeMachine) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await machine.client.call(CommandName.GO_TO, NamedValue("Bogus", (Number.of(1),)))
    assert excinfo.value.error.number == "0506"
    await machine.server.clear_all_errors()
    with pytest.raises(IppDmeServerError) as excinfo:
        await machine.client.call(CommandName.GO_TO, NamedValue("Sync", (Number.of(2),)))
    assert excinfo.value.error.number == "0509"
    await machine.server.clear_all_errors()
    # A proprietary argument (6.1) is ignored.
    await machine.client.call(CommandName.GO_TO, NamedValue("XXFoo", (Number.of(2),)))


async def test_go_to_moves_the_rotary_table_with_r_and_step_adds(machine: IppDmeMachine) -> None:
    await machine.cart_cmm.go_to(r=30)
    await machine.cart_cmm.step(r=15)
    assert (await machine.cart_cmm.get("R")).number("R") == 45.0


async def test_pt_meas_takes_a_part_alignment(machine: IppDmeMachine) -> None:
    from pyippdme.client.builders import PartAlignment

    await machine.cart_cmm.pt_meas(1, align_part=PartAlignment((1, 0, 0), (0, 1, 0), 0))
    assert (await machine.cart_cmm.get("R")).number("R") == pytest.approx(90.0)


async def test_pt_meas_reports_the_tool_alignment_and_angles(machine: IppDmeMachine) -> None:
    await machine.tool_changer.change_tool("AlignProbe")
    await machine.cart_cmm.on_pt_meas_report("X", "Tool.Alignment", "Tool.A", "IJKAct")
    await machine.cart_cmm.go_to(alignment=ToolAlignment((1, 0, 0)))
    report = await machine.cart_cmm.pt_meas(x=1)
    assert report.vector("Tool.Alignment") == pytest.approx((1.0, 0.0, 0.0))
    assert report.number("Tool.A") == pytest.approx(90.0)
    assert report.number("IJKAct") == 2.0


# -- OnMoveReport ----------------------------------------------------------------------


async def test_on_move_report_needs_a_time_of_at_least_a_tenth_of_a_second(
    machine: IppDmeMachine,
) -> None:
    """Table 68: "Time must be greater or equal 0.1"."""
    with pytest.raises(IppDmeServerError):
        await machine.mover.on_move_report(0.05, 1, "X")


async def test_on_move_report_sends_null_for_axes_the_tool_does_not_have(
    machine: IppDmeMachine,
) -> None:
    """6.10.2: "the server responds NULL as value of the respective information"."""
    txn = await machine.mover.on_move_report(0.5, 1, "X", "Tool.A")
    await machine.cart_cmm.go_to(x=2)
    report = await asyncio.wait_for(anext(txn.events()), 1.0)
    assert isinstance(report, Items)
    assert report.values[0] == NamedValue("X", (Number.of(2),))
    assert report.values[1] == NamedValue("Tool.A", (BasicName("NULL"),))
    assert isinstance(txn.tag, EventTag)
    await machine.server.stop_daemon(txn.tag)


# -- properties -----------------------------------------------------------------------


async def test_a_proprietary_property_is_stored(machine: IppDmeMachine) -> None:
    await machine.server.set_prop("XXLimit", 5)
    assert await machine.server.get_prop("XXLimit") == 5.0


async def test_a_proprietary_property_that_was_never_set_is_not_supported(
    machine: IppDmeMachine,
) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await machine.server.get_prop("XXNever")
    assert excinfo.value.error.number == "0506"


async def test_a_lock_axis_unlocks_the_locked_positions(machine: IppDmeMachine) -> None:
    await machine.form_tester.lock_position("RFR")
    await machine.form_tester.lock_axis("X")
    await machine.form_tester.lock_position("XFR")  # no conflict: RFR was unlocked


# -- session, home and properties -----------------------------------------------------------


async def test_end_session_stops_the_daemons(machine: IppDmeMachine) -> None:
    """6.3.1: "The method must make sure that all daemons are stopped"."""
    await machine.mover.on_move_report(0.5, 1, "X")
    await machine.end_session()
    await machine.start_session()
    # The old daemon is gone: starting another one is no longer "Daemon exists already".
    txn = await machine.mover.on_move_report(0.5, 1, "Y")
    assert isinstance(txn.tag, EventTag)


async def test_home_leaves_the_machine_at_its_home_position(machine: IppDmeMachine) -> None:
    await machine.cart_cmm.go_to(x=5, y=6, z=7)
    await machine.dme.home()
    assert await machine.cart_cmm.get_position() == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("name", ["X", "IJK", "ER", "Q", "Name", "Alignment", "Temperature"])
async def test_a_property_sent_as_a_command_is_a_bad_context(
    machine: IppDmeMachine, name: str
) -> None:
    """6.3.1.1: properties used directly, not as an argument, are a "Bad context" (0508)."""
    with pytest.raises(IppDmeServerError) as excinfo:
        await machine.client.call(name)
    assert excinfo.value.error.number == "0508"


async def test_the_scale_temperatures_can_be_set(machine: IppDmeMachine) -> None:
    """6.7.1: they can be set unless the controller corrects the temperature itself (1014)."""
    sensors = await machine.cart_cmm.get_temperature_sensors()
    assert not any(sensor.cmm_temp_correction for sensor in sensors)
    await machine.mover.set_scale_temperatures(X=21.5)
    assert (await machine.mover.get_scale_temperatures("X"))["X"] == 21.5
