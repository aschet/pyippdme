# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for the object-model methods that wrap the less common server commands."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from pyippdme import IppDmeMachine, VirtualCMM
from pyippdme.exceptions import IppDmeServerError
from pyippdme.protocol.network import MemoryNetwork
from pyippdme.rawdata.transfer import read_samples
from pyippdme.simulation.classes.tool_class import DEFAULT_TOOL_COLLECTION
from pyippdme.types.csy import InMemoryCsyStore


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


# -- Server ------------------------------------------------------------------


async def test_clear_all_errors_recovers_from_a_severe_error(machine: IppDmeMachine) -> None:
    with pytest.raises(IppDmeServerError):
        await machine.client.call("NoSuchCommand")
    assert await machine.server.get_err_status() is True
    status = await machine.server.get_xtd_err_status()
    assert status["ActiveError"] == 501.0
    await machine.server.clear_all_errors()
    assert await machine.server.get_err_status() is False
    assert await machine.dme.get_machine_class()


async def test_get_xtd_err_status_reports_status_lines(machine: IppDmeMachine) -> None:
    status = await machine.server.get_xtd_err_status()
    assert status == {"IsHomed": 0.0}
    await machine.dme.home()
    assert (await machine.server.get_xtd_err_status())["IsHomed"] == 1.0


async def test_enum_name_spaces_is_empty_without_proprietary_namespaces(
    machine: IppDmeMachine,
) -> None:
    assert await machine.server.enum_name_spaces() == ()


async def test_properties_can_be_written_and_read_by_name(machine: IppDmeMachine) -> None:
    await machine.server.set_prop("XXCustom.Number", 1.5)
    await machine.server.set_prop("XXCustom.Text", "hello")
    assert await machine.server.get_prop("XXCustom.Number") == 1.5
    assert await machine.server.get_prop_e("XXCustom.Text") == "hello"


async def test_session_commands_are_available_on_the_server_namespace(
    machine: IppDmeMachine,
) -> None:
    await machine.server.end_session()
    await machine.server.start_session()


async def test_abort_e_completes_when_nothing_is_running(machine: IppDmeMachine) -> None:
    await machine.server.abort_e()


async def test_stop_all_daemons_without_a_daemon_raises(machine: IppDmeMachine) -> None:
    with pytest.raises(IppDmeServerError):
        await machine.server.stop_all_daemons()


# -- Mover -------------------------------------------------------------------


async def test_move_report_daemon_delivers_reports_until_stopped(machine: IppDmeMachine) -> None:
    transaction = await machine.mover.on_move_report(0.1, 1.0, "X", "Y")
    await machine.cart_cmm.go_to(x=5, y=2)
    events = transaction.events()
    report = await asyncio.wait_for(anext(events), 1.0)
    assert report.to_wire() == "X(5),Y(2)"
    await machine.server.stop_daemon(transaction.tag)  # type: ignore[arg-type]


async def test_prioritized_move_report_daemon_can_be_started(machine: IppDmeMachine) -> None:
    transaction = await machine.mover.on_move_report(0.1, 1.0, "X", prioritized=True)
    assert transaction.tag.to_wire().startswith("E")
    await machine.server.stop_all_daemons()


async def test_scale_temperature_commands(machine: IppDmeMachine) -> None:
    await machine.mover.update_scale_temperatures()
    await machine.mover.set_temperature_compensation_origin(x=1.0, z=3.0)
    with pytest.raises(IppDmeServerError):
        await machine.mover.set_temperature_compensation_origin()  # at least one axis


# -- CartCmm -----------------------------------------------------------------


async def test_pt_meas_report_selects_the_reported_values(machine: IppDmeMachine) -> None:
    await machine.cart_cmm.on_pt_meas_report("X", "Z")
    assert await machine.cart_cmm.pt_meas(x=1, y=2, z=3) == {"X": 1.0, "Z": 3.0}


async def test_self_centering_measurements(machine: IppDmeMachine) -> None:
    measured = await machine.cart_cmm.pt_meas_self_center(1, 2, 3, ijk=(0, 0, 1))
    assert measured == {"X": 1.0, "Y": 2.0, "Z": 3.0}
    locked = await machine.cart_cmm.pt_meas_self_center_locked(
        1, 2, 3, ijk=(0, 0, 1), lmn=(1, 0, 0)
    )
    assert locked == {"X": 1.0, "Y": 2.0, "Z": 3.0}
    with pytest.raises(IppDmeServerError):
        await machine.cart_cmm.pt_meas_self_center_locked(1, 2, 3, ijk=(0, 0, 1), lmn=(0, 0, 1))


# -- Tool --------------------------------------------------------------------


async def test_tool_alignment_commands(machine: IppDmeMachine) -> None:
    await machine.tool_changer.change_tool("AlignProbe")
    await machine.tool.align_tool((0, 0, 1), 0.0)
    assert await machine.tool.calc_tool_alignment() == pytest.approx((0.0, 0.0, 1.0))
    angles = await machine.tool.calc_tool_angles((0, 0, 1))
    assert set(angles) == {"A", "B"}
    assert await machine.tool.calc_tool_angles((0, 0, 1), (1, 0, 0)) != {}
    await machine.tool.align_tool((0, 0, 1), 0.0, (1, 0, 0), 0.0)
    assert len(await machine.tool.calc_tool_alignment()) == 6


async def test_align_tool_requires_secondary_and_beta_together(machine: IppDmeMachine) -> None:
    await machine.tool_changer.change_tool("AlignProbe")
    with pytest.raises(ValueError, match="together"):
        await machine.tool.align_tool((0, 0, 1), 0.0, secondary=(1, 0, 0))


async def test_optimize_flag_and_smallest_angle(machine: IppDmeMachine) -> None:
    await machine.tool_changer.change_tool("AlignProbe")
    assert await machine.tool.is_optimize_enabled() is False
    await machine.tool.enable_optimize()
    assert await machine.tool.is_optimize_enabled() is True
    await machine.tool.disable_optimize()
    assert await machine.tool.is_optimize_enabled() is False
    await machine.tool.use_smallest_angle_to_align_tool(True)


async def test_tool_geometry_queries(machine: IppDmeMachine) -> None:
    await machine.tool_changer.change_tool("AlignProbe")
    assert await machine.tool.avr_radius() == 0.0
    assert await machine.tool.get_avr_offsets() == (0.0, 0.0, 0.0)
    assert await machine.tool.get_collision_volume() == ()
    assert await machine.tool.get_alignment_volume() == ()
    await machine.tool.re_qualify()


# -- ToolChanger -------------------------------------------------------------


async def test_tool_collections(machine: IppDmeMachine) -> None:
    tools = await machine.tool_changer.enum_tool_collection(DEFAULT_TOOL_COLLECTION)
    assert ("AlignProbe", "Tool") in tools
    assert await machine.tool_changer.enum_all_tool_collections(DEFAULT_TOOL_COLLECTION) == tools
    await machine.tool_changer.open_tool_collection(DEFAULT_TOOL_COLLECTION)
    with pytest.raises(IppDmeServerError):
        await machine.tool_changer.enum_tool_collection("NoSuchCollection")


# -- Scanning ----------------------------------------------------------------


async def test_scan_on_circle_and_helix(machine: IppDmeMachine) -> None:
    arc = [
        point
        async for point in machine.scanning.scan_on_circle(
            (0, 0, 0), (5, 0, 0), (0, 0, 1), 90, 0, 30
        )
    ]
    assert len(arc) == 4
    assert arc[-1] == pytest.approx((0.0, 5.0, 0.0), abs=1e-9)
    helix = [
        point
        async for point in machine.scanning.scan_on_helix(
            (0, 0, 0), (5, 0, 0), (0, 0, 1), 90, 0, 30, 2.0
        )
    ]
    assert len(helix) == 4
    assert helix[-1][2] > 0


async def test_scan_on_circle_acknowledged_is_the_scan_not_the_setup(
    machine: IppDmeMachine,
) -> None:
    scan = machine.scanning.scan_on_circle((0, 0, 0), (5, 0, 0), (0, 0, 1), 90, 0, 30, True)
    await scan.acknowledged()
    transaction = await scan.transaction()
    assert (await transaction.wait_complete())[0] is not None


async def test_unknown_contour_scans(machine: IppDmeMachine) -> None:
    origin, x_axis, z_axis, y_axis = (0, 0, 0), (1, 0, 0), (0, 0, 1), (0, 1, 0)
    near = (0.1, 0, 0)
    end = (10, 0, 0)
    scans = [
        machine.scanning.scan_in_plane_end_is_sphere(
            origin, x_axis, z_axis, near, 2.0, end, 1.0, 1, x_axis
        ),
        machine.scanning.scan_in_plane_end_is_plane(
            origin, x_axis, z_axis, near, 2.0, end, x_axis, 1, x_axis
        ),
        machine.scanning.scan_in_plane_end_is_cyl(
            origin, x_axis, z_axis, near, 2.0, end, y_axis, 1.0, 1, x_axis
        ),
        machine.scanning.scan_in_cyl_end_is_sphere(
            (0, 5, 0), z_axis, origin, x_axis, near, 2.0, end, 1.0, 1, x_axis
        ),
        machine.scanning.scan_in_cyl_end_is_plane(
            (0, 5, 0), z_axis, origin, x_axis, near, 2.0, end, x_axis, 1, x_axis
        ),
    ]
    for scan in scans:
        points = [point async for point in scan]
        assert points
        assert points[0] == (0.0, 0.0, 0.0)


async def test_scan_hints_and_density_are_accepted(machine: IppDmeMachine) -> None:
    await machine.scanning.scan_on_line_hint(5.0, 0.1)
    await machine.scanning.scan_on_circle_hint(0.5, 0.1)
    await machine.scanning.scan_on_curve_hint(0.1, 2.0)
    await machine.scanning.scan_on_curve_density(dis=0.5, angle=5.0, angle_base_length=1.0)
    await machine.scanning.scan_on_curve_density(at_nominals=True)
    await machine.scanning.scan_unknown_hint(2.0)
    await machine.scanning.scan_unknown_density(dis=0.5)
    with pytest.raises(IppDmeServerError):
        await machine.scanning.scan_unknown_hint(0.0)


# -- Raw data ----------------------------------------------------------------


async def test_raw_data_binary_transfer_through_the_model(
    machine: IppDmeMachine, network: MemoryNetwork
) -> None:
    await machine.raw_data.data_acquire("Acq1", "SingleShot", "Settings")
    await machine.raw_data.raw_data_bin_setup("double", 5000)
    transfer = machine.raw_data.get_raw_data_bin("Acq1")
    samples = await read_samples("x", 5000, "double", network=network)
    await transfer
    assert samples.shape == (1, 3)
    assert samples[0] == pytest.approx((0.0, 0.0, 0.0), abs=0.05)


async def test_raw_data_float_transfer_is_unpacked_as_float32(
    machine: IppDmeMachine, network: MemoryNetwork
) -> None:
    await machine.raw_data.data_acquire("Acq1", "SingleShot", "Settings")
    await machine.raw_data.raw_data_bin_setup("float", 5001, live_mode=True)
    transfer = machine.raw_data.get_raw_data_bin("Acq1")
    samples = await read_samples("x", 5001, "float", network=network)
    await transfer
    assert samples.shape == (1, 3)


async def test_read_samples_gives_up_when_nothing_listens(network: MemoryNetwork) -> None:
    with pytest.raises(ConnectionRefusedError):
        await read_samples("x", 5002, "double", network=network, timeout=0.05)


# -- Tool and FoundTool properties ---------------------------------------------


async def test_tool_properties(machine: IppDmeMachine) -> None:
    assert await machine.tool.get_name() == "RefTool"
    assert await machine.tool.get_collection() == DEFAULT_TOOL_COLLECTION
    assert await machine.tool.get_last_qualified() == "00000000T000000Z"
    tool_id = await machine.tool.get_id()
    assert tool_id.id == "RefTool"
    with pytest.raises(IppDmeServerError):
        await machine.tool.get_alignment()  # the default tool is not alignable


async def test_alignment_property_of_an_alignable_tool(machine: IppDmeMachine) -> None:
    await machine.tool_changer.change_tool("AlignProbe")
    await machine.tool.align_tool((0, 0, 1), 0.0)
    assert await machine.tool.get_alignment() == pytest.approx((0.0, 0.0, 1.0))
    await machine.tool.re_qualify()
    assert await machine.tool.get_last_qualified() != "00000000T000000Z"


async def test_found_tool_properties(machine: IppDmeMachine) -> None:
    assert await machine.found_tool.get_name() == "UnDefTool"
    await machine.tool_changer.find_tool("RefTool2")
    assert await machine.found_tool.get_name() == "RefTool2"
    assert await machine.found_tool.get_collection() == DEFAULT_TOOL_COLLECTION
    assert (await machine.found_tool.get_id()).id == "RefTool2"
    assert await machine.found_tool.get_last_qualified() == "00000000T000000Z"


# -- Values with several numbers ---------------------------------------------


async def test_pt_meas_returns_the_probing_direction_when_it_is_reported(
    machine: IppDmeMachine,
) -> None:
    await machine.cart_cmm.on_pt_meas_report("X", "Y", "Z", "IJK")
    report = await machine.cart_cmm.pt_meas(1, 2, 3, ijk=(0, 0, 1))
    assert report.number("Z") == 3.0
    assert len(report.vector("IJK")) == 3


async def test_scans_can_report_more_than_the_position(machine: IppDmeMachine) -> None:
    scan = machine.scanning.reporting("X", "Y", "Z", "IJK")
    points = [p async for p in scan.scan_on_line((0, 0, 0), (4, 0, 0), (0, 0, 1), 2.0)]
    assert [p.number("X") for p in points] == [0.0, 2.0, 4.0]
    assert all(p.vector("IJK") == (0.0, 0.0, 1.0) for p in points)


async def test_a_raw_data_file_that_cannot_be_written_is_error_2005(
    machine: IppDmeMachine, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = tmp_path / "a_file"
    blocker.write_text("not a directory")
    monkeypatch.setattr(
        "pyippdme.simulation.classes.rawdata_class.default_raw_data_directory",
        lambda: blocker / "raw",
    )
    await machine.raw_data.data_acquire("Acq1", "SingleShot", "Settings")
    with pytest.raises(IppDmeServerError) as error:
        await machine.raw_data.get_raw_data_file("Acq1")
    assert error.value.error.number == "2005"
