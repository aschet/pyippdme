# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for the high-level, typed IppDmeMachine client (pyippdme.client.model)."""

from __future__ import annotations

import pytest

from pyippdme import IppDmeMachine
from pyippdme.client import IppDmeClient
from pyippdme.client.model import AcquisitionPoint, CurvePoint
from pyippdme.exceptions import IppDmeServerError
from pyippdme.types.csy import CoordinateTransform


@pytest.fixture
async def machine(client: IppDmeClient) -> IppDmeMachine:
    m = IppDmeMachine(client)
    await m.start_session()
    return m


async def test_dme_namespace(machine: IppDmeMachine) -> None:
    assert await machine.dme.get_dme_version() == "2.5"
    assert "GoTo" in await machine.dme.get_supported_commands()
    args = await machine.dme.get_supported_arguments("GoTo")
    assert args == {"Positions": "enum", "Sync": "int"}
    assert "CartCMM" in await machine.dme.get_machine_class()
    assert await machine.dme.is_homed() is False
    await machine.dme.home()
    assert await machine.dme.is_homed() is True


async def test_cart_cmm_motion_and_position(machine: IppDmeMachine) -> None:
    await machine.cart_cmm.go_to(x=1.0, y=2.0, z=3.0)
    assert await machine.cart_cmm.get_position() == (1.0, 2.0, 3.0)
    assert await machine.cart_cmm.get("X") == {"X": 1.0}

    result = await machine.cart_cmm.pt_meas(x=5.0)
    assert result["X"] == 5.0


async def test_cart_cmm_step_and_circle_spiral(machine: IppDmeMachine) -> None:
    await machine.cart_cmm.go_to(x=10.0, y=0.0, z=0.0)
    await machine.cart_cmm.step(x=1.0, z=2.0)
    assert await machine.cart_cmm.get_position() == (11.0, 0.0, 2.0)

    await machine.cart_cmm.go_to(x=10.0, y=0.0, z=0.0)
    await machine.cart_cmm.go_to_on_circle((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 10.0, 0.0))
    assert await machine.cart_cmm.get_position() == (0.0, 10.0, 0.0)

    await machine.cart_cmm.go_to_on_spiral((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 5.0, 3.0))
    assert await machine.cart_cmm.get_position() == (0.0, 5.0, 3.0)


async def test_cart_cmm_coord_system(machine: IppDmeMachine) -> None:
    await machine.cart_cmm.set_coord_system("PartCsy")
    assert await machine.cart_cmm.get_coord_system() == "PartCsy"


async def test_cart_cmm_csy_transformation_round_trip(machine: IppDmeMachine) -> None:
    t = CoordinateTransform(1, 2, 3, 45, 90, 180)
    await machine.cart_cmm.set_csy_transformation("PartCsy", t)
    assert await machine.cart_cmm.get_csy_transformation("PartCsy") == t


async def test_cart_cmm_named_csy_transformation_and_enum(machine: IppDmeMachine) -> None:
    t = CoordinateTransform.identity()
    await machine.cart_cmm.save_named_csy_transformation("Fixture1", t)
    assert await machine.cart_cmm.get_named_csy_transformation("Fixture1") == t
    assert await machine.cart_cmm.enum_coord_systems() == ("Fixture1",)
    await machine.cart_cmm.delete_coord_system("Fixture1")
    assert await machine.cart_cmm.enum_coord_systems() == ()


async def test_cart_cmm_temperature_sensors(machine: IppDmeMachine) -> None:
    sensors = await machine.cart_cmm.get_temperature_sensors()
    assert {s.name for s in sensors} == {
        "PartSensor",
        "CMMSensor",
        "XAxisSensor",
        "YAxisSensor",
        "ZAxisSensor",
    }
    x_axis = next(s for s in sensors if s.name == "XAxisSensor")
    assert x_axis.kind == "Mover"
    assert x_axis.scale_axis == "X"

    assert await machine.cart_cmm.read_temperature_sensor("CMMSensor") == 20.0
    assert await machine.cart_cmm.read_temperature_sensor("Bogus") == -273.0

    readings = await machine.cart_cmm.read_all_temperatures()
    assert readings["CMMSensor"] == 20.0
    assert len(readings) == 5


async def test_cart_cmm_save_active_and_load(machine: IppDmeMachine) -> None:
    t = CoordinateTransform(5, 6, 7, 0, 0, 0)
    await machine.cart_cmm.set_csy_transformation("PartCsy", t)
    await machine.cart_cmm.save_active_coord_system("MyPart")
    await machine.cart_cmm.set_csy_transformation("PartCsy", CoordinateTransform.identity())
    await machine.cart_cmm.load_coord_system("MyPart")
    assert await machine.cart_cmm.get_csy_transformation("PartCsy") == t


async def test_tool_parameter_get_set(machine: IppDmeMachine) -> None:
    param = await machine.tool.get_parameter("GoToPar", "Speed")
    assert param.act == pytest.approx(100.0)
    await machine.tool.set_parameter("GoToPar", "Speed", 50.0)
    updated = await machine.tool.get_parameter("GoToPar", "Speed")
    assert updated.act == 50.0


async def test_tool_id_and_name_and_alignable(machine: IppDmeMachine) -> None:
    assert await machine.tool.get_name() == "RefTool"
    assert await machine.tool.is_alignable() is False
    tool = await machine.tool.get_id()
    assert tool.id == "RefTool"


async def test_scanning_scan_on_line_yields_points(machine: IppDmeMachine) -> None:
    points = [p async for p in machine.scanning.scan_on_line((0, 0, 0), (4, 0, 0), (0, 0, 1), 2.0)]
    assert points[0] == (0.0, 0.0, 0.0)
    assert points[-1] == (4.0, 0.0, 0.0)
    assert len(points) >= 2


async def test_scanning_scan_on_curve_yields_nominal_points(machine: IppDmeMachine) -> None:
    curve = [
        CurvePoint((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1),
        CurvePoint((100.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1),
    ]
    points = [p async for p in machine.scanning.scan_on_curve(curve)]
    assert points == [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0)]


async def test_form_tester_center_part(machine: IppDmeMachine) -> None:
    assert await machine.form_tester.center_part(0.0, 0.0, 0.0, 5.0) is True
    assert await machine.form_tester.center_part(100.0, 0.0, 0.0, 5.0) is False


async def test_form_tester_lock_axis(machine: IppDmeMachine) -> None:
    await machine.form_tester.lock_axis("X")
    await machine.cart_cmm.go_to(x=99.0)
    assert await machine.cart_cmm.get("X") == {"X": 0.0}


async def test_mover_namespace(machine: IppDmeMachine) -> None:
    assert await machine.mover.is_user_enabled() is True
    await machine.mover.disable_user()
    assert await machine.mover.is_user_enabled() is False
    assert await machine.mover.enumerate_mover_axes() == ("X", "Y", "Z")
    await machine.mover.set_scale_temperatures(X=21.0)
    assert await machine.mover.get_scale_temperatures("X") == {"X": 21.0}


async def test_rotary_table_align_and_enable(machine: IppDmeMachine) -> None:
    await machine.rotary_table.enable_rotary_table_var_csy(True)
    part_v, machine_v = await machine.rotary_table.align_part((1, 0, 0), (0, 1, 0), 0.0)
    assert part_v == pytest.approx((1.0, 0.0, 0.0))
    assert machine_v == pytest.approx((0.0, 1.0, 0.0))


async def test_raw_data_namespace_round_trip(machine: IppDmeMachine) -> None:
    struct_data = await machine.raw_data.get_adv_data_struct()
    assert struct_data.return_technologies == {"SocBin", "ShaMem", "File"}

    await machine.raw_data.data_acquire("Acq1", "SingleShot", "Settings1")
    url = await machine.raw_data.get_raw_data_file("Acq1")
    assert url.startswith("file://")
    await machine.raw_data.del_raw_data_file("Acq1")

    await machine.raw_data.data_acquire("Acq2", "SingleShot", "Settings1")
    _name, offset, size = await machine.raw_data.get_raw_data_sha_mem("Acq2")
    assert offset == 0
    assert size == 24
    await machine.raw_data.release_sha_mem("Acq2")

    await machine.raw_data.data_acquire("Acq3", "SingleShot", "Settings1")
    await machine.raw_data.data_acquire("Acq4", "SingleShot", "Settings1")
    await machine.raw_data.delete_acquisition("Acq3")
    await machine.raw_data.delete_all_acquisitions()
    for name in ("Acq3", "Acq4"):
        with pytest.raises(IppDmeServerError):
            await machine.raw_data.get_raw_data_file(name)


async def test_raw_data_acquire_with_a_sweep_scan_path(machine: IppDmeMachine) -> None:
    await machine.raw_data.data_acquire(
        "Sweep1",
        "Sweep",
        "Settings1",
        points=[
            AcquisitionPoint((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
            AcquisitionPoint((10.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
        ],
    )
    _name, _offset, size = await machine.raw_data.get_raw_data_sha_mem("Sweep1")
    # Densified into more points than the 2 control points given (one
    # MeasPoint == 24 bytes).
    assert size > 2 * 24


async def test_server_namespace(machine: IppDmeMachine) -> None:
    assert await machine.server.get_err_status() is False
    description = await machine.server.get_error_info(501)
    assert "command" in description.lower()

    assert await machine.server.enum_prop("Part") == (
        ("Temperature", "Number"),
        ("XpanCoefficient", "Number"),
        ("Approach", "Number"),
        ("Search", "Number"),
        ("Retract", "Number"),
    )
    all_tool_props = await machine.server.enum_all_prop("Tool.GoToPar")
    assert all_tool_props[0] == ("Speed", "Property")
    assert ("Min", "Number") in all_tool_props


async def test_tool_changer_namespace(machine: IppDmeMachine) -> None:
    assert set(await machine.tool_changer.enum_tools()) == {"RefTool", "RefTool2", "AlignProbe"}
    assert await machine.tool_changer.found_tool() == "NoTool"

    await machine.tool_changer.find_tool("RefTool2")
    assert await machine.tool_changer.found_tool() == "RefTool2"

    await machine.tool_changer.change_tool("RefTool2")
    assert await machine.tool.get_name() == "RefTool2"

    action, offset = await machine.tool_changer.get_change_tool_action("RefTool")
    assert action == "MoveAuto"  # ~51mm offset, past the in-place "Switch" threshold
    assert offset == pytest.approx((-10.0, 0.0, -50.0))

    await machine.tool_changer.set_tool("RefTool")
    assert await machine.tool.get_name() == "RefTool"


async def test_part_namespace_defaults_and_round_trip(machine: IppDmeMachine) -> None:
    assert await machine.part.get_temperature() == 20.0
    assert await machine.part.get_xpan_coefficient() == 0.0
    assert await machine.part.get_approach() == 0.0
    assert await machine.part.get_search() == 0.0
    assert await machine.part.get_retract() == 0.0

    await machine.part.set_temperature(21.5)
    await machine.part.set_xpan_coefficient(11.7)
    await machine.part.set_approach(2.0)
    await machine.part.set_search(3.0)
    await machine.part.set_retract(4.0)

    assert await machine.part.get_temperature() == 21.5
    assert await machine.part.get_xpan_coefficient() == 11.7
    assert await machine.part.get_approach() == 2.0
    assert await machine.part.get_search() == 3.0
    assert await machine.part.get_retract() == 4.0
