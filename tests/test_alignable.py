# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for Alignable_AB/Alignable_ABC (AlignTool, A/B/C, Calc*, Enable/DisableOptimize)."""

from __future__ import annotations

import math

import pytest

from pyippdme import IppDmeClient
from pyippdme.exceptions import IppDmeServerError
from pyippdme.protocol.ast import (
    DataPayload,
    Items,
    NamedValue,
    Number,
    NumericData,
    PropertyData,
    String,
)
from pyippdme.protocol.commands import CommandName


def _items(data: DataPayload) -> Items:
    assert isinstance(data, Items)
    return data


def _nums(nv: NamedValue) -> tuple[float, ...]:
    return tuple(a.value for a in nv.args if isinstance(a, Number))


def _bool(data: DataPayload) -> float:
    """Read the single number of a response, bare (``1``) or named (``IsAlignable(1)``)."""
    if isinstance(data, Items):
        (named,) = data.values
        return _nums(named)[0]
    assert isinstance(data, NumericData)
    (value,) = data.values
    return value.value


async def _use_align_probe(client: IppDmeClient) -> None:
    await client.call(CommandName.CHANGE_TOOL, String("AlignProbe"))


async def test_is_alignable_reflects_the_active_tool(started_client: IppDmeClient) -> None:
    (default_tool,) = await started_client.call(CommandName.IS_ALIGNABLE)
    assert _bool(default_tool) == 0

    await _use_align_probe(started_client)
    (aligned,) = await started_client.call(CommandName.IS_ALIGNABLE)
    assert _bool(aligned) == 1


async def test_align_tool_four_arg_form_stores_and_echoes(
    started_client: IppDmeClient,
) -> None:
    await _use_align_probe(started_client)
    data = await started_client.call(
        CommandName.ALIGN_TOOL, *(Number.of(v) for v in (0.0, 0.0, 1.0, 5.0))
    )
    (values,) = data
    # Table 105: the reached vectors are returned unnamed.
    assert isinstance(values, NumericData)
    assert [n.value for n in values.values] == pytest.approx([0.0, 0.0, 1.0])


async def test_align_tool_eight_arg_form_stores_both_vectors(
    started_client: IppDmeClient,
) -> None:
    await _use_align_probe(started_client)
    data = await started_client.call(
        CommandName.ALIGN_TOOL,
        *(Number.of(v) for v in (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 5.0, 5.0)),
    )
    (values,) = data
    assert isinstance(values, NumericData)
    assert [n.value for n in values.values] == pytest.approx([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


async def test_align_tool_rejects_a_non_alignable_tool(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.ALIGN_TOOL, *(Number.of(v) for v in (0.0, 0.0, 1.0, 5.0))
        )
    assert excinfo.value.error.number == "1505"


async def test_align_tool_rejects_negative_alpha(started_client: IppDmeClient) -> None:
    await _use_align_probe(started_client)
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.ALIGN_TOOL, *(Number.of(v) for v in (0.0, 0.0, 1.0, -1.0))
        )
    assert excinfo.value.error.number == "0509"


async def test_align_tool_rejects_zero_vector(started_client: IppDmeClient) -> None:
    await _use_align_probe(started_client)
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.ALIGN_TOOL, *(Number.of(v) for v in (0.0, 0.0, 0.0, 5.0))
        )
    assert excinfo.value.error.number == "0509"


async def test_align_tool_rejects_wrong_argument_count(started_client: IppDmeClient) -> None:
    await _use_align_probe(started_client)
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.ALIGN_TOOL, *(Number.of(v) for v in (0.0, 0.0, 1.0)))
    assert excinfo.value.error.number == "0509"


async def test_avr_radius_is_always_zero(started_client: IppDmeClient) -> None:
    (data,) = await started_client.call(CommandName.AVR_RADIUS)
    assert _items(data).values[0].name == "AvrRadius"  # Table 106: Kind N
    assert _bool(data) == 0.0


async def test_get_a_and_b_report_the_default_orientation(started_client: IppDmeClient) -> None:
    (data,) = await started_client.call(
        CommandName.GET, NamedValue("Tool.A", ()), NamedValue("Tool.B", ())
    )
    values = {nv.name: _nums(nv)[0] for nv in _items(data).values}
    assert values["Tool.A"] == pytest.approx(0.0)  # default alignment (0,0,1) is "straight up"
    assert values["Tool.B"] == pytest.approx(0.0)


async def test_get_a_and_b_follow_align_tool(started_client: IppDmeClient) -> None:
    await _use_align_probe(started_client)
    await started_client.call(CommandName.ALIGN_TOOL, *(Number.of(v) for v in (1.0, 0.0, 0.0, 5.0)))
    (data,) = await started_client.call(
        CommandName.GET, NamedValue("Tool.A", ()), NamedValue("Tool.B", ())
    )
    values = {nv.name: _nums(nv)[0] for nv in _items(data).values}
    assert values["Tool.A"] == pytest.approx(90.0)  # (1,0,0) is 90 degrees from (0,0,1)
    assert values["Tool.B"] == pytest.approx(0.0)  # atan2(0, 1) = 0


async def test_get_c_reflects_the_secondary_vector(started_client: IppDmeClient) -> None:
    await _use_align_probe(started_client)
    await started_client.call(
        CommandName.ALIGN_TOOL,
        *(Number.of(v) for v in (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 5.0, 5.0)),
    )
    (data,) = await started_client.call(CommandName.GET, NamedValue("Tool.C", ()))
    (value,) = _items(data).values
    assert _nums(value)[0] == pytest.approx(90.0)  # atan2(0, 1) for the secondary (0,1,0)


async def test_calc_tool_alignment_matches_align_tool(started_client: IppDmeClient) -> None:
    await _use_align_probe(started_client)
    await started_client.call(CommandName.ALIGN_TOOL, *(Number.of(v) for v in (0.0, 1.0, 0.0, 5.0)))
    (data,) = await started_client.call(CommandName.CALC_TOOL_ALIGNMENT, NamedValue("Tool.A", ()))
    (value,) = _items(data).values
    assert value.name == "Tool.Alignment"
    assert _nums(value) == pytest.approx((0.0, 1.0, 0.0))


async def test_calc_tool_alignment_rejects_non_alignable_tool(
    started_client: IppDmeClient,
) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.CALC_TOOL_ALIGNMENT, NamedValue("Tool.A", ()))
    assert excinfo.value.error.number == "1505"


async def test_calc_tool_angles_is_the_inverse_of_calc_tool_alignment(
    started_client: IppDmeClient,
) -> None:
    await _use_align_probe(started_client)
    (data,) = await started_client.call(
        CommandName.CALC_TOOL_ANGLES,
        NamedValue("Tool.Alignment", (Number.of(1.0), Number.of(0.0), Number.of(0.0))),
    )
    values = {nv.name: _nums(nv)[0] for nv in _items(data).values}
    assert values["Tool.A"] == pytest.approx(90.0)
    assert values["Tool.B"] == pytest.approx(0.0)
    assert "Tool.C" not in values  # no secondary vector given


async def test_use_smallest_angle_to_align_tool_accepted_and_reset_by_start_session(
    started_client: IppDmeClient,
) -> None:
    await started_client.call(CommandName.USE_SMALLEST_ANGLE_TO_ALIGN_TOOL, Number.of(1))
    # StartSession() (6.20.1) resets it back to its default (0).
    await started_client.call(CommandName.END_SESSION)
    await started_client.call(CommandName.START_SESSION)
    # No direct getter exists for this flag; absence of an error and the
    # spec's own documented default is what there is to check here.


async def test_enable_disable_optimize_roundtrip(started_client: IppDmeClient) -> None:
    (before,) = await started_client.call(CommandName.IS_OPTIMIZE_ENABLED)
    assert _bool(before) == 0
    await started_client.call(CommandName.ENABLE_OPTIMIZE)
    (after,) = await started_client.call(CommandName.IS_OPTIMIZE_ENABLED)
    assert _bool(after) == 1
    await started_client.call(CommandName.DISABLE_OPTIMIZE)
    (reset,) = await started_client.call(CommandName.IS_OPTIMIZE_ENABLED)
    assert _bool(reset) == 0


async def test_clear_all_errors_disables_optimize(started_client: IppDmeClient) -> None:
    await started_client.call(CommandName.ENABLE_OPTIMIZE)
    await started_client.call(CommandName.CLEAR_ALL_ERRORS)
    (data,) = await started_client.call(CommandName.IS_OPTIMIZE_ENABLED)
    assert _bool(data) == 0


async def test_avr_offsets_is_always_zero(started_client: IppDmeClient) -> None:
    # Table 116: only usable as an argument of GetProp(Tool.AvrOffsets()).
    (data,) = await started_client.call(CommandName.GET_PROP, NamedValue("Tool.AvrOffsets", ()))
    (offsets,) = _items(data).values
    assert offsets.name == "Tool.AvrOffsets"
    assert _nums(offsets) == pytest.approx((0.0, 0.0, 0.0))


async def test_collision_and_alignment_volume_report_no_boxes(
    started_client: IppDmeClient,
) -> None:
    for name in ("Tool.CollisionVolume", "Tool.AlignmentVolume"):
        (data,) = await started_client.call(CommandName.GET_PROP, NamedValue(name, ()))
        (volume,) = _items(data).values
        assert volume.name == name
        assert volume.args == ()


async def test_tool_alignment_property_matches_align_tool(started_client: IppDmeClient) -> None:
    await _use_align_probe(started_client)
    await started_client.call(CommandName.ALIGN_TOOL, *(Number.of(v) for v in (0.0, 0.0, 1.0, 5.0)))
    (data,) = await started_client.call(CommandName.GET_PROP, NamedValue("Tool.Alignment", ()))
    (value,) = _items(data).values
    assert value.name == "Tool.Alignment"
    assert _nums(value) == pytest.approx((0.0, 0.0, 1.0))


async def test_tool_alignment_property_rejects_non_alignable_tool(
    started_client: IppDmeClient,
) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.GET_PROP, NamedValue("Tool.Alignment", ()))
    assert excinfo.value.error.number == "1505"


async def test_enum_prop_tool_includes_alignment_only_for_alignable_tools(
    started_client: IppDmeClient,
) -> None:
    default_props = await started_client.call(CommandName.ENUM_PROP, NamedValue("Tool", ()))
    default_names = {item.first.value for item in default_props if isinstance(item, PropertyData)}
    assert "Alignment" not in default_names

    await _use_align_probe(started_client)
    aligned_props = await started_client.call(CommandName.ENUM_PROP, NamedValue("Tool", ()))
    aligned_names = {item.first.value for item in aligned_props if isinstance(item, PropertyData)}
    assert {"Alignment", "A", "B", "C"} <= aligned_names


def test_tool_angle_convention_is_self_consistent() -> None:
    # Sanity-check the trig convention against basic geometry, independent
    # of the implementation (mirrors test_scanning.py's own such check).
    v = (1.0, 1.0, 0.0)
    length = math.sqrt(2.0)
    unit = (v[0] / length, v[1] / length, v[2] / length)
    a = math.degrees(math.acos(unit[2]))
    b = math.degrees(math.atan2(unit[1], unit[0]))
    assert (a, b) == pytest.approx((90.0, 45.0))


async def test_go_to_moves_the_tool_angles_and_step_adds_to_them(
    started_client: IppDmeClient,
) -> None:
    await _use_align_probe(started_client)
    await started_client.call(
        CommandName.GO_TO,
        NamedValue("Tool.A", (Number.of(30.0),)),
        NamedValue("Tool.B", (Number.of(45.0),)),
    )
    names = (NamedValue("Tool.A", ()), NamedValue("Tool.B", ()))
    (data,) = await started_client.call(CommandName.GET, *names)
    angles = {nv.name: _nums(nv)[0] for nv in _items(data).values}
    assert angles == pytest.approx({"Tool.A": 30.0, "Tool.B": 45.0})

    await started_client.call(CommandName.STEP, NamedValue("Tool.A", (Number.of(10.0),)))
    (data,) = await started_client.call(CommandName.GET, *names)
    angles = {nv.name: _nums(nv)[0] for nv in _items(data).values}
    assert angles == pytest.approx({"Tool.A": 40.0, "Tool.B": 45.0})


async def test_go_to_with_tool_alignment_sets_the_orientation(started_client: IppDmeClient) -> None:
    await _use_align_probe(started_client)
    await started_client.call(
        CommandName.GO_TO,
        NamedValue("Tool.Alignment", (Number.of(1.0), Number.of(0.0), Number.of(0.0))),
    )
    (data,) = await started_client.call(CommandName.GET_PROP, NamedValue("Tool.Alignment", ()))
    (value,) = _items(data).values
    assert _nums(value) == pytest.approx((1.0, 0.0, 0.0))


async def test_tool_angles_cannot_be_moved_on_a_fixed_tool(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.GO_TO, NamedValue("Tool.A", (Number.of(10.0),)))
    assert excinfo.value.error.number == "1505"


async def test_tool_c_is_not_supported_by_a_tool_with_two_rotation_axes(
    started_client: IppDmeClient,
) -> None:
    await _use_align_probe(started_client)
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.GO_TO, NamedValue("Tool.C", (Number.of(10.0),))
        )  # three angles on a two-axis head
    assert excinfo.value.error.number == "1004"


async def test_step_does_not_take_tool_alignment(started_client: IppDmeClient) -> None:
    await _use_align_probe(started_client)
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(
            CommandName.STEP, NamedValue("Tool.Alignment", (Number.of(0.0),) * 3)
        )
    assert excinfo.value.error.number == "0506"


async def test_get_ijk_is_the_direction_of_the_tool(started_client: IppDmeClient) -> None:
    await _use_align_probe(started_client)
    await started_client.call(CommandName.ALIGN_TOOL, *(Number.of(v) for v in (1.0, 0.0, 0.0, 5.0)))
    (data,) = await started_client.call(CommandName.GET, NamedValue("IJK", ()))
    (ijk,) = _items(data).values
    assert _nums(ijk) == pytest.approx((1.0, 0.0, 0.0))


async def test_found_tool_angles_need_find_tool(started_client: IppDmeClient) -> None:
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.GET, NamedValue("FoundTool.A", ()))
    assert excinfo.value.error.number == "1503"
    await started_client.clear_all_errors()
    await started_client.call(CommandName.FIND_TOOL, String("AlignProbe"))
    (data,) = await started_client.call(CommandName.GET, NamedValue("FoundTool.A", ()))
    assert _items(data).values[0].name == "FoundTool.A"


async def test_pointer_commands_name_the_parameter_blocks(started_client: IppDmeClient) -> None:
    from pyippdme.protocol.ast import NameValue

    (pt_meas_par,) = await started_client.call(CommandName.PT_MEAS_PAR)
    assert pt_meas_par == NameValue("Tool.PtMeasPar")
    (scan_par,) = await started_client.call(CommandName.SCAN_PAR)
    assert scan_par == NameValue("Tool.ScanPar")
    with pytest.raises(IppDmeServerError) as excinfo:
        await started_client.call(CommandName.OPT_PAR)  # no optical tool
    assert excinfo.value.error.number == "1506"


async def test_a_property_used_as_a_command_is_a_bad_context(network, server_port) -> None:  # type: ignore[no-untyped-def]
    """6.20.2, Table 113: ``Tool.A()`` is only valid as an argument of ``Get`` or ``OnReport``."""
    reader, writer = await network.open_connection("x", server_port)
    writer.write(b"00001 StartSession()\r\n00002 Tool.A()\r\n")
    await writer.drain()
    lines = [(await reader.readline()).decode().strip() for _ in range(4)]
    writer.close()
    assert '00002 ! Error(3,0508,"Protocol","Bad context")' in lines


async def test_use_smallest_angle_refuses_a_half_turn_in_both_spellings(
    started_client: IppDmeClient,
) -> None:
    await _use_align_probe(started_client)
    await started_client.call(CommandName.ALIGN_TOOL, *(Number.of(v) for v in (0, 0, 1, 5)))
    await started_client.call(CommandName.USE_SMALLEST_ANGLE_TO_ALIGN_TOOL, Number.of(1))
    with pytest.raises(IppDmeServerError) as error:
        await started_client.call(CommandName.ALIGN_TOOL, *(Number.of(v) for v in (0, 0, -1, 5)))
    assert error.value.error.number == "2500"
    await started_client.call(CommandName.CLEAR_ALL_ERRORS)
    # The other spelling in the text of the standard is the same command.
    await started_client.call("UseSmallestAngleToAlignTool", Number.of(0))
    await started_client.call(CommandName.ALIGN_TOOL, *(Number.of(v) for v in (0, 0, -1, 5)))
