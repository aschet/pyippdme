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
    fields = {nv.name: _nums(nv)[0] for nv in _items(values).values}
    assert fields == pytest.approx({"i1": 0.0, "j1": 0.0, "k1": 1.0})


async def test_align_tool_eight_arg_form_stores_both_vectors(
    started_client: IppDmeClient,
) -> None:
    await _use_align_probe(started_client)
    data = await started_client.call(
        CommandName.ALIGN_TOOL,
        *(Number.of(v) for v in (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 5.0, 5.0)),
    )
    (values,) = data
    fields = {nv.name: _nums(nv)[0] for nv in _items(values).values}
    assert fields == pytest.approx(
        {"i1": 1.0, "j1": 0.0, "k1": 0.0, "i2": 0.0, "j2": 1.0, "k2": 0.0}
    )


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
    assert _bool(data) == 0.0


async def test_get_a_and_b_report_the_default_orientation(started_client: IppDmeClient) -> None:
    (data,) = await started_client.call(CommandName.GET, NamedValue("A", ()), NamedValue("B", ()))
    values = {nv.name: _nums(nv)[0] for nv in _items(data).values}
    assert values["A"] == pytest.approx(0.0)  # default alignment (0,0,1) is "straight up"
    assert values["B"] == pytest.approx(0.0)


async def test_get_a_and_b_follow_align_tool(started_client: IppDmeClient) -> None:
    await _use_align_probe(started_client)
    await started_client.call(CommandName.ALIGN_TOOL, *(Number.of(v) for v in (1.0, 0.0, 0.0, 5.0)))
    (data,) = await started_client.call(CommandName.GET, NamedValue("A", ()), NamedValue("B", ()))
    values = {nv.name: _nums(nv)[0] for nv in _items(data).values}
    assert values["A"] == pytest.approx(90.0)  # (1,0,0) is 90 degrees from (0,0,1)
    assert values["B"] == pytest.approx(0.0)  # atan2(0, 1) = 0


async def test_get_c_reflects_the_secondary_vector(started_client: IppDmeClient) -> None:
    await _use_align_probe(started_client)
    await started_client.call(
        CommandName.ALIGN_TOOL,
        *(Number.of(v) for v in (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 5.0, 5.0)),
    )
    (data,) = await started_client.call(CommandName.GET, NamedValue("C", ()))
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
    (data,) = await started_client.call(CommandName.AVR_OFFSETS)
    values = {nv.name: _nums(nv)[0] for nv in _items(data).values}
    assert values == pytest.approx({"x": 0.0, "y": 0.0, "z": 0.0})


async def test_collision_and_alignment_volume_report_no_boxes(
    started_client: IppDmeClient,
) -> None:
    assert await started_client.call(CommandName.COLLISION_VOLUME) == ()
    assert await started_client.call(CommandName.ALIGNMENT_VOLUME) == ()


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
